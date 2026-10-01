"""Protocol, objective, checkpoint and restart tests on small synthetic folds."""
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, f1_score

from model_lstm.ablation_step_supervision.test_ablation import make_fold
from model_lstm.LSTM_model_train_tune import AssistLSTM, AssistLoss
from .data import FEATURE_KEYS, WindowDataset
from .engine import fit
from .models import MODELS
from .run_loso import run_loso


def make_l1_fold(root):
    fold = make_fold(root)
    for path in fold.glob('*/*.npz'):
        with np.load(path) as archive:
            data = dict(archive)
        frames = len(data['task_id'])
        for key in FEATURE_KEYS:
            dim = 9 if key == 'joint_angles' else 2 if key == 'ratios' else 32 if key == 'polar_azimuth' else 16
            data[key] = np.repeat((np.arange(frames, dtype=np.float32) / 100)[:, None], dim, axis=1)
        data['task_id'][119] = 7
        data['task_id_plateau_vector'][:, 2] = 1  # Intentionally disagrees with scalar truth.
        data['task_progress_vector'] = np.full((frames, 8), .4, dtype=np.float32)
        np.savez(path, **data)
    manifest_path = fold / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['feature_keys'] = list(FEATURE_KEYS)
    manifest_path.write_text(json.dumps(manifest))
    return fold


class L1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_l1_network_and_initialization(self):
        torch.manual_seed(3)
        original = AssistLSTM()
        for arm in MODELS:
            torch.manual_seed(3)
            model = MODELS[arm]([2.] * 7, 3.)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
            self.assertEqual(model.lstm.input_size, 251)
            self.assertEqual(model.lstm.hidden_size, 128)
            self.assertEqual(model.progress_head.out_features, 7)

    def test_losses_match_l1_auxiliaries_and_shared_step_mask(self):
        y = dict(step=torch.tensor([2, 7, 5]), peak=torch.rand(3, 7),
                 plateau=F.one_hot(torch.tensor([2, 1, 5]), 7).float(),
                 progress=torch.rand(3, 7), mistake=torch.tensor([0., 1., 0.]),
                 idle=torch.tensor([0., 1., 0.]))
        out = dict(step=torch.randn(3, 7, requires_grad=True), progress=torch.randn(3, 7),
                   mistake=torch.randn(3), background=torch.randn(3))
        original = AssistLoss([2.] * 7, 3.)
        _, old_parts = original((out['step'], out['progress'], out['mistake'], out['background']),
                                {**y, 'step': y['peak']})
        for arm in MODELS:
            model = MODELS[arm]([2.] * 7, 3.)
            total, parts, count = model.losses(out, y)
            self.assertEqual(int(count), 2)
            expected = F.cross_entropy(out['step'][[0, 2]], y['step'][[0, 2]]) if arm == 'B' else F.binary_cross_entropy_with_logits(
                out['step'][[0, 2]], y['peak'][[0, 2]], pos_weight=torch.full((7,), 2.))
            torch.testing.assert_close(parts['step'], expected)
            for key, old_key in [('progress', 'progress'), ('mistake', 'mistake'), ('background', 'idle')]:
                torch.testing.assert_close(parts[key], old_parts[old_key])
            torch.testing.assert_close(total, expected + .3 * parts['progress'] + .3 * parts['mistake'] + .2 * parts['background'])
            gradient = torch.autograd.grad(parts['step'], out['step'], retain_graph=True)[0]
            self.assertTrue((gradient[1] == 0).all())
            bg = {**y, 'step': torch.full((3,), 7), 'idle': torch.ones(3)}
            loss, parts, _ = model.losses(out, bg)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(parts['step'].detach()), 0.)
            gradient = torch.autograd.grad(parts['step'], out['step'], retain_graph=True)[0]
            self.assertTrue((gradient == 0).all())

    def test_scalar_truth_and_consecutive_120_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold = make_l1_fold(root / 'data')
            path = next((fold / 'test').glob('*.npz'))
            dataset = WindowDataset(path, 'test_norm', root / 'cache')
            x, y = dataset[0]
            self.assertEqual(tuple(x.shape), (120, 251))
            np.testing.assert_allclose(x[:, 0], np.arange(120) / 100, rtol=1e-6)
            self.assertEqual(int(y['step']), 7)
            self.assertEqual(float(y['idle']), 1)
            self.assertEqual(int(y['plateau'].argmax()), 2)
            self.assertTrue((y['peak'] > 0).all())
            self.assertEqual(dataset.target_frames.tolist(), [119, 129, 139, 149, 159])
            del dataset
            gc.collect()

    def test_checkpoint_selection_and_optimizer(self):
        model = MODELS['B']([2.] * 7, 3.)
        epoch = 0
        lrs = []
        def fake_epoch(model, loader, device, optimizer=None):
            nonlocal epoch
            if optimizer is not None:
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertEqual(optimizer.param_groups[0]['weight_decay'], .0001)
                lrs.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                epoch += 1
                with torch.no_grad():
                    model.step_head.bias.fill_(epoch)
                return {'loss': .1}, None, None
            return {'loss': [.1, .3, .2][epoch - 1], 'macro_f1_8': [.4, .7, .6][epoch - 1]}, None, None
        with tempfile.TemporaryDirectory() as temporary, patch(
            'model_lstm.ablation_l1_supervision.engine.run_epoch', side_effect=fake_epoch
        ):
            fit(model, [], [], torch.device('cpu'), temporary, 3)
            self.assertTrue((model.step_head.bias == 2).all())
            self.assertEqual(json.loads((Path(temporary) / 'best_val_loss.json').read_text())['epoch'], 1)
            np.testing.assert_allclose(lrs, [.001, .00075, .00025])

    def test_two_arm_pipeline_restart_and_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_l1_fold(root / 'data')
            run = run_loso(root / 'data', root / 'run', folds=[1], epochs=1, device='cpu', cache_dir=root / 'cache')
            previous = None
            for arm in MODELS:
                directory = run / arm / 'fold_01'
                frame = pd.read_csv(directory / 'test_predictions.csv')
                identity = frame[['file', 'target_frame', 'true_step']]
                if previous is not None:
                    pd.testing.assert_frame_equal(identity, previous)
                previous = identity
                result = json.loads((directory / 'test_results.json').read_text())
                self.assertAlmostEqual(result['macro_f1_8'], f1_score(frame.true_step, frame.pred_step, labels=range(8), average='macro', zero_division=0))
                cm = pd.read_csv(directory / 'confusion_matrix.csv', index_col=0).to_numpy()
                np.testing.assert_array_equal(cm, confusion_matrix(frame.true_step, frame.pred_step, labels=range(8)))
                config = json.loads((directory / 'config.json').read_text())
                self.assertEqual(config['input_dim'], 251)
                self.assertEqual(config['batch_size'], 256)
            self.assertEqual(len(pd.read_csv(run / 'paired_differences.csv')), 1)
            completed = run / 'B/fold_01/test_results.json'
            timestamp = completed.stat().st_mtime_ns
            (run / 'comparison.csv').unlink()
            with patch('model_lstm.ablation_l1_supervision.run_loso.load_fold') as load:
                run_loso(root / 'data', run, folds=[1], epochs=1, device='cpu')
                load.assert_not_called()
            self.assertEqual(len(pd.read_csv(run / 'comparison.csv')), 2)
            (run / 'C/fold_01/test_results.json').unlink()
            run_loso(root / 'data', run, folds=[1], epochs=1, device='cpu', cache_dir=root / 'cache')
            self.assertEqual(len(pd.read_csv(run / 'C/fold_01/metrics.csv')), 1)
            self.assertEqual(completed.stat().st_mtime_ns, timestamp)
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                run_loso(root / 'data', run, folds=[1], epochs=2, device='cpu')
            self.assertTrue((run / 'source/dependencies/model_lstm/LSTM_model_train_tune.py').exists())
            gc.collect()


if __name__ == '__main__':
    unittest.main()
