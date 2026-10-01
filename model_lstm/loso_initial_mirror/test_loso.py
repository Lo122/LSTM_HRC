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
from sklearn.metrics import f1_score
from model_lstm.LSTM_model_train import AssistLSTM
from .data import SETTINGS, REFERENCE, FEATURE_KEYS, load_fold, training_weights
from .model import MirrorLSTM
from .run_loso import run_loso


def make_sources(root):
    for directory in ('original', 'augmented_mirror'):
        (root / directory).mkdir(parents=True)
    for uid in range(1, 16):
        for mirror in (False, True):
            time = torch.arange(400, dtype=torch.float32) / 100
            offset = uid + .2 * mirror
            features = {}
            for key in FEATURE_KEYS:
                width = 9 if key == 'joint_angles' else 2 if key == 'ratios' else 16
                features[key] = (time[:, None] + offset).repeat(1, width)
            labels = dict(task_id=torch.zeros(400, dtype=torch.long),
                          task_progress=torch.arange(400, dtype=torch.float32) % 101,
                          mistake=torch.arange(400) % 2)
            labels['task_id'][159::30] = torch.arange(len(labels['task_id'][159::30])) % 8
            directory = root / ('augmented_mirror' if mirror else 'original')
            suffix = '_aug-01' if mirror else ''
            torch.save(dict(features=features, labels=labels), directory / f'features__cam-05_uid-{uid:02d}_take-01{suffix}.pt')


class MirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_network_and_weighted_sample_mean(self):
        torch.manual_seed(42)
        original = AssistLSTM(89, 64, 8, dropout=.2, num_layers=1, num_mistakes=2)
        torch.manual_seed(42)
        model = MirrorLSTM(89, [1., 2., 3., 4., 5., 6., 7., .5])
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        out = dict(step=torch.randn(3, 8), progress=torch.randn(3), mistake=torch.randn(3, 2))
        y = dict(step=torch.tensor([7, 7, 7]), progress=torch.rand(3), mistake=torch.tensor([0, 1, 0]))
        total, parts, count = model.losses(out, y)
        expected = F.cross_entropy(out['step'], y['step']) * .5
        torch.testing.assert_close(parts['step'], expected)
        torch.testing.assert_close(total, expected + F.mse_loss(out['progress'], y['progress']) + F.cross_entropy(out['mistake'], y['mistake']))
        self.assertEqual(count, 3)

    def test_train_only_legacy_norm_windows_and_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sources(root / 'raw')
            datasets, manifest = load_fold(root / 'raw', root / 'fold', 1, root / 'cache')
            with np.load(root / 'fold/norm_stats.npz') as archive:
                stats = dict(archive)
            recordings = [d.recording for d in datasets['train'].datasets]
            raw = np.concatenate([r.features for r in recordings])
            # Compare each group with the original all-frame mean/std formula.
            start = 0
            for key, width in zip(FEATURE_KEYS, manifest['dimensions']):
                values = raw[:, start:start + width].copy()
                np.testing.assert_allclose(stats[f'{key}_mean'], values.mean(0), rtol=1e-6)
                np.testing.assert_allclose(stats[f'{key}_std'], values.std(0) + 1e-6, rtol=1e-6)
                start += width
            self.assertNotIn(1, manifest['subjects']['train'])
            self.assertEqual(manifest['input_dim'], 89)
            d = datasets['test'].datasets[0]
            x, y = d[0]
            self.assertEqual(tuple(x.shape), (160, 89))
            self.assertEqual(d.target_frames.tolist(), list(range(159, 400, 30)))
            self.assertAlmostEqual(float(y['progress']), 58 / 100, places=6)
            np.testing.assert_allclose(x, (d.recording.features[:160] - d.mean) / d.std)
            sampler, counts, weights = training_weights(datasets['train'], 42)
            targets = np.concatenate([p.recording.step[p.target_frames] for p in datasets['train'].datasets])
            np.testing.assert_allclose(sampler.weights, 1 / np.sqrt(np.array(counts)[targets]))
            expected = 1 / np.sqrt(counts)
            expected /= np.average(expected, weights=counts)
            np.testing.assert_allclose(weights, expected)
            self.assertTrue(sampler.replacement)
            self.assertEqual(len(sampler), len(datasets['train']))
            # Existing folds reject changed held-out source files, never silently reuse exports.
            path = next((root / 'raw/original').glob('*uid-01*'))
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Source data'):
                load_fold(root / 'raw', root / 'fold', 1, root / 'cache')
            del datasets, d, recordings, sampler
            gc.collect()

    def test_reference_saved_weights_are_reproduced(self):
        from types import SimpleNamespace
        counts = np.array(REFERENCE['step_class_counts'])
        targets = np.repeat(np.arange(8), counts)
        dataset = SimpleNamespace(datasets=[SimpleNamespace(
            recording=SimpleNamespace(step=targets), target_frames=np.arange(len(targets)))])
        sampler, actual_counts, weights = training_weights(dataset, 42)
        self.assertEqual(actual_counts, counts.tolist())
        np.testing.assert_allclose(weights, REFERENCE['step_class_weights'], rtol=1e-6)
        np.testing.assert_allclose(sampler.weights, 1 / np.sqrt(counts[targets]))
        self.assertEqual(SETTINGS['class7_loss_factor'], 1.)

    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sources(root / 'raw')
            run = run_loso(root / 'raw', root / 'run', folds=[1], epochs=1, device='cpu', cache_dir=root / 'cache')
            fold = run / 'mirror/fold_01'
            frame = pd.read_csv(fold / 'test_predictions.csv')
            results = json.loads((fold / 'test_results.json').read_text())
            self.assertAlmostEqual(results['macro_f1_8'], f1_score(frame.true_step, frame.pred_step, labels=range(8), average='macro', zero_division=0))
            self.assertEqual(json.loads((fold / 'train_results.json').read_text())['windows'], 24 * 9)
            self.assertEqual(frame.target_frame.tolist(), list(range(159, 400, 30)))
            self.assertTrue((fold / 'best_val_loss.pth').exists())
            self.assertTrue((fold / 'train_evaluation/confusion_matrix.csv').exists())
            (run / 'comparison.csv').unlink()
            marker = fold / 'test_results.json'
            previous = marker.stat().st_mtime_ns
            with patch('model_lstm.loso_initial_mirror.run_loso.load_fold') as load:
                run_loso(root / 'raw', run, folds=[1], epochs=1, device='cpu')
                load.assert_not_called()
            self.assertEqual(marker.stat().st_mtime_ns, previous)
            self.assertTrue((run / 'comparison.csv').exists())
            marker.unlink()
            run_loso(root / 'raw', run, folds=[1], epochs=1, device='cpu', cache_dir=root / 'cache')
            self.assertEqual(len(pd.read_csv(fold / 'metrics.csv')), 1)
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                run_loso(root / 'raw', run, folds=[1], epochs=1, selection='val_loss', device='cpu')
            # Legacy checkpoint selection remains available as a separate protocol.
            second = run_loso(root / 'raw', root / 'loss_run', folds=[1], epochs=1, selection='val_loss', device='cpu', cache_dir=root / 'cache')
            self.assertEqual(json.loads((second / 'protocol.json').read_text())['test_checkpoint'], 'best_val_loss.pth')
            gc.collect()


if __name__ == '__main__':
    unittest.main()
