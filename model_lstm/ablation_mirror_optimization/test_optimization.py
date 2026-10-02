import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_lstm.loso_initial_mirror.test_loso import make_sources
from model_lstm.loso_initial_mirror.model import MirrorLSTM
from model_lstm.loso_initial_mirror.run_loso import run_loso as run_control
from model_lstm.ablation_step_supervision.engine import run_epoch
from .engine import fit
from .run_loso import run_loso


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_clips_gradients_before_optimizer_step(self):
        model = MirrorLSTM(89, [1.] * 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        batch = (torch.randn(2, 160, 89), dict(step=torch.tensor([1, 2]),
                 progress=torch.full((2,), 100.), mistake=torch.tensor([0, 1])))
        original_step = optimizer.step
        def checked_step():
            norm = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm()
            self.assertLessEqual(float(norm), 1.00001)
            return original_step()
        with patch.object(optimizer, 'step', side_effect=checked_step), patch(
            'torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_
        ) as clip:
            run_epoch(model, [batch], torch.device('cpu'), optimizer, gradient_clip=1.)
            clip.assert_called_once()
        with patch('torch.nn.utils.clip_grad_norm_') as clip:
            run_epoch(model, [batch], torch.device('cpu'))
            clip.assert_not_called()

    def test_optimizer_schedule_and_best_checkpoint(self):
        model = MirrorLSTM(89, [1.] * 8)
        epoch, rates = 0, []
        def fake_epoch(model, loader, device, optimizer=None, gradient_clip=None):
            nonlocal epoch
            if optimizer is not None:
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertEqual(optimizer.param_groups[0]['weight_decay'], .0001)
                self.assertEqual(gradient_clip, 1.)
                rates.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                epoch += 1
                with torch.no_grad():
                    model.step_head.bias.fill_(epoch)
                return {'loss': .1}, None, None
            return {'loss': [.1, .3, .2][epoch - 1], 'macro_f1_8': [.4, .7, .6][epoch - 1]}, None, None
        with tempfile.TemporaryDirectory() as temporary, patch(
            'model_lstm.ablation_mirror_optimization.engine.run_epoch', side_effect=fake_epoch
        ):
            fit(model, [], [], torch.device('cpu'), temporary, 3)
            np.testing.assert_allclose(rates, [.001, .00075, .00025])
            self.assertTrue((model.step_head.bias == 2).all())
            self.assertEqual(json.loads((Path(temporary) / 'best_val_loss.json').read_text())['epoch'], 1)

    def test_paired_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sources(root / 'raw')
            baseline = run_control(root / 'raw', root / 'control', folds=[1], epochs=1,
                                   device='cpu', cache_dir=root / 'cache')
            marker = baseline / 'mirror/fold_01/test_results.json'
            timestamp = marker.stat().st_mtime_ns
            run = run_loso(root / 'raw', root / 'new', folds=[1], epochs=1, device='cpu',
                           cache_dir=root / 'cache', baseline_run=baseline)
            base_config = json.loads((baseline / 'mirror/fold_01/config.json').read_text())
            new_config = json.loads((run / 'mirror/fold_01/config.json').read_text())
            changed = {k for k in base_config if base_config[k] != new_config[k]}
            self.assertEqual(changed, {'experiment', 'optimizer', 'lr', 'weight_decay', 'scheduler', 'gradient_clip'})
            self.assertEqual(new_config['batch_size'], 32)
            self.assertEqual(new_config['scheduler_t_max'], 1)
            self.assertEqual(marker.stat().st_mtime_ns, timestamp)
            summary = json.loads((run / 'optimizer_comparison_summary.json').read_text())
            self.assertEqual(summary['test_uids'], [1])
            score = summary['metrics']['macro_f1_8']
            self.assertAlmostEqual(score['delta']['mean'], score['l1opt']['mean'] - score['baseline']['mean'])
            (run / 'paired_optimizer_comparison.csv').unlink()
            with patch('model_lstm.ablation_mirror_optimization.run_loso.load_fold') as load:
                run_loso(root / 'raw', run, folds=[1], epochs=1, device='cpu', baseline_run=baseline)
                load.assert_not_called()
            self.assertEqual(len(pd.read_csv(run / 'paired_optimizer_comparison.csv')), 1)
            (run / 'mirror/fold_01/test_results.json').unlink()
            run_loso(root / 'raw', run, folds=[1], epochs=1, device='cpu', cache_dir=root / 'cache', baseline_run=baseline)
            self.assertEqual(len(pd.read_csv(run / 'mirror/fold_01/metrics.csv')), 1)
            with self.assertRaisesRegex(ValueError, 'Baseline settings differ'):
                run_loso(root / 'raw', root / 'bad', folds=[1], epochs=2, device='cpu', baseline_run=baseline)
            self.assertFalse((root / 'bad').exists())
            gc.collect()


if __name__ == '__main__':
    unittest.main()
