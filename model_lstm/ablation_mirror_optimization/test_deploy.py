import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from model_lstm.loso_initial_mirror.test_loso import make_sources
from model_lstm.loso_initial_mirror.data import SETTINGS, RawRecording, fit_norm, WindowDataset
from .deploy_train import train_full_dataset
from .engine import OPTIMIZATION
from .inference import DeploymentModel


def reference(root):
    root.mkdir()
    config = dict(SETTINGS)
    config.update(OPTIMIZATION, experiment='mirror_l1_optimization_v1')
    (root / 'protocol.json').write_text(json.dumps(config))
    return root


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_data_training_and_portable_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sources(root / 'raw')
            loso = reference(root / 'loso')
            run = train_full_dataset(root / 'raw', root / 'deploy', loso, epochs=2,
                                     device='cpu', cache_dir=root / 'cache')
            config = json.loads((run / 'config.json').read_text())
            manifest = json.loads((run / 'dataset_manifest.json').read_text())
            self.assertEqual(config['train_uids'], list(range(1, 16)))
            self.assertEqual(config['val_uids'], [])
            self.assertEqual(config['test_uids'], [])
            self.assertEqual(len(manifest['sources']['train']), 30)
            self.assertEqual(config['sampler_num_samples'], 270)
            self.assertEqual(config['scheduler_t_max'], 2)
            self.assertEqual(config['batch_size'], 32)
            self.assertEqual(config['optimizer'], 'AdamW')
            self.assertEqual(config['gradient_clip'], 1.)
            metrics = pd.read_csv(run / 'metrics.csv')
            np.testing.assert_allclose(metrics.learning_rate, [.001, .0005])
            self.assertFalse((run / 'best_macro_f1.pth').exists())
            self.assertTrue((run / 'model_weights.pth').exists())
            self.assertFalse((run / 'last_model_weights.pth').exists())
            self.assertFalse((run / 'last_checkpoint.json').exists())
            records = [RawRecording(p['path'], root / 'cache') for p in manifest['sources']['train']]
            stats = fit_norm(records)
            with np.load(run / 'norm_stats.npz') as saved:
                for k in stats:
                    np.testing.assert_array_equal(saved[k], stats[k])
            predictor = DeploymentModel(run)
            raw = records[0].features[:160]
            result = predictor.predict(raw)
            self.assertEqual(result['step_logits'].shape, (1, 8))
            np.testing.assert_allclose(result['step_probabilities'].sum(1), [1.], atol=1e-6)
            x, _ = WindowDataset(records[0], stats)[0]
            with torch.no_grad():
                expected = predictor.model(x.unsqueeze(0))[0].numpy()
            np.testing.assert_allclose(result['step_logits'], expected, atol=1e-6)
            self.assertNotIn('class_weights', torch.load(run / 'model_weights.pth', weights_only=True))
            with self.assertRaisesRegex(ValueError, 'Expected finite raw features'):
                predictor.predict(np.zeros((120, 89)))
            with self.assertRaisesRegex(ValueError, 'new output'):
                train_full_dataset(root / 'raw', run, loso, epochs=2, device='cpu')
            del records, predictor, raw
            gc.collect()

    def test_interruption_preserves_last_completed_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_sources(root / 'raw')
            loso = reference(root / 'loso')
            from .deploy_train import run_epoch
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('simulated interruption')
                return run_epoch(*args, **kwargs)
            with patch('model_lstm.ablation_mirror_optimization.deploy_train.run_epoch', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    train_full_dataset(root / 'raw', root / 'deploy', loso, epochs=2,
                                       device='cpu', cache_dir=root / 'cache')
            out = root / 'deploy'
            self.assertEqual(json.loads((out / 'last_checkpoint.json').read_text())['completed_epochs'], 1)
            self.assertTrue((out / 'last_model_weights.pth').exists())
            self.assertFalse((out / 'model_weights.pth').exists())
            self.assertFalse((out / 'training_summary.json').exists())
            gc.collect()


if __name__ == '__main__':
    unittest.main()
