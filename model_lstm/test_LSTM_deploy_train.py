"""Full-data export, validation-only epoch selection and deployment reload checks."""

import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys


PROJECT_SRC_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_ROOT))

import numpy as np
import torch

from model_lstm import LSTM_deploy_train as deploy
from model_lstm.test_LSTM_tune import make_l1_corpus


class DeployTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_epoch_selection_needs_only_validation_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epochs = [7, 12, 9, 4, 9, 8, 9, 9, 9, 7, 9, 7, 5, 9, 6]
            for uid, epoch in enumerate(epochs, 1):
                fold = root / f"fold_{uid:02d}"
                fold.mkdir()
                (fold / "config.json").write_text(json.dumps(deploy.L1_SETTINGS))
                (fold / "summary_step_macro_f1.json").write_text(json.dumps({"best_epoch": epoch}))
            self.assertEqual(deploy.select_epochs(root)[0], 9)
            self.assertEqual(deploy.select_epochs(root / "nonexistent", 2)[0], 2)
            with self.assertRaises(ValueError):
                deploy.select_epochs(root, 13)
            (root / "fold_15/summary_step_macro_f1.json").unlink()
            with self.assertRaises(FileNotFoundError):
                deploy.select_epochs(root)

    def test_all_subjects_one_mirror_and_missing_mirror_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = make_l1_corpus(Path(temporary) / "raw")
            mirror = next((root / "augmented_mirror").glob("*uid-15*_aug-01.pt"))
            legacy = root / "aug_test/raw" / mirror.name
            legacy.parent.mkdir(parents=True)
            mirror.rename(legacy)
            files = deploy.full_training_files(root)
            self.assertEqual(len(files), 30)
            self.assertEqual(sum("_aug-01" in p.name for p in files), 15)
            self.assertFalse(any("_aug-02" in p.name for p in files))
            self.assertIn(legacy, files)
            legacy.unlink()
            with self.assertRaisesRegex(ValueError, "Missing aug-01"):
                deploy.full_training_files(root)

    def test_full_export_training_and_portable_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = make_l1_corpus(root / "raw")
            files = deploy.full_training_files(raw)
            # Keep real L1 feature/window/model dimensions; make a tiny synthetic corpus.
            for i, path in enumerate(files):
                content = torch.load(path, weights_only=False)
                content["features"] = {k: v.repeat(40, 1) + i / 10
                                       for k, v in content["features"].items()}
                content["labels"] = {k: v.repeat((40, 1) if v.ndim == 2 else (40,))
                                     for k, v in content["labels"].items()}
                torch.save(content, path)
            source_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
            # Keep the mmap test cache temporary instead of populating the real workspace.
            from unittest.mock import patch
            with patch.object(deploy, "ROOT", root):
                output = deploy.train_full_dataset(raw, root / "export", root / "output",
                                                   epochs=1, device="cpu")
            config = json.loads((output / "config.json").read_text())
            summary = json.loads((output / "training_summary.json").read_text())
            self.assertEqual(config["train_uids"], list(range(1, 16)))
            self.assertEqual(config["val_uids"], [])
            self.assertEqual(config["test_uids"], [])
            self.assertEqual(config["scheduler_t_max"], 12)
            self.assertEqual(summary["training_windows"], 630)
            self.assertEqual(config["feature_layout"][-1]["stop"], 251)
            self.assertEqual(hashlib.sha256((output / "norm_stats.npz").read_bytes()).hexdigest(),
                             config["normalization_sha256"])
            with np.load(output / "norm_stats.npz") as stats:
                values = np.concatenate([torch.load(p, weights_only=False)["features"]
                                         ["position_x_relative_to_pelvis"].numpy()[::7] for p in files])
                np.testing.assert_allclose(stats["position_x_relative_to_pelvis_mean"],
                                           values.mean(axis=0, dtype=np.float64), rtol=1e-6)
            model = deploy.AssistLSTM()
            model.load_state_dict(torch.load(output / "model_weights.pth", weights_only=True))
            model.eval()
            with torch.no_grad():
                outputs = model(torch.zeros(2, 120, 251))
            self.assertEqual([tuple(v.shape) for v in outputs], [(2, 7), (2, 7), (2,), (2,)])
            self.assertTrue(all(torch.isfinite(v).all() for v in outputs))
            self.assertEqual(source_hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in files])
            self.assertEqual(deploy.prepare_full_dataset(raw, root / "export")["normalization_id"],
                             config["normalization_id"])
            with self.assertRaisesRegex(ValueError, "new output"):
                deploy.train_full_dataset(raw, root / "export", output, epochs=1, device="cpu")
            with (root / "export/norm_stats.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Normalization file changed"):
                deploy.prepare_full_dataset(raw, root / "export")
            gc.collect()


if __name__ == "__main__":
    unittest.main()
