"""Behavioral checks for matched A/B/C supervision and the shared LOSO pipeline."""

import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

from .data import FEATURE_KEYS, WindowDataset
from .engine import class_scores, fit
from .models import MODELS
from .run_loso import run_loso


def write_npz(path, norm_id="test_norm", frames=160, invalid_frame=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = {key: (9 if key == "joint_angles" else 2 if key == "ratios" else 16) for key in FEATURE_KEYS}
    time = np.arange(frames, dtype=np.float32)
    data = {key: np.repeat((time / 100)[:, None], dim, axis=1) for key, dim in dimensions.items()}
    step = (time.astype(np.int64) // 10) % 8
    step[117] = 7  # Deliberately conflict with soft labels and plateau.
    valid = np.ones(frames, dtype=bool)
    if invalid_frame is not None:
        valid[invalid_frame] = False
    data.update(task_id=step, task_id_vector=np.full((frames, 8), .6, dtype=np.float32),
                task_id_plateau_vector=np.zeros((frames, 8), dtype=np.float32),
                task_progress=time / frames, mistake=time.astype(np.int64) % 2,
                valid_frame=valid, normalization_id=np.array(norm_id), feature_profile=np.array("claude_l1_v1"))
    np.savez(path, **data)


def make_fold(root, uid=1):
    directory = root / f"fold_{uid:02d}"
    directory.mkdir(parents=True)
    rest = [u for u in range(1, 16) if u != uid]
    val = sorted(np.random.RandomState(1000 + uid).permutation(rest)[:2].tolist())
    groups = dict(train=sorted(set(rest) - set(val)), val=val, test=[uid])
    np.savez(directory / "norm_stats.npz", marker=np.array([uid]))
    files = {}
    for split, subjects in groups.items():
        files[split] = []
        for subject in subjects:
            for aug in ("", "_aug-01") if split == "train" else ("",):
                name = f"{split}/features__cam-05_uid-{subject:02d}_take-01{aug}_norm.npz"
                write_npz(directory / name)
                files[split].append(name)
    manifest = dict(subjects=groups, files=files, profile="claude_l1_v1", feature_keys=list(FEATURE_KEYS),
                    normalization_id="test_norm",
                    normalization_sha256=hashlib.sha256((directory / "norm_stats.npz").read_bytes()).hexdigest())
    (directory / "dataset_manifest.json").write_text(json.dumps(manifest))
    return directory


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampling_last_frame_and_unsampled_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_npz(root / "sample.npz")
            dataset = WindowDataset(root / "sample.npz", "test_norm", root / "cache")
            x, y = dataset[0]
            self.assertEqual(x.shape, (40, 91))
            np.testing.assert_allclose(x[:, 0], np.arange(0, 118, 3) / 100, rtol=1e-6)
            self.assertEqual(dataset.target_frames.tolist(), [117, 127, 137, 147, 157])
            self.assertEqual(int(y["step"]), 7)  # Never infer truth from plateau/peak.
            self.assertAlmostEqual(float(y["progress"]), 117 / 160)
            self.assertTrue((y["peak"] > 0).all())  # Do not zero original peaks.
            with self.assertRaisesRegex(ValueError, "normalization"):
                WindowDataset(root / "sample.npz", "other_fold", root / "cache")
            write_npz(root / "gap.npz", invalid_frame=1)
            gap = WindowDataset(root / "gap.npz", "test_norm", root / "cache")
            self.assertEqual(gap.target_frames.tolist(), [127, 137, 147, 157])
            del dataset, gap
            gc.collect()

    def test_masks_unweighted_losses_and_all_background_gradient(self):
        targets = dict(step=torch.tensor([2, 7, 5]), peak=torch.rand(3, 7),
                       progress=torch.tensor([.1, .2, .3]), mistake=torch.tensor([0, 1, 0]))
        for arm in ("B", "C"):
            model = MODELS[arm]()
            logits = torch.randn(3, 7, requires_grad=True)
            out = dict(step=logits, background=torch.tensor([-.4, .9, .2], requires_grad=True),
                       progress=torch.zeros(3, requires_grad=True), mistake=torch.zeros(3, 2, requires_grad=True))
            step, bg, count = model.step_losses(out, targets)
            self.assertEqual(int(count), 2)
            expected = (F.cross_entropy(logits[[0, 2]], targets["step"][[0, 2]]) if arm == "B"
                        else F.binary_cross_entropy_with_logits(logits[[0, 2]], targets["peak"][[0, 2]]))
            torch.testing.assert_close(step, expected)
            torch.testing.assert_close(bg, F.binary_cross_entropy_with_logits(out["background"], torch.tensor([0., 1., 0.])))
            step.backward()
            self.assertTrue((logits.grad[1] == 0).all())
            self.assertGreater(float(logits.grad[[0, 2]].abs().sum()), 0)
            # Even nonzero peak targets on all-background batches must yield no task gradient.
            only_bg = {**targets, "step": torch.full((3,), 7)}
            out["step"] = torch.randn(3, 7, requires_grad=True)
            total, parts, _ = model.losses(out, only_bg)
            self.assertEqual(float(parts["step"].detach()), 0)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertTrue((out["step"].grad == 0).all())
        model = MODELS["A"]()
        logits = torch.randn(3, 8)
        torch.testing.assert_close(model.step_losses({"step": logits}, targets)[0],
                                   F.cross_entropy(logits, targets["step"]))

    def test_common_initialization_and_same_background_threshold(self):
        states = {}
        for arm in MODELS:
            torch.manual_seed(42)
            states[arm] = MODELS[arm]().state_dict()
        common = [k for k in states["A"] if not k.startswith("step_head")]
        for name in common:
            for arm in ("B", "C"):
                torch.testing.assert_close(states["A"][name], states[arm][name], rtol=0, atol=0)
        for name in states["B"]:
            torch.testing.assert_close(states["B"][name], states["C"][name], rtol=0, atol=0)
        logits = torch.zeros(3, 7)
        logits[:, 2] = 1
        for arm in ("B", "C"):
            pred, _ = MODELS[arm]().decode(dict(step=logits, background=torch.tensor([-.01, 0., .01])))
            self.assertEqual(pred.tolist(), [2, 2, 7])

    def test_complete_eight_class_metric_counts_background_misses(self):
        true = np.array([0, 0, 1, 7, 7])
        pred = np.array([0, 7, 1, 0, 7])
        cm = confusion_matrix(true, pred, labels=range(8))
        self.assertAlmostEqual(class_scores(cm)[2].mean(), f1_score(true, pred, labels=range(8), average="macro", zero_division=0))
        self.assertEqual(cm.sum(), 5)

    def test_checkpoint_uses_validation_full_f1_not_loss_or_last_epoch(self):
        model = MODELS["A"]()
        epoch = 0
        def fake_epoch(model, loader, device, optimizer=None):
            nonlocal epoch
            if optimizer is not None:
                epoch += 1
                with torch.no_grad():
                    model.step_head.bias.fill_(epoch)
                return {"loss": .1}, None, None
            return {"loss": [.1, .3, .2][epoch - 1], "macro_f1_8": [.4, .7, .6][epoch - 1]}, None, None
        with tempfile.TemporaryDirectory() as temporary, patch(
            "model_lstm.ablation_step_supervision.engine.run_epoch", side_effect=fake_epoch
        ):
            fit(model, [], [], torch.device("cpu"), temporary, 3)
            self.assertTrue((model.step_head.bias == 2).all())
            root = Path(temporary)
            self.assertEqual(json.loads((root / "best_macro_f1.json").read_text())["epoch"], 2)
            self.assertEqual(json.loads((root / "best_val_loss.json").read_text())["epoch"], 1)

    def test_three_arm_end_to_end_same_windows_and_saved_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fold = make_fold(root / "data")
            run = run_loso(root / "data", root / "run", folds=[1], epochs=1, device="cpu", cache_dir=root / "cache")
            identity = None
            for arm in MODELS:
                directory = run / arm / "fold_01"
                frame = pd.read_csv(directory / "test_predictions.csv")
                current = frame[["file", "target_frame", "true_step"]]
                if identity is not None:
                    pd.testing.assert_frame_equal(identity, current)
                identity = current
                result = json.loads((directory / "test_results.json").read_text())
                score = f1_score(frame.true_step, frame.pred_step, labels=range(8), average="macro", zero_division=0)
                self.assertAlmostEqual(score, result["macro_f1_8"])
                cm = pd.read_csv(directory / "confusion_matrix.csv", index_col=0).to_numpy()
                np.testing.assert_array_equal(cm, confusion_matrix(frame.true_step, frame.pred_step, labels=range(8)))
                np.testing.assert_array_equal(cm, pd.read_csv(run / arm / "confusion_matrix.csv", index_col=0).to_numpy())
                self.assertEqual((directory / "norm_stats.npz").read_bytes(), (fold / "norm_stats.npz").read_bytes())
                summary = json.loads((run / arm / "loso_summary.json").read_text())
                self.assertEqual(summary["completed_folds"], 1)
                self.assertEqual(summary["metrics"]["macro_f1_8"]["std"], 0)
                config = json.loads((directory / "config.json").read_text())
                self.assertIsNone(config["class_weights"])
                self.assertIsNone(config["pos_weight"])
                self.assertEqual(config["sampler"], "RandomSampler")
            self.assertEqual(len(pd.read_csv(run / "comparison.csv")), 3)
            self.assertEqual(len(pd.read_csv(run / "paired_differences.csv")), 3)
            self.assertTrue((run / "source/models.py").is_file())
            completed = run / "A/fold_01/test_results.json"
            original_time = completed.stat().st_mtime_ns
            (run / "comparison.csv").unlink()
            with patch("model_lstm.ablation_step_supervision.run_loso.load_fold") as load:
                run_loso(root / "data", run, arms=["C"], folds=[1], epochs=1, device="cpu")
                load.assert_not_called()
            self.assertEqual(len(pd.read_csv(run / "comparison.csv")), 3)
            # Restart only incomplete C, without appending duplicate epoch rows.
            (run / "C/fold_01/test_results.json").unlink()
            run_loso(root / "data", run, folds=[1], epochs=1, device="cpu", cache_dir=root / "cache")
            self.assertEqual(len(pd.read_csv(run / "C/fold_01/metrics.csv")), 1)
            self.assertEqual(completed.stat().st_mtime_ns, original_time)
            self.assertEqual(len(pd.read_csv(run / "paired_differences.csv")), 3)
            # Add another fold and retain the previous fold in pooled reports.
            make_fold(root / "data", uid=2)
            run_loso(root / "data", run, arms=["A"], folds=[2], epochs=1, device="cpu", cache_dir=root / "cache")
            self.assertEqual(json.loads((run / "A/loso_summary.json").read_text())["test_uids"], [1, 2])
            self.assertEqual(json.loads((run / "protocol.json").read_text())["folds"], [1, 2])
            with self.assertRaisesRegex(ValueError, "settings differ"):
                run_loso(root / "data", run, folds=[1], epochs=2, device="cpu")
            np.savez(fold / "norm_stats.npz", marker=np.array([999]))
            with self.assertRaisesRegex(ValueError, "normalization differs"):
                run_loso(root / "data", run, folds=[1], epochs=1, device="cpu")
            gc.collect()


if __name__ == "__main__":
    unittest.main()
