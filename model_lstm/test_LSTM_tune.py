"""Semantic checks and a small end-to-end run of the stage-3 notebook.

Run from the repository root: python -B -m unittest model_lstm.test_LSTM_tune -v
"""

import contextlib
import gc
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model_lstm.LSTM_databuilder_tune import (
    AssistSequenceDataset, training_pos_weights,
)
from model_lstm.LSTM_model_train_tune import AssistLSTM, AssistLoss
from data_proc_2d.app.build_norm_dataset_tune import (
    NormDatasetBuilder, compute_global_norm_stats, export_datasets, prepare_splits, split_by_uid,
    FEATURE_KEYS, L1_PROFILE, feature_arrays, prepare_l1_fold, l1_subject_split,
)


MODEL_DIR = Path(__file__).resolve().parent
FEATURE_DIMS = {
    "distance_from_center": 16, "position_x_relative_to_pelvis": 16,
    "position_y_relative_to_pelvis": 16, "position_z_relative_to_pelvis": 16,
    "joint_angles": 9, "polar_elevation": 16, "ratios": 2,
}
FEATURE_DIMS.update({key: 16 for key in FEATURE_KEYS if key not in FEATURE_DIMS})


def make_pair(root, split="aug_train", uid=3):
    directory = Path(root) / split
    for child in ("raw", "norm_tune"):
        (directory / child).mkdir(parents=True, exist_ok=True)
    name = f"features__cam-05_uid-{uid:02d}_take-01_aug-01"
    peak = torch.zeros(8, 8)
    plateau = torch.zeros(8, 8)
    progress = torch.zeros(8, 8)
    peak[1, 1:3] = torch.tensor([0.75, 0.8])
    plateau[1, 1:3] = 1
    progress[1, 1:3] = torch.tensor([30, 70])
    peak[3, 7], plateau[3, 7] = 1, 1
    peak[3, 0] = 0.1  # Preserve a soft task target even when idle is true.
    progress[3, :7] = 90  # Inactive lanes must not enter progress loss/MAE.
    peak[5, 0], plateau[5, 0], progress[5, 0] = 0.5, 0.5, 60
    peak[[1, 5, 7], 4] = 0.7  # peak counts, not plateau or argmax counts
    plateau[7, 4], progress[7, 4] = 1, 70
    mistake = torch.zeros(8, dtype=torch.int64)
    mistake[5] = 1
    labels = dict(task_id_vector=peak, task_id_plateau_vector=plateau,
                  task_progress_vector=progress, mistake=mistake, task_id=peak.argmax(1),
                  task_progress=progress.gather(1, peak.argmax(1)[:, None]).squeeze(1))
    features = {key: np.arange(8 * dim, dtype=np.float32).reshape(8, dim) / 100
                for key, dim in FEATURE_DIMS.items()}
    npz = directory / "norm_tune" / f"{name}_norm.npz"
    raw = directory / "raw" / f"{name}.pt"
    torch.save({"labels": labels, "features": {k: torch.from_numpy(v) for k, v in features.items()}}, raw)
    stats = {f"{key}_{kind}": np.full(dim, value, dtype=np.float32)
             for key, dim in FEATURE_DIMS.items() for kind, value in [("mean", 0), ("std", 1)]}
    NormDatasetBuilder(stats, FEATURE_DIMS).build_dataset(raw, npz)
    return npz, raw


def make_l1_corpus(root):
    root = Path(root)
    (root / "original").mkdir(parents=True)
    (root / "augmented_mirror").mkdir()
    for uid in range(1, 16):
        _, raw = make_pair(root, "fixtures", uid)
        content = raw.read_bytes()
        (root / "original" / raw.name.replace("_aug-01", "")).write_bytes(content)
        (root / "augmented_mirror" / raw.name).write_bytes(content)
        # Duplicate augmentations must not be selected.
        (root / "augmented_mirror" / raw.name.replace("_aug-01", "_aug-02")).write_bytes(content)
    return root


class TuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.npz, self.raw = make_pair(self.root)

    def tearDown(self):
        gc.collect()
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return AssistSequenceDataset(self.npz, window_size=2, stride=2,
                                     cache_dir=self.root / "cache", **kwargs)

    def test_last_frame_vectors_and_unchanged_features(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 4)
        x, y = dataset[0]
        with np.load(self.npz) as original:
            np.testing.assert_array_equal(x.numpy(), original["ratios"][:2])
        torch.testing.assert_close(y["step"][[1, 2]], torch.tensor([0.75, 0.8]))
        torch.testing.assert_close(y["progress"][[1, 2]], torch.tensor([0.3, 0.7]))
        self.assertEqual(int((y["plateau"] + 0.001 * y["step"]).argmax()), 2)
        _, idle = dataset[1]
        self.assertEqual(float(idle["idle"]), 1)
        self.assertAlmostEqual(float(idle["step"][0]), 0.1)
        self.assertEqual(tuple(idle["step"].shape), (7,))
        offset = self.dataset(predict_offset=1)
        self.assertEqual(len(offset), 3)
        torch.testing.assert_close(offset[0][1]["step"], torch.zeros(7))

    def test_weights_use_window_peaks(self):
        dataset = self.dataset()
        stats = training_pos_weights([dataset])
        self.assertEqual(stats["windows"], 4)
        self.assertEqual(stats["task_positive"], [1, 1, 1, 0, 3, 0, 0])
        np.testing.assert_allclose(stats["task_pos_weight"], [2, 2, 2, 2, 1 / 3, 2, 2])
        self.assertEqual(stats["mistake_pos_weight"], 3)

    def test_old_npz_without_vectors_fails(self):
        with np.load(self.npz) as archive:
            arrays = {k: archive[k] for k in archive.files}
        del arrays["task_id_plateau_vector"]
        np.savez_compressed(self.npz, **arrays)
        with self.assertRaisesRegex(ValueError, "regenerate with build_norm_dataset_tune.py"):
            self.dataset()

    def test_dataset_needs_no_raw_pt(self):
        self.raw.unlink()
        with patch("torch.load", side_effect=AssertionError("Dataset must not read PT files")):
            dataset = self.dataset()
            self.assertEqual(len(dataset), 4)
            torch.testing.assert_close(dataset[0][1]["progress"][[1, 2]], torch.tensor([0.3, 0.7]))

    def test_export_all_splits_uses_same_stats_and_scales_once(self):
        import hashlib
        paths = [self.raw]
        for split, uid in [("aug_val", 1), ("aug_test", 2)]:
            _, raw = make_pair(self.root, split, uid)
            paths.append(raw)
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        stats = dict(ratios_mean=np.array([1., 2.]), ratios_std=np.array([2., 4.]))
        split_dirs = {split: self.root / split / "raw"
                      for split in ("aug_train", "aug_val", "aug_test")}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            export_datasets(split_dirs, stats, ["ratios"])
        for raw in paths:
            data = torch.load(raw, weights_only=False)
            out = raw.parent.parent / "norm_tune" / f"{raw.stem}_norm.npz"
            with np.load(out) as exported:
                expected = ((data["features"]["ratios"].numpy() - [1., 2.]) / [2., 4.]).astype(np.float32)
                np.testing.assert_array_equal(exported["ratios"], expected)
                for key in ("task_id_vector", "task_id_plateau_vector"):
                    np.testing.assert_array_equal(exported[key], data["labels"][key].numpy())
                for key in ("task_progress", "task_progress_vector"):
                    np.testing.assert_array_equal(exported[key], data["labels"][key].numpy() / 100)
                self.assertIn("mistake", exported.files)
                self.assertIn("task_id", exported.files)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])

    def test_uid_split_matches_original_and_second_run_skips(self):
        from data_proc_2d.app.build_norm_dataset import split_by_uid as original_split
        source = self.root / "unsplit"
        source.mkdir()
        files = [f"features__cam-{cam:02d}_uid-{uid:02d}_take-01.pt"
                 for uid in range(10) for cam in range(2)]
        for name in files:
            (source / name).write_bytes(name.encode())
        self.assertEqual(split_by_uid(files), original_split(files))
        split_dirs = {split: self.root / "fresh" / split / "raw"
                      for split in ("train", "val", "test")}
        prepare_splits(source, split_dirs)
        actual = [[path.name for path in sorted(directory.glob("*.pt"))]
                  for directory in split_dirs.values()]
        self.assertEqual(actual, [sorted(part) for part in original_split(sorted(files))])
        self.assertEqual([len(part) for part in actual], [16, 2, 2])
        uid_sets = [{re.search(r"uid-(\d+)", name).group(1) for name in part} for part in actual]
        self.assertFalse(uid_sets[0] & uid_sets[1] or uid_sets[0] & uid_sets[2] or uid_sets[1] & uid_sets[2])
        self.assertFalse(list(source.iterdir()))
        for directory in split_dirs.values():
            for path in directory.glob("*.pt"):
                self.assertEqual(path.read_bytes(), path.name.encode())
        with patch("data_proc_2d.app.build_norm_dataset_tune.split_by_uid",
                   side_effect=AssertionError("Must skip an existing split")):
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_splits(self.root / "missing_source", split_dirs)

    def test_partial_split_stops_before_moving_files(self):
        split_dirs = {split: self.root / split / "raw"
                      for split in ("aug_train", "aug_val", "aug_test")}
        before = self.raw.read_bytes()
        with self.assertRaisesRegex(ValueError, "Only some split folders"):
            prepare_splits(self.root / "missing_source", split_dirs)
        self.assertEqual(self.raw.read_bytes(), before)

    def test_norm_stats_fit_train_only_and_export_all_splits(self):
        train_values = torch.load(self.raw, weights_only=False)["features"]["ratios"].numpy()
        for split, uid in [("aug_val", 1), ("aug_test", 2)]:
            _, raw = make_pair(self.root, split, uid)
            data = torch.load(raw, weights_only=False)
            data["features"]["ratios"] += 100
            torch.save(data, raw)
        split_dirs = {split: self.root / split / "raw"
                      for split in ("aug_train", "aug_val", "aug_test")}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            stats = compute_global_norm_stats(sorted(split_dirs["aug_train"].glob("*.pt")), ["ratios"])
            export_datasets(split_dirs, stats, ["ratios"])
        np.testing.assert_array_equal(stats["ratios_mean"], train_values.mean(axis=0))
        np.testing.assert_array_equal(stats["ratios_std"], train_values.std(axis=0) + 1e-6)
        for split, raw_dir in split_dirs.items():
            raw = next(raw_dir.glob("*.pt"))
            values = torch.load(raw, weights_only=False)["features"]["ratios"].numpy()
            with np.load(raw_dir.parent / "norm_tune" / f"{raw.stem}_norm.npz") as archive:
                expected = (values - stats["ratios_mean"]) / stats["ratios_std"]
                np.testing.assert_array_equal(archive["ratios"], expected)

    def test_interrupted_cache_can_be_retried(self):
        def interrupt_copy(source, target, **kwargs):
            target.write(b"incomplete array")
            raise OSError("Interrupted copy")

        with patch("model_lstm.LSTM_databuilder_tune.shutil.copyfileobj", side_effect=interrupt_copy):
            with self.assertRaisesRegex(OSError, "Interrupted copy"):
                self.dataset()
        self.assertFalse(any(p.is_file() for p in (self.root / "cache").rglob("*")))
        dataset = self.dataset()
        with np.load(self.npz) as original:
            np.testing.assert_array_equal(dataset[0][0].numpy(), original["ratios"][:2])

    def test_no_positive_mistake_has_explicit_error(self):
        with np.load(self.npz) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["mistake"] = np.zeros(8, dtype=np.int64)
        np.savez_compressed(self.npz, **arrays)
        with self.assertRaisesRegex(ValueError, "No positive mistake"):
            training_pos_weights([self.dataset()])

    def test_four_head_shapes_and_checkpoint_roundtrip(self):
        model = AssistLSTM(2, 4, dropout=0.2)
        model.eval()
        x = torch.randn(1, 2, 2)
        outputs = model(x)
        self.assertEqual([tuple(t.shape) for t in outputs], [(1, 7), (1, 7), (1,), (1,)])
        self.assertEqual(model.lstm.dropout, 0)
        self.assertEqual(model.shared[-1].p, 0.2)
        checkpoint = self.root / "model.pth"
        torch.save(model.state_dict(), checkpoint)
        restored = AssistLSTM(2, 4, dropout=0.2).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        for actual, expected in zip(restored(x), outputs):
            torch.testing.assert_close(actual, expected)

    def test_idle_still_has_task_gradients_and_progress_mask(self):
        outputs = [torch.zeros(2, 7, requires_grad=True),
                   torch.ones(2, 7, requires_grad=True),
                   torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)]
        targets = dict(step=torch.zeros(2, 7), plateau=torch.zeros(2, 7),
                       progress=torch.zeros(2, 7), mistake=torch.zeros(2), idle=torch.tensor([0., 1.]))
        targets["plateau"][0, :2] = 1
        criterion = AssistLoss(torch.ones(7), 1)
        loss, parts = criterion(outputs, targets)
        self.assertAlmostEqual(float(parts["progress"].detach()), 1.0)
        self.assertAlmostEqual(float(loss.detach()), 1.5 * np.log(2) + 0.3, places=6)
        loss.backward()
        self.assertTrue(torch.all(outputs[0].grad[1] > 0))
        torch.testing.assert_close(outputs[1].grad[1], torch.zeros(7))
        torch.testing.assert_close(outputs[1].grad[0, 2:], torch.zeros(5))
        self.assertLess(float(outputs[3].grad[1]), 0)

    def test_empty_progress_mask_and_soft_bce(self):
        outputs = [torch.zeros(2, 7, requires_grad=True), torch.ones(2, 7, requires_grad=True),
                   torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)]
        targets = dict(step=torch.full((2, 7), 0.25), plateau=torch.zeros(2, 7),
                       progress=torch.zeros(2, 7), mistake=torch.zeros(2), idle=torch.ones(2))
        loss, parts = AssistLoss(torch.full((7,), 2.), 1)(outputs, targets)
        self.assertEqual(float(parts["progress"].detach()), 0)
        self.assertAlmostEqual(float(parts["step"].detach()), 1.25 * np.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(outputs[1].grad, torch.zeros(2, 7))

    def test_l1_transforms_finite_sampling_and_window_gaps(self):
        data = torch.load(self.raw, weights_only=False)
        data["features"]["polar_azimuth"][:] = 90
        data["features"]["ratios"][0] = torch.tensor([-1., 6.])
        data["features"]["joint_speed"][2, 0] = float("nan")
        torch.save(data, self.raw)
        features, valid = feature_arrays(data, FEATURE_KEYS, L1_PROFILE)
        self.assertEqual(sum(x.shape[1] for x in features.values()), 251)
        np.testing.assert_allclose(features["polar_azimuth"][:, :16], 1, atol=1e-6)
        np.testing.assert_allclose(features["polar_azimuth"][:, 16:], 0, atol=1e-6)
        np.testing.assert_array_equal(features["ratios"][0], [0, 4])
        self.assertFalse(valid[2])
        stats = compute_global_norm_stats([self.raw], profile=L1_PROFILE)
        np.testing.assert_allclose(stats["ratios_mean"], features["ratios"][[0, 7]].mean(axis=0))
        builder = NormDatasetBuilder(stats, profile=L1_PROFILE)
        builder.build_dataset(self.raw, self.npz)
        dataset = self.dataset(feature_keys=FEATURE_KEYS, expected_normalization_id=builder.normalization_id)
        self.assertEqual(dataset.indices.tolist(), [0, 4, 6])
        self.assertEqual(dataset.target_slice.tolist(), [1, 5, 7])
        self.assertEqual(training_pos_weights([dataset])["windows"], 3)
        self.assertTrue(torch.isfinite(dataset[1][0]).all())
        with self.assertRaisesRegex(ValueError, "Wrong feature profile/normalization"):
            self.dataset(expected_normalization_id="wrong-fold")

    def test_l1_fold_split_and_train_only_statistics(self):
        corpus = make_l1_corpus(self.root / "l1")
        for uid in (2, 5, 14):
            path = next((corpus / "original").glob(f"*uid-{uid:02d}_*.pt"))
            data = torch.load(path, weights_only=False)
            data["features"]["joint_speed"] += 1000
            torch.save(data, path)
        fold = prepare_l1_fold(corpus, corpus / "claude_l1", 2)
        manifest = json.loads((fold / "dataset_manifest.json").read_text())
        self.assertEqual(manifest["subjects"]["val"], [5, 14])
        self.assertEqual({k: len(v) for k, v in manifest["files"].items()}, dict(train=24, val=2, test=1))
        self.assertTrue(all("aug" not in p for k in ("val", "test") for p in manifest["files"][k]))
        self.assertFalse(any("aug-02" in p for v in manifest["files"].values() for p in v))
        with np.load(fold / "norm_stats.npz") as stats:
            self.assertLess(float(stats["joint_speed_mean"].max()), 2)
        self.assertEqual(prepare_l1_fold(corpus, corpus / "claude_l1", 2), fold)
        for uid in range(1, 16):
            split = l1_subject_split(uid)
            self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [12, 2, 1])
            self.assertEqual(len(set(sum(split.values(), []))), 15)

    def test_l1_runner_checkpoint_selection_cosine_and_clipping(self):
        import pandas as pd
        from model_lstm.LSTM_loso_tune import run_fold
        corpus = make_l1_corpus(self.root / "runner_l1")
        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
            result = run_fold(corpus, corpus / "claude_l1", self.root / "runs", 2,
                              torch.device("cpu"), settings=dict(
                                  epochs=2, batch_size=128, window_size=2, stride=2, hidden_dim=4,
                                  dataset_cache_dir=str(self.root / "runner_cache")))
            self.assertEqual(clip.call_count, 2)
            self.assertTrue(all(call.args[1] == 1.0 for call in clip.call_args_list))
        run = Path(result["run_dir"])
        metrics = pd.read_csv(run / "metrics.csv")
        np.testing.assert_allclose(metrics.learning_rate, [1e-3, 5e-4])
        f1_summary = json.loads((run / "summary_step_macro_f1.json").read_text())
        loss_summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(f1_summary["best_epoch"], int(metrics.loc[metrics.val_step_macro_f1.idxmax(), "epoch"]))
        self.assertEqual(loss_summary["best_epoch"], int(metrics.loc[metrics.val_loss.idxmin(), "epoch"]))
        self.assertEqual(result["windows"], 4)
        self.assertEqual(result["task_windows"], 3)
        self.assertIn("end_to_end_macro_f1", result)
        predictions = pd.read_csv(run / "test_predictions.csv")
        self.assertEqual(predictions.target_frame.tolist(), [1, 3, 5, 7])
        self.assertEqual(predictions.true_step_8.tolist(), [2, 7, 0, 4])

    def test_loso_summary_averages_subjects(self):
        from model_lstm.LSTM_loso_tune import run_loso
        keys = ("step_macro_f1", "step_balanced_acc", "idle_f1", "mistake_f1",
                "progress_mae_percent", "end_to_end_macro_f1", "end_to_end_acc")
        def fake_fold(data_root, export_root, run_root, uid, device):
            return {"test_uid": uid, **{key: uid / 20 for key in keys}}
        with patch("model_lstm.LSTM_loso_tune.run_fold", side_effect=fake_fold) as fold:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_loso(self.root, self.root, self.root / "summary", torch.device("cpu"))
        self.assertEqual(fold.call_count, 15)
        stats = summary["metrics"]["step_macro_f1"]
        self.assertAlmostEqual(stats["mean"], 0.4)
        self.assertAlmostEqual(stats["std"], np.std(np.arange(1, 16) / 20))

    def test_notebook_one_epoch_through_export(self):
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        notebook = json.loads((MODEL_DIR / "LSTM_train_mirror_tune.ipynb").read_text(encoding="utf-8"))
        corpus = make_l1_corpus(self.root / "notebook_l1")
        prepare_l1_fold(corpus, corpus / "claude_l1", 2)
        # Training/evaluation must be self-contained after the preprocessing step.
        for raw in corpus.rglob("*.pt"):
            raw.unlink()
        ns = {"__name__": "__tune_smoke__"}
        original_show = plt.show
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    self.assertFalse(any(ord(char) < 32 and char not in "\n\r\t" for char in source))
                    if index == 9:
                        source = source[source.index("fold_dir ="):]
                        ns["data_root"] = corpus
                    exec(compile(source, f"tune_cell_{index}", "exec"), ns)
                    if index == 0:
                        ns["ROOT"] = self.root
                        plt.show = lambda: None
                    elif index == 3:
                        ns["device"] = torch.device("cpu")
                    elif index == 7:
                        ns.update(epochs=1, batch_size=2, window_size=2, stride=2, hidden_layer=4)
                    elif index == 21:
                        sampler = ns["loader"].sampler
                        self.assertIsInstance(sampler, torch.utils.data.RandomSampler)
                        self.assertFalse(sampler.replacement)
                        for _ in range(2):
                            self.assertEqual(sorted(sampler), list(range(len(ns["full_dataset"]))))
                run_dir = ns["run_dir"]
                self.assertTrue((run_dir / "best_step_macro_f1.pth").exists())
                self.assertTrue((run_dir / "idle_confusion_matrix.png").exists())
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                self.assertTrue(config["training_shuffle"])
                self.assertEqual(config["optimizer"], "AdamW")
                self.assertEqual(config["scheduler"], "CosineAnnealingLR")
                self.assertEqual(config["gradient_clip"], 1)
                self.assertEqual(config["input_dim"], 251)
                self.assertEqual(config["val_uids"], [5, 14])
                self.assertEqual(ns["best_model_path"].name, "best_step_macro_f1.pth")
                self.assertEqual(config["train_metrics_distribution"], "original_windows")
                self.assertEqual(ns["results"]["windows"], 4)
                self.assertEqual(ns["results"]["task_windows"], 3)
                self.assertEqual(ns["results"]["progress_metric_lanes"], 4)
                self.assertEqual(ns["pred_df"].loc[1, "true_step"], -1)
                self.assertEqual(ns["pred_df"].loc[0, "true_step"], 2)
                self.assertIn("pred_progress_6", ns["pred_df"].columns)
                self.assertEqual(ns["pred_df"].target_frame.tolist(), [1, 3, 5, 7])
        finally:
            plt.show = original_show
            plt.close("all")
            ns.clear()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
