"""Memory-mapped normalized NPZ features, vector targets and valid temporal windows."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


TASK_NAMES = (
    "Pull Cables", "Lift", "Place", "Align", "Screw", "Connect Cables", "Clamp Coupling",
)
LABEL_KEYS = [
    "task_id_vector", "task_id_plateau_vector", "task_progress_vector", "mistake", "task_id",
]


def cached_arrays(source_path, keys, cache_dir):
    """Stream NPZ arrays to the local cache, then open read-only disk mappings."""
    source_path = Path(source_path).resolve()
    stat = source_path.stat()
    signature = json.dumps([str(source_path), stat.st_size, stat.st_mtime_ns, keys, "tune-npz-v2"])
    directory = Path(cache_dir) / hashlib.sha256(signature.encode()).hexdigest()
    paths = [directory / f"{i}.npy" for i in range(len(keys))]
    missing = [(key, path) for key, path in zip(keys, paths) if not path.exists()]
    if missing:
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path) as data, tempfile.TemporaryDirectory(dir=directory) as temporary_dir:
            absent = [key for key in keys if f"{key}.npy" not in data.namelist()]
            if absent:
                raise ValueError(f"Missing NPZ fields {absent}; regenerate with build_norm_dataset_tune.py")
            for key, path in missing:
                temporary = Path(temporary_dir) / path.name
                with data.open(f"{key}.npy") as member, temporary.open("wb") as target:
                    shutil.copyfileobj(member, target, length=1024 * 1024)
                temporary.replace(path)  # Publish only complete arrays.
    return [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]


class AssistSequenceDataset(Dataset):
    def __init__(self, npz_path, window_size=120, stride=10,
                 predict_offset=0, mode="seq2one", feature_keys=None, cache_dir=None,
                 expected_normalization_id=None):
        super().__init__()
        if mode != "seq2one":
            raise ValueError("The four-head LSTM uses seq2one targets only")
        if window_size <= 0 or stride <= 0 or predict_offset < 0:
            raise ValueError("Invalid window_size, stride or predict_offset")
        self.npz_path = Path(npz_path)
        if expected_normalization_id is not None:
            with np.load(self.npz_path, allow_pickle=False) as archive:
                if ("normalization_id" not in archive
                        or archive["normalization_id"].item() != expected_normalization_id
                        or "feature_profile" not in archive
                        or archive["feature_profile"].item() != "claude_l1_v1"):
                    raise ValueError(f"Wrong feature profile/normalization: {self.npz_path}")
        self.feature_keys = ["ratios"] if feature_keys is None else list(feature_keys)
        if not self.feature_keys:
            raise ValueError("feature_keys must not be empty")
        cache_dir = Path(__file__).parent / ".dataset_cache" / "tune" if cache_dir is None else Path(cache_dir)
        arrays = cached_arrays(self.npz_path, self.feature_keys + LABEL_KEYS, cache_dir)
        self.features = arrays[:len(self.feature_keys)]
        self.peak, self.plateau, self.progress, self.mistake, self.task_id = arrays[len(self.feature_keys):]
        self.window_size, self.stride = window_size, stride  # stride remains window hop
        self.predict_offset, self.mode = predict_offset, mode
        self.T = len(self.features[0])
        if any(f.ndim != 2 or len(f) != self.T for f in self.features):
            raise ValueError("Features must have matching (frames, dimensions) shapes")
        if any(v.shape != (self.T, 8) for v in (self.peak, self.plateau, self.progress)):
            raise ValueError("Task vectors must have 8 lanes: tasks 0..6, idle at position 7")
        if self.mistake.shape != (self.T,) or self.task_id.shape != (self.T,):
            raise ValueError("Scalar labels must match the feature frame count")
        starts = np.arange(0, max(0, self.T - window_size - predict_offset + 1), stride)
        with np.load(self.npz_path, allow_pickle=False) as archive:
            valid = archive["valid_frame"] if "valid_frame" in archive else np.ones(self.T, dtype=bool)
        if valid.shape != (self.T,) or valid.dtype != np.bool_:
            raise ValueError("valid_frame must contain one Boolean per original frame")
        for feature in self.features:
            valid &= np.isfinite(feature).all(axis=1)
        invalid_prefix = np.r_[0, np.cumsum(~valid)]
        ends = starts + window_size
        keep = (invalid_prefix[ends] == invalid_prefix[starts]) & valid[ends - 1 + predict_offset]
        self.indices = starts[keep]
        # Actual surviving endpoints, also used by weight/statistics calculations.
        self.target_slice = self.indices + window_size - 1 + predict_offset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        target = end - 1 + self.predict_offset
        x = torch.from_numpy(np.concatenate(
            [feature[start:end] for feature in self.features],
            axis=1, dtype=np.float32, casting="unsafe",
        ))
        targets = {
            "step": self.peak[target, :7],
            "plateau": self.plateau[target, :7],
            "progress": self.progress[target, :7],  # Already scaled to [0, 1] by the exporter.
            "mistake": self.mistake[target],
            "idle": self.plateau[target, 7] >= 0.5,
        }
        return x, {key: torch.tensor(value, dtype=torch.float32) for key, value in targets.items()}


def window_class_counts(datasets):
    """Legacy scalar-label window counts, used only for distribution plots."""
    counts = np.zeros(8, dtype=np.int64)
    for dataset in datasets:
        counts += np.bincount(dataset.task_id[dataset.target_slice].astype(np.int64), minlength=8)
    return counts


def training_pos_weights(datasets, cap=2.0):
    """Count training window targets once to set BCE positive-class weights."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Task pos_weight cap must be positive")
    n_windows = sum(len(d) for d in datasets)
    if n_windows == 0:
        raise ValueError("No training windows")
    task_positive = np.zeros(7, dtype=np.int64)
    mistake_positive = 0
    for dataset in datasets:
        task_positive += (dataset.peak[dataset.target_slice, :7] >= 0.5).sum(axis=0)
        mistake_positive += int(dataset.mistake[dataset.target_slice].sum())
    # A missing task's infinite ratio becomes the documented finite cap.
    task_weight = np.full(7, cap, dtype=np.float64)
    np.divide(n_windows - task_positive, task_positive, out=task_weight, where=task_positive > 0)
    if mistake_positive == 0:
        raise ValueError("No positive mistake windows; uncapped mistake pos_weight is undefined")
    return {
        "windows": n_windows,
        "task_positive": task_positive.tolist(),
        "task_pos_weight": np.minimum(task_weight, cap).tolist(),
        "mistake_positive": mistake_positive,
        "mistake_pos_weight": (n_windows - mistake_positive) / mistake_positive,
    }
