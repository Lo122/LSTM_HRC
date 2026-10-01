"""Read existing fold exports once; share windows, labels and norm across A/B/C."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from model_lstm.LSTM_databuilder_tune import cached_arrays


FEATURE_KEYS = (
    "distance_from_center", "position_x_relative_to_pelvis", "position_y_relative_to_pelvis",
    "position_z_relative_to_pelvis", "joint_angles", "polar_elevation", "ratios",
)
CLASS_NAMES = ("Pull Cables", "Lift", "Place", "Align", "Screw", "Connect Cables", "Clamp Coupling", "Background")


class WindowDataset(Dataset):
    def __init__(self, path, norm_id, cache_dir, samples=40, sample_interval=3, hop=10):
        if min(samples, sample_interval, hop) <= 0:
            raise ValueError("Window parameters must be positive")
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            if archive["normalization_id"].item() != norm_id or archive["feature_profile"].item() != "claude_l1_v1":
                raise ValueError(f"Wrong normalization/profile: {path}")
            valid = archive["valid_frame"].copy()
        arrays = cached_arrays(self.path, list(FEATURE_KEYS) + [
            "task_id", "task_id_vector", "task_progress", "mistake",
        ], cache_dir)
        self.features = arrays[:len(FEATURE_KEYS)]
        self.step, self.peak, self.progress, self.mistake = arrays[len(FEATURE_KEYS):]
        frames = len(self.step)
        if valid.shape != (frames,) or valid.dtype != np.bool_:
            raise ValueError(f"Invalid frame mask: {path}")
        if any(f.ndim != 2 or len(f) != frames for f in self.features):
            raise ValueError(f"Invalid feature shapes: {path}")
        if sum(f.shape[1] for f in self.features) != 91:
            raise ValueError(f"Expected 91 features: {path}")
        if (self.step.shape != (frames,) or self.peak.shape != (frames, 8)
                or self.progress.shape != (frames,) or self.mistake.shape != (frames,)):
            raise ValueError(f"Invalid label shapes: {path}")
        if not np.isin(self.step, range(8)).all() or not np.isin(self.mistake, [0, 1]).all():
            raise ValueError(f"Invalid integer labels: {path}")
        if (not np.isfinite(self.peak).all() or (self.peak < 0).any() or (self.peak > 1).any()
                or not np.isfinite(self.progress).all()):
            raise ValueError(f"Invalid soft/progress labels: {path}")
        for feature in self.features:
            valid &= np.isfinite(feature).all(axis=1)
        self.sample_interval = sample_interval
        self.span = (samples - 1) * sample_interval + 1
        starts = np.arange(0, max(0, frames - self.span + 1), hop)
        invalid = np.r_[0, np.cumsum(~valid)]
        self.starts = starts[invalid[starts + self.span] == invalid[starts]]
        self.target_frames = self.starts + self.span - 1

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        end = start + self.span
        target = end - 1
        x = np.concatenate([f[start:end:self.sample_interval] for f in self.features],
                           axis=1, dtype=np.float32)
        y = dict(step=torch.tensor(self.step[target], dtype=torch.long),
                 peak=torch.tensor(self.peak[target, :7], dtype=torch.float32),
                 progress=torch.tensor(self.progress[target], dtype=torch.float32),
                 mistake=torch.tensor(self.mistake[target], dtype=torch.long))
        return torch.from_numpy(x), y


def load_fold(data_root, uid, cache_dir, samples=40, sample_interval=3, hop=10):
    fold = Path(data_root) / f"fold_{uid:02d}"
    manifest = json.loads((fold / "dataset_manifest.json").read_text(encoding="utf-8"))
    stats = fold / "norm_stats.npz"
    if hashlib.sha256(stats.read_bytes()).hexdigest() != manifest["normalization_sha256"]:
        raise ValueError(f"Normalization hash mismatch: {stats}")
    if manifest["profile"] != "claude_l1_v1" or not set(FEATURE_KEYS).issubset(manifest["feature_keys"]):
        raise ValueError(f"Incompatible feature profile: {fold}")
    tr, va, te = (set(manifest["subjects"][s]) for s in ("train", "val", "test"))
    expected_val = set(np.random.RandomState(1000 + uid).permutation([u for u in range(1, 16) if u != uid])[:2])
    if (len(tr) != 12 or va != expected_val or te != {uid} or tr & va or tr & te or va & te
            or tr | va | te != set(range(1, 16))):
        raise ValueError(f"Invalid LOSO split: {fold}")
    datasets = {}
    for split in ("train", "val", "test"):
        names = manifest["files"][split]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate files in {fold}/{split}")
        parts = []
        for name in names:
            path = fold / name
            file_uid = int(re.search(r"uid-(\d+)", path.name)[1])
            if file_uid not in manifest["subjects"][split] or (split != "train" and "_aug-" in name):
                raise ValueError(f"File does not belong to split: {path}")
            parts.append(WindowDataset(path, manifest["normalization_id"], cache_dir, samples, sample_interval, hop))
        datasets[split] = ConcatDataset(parts)
        if not len(datasets[split]):
            raise ValueError(f"No valid windows: {fold}/{split}")
    return datasets, manifest
