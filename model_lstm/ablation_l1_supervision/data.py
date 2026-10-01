"""L1 features/windows with shared scalar task truth and background targets."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data_proc_2d.app.build_norm_dataset_tune import FEATURE_KEYS
from model_lstm.LSTM_databuilder_tune import AssistSequenceDataset, training_pos_weights
from model_lstm.ablation_step_supervision.data import CLASS_NAMES


class WindowDataset(AssistSequenceDataset):
    def __init__(self, path, norm_id, cache_dir, samples=120, sample_interval=1, hop=10):
        if sample_interval != 1:
            raise ValueError("L1 uses consecutive frames at 30 fps")
        super().__init__(path, window_size=samples, stride=hop, feature_keys=FEATURE_KEYS,
                         cache_dir=cache_dir, expected_normalization_id=norm_id)
        if sum(f.shape[1] for f in self.features) != 251:
            raise ValueError("Expected 251 L1 features")
        if not np.isin(self.task_id, range(8)).all():
            raise ValueError("Invalid integer task_id")
        for values in (self.peak, self.plateau, self.progress):
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError("Invalid vector labels")
        if not np.isin(self.mistake, [0, 1]).all():
            raise ValueError("Invalid mistake labels")
        self.path, self.target_frames = self.npz_path, self.target_slice

    def __getitem__(self, index):
        x, target = super().__getitem__(index)
        step = int(self.task_id[self.target_slice[index]])
        target["peak"] = target.pop("step")
        target["step"] = torch.tensor(step, dtype=torch.long)
        target["idle"] = torch.tensor(float(step == 7))
        return x, target


def load_fold(data_root, uid, cache_dir, samples=120, sample_interval=1, hop=10):
    fold = Path(data_root) / f"fold_{uid:02d}"
    manifest = json.loads((fold / "dataset_manifest.json").read_text(encoding="utf-8"))
    stats = fold / "norm_stats.npz"
    if hashlib.sha256(stats.read_bytes()).hexdigest() != manifest["normalization_sha256"]:
        raise ValueError(f"Normalization hash mismatch: {stats}")
    if manifest["profile"] != "claude_l1_v1" or list(FEATURE_KEYS) != manifest["feature_keys"]:
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
