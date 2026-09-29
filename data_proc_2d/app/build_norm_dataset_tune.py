"""Export Claude L1 features/labels with train-only statistics for each LOSO fold.

Edit data_root and test_uid in __main__. Legacy helpers remain for older exports.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
FEATURE_KEYS = [
    "joint_speed", "joint_acceleration",
    "joint_velocity_x", "joint_velocity_y", "joint_velocity_z",
    "joint_acceleration_x", "joint_acceleration_y", "joint_acceleration_z",
    "position_x_relative_to_pelvis", "position_y_relative_to_pelvis", "position_z_relative_to_pelvis",
    "polar_azimuth", "polar_elevation", "joint_angles", "ratios", "distance_from_center",
]
VECTOR_KEYS = ["task_id_vector", "task_id_plateau_vector", "task_progress_vector"]
L1_PROFILE = "claude_l1_v1"


def feature_arrays(data, feature_keys, profile="legacy"):
    """Transform before fitting statistics; preserve frame positions, including gaps."""
    features = {key: data["features"][key].numpy() for key in feature_keys}
    valid = np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in features.values()])
    if profile == L1_PROFILE:
        if "polar_azimuth" in features:
            radians = np.deg2rad(features["polar_azimuth"])
            features["polar_azimuth"] = np.concatenate([np.sin(radians), np.cos(radians)], axis=1)
        if "ratios" in features:
            features["ratios"] = np.clip(features["ratios"], 0, 4)
    elif profile != "legacy":
        raise ValueError(f"Unknown feature profile: {profile}")
    return features, valid


def normalization_id(stats):
    digest = hashlib.sha256()
    for key in sorted(stats):
        value = np.ascontiguousarray(stats[key])
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def compute_global_norm_stats(pt_files, feature_keys=None, profile="legacy"):
    feature_keys = FEATURE_KEYS if feature_keys is None else feature_keys
    feats = {key: [] for key in feature_keys}
    for pt_path in tqdm(pt_files, desc="Computing norm stats"):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        features, valid = feature_arrays(data, feature_keys, profile)
        for key in feature_keys:
            values = features[key]
            # Sample original frame indices first, then reject non-finite rows.
            feats[key].append(values[::7][valid[::7]] if profile == L1_PROFILE else values)

    stats = {}
    for key in feature_keys:
        values = np.concatenate(feats[key], axis=0)
        if not len(values):
            raise ValueError(f"No finite training frames for {key}")
        dtype = np.float64 if profile == L1_PROFILE else None
        stats[f"{key}_mean"] = values.mean(axis=0, dtype=dtype).astype(np.float32)
        stats[f"{key}_std"] = (values.std(axis=0, dtype=dtype) + 1e-6).astype(np.float32)
    return stats


class NormDatasetBuilder:
    def __init__(self, norm_stats, feature_keys=None, profile="legacy"):
        self.norm_stats = norm_stats
        self.feature_keys = FEATURE_KEYS if feature_keys is None else list(feature_keys)
        self.profile = profile
        self.normalization_id = normalization_id(norm_stats)

    def load_and_normalize(self, pt_path):
        # These are trusted, locally generated tensors, including metadata.
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        transformed, valid = feature_arrays(data, self.feature_keys, self.profile)
        features = {}
        for key in self.feature_keys:
            x = transformed[key]
            mean = self.norm_stats[f"{key}_mean"]
            std = self.norm_stats[f"{key}_std"]
            if self.profile == "legacy":
                std = np.where(std < 1e-6, 1.0, std)
            features[key] = ((x - mean) / std).astype(np.float32)
        if self.profile == L1_PROFILE:
            features["valid_frame"] = valid
            features["normalization_id"] = np.array(self.normalization_id)
            features["feature_profile"] = np.array(self.profile)
        return features, data["labels"]

    def build_dataset(self, pt_path, out_path):
        save_dict, labels = self.load_and_normalize(pt_path)
        frames = len(next(iter(save_dict.values())))
        for key in VECTOR_KEYS:
            values = labels[key].numpy().astype(np.float32)
            if values.shape != (frames, 8):
                raise ValueError(f"{pt_path}: {key} must have shape ({frames}, 8)")
            save_dict[key] = values
        # Progress is scaled exactly once, here, for both old and new loaders.
        save_dict["task_progress_vector"] /= 100.0
        save_dict["task_progress"] = labels["task_progress"].numpy().astype(np.float32) / 100.0
        save_dict["mistake"] = labels["mistake"].numpy().astype(np.int64)
        task_id = labels["task_id"].numpy().astype(np.int64)
        save_dict["task_id"] = np.where(task_id == 8, 7, task_id)
        np.savez_compressed(out_path, **save_dict)


def split_by_uid(files, ratios=(0.8, 0.1, 0.1), n_trials=5000, seed=42):
    groups = defaultdict(list)
    for file in files:
        uid = re.search(r"uid-(\d+)", str(file)).group(1)
        groups[uid].append(file)

    if len(groups) < 3:
        raise ValueError("Not enough unique uids to split into 3 parts.")

    rng = random.Random(seed)
    uids = list(groups)
    total = len(files)
    targets = [total * ratio for ratio in ratios]
    best_score = float("inf")
    best_split = None

    for _ in range(n_trials):
        rng.shuffle(uids)
        split = [[], [], []]
        counts = [0, 0, 0]

        for position, uid in enumerate(uids):
            size = len(groups[uid])
            empty = [i for i in range(3) if not split[i]]
            remaining = len(uids) - position
            candidates = empty if remaining == len(empty) else range(3)

            index = min(
                candidates,
                key=lambda i: (
                    (counts[i] + size - targets[i]) ** 2
                    - (counts[i] - targets[i]) ** 2
                ),
            )
            split[index].append(uid)
            counts[index] += size

        score = sum(
            (count - target) ** 2
            for count, target in zip(counts, targets)
        )
        if score < best_score:
            best_score = score
            best_split = [part.copy() for part in split]

    return tuple(
        [file for uid in part for file in groups[uid]]
        for part in best_split
    )


def prepare_splits(original_pt_dir, split_dirs):
    # split_dirs is ordered as train, val, test.
    existing = [list(directory.glob("*.pt")) for directory in split_dirs.values()]
    if all(existing):
        print("Train/val/test already split; skipping split.")
        return
    if any(existing):
        raise ValueError("Only some split folders contain PT files; check the three paths.")

    all_files = sorted(path.name for path in original_pt_dir.glob("*.pt"))
    split_files = split_by_uid(all_files)
    for directory, files in zip(split_dirs.values(), split_files):
        directory.mkdir(parents=True, exist_ok=True)
        for filename in files:
            (original_pt_dir / filename).rename(directory / filename)


def export_datasets(split_dirs, norm_stats, feature_keys=None):
    builder = NormDatasetBuilder(norm_stats, feature_keys)
    for split, raw_dir in split_dirs.items():
        paths = sorted(raw_dir.glob("*.pt"))
        out_dir = raw_dir.parent / "norm_tune"
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in tqdm(paths, desc=f"Export {split}"):
            builder.build_dataset(path, out_dir / f"{path.stem}_norm.npz")
        print(f"Saved {len(paths)} files to {out_dir}")


def l1_subject_split(test_uid):
    if test_uid not in range(1, 16):
        raise ValueError("Claude L1 expects test UID 1..15")
    rest = [uid for uid in range(1, 16) if uid != test_uid]
    val = sorted(np.random.RandomState(1000 + test_uid).permutation(rest)[:2].tolist())
    return {"train": sorted(set(rest) - set(val)), "val": val, "test": [test_uid]}


def l1_file_splits(data_root, test_uid):
    """Select originals and exactly one mirror without moving source files."""
    data_root = Path(data_root)
    originals = sorted((data_root / "original").glob("*.pt"))
    if not originals:
        raise ValueError(f"No originals in {data_root / 'original'}")
    mirrors = {}
    mirror_dirs = [data_root / "augmented_mirror"] + [
        data_root / split / "raw" for split in ("aug_train", "aug_val", "aug_test")
    ]
    for directory in mirror_dirs:
        for path in sorted(directory.rglob("*_aug-01.pt")):
            key = path.stem.removesuffix("_aug-01")
            if key in mirrors and mirrors[key].resolve() != path.resolve():
                raise ValueError(f"Ambiguous mirror copies for {key}")
            mirrors[key] = path
    uid_groups = l1_subject_split(test_uid)
    result = {split: [] for split in uid_groups}
    observed = set()
    for original in originals:
        uid = int(re.search(r"uid-(\d+)", original.name)[1])
        observed.add(uid)
        split = next((name for name, uids in uid_groups.items() if uid in uids), None)
        if split is None:
            raise ValueError(f"Unexpected subject: {original}")
        result[split].append(original)
        if split == "train":
            if original.stem not in mirrors:
                raise ValueError(f"Missing _aug-01 mirror for {original.name}")
            result[split].append(mirrors[original.stem])
    if observed != set(range(1, 16)):
        raise ValueError(f"Expected original recordings for all 15 subjects, found {sorted(observed)}")
    return result


def prepare_l1_fold(data_root, output_root, test_uid):
    """Export one LOSO fold with its own statistics; reuse only matching exports."""
    splits = l1_file_splits(data_root, test_uid)
    fold_dir = Path(output_root) / f"fold_{test_uid:02d}"
    manifest_path = fold_dir / "dataset_manifest.json"
    sources = {name: [{"path": str(p.resolve()), "size": p.stat().st_size,
                       "mtime_ns": p.stat().st_mtime_ns} for p in paths]
               for name, paths in splits.items()}
    identity = dict(profile=L1_PROFILE, feature_keys=FEATURE_KEYS,
                    subjects=l1_subject_split(test_uid), sources=sources,
                    statistics_frame_step=7, statistics_dtype="float64", std_ddof=0,
                    sampling_order="original frame indices, then finite-row filter")
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError(f"Source data/protocol changed; choose a new output_root for {fold_dir}")
        stats_path = fold_dir / "norm_stats.npz"
        if hashlib.sha256(stats_path.read_bytes()).hexdigest() != saved["normalization_sha256"]:
            raise ValueError(f"Normalization file changed: {stats_path}")
        if not all((fold_dir / name).is_file() for paths in saved["files"].values() for name in paths):
            raise ValueError(f"Incomplete export in {fold_dir}; choose a new output_root")
        return fold_dir
    fold_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_global_norm_stats(splits["train"], profile=L1_PROFILE)
    stats_path = fold_dir / "norm_stats.npz"
    np.savez(stats_path, **stats)
    builder = NormDatasetBuilder(stats, profile=L1_PROFILE)
    files = {}
    for split, paths in splits.items():
        (fold_dir / split).mkdir(exist_ok=True)
        files[split] = []
        for path in tqdm(paths, desc=f"L1 UID {test_uid}: {split}"):
            relative = f"{split}/{path.stem}_norm.npz"
            builder.build_dataset(path, fold_dir / relative)
            files[split].append(relative)
    manifest = dict(identity, files=files, normalization_id=builder.normalization_id,
                    normalization_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                    fold_index=test_uid - 1, test_uid=test_uid, input_dim=251)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return fold_dir


if __name__ == "__main__":
    data_root = Path(
        "G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/"
        "ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04"
    )
    # Match test_uid in LSTM_train_mirror_tune.ipynb. No source files are moved.
    test_uid = 2
    print(prepare_l1_fold(data_root, data_root / "claude_l1", test_uid))
