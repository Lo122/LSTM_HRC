# concat various features, normalize data and build frame-level dataset for modeling
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import csv
import torch
import datetime
from tqdm import tqdm

'''
 - read segmented pt files
 - split pt files into train/test/val folders
 - compute global mean/std for each feature type using train set
 - store mean/std for later normalization when building dataset
 - build dataset by normalizing features and stacking them together, then save as npz file
'''
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# ============================================================
# Config
# ===========================================================
# json_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\annotations\all"
pt_dir = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos\dataset\original"
pt_train_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\train\raw"
# pt_val_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\val\raw"
pt_test_norm_dir = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos\dataset\original\norm"

FEATURE_KEYS = [
    "velocity_scale",
    "acceleration_scale",
    "velocity_xy",
    "acceleration_xy",
    "pol_vectors",
    "pol_distance",
    "pol_angles",
    "pol_distance_velocity",
    "pol_angluer_velocity",
    "joint_angles",
    "ratios",
    "dist_ratios",
]

def compute_global_norm_stats(pt_files):
    feats = {k: [] for k in FEATURE_KEYS}

    for fname in tqdm(pt_files, desc="Computing norm stats"):
        pt_path = os.path.join(pt_train_dir, fname)
        data = torch.load(pt_path, map_location="cpu")

        for k in FEATURE_KEYS:
            feats[k].append(data["features"][k].numpy())

    stats = {}

    for k in FEATURE_KEYS:
        v = np.concatenate(feats[k], axis=0)

        stats[f"{k}_mean"] = v.mean(axis=0).astype(np.float32)
        stats[f"{k}_std"] = (v.std(axis=0) + 1e-6).astype(np.float32)

    return stats


# ============================================================
# Build dataset (PURE FRAME LEVEL VERSION)
# ============================================================
class NormDatasetBuilder:
    def __init__(self, norm_stats: dict, feature_keys=None):
        self.norm_stats = norm_stats
        self.feature_keys = FEATURE_KEYS if feature_keys is None else feature_keys

    def load_and_normalize(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu")

        feats = data["features"]
        labels = data["labels"]

        out = {}

        for k in self.feature_keys:
            x = feats[k].numpy()

            mean = self.norm_stats[f"{k}_mean"]
            std = self.norm_stats[f"{k}_std"]

            std = np.where(std < 1e-6, 1.0, std)

            out[k] = ((x - mean) / std).astype(np.float32)

        # ========================================================
        # Feature Engineering Conclusion
        # ========================================================

        # feature combo (add more if resonable)
        # pol_angles (T,13) + joint_angles (T,7) -> (T,20)
        out["angles_combined"] = np.concatenate(
            [out["pol_angles"], out["joint_angles"]],
            axis=1
        ).astype(np.float32)

        return out, labels

    def build_dataset(self, pt_path, out_path):
        feat_dict, labels = self.load_and_normalize(pt_path)

        save_dict = {}

        # ===== save features =====
        for k, v in feat_dict.items():
            save_dict[k] = v

        # ===== save labels =====
        save_dict["step_id"] = labels["step_id"].numpy().astype(np.int64)
        save_dict["step_id_vector"] = labels["step_id_vector"].numpy().astype(np.float32)
        save_dict["status_id"] = labels["status_id"].numpy().astype(np.int64)
        save_dict["task_progress"] = labels["task_progress"].numpy().astype(np.float32)/ 100.0  # normalize to [0,1]

        # ===== save npz =====
        np.savez_compressed(out_path, **save_dict)

        print(f"Saved: {out_path}")

def split_raw_pt_files(pt_dir, pt_test_dir, test_ratio=0.2, seed=42):
    pt_files = sorted([f for f in os.listdir(pt_dir) if f.endswith(".pt")])

    if len(os.listdir(pt_test_dir)) <= 1:
        rng = np.random.default_rng(seed)
        rng.shuffle(pt_files)

        split_idx = int(len(pt_files) * test_ratio)
        test_files = pt_files[:split_idx]
        train_files = pt_files[split_idx:]

        for f in test_files:
            os.rename(
                os.path.join(pt_dir, f),
                os.path.join(pt_test_dir, f)
            )

    else:
        train_files = sorted([f for f in os.listdir(pt_dir) if f.endswith(".pt")])
        test_files = sorted([f for f in os.listdir(pt_test_dir) if f.endswith(".pt")])

    return train_files, test_files



if __name__ == "__main__":

    today_date = datetime.datetime.now().strftime("%Y-%m-%d")

    seg_pt_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\segment"
    seg_pt_test_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\test\raw"
    seg_pt_val_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\val\raw"
    seg_pt_train_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\train\raw"
    
    #split train/test dataset 80/20
    # train_files, test_files = split_raw_pt_files(seg_pt_train_dir, seg_pt_test_dir, test_ratio=0.2)

    #split test/val dataset 10/10
    # test_files, val_files = split_raw_pt_files(seg_pt_test_dir,seg_pt_val_dir,test_ratio=0.5)

    train_files = sorted([f for f in os.listdir(seg_pt_train_dir) if f.endswith(".pt")])
    #return mean and std for each feature type (degree, ratio, speed, accel) in order to normalize data when building dataset
    norm_stats  = compute_global_norm_stats(train_files)
    np.savez(f"data_proc_2d/dataset/norm_{today_date}.npz", **norm_stats)
    builder = NormDatasetBuilder(norm_stats)

    out_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\train\norm" 
    for pt_file in train_files:
        pt_path = os.path.join(seg_pt_train_dir, pt_file)
        
        
        
        out_path = os.path.join(out_dir, os.path.basename(pt_path).replace(".pt", "_norm.npz"))
    
        # json_paths = os.path.join(json_dir, os.path.basename(pt_path).replace("pt","json").replace("features","label"))

        builder.build_dataset(
            pt_path=pt_path,
            out_path=out_path
        )

    print("\nAll datasets built successfully.")