# concat various features, normalize data and build frame-level dataset for modeling
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import csv
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

json_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\annotations\all"
pt_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\train\raw"
pt_test_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\test\raw"
pt_val_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\val\raw"
pt_test_norm_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\train\norm"
# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    fps: int = 30

    # urgency decay parameter (seconds)
    urgency_lambda: float = 2.0

    # dataset output
    out_path: str = "dataset.npz"


# ============================================================
# Helpers
# ============================================================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def time_to_frame(t: float, fps: int) -> int:
    return int(round(t * fps))


def safe_slice(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return arr[start:end], but pad if out of bounds."""
    n = len(arr)
    start0 = max(0, start)
    end0 = min(n, end)

    chunk = arr[start0:end0]

    # pad left/right with zeros if needed
    if start < 0 or end > n:
        pad_left = max(0, -start)
        pad_right = max(0, end - n)

        if pad_left > 0:
            chunk = np.concatenate([np.zeros((pad_left, arr.shape[1]), dtype=arr.dtype), chunk], axis=0)
        if pad_right > 0:
            chunk = np.concatenate([chunk, np.zeros((pad_right, arr.shape[1]), dtype=arr.dtype)], axis=0)

    return chunk


def compute_urgency_and_type(
    t_sec,
    step_id,
    labels,
    lambda_sec
):
    segments = labels.get("assist_events", [])

    seg = None
    for s in segments:
        if int(s["step_id"]) == step_id:
            seg = s
            break

    if seg is None:
        return 0, 0.0, -1.0
    
    delta = 0.0

    start = float(seg["start"])
    end   = float(seg["end"])
    assist_type = int(seg.get("assist_type", 0))

    if t_sec < start:
        delta = start - t_sec
        urgency = np.exp(-delta / lambda_sec)

    elif t_sec <= end:
        urgency = 1.0

    else:
        delta = t_sec - end
        urgency = np.exp(-delta / lambda_sec)

    return assist_type, float(urgency), float(delta)


# ======ELEMENT ENVRIONMENT STEP LOCAL PARAMETERS=========== NEEDS TO CHANGE=================================================
# ============================================================
# Core: step_id lookup
# ============================================================
# def lookup_step_meta(t: float, meta: dict):
#     """
#     Returns (step_id, element_feat, env_feat, pos_feat)
#     """
#     segments = meta.get("step_segments", [])
#     for seg in segments:
#         if seg["start"] <= t < seg["end"]:
#             step_id = int(seg.get("step_id", 0))
#             element_feat = np.array(seg.get("element_feat", []), dtype=np.float32)
#             env_feat = np.array(seg.get("env_feat", []), dtype=np.float32)
#             pos_feat = np.array(seg.get("pos_feat", []), dtype=np.float32)
#             return step_id, element_feat, env_feat, pos_feat

#     # fallback
#     return 0, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

# ============================================================
# Pose loading: pose.json -> pose.npy (in-memory)
# ============================================================

# normalize features

        # velocity_xy = data['velocity_xy']          # [T, 17]
        # acceleration_xy = data['acceleration_xy']  # [T, 17]

        # pol_vectors = data['pol_vectors']          # [T, 17]
        # angles = data['pol_angles']                    # [T, 17]

        # joint_angles = data['joint_angles']                   
        # ratios = data['ratios']                    # [T, 1] 
        # dist_ratios = data['dist_ratios']            # [T, 1]

def load_feature_from_pt(pt_path: str, mean_std) -> np.ndarray:
    data = torch.load(pt_path, weights_only=False)

    velocity_xy = data['velocity_xy']          # [T, 17]
    acceleration_xy = data['acceleration_xy']  # [T, 17]

    pol_vectors = data['pol_vectors']          # [T, 17]
    angles = data['pol_angles']                    # [T, 17]

    joint_angles = data['joint_angles']                   
    ratios = data['ratios']                    # [T, 1] 
    dist_ratios = data['dist_ratios']            # [T, 1]

    #norm them with stored global std/mean
    feat_vel_xy = (velocity_xy - mean_std[0])/mean_std[1]
    feat_acc_xy = (acceleration_xy - mean_std[2])/mean_std[3]
    feat_pool_vectors = (pol_vectors - mean_std[4])/mean_std[5]
    feat_angles = (angles - mean_std[6])/mean_std[7]
    feat_joint_angles = (joint_angles - mean_std[8])/mean_std[9]
    feat_ratios = (ratios - mean_std[10])/mean_std[11]
    feat_dist_ratios = (dist_ratios - mean_std[12])/mean_std[13]

    return [feat_vel_xy,feat_acc_xy,feat_pool_vectors,feat_angles,feat_joint_angles,feat_ratios,feat_dist_ratios]


def generate_step_soft_labels(
    step_markers,
    fps,
    total_frames,
    num_steps=None,
    sigma_sec=5.0
):

    # ---- step_id mapping ----
    if num_steps is None:
        step_ids = sorted(list(set(m["step_id"] for m in step_markers)))
        step_to_idx = {sid: i for i, sid in enumerate(step_ids)}
        num_steps = len(step_ids)
    else:
        step_to_idx = {sid: sid for sid in range(num_steps)}

    # ---- convert peaks ----
    peak_frames = []
    peak_steps = []

    for m in step_markers:
        frame = int(round(m["timestamp"] * fps))
        peak_frames.append(frame)
        peak_steps.append(step_to_idx[m["step_id"]])

    peak_frames = np.array(peak_frames)
    peak_steps = np.array(peak_steps)

    # ---- build field ----
    T = total_frames
    soft = np.zeros((T, num_steps), dtype=np.float32)

    sigma = sigma_sec * fps
    t_axis = np.arange(T)

    for pf, ps in zip(peak_frames, peak_steps):
        prob = np.exp(- (t_axis - pf) ** 2 / (2 * sigma ** 2))
        soft[:, ps] += prob

    # ---- normalize ----
    s = soft.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    soft = soft / s

    return soft

# ============================================================
# Build dataset (PURE FRAME LEVEL VERSION)
# ============================================================
def build_dataset(
    pt_path: str,
    labels_json_path: str,
    cfg: Config,
    mean_std: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    # 1) load data
    # feat_list, xyn_feat = load_feature_from_pt(pt_path, mean_std)  # [T, pose_dim]
    feat_res = load_feature_from_pt(pt_path, mean_std)

    labels = load_json(labels_json_path)

    step_markers = labels.get("step_markers", [])

    num_frames = feat_res[0].shape[0]

    step_soft = generate_step_soft_labels(
    step_markers=step_markers,
    fps=cfg.fps,
    total_frames=num_frames,
    sigma_sec=2.0
    )

    #stack all
    (
        X_vel_xy,
        X_acc_xy,
        X_pool_vec,
        X_angles,
        X_joint_angles,
        X_ratios,
        X_dist_ratios,
    ) = feat_res

    X_step = step_soft

    def stack_float32(x):
        return np.stack(x, axis=0).astype(np.float32)

    to_stack = {
        "X_vel_xy": X_vel_xy,
        "X_acc_xy": X_acc_xy,
        "X_pool_vec": X_pool_vec,
        "X_angles": X_angles,
        "X_joint_angles": X_joint_angles,
        "X_ratios": X_ratios,
        "X_dist_ratios": X_dist_ratios,
        "X_step": X_step,
    }

    stacked = {
        k: stack_float32(v) if isinstance(v, list) else np.asarray(v, dtype=np.float32)
        for k, v in to_stack.items()
    }

    # save
    np.savez(cfg.out_path, **stacked)
    
#region - not used for now
# CSV for visualization
    # with open(cfg.out_path.replace(".npz", ".csv"), "w", newline='') as f:
    #     writer = csv.writer(f)
    #     header = ["frame_index", "X_degree", "X_ratio", "X_speed", "X_acc", "X_step"] #, "X_elem", "X_env", "Y_type", "Y_urgency", "Y_eta"]  
    #     # header = ["frame_idex","X_pose", "X_step", "X_elem", "X_env", "Y_type", "Y_urgency", "Y_eta"]
    #     writer.writerow(header)
    #     for i in range(len(X_degree)):
    #         row = [
    #             i,
    #             X_degree[i].tolist(),
    #             X_ratio[i].tolist(),
    #             X_speed[i].tolist(),
    #             X_acc[i].tolist(),
    #             X_step[i].tolist(),
    #         ]
    #         writer.writerow(row)
#endregion

    print("\n==============================")
    print("Frame-level dataset built successfully!")
    print("==============================")
    print(f"Saved: {cfg.out_path}")

    print("==============================\n")

import torch
import numpy as np
from tqdm import tqdm

def compute_global_norm_stats(pt_files):
    feats = {
        "vel_xy": [],
        "acc_xy": [],
        "pol_vec": [],
        "pol_ang": [],
        "joint_ang": [],
        "ratios": [],
        "dist_ratios": [],
    }

    for pt_path in pt_files:
        pt_path = os.path.join(pt_dir, pt_path)
        data = torch.load(pt_path, weights_only=False)

        feats["vel_xy"].append(data["velocity_xy"].numpy())
        feats["acc_xy"].append(data["acceleration_xy"].numpy())
        feats["pol_vec"].append(data["pol_vectors"].numpy())
        feats["pol_ang"].append(data["pol_angles"].numpy())
        feats["joint_ang"].append(data["joint_angles"].numpy())
        feats["ratios"].append(data["ratios"].numpy())
        feats["dist_ratios"].append(data["dist_ratios"].numpy())

        #nan check:
        np_check = data["dist_ratios"].numpy()
        nan_mask = np.isnan(np_check)
        if nan_mask.sum() != 0: 
            idx = np.argwhere(nan_mask)
            print("First few NaN positions:", idx[:10])

            print(f"problematic pt: {pt_path}")

    stats = {}

    for k, v in feats.items():
        v = np.concatenate(v, axis=0)

        stats[f"{k}_mean"] = v.mean(axis=0).astype(np.float32)
        stats[f"{k}_std"] = (v.std(axis=0) + 1e-6).astype(np.float32)




    ordered_keys = [
        "vel_xy",
        "acc_xy",
        "pol_vec",
        "pol_ang",
        "joint_ang",
        "ratios",
        "dist_ratios",
    ]

    out_list = []
    for k in ordered_keys:
        mean = stats[f"{k}_mean"]
        std = stats[f"{k}_std"]
        out_list.extend([mean, std])

    return out_list


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

    #split train/test dataset 20/80
    train_files, test_files = split_raw_pt_files(pt_dir, pt_test_dir, test_ratio=0.2)

    #split test/val dataset 10/10
    test_files, val_files = split_raw_pt_files(pt_test_dir,pt_val_dir,test_ratio=0.5)

    #return mean and std for each feature type (degree, ratio, speed, accel) in order to normalize data when building dataset
    res_list = compute_global_norm_stats(train_files)
    np.savez("data_proc_2d/dataset/norm.npz", mean_vel_xy=res_list[0], std_vel_xy=res_list[1], mean_acc_xy=res_list[2], std_acc_xy=res_list[3], 
             mean_pol_vec=res_list[4], std_pol_vec=res_list[5], mean_pol_ang=res_list[6], std_pol_ang=res_list[7], 
             mean_joint_ang=res_list[8], std_joint_ang=res_list[9], mean_ratios=res_list[10], std_ratios=res_list[11], 
             mean_dist_ratios=res_list[12], std_dist_ratios=res_list[13])


    for pt_file in train_files:
        pt_path = os.path.join(pt_dir, pt_file)
        out_dir = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\dataset\train\norm"
        
        out_path = os.path.join(pt_test_norm_dir, os.path.basename(pt_path).replace(".pt", "_norm.npz"))
    

        json_paths = os.path.join(json_dir, os.path.basename(pt_path).replace("pt","json").replace("features","label"))


        cfg = Config(
            fps=30,
            out_path=out_path
        )

        build_dataset(
            pt_path=pt_path,
            labels_json_path=json_paths,
            # design_json_path="data/dataset/design_data.json",
            cfg=cfg,
            mean_std = res_list
        )