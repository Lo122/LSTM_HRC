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

json_dir = r"data\video_labels"
pt_dir = r"data\dataset"

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
def load_feature_from_pt(pt_path: str, mean_std) -> np.ndarray:
    data = torch.load(pt_path, weights_only=False)

    xyn_feat = data['landmarks']

    degree_feat = data['feat_degree']
    ratio_feat = data['feat_ratio']
    speed_feat = data['feat_speed']
    acc_feat = data['feat_acc']

    angle_rad = torch.deg2rad(degree_feat)
    sin_feat = torch.sin(angle_rad)
    cos_feat = torch.cos(angle_rad)

    angle_feat = torch.cat([sin_feat, cos_feat], dim=1)

    degree_feat = (angle_feat - mean_std[0]) / mean_std[1]
    ratio_feat = (ratio_feat - mean_std[2]) / mean_std[3]
    speed_feat = (speed_feat - mean_std[4]) / mean_std[5]
    acc_feat = (acc_feat - mean_std[6]) / mean_std[7]

    return [degree_feat.numpy(), ratio_feat.numpy(), speed_feat.numpy(), acc_feat.numpy()],xyn_feat.numpy()



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
    feat_list, xyn_feat = load_feature_from_pt(pt_path, mean_std)  # [T, pose_dim]
    labels = load_json(labels_json_path)

    step_markers = labels.get("step_markers", [])

    num_frames = feat_list[0].shape[0]

    step_soft = generate_step_soft_labels(
    step_markers=step_markers,
    fps=cfg.fps,
    total_frames=num_frames,
    sigma_sec=2.0
    )

    #normalized features
    X_degree = []
    X_ratio = []
    X_speed = []
    X_acc = []

    X_xyn = []  # add raw keypoints as features for visualization and potential future use

    X_step = []

    for frame_idx in range(num_frames):

        t_sec = frame_idx / cfg.fps

        # feature per frame
        x_deg = feat_list[0][frame_idx]  # [pose_dim]
        x_ratio = feat_list[1][frame_idx]  # [pose_dim]
        x_speed = feat_list[2][frame_idx]  # [pose_dim]
        x_acc = feat_list[3][frame_idx]  # [pose_dim]
        x_xyn = xyn_feat[frame_idx]  # [2]

        X_xyn.append(x_xyn)

        X_degree.append(x_deg)
        X_ratio.append(x_ratio)
        X_speed.append(x_speed)
        X_acc.append(x_acc)
        X_step.append(step_soft[frame_idx])

    # -----------------------------
    # pack arrays
    # -----------------------------
    X_degree = np.stack(X_degree, axis=0).astype(np.float32)  # [T, pose_dim]
    X_ratio = np.stack(X_ratio, axis=0).astype(np.float32)   # [T, pose_dim]
    X_speed = np.stack(X_speed, axis=0).astype(np.float32)    # [T, pose_dim]
    X_acc = np.stack(X_acc, axis=0).astype(np.float32)        # [T, pose_dim]

    X_xyn = np.stack(X_xyn, axis=0).astype(np.float32)        # [T, 2]

    X_step = np.stack(X_step, axis=0).astype(np.float32)      # [T]


    np.savez(
        cfg.out_path,

        X_degree=X_degree,      # [T, pose_dim]
        
        X_ratio=X_ratio,
        X_speed=X_speed,
        X_acc=X_acc,

        X_xyn=X_xyn,          # raw keypoints for potential future use

        X_step=X_step,
        pose_dim=X_degree.shape[1],

    )


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
    print(f"Frames: {len(X_degree)}")
    print(f"X_degree: {X_degree.shape}")
    print(f"X_ratio: {X_ratio.shape}")
    print(f"X_speed: {X_speed.shape}")
    print(f"X_acc: {X_acc.shape}")
    print(f"X_xyn: {X_xyn.shape}")
    print(f"X_step: {X_step.shape}")

    print("==============================\n")

import torch
import numpy as np
from tqdm import tqdm

def compute_global_norm_stats(pt_files):

    # all_landmark = [] 
    all_degrees = []
    all_ratios = []
    all_speed = []
    all_accel = []


    for pt_path in pt_files:
        pt_path = os.path.join(pt_dir, pt_path)
        data = torch.load(pt_path, weights_only=False)

        landmark_feat = data['landmarks']       # [T, 17, 2]
        degree_feat = data['feat_degree']       # [T, 8]
        ratio_feat = data['feat_ratio']         # [T, 1]
        speed_feat = data['feat_speed']          # [T, 17]
        acc_feat = data['feat_acc']  # [T, 17]

        angle_rad = torch.deg2rad(degree_feat)
        sin_feat = torch.sin(angle_rad)
        cos_feat = torch.cos(angle_rad)

        angle_feat = torch.cat([sin_feat, cos_feat], dim=1)  # [T, 18]

        # pose_feat = torch.cat([angle_feat, acceleration], dim=1)  # [T, F]

        all_degrees.append(angle_feat.numpy())
        all_ratios.append(ratio_feat.numpy())
        all_speed.append(speed_feat.numpy())
        all_accel.append(acc_feat.numpy())

    # all_feats = np.concatenate(all_feats, axis=0)  # [N, F]
    all_degrees = np.concatenate(all_degrees, axis=0)  # [N, 18]
    all_ratios = np.concatenate(all_ratios, axis=0)    # [N, 1]
    all_speed = np.concatenate(all_speed, axis=0)      # [N, 17]
    all_accel = np.concatenate(all_accel, axis=0)      # [N, 17]

    # mean = all_feats.mean(axis=0)
    # std = all_feats.std(axis=0) + 1e-6
    mean_degree = all_degrees.mean(axis=0)
    std_degree = all_degrees.std(axis=0) + 1e-6
    mean_ratio = all_ratios.mean(axis=0)
    std_ratio = all_ratios.std(axis=0) + 1e-6
    mean_speed = all_speed.mean(axis=0)
    std_speed = all_speed.std(axis=0) + 1e-6
    mean_accel = all_accel.mean(axis=0)
    std_accel = all_accel.std(axis=0) + 1e-6

    return [mean_degree.astype(np.float32), std_degree.astype(np.float32), mean_ratio.astype(np.float32), std_ratio.astype(np.float32), mean_speed.astype(np.float32), std_speed.astype(np.float32), mean_accel.astype(np.float32), std_accel.astype(np.float32)]

if __name__ == "__main__":


    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]

    #return mean and std for each feature type (degree, ratio, speed, accel) in order to normalize data when building dataset
    res_list = compute_global_norm_stats(pt_files)
    np.savez("norm_stats.npz", mean_degree=res_list[0], std_degree=res_list[1], mean_ratio=res_list[2], std_ratio=res_list[3], mean_speed=res_list[4], std_speed=res_list[5], mean_accel=res_list[6], std_accel=res_list[7])


    for pt_file in pt_files:
        pt_path = os.path.join(pt_dir, pt_file)
        out_path = os.path.join("data/built_dataset", os.path.basename(pt_path).replace(".pt", "_norm.npz"))

        json_paths = os.path.join(json_dir, os.path.basename(pt_path).replace(".pt", "_steps.json"))


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