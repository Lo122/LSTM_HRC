# build_dataset.py
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

    # window settings
    window_sec: float = 4.0
    stride_sec: float = 1.0

    # step-local history: keep last M windows from SAME step_id
    history_M: int = 3

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

'''
#compute urgency (assistance probability) and type label for a given window based on future assist events
def compute_urgency_and_type(
    t_end: float,
    step_id: int,
    labels: dict,
    lambda_sec: float
):
    events = labels.get("assist_events", [])

    # filter same step & future
    future_events = [
        e for e in events
        if int(e.get("step_id", -1)) == step_id
        and float(e["t"]) >= t_end
    ]

    if len(future_events) == 0:
        return 0, 0.0, -1.0  # no future event

    future_events.sort(key=lambda x: float(x["t"]))
    e = future_events[0]

    t_event = float(e["t"])
    delta_t = t_event - t_end

    if delta_t < 0:
        return 0, 0.0, -1.0

    urgency = np.exp(-delta_t / lambda_sec)

    y_type = int(e.get("type", 0))

    return y_type, float(urgency), float(delta_t)
'''
'''
{
    "pose_seq": ...,
    "element_context": [L, W],
    "env_context": [distance, travel_time],
    "step_label": int,
    "assist_label": int,
    "urgency_label": float (0~1),
}
'''

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
def load_pose_from_json(pose_json_path: str) -> np.ndarray:
    """
    Returns:
      pose_arr: [num_frames, pose_dim] float32

    We flatten 33 landmarks per frame.
    Each landmark: x,y,z,visibility (if present)
    """
    data = load_json(pose_json_path)
    frames = data["frames"]

    # infer landmark dim
    first_nonempty = None
    for fr in frames:
        if len(fr.get("landmarks", [])) > 0:
            first_nonempty = fr
            break

    if first_nonempty is None:
        raise ValueError("pose.json contains no detected landmarks at all.")

    lm0 = first_nonempty["landmarks"][0]
    use_visibility = "visibility" in lm0
    per_lm_dim = 4 if use_visibility else 3
    pose_dim = 33 * per_lm_dim

    all_feats = []
    for fr in frames:
        lms = fr.get("landmarks", [])
        if len(lms) == 0:
            feat = np.zeros((pose_dim,), dtype=np.float32)
        else:
            pts = []
            for lm in lms:
                pts.extend([lm["x"], lm["y"], lm["z"]])
                if use_visibility:
                    pts.append(lm["visibility"])
            feat = np.array(pts, dtype=np.float32)
        all_feats.append(feat)

    pose_arr = np.stack(all_feats, axis=0).astype(np.float32)
    return pose_arr


# change the key if npz structure changes
def load_feature_from_pt(pt_path: str, mean,std) -> np.ndarray:
    data = torch.load(pt_path, weights_only=False)
    landmarks = data['landmarks']
    degree_feat = data['features']
    speed = data['speed']
    acceleration = data['acceleration']

    T = landmarks.shape[0]
    landmarks_flatten = landmarks.reshape(T, -1)

    angle_rad = torch.deg2rad(degree_feat)
    sin_feat = torch.sin(angle_rad)
    cos_feat = torch.cos(angle_rad)

    angle_feat = torch.cat([sin_feat, cos_feat], dim=1)

    #concatenate
    pose_feat = torch.cat([angle_feat, acceleration], dim=1)

    #normalize
    pose_feat = pose_feat.numpy()
    pose_feat = (pose_feat - mean) / std
    return pose_feat.astype(np.float32)



def generate_step_soft_labels(
    step_markers,
    fps,
    total_frames,
    num_steps=None,
    sigma_sec=5.0
):
    """
    Generate soft step labels [T, num_steps] using Gaussian peaks.

    Args:
        step_markers: [{"timestamp": float, "step_id": int}, ...]
        fps: int
        total_frames: int
        num_steps: optional int
        sigma_sec: float (spread in seconds)

    Returns:
        soft_labels: np.ndarray [T, num_steps]
    """

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
    mean: np.ndarray,
    std: np.ndarray
):
    # 1) load data
    pose = load_feature_from_pt(pt_path, mean, std)  # [T, pose_dim]
    labels = load_json(labels_json_path)

    step_markers = labels.get("step_markers", [])


    # events = labels.get("assist_events", [])
    # if len(events) > 0:
    #     step_id = events[0].get("step_id", 0)
    #     element_id = events[0].get("element_id", 0)
    # else:
    #     step_id = 0
    #     element_id = 0

    # design_data = load_json(design_json_path)

    # elem_feat = design_data.get(str(element_id), {}).get("element_features", [])
    # env_feat = design_data.get(str(step_id), {}).get("env_features", [])

    # elem_feat = np.array(elem_feat, dtype=np.float32)
    # env_feat = np.array(env_feat, dtype=np.float32)

    # pose shape: [num_frames, pose_dim]
    assert pose.ndim == 2, f"pose must be [frames, pose_dim], got {pose.shape}"
    num_frames, pose_dim = pose.shape

    # ============================================================
    # FRAME-LEVEL LOOP
    # ============================================================

    step_soft = generate_step_soft_labels(
    step_markers=step_markers,
    fps=cfg.fps,
    total_frames=num_frames,
    sigma_sec=2.0
    )

    X_pose = []
    X_step = []
    X_elem = []
    X_env = []

    Y_type = []
    Y_urgency = []
    Y_eta = []

    for frame_idx in range(num_frames):

        t_sec = frame_idx / cfg.fps

        # pose per frame
        x = pose[frame_idx]  # [pose_dim]

        # y_type, y_urgency, y_eta = compute_urgency_and_type(
        #     t_sec=t_sec,
        #     step_id=step_id,
        #     labels=labels,
        #     lambda_sec=cfg.urgency_lambda
        # )

        X_pose.append(x)
        X_step.append(step_soft[frame_idx])
        # X_elem.append(elem_feat)
        # X_env.append(env_feat)

        # Y_type.append(y_type)
        # Y_urgency.append(y_urgency)
        # Y_eta.append(y_eta)

    # -----------------------------
    # pack arrays
    # -----------------------------
    X_pose = np.stack(X_pose, axis=0).astype(np.float32)  # [T, pose_dim]
    X_step = np.stack(X_step, axis=0).astype(np.float32)   # [T]
    # Y_type = np.array(Y_type, dtype=np.int64)
    # Y_urgency = np.array(Y_urgency, dtype=np.float32)
    # Y_eta = np.array(Y_eta, dtype=np.float32)

    # pad static feats

    # def pad_feats(list_of_arr):
    #     max_d = max([a.shape[0] for a in list_of_arr]) if len(list_of_arr) > 0 else 0
    #     out = np.zeros((len(list_of_arr), max_d), dtype=np.float32)
    #     for i, a in enumerate(list_of_arr):
    #         if a.shape[0] > 0:
    #             out[i, :a.shape[0]] = a
    #     return out, max_d

    # X_elem, elem_dim = pad_feats(X_elem)
    # X_env, env_dim = pad_feats(X_env)

    # -----------------------------
    # save
    # -----------------------------
    np.savez(
        cfg.out_path,

        X_pose=X_pose,      # [T, pose_dim]
        X_step=X_step,
        # X_elem=X_elem,
        # X_env=X_env,

        # Y_type=Y_type,
        # Y_urgency=Y_urgency,
        # Y_eta=Y_eta,

        pose_dim=pose_dim,
        # elem_dim=elem_dim,
        # env_dim=env_dim,

        # fps=cfg.fps,
        # urgency_lambda=cfg.urgency_lambda,
    )


#region - not used for now
# CSV for visualization
    # with open(cfg.out_path.replace(".npz", ".csv"), "w", newline='') as f:
    #     writer = csv.writer(f)
    #     header = ["frame_index", "X_pose", "X_step"] #, "X_elem", "X_env", "Y_type", "Y_urgency", "Y_eta"]  
    #     # header = ["frame_idex","X_pose", "X_step", "X_elem", "X_env", "Y_type", "Y_urgency", "Y_eta"]
    #     writer.writerow(header)
    #     for i in range(len(X_pose)):
    #         row = [
    #             i,
    #             X_pose[i].tolist(),
    #             X_step[i].tolist(),
    #             # X_elem[i].tolist(),
    #             # X_env[i].tolist(),
    #             # Y_type[i].item(),
    #             # Y_urgency[i].item(),
    #             # Y_eta[i].item()
    #         ]
    #         writer.writerow(row)
#endregion

    print("\n==============================")
    print("Frame-level dataset built successfully!")
    print("==============================")
    print(f"Saved: {cfg.out_path}")
    print(f"Frames: {len(X_pose)}")
    print(f"X_pose: {X_pose.shape}")
    print(f"X_step: {X_step.shape}")
    # print(f"X_elem: {X_elem.shape}")
    # print(f"X_env : {X_env.shape}")
    # print(f"Y_urgency: {Y_urgency.shape}")
    print("==============================\n")

import torch
import numpy as np
from tqdm import tqdm

def compute_global_norm_stats(pt_files):

    all_feats = []

    for pt_path in pt_files:
        pt_path = os.path.join(pt_dir, pt_path)
        data = torch.load(pt_path, weights_only=False)

        degree_feat = data['features']       # [T, 9]
        acceleration = data['acceleration']  # [T, 17]

        angle_rad = torch.deg2rad(degree_feat)
        sin_feat = torch.sin(angle_rad)
        cos_feat = torch.cos(angle_rad)

        angle_feat = torch.cat([sin_feat, cos_feat], dim=1)  # [T, 18]

        pose_feat = torch.cat([angle_feat, acceleration], dim=1)  # [T, F]

        all_feats.append(pose_feat.numpy())

    all_feats = np.concatenate(all_feats, axis=0)  # [N, F]

    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-6

    return mean.astype(np.float32), std.astype(np.float32)


if __name__ == "__main__":


    pt_files = [f for f in os.listdir(pt_dir) if f.endswith(".pt")]

    #degree and acceleration
    mean, std = compute_global_norm_stats(pt_files)
    np.savez("norm_stats.npz", mean=mean, std=std)
    print("Global feature mean:", mean)
    print("Global feature std:", std)
    
    for pt_file in pt_files:
        pt_path = os.path.join(pt_dir, pt_file)
        out_path = os.path.join("data/built_dataset", os.path.basename(pt_path).replace(".pt", "_window.npz"))

        json_paths = os.path.join(json_dir, os.path.basename(pt_path).replace(".pt", "_steps.json"))


        cfg = Config(
            fps=30,
            window_sec=1, #float
            stride_sec=1, #float
            history_M=1, #int
            out_path=out_path
        )

        build_dataset(
            pt_path=pt_path,
            labels_json_path=json_paths,
            # design_json_path="data/dataset/design_data.json",
            cfg=cfg,
            mean=mean,
            std=std
        )