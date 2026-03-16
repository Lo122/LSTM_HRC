# build_dataset.py
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import csv


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


# ============================================================
# Build dataset (PURE FRAME LEVEL VERSION)
# ============================================================
def build_dataset(
    pose_json_path: str,
    labels_json_path: str,
    design_json_path: str,
    cfg: Config
):
    # 1) load data
    pose = load_pose_from_json(pose_json_path)
    labels = load_json(labels_json_path)

    events = labels.get("assist_events", [])
    if len(events) > 0:
        step_id = events[0].get("step_id", 0)
        element_id = events[0].get("element_id", 0)
    else:
        step_id = 0
        element_id = 0

    design_data = load_json(design_json_path)

    elem_feat = design_data.get(str(element_id), {}).get("element_features", [])
    env_feat = design_data.get(str(step_id), {}).get("env_features", [])

    elem_feat = np.array(elem_feat, dtype=np.float32)
    env_feat = np.array(env_feat, dtype=np.float32)

    # pose shape: [num_frames, pose_dim]
    assert pose.ndim == 2, f"pose must be [frames, pose_dim], got {pose.shape}"
    num_frames, pose_dim = pose.shape

    # ============================================================
    # FRAME-LEVEL LOOP
    # ============================================================

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

        y_type, y_urgency, y_eta = compute_urgency_and_type(
            t_sec=t_sec,
            step_id=step_id,
            labels=labels,
            lambda_sec=cfg.urgency_lambda
        )

        X_pose.append(x)
        X_step.append(step_id)
        X_elem.append(elem_feat)
        X_env.append(env_feat)

        Y_type.append(y_type)
        Y_urgency.append(y_urgency)
        Y_eta.append(y_eta)

    # -----------------------------
    # pack arrays
    # -----------------------------
    X_pose = np.stack(X_pose, axis=0).astype(np.float32)  # [T, pose_dim]
    X_step = np.array(X_step, dtype=np.int64)             # [T]
    Y_type = np.array(Y_type, dtype=np.int64)
    Y_urgency = np.array(Y_urgency, dtype=np.float32)
    Y_eta = np.array(Y_eta, dtype=np.float32)

    # pad static feats
    def pad_feats(list_of_arr):
        max_d = max([a.shape[0] for a in list_of_arr]) if len(list_of_arr) > 0 else 0
        out = np.zeros((len(list_of_arr), max_d), dtype=np.float32)
        for i, a in enumerate(list_of_arr):
            if a.shape[0] > 0:
                out[i, :a.shape[0]] = a
        return out, max_d

    X_elem, elem_dim = pad_feats(X_elem)
    X_env, env_dim = pad_feats(X_env)

    # -----------------------------
    # save
    # -----------------------------
    np.savez(
        cfg.out_path,

        X_pose=X_pose,      # [T, pose_dim]
        X_step=X_step,
        X_elem=X_elem,
        X_env=X_env,

        Y_type=Y_type,
        Y_urgency=Y_urgency,
        Y_eta=Y_eta,

        pose_dim=pose_dim,
        elem_dim=elem_dim,
        env_dim=env_dim,

        fps=cfg.fps,
        urgency_lambda=cfg.urgency_lambda,
    )

    # CSV for visualization
    with open(cfg.out_path.replace(".npz", ".csv"), "w", newline='') as f:
        writer = csv.writer(f)
        header = ["frame_idex","X_pose", "X_step", "X_elem", "X_env", "Y_type", "Y_urgency", "Y_eta"]
        writer.writerow(header)
        for i in range(len(X_pose)):
            row = [
                i,
                X_pose[i].tolist(),
                X_step[i].item(),
                X_elem[i].tolist(),
                X_env[i].tolist(),
                Y_type[i].item(),
                Y_urgency[i].item(),
                Y_eta[i].item()
            ]
            writer.writerow(row)


    print("\n==============================")
    print("Frame-level dataset built successfully!")
    print("==============================")
    print(f"Saved: {cfg.out_path}")
    print(f"Frames: {len(X_pose)}")
    print(f"X_pose: {X_pose.shape}")
    print(f"X_step: {X_step.shape}")
    print(f"X_elem: {X_elem.shape}")
    print(f"X_env : {X_env.shape}")
    print(f"Y_urgency: {Y_urgency.shape}")
    print("==============================\n")


# ============================================================
# dataloader from frame-level to window-level
# ============================================================



if __name__ == "__main__":
    # Example:
    # - pose_json_path: generated by extract_pose_json.py
    # - labels_json_path: generated by label_video.py (unified)
    # - meta_json_path: optional
    cfg = Config(
        fps=30,
        window_sec=1, #float
        stride_sec=1, #float
        history_M=1, #int
        out_path="data/dataset/dataset_lift.npz"
    )

    build_dataset(
        pose_json_path="data/dataset/pose_lift.json",
        labels_json_path="data/dataset/labels_lift.json",
        design_json_path="data/dataset/design_data.json",
        cfg=cfg
    )