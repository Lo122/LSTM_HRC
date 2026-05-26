#HAVEN'T DEBUG YET!!DONT USE IT DIRECTLY!!
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from data.extract_pose import normalize_keypoints, extract_posture_features
from LSTM.LSTM_model_train import AssistLSTM

# =========================
# CONFIG
# =========================
MODEL_PATH = r"application\lstm_hrc.pth"
NORM_PATH = r"data_proc_2d\norm_stats.npz"

WINDOW_SIZE = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================

#Hybrid (Deg+Sp):   (61104, 33)
model = AssistLSTM(input_dim=33)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# LOAD NORMALIZATION
# =========================
norm = np.load(NORM_PATH)
# mean = norm["mean"]
# std = norm["std"]

mean_degree = norm["mean_degree"]
std_degree = norm["std_degree"]
mean_speed=norm["mean_speed"]
std_speed=norm["std_speed"]

# =========================
# YOLO
# =========================
yolo_model = YOLO("yolo26n-pose.pt")

# =========================
# BUFFER
# =========================
buffer = deque(maxlen=WINDOW_SIZE)

# =========================
# STATE
# =========================
prev_kpts = None
prev_speed = None


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = torch.dot(ba, bc) / (torch.norm(ba)*torch.norm(bc)+1e-6)
    angle = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    return torch.rad2deg(angle)


# =========================
# FEATURE BUILDER
# =========================
def build_feature(kpts):
    global prev_kpts, prev_speed, mean_degree, std_degree, mean_speed, std_speed

    # ---------- features ----------
    degree_feat, _ = extract_posture_features(kpts)

    # ---------- speed ----------
    if prev_kpts is None:
        speed = torch.zeros(17)
    else:
        velocity = kpts - prev_kpts
        speed = torch.norm(velocity, dim=1)

    # # ---------- acceleration ----------
    # if prev_speed is None:
    #     accel = torch.zeros(17)
    # else:
    #     accel = torch.abs(speed - prev_speed)

    prev_kpts = kpts.clone()
    prev_speed = speed.clone()

    # ---------- angle → sin/cos ----------
    angle_rad = torch.deg2rad(degree_feat)
    sin_feat = torch.sin(angle_rad)
    cos_feat = torch.cos(angle_rad)

    angle_feat = torch.cat([sin_feat, cos_feat], dim=0)  # [18]

    #norm angle and speed
    norm_degree = (angle_feat - mean_degree) / std_degree
    norm_speed = (speed - mean_speed) / std_speed

    #angle and speed concatenate
    hybrid_feat = np.hstack([norm_degree, norm_speed]).astype(np.float32)  # [35]
    feat = torch.tensor(hybrid_feat, dtype=torch.float32)

    # # ---------- normalize ----------
    # pose_feat = pose_feat.numpy()
    # pose_feat = (pose_feat - mean) / std

    return feat


# =========================
# MAIN LOOP
# =========================

# use another video for testing
test_vid = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\cropped\cam-01\video__cam-01_uid-01_take-03.mp4"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    result = results[0]

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        raw_kpts = result.keypoints.xyn[0].cpu()
        kpts = normalize_keypoints(raw_kpts)
    else:
        kpts = torch.zeros((17, 2))

    ##CORE!
    feat = build_feature(kpts)
    buffer.append(feat)

    if len(buffer) == WINDOW_SIZE:
        x = np.stack(buffer)
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            step_logits = model(x)
            probs = torch.softmax(step_logits, dim=1)
            pred = torch.argmax(probs, dim=1)

        print(f"Step: {pred.item()} | Prob: {probs.cpu().numpy()}")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()