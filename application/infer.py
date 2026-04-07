#HAVEN'T DEBUG YET!!DONT USE IT DIRECTLY!!
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# =========================
# CONFIG
# =========================
MODEL_PATH = "assist_model.pth"
NORM_PATH = "norm_stats.npz"

WINDOW_SIZE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODEL
# =========================
from model import AssistLSTM

model = AssistLSTM(input_dim=35)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# LOAD NORMALIZATION
# =========================
norm = np.load(NORM_PATH)
mean = norm["mean"]
std = norm["std"]

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

# =========================
# SAME FUNCTIONS
# =========================
def normalize_keypoints(kpts_tensor):
    if torch.all(kpts_tensor == 0):
        return kpts_tensor
    center = kpts_tensor.mean(dim=0)
    kpts = kpts_tensor - center
    scale = torch.norm(kpts, dim=1).mean() + 1e-6
    return kpts / scale


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = torch.dot(ba, bc) / (torch.norm(ba)*torch.norm(bc)+1e-6)
    angle = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    return torch.rad2deg(angle)


def extract_posture_features(kpts):
    try:
        angle1 = calculate_angle(kpts[7], kpts[5], kpts[11])
        angle2 = calculate_angle(kpts[8], kpts[6], kpts[12])
        angle3 = calculate_angle(kpts[5], kpts[7], kpts[9])
        angle4 = calculate_angle(kpts[6], kpts[8], kpts[10])
        angle5 = calculate_angle(kpts[5], kpts[11], kpts[13])
        angle6 = calculate_angle(kpts[6], kpts[12], kpts[14])
        angle7 = calculate_angle(kpts[11], kpts[13], kpts[15])
        angle8 = calculate_angle(kpts[12], kpts[14], kpts[16])

        dist_wrists = torch.norm(kpts[9] - kpts[10])
        dist_shoulders = torch.norm(kpts[5] - kpts[6]) + 1e-6
        ratio = dist_wrists / dist_shoulders

        return torch.tensor([angle1,angle2,angle3,angle4,
                             angle5,angle6,angle7,angle8,ratio])
    except:
        return torch.zeros(9)


# =========================
# FEATURE BUILDER
# =========================
def build_feature(kpts):
    global prev_kpts, prev_speed

    # ---------- posture ----------
    degree_feat = extract_posture_features(kpts)

    # ---------- speed ----------
    if prev_kpts is None:
        speed = torch.zeros(17)
    else:
        velocity = kpts - prev_kpts
        speed = torch.norm(velocity, dim=1)

    # ---------- acceleration ----------
    if prev_speed is None:
        accel = torch.zeros(17)
    else:
        accel = torch.abs(speed - prev_speed)

    prev_kpts = kpts.clone()
    prev_speed = speed.clone()

    # ---------- angle → sin/cos ----------
    angle_rad = torch.deg2rad(degree_feat)
    sin_feat = torch.sin(angle_rad)
    cos_feat = torch.cos(angle_rad)

    angle_feat = torch.cat([sin_feat, cos_feat], dim=0)  # [18]

    pose_feat = torch.cat([angle_feat, accel], dim=0)    # [35]

    # ---------- normalize ----------
    pose_feat = pose_feat.numpy()
    pose_feat = (pose_feat - mean) / std

    return pose_feat


# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)

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

    feat = build_feature(kpts)
    buffer.append(feat)

    if len(buffer) == WINDOW_SIZE:
        x = np.stack(buffer)
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            step_logits, _ = model(x)
            probs = torch.softmax(step_logits, dim=1)
            pred = torch.argmax(probs, dim=1)

        print(f"Step: {pred.item()} | Prob: {probs.cpu().numpy()}")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()