import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

import random

import sys
import socket

import time
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_lstm.LSTM_model_train import AssistLSTM

from step_id_stabilizer import StepIdStabilizer
from activation_logic import (
    build_current_task,
    command_grants_permission,
    permission_should_be_requested,
    add_pending_task,
)

import json
import re
import math

from copy import deepcopy

from extract_feat_rlt import setup_filtering, smooth_kpt,extract_features
from norm_feat_rlt import NormRealTime

# =========================
# CONFIG
# =========================
SELECTED_FEATS = ["pol_angles","joint_angles","ratios"]

BEST_MODEL_DIR = Path(r"C:\Users\loy49\Desktop\REPO\LSTM_HRC\model_lstm\runs\exp_2026-06-28_15-27-49")
MODEL_PATH = Path(BEST_MODEL_DIR) / "best_model.pth"
MODEL_CONFIG_PATH = Path(BEST_MODEL_DIR) / "config.json"

NORM_DIR = Path(r"C:\Users\loy49\Desktop\REPO\LSTM_HRC\data_proc_2d\dataset")
#latest norm npz
npz_files = list(NORM_DIR.glob("*.npz"))
if not npz_files:
    raise FileNotFoundError(f"No .npz files found in {NORM_DIR}")

NORM_PATH = str(max(npz_files, key=lambda p: p.stat().st_mtime))


yolo_model = YOLO("yolo26n-pose.pt")

#MODEL CONFIGURATION FROM JSON

with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    model_config = json.load(config_file)

WINDOW_SIZE = model_config["window_size"]
INPUT_DIM = model_config["input_dim"]
HINDDEN_DIM = model_config["hidden_dim"]
NUM_STEPS = model_config["num_steps"]
NUM_LAYERS = model_config.get("num_layers", 1)
DROPOUT = model_config.get("dropout", 0.0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AssistLSTM(
    input_dim = INPUT_DIM,
    hidden_dim = HINDDEN_DIM,
    num_steps = NUM_STEPS,
    dropout = DROPOUT,
    num_layers = NUM_LAYERS
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

STEP_SMOOTHING_WINDOW = 5
STEP_CONFIRMATION_COUNT = 3
STEP_MIN_CONFIDENCE = 0.6
STEP_MIN_MARGIN = 0.15



class ChangedMessageSender:
    def __init__(self, send_callback):
        self.send_callback = send_callback
        self.last_message = None

    def send_if_changed(self, message):
        if message == self.last_message:
            return False

        self.send_callback(message)
        self.last_message = deepcopy(message)
        return True

    def check_be_ready(self):
        pass

#protocol fields:
msg_gh = {
    "step_id": None,
    "progress": 0.0,

    # NEW: framework MVP fields
    "robot_capable": False,
    "opportunity_detected": False,
    "activation_state": "MONITORING",
    "permission_required": False,
    "permission_state": "NOT_REQUIRED",
    "suggested_action": "wait",
    "robot_command": "NO_ACTION",
    "reason": "monitoring"
}

UDP_HOST = "127.0.0.1"
UDP_PORT = 5006

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_message_to_gh(message):
    payload = json.dumps(message).encode("utf-8")
    udp_socket.sendto(payload, (UDP_HOST, UDP_PORT))

#the last checking gate
gh_sender = ChangedMessageSender(send_message_to_gh)

class ActivationState(Enum):
    MONITORING = "MONITORING"
    ASK_PERMISSION = "ASK_PERMISSION"
    READY_TO_ACTIVATE = "READY_TO_ACTIVATE"
    ACTIVE = "ACTIVE"
    HOLD = "HOLD"
    CANCELLED = "CANCELLED"


activation_state = ActivationState.MONITORING
CONFIRMATION_TIMEOUT = 5.0

# Store opportunities that were not confirmed within the time window.
pending_task_pool = deque(maxlen=5)

def timed_input(prompt, timeout=5.0):
    """
    Blocking user input with timeout.

    The main inference loop is paused while waiting.
    If no input is received within timeout, return None.

    On Windows, use non-blocking console polling instead of a background
    input() thread. This avoids stale input threads consuming a later response.
    """
    print(prompt, end="", flush=True)

    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + timeout
        chars = []

        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                char = msvcrt.getwch()

                if char in ("\r", "\n"):
                    print()
                    return "".join(chars).strip().lower()

                if char == "\b":
                    if chars:
                        chars.pop()
                        print("\b \b", end="", flush=True)
                    continue

                chars.append(char)
                print(char, end="", flush=True)

            time.sleep(0.05)

        print()
        return None

    # Fallback for non-Windows terminals. This script is currently run on
    # Windows, so the MVP path above avoids orphaned input threads.
    return input().strip().lower()

# GLOBAL_ACTIVATION_CONFIG = {
#     "progress_threshold": 0.8,
#     "requires_confirmation": True
# }

TRAJECTORY_CONFIG = {
    0: {
        "robot_capable": True,
        "suggested_action": "assist_lifting",
        "threshold": 0.01
    },

    1: {
        "robot_capable": True,
        "suggested_action": "wait",
        "threshold": 0.01
    },

    2: {
        "robot_capable": True,
        "suggested_action": "wait",
        "threshold": 0.01
    },

    3: {
        "robot_capable": True,
        "suggested_action": "wait",
        "threshold": 0.01
    },

    4: {
        "robot_capable": True,
        "suggested_action": "continue_screwing",
        "threshold": 0.01
    },
    5: {
        "robot_capable": True,
        "suggested_action": "hold_panel",
        "threshold": 0.01
    },
}

len_robot_task = sum(
    1 for task in TRAJECTORY_CONFIG.values() if task.get("robot_capable", False)
)


def build_msg(
    stable_step_id,
    progress,
    robot_capable=False,
    opportunity_detected=False,
    activation_state_value="MONITORING",
    permission_required=False,
    permission_state="NOT_REQUIRED",
    suggested_action="wait",
    robot_command="NO_ACTION",
    reason="monitoring"
):
    """
    Centralized message builder for Grasshopper / robot.
    """
    return {
        "step_id": int(stable_step_id) if stable_step_id is not None else None,
        "progress": float(progress),

        "robot_capable": bool(robot_capable),
        "opportunity_detected": bool(opportunity_detected),
        "activation_state": activation_state_value,
        "permission_required": bool(permission_required),
        "permission_state": permission_state,
        "suggested_action": suggested_action,
        "robot_command": robot_command,
        "reason": reason
    }



# =========================
# BUFFER
# =========================
buffer = deque(maxlen=WINDOW_SIZE)

# =========================
# STATE
# =========================
prev_kpts = None
prev_speed = None
step_stabilizer = None


# =========================
# MAIN LOOP
# =========================

# use another video for testing
test_vid = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"
cap = cv2.VideoCapture(test_vid)

# NEW: store last proposed task while waiting for human confirmation.
# This prevents switching to a new task during confirmation.
current_proposed_task = None

print("CLI commands: y=approve, n=reject, h=hold, s=stop, r=resume latest pending task")

kinematic_tracker = setup_filtering()
norm_real_time = NormRealTime(NORM_PATH, SELECTED_FEATS)

last_stable_step_id = None
last_progress_by_step = {}
requested_task_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    
    result = results[0]

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts = result.keypoints.xyn[0].cpu()
    else:
        kpts = torch.zeros((17, 2))

    ## SMOOTHING AND FEATURE EXTRACTION
    smoothed_kpts = smooth_kpt(kpts, kinematic_tracker)
    feat = extract_features(smoothed_kpts, selected_feats=SELECTED_FEATS)
    
    ## NORMALIZATION FEATURES
    feat = norm_real_time.normalize_features(feat)

    feat_vector = np.concatenate([
        np.asarray(feat[key].detach().cpu().numpy() if isinstance(feat[key], torch.Tensor) else feat[key], dtype=np.float32).reshape(-1)
        for key in SELECTED_FEATS
    ], axis=0)
    buffer.append(feat_vector)

    if len(buffer) == WINDOW_SIZE:
        x = np.stack(buffer).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

#MODEL PREDICTION
            step_logits, progress_pred = model(x)

            step_probs = torch.softmax(step_logits, dim=1)
            step_pred = torch.argmax(step_probs, dim=1)
            confidence = torch.max(step_probs, dim=1).values

            step_id = step_pred.item()
            progress = progress_pred.item()
            conf = confidence.item()



        probs_np = step_probs.squeeze(0).cpu().numpy()

        if step_stabilizer is None:
            step_stabilizer = StepIdStabilizer(
                num_steps=probs_np.shape[0],
                smoothing_window=STEP_SMOOTHING_WINDOW,
                confirmation_count=STEP_CONFIRMATION_COUNT,
                min_confidence=STEP_MIN_CONFIDENCE,
                min_margin=STEP_MIN_MARGIN,
            )

        stable_step_id = step_stabilizer.update(probs_np)

        print(f"Raw Step: {step_id} | Stable Step: {stable_step_id} | Progress: {progress}")

        # send message to gh/robot through local UDP
        if step_id is not None:

#need to add ros and rosbridge to send the message to the robot
#CHANGE THIS LOOP LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if last_stable_step_id is not None and step_id != last_stable_step_id:
                last_progress_by_step.clear()

            last_stable_step_id = step_id

            last_progress = last_progress_by_step.get(step_id,0)
            
            #GET THRESHOLD:
            threshold = TRAJECTORY_CONFIG[step_id]["threshold"]

            cross_threshold = (last_progress is not None and last_progress<=threshold and progress>threshold)
            
            msg_gh = None  

            current_proposed_task = build_current_task(
                step_id,
                progress,
                TRAJECTORY_CONFIG,
            )

            last_progress_by_step[step_id] = progress
            task_id = current_proposed_task.get("task_id", f"step_{step_id}")

            if len(requested_task_ids)==len_robot_task:
                requested_task_ids.clear()
            
            if cross_threshold and task_id not in requested_task_ids:
                # In this MVP version, inference pauses during the confirmation
                # window so the terminal prompt stays readable.
                requested_task_ids.add(task_id)
                activation_state = ActivationState.ASK_PERMISSION

                print("\n" + "=" * 60)
                print("[CONFIRMATION WINDOW]")
                print(f"Robot proposes: {current_proposed_task['suggested_action']}")
                print(f"Step ID for GH trajectory: {current_proposed_task['step_id']}")
                print(f"Normalized progress: {progress:.2f}")
                print(f"Please respond within {CONFIRMATION_TIMEOUT:.1f}s")
                print("Input: y=approve, n=reject / add to pending")
                print("=" * 60)

                command = timed_input("Your response [y/n]: ", timeout=CONFIRMATION_TIMEOUT)

                if command_grants_permission(command):
                    activation_state = ActivationState.READY_TO_ACTIVATE

                    # Send START_TASK to GH / robot only after permission is granted.
                    msg_gh = build_msg(
                        stable_step_id=current_proposed_task["step_id"],
                        progress=current_proposed_task["step_progress"],
                        robot_capable=True,
                        opportunity_detected=True,
                        activation_state_value=activation_state.value,
                        permission_required=True,
                        permission_state="GRANTED",
                        suggested_action=current_proposed_task["suggested_action"],
                        robot_command="START_TASK",
                        reason="permission_granted"
                    )

                    # In a real robot setup, ACTIVE should be set by robot feedback.
                    activation_state = ActivationState.ACTIVE

                else:
                    add_pending_task(pending_task_pool, current_proposed_task, command)

                    print(
                        f"\n[PENDING] Permission not granted. "
                        f"Task added to pending pool. Pending count: {len(pending_task_pool)}\n"
                    )

                    activation_state = ActivationState.MONITORING

                current_proposed_task = None

            else:
                reason = (
                    "progress_below_global_threshold"
                    if current_proposed_task["robot_capable"]
                    else "step_not_robot_capable"
                )

                print(f"Skipped UDP to GH ({reason})")

            if msg_gh is None:
                print("Skipped UDP to GH (permission not granted)")
            else:
                send_message_to_gh(msg_gh)
                print(f"Sent UDP to GH {UDP_HOST}:{UDP_PORT} | {json.dumps(msg_gh)}")

        else:
            print("Skipped UDP to GH (step id not stable yet)")

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
udp_socket.close()
cv2.destroyAllWindows()
