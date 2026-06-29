import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

import random

import os
import sys
import socket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from data.extract_pose import normalize_keypoints, extract_posture_features
from LSTM.LSTM_model_train import AssistLSTM
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

# =========================
# NEW: activation / confirmation imports
# =========================
import time
from enum import Enum


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


# region UDP connection
# set up msg protocol with gh/robot through local UDP
#
# Original protocol:
# msg_gh = {
#     "step_id": "",
#     "step_progress": 0.0
# }
#
# Extended MVP protocol:
# - step_id / step_progress still come from perception
# - activation_state / permission_state / suggested_action / robot_command
#   represent the proactive assistance framework state
#
# Important:
# - step_progress is normalized by the model / progress estimator.
# - step_progress controls WHEN communication / activation is triggered.
# - step_id is mainly used by GH / robot to select the correct trajectory.

#protocol fields:
msg_gh = {
    "step_id": None,
    "step_progress": 0.0,

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

# endregion


# =========================
# NEW: Activation State Machine
# =========================
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

# Blocking confirmation is intentional for the MVP.
# During the confirmation window, the model/video loop is paused so that:
# - terminal output does not keep refreshing
# - the worker has a clear 5s window to respond
# - no new prediction can overwrite the current proposed task
#
# This is different from the long-term architecture, where perception may continue
# in a separate thread. For the current CLI-based MVP, pausing inference is simpler
# and easier to demonstrate.
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


# =========================
# NEW: Global Activation + Step Action Mapping
# =========================
# The progress threshold is global because step_progress is normalized.
# Therefore, progress does NOT need to be interpreted per step.
#
# step_progress controls:
#   WHEN the system should trigger communication / permission acquisition.
#
# step_id controls:
#   WHICH robot-capable action / GH trajectory should be selected.
GLOBAL_ACTIVATION_CONFIG = {
    "progress_threshold": 0.8,
    "requires_confirmation": True
}

# Step/action mapping.
# Replace the example step IDs and actions with your real climate ceiling workflow.
#
# robot_capable means:
#   the robot MAY participate in this step.
#   it does NOT mean the robot MUST execute it.
#
# suggested_action is mainly a semantic label for debugging / thesis demo.
# GH / robot can still use step_id to select the correct trajectory.
TRAJECTORY_CONFIG = {
    0: {
        "robot_capable": True,
        "suggested_action": "assist_lifting"
    },

    1: {
        "robot_capable": False,
        "suggested_action": "wait"
    },

    # Add more robot-capable steps here if needed.
    # 4: {
    #     "robot_capable": True,
    #     "suggested_action": "continue_screwing"
    # },
    # 5: {
    #     "robot_capable": True,
    #     "suggested_action": "hold_panel"
    # },
}


def get_step_progress(stable_step_id, probs_np=None):
    """
    Placeholder progress estimation.

    Current code used a constant 0.5.

    In your final system, this should be replaced by your normalized progress
    output from the model / progress estimator.

    Expected range:
        0.0 = current step just started
        1.0 = current step completed

    Because progress is normalized, the same global threshold can be used
    across different step IDs.
    """
    prog = random.uniform(0.7, 1)
    return prog


def build_msg(
    stable_step_id,
    step_progress,
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
        "step_progress": float(step_progress),

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
# CONFIG
# =========================
MODEL_PATH = r"lstm_hrc.pth"
NORM_PATH = r"norm_stats.npz"

WINDOW_SIZE = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STEP_SMOOTHING_WINDOW = 5
STEP_CONFIRMATION_COUNT = 3
STEP_MIN_CONFIDENCE = 0.6
STEP_MIN_MARGIN = 0.15

# =========================
# LOAD MODEL
# =========================

# Hybrid (Deg+Sp):   (61104, 33)
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
mean_speed = norm["mean_speed"]
std_speed = norm["std_speed"]

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
step_stabilizer = None



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

    # norm angle and speed
    norm_degree = (angle_feat - mean_degree) / std_degree
    norm_speed = (speed - mean_speed) / std_speed

    # angle and speed concatenate
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
cap = cv2.VideoCapture(test_vid)

# NEW: store last proposed task while waiting for human confirmation.
# This prevents switching to a new task during confirmation.
current_proposed_task = None

print("CLI commands: y=approve, n=reject, h=hold, s=stop, r=resume latest pending task")

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

    ## CORE!
    feat = build_feature(kpts)
    buffer.append(feat)

    if len(buffer) == WINDOW_SIZE:
        x = np.stack(buffer)
        x = torch.tensor(x).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            step_logits = model(x)
            probs = torch.softmax(step_logits, dim=1)
            raw_pred = torch.argmax(probs, dim=1)

        probs_np = probs.squeeze(0).cpu().numpy()
        if step_stabilizer is None:
            step_stabilizer = StepIdStabilizer(
                num_steps=probs_np.shape[0],
                smoothing_window=STEP_SMOOTHING_WINDOW,
                confirmation_count=STEP_CONFIRMATION_COUNT,
                min_confidence=STEP_MIN_CONFIDENCE,
                min_margin=STEP_MIN_MARGIN,
            )

        stable_step_id = step_stabilizer.update(probs_np)

        print(f"Raw Step: {raw_pred.item()} | Stable Step: {stable_step_id} | Prob: {probs_np}")

        # send message to gh/robot through local UDP
        if stable_step_id is not None:
            msg_gh = None
            step_progress = get_step_progress(stable_step_id, probs_np)
            current_proposed_task = build_current_task(
                stable_step_id,
                step_progress,
                TRAJECTORY_CONFIG,
            )

            if permission_should_be_requested(
                current_proposed_task,
                GLOBAL_ACTIVATION_CONFIG["progress_threshold"],
            ):
                # In this MVP version, inference pauses during the confirmation
                # window so the terminal prompt stays readable.
                activation_state = ActivationState.ASK_PERMISSION

                print("\n" + "=" * 60)
                print("[CONFIRMATION WINDOW]")
                print(f"Robot proposes: {current_proposed_task['suggested_action']}")
                print(f"Step ID for GH trajectory: {current_proposed_task['step_id']}")
                print(f"Normalized progress: {step_progress:.2f}")
                print(f"Please respond within {CONFIRMATION_TIMEOUT:.1f}s")
                print("Input: y=approve, n=reject / add to pending")
                print("=" * 60)

                command = timed_input("Your response [y/n]: ", timeout=CONFIRMATION_TIMEOUT)

                if command_grants_permission(command):
                    activation_state = ActivationState.READY_TO_ACTIVATE

                    # Send START_TASK to GH / robot only after permission is granted.
                    msg_gh = build_msg(
                        stable_step_id=current_proposed_task["step_id"],
                        step_progress=current_proposed_task["step_progress"],
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
