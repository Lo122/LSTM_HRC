# extract_pose_json.py
from math import dist
import os
import json
import cv2
# import mediapipe as mp
from PIL import Image
from sympy import fps
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
# import dataset_tools as dtools

#construction sites tools dataset: https://datasetninja.com/small-size-construction-tools
#use this dataset to fine tune the yolo model for better detection of tools/objects in the construction site environment

# Load the yolo model
yolo_model = YOLO("yolo26n-pose.pt")  #fast 

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos\processed\1_spacer\cam1\cam1_spacer_M2.mp4"
# OUTPUT_JSON = r"data/dataset/cam1_spacer_Y1.json"
OUTPUT_PT = r"data/dataset/cam1_spacer_M2.pt"

# If True -> each landmark has x,y,z,visibility
USE_VISIBILITY = True

SHOW_EVERY_N_FRAMES = 20
# ==========================
# Helper
# ==========================
def sec_from_frame(frame_idx, fps):
    return frame_idx / fps

#region - not used for now
# extract the two cloosest objects/tools to the hands from each frame
# def find_closest_obbs(obbs_data, landmarks_tensor):
#     if obbs_data is None or len(obbs_data.xywhr) == 0 or landmarks_tensor.shape[1] < 11:
#             return []
    
#     #get hands
#     keypoints = landmarks_tensor[0].cpu().numpy() 
#     left_hand = keypoints[9]   # [x, y]
#     right_hand = keypoints[10] # [x, y]
#     hands = {"left": left_hand, "right": right_hand}

#     #get obbs info
#     obb_coords = obbs_data.xywhr.cpu().numpy()
#     cls_ids = obbs_data.cls.cpu().numpy().astype(int)
#     confs = obbs_data.conf.cpu().numpy()
    
#     closest_per_hand = []

#     #get closest obbs to each hand (max 1 per hand)
#     for side, hand_pos in hands.items():
#         if np.all(hand_pos == 0):
#             continue  # skip if hand not detected
#         min_dist = float('inf')
#         best_obb = None

#         for i, obb in enumerate(obb_coords):
#             obj_center = obb[:2] #get centroid
#             dist = np.linalg.norm(hand_pos - obj_center)
#             if dist < min_dist:
#                 min_dist = dist
#                 best_obj = {
#                     "hand_side": side,
#                     "object_name": str(obbs_data.names.get(cls_ids[i], "nan")),
#                     "obb_xywhr": obb.tolist(),
#                     "center": obj_center.tolist(),
#                     "confidence": float(confs[i]),
#                     "distance": float(dist)
#                 }
#         if best_obj:
#             closest_per_hand.append(best_obj)

#     return closest_per_hand
#endregion

def normalize_keypoints(kpts_tensor):
    if torch.all(kpts_tensor == 0):
        return kpts_tensor  # Return as is if all keypoints are zero (no detection)
    
    center = kpts_tensor.mean(dim=0)  # Calculate the center of the keypoints
    kpts_preprocessed = kpts_tensor - center  # Center the keypoints around (0,0)

    dist = torch.norm(kpts_preprocessed, dim=1) # Get the maximum distance from the center
    scale = dist.mean()+1e-6  # Add a small epsilon to avoid division by zero
    kpts_preprocessed = kpts_preprocessed / scale  # Scale the keypoints to fit within a unit circle

    return kpts_preprocessed
    
# ==========================
# Main
# ==========================
def run_pose_extraction():
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #add finger landmarks to pose visualization??? not sure now

    frames_out = []

    print("\n==============================")
    print(" YOLO Pose Extraction Tool")
    print("==============================")
    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print("==============================\n")

    frame_idx = 0

    for _ in tqdm(range(total_frames), desc="Extracting pose"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_sec = sec_from_frame(frame_idx, fps)

        # get all the results
        results = yolo_model(frame)
        result = results[0]  # get the first result (assuming batch size of 1)
        
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            raw_kpts = result.keypoints.xyn[0].cpu()  # get the keypoints for the first detected person
            kpts = normalize_keypoints(raw_kpts)
        else:
            kpts = torch.zeros((17, 2))  # create a dummy tensor if no keypoints detected

        frames_out.append({
                    "norm_kpts": kpts,
                    "t": frame_idx / fps
                })
        if frame_idx % SHOW_EVERY_N_FRAMES == 0:
            cv_plot = result.plot()
            cv2.imshow("Debug", cv_plot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    #save
    save_data = {
        "metadata": {"video": VIDEO_PATH, "fps": fps},
        "landmarks": torch.stack([f["norm_kpts"] for f in frames_out]),
        "t_steps": torch.tensor([f["t"] for f in frames_out])
    }

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(save_data, OUTPUT_PT)
    print(f"\nSaved {len(frames_out)} frames to {OUTPUT_PT}")


if __name__ == "__main__":
    run_pose_extraction()