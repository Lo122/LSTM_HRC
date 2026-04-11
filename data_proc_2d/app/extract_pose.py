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

# OUTPUT_JSON = r"data/dataset/cam1_spacer_Y1.json"
OUTPUT_DIR = r"data/dataset"

# If True -> each landmark has x,y,z,visibility
USE_VISIBILITY = True

SHOW_EVERY_N_FRAMES = 20
# ==========================
# Helper
# ==========================
def sec_from_frame(frame_idx, fps):
    return frame_idx / fps
# Nose
# Left Eye
# Right Eye
# Left Ear
# Right Ear
# Left Shoulder
# Right Shoulder
# Left Elbow
# Right Elbow
# Left Wrist
# Right Wrist
# Left Hip
# Right Hip
# Left Knee
# Right Knee
# Left Ankle
# Right Ankle

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

def calculate_angle(a, b, c):
    # Calculate the angle at point b formed by points a and c
    ba = a - b
    bc = c - b

    cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc) + 1e-6)
    angle = torch.acos(torch.clamp(cosine_angle, -1.0, 1.0))  # Clamp to avoid numerical issues
    return torch.rad2deg(angle)

def extract_posture_features(kpts_tensor):
    if torch.all(kpts_tensor == 0):
        return torch.zeros(8), torch.zeros(1)  # Return zero vectors if no keypoints detected

    # angle list 1: [degree between torso and left upper arm, degree between torso and right upper arm]
    # [∠7-5-11，∠8-6-12] 
    try:
        angle1 = calculate_angle(kpts_tensor[7], kpts_tensor[5], kpts_tensor[11])
        angle2 = calculate_angle(kpts_tensor[8], kpts_tensor[6], kpts_tensor[12])
    
    # angle list 2: [degree between left upper arm and left forearm, degree between right upper arm and right forearm]
    # [∠5-7-9，∠6-8-10]
        angle3 = calculate_angle(kpts_tensor[5], kpts_tensor[7], kpts_tensor[9])
        angle4 = calculate_angle(kpts_tensor[6], kpts_tensor[8], kpts_tensor[10])

    # angle list 3: [degree between bottom torsor and left leg, degree between bottom torsor and right leg]
    # [∠5-11-13，∠6-12-14]
        angle5 = calculate_angle(kpts_tensor[5], kpts_tensor[11], kpts_tensor[13])
        angle6 = calculate_angle(kpts_tensor[6], kpts_tensor[12], kpts_tensor[14])
    
    # angle list 4: [degree of left knee, degree of right knee]
    # [∠11-13-15，∠12-14-16]
        angle7 = calculate_angle(kpts_tensor[11], kpts_tensor[13], kpts_tensor[15])
        angle8 = calculate_angle(kpts_tensor[12], kpts_tensor[14], kpts_tensor[16])

    # divide(distance between two wrists, distance between two shoulders)
    # l(9,10), l(5,6)
        dist_wrists = torch.norm(kpts_tensor[9] - kpts_tensor[10])
        dist_shoulders = torch.norm(kpts_tensor[5] - kpts_tensor[6]) + 1e-6  # Add epsilon to avoid division by zero
        ratio9 = dist_wrists / dist_shoulders

        return torch.tensor([angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8]), torch.tensor([ratio9])
    
    except Exception as e:
        return torch.zeros(8), torch.zeros(1)  # Return zero vectors if any error occurs during angle calculation (e.g., due to missing keypoints)
# ==========================
# Main
# ==========================
def run_pose_extraction(VIDEO_PATH):

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
            # xy 
            raw_kpts = result.keypoints.xyn[0].cpu()  # get the keypoints for the first detected person
            kpts = normalize_keypoints(raw_kpts)

        else:
            kpts = torch.zeros((17, 2))  # create a dummy tensor if no keypoints detected


# organize features
        kpts_degree, kpts_ratio = extract_posture_features(kpts)

        frames_out.append({
                    "norm_kpts": kpts,
                    "features_degree": kpts_degree,
                    "features_ratio": kpts_ratio,
                    "t": frame_idx / fps
                })
        if frame_idx % SHOW_EVERY_N_FRAMES == 0:
            cv_plot = result.plot()
            cv2.imshow("Debug", cv_plot)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # --- Final Data Organization ---
    all_landmarks = torch.stack([f["norm_kpts"] for f in frames_out]) # (T, 17, 2)
    all_degrees = torch.stack([f["features_degree"] for f in frames_out])   # (T, 8)
    all_ratios = torch.stack([f["features_ratio"] for f in frames_out])   # (T, 1)

    # 1. Calculate Velocity (Speed) - (T-1, 17)
    # diff calculates: x[i+1] - x[i]
    velocity = torch.diff(all_landmarks, dim=0)
    speed = torch.norm(velocity, dim=2) 
    
    # 2. Calculate Acceleration - (T-2, 17)
    # accel calculates: |speed[i+1] - speed[i]|
    accel = torch.abs(torch.diff(speed, dim=0))
    
    # 3. Padding to maintain original frame count T
    # Speed: Pad 1 row of zeros at the start
    speed_padded = torch.cat([torch.zeros((1, 17)), speed], dim=0)
    
    # Acceleration: Pad 2 rows of zeros at the start 
    # (or 1 at start, 1 at end) to keep T frames
    accel_padded = torch.cat([torch.zeros((2, 17)), accel], dim=0)


    #save
    save_data = {
        "metadata": {"video": VIDEO_PATH, "fps": fps},
        "landmarks": all_landmarks,
        "feat_degree": all_degrees,
        "feat_ratio": all_ratios,
        "feat_speed": speed_padded,
        "feat_acc": accel_padded,
        "t_steps": torch.tensor([f["t"] for f in frames_out])
    }

    OUTPUT_PT = os.path.join(OUTPUT_DIR, os.path.basename(VIDEO_PATH).replace(".mp4", ".pt"))

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(save_data, OUTPUT_PT)
    print(f"\nSaved {len(frames_out)} frames to {OUTPUT_PT}")


if __name__ == "__main__":
    pass

    # path = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos\processed\cam3\cam3_Y1.mp4"
    # run_pose_extraction(path)

    # VIDEO_DIR = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos\processed\cam3"

    # VIDEO_PATHS = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    # for video_path in VIDEO_PATHS:
    #     run_pose_extraction(video_path)
