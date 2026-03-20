# extract_pose_json.py
import os
import json
import cv2
# import mediapipe as mp
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Load the yolo model
yolo_model = YOLO("yolo26n-pose.pt")  #fast 

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = r"data\video\lift.mp4"
OUTPUT_JSON = r"data/dataset/lift.json"

# If True -> each landmark has x,y,z,visibility
USE_VISIBILITY = True

SHOW_EVERY_N_FRAMES = 10
# ==========================
# Helper
# ==========================
def sec_from_frame(frame_idx, fps):
    return frame_idx / fps

# extract the two cloosest objects/tools to the hands from each frame
def find_closest_objects(rgb_frame, num_objects=2):
    return []

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

        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = yolo_model(frame)

        # if result.pose_landmarks and finger_result.multi_hand_landmarks is None:
        if not results:
            landmarks = []
        else:
            landmarks = []
            # add pose landmarks
            for result in results:
                img_plot = result.plot()  # visualize the results on the image (optional)
                img_plot = Image.fromarray(img_plot[..., ::-1])
                img_plot.show()

                
                xy = result.keypoints.xy
                landmarks.append(xy)


        frames_out.append({
            "frame": int(frame_idx),
            "t": round(float(t_sec), 6),
            "landmarks": landmarks # tensor format
        })



    cap.release()
    # pose.close()
    cv2.destroyAllWindows() 

    
    out = {
        "schema": "pose_v1",
        "video_path": VIDEO_PATH,
        "created_at": datetime.now().isoformat(),
        "fps": float(fps),
        "num_frames": int(len(frames_out)),
        "pose_model": "MediaPipe Pose (33 landmarks)",
        "use_visibility": USE_VISIBILITY,
        "frames": frames_out,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved pose JSON to: {OUTPUT_JSON}")
    print(f"Frames saved: {len(frames_out)}")


if __name__ == "__main__":
    run_pose_extraction()