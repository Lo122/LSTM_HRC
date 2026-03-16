# extract_pose_json.py
import os
import json
import cv2
import mediapipe as mp
from datetime import datetime
from tqdm import tqdm
import numpy as np

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = r"data\video\spacer.mp4"
OUTPUT_JSON = r"data/dataset/spacer.json"

# If True -> each landmark has x,y,z,visibility
USE_VISIBILITY = True

SHOW_EVERY_N_FRAMES = 10
# ==========================
# Helper
# ==========================
def sec_from_frame(frame_idx, fps):
    return frame_idx / fps


# ==========================
# Main
# ==========================
def run_pose_extraction():
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_pose = mp.solutions.pose
    mp_finger = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils 
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fingers = mp_finger.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    #add finger landmarks to pose visualization


    frames_out = []

    print("\n==============================")
    print(" MediaPipe Pose Extraction Tool")
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pose.process(rgb)
        finger_result = fingers.process(rgb)



        if result.pose_landmarks and finger_result.multi_hand_landmarks is None:
            landmarks = []
        else:
            landmarks = []
            # add pose landmarks
            for lm in result.pose_landmarks.landmark:
                if USE_VISIBILITY:
                    landmarks.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility),
                    })
                else:
                    landmarks.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    })
            # add finger landmarks
            for hand_landmarks in finger_result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    if USE_VISIBILITY:
                        landmarks.append({
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility),
                        })
                    else:
                        landmarks.append({
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                        })

        frames_out.append({
            "frame": int(frame_idx),
            "t": round(float(t_sec), 6),
            "landmarks": landmarks
        })

        # ===============================
        if result.pose_landmarks and frame_idx % SHOW_EVERY_N_FRAMES == 0:
            h, w, _ = frame.shape

            white_bg = 255 * np.ones((h, w, 3), dtype=np.uint8)
            
            mp_drawing.draw_landmarks(
                white_bg,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        
            cv2.putText(
                white_bg,
                f"Frame: {frame_idx}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Pose Debug Window", white_bg)
            #export the image
            frame_output_path = f"C:/Users/loy49/Desktop/white/frame_{frame_idx:04d}.jpg"
            os.makedirs(os.path.dirname(frame_output_path), exist_ok=True)
            cv2.imwrite(frame_output_path, white_bg)   

          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    pose.close()
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