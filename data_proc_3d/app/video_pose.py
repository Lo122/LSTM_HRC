import os
import time
import subprocess
import cv2
import numpy as np
import requests
from ultralytics import YOLO

# --- Configuration ---
# Place your test video in the VideoPose3D folder and update this name
input_video = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"
output_video = 'output_3d_animation.mp4'
dataset_name = 'yolo_extraction'
subject_name = os.path.basename(input_video)
action_name = 'custom'

VIDEOPOSE3D_DIR = r"C:\Users\Owner\OneDrive - Universität Stuttgart\2025_26_Thesis\codes\VideoPose3D"
CONDA_ENV_DIR = r"C:\Users\Owner\anaconda3\envs\videopose3d"
PYTHON_EXE = os.path.join(CONDA_ENV_DIR, "python.exe")

FRAME_SIZE = (640, 360)  # (width, height)
START_SEC = 30  # crop starting at this timestamp
END_SEC = 120  # crop ending at this timestamp


def resize_video(src_path, target_size, start_sec=None, end_sec=None):
    """Resize src_path to target_size, optionally cropping to [start_sec, end_sec).

    Returns the path of the processed copy, or src_path unchanged if neither
    resizing nor cropping is needed.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(fps * start_sec) if start_sec else 0
    end_frame = int(fps * end_sec) if end_sec is not None else total_frames
    needs_crop = start_frame > 0 or end_frame < total_frames

    if orig_size == target_size and not needs_crop:
        cap.release()
        return src_path

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    base, ext = os.path.splitext(src_path)
    resized_path = f"{base}_resized{ext}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(resized_path, fourcc, fps, target_size)

    frame_count = start_frame
    while frame_count < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    return resized_path


def download_file(url, dest_path, retries=3, timeout=30):
    """Download url to dest_path with retries, streaming and a size check."""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                expected_size = int(resp.headers.get("Content-Length", 0))

                tmp_path = f"{dest_path}.part"
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)

            if expected_size and downloaded != expected_size:
                raise IOError(f"Incomplete download: got {downloaded} of {expected_size} bytes")

            os.replace(tmp_path, dest_path)
            return
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Download attempt {attempt}/{retries} failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def main():
    if not os.path.exists(input_video):
        print(f"Error: Could not find '{input_video}'. Please put a video in the folder.")
        return

    # 0. Resize video to a fixed frame size and crop to [START_SEC, END_SEC)
    print(f"\n--- Step 0: Resizing Video to {FRAME_SIZE[0]}x{FRAME_SIZE[1]}, "
          f"cropping to {START_SEC}s-{END_SEC}s ---")
    resized_video = resize_video(input_video, FRAME_SIZE, start_sec=START_SEC, end_sec=END_SEC)

    # 1. Run YOLO to get 2D Keypoints
    print("\n--- Step 1: Running YOLO 2D Pose Estimation ---")
    model = YOLO('yolov8n-pose.pt')
    results = model(resized_video, stream=True)

    all_keypoints = []
    for r in results:
        # Check if a person was detected in the frame
        if r.keypoints is not None and r.keypoints.xy.numel() > 0:
            # Grab the 17x2 coordinates for the first person detected
            kpts = r.keypoints.xy[0].cpu().numpy()
        else:
            # Fallback for empty frames to keep the timeline synced
            kpts = np.zeros((17, 2))
        all_keypoints.append(kpts)

    keypoints_array = np.array(all_keypoints)

    # 2. Save to VideoPose3D Format
    print("\n--- Step 2: Saving Data for VideoPose3D ---")
    data_dir = os.path.join(VIDEOPOSE3D_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    npz_path = os.path.join(data_dir, f'data_2d_custom_{dataset_name}.npz')
    
    # This specific nested dictionary structure is strictly required by the library
    output_dict = {
        subject_name: {
            action_name: [keypoints_array]
        }
    }
    # CustomDataset also requires per-video resolution metadata, keyed by subject_name
    metadata = {
        'layout_name': 'coco',
        'num_joints': 17,
        'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]],
        'video_metadata': {
            subject_name: {'w': FRAME_SIZE[0], 'h': FRAME_SIZE[1]}
        }
    }
    np.savez_compressed(npz_path, positions_2d=output_dict, metadata=metadata)

    # 3. Download Pretrained Weights (if missing)
    print("\n--- Step 3: Checking Pretrained Weights ---")
    checkpoint_dir = os.path.join(VIDEOPOSE3D_DIR, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_url = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
    model_path = os.path.join(checkpoint_dir, "pretrained_h36m_detectron_coco.bin")

    if not os.path.exists(model_path):
        print("Downloading the 243-frame VideoPose3D model...")
        download_file(model_url, model_path)

    # 4. Execute VideoPose3D Rendering
    print("\n--- Step 4: Running VideoPose3D 3D Lifting ---")

    run_py_path = os.path.join(VIDEOPOSE3D_DIR, "run.py")
    if not os.path.exists(run_py_path):
        print(
            f"Error: 'run.py' not found in VIDEOPOSE3D_DIR ('{VIDEOPOSE3D_DIR}').\n"
            "Update VIDEOPOSE3D_DIR to point at your VideoPose3D repo checkout "
            "(https://github.com/facebookresearch/VideoPose3D)."
        )
        return

    command = [
        PYTHON_EXE, run_py_path,
        "-d", "custom",                           # Use custom dataset loader
        "-k", dataset_name,                       # Name of the .npz file (without prefix/suffix)
        "-arc", "3,3,3,3,3",                      # Architecture matching the pretrained model
        "-c", "checkpoint",                       # Folder containing weights
        "--evaluate", "pretrained_h36m_detectron_coco.bin", 
        "--render",                               # Enable visualization
        "--viz-subject", subject_name,            # Must match the dictionary key
        "--viz-action", action_name,              # Must match the dictionary key
        "--viz-camera", "0",                      # Default camera angle
        "--viz-video", os.path.abspath(resized_video), # Absolute path to source video
        "--viz-output", output_video,             # Final rendered file
        "--viz-export", "output_3d_data.npy"      # Save the raw 3D coordinates for later use
    ]
    
    # Run the command from inside the VideoPose3D repo, since it resolves
    # its "-c checkpoint" / "-k dataset" arguments relative to the cwd.
    # Prepend the conda env's Library/bin so matplotlib's ffmpeg writer can
    # find ffmpeg.exe (only happens automatically on "conda activate", not
    # when just pointing at this python.exe as an interpreter).
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([
        os.path.join(CONDA_ENV_DIR, "Library", "bin"),
        os.path.join(CONDA_ENV_DIR, "Scripts"),
        CONDA_ENV_DIR,
        env.get("PATH", ""),
    ])
    # PyTorch (libiomp5md.dll) and another dependency (libomp.dll) both bundle
    # an OpenMP runtime; letting both load avoids the OMP Error #15 abort.
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    result = subprocess.run(command, cwd=VIDEOPOSE3D_DIR, env=env)
    if result.returncode != 0:
        print(f"\n--- VideoPose3D exited with code {result.returncode}; no output was generated ---")
        return

    print(f"\n--- Done! Video saved to {os.path.join(VIDEOPOSE3D_DIR, output_video)} ---")
    print(f"--- Raw 3D coordinates saved to {os.path.join(VIDEOPOSE3D_DIR, 'output_3d_data.npy')} ---")

if __name__ == "__main__":
    main()