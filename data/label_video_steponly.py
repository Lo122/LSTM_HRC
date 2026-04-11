import cv2
import json
import os
from datetime import datetime

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos\processed\cam1\cam1_M4.mp4"
OUTPUT_JSON = os.path.join("data/video_labels", os.path.basename(VIDEO_PATH).replace(".mp4", "_steps.json"))

def sec_from_frame(frame_idx, fps):
    return frame_idx / fps

def run_step_label_tool():
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_step_id = 0
    paused = False
    step_events = [] # To store markers: {"timestamp": xxx, "step_id": xxx}

    print("\n==============================")
    print(" Multi-Step Label Tool (Manual ID)")
    print("==============================")
    print("SPACE : pause / resume")
    print("ENTER : Mark current time with CURRENT Step ID")
    print("[     : Decrease Step ID")
    print("]     : Increase Step ID")
    print("u     : Undo last marker")
    print("q     : Quit and save")
    print("==============================\n")

    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx += 1

        t_sec = sec_from_frame(frame_idx, fps)

        # UI Overlay
        overlay = frame.copy()
        info_lines = [
            f"Time: {t_sec:.2f}s | Frame: {frame_idx}/{total_frames}",
            f"MANUAL Step ID: {current_step_id}",
            f"Total Markers: {len(step_events)}",
        ]
        
        if len(step_events) > 0:
            last = step_events[-1]
            info_lines.append(f"Last Saved: {last['timestamp']}s -> Step {last['step_id']}")

        y = 30
        for line in info_lines:
            cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30

        cv2.imshow("Multi-Step Label Tool", overlay)

        # Key handling
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord("q"): 
            break

        # Mark Step Point (No auto-increment)
        if key == 13: # ENTER
            marker = {
                "timestamp": round(float(t_sec), 3),
                "step_id": int(current_step_id)
            }
            step_events.append(marker)
            print(f"[MARKED] Saved Step {current_step_id} at {marker['timestamp']}s")
            continue

        if key == ord(" "):
            paused = not paused
            continue

        # Manual Step ID Controls
        if key == ord("["):
            current_step_id = max(0, current_step_id - 1)
            print(f"[ID CHANGE] Current Step ID set to: {current_step_id}")
            continue
        if key == ord("]"):
            current_step_id += 1
            print(f"[ID CHANGE] Current Step ID set to: {current_step_id}")
            continue

        # Undo
        if key == ord("u"):
            if step_events:
                removed = step_events.pop()
                print(f"[UNDO] Removed marker: {removed}")
            continue

    # Final Output
    out = {
        "video_path": VIDEO_PATH,
        "fps": float(fps),
        "step_markers": step_events,
        "labeling_date": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSuccessfully saved {len(step_events)} markers to {OUTPUT_JSON}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_step_label_tool()