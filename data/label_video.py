import cv2
import json
import os
from datetime import datetime

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\videos\processed\cam2\cam2_Y1.mp4"
OUTPUT_JSON = r"data/video_labels/labels_lift.json"

ASSIST_KEYS = {
    "h": ("HOLD", 1),
}

def sec_from_frame(frame_idx, fps):
    return frame_idx / fps


def run_label_tool():
    assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_step_id = 0
    current_element_id = 0
    current_assist_type = None
    current_assist_name = None

    paused = False
    events = []
    pending_segment = None

    print("\n==============================")
    print(" WoZ Video Label Tool (Segment Mode)")
    print("==============================")
    print("SPACE : pause / resume")
    print("ENTER : mark END")
    print("h     : select assist type")
    print("u     : undo")
    print("q     : quit and save")
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

        overlay = frame.copy()
        info_lines = [
            f"Time: {t_sec:.2f}s   Frame: {frame_idx}/{total_frames}",
            f"Step ID: {current_step_id}",
            f"Element ID: {current_element_id}",
            f"Assist Type: {current_assist_name}",
        ]

        if pending_segment is not None:
            info_lines.append("STATUS: START marked, waiting END (ENTER)")
        elif len(events) > 0:
            info_lines.append("STATUS: Segment completed")
        else:
            info_lines.append("STATUS: No segment")

        y = 30
        for line in info_lines:
            cv2.putText(
                overlay,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 28

        cv2.imshow("WoZ Label Tool", overlay)

        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == 255:
            continue

        # Quit
        if key == ord("q"):
            print("Quit requested.")
            break

        # Pause / Resume
        if key == ord(" "):
            paused = not paused
            if paused and current_assist_type is not None and pending_segment is None:
                pending_segment = {
                    "start": round(float(t_sec), 3),
                    "step_id": int(current_step_id),
                    "element_id": int(current_element_id),
                    "type": int(current_assist_type),
                    "type_name": current_assist_name,
                }
                print(f"[START] {pending_segment}")

            continue

        # ENTER = mark END
        if key == 13:
            if pending_segment is not None:
                end_time = round(float(t_sec), 3)

                if end_time <= pending_segment["start"]:
                    print("[WARNING] End must be after start")
                else:
                    segment = {
                        "start": pending_segment["start"],
                        "end": end_time,
                        "step_id": pending_segment["step_id"],
                        "element_id": pending_segment["element_id"],
                        "type": pending_segment["type"],
                        "type_name": pending_segment["type_name"],
                    }

                    events.clear()
                    events.append(segment)

                    print(f"[SEGMENT COMPLETED] {segment}")

                pending_segment = None
                paused = False  

            continue

        # Step change
        if key == ord("["):
            current_step_id = max(0, current_step_id - 1)
            print(f"[INFO] step_id -> {current_step_id}")
            continue

        if key == ord("]"):
            current_step_id += 1
            print(f"[INFO] step_id -> {current_step_id}")
            continue

        # Element change
        if key == ord(","):
            current_element_id = max(0, current_element_id - 1)
            print(f"[INFO] element_id -> {current_element_id}")
            continue
        if key == ord("."):
            current_element_id += 1
            print(f"[INFO] element_id -> {current_element_id}")
            continue

        # Undo
        if key == ord("u"):
            if pending_segment is not None:
                print("[UNDO] cancel START")
                pending_segment = None
            elif len(events) > 0:
                removed = events.pop()
                print(f"[UNDO] removed segment: {removed}")
            else:
                print("[UNDO] nothing to undo")
            continue

        ch = chr(key).lower()
        if ch in ASSIST_KEYS:
            current_assist_name, current_assist_type = ASSIST_KEYS[ch]
            print(f"[SELECT] Assist type -> {current_assist_name}")
            continue

    out = {
        "schema": "assist_events_v1",
        "video_path": VIDEO_PATH,
        "created_at": datetime.now().isoformat(),
        "fps": float(fps),
        "assist_events": events,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved labels to: {OUTPUT_JSON}")
    print(f"Total segments: {len(events)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_label_tool()