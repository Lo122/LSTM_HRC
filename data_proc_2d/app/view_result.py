
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.file_io_utils import iter_files, load_torch
from src.human_pose_extractor import VideoPoseExtractor
from utilities import log_utils

video_root_path = Path(r"G:\.shortcut-targets-by-id\1Ykdzx6UjCe0KPKy_6M4LgCOTxKK6Awgy\Videos")
PT_ROOT = video_root_path / "dataset" / "keypoints"
FILE_NAME_FILTER = "keypoints__cam_04_uid-05_take-1"
USE_SMOOTHED_LANDMARKS = True
WINDOW_NAME = "Skeleton Result Preview"


def _select_pt_file(root: Path, file_name_filter: str | None, logger) -> Path:
    pt_files = list(iter_files(root, extension=".pt"))
    if file_name_filter:
        filter_text = file_name_filter.lower()
        pt_files = [path for path in pt_files if filter_text in path.stem.lower()]

    if not pt_files:
        raise FileNotFoundError(f"No PT files found under {root} matching filter={file_name_filter!r}")

    selected_file = pt_files[0]
    logger.info("Selected PT file: %s", selected_file)
    if len(pt_files) > 1:
        logger.info("Multiple matches found (%s). Using the first match.", len(pt_files))
    return selected_file


def _as_float_tensor(data: object | None) -> torch.Tensor | None:
    if data is None:
        return None

    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    return tensor.detach().clone().to(dtype=torch.float32)


def _as_int(value: object | None, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, (bool, int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating, str)):
            return int(value)
        return default
    except (TypeError, ValueError):
        return default


def _as_float(value: object | None, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        return default
    except (TypeError, ValueError):
        return default


def _get_preview_landmarks(source_data: dict, use_smoothed_landmarks: bool) -> torch.Tensor:
    preferred_keys = ["smoothed_landmarks", "raw_landmarks"] if use_smoothed_landmarks else ["raw_landmarks", "smoothed_landmarks"]
    for key in preferred_keys:
        tensor = _as_float_tensor(source_data.get(key))
        if tensor is not None:
            return tensor

    raise KeyError("The PT file does not contain smoothed_landmarks or raw_landmarks.")


def _build_blank_frame(metadata: dict[str, object]) -> np.ndarray:
    frame_width = _as_int(metadata.get("video_frame_width"), 1280)
    frame_height = _as_int(metadata.get("video_frame_height"), 720)
    return np.zeros((max(frame_height, 1), max(frame_width, 1), 3), dtype=np.uint8)


def preview_pt_file(pt_path: Path, use_smoothed_landmarks: bool = True) -> None:
    logger = log_utils.setup_logger("view_result")
    source_data = load_torch(str(pt_path), logger=logger)
    metadata = dict(source_data.get("metadata", {}))

    preview_landmarks = _get_preview_landmarks(source_data, use_smoothed_landmarks)
    raw_landmarks = _as_float_tensor(source_data.get("raw_landmarks"))
    joint_confidence = _as_float_tensor(source_data.get("joint_confidence"))
    detection_confidence = _as_float_tensor(source_data.get("detection_confidence"))

    preview_extractor = VideoPoseExtractor(
        input_pt=str(pt_path),
        output_pt=str(pt_path),
        output_video=None,
        show_every_n_frames=0,
        logger=logger,
    )

    if raw_landmarks is None:
        normalized_input = bool(metadata.get("raw_landmarks_normalized", True))
    else:
        normalized_input = preview_extractor._raw_landmarks_are_normalized(raw_landmarks, metadata)

    source_video_path = preview_extractor._resolve_video_path(metadata.get("video"))
    cap: cv2.VideoCapture | None = None
    fps = _as_float(metadata.get("fps"), 0.0)
    frame_width = _as_int(metadata.get("video_frame_width"), 0)
    frame_height = _as_int(metadata.get("video_frame_height"), 0)

    if source_video_path and Path(source_video_path).exists():
        cap = cv2.VideoCapture(source_video_path)
        if cap.isOpened():
            fps_from_video = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 and fps_from_video > 0:
                fps = fps_from_video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("Previewing with source video: %s", source_video_path)
        else:
            logger.warning("Could not open source video for preview: %s", source_video_path)
            cap.release()
            cap = None
    else:
        logger.warning("Source video missing in metadata or on disk. Preview will use a blank background.")

    if frame_width <= 0 or frame_height <= 0:
        blank_frame = _build_blank_frame(metadata)
        frame_height, frame_width = blank_frame.shape[:2]
    else:
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    total_frames = int(preview_landmarks.shape[0])
    playback_delay_ms = max(1, int(round(1000.0 / fps))) if fps > 0 else 1

    logger.info("Frames available in PT: %s", total_frames)
    logger.info("Controls: q=quit, space=pause/resume")

    paused = False
    frame_index = 0
    while frame_index < total_frames:
        if not paused:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Source video ended before PT landmarks. Switching to blank background.")
                    cap.release()
                    cap = None
                    frame = blank_frame.copy()
            else:
                frame = blank_frame.copy()

            smoothed_kpts = preview_landmarks[frame_index]
            joint_conf = None
            det_conf = None
            if joint_confidence is not None and frame_index < int(joint_confidence.shape[0]):
                joint_conf = joint_confidence[frame_index]
            if detection_confidence is not None and frame_index < int(detection_confidence.shape[0]):
                det_conf = detection_confidence[frame_index]

            preview_frame = preview_extractor._render_smoothed_pose_frame(
                frame_bgr=frame,
                smoothed_kpts=smoothed_kpts,
                frame_width=frame_width,
                frame_height=frame_height,
                normalized_input=normalized_input,
                joint_conf=joint_conf,
                det_conf=det_conf,
            )
            preview_extractor._draw_text_with_outline(
                preview_frame,
                f"frame {frame_index + 1}/{total_frames}",
                (12, 48),
                colour=(0, 255, 0),
                font_scale=0.55,
            )
            cv2.imshow(WINDOW_NAME, preview_frame)
            frame_index += 1

        key = cv2.waitKey(0 if paused else playback_delay_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    logger = log_utils.setup_logger("view_result")
    selected_pt = _select_pt_file(PT_ROOT, FILE_NAME_FILTER, logger)
    preview_pt_file(selected_pt, use_smoothed_landmarks=USE_SMOOTHED_LANDMARKS)


if __name__ == "__main__":
    main()
        