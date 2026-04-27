import math
import os
import logging
import sys
from typing import Any, Dict, List
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent))
from file_io_utils import load_torch, save_torch
from filter_utils import RealTimeSGFilter, PerJointKinematicTracker, FILTER_CONFIG
from viewer.view.overlay import draw_pose_overlay


SMOOTHED_SEG_COLOURS = {
    "head": "#f9c74f",
    "arm": "#90be6d",
    "torso": "#4ecdc4",
    "leg": "#577590",
}
SMOOTHED_JOINT_COLOUR = "#ff6b35"
SMOOTHED_BODY_CENTER_CFG = [
    ("mid-hip", "#ff6b35"),
    ("mid-shoulder", "#00b4d8"),
    ("half-body", "#c77dff"),
]


class VideoPoseExtractor:
    def __init__(
        self,
        video_path: str | None = None,
        output_pt: str | None = None,
        output_video: str | None = None,
        model_path: str = "yolo26n-pose.pt",
        show_every_n_frames: int = 20,
        use_visibility: bool = False,
        use_tracking: bool = True,
        input_pt: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if output_pt is None:
            raise ValueError("output_pt must be provided.")
        if video_path is None and input_pt is None:
            raise ValueError("Either video_path or input_pt must be provided.")

        self.video_path = video_path
        self.input_pt = input_pt
        self.output_pt = output_pt
        self.output_video = output_video
        self.model_path = model_path
        self.show_every_n_frames = show_every_n_frames
        self.use_visibility = use_visibility
        self.use_tracking = use_tracking
        self.logger = logger or logging.getLogger(__name__)
        self.source_mode = "pt" if self.input_pt else "video"

        self.device = "cpu" if self.input_pt else self._get_inference_device()
        self.yolo_model = None if self.input_pt else YOLO(self.model_path)
        self.filter = RealTimeSGFilter(
            window_size=FILTER_CONFIG["window_size"],
            poly_order=FILTER_CONFIG["poly_order"]
            )
        self.kinematic_tracker = PerJointKinematicTracker(
            sg_filter=self.filter,
            num_joints=17,
            dims_per_joint=2,
            max_jump=FILTER_CONFIG["max_jump"],  # Max allowed jump in normalized coordinates (e.g., 0.5 means 50% of the frame)
            max_hold_frames=FILTER_CONFIG["max_hold_frames"],  # Max frames to hold a lost joint before resetting
        )
    
    def run_pose_extraction(self) -> Dict[str, Any]:
        self.filter.reset()
        self.kinematic_tracker.reset()

        if self.input_pt is not None:
            return self._run_pose_extraction_from_pt()
        return self._run_pose_extraction_from_video()


    def _create_video_writer(
        self,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> cv2.VideoWriter | None:
        if not self.output_video:
            return None

        output_dir = os.path.dirname(self.output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        output_fps = fps if fps > 0 else 30.0
        video_writer = cv2.VideoWriter(self.output_video, fourcc, output_fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for output: {self.output_video}")
        return video_writer


    def _filter_pose_frame(
        self,
        raw_kpts: torch.Tensor,
        det_conf: torch.Tensor | None,
        joint_conf: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        try:
            raw_kpts = self._purse_fileter_pose_data(raw_kpts)
        
        except ValueError as e:
            self.logger.warning("Error processing raw keypoints: %s. Using zeros for this frame.", e)
            zeros = torch.zeros((self.kinematic_tracker.num_joints, self.kinematic_tracker.dims_per_joint), dtype=torch.float32)
            return zeros, zeros
        
        
        smoothed_kpts = self.kinematic_tracker.update(raw_kpts, joint_conf, det_conf)
        if smoothed_kpts is None:
            smoothed_kpts = torch.zeros_like(raw_kpts)
        return smoothed_kpts, self._normalize_keypoints(smoothed_kpts)


    def _purse_fileter_pose_data(self, tensor: torch.Tensor) -> torch.Tensor:
        
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.detach().clone()
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        
        tensor = tensor.reshape(self.kinematic_tracker.num_joints,
                                self.kinematic_tracker.dims_per_joint)
        
        if torch.all(tensor == 0):
            raise ValueError("All elements in the tensor are zero")
        
        return tensor


    def _build_save_data(
        self,
        frames_out: List[Dict[str, torch.Tensor]],
        fps: float,
        metadata: Dict[str, Any],
        base_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        save_data = dict(base_data) if base_data is not None else {}
        updated_metadata = dict(metadata)
        updated_metadata["fps"] = fps if fps > 0 else metadata.get("fps", 0.0)
        updated_metadata["filter_config"] = {
            "window_size": self.filter.window_size,
            "poly_order": self.filter.poly_order,
            "max_jump": self.kinematic_tracker.max_jump,
            "max_hold_frames": self.kinematic_tracker.max_hold_frames,
        }

        if self.input_pt is None:
            updated_metadata.update({
                "video": self.video_path,
                "model": self.model_path,
                "device": self.device,
                "tracking": self.use_tracking,
            })
        else:
            updated_metadata["video"] = self.video_path or metadata.get("video")
            updated_metadata["refilter_source_pt"] = self.input_pt
            updated_metadata["refilter_device"] = self.device
            updated_metadata["processing_source"] = "pt"

        save_data.update({
            "metadata": updated_metadata,
            "norm_landmarks": torch.stack([frame["norm_kpts"] for frame in frames_out]),
            "smoothed_landmarks": torch.stack([frame["smoothed_kpts"] for frame in frames_out]),
            "raw_landmarks": torch.stack([frame["raw_kpts"] for frame in frames_out]),
            "joint_confidence": torch.stack([frame["joint_conf"] for frame in frames_out]),
            "detection_confidence": torch.stack([frame["det_conf"] for frame in frames_out]),
            "t_steps": torch.stack([frame["t"] for frame in frames_out]),
        })
        return save_data


    def _run_pose_extraction_from_video(self) -> Dict[str, Any]:
        video_path = self.video_path
        assert video_path is not None
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_out: List[Dict[str, torch.Tensor]] = []
        video_writer = self._create_video_writer(fps, frame_width, frame_height)

        self.logger.info("==============================")
        self.logger.info("YOLO Pose Extraction Tool")
        self.logger.info("Video: %s", video_path)
        self.logger.info("FPS: %s", fps)
        self.logger.info("Total frames: %s", total_frames)
        self.logger.info("Device: %s", self.device)
        if self.output_video:
            self.logger.info("Output video: %s", self.output_video)
        self.logger.info("==============================")

        frame_idx = 0
        raw_landmarks_normalized: bool | None = None
        assert self.yolo_model is not None
        for _ in tqdm(range(total_frames), desc="Extracting pose"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_sec = self._sec_from_frame(frame_idx, fps)

            if self.use_tracking:
                results = self.yolo_model.track(frame, device=self.device, persist=True, verbose=False)
            else:
                results = self.yolo_model(frame, device=self.device, verbose=False)
            result = results[0]
            
            # confidence scores for the whole detection and individual joints (if available)
            det_index = 0
            det_conf = torch.tensor(0.0, dtype=torch.float32)
            joint_conf = torch.zeros(17, dtype=torch.float32)
            det_conf_for_tracker = None
            joint_conf_for_tracker = None
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                keypoint_data = result.keypoints.data[det_index].cpu()
                if keypoint_data.shape[-1] > 2:
                    joint_conf = keypoint_data[..., 2].to(dtype=torch.float32)
                    joint_conf_for_tracker = joint_conf
                if result.boxes is not None and len(result.boxes) > det_index:
                    det_conf = result.boxes.conf[det_index].detach().cpu().to(dtype=torch.float32)
                    det_conf_for_tracker = det_conf

            # YOLOv8n-pose returns keypoints in xyn format
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                if self.use_visibility and hasattr(result.keypoints, "data"):
                    raw_kpts = result.keypoints.data[det_index].cpu()[..., :2]
                    keypoints_are_normalized = False
                else:
                    raw_kpts = result.keypoints.xyn[det_index].cpu()[..., :2]
                    keypoints_are_normalized = True

                if raw_landmarks_normalized is None:
                    raw_landmarks_normalized = keypoints_are_normalized
                
                smoothed_kpts, kpts = self._filter_pose_frame(
                    raw_kpts,
                    det_conf_for_tracker,
                    joint_conf_for_tracker,
                )
                
            else:
                raw_kpts = torch.zeros((17, 2), dtype=torch.float32)
                kpts     = torch.zeros((17, 2), dtype=torch.float32)
                smoothed_kpts = torch.zeros((17, 2), dtype=torch.float32)
                keypoints_are_normalized = True

            plotted_frame = self._render_smoothed_pose_frame(
                frame_bgr=frame,
                smoothed_kpts=smoothed_kpts,
                frame_width=frame_width,
                frame_height=frame_height,
                normalized_input=keypoints_are_normalized,
                joint_conf=joint_conf,
                det_conf=det_conf,
            )

            if video_writer is not None:
                video_writer.write(plotted_frame)

            frames_out.append({
                "norm_kpts": kpts,
                "smoothed_kpts": smoothed_kpts,
                "raw_kpts":  raw_kpts,   # xyn in [0,1] — used for video overlay
                "joint_conf": joint_conf,
                "det_conf": det_conf,
                "t":         torch.tensor(t_sec, dtype=torch.float32),
            })

            if self.show_every_n_frames > 0 and frame_idx % self.show_every_n_frames == 0:
                cv2.imshow("Debug", plotted_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if video_writer is not None:
            video_writer.release()
            self.logger.info("Saved smoothed pose video to %s", self.output_video)
        cv2.destroyAllWindows()

        if not frames_out:
            raise RuntimeError("No frames were processed. Check your input video path and codec support.")

        save_data = self._build_save_data(
            frames_out,
            fps=fps,
            metadata={
                "video": video_path,
                "total_frames": total_frames,
                "video_frame_width": frame_width,
                "video_frame_height": frame_height,
                "raw_landmarks_normalized": raw_landmarks_normalized if raw_landmarks_normalized is not None else (not self.use_visibility),
                "filter_config": {
                    "window_size": self.filter.window_size,
                    "poly_order": self.filter.poly_order,
                    "max_jump": self.kinematic_tracker.max_jump,
                    "max_hold_frames": self.kinematic_tracker.max_hold_frames
                    }
            },
        )

        self._save_posture_data(save_data, len(frames_out))
        return save_data


    def _run_pose_extraction_from_pt(self) -> Dict[str, Any]:
        assert self.input_pt is not None
        if not os.path.exists(self.input_pt):
            raise FileNotFoundError(f"Skeleton PT file not found: {self.input_pt}")

        source_data = load_torch(self.input_pt, logger=self.logger)
        source_metadata = dict(source_data.get("metadata", {}))
        raw_landmarks = source_data.get("raw_landmarks")
        if raw_landmarks is None:
            raise KeyError(
                "raw_landmarks not found in the PT file. Re-filtering from PT requires the saved raw skeleton tensor."
            )

        if not isinstance(raw_landmarks, torch.Tensor):
            raw_landmarks = torch.as_tensor(raw_landmarks)
        raw_landmarks = raw_landmarks.detach().clone()
        if not raw_landmarks.is_floating_point():
            raw_landmarks = raw_landmarks.to(dtype=torch.float32)

        expected_joint_shape = (
            self.kinematic_tracker.num_joints,
            self.kinematic_tracker.dims_per_joint,
        )
        if raw_landmarks.ndim != 3 or tuple(raw_landmarks.shape[1:]) != expected_joint_shape:
            raise ValueError(
                f"Expected raw_landmarks with shape (T, {expected_joint_shape[0]}, {expected_joint_shape[1]}), "
                f"but got {tuple(raw_landmarks.shape)}"
            )

        raw_landmarks_normalized = self._raw_landmarks_are_normalized(raw_landmarks, source_metadata)
        total_frames = int(raw_landmarks.shape[0])

        fps = float(source_metadata.get("fps", 0.0) or 0.0)
        time_steps = source_data.get("t_steps")
        if time_steps is not None:
            if not isinstance(time_steps, torch.Tensor):
                time_steps = torch.as_tensor(time_steps)
            time_steps = time_steps.detach().clone().reshape(-1).to(dtype=torch.float32)

        joint_confidence = source_data.get("joint_confidence")
        if joint_confidence is not None:
            if not isinstance(joint_confidence, torch.Tensor):
                joint_confidence = torch.as_tensor(joint_confidence)
            joint_confidence = joint_confidence.detach().clone().to(dtype=torch.float32)
            if joint_confidence.ndim == 1:
                joint_confidence = joint_confidence.reshape(1, -1)
            if (
                joint_confidence.ndim != 2
                or int(joint_confidence.shape[1]) != self.kinematic_tracker.num_joints
            ):
                self.logger.warning(
                    "Ignoring joint_confidence with unexpected shape: %s",
                    tuple(joint_confidence.shape),
                )
                joint_confidence = None

        detection_confidence = source_data.get("detection_confidence")
        if detection_confidence is not None:
            if not isinstance(detection_confidence, torch.Tensor):
                detection_confidence = torch.as_tensor(detection_confidence)
            detection_confidence = detection_confidence.detach().clone().reshape(-1).to(dtype=torch.float32)

        source_video_path = self._resolve_video_path(self.video_path or source_metadata.get("video"))
        cap: cv2.VideoCapture | None = None
        video_writer: cv2.VideoWriter | None = None
        frame_width = 0
        frame_height = 0

        if self.output_video or self.show_every_n_frames > 0:
            if source_video_path and os.path.exists(source_video_path):
                cap = cv2.VideoCapture(source_video_path)
                fps_from_video = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 and fps_from_video > 0:
                    fps = fps_from_video
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self.output_video:
                    video_writer = self._create_video_writer(fps, frame_width, frame_height)
            else:
                self.logger.warning(
                    "Skipping PT video preview/output because the source video could not be resolved from PT metadata: %s",
                    source_video_path,
                )

        frames_out: List[Dict[str, torch.Tensor]] = []

        self.logger.info("==============================")
        self.logger.info("Skeleton PT Re-filter Tool")
        self.logger.info("Source PT: %s", self.input_pt)
        self.logger.info("Frames: %s", total_frames)
        self.logger.info("Source video: %s", source_video_path)
        self.logger.info("Raw landmarks normalized: %s", raw_landmarks_normalized)
        if self.output_video:
            self.logger.info("Output video: %s", self.output_video)
        self.logger.info("==============================")

        video_exhausted = False
        for frame_index in tqdm(range(total_frames), desc="Re-filtering pose"):
            raw_kpts = raw_landmarks[frame_index]
            joint_conf = torch.zeros(self.kinematic_tracker.num_joints, dtype=torch.float32)
            det_conf = torch.tensor(0.0, dtype=torch.float32)
            joint_conf_for_tracker = None
            det_conf_for_tracker = None
            if joint_confidence is not None and frame_index < int(joint_confidence.shape[0]):
                joint_conf = joint_confidence[frame_index].detach().clone().reshape(-1).to(dtype=torch.float32)
                if joint_conf.numel() != self.kinematic_tracker.num_joints:
                    joint_conf = torch.zeros(self.kinematic_tracker.num_joints, dtype=torch.float32)
                else:
                    joint_conf_for_tracker = joint_conf
            if detection_confidence is not None and frame_index < int(detection_confidence.shape[0]):
                det_conf = detection_confidence[frame_index].detach().clone().to(dtype=torch.float32)
                det_conf_for_tracker = det_conf

            smoothed_kpts, kpts = self._filter_pose_frame(
                raw_kpts,
                det_conf_for_tracker,
                joint_conf_for_tracker,
            )

            if time_steps is not None and frame_index < int(time_steps.shape[0]):
                t_sec = time_steps[frame_index].clone()
            else:
                t_sec = torch.tensor(self._sec_from_frame(frame_index + 1, fps), dtype=torch.float32)

            if cap is not None and not video_exhausted:
                ret, frame = cap.read()
                if ret:
                    plotted_frame = self._render_smoothed_pose_frame(
                        frame_bgr=frame,
                        smoothed_kpts=smoothed_kpts,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        normalized_input=raw_landmarks_normalized,
                        joint_conf=joint_conf,
                        det_conf=det_conf,
                    )
                    if video_writer is not None:
                        video_writer.write(plotted_frame)

                    if self.show_every_n_frames > 0 and (frame_index + 1) % self.show_every_n_frames == 0:
                        cv2.imshow("Debug", plotted_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                else:
                    self.logger.warning(
                        "Source video ended before all PT frames were rendered. Output video will stop at frame %s.",
                        frame_index,
                    )
                    video_exhausted = True

            frames_out.append({
                "norm_kpts": kpts,
                "smoothed_kpts": smoothed_kpts,
                "raw_kpts": raw_kpts.detach().clone().to(dtype=torch.float32),
                "joint_conf": joint_conf,
                "det_conf": det_conf,
                "t": t_sec,
            })

        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
            self.logger.info("Saved smoothed pose video to %s", self.output_video)
        cv2.destroyAllWindows()

        if not frames_out:
            raise RuntimeError("No frames were loaded from the skeleton PT file.")

        save_data = self._build_save_data(
            frames_out,
            fps=fps,
            metadata={
                **source_metadata,
                "raw_landmarks_normalized": raw_landmarks_normalized,
            },
            base_data=source_data,
        )
        self._save_posture_data(save_data, len(frames_out))
        return save_data
            

    # =========================================================================
    # Internal helper methods
    # =========================================================================
    @staticmethod
    def _sec_from_frame(frame_idx: int, fps: float) -> float:
        return frame_idx / fps if fps > 0 else 0.0


    @staticmethod
    def _raw_landmarks_are_normalized(raw_landmarks: torch.Tensor, metadata: Dict[str, Any]) -> bool:
        metadata_value = metadata.get("raw_landmarks_normalized")
        if metadata_value is not None:
            return bool(metadata_value)

        valid_values = raw_landmarks[torch.isfinite(raw_landmarks) & (raw_landmarks.abs() > 1e-6)]
        if valid_values.numel() == 0:
            return True

        min_value = float(valid_values.min().item())
        max_value = float(valid_values.max().item())
        return min_value >= -1e-6 and max_value <= 1.5


    @staticmethod
    def _normalize_keypoints(kpts_tensor: torch.Tensor, nomalize_by_uppper_body_center:bool = True) -> torch.Tensor:
        if torch.all(kpts_tensor == 0):
            return kpts_tensor

        if nomalize_by_uppper_body_center:
            center = kpts_tensor[[5, 6, 11, 12]].mean(dim=0)  # shoulders and hips center   
        else:
            center = kpts_tensor.mean(dim=0)
        kpts_preprocessed = kpts_tensor - center

        distances = torch.norm(kpts_preprocessed, dim=1)
        scale = distances.mean() + 1e-6
        return kpts_preprocessed / scale


    @staticmethod
    def _get_inference_device() -> str:
        if not torch.cuda.is_available():
            return "cpu"

        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as error:
            logging.getLogger(__name__).warning("CUDA unavailable, falling back to CPU: %s", error)
            return "cpu"


    def _save_posture_data(self, save_data: Dict[str, Any], num_frames: int) -> None:
        save_torch(save_data, self.output_pt, logger=self.logger)
        self.logger.info("Saved %s frames to %s", num_frames, self.output_pt)


    def _resolve_video_path(self, source_video_path: str | None) -> str | None:
        if not source_video_path:
            return None

        source_path = Path(source_video_path)
        if source_path.exists():
            return str(source_path)

        if self.input_pt is not None and not source_path.is_absolute():
            candidate = Path(self.input_pt).resolve().parent / source_path
            if candidate.exists():
                return str(candidate)

        return source_video_path


    @staticmethod
    def _draw_text_with_outline(
        frame_bgr: np.ndarray,
        text: str,
        origin: tuple[int, int],
        colour: tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.38,
    ) -> None:
        cv2.putText(
            frame_bgr,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            1,
            cv2.LINE_AA,
        )


    def _annotate_confidence_overlay(
        self,
        frame_bgr: np.ndarray,
        smoothed_kpts: torch.Tensor,
        frame_width: int,
        frame_height: int,
        normalized_input: bool,
        joint_conf: torch.Tensor | None,
        det_conf: torch.Tensor | None,
    ) -> np.ndarray:
        annotated_frame = frame_bgr.copy()
        det_conf_value = 0.0
        if det_conf is not None:
            det_conf_value = float(torch.as_tensor(det_conf, dtype=torch.float32).item())

        self._draw_text_with_outline(
            annotated_frame,
            f"det conf: {det_conf_value:.2f}",
            (12, 24),
            colour=(0, 255, 255),
            font_scale=0.55,
        )

        if torch.all(smoothed_kpts == 0):
            return annotated_frame

        if joint_conf is None:
            joint_conf = torch.zeros(self.kinematic_tracker.num_joints, dtype=torch.float32)
        else:
            joint_conf = torch.as_tensor(joint_conf, dtype=torch.float32).reshape(-1)

        pixel_points = (
            self._to_normalized_keypoints(
                smoothed_kpts,
                frame_width=frame_width,
                frame_height=frame_height,
                normalized_input=normalized_input,
            )
            * np.array([max(frame_width, 1), max(frame_height, 1)], dtype=np.float32)
        ).round().astype(int)

        for joint_index, point in enumerate(pixel_points):
            if point.sum() == 0:
                continue

            conf_value = 0.0
            if joint_index < joint_conf.numel():
                conf_value = float(joint_conf[joint_index].item())

            text_x = int(np.clip(point[0] + 6, 0, max(frame_width - 40, 0)))
            text_y = int(np.clip(point[1] - 6, 12, max(frame_height - 4, 12)))
            self._draw_text_with_outline(
                annotated_frame,
                f"{conf_value:.2f}",
                (text_x, text_y),
            )

        return annotated_frame


    @staticmethod
    def _to_normalized_keypoints(
        kpts_tensor: torch.Tensor,
        frame_width: int,
        frame_height: int,
        normalized_input: bool,
    ) -> np.ndarray:
        kpts_np = kpts_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if normalized_input:
            return np.clip(kpts_np, 0.0, 1.0)

        scale = np.array([max(frame_width, 1), max(frame_height, 1)], dtype=np.float32)
        return np.clip(kpts_np / scale, 0.0, 1.0)


    def _render_smoothed_pose_frame(
        self,
        frame_bgr: np.ndarray,
        smoothed_kpts: torch.Tensor,
        frame_width: int,
        frame_height: int,
        normalized_input: bool,
        joint_conf: torch.Tensor | None = None,
        det_conf: torch.Tensor | None = None,
    ) -> np.ndarray:
        if torch.all(smoothed_kpts == 0):
            overlay_bgr = frame_bgr.copy()
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = draw_pose_overlay(
                frame_rgb=frame_rgb,
                kpts_xyn=self._to_normalized_keypoints(
                    smoothed_kpts,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    normalized_input=normalized_input,
                ),
                seg_colours=SMOOTHED_SEG_COLOURS,
                joint_colour=SMOOTHED_JOINT_COLOUR,
                body_center_cfg=SMOOTHED_BODY_CENTER_CFG,
            )
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        return self._annotate_confidence_overlay(
            frame_bgr=overlay_bgr,
            smoothed_kpts=smoothed_kpts,
            frame_width=frame_width,
            frame_height=frame_height,
            normalized_input=normalized_input,
            joint_conf=joint_conf,
            det_conf=det_conf,
        )


    


if __name__ == "__main__":
    extractor = VideoPoseExtractor(
        video_path=r"G:\My Drive\University of Stuttgart\ITECH_Thesis\videos\originals\camera_01\Video__CID-01_UID-01_01.MOV",
        output_pt=r"dataset/Video__CID-01_UID-01_01.pt",
        output_video=r"dataset/Video__CID-01_UID-01_01_pose.mp4",
        model_path="yolo26n-pose.pt",
        show_every_n_frames=0,
        use_visibility=False,
    )
    extractor.run_pose_extraction()