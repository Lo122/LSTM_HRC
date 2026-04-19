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
from file_io_utils import save_torch
from filter_utils import RealTimeSGFilter, FILTER_CONFIG
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
        video_path: str,
        output_pt: str,
        output_video: str | None = None,
        model_path: str = "yolo26n-pose.pt",
        show_every_n_frames: int = 20,
        use_visibility: bool = False,
        use_tracking: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        
        self.video_path = video_path
        self.output_pt = output_pt
        self.output_video = output_video
        self.model_path = model_path
        self.show_every_n_frames = show_every_n_frames
        self.use_visibility = use_visibility
        self.use_tracking = use_tracking
        self.logger = logger or logging.getLogger(__name__)

        self.device = self._get_inference_device()
        self.yolo_model = YOLO(self.model_path)
        self.filter = RealTimeSGFilter(
            window_size=FILTER_CONFIG["window_size"],
            poly_order=FILTER_CONFIG["poly_order"]
            )
    
    
    def run_pose_extraction(self) -> Dict[str, Any]:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_out: List[Dict[str, torch.Tensor]] = []
        video_writer: cv2.VideoWriter | None = None

        if self.output_video:
            output_dir = os.path.dirname(self.output_video)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            output_fps = fps if fps > 0 else 30.0
            video_writer = cv2.VideoWriter(self.output_video, fourcc, output_fps, (frame_width, frame_height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not open video writer for output: {self.output_video}")

        self.logger.info("==============================")
        self.logger.info("YOLO Pose Extraction Tool")
        self.logger.info("Video: %s", self.video_path)
        self.logger.info("FPS: %s", fps)
        self.logger.info("Total frames: %s", total_frames)
        self.logger.info("Device: %s", self.device)
        if self.output_video:
            self.logger.info("Output video: %s", self.output_video)
        self.logger.info("==============================")

        frame_idx = 0
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

            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                if self.use_visibility and hasattr(result.keypoints, "data"):
                    raw_kpts = result.keypoints.data[0].cpu()[..., :2]
                    keypoints_are_normalized = False
                else:
                    raw_kpts = result.keypoints.xyn[0].cpu()[..., :2]
                    keypoints_are_normalized = True
                    
                smoothed_kpts = self.filter.update(raw_kpts.clone())
                kpts = self._normalize_keypoints(smoothed_kpts)
                
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
            )

            if video_writer is not None:
                video_writer.write(plotted_frame)

            frames_out.append({
                "norm_kpts": kpts,
                "smoothed_kpts": smoothed_kpts,
                "raw_kpts":  raw_kpts,   # xyn in [0,1] — used for video overlay
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

        save_data = {
            "metadata": {
                "video": self.video_path,
                "fps": fps,
                "model": self.model_path,
                "device": self.device,
                "tracking": self.use_tracking,
                "filter_config": {
                    "window_size": self.filter.window_size,
                    "poly_order": self.filter.poly_order,
                },
            },
            "norm_landmarks":     torch.stack([frame["norm_kpts"] for frame in frames_out]),
            "smoothed_landmarks":     torch.stack([frame["smoothed_kpts"] for frame in frames_out]),
            "raw_landmarks": torch.stack([frame["raw_kpts"] for frame in frames_out]),
            "t_steps":       torch.stack([frame["t"] for frame in frames_out]),
        }

        self._save_posture_data(save_data, len(frames_out))
        return save_data
            

    # =========================================================================
    # Internal helper methods
    # =========================================================================
    @staticmethod
    def _sec_from_frame(frame_idx: int, fps: float) -> float:
        return frame_idx / fps if fps > 0 else 0.0


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
    ) -> np.ndarray:
        if torch.all(smoothed_kpts == 0):
            return frame_bgr.copy()

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
        return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)


    


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