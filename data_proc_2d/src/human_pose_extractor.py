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


class _OneEuroFilter:
    """Per-scalar One Euro Filter (Casiez et al., 2012).

    Smooths a 1-D signal adaptively: slow motion gets heavy smoothing,
    fast motion gets light smoothing to avoid lag.
    """

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self._freq = max(freq, 1e-6)
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self._freq
        a_d = self._alpha(self._d_cutoff, self._freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, self._freq)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
    


class SGFilter():
    def __init__(self) -> None:
        pass


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
        smoothing: str = "one_euro",
        logger: logging.Logger | None = None,
    ) -> None:
        
        self.video_path = video_path
        self.output_pt = output_pt
        self.output_video = output_video
        self.model_path = model_path
        self.show_every_n_frames = show_every_n_frames
        self.use_visibility = use_visibility
        self.use_tracking = use_tracking
        self.smoothing = smoothing
        self.logger = logger or logging.getLogger(__name__)

        self.device = self._get_inference_device()
        self.yolo_model = YOLO(self.model_path)
        self.logger.info("Tracking: %s | Smoothing: %s", self.use_tracking, self.smoothing)


    @staticmethod
    def _sec_from_frame(frame_idx: int, fps: float) -> float:
        return frame_idx / fps if fps > 0 else 0.0


    @staticmethod
    def _normalize_keypoints(kpts_tensor: torch.Tensor) -> torch.Tensor:
        if torch.all(kpts_tensor == 0):
            return kpts_tensor

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


    def _smooth_landmarks(self, landmarks: torch.Tensor, fps: float) -> torch.Tensor:
        """Apply One Euro Filter per keypoint per axis over the time dimension."""
        arr = landmarks.numpy()           # (T, K, D)
        T, K, D = arr.shape
        freq = fps if fps > 0 else 30.0
        out = arr.copy()
        for k in range(K):
            for d in range(D):
                f = _OneEuroFilter(freq=freq)
                for t in range(T):
                    out[t, k, d] = f(float(arr[t, k, d]))
        return torch.from_numpy(out.astype(np.float32))


    def _save_posture_data(self, save_data: Dict[str, Any], num_frames: int) -> None:
        save_torch(save_data, self.output_pt, logger=self.logger)
        self.logger.info("Saved %s frames to %s", num_frames, self.output_pt)


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
            plotted_frame = result.plot()

            if video_writer is not None:
                video_writer.write(plotted_frame)

            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                if self.use_visibility and hasattr(result.keypoints, "data"):
                    raw_kpts = result.keypoints.data[0].cpu()[..., :2]
                else:
                    raw_kpts = result.keypoints.xyn[0].cpu()[..., :2]
                kpts = self._normalize_keypoints(raw_kpts.clone())
            else:
                raw_kpts = torch.zeros((17, 2), dtype=torch.float32)
                kpts     = torch.zeros((17, 2), dtype=torch.float32)

            frames_out.append({
                "norm_kpts": kpts,
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
            self.logger.info("Saved annotated video to %s", self.output_video)
        cv2.destroyAllWindows()

        if not frames_out:
            raise RuntimeError("No frames were processed. Check your input video path and codec support.")

        landmarks = torch.stack([frame["norm_kpts"] for frame in frames_out])

        if self.smoothing == "one_euro":
            self.logger.info("Applying One Euro Filter to %s frames ...", len(frames_out))
            landmarks = self._smooth_landmarks(landmarks, fps)

        save_data = {
            "metadata": {
                "video": self.video_path,
                "fps": fps,
                "model": self.model_path,
                "device": self.device,
                "tracking": self.use_tracking,
                "smoothing": self.smoothing,
            },
            "landmarks":     landmarks,
            "raw_landmarks": torch.stack([frame["raw_kpts"] for frame in frames_out]),
            "t_steps":       torch.stack([frame["t"] for frame in frames_out]),
        }

        self._save_posture_data(save_data, len(frames_out))

        return save_data



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