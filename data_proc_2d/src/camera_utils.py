
from dataclasses import dataclass
import logging
from pathlib import Path
import platform
from typing import Optional, Any

from pygrabber.dshow_graph import FilterGraph

import cv2


logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    # --- HARDWARE & ROI ---
    camera_index: int = 1       # Try 0 or 1
    roi_x_start: int = 150      # Adjust based on your setup
    roi_x_end: int = 950
    roi_y_start: int = 0
    roi_y_end: int = 350
    
    # --- ALGORITHM TUNING (Default Values) ---
    clahe_clip: float = 3.0     # Contrast enhancement (0.1 to 10.0)
    threshold: int = 25         # Sensitivity (0 to 255)
    kernel_width: int = 25      # Line length filter (odd numbers only)
    kernel_height: int = 5
    gap_fill_kernel: int = 35    # NEW: How wide a gap to bridge (in pixels)
    min_area: int = 20        # Keep even SMALL chunks (reflections)
    buffer_ratio: float = 0.5



class CameraProcessing:
    def __init__(self, config: Optional[CameraConfig] = None,
                  config_file: Path = Path(),
                  config_key: str = 'Software.Camera') -> None:
        
        
        self.config_file = config_file
        # Load config from provided object or file, fallback to defaults
        if config is not None:
            self.config: CameraConfig = config
        else:
            self.config = CameraConfig()


    def setup_camera(self) -> cv2.VideoCapture:
        """Initialize and configure the camera. Returns an open VideoCapture."""
        api_preference = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.config.camera_index, )

        # Best-effort settings (not all devices support these)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
            cap.set(cv2.CAP_PROP_FOCUS, 0)
        except Exception:
            logger.debug("Some camera properties could not be set on this device.")

        if not cap.isOpened():
            logger.error("Could not open video device (index=%s)", self.config.camera_index)
            raise RuntimeError("Could not open video device.")

        logger.info("Camera opened (index=%s)", self.config.camera_index)
        return cap
    

    def setup_camera_wifi(self, ip_address: str, port: int, frame_size: tuple[int, int]) -> cv2.VideoCapture:
        """Initialize a VideoCapture for an iPhone camera stream over Wi-Fi."""
        cap = cv2.VideoCapture(f"http://{ip_address}:{port}/video")
        if not cap.isOpened():
            logger.error("Could not open iPhone camera stream at %s:%s", ip_address, port)
            raise RuntimeError("Could not open iPhone camera stream.")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        
        logger.info("iPhone camera stream opened at %s:%s", ip_address, port)
        return cap
    


def iphone_camera_connection(ip_address: str, port: int):
    cap = cv2.VideoCapture(f"http://{ip_address}:{port}/video")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from iPhone camera stream.")
            break
        # Process the frame (e.g., display or save)
        cv2.imshow('iPhone Camera Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def iphone_usb_connection():
    # 0 is usually your laptop's built-in webcam. 
    # 1 or 2 will likely be the DroidCam Virtual Webcam. 
    # Change this number if it opens the wrong camera.
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        cv2.imshow('iPhone USB Camera Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def list_cameras() -> list[dict[str, Any]]:
    """Return available DirectShow camera devices and whether each can produce frames."""
    try:
        devices = FilterGraph().get_input_devices()  # index order for DirectShow
    except Exception as exc:
        logger.exception("Failed to enumerate camera devices: %s", exc)
        return []

    cams = []
    for i, name in enumerate(devices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        try:
            ok, _ = cap.read()
        except Exception:
            ok = False
        finally:
            cap.release()
        cams.append({"index": i, "name": name, "working": ok})
        
    return cams


if __name__ == "__main__":
    # Example usage
    # list_cameras()
    # iphone_usb_connection()
    # iphone_camera_connection(ip_address="192.168.0.101", port=4747)

    # camera_processor = CameraProcessing()
    # cap = camera_processor.setup_camera()

    camera_processor = CameraProcessing()
    cap = camera_processor.setup_camera()
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera.")
            break
    
        cv2.imshow('iPhone USB Camera Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
