import os
import cv2
import yaml
import numpy as np
from .base import CameraSource

DEFAULT_CONFIG_PATH = "configs/camera_webcam.yaml"


class WebcamSource(CameraSource):
    """
    Zero-setup by default: if no calibration file exists yet, approximates
    focal_length_px from frame width (a widely-used rough heuristic for
    webcams with ~60-70 deg horizontal FOV). If scripts/auto_calibrate.py
    has been run, it loads the real calibrated value instead — no code
    change needed, just drop the generated YAML in configs/.
    """

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720,
                 config_path: str = DEFAULT_CONFIG_PATH):
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)  # CAP_DSHOW = faster startup on Windows
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        self.focal_length_px = self._load_or_default(config_path, width)

    @staticmethod
    def _load_or_default(config_path: str, width: int) -> float:
     if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Handles: empty file (cfg is None), missing key, or wrong nesting
        if cfg:
            focal = cfg.get("calibration", {}).get("focal_length_px") or cfg.get("focal_length_px")
            if focal:
                return float(focal)

    # No file, empty file, or malformed — use the rough default
        return float(width)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def get_focal_length_px(self) -> float:
        return self.focal_length_px

    def release(self) -> None:
        self.cap.release()