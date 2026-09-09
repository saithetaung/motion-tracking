import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class MonocularDepthEstimator:
    def __init__(self, model_size: str = "base", device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_id = f"depth-anything/Depth-Anything-V2-Metric-Indoor-{model_size.capitalize()}-hf"
        self.pipe = pipeline(task="depth-estimation", model=model_id, device=device,
                             torch_dtype=torch.float16 if device == "cuda" else torch.float32)

    def estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)          # <-- the fix: numpy array -> PIL Image

        result = self.pipe(pil_image)
        depth = np.array(result["depth"], dtype=np.float32)

        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        return depth

    @staticmethod
    def read_depth_at(depth_map: np.ndarray, x: int, y: int, patch: int = 5) -> float:
        h, w = depth_map.shape
        x0, x1 = max(0, x - patch), min(w, x + patch)
        y0, y1 = max(0, y - patch), min(h, y + patch)
        region = depth_map[y0:y1, x0:x1]
        return float(np.median(region))