import cv2
import numpy as np
from src.schemas import Track


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID {t.track_id}"
        if t.distance_m is not None:
            label += f" | {t.distance_m}m"
        if t.speed_kmh is not None:
            label += f" | {t.speed_kmh} km/h"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame