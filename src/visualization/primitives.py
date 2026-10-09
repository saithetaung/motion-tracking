"""Small, stateless OpenCV drawing helpers. No tracking logic in here."""
import math

import cv2
import numpy as np

AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_DUPLEX


def text_size(txt: str, scale: float, thick: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(txt, FONT, scale, thick)
    return w, h


def put_text(img: np.ndarray, txt: str, org: tuple[int, int], scale: float,
             color: tuple, thick: int = 1) -> None:
    cv2.putText(img, txt, org, FONT, scale, color, thick, AA)


def blend_rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
                       color: tuple, alpha: float, radius: int) -> None:
    """Translucent rounded rectangle. Only blends the small ROI, never the full frame."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()

    rx, ry = x - x0, y - y0
    r = max(1, min(radius, h // 2, w // 2))
    cv2.rectangle(layer, (rx + r, ry), (rx + w - r, ry + h), color, -1)
    cv2.rectangle(layer, (rx, ry + r), (rx + w, ry + h - r), color, -1)
    for cx, cy in ((rx + r, ry + r), (rx + w - r, ry + r),
                   (rx + r, ry + h - r), (rx + w - r, ry + h - r)):
        cv2.circle(layer, (cx, cy), r, color, -1, AA)

    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)


def corner_brackets(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                    color: tuple, length: int, thick: int) -> None:
    """Four L-shaped corners instead of a full rectangle: far less visual clutter."""
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * length, py), color, thick, AA)
        cv2.line(img, (px, py), (px, py + dy * length), color, thick, AA)
        cv2.circle(img, (px, py), max(1, thick // 2), color, -1, AA)


def arrow(img: np.ndarray, start: tuple[int, int], end: tuple[int, int],
          color: tuple, thickness: int, head: int, shadow: tuple) -> None:
    """Solid arrow with a dark outline so it stays readable on bright and dark backgrounds."""
    sx, sy = start
    ex, ey = end
    ang = math.atan2(ey - sy, ex - sx)
    left = (int(ex - head * math.cos(ang - 0.5)), int(ey - head * math.sin(ang - 0.5)))
    right = (int(ex - head * math.cos(ang + 0.5)), int(ey - head * math.sin(ang + 0.5)))
    base = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
    tri = np.array([(ex, ey), left, right], dtype=np.int32)

    cv2.line(img, (sx, sy), base, shadow, thickness + 4, AA)          # outline
    cv2.polylines(img, [tri], True, shadow, 4, AA)
    cv2.line(img, (sx, sy), base, color, thickness, AA)               # body
    cv2.fillPoly(img, [tri], color, AA)


def triangle(img: np.ndarray, cx: int, cy: int, size: int, down: bool, color: tuple) -> None:
    d = size if down else -size
    pts = np.array([(cx - size, cy - d // 2), (cx + size, cy - d // 2), (cx, cy + d // 2)],
                   dtype=np.int32)
    cv2.fillPoly(img, [pts], color, AA)
