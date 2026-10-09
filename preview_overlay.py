"""Renders a synthetic 1080p frame so you can restyle the overlay without a camera or models.

Run from the project root:  python preview_overlay.py
"""
import sys
import time

import cv2
import numpy as np

from src.schemas import Track
from src.visualization.overlay import OverlayRenderer


def background(w=1920, h=1080):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (205, 210, 214)                                   # bright wall
    img[int(h * 0.62):] = (120, 130, 140)                      # floor
    cv2.rectangle(img, (1300, 80), (1750, 520), (250, 252, 255), -1)   # very bright window
    cv2.rectangle(img, (100, 120), (520, 330), (60, 70, 80), -1)       # dark poster
    return img


def person(img, box, shade):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cv2.rectangle(img, (x1 + 10, y1 + (y2 - y1) // 5), (x2 - 10, y2), shade, -1)
    cv2.circle(img, (cx, y1 + (y2 - y1) // 9), (x2 - x1) // 4, (150, 170, 200), -1)


# (box, vx_px_per_step, speed, dist, lateral, radial, name)
SPECS = [
    ((700, 300, 1000, 1000), 0, 14.2, 3.2, 1.0, -3.7, "Alex"),        # running, toward camera
    ((1250, 380, 1450, 900), 6, 4.6, 6.8, 1.2, 0.1, None),            # walking, sideways, unnamed
    ((200, 450, 380, 950), 0, 0.0, 5.1, 0.0, 0.0, "Maria"),           # standing still
    ((1550, 20, 1750, 520), -2, 8.3, 9.4, -0.5, 2.1, None),           # jogging away, near top edge
    ((1000, 600, 1150, 1075), 0, 5.0, 2.0, 1.2, -0.5, "Jonathan K."), # long name, near bottom edge
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "overlay_preview.png"
    r = OverlayRenderer()
    base = time.time()
    final = None
    steps = range(-24, 1)
    for step in steps:
        frame = background()
        tracks = []
        for i, (box, vx, sp, d, lat, rad, name) in enumerate(SPECS):
            x1, y1, x2, y2 = box
            b = (x1 + vx * step, y1, x2 + vx * step, y2)
            person(frame, tuple(int(v) for v in b), (50 + 25 * i, 55, 70))
            tracks.append(Track(track_id=i + 1, bbox=tuple(float(v) for v in b), timestamp=base,
                                distance_m=d, speed_kmh=sp, name=name, lateral_mps=lat, radial_mps=rad))
        final = r.draw(frame, tracks, now=base + step * 0.04)
    cv2.imwrite(out, final)
    print("wrote", out, final.shape)
