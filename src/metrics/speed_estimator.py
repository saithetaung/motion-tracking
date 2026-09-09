import numpy as np
from src.schemas import Track


class SpeedEstimator:
    """
    Approximates lateral+depth displacement from bbox center movement (pixels)
    scaled by distance, converted to real-world speed. Smoothed over a window
    to avoid frame-to-frame jitter.
    """

    def __init__(self, focal_length_px: float, smoothing_window: int = 5):
        self.focal_length_px = focal_length_px
        self.smoothing_window = smoothing_window

    def update(self, track: Track) -> Track:
        if track.distance_m is None:
            return track

        x1, y1, x2, y2 = track.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        track.history.append((track.timestamp, cx, cy, track.distance_m))

        if len(track.history) < 2:
            track.speed_kmh = 0.0
            return track

        window = list(track.history)[-self.smoothing_window:]
        speeds = []
        for (t0, x0, y0, d0), (t1, x1_, y1_, d1) in zip(window, window[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            # pixel displacement converted to meters at that depth (pinhole approx)
            dx_m = ((x1_ - x0) / self.focal_length_px) * d0
            dy_m = ((y1_ - y0) / self.focal_length_px) * d0
            dz_m = d1 - d0  # change in distance = movement toward/away from camera
            dist_m = float(np.sqrt(dx_m**2 + dy_m**2 + dz_m**2))
            speeds.append((dist_m / dt) * 3.6)  # m/s -> km/h

        track.speed_kmh = round(float(np.mean(speeds)), 2) if speeds else 0.0
        return track