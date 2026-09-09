import numpy as np
from collections import deque, defaultdict
from src.schemas import Track


class SpeedEstimator:
    """
    Maintains its own per-track_id history, since Track objects are
    recreated fresh every frame by the tracker (history can't live on
    the Track instance itself, or it resets every frame).
    """

    def __init__(self, focal_length_px: float, smoothing_window: int = 5, max_history: int = 30):
        self.focal_length_px = focal_length_px
        self.smoothing_window = smoothing_window
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))

    def update(self, track: Track) -> Track:
        if track.distance_m is None:
            return track

        x1, y1, x2, y2 = track.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        history = self._history[track.track_id]
        history.append((track.timestamp, cx, cy, track.distance_m))

        if len(history) < 2:
            track.speed_kmh = 0.0
            return track

        window = list(history)[-self.smoothing_window:]
        speeds = []
        for (t0, x0, y0, d0), (t1, x1_, y1_, d1) in zip(window, window[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            dx_m = ((x1_ - x0) / self.focal_length_px) * d0
            dy_m = ((y1_ - y0) / self.focal_length_px) * d0
            dz_m = d1 - d0
            dist_m = float(np.sqrt(dx_m**2 + dy_m**2 + dz_m**2))
            speeds.append((dist_m / dt) * 3.6)

        track.speed_kmh = round(float(np.mean(speeds)), 2) if speeds else 0.0
        return track

    def reset_track(self, track_id: int) -> None:
        """Call when a track is lost/removed to avoid unbounded memory growth over a long-running session."""
        self._history.pop(track_id, None)