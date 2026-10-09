import numpy as np
from collections import deque, defaultdict
from src.schemas import Track


class SpeedEstimator:
    """
    Speed + motion direction from net displacement over a real-time window
    (not adjacent-frame differences), so it is robust to fps changes and to
    per-frame bbox jitter. Outputs, all EMA-smoothed:

      speed_kmh    - overall speed
      lateral_mps  - sideways velocity (+ = moving right in the image)
      radial_mps   - velocity along the camera axis (- = approaching, + = moving away)

    lateral/radial feed the overlay's direction arrow and closer/farther indicator.
    """

    def __init__(self, focal_length_px: float, window_seconds: float = 0.5,
                 min_movement_m: float = 0.03, ema_alpha: float = 0.3):
        self.focal_length_px = focal_length_px
        self.window_seconds = window_seconds
        self.min_movement_m = min_movement_m   # ignore jitter below this: treat as stationary
        self.ema_alpha = ema_alpha
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
        # per track: (speed_kmh, lateral_mps, radial_mps)
        self._smoothed: dict[int, tuple[float, float, float]] = {}

    def _publish(self, track: Track) -> Track:
        speed, lateral, radial = self._smoothed.get(track.track_id, (0.0, 0.0, 0.0))
        track.speed_kmh = round(speed, 2)
        track.lateral_mps = round(lateral, 3)
        track.radial_mps = round(radial, 3)
        return track

    def update(self, track: Track) -> Track:
        if track.distance_m is None:
            return track

        x1, y1, x2, y2 = track.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now = track.timestamp

        history = self._history[track.track_id]
        history.append((now, cx, cy, track.distance_m))

        # keep a true rolling time window, independent of fps
        while history and (now - history[0][0]) > self.window_seconds:
            history.popleft()

        if len(history) < 2:
            return self._publish(track)

        t0, x0, y0, d0 = history[0]
        t1, x1_, y1_, d1 = history[-1]
        dt = t1 - t0
        if dt <= 0:
            return self._publish(track)

        dx_m = ((x1_ - x0) / self.focal_length_px) * d0
        dy_m = ((y1_ - y0) / self.focal_length_px) * d0
        dz_m = d1 - d0
        dist_m = float(np.sqrt(dx_m**2 + dy_m**2 + dz_m**2))

        if dist_m < self.min_movement_m:
            raw = (0.0, 0.0, 0.0)
        else:
            raw = ((dist_m / dt) * 3.6, dx_m / dt, dz_m / dt)

        prev = self._smoothed.get(track.track_id, raw)
        a = self.ema_alpha
        self._smoothed[track.track_id] = tuple(a * r + (1 - a) * p for r, p in zip(raw, prev))
        return self._publish(track)

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._smoothed.pop(track_id, None)
