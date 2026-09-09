from src.schemas import Track
from src.metrics.depth_model import MonocularDepthEstimator


class DistanceEstimator:
    """
    Reads distance from a pre-computed depth map at each track's centroid.
    The depth map itself is computed once per frame (see pipeline.py /
    worker.py) and reused across all tracks in that frame — running the
    depth model once, not once per person, matters for FPS.
    """

    def __init__(self, depth_model: MonocularDepthEstimator):
        self.depth_model = depth_model
        self._current_depth_map = None

    def set_frame(self, frame) -> None:
        self._current_depth_map = self.depth_model.estimate_depth_map(frame)

    def estimate(self, track: Track) -> Track:
        if self._current_depth_map is None:
            return track

        x1, y1, x2, y2 = track.bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        track.distance_m = round(
            self.depth_model.read_depth_at(self._current_depth_map, cx, cy), 2
        )
        return track