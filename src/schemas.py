from dataclasses import dataclass, field
import time


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    timestamp: float = field(default_factory=time.time)
    distance_m: float | None = None
    speed_kmh: float | None = None
    name: str | None = None                  # set by FaceIdentifier when a face matches
    lateral_mps: float | None = None         # sideways velocity, + = moving right in the image
    radial_mps: float | None = None          # velocity along the camera axis, - = approaching
