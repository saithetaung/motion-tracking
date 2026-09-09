import time
from ultralytics import YOLO
import numpy as np
from src.schemas import Track

PERSON_CLASS_ID = 0  # COCO "person"


class PersonTracker:
    """
    Wraps YOLO26 + Ultralytics' built-in tracker (ByteTrack by default).
    Handles detection + persistent ID assignment in a single call.
    """

    def __init__(self, model_path: str = "yolo26n.pt", tracker_cfg: str = "bytetrack.yaml",
                 conf_threshold: float = 0.4, max_people: int | None = None):
        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg
        self.conf_threshold = conf_threshold
        self.max_people = max_people  # None = track everyone detected; set 1 for single-person demo mode

    def update(self, frame: np.ndarray) -> list[Track]:
        results = self.model.track(
            frame,
            imgsz=640,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )[0]

        if results.boxes.id is None:
            return []

        tracks = []
        now = time.time()
        for box, track_id in zip(results.boxes, results.boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tracks.append(Track(track_id=int(track_id), bbox=(x1, y1, x2, y2), timestamp=now))

        if self.max_people:
            # keep the earliest-assigned (lowest) track IDs = the person(s) tracked first
            tracks = sorted(tracks, key=lambda t: t.track_id)[: self.max_people]

        return tracks