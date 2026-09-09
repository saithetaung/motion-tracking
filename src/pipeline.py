import cv2
from src.camera.webcam_source import WebcamSource
from src.detection_tracking.tracker import PersonTracker
from src.metrics.depth_model import MonocularDepthEstimator
from src.metrics.distance_estimator import DistanceEstimator
from src.metrics.speed_estimator import SpeedEstimator
from src.visualization.overlay import draw_tracks


class TrackingPipeline:
    def __init__(self, camera, tracker, distance_estimator, speed_estimator):
        self.camera = camera
        self.tracker = tracker
        self.distance_estimator = distance_estimator
        self.speed_estimator = speed_estimator

    def run(self):
        while True:
            ok, frame = self.camera.read()
            if not ok:
                break

            self.distance_estimator.set_frame(frame)  # depth map computed once per frame

            tracks = self.tracker.update(frame)
            tracks = [self.distance_estimator.estimate(t) for t in tracks]
            tracks = [self.speed_estimator.update(t) for t in tracks]

            frame = draw_tracks(frame, tracks)
            cv2.imshow("Human Tracking Demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()


def build_pipeline(single_person_mode: bool = True) -> TrackingPipeline:
    camera = WebcamSource()
    focal_length_px = camera.get_focal_length_px()  # still needed by SpeedEstimator below

    tracker = PersonTracker(max_people=1 if single_person_mode else None)
    depth_model = MonocularDepthEstimator(model_size="small")
    distance_estimator = DistanceEstimator(depth_model=depth_model)
    speed_estimator = SpeedEstimator(focal_length_px=focal_length_px)

    return TrackingPipeline(camera, tracker, distance_estimator, speed_estimator)