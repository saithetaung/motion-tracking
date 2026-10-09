import random

from src.metrics.speed_estimator import SpeedEstimator
from src.schemas import Track

FPS = 30
FOCAL = 1000.0


def run(est, n_frames, bbox_at, dist_at, track_id=1):
    out = None
    for i in range(n_frames):
        t = Track(track_id=track_id, bbox=bbox_at(i), timestamp=i / FPS, distance_m=dist_at(i))
        out = est.update(t)
    return out


def box(cx, cy=500.0, w=100.0, h=300.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_stationary_person_with_pixel_jitter_reads_zero():
    random.seed(1)
    est = SpeedEstimator(focal_length_px=FOCAL)
    t = run(est, 60, lambda i: box(640 + random.uniform(-1, 1)), lambda i: 4.0)
    assert t.speed_kmh < 1.0
    assert abs(t.lateral_mps) < 0.2 and abs(t.radial_mps) < 0.2


def test_sideways_walk_reports_lateral_speed_and_no_radial():
    est = SpeedEstimator(focal_length_px=FOCAL)
    # 2 m/s at 4 m depth -> 2/30 m per frame -> (2/30) * 1000 / 4 px per frame
    px_per_frame = (2.0 / FPS) * FOCAL / 4.0
    t = run(est, 60, lambda i: box(300 + px_per_frame * i), lambda i: 4.0)
    assert abs(t.speed_kmh - 7.2) < 0.6
    assert abs(t.lateral_mps - 2.0) < 0.2
    assert abs(t.radial_mps) < 0.2


def test_approaching_camera_has_negative_radial_speed():
    est = SpeedEstimator(focal_length_px=FOCAL)
    t = run(est, 60, lambda i: box(640), lambda i: 9.0 - (3.0 / FPS) * i)   # 3 m/s toward camera
    assert abs(t.radial_mps - (-3.0)) < 0.3
    assert abs(t.speed_kmh - 10.8) < 1.0


def test_moving_away_has_positive_radial_speed():
    est = SpeedEstimator(focal_length_px=FOCAL)
    t = run(est, 60, lambda i: box(640), lambda i: 3.0 + (2.0 / FPS) * i)
    assert t.radial_mps > 1.5


def test_tracks_do_not_share_state():
    est = SpeedEstimator(focal_length_px=FOCAL)
    px = (2.0 / FPS) * FOCAL / 4.0
    run(est, 60, lambda i: box(300 + px * i), lambda i: 4.0, track_id=1)
    still = run(est, 60, lambda i: box(900), lambda i: 4.0, track_id=2)
    assert still.speed_kmh == 0.0
