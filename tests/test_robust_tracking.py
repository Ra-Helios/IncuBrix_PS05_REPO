"""Unit tests: robust/ conservative multi-face tracking.

Covers time-based gaps, IoU gating, velocity prediction, and people
passing/crossing without heavyweight tracking frameworks.
"""

import pytest

from src.tracking import SimpleTracker, _iou
from src.vision import FaceDetection


def make_detection(cx, cy, confidence=0.9, w=100, h=150, ts=None) -> FaceDetection:
    """Create a synthetic face detection centered at (cx, cy)."""
    return FaceDetection(
        bbox_x1=cx - w / 2,
        bbox_y1=cy - h / 2,
        bbox_x2=cx + w / 2,
        bbox_y2=cy + h / 2,
        center_x=cx,
        center_y=cy,
        face_width=w,
        face_height=h,
        confidence=confidence,
        timestamp_ms=ts,
    )


def test_config_parameters_applied():
    tracker = SimpleTracker(
        max_gap_frames=15,
        match_distance=0.5,
        iou_threshold=0.2,
        max_track_gap_ms=1000.0,
        analysis_fps=15.0,
    )
    assert tracker.max_gap_frames == 15
    assert tracker.match_distance == 0.5
    assert tracker.iou_threshold == 0.2


def test_iou_metric():
    a = (0.0, 0.0, 100.0, 100.0)
    b = (50.0, 0.0, 150.0, 100.0)
    assert _iou(a, b) == pytest.approx(1 / 3)
    assert _iou(a, (200.0, 200.0, 300.0, 300.0)) == 0.0


def test_iou_gate_prevents_track_jump():
    """A track whose bbox does not overlap a detection does not take it over."""
    tracker = SimpleTracker(max_gap_frames=5)
    tracker.update_all([make_detection(300, 200)], 640)
    # Next frame: face_1's bbox remains; a far detection has zero overlap.
    tracks = tracker.update_all([make_detection(560, 200)], 640)
    assert "face_1" in tracks  # still alive but lost this frame
    assert tracks["face_1"].frames_since_seen == 1
    assert "face_2" in tracks  # the far detection became a new track
    assert tracks["face_2"].last_center == (560, 200)


def test_time_based_gap_expires():
    tracker = SimpleTracker(max_track_gap_ms=500.0, analysis_fps=15.0)
    tracker.update_all([make_detection(300, 200, ts=0.0)], 640, timestamp_ms=0.0)
    tracks = tracker.update_all([], 640, timestamp_ms=600.0)
    assert tracks == {}
    assert "face_1" not in tracks


def test_time_based_gap_keeps_within_limit():
    tracker = SimpleTracker(max_track_gap_ms=1000.0, analysis_fps=15.0)
    tracker.update_all([make_detection(300, 200, ts=0.0)], 640, timestamp_ms=0.0)
    tracker.update_all([], 640, timestamp_ms=200.0)
    tracks = tracker.update_all([make_detection(300, 200, ts=400.0)], 640, timestamp_ms=400.0)
    assert "face_1" in tracks


def test_velocity_prediction_recovers_moving_face():
    """A moving face that disappears for one frame is re-acquired on the
    predicted position even when it moved beyond its old bbox."""
    tracker = SimpleTracker(iou_threshold=0.3)
    tracker.update_all([make_detection(200, 200, w=60, h=90)], 640)
    tracker.update_all([make_detection(230, 200, w=60, h=90)], 640)
    tracker.update_all([make_detection(260, 200, w=60, h=90)], 640)
    # One empty frame (face occluded/briefly missed).
    tracker.update_all([], 640)
    # It reappears ahead along its trajectory (kept moving while missed).
    tracks = tracker.update_all([make_detection(320, 200, w=60, h=90)], 640)
    assert "face_1" in tracks, "moving face was not re-acquired"
    assert tracks["face_1"].last_center == (320, 200)


def test_crossing_people_keep_ids():
    """Two people passing each other retain distinct persistent IDs."""
    tracker = SimpleTracker(iou_threshold=0.1)
    # face_1 walks right, face_2 walks left, crossing near the center.
    positions_a = [100, 140, 180, 220, 260, 300, 340, 380]
    positions_b = [380, 340, 300, 260, 220, 180, 140, 100]
    first_ids = None
    last_ids = None
    for i in range(len(positions_a)):
        dets = [
            make_detection(positions_a[i], 200, w=60, h=90),
            make_detection(positions_b[i], 210, w=60, h=90),
        ]
        tracks = tracker.update_all(dets, 640)
        ids = (_id_at(tracks, positions_a[i]), _id_at(tracks, positions_b[i]))
        if first_ids is None:
            first_ids = ids
        last_ids = ids
    assert first_ids is not None and last_ids is not None
    assert first_ids == last_ids, "identities swapped while crossing"
    assert first_ids[0] != first_ids[1]


def _id_at(tracks, x):
    for fid, t in tracks.items():
        if abs(t.last_center[0] - x) < 1.0:
            return fid
    return None