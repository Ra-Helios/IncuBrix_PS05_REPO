"""Unit tests for the baseline multi-face tracker (update_all)."""

from src.tracking import SimpleTracker
from src.vision import FaceDetection


def make_detection(cx, cy, confidence=0.9, w=100, h=150) -> FaceDetection:
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
    )


def test_update_all_tracks_two_faces():
    tracker = SimpleTracker()
    tracks = tracker.update_all(
        [make_detection(200, 200), make_detection(500, 200)], 640
    )
    assert set(tracks.keys()) == {"face_1", "face_2"}


def test_face_ids_stable_across_frames():
    tracker = SimpleTracker()
    tracker.update_all([make_detection(200, 200), make_detection(500, 200)], 640)
    tracks = tracker.update_all(
        [make_detection(205, 205), make_detection(495, 198)], 640
    )
    assert set(tracks.keys()) == {"face_1", "face_2"}


def test_dropped_face_expires_other_survives():
    tracker = SimpleTracker(max_gap_frames=5)
    tracker.update_all([make_detection(200, 200), make_detection(500, 200)], 640)
    # Only face_1 kept present; face_2 lost for 6 frames => expires.
    for _ in range(6):
        tracks = tracker.update_all([make_detection(200, 200)], 640)
    assert "face_1" in tracks
    assert "face_2" not in tracks


def test_new_face_gets_new_id_after_expiry():
    tracker = SimpleTracker(max_gap_frames=5)
    tracker.update_all([make_detection(200, 200)], 640)
    for _ in range(6):
        tracker.update_all([], 640)
    tracks = tracker.update_all([make_detection(200, 200)], 640)
    assert list(tracks.keys()) == ["face_2"]


def test_recovery_within_gap_keeps_id():
    tracker = SimpleTracker(max_gap_frames=5)
    tracker.update_all([make_detection(200, 200)], 640)
    for _ in range(3):
        tracker.update_all([], 640)
    tracks = tracker.update_all([make_detection(200, 200)], 640)
    assert "face_1" in tracks
