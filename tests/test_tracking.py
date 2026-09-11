"""Tests for the simple face tracker.

Uses synthetic detections and does not require a GPU or real video.
"""

from src.tracking import SimpleTracker
from src.vision import FaceDetection


def make_detection(cx, cy, confidence=0.9, w=100, h=150) -> FaceDetection:
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
    )


def test_face_id_persists():
    """A face detected repeatedly should keep the same ID."""
    tracker = SimpleTracker()
    frame_width = 640

    det = make_detection(320, 300)
    first = tracker.update([det], frame_width)
    assert first is not None
    face_id = first.face_id
    assert face_id == "face_1"

    # Slight movement keeps same ID
    det2 = make_detection(330, 305)
    second = tracker.update([det2], frame_width)
    assert second is not None
    assert second.face_id == face_id


def test_temporary_face_loss_recovers():
    """Short gap should keep ID; re-appearing face resumes same ID."""
    tracker = SimpleTracker(max_gap_frames=5)
    frame_width = 640

    det = make_detection(320, 300)
    first = tracker.update([det], frame_width)
    face_id = first.face_id

    # Lost for 3 frames (within gap tolerance)
    for _ in range(3):
        tracker.update([], frame_width)

    # Face reappears
    det2 = make_detection(320, 300)
    recovered = tracker.update([det2], frame_width)
    assert recovered is not None
    assert recovered.face_id == face_id


def test_face_loss_over_threshold_resets():
    """Face lost beyond the threshold should reset the track."""
    tracker = SimpleTracker(max_gap_frames=5)
    frame_width = 640

    det = make_detection(320, 300)
    first = tracker.update([det], frame_width)
    face_id = first.face_id

    # Exceed the gap threshold
    for _ in range(6):
        tracker.update([], frame_width)

    # New detection should create a new track ID
    det2 = make_detection(320, 300)
    result = tracker.update([det2], frame_width)
    assert result is not None
    assert result.face_id != face_id


def test_empty_detections_returns_none_when_no_track():
    """With no track and no detections, update returns None."""
    tracker = SimpleTracker()
    assert tracker.update([], 640) is None
