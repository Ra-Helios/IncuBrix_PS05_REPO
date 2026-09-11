"""Unit tests for the MouthMotionTracker (mouth-motion evidence)."""

from src.active_speaker import MouthMotionTracker


def test_first_sample_returns_none():
    tracker = MouthMotionTracker(smoothing_window=3, motion_scale=0.10)
    assert tracker.update("face_1", 0.01) is None


def test_flat_aperture_gives_zero_evidence():
    tracker = MouthMotionTracker(smoothing_window=3, motion_scale=0.10)
    tracker.update("face_1", 0.01)
    for _ in range(10):
        ev = tracker.update("face_1", 0.01)
    assert ev == 0.0


def test_moving_aperture_gives_evidence():
    tracker = MouthMotionTracker(smoothing_window=3, motion_scale=0.10)
    tracker.update("face_1", 0.01)
    ev = None
    # Big alternating delta => strong evidence.
    for v in [0.1, 0.01, 0.1, 0.01, 0.1, 0.01]:
        ev = tracker.update("face_1", v)
    assert ev is not None
    assert 0.0 < ev <= 1.0


def test_evidence_saturates_at_one():
    tracker = MouthMotionTracker(smoothing_window=2, motion_scale=0.05)
    tracker.update("face_1", 0.0)
    # delta of 0.2 / 0.05 = 4 => saturate at 1.0
    ev = tracker.update("face_1", 0.2)
    assert ev == 1.0


def test_missing_aperture_keeps_history():
    tracker = MouthMotionTracker(smoothing_window=3, motion_scale=0.10)
    tracker.update("face_1", 0.01)
    ev = tracker.update("face_1", 0.08)
    assert ev is not None
    # Missing sample shouldn't crash and keeps previous history.
    ev2 = tracker.update("face_1", None)
    assert ev2 is not None


def test_missing_aperture_first_sample_returns_none():
    tracker = MouthMotionTracker()
    assert tracker.update("face_1", None) is None


def test_reset_single_face():
    tracker = MouthMotionTracker()
    tracker.update("face_1", 0.01)
    tracker.update("face_1", 0.05)
    tracker.reset("face_1")
    assert tracker.update("face_1", 0.05) is None  # history cleared


def test_reset_all_faces():
    tracker = MouthMotionTracker()
    tracker.update("face_1", 0.01)
    tracker.update("face_2", 0.02)
    tracker.reset()
    assert tracker.update("face_1", 0.03) is None


def test_evidence_rounded():
    tracker = MouthMotionTracker(smoothing_window=2, motion_scale=0.10)
    tracker.update("face_1", 0.0)
    ev = tracker.update("face_1", 0.05)  # 0.5 expected
    assert ev == 0.5
