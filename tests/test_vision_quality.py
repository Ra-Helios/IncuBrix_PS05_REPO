"""Explainable visual-quality evidence.

Covers the model-free heuristics in ``src/vision.compute_face_quality``
(visibility, size, pose-frontalness, mouth visibility) and the per-track
``track_quality`` maintained by ``src.tracking``.
"""

import os

import pytest

from src.tracking import SimpleTracker
from src.vision import FaceDetection, compute_face_quality, combined_visual_quality

QUALITY_COMPONENTS = (
    "face_visibility",
    "size_quality",
    "head_pose_quality",
    "mouth_visibility",
    "landmark_quality",
    "visual_quality",
)


def make_landmarks(
    n=478,
    clipped=0,
    nose=(0.5, 0.5),
    eye_left=(0.40, 0.42),
    eye_right=(0.60, 0.42),
):
    """Build a normalized landmark list with (default) all points inside."""
    pts = [(0.3, 0.3) for _ in range(n)]
    pts[1] = nose
    pts[33] = eye_left
    pts[263] = eye_right
    for i in range(min(clipped, n)):
        pts[i] = (0.0, 0.3)  # clipped at the image edge
    return pts


def make_detection(x, y, w=60, h=90, confidence=1.0):
    return FaceDetection(
        bbox_x1=x - w / 2,
        bbox_y1=y - h / 2,
        bbox_x2=x + w / 2,
        bbox_y2=y + h / 2,
        center_x=x,
        center_y=y,
        face_width=w,
        face_height=h,
        confidence=confidence,
    )


def test_quality_components_all_inside():
    # face_width=160 on 640 is >= the 20% "full quality" mark -> size 1.0.
    q = compute_face_quality(make_landmarks(), face_width=160, frame_width=640)
    for key in QUALITY_COMPONENTS:
        assert q[key] == pytest.approx(1.0)
    assert q["head_pose_quality"] == pytest.approx(1.0)


def test_face_visibility_half_clipped():
    q = compute_face_quality(
        make_landmarks(clipped=239), face_width=60, frame_width=640
    )
    assert q["face_visibility"] == pytest.approx(0.5, abs=0.05)


def test_size_quality_small_face():
    # 40px face on a 640px frame is well below the 20% "full quality" mark.
    q = compute_face_quality(
        make_landmarks(clipped=0), face_width=40, frame_width=640
    )
    assert q["size_quality"] == pytest.approx(40 / 128, abs=1e-4)
    # Weakest-link semantics: tiny size drives landmark_quality down.
    assert q["landmark_quality"] == pytest.approx(40 / 128, abs=1e-4)


def test_head_pose_turned_face_drops_quality():
    # Frontal face: nose projects onto the eye-line midpoint.
    frontal = make_landmarks()
    assert compute_face_quality(frontal, 60, 640)["head_pose_quality"] > 0.9
    # Turned face: nose slides toward the right eye -> ratio ~1 -> quality 0.
    turned = make_landmarks(nose=(0.63, 0.5))
    turned_q = compute_face_quality(turned, 60, 640)["head_pose_quality"]
    frontal_q = compute_face_quality(frontal, 60, 640)["head_pose_quality"]
    assert turned_q == pytest.approx(0.0)
    assert turned_q < frontal_q - 0.5


def test_mouth_visibility_misses_clipped_lip():
    no_clip = compute_face_quality(make_landmarks(), 60, 640)
    assert no_clip["mouth_visibility"] == pytest.approx(1.0)
    # Clipping the first landmark (a mouth landmark) drops mouth visibility.
    partially = compute_face_quality(make_landmarks(clipped=1), 60, 640)
    assert partially["mouth_visibility"] < 1.0


def test_detection_defaults_full_quality():
    det = make_detection(10, 10)
    for key in QUALITY_COMPONENTS:
        assert getattr(det, key) == 1.0
    assert combined_visual_quality(det) == 1.0


def test_track_quality_rises_with_continuity():
    tracker = SimpleTracker()
    qualities = []
    for _ in range(5):
        tracker.update_all([make_detection(100 + _ * 2, 200, confidence=0.6)], 640)
        qualities.append(tracker.tracks["face_1"].track_quality)
    # Each tracked frame smooths confidence and ramps continuity upward.
    assert qualities[0] == pytest.approx(0.42, abs=1e-4)
    assert all(b > a for a, b in zip(qualities, qualities[1:]))
    assert 0.0 < qualities[-1] < 1.0


def test_track_quality_low_confidence():
    low_tracker = SimpleTracker()
    low_tracker.update_all([make_detection(100, 200, confidence=0.05)], 640)
    low = low_tracker.tracks["face_1"].track_quality
    normal_tracker = SimpleTracker()
    normal_tracker.update_all([make_detection(100, 200, confidence=0.6)], 640)
    normal = normal_tracker.tracks["face_1"].track_quality
    assert low < normal
    assert low < 0.1


@pytest.mark.skipif(
    not os.path.exists("test_assets/single_speaker_speech.mp4"),
    reason="test asset not present",
)
def test_real_frontal_face_quality():
    """Empirical check of the heuristics on a real frontal talking face."""
    import cv2

    from src.vision import FaceDetector

    cap = cv2.VideoCapture("test_assets/single_speaker_speech.mp4")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok, "failed to read first frame of the asset"

    detector = FaceDetector()
    detections = detector.detect(frame)
    detector.close()
    assert detections, "no face detected on the first frame"

    face = detections[0]
    assert face.face_visibility > 0.9, "frontal face should be fully visible"
    assert face.mouth_visibility > 0.8
    assert face.head_pose_quality > 0.3