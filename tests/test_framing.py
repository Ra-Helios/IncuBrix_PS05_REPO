"""Tests for the framing module (9:16 crop calculation and EMA smoothing).

Uses synthetic bounding boxes and does not require a GPU or real video.
"""

import pytest

from src.config import Config
from src.framing import CropCalculator
from src.tracking import TrackedFace


def make_config(**overrides) -> Config:
    """Create a Config with default values plus any overrides."""
    base = Config()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def make_face(cx, cy, w=100, h=150) -> TrackedFace:
    """Create a synthetic tracked face centered at (cx, cy)."""
    bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return TrackedFace(
        face_id="face_1",
        last_bbox=bbox,
        last_center=(cx, cy),
        last_confidence=0.9,
        frames_since_seen=0,
        total_frames_tracked=1,
    )


def test_crop_has_correct_aspect_ratio():
    """Crop must preserve the 9:16 aspect ratio."""
    cfg = make_config()
    calc = CropCalculator(cfg)
    face = make_face(cx=320, cy=300)

    crop = calc.calculate(face, frame_width=640, frame_height=720)

    crop_w = crop.x2 - crop.x1
    crop_h = crop.y2 - crop.y1
    ratio = crop_w / crop_h
    assert abs(ratio - (9 / 16)) < 0.02, f"Unexpected ratio: {ratio}"


def test_crop_stays_916_when_padding_exceeds_frame():
    """A tall face with generous padding must still yield a 9:16 crop.

    In a wide 16:9 analysis frame a padded face crop can exceed the frame
    height; it must be scaled down (not clamped) to keep the ratio, so the
    renderer never stretches content.
    """
    cfg = make_config()
    calc = CropCalculator(cfg)
    # 640x360 analysis frame; face height 300 forces padding > frame height.
    face = make_face(cx=320, cy=200, w=120, h=300)

    crop = calc.calculate(face, frame_width=640, frame_height=360)

    crop_w = crop.x2 - crop.x1
    crop_h = crop.y2 - crop.y1
    assert crop_h <= 360
    ratio = crop_w / crop_h
    assert abs(ratio - (9 / 16)) < 0.02, f"Unexpected ratio: {ratio}"


def test_crop_stays_in_bounds():
    """Crop must always stay within the frame boundary (corners)."""
    cfg = make_config()
    calc = CropCalculator(cfg)

    # Face near top-left corner
    face_tl = make_face(cx=10, cy=10)
    crop = calc.calculate(face_tl, frame_width=640, frame_height=720)
    assert crop.x1 >= 0
    assert crop.y1 >= 0

    # Face near bottom-right corner
    face_br = make_face(cx=630, cy=700)
    crop = calc.calculate(face_br, frame_width=640, frame_height=720)
    assert crop.x2 <= 640
    assert crop.y2 <= 720


def test_crop_never_exceeds_frame_on_loss():
    """Fallback crop must stay within bounds when face is lost."""
    cfg = make_config(max_detection_gap_frames=5)
    calc = CropCalculator(cfg)

    # Feed a face then lose it for several frames
    face = make_face(cx=320, cy=300)
    calc.calculate(face, frame_width=640, frame_height=720)

    last_crop = None
    for _ in range(50):
        crop = calc.calculate(None, frame_width=640, frame_height=720)
        assert crop.x1 >= 0 and crop.y1 >= 0
        assert crop.x2 <= 640 and crop.y2 <= 720
        last_crop = crop

    assert last_crop is not None


def test_ema_smoothing_smoothes_jump():
    """A large face jump should result in a smaller smoothed movement."""
    cfg = make_config(ema_alpha=0.18)
    calc = CropCalculator(cfg)

    # Initial position
    face1 = make_face(cx=320, cy=300)
    calc.calculate(face1, frame_width=640, frame_height=720)

    # Huge jump downward
    face2 = make_face(cx=320, cy=680)
    crop_after_jump = calc.calculate(face2, frame_width=640, frame_height=720)

    # EMA should not allow the crop to move the full raw amount in one frame.
    # The center before the jump (it was the raw first-frame center)
    raw_first = make_face(cx=320, cy=300)
    calc2 = CropCalculator(make_config(ema_alpha=1.0))
    first_crop = calc2.calculate(raw_first, frame_width=640, frame_height=720)
    # With alpha=1.0, first frame and jump are both instant:
    jump = make_face(cx=320, cy=680)
    jump_crop = calc2.calculate(jump, frame_width=640, frame_height=720)
    first_center = (first_crop.y1 + first_crop.y2) / 2
    jump_center = (jump_crop.y1 + jump_crop.y2) / 2
    raw_movement = abs(jump_center - first_center)

    # Now measure the smoothed movement
    smoothed_center = (crop_after_jump.y1 + crop_after_jump.y2) / 2
    smoothed_movement = abs(smoothed_center - first_center)

    assert raw_movement > 0
    assert smoothed_movement < raw_movement, (
        f"Smoothed movement {smoothed_movement} should be less than raw {raw_movement}"
    )


def test_ema_alpha_one_is_instant():
    """Alpha of 1.0 means raw crop is used directly each frame."""
    cfg = make_config(ema_alpha=1.0)
    calc = CropCalculator(cfg)

    face1 = make_face(cx=320, cy=300)
    calc.calculate(face1, frame_width=640, frame_height=720)

    face2 = make_face(cx=320, cy=680)
    crop = calc.calculate(face2, frame_width=640, frame_height=720)

    # With no smoothing, the crop should follow the raw position exactly
    # (center for face at 680, clamped to bottom: crop spans [240, 720])
    center = (crop.y1 + crop.y2) / 2
    assert center == pytest.approx(480, abs=2)
