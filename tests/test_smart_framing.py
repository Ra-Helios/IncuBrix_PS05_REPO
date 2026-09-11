"""Smarter framing (deadband, adaptive smoothing, velocity cap,
and crop-center jitter/velocity reporting).

The default configuration (deadband 0, no velocity cap) must reproduce the
baseline EMA math exactly; the new features only activate when configured.
Assertions use ``last_jitter``/``last_velocity`` (float, pixel-exact) rather
than the integer CropRect corners.
"""

import pytest

from src.config import Config
from src.framing import CropCalculator
from src.tracking import TrackedFace


def make_config(ema_alpha=0.18, deadband=0.0, max_velocity=None) -> Config:
    cfg = Config()
    cfg.ema_alpha = ema_alpha
    cfg.framing.deadband = deadband
    cfg.framing.max_crop_velocity_percent = max_velocity
    return cfg


def make_face(cx, cy, w=100, h=150) -> TrackedFace:
    bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return TrackedFace(
        face_id="face_1",
        last_bbox=bbox,
        last_center=(cx, cy),
        last_confidence=0.9,
        frames_since_seen=0,
        total_frames_tracked=1,
    )


def settle(calc, cx, cy):
    calc.calculate(make_face(cx, cy), 640, 720)
    calc.calculate(make_face(cx, cy), 640, 720)
    calc.calculate(make_face(cx, cy), 640, 720)
    return calc


def center_x(crop):
    return (crop.x1 + crop.x2) / 2


def test_default_matches_phase2_ema():
    """Deadband 0 and no velocity cap reproduce the baseline EMA movement."""
    calc = settle(CropCalculator(make_config(ema_alpha=0.18)), 320, 300)
    before = center_x(calc.calculate(make_face(320, 300), 640, 720))

    crop = calc.calculate(make_face(370, 300), 640, 720)  # 50px raw move
    # 0.18 alpha -> 9px smoothed displacement (float jitter, exact).
    assert calc.last_jitter == pytest.approx(9.0, abs=0.05)
    assert center_x(crop) > before


def test_deadband_suppresses_sub_threshold_motion():
    """Moves smaller than the deadband do not move the crop (no jitter)."""
    calc = settle(CropCalculator(make_config(deadband=0.02)), 320, 300)
    before = center_x(calc.calculate(make_face(320, 300), 640, 720))

    crop = calc.calculate(make_face(325, 300), 640, 720)  # 5px raw < deadband
    assert center_x(crop) == pytest.approx(before, abs=1e-3)
    assert calc.last_jitter == 0.0

    # A big raw move (ema delta 18px > deadband 12.8px) breaks through.
    crop = calc.calculate(make_face(420, 300), 640, 720)  # 100px raw
    assert center_x(crop) > before
    assert calc.last_jitter == pytest.approx(18.0, abs=0.5)


def test_velocity_cap_limits_per_frame_jump():
    """With a velocity cap the crop center moves at most cap/100 * width."""
    calc = settle(CropCalculator(make_config(max_velocity=2.0)), 200, 300)
    before = center_x(calc.calculate(make_face(200, 300), 640, 720))

    crop = calc.calculate(make_face(500, 300), 640, 720)  # huge raw jump
    moved = abs(center_x(crop) - before)
    assert 0 < moved <= 13.0  # integer-crop rounding, cap is 12.8px
    assert calc.last_jitter == pytest.approx(12.8, abs=0.2)


def test_adaptive_alpha_extra_smoothing_on_small_moves():
    """Small moves (speed < 4% width) receive extra smoothing when the
    deadband is enabled."""
    adaptive = settle(CropCalculator(make_config(deadband=0.0005)), 500, 300)
    adaptive.calculate(make_face(505, 300), 640, 720)  # 5px = 0.78% width
    plain = settle(CropCalculator(make_config(deadband=0.0)), 500, 300)
    plain.calculate(make_face(505, 300), 640, 720)

    assert adaptive.last_jitter == pytest.approx(5 * 0.09, abs=0.05)
    assert plain.last_jitter == pytest.approx(5 * 0.18, abs=0.05)
    assert adaptive.last_jitter < plain.last_jitter


def test_jitter_resets_after_reset():
    calc = CropCalculator(make_config())
    calc.calculate(make_face(320, 300), 640, 720)
    calc.calculate(make_face(500, 300), 640, 720)
    assert calc.last_jitter > 0
    assert calc.last_velocity == calc.last_jitter
    calc.reset()
    assert calc.last_jitter == 0.0
    assert calc.last_velocity == 0.0


def test_timeline_serializes_jitter():
    from src.timeline import FrameRecord, TimelineBuilder

    builder = TimelineBuilder(source_file="x.mp4", analysis_fps=15, strategy="B")
    record = FrameRecord(
        frame_idx=1,
        timestamp_ms=100.0,
        active_speaker_id="face_1",
        confidence=0.8,
        crop_jitter=1.234,
        crop_velocity=1.234,
    )
    builder.add_record(record)
    frame = builder.build()["frames"][0]
    assert frame["crop_jitter"] == pytest.approx(1.234, abs=1e-4)
    assert frame["crop_velocity"] == pytest.approx(1.234, abs=1e-4)