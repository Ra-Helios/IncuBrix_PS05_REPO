"""9:16 and 1:1 aspect-ratio validation.

Covers the aspect helpers (parse/derive), 1:1 framing (square crops), the
CLI rejection of unsupported ratios, and end-to-end runs proving the
timeline metadata and crop shapes match the requested ratio.
"""

import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.aspect import parse_ratio, resolve_output_dimensions
from src.config import Config
from src.framing import CropCalculator
from src.tracking import TrackedFace


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


def make_config(w, h) -> Config:
    cfg = Config()
    cfg.target_width = w
    cfg.target_height = h
    return cfg


# --------------------------------------------------------------------------
# Aspect helpers
# --------------------------------------------------------------------------


def test_parse_ratio_valid():
    assert parse_ratio("9:16") == (9, 16)
    assert parse_ratio("1:1") == (1, 1)


def test_parse_ratio_rejects_malformed():
    for bad in ("916", "9/16", "9:16:1", "", "a:b"):
        with pytest.raises(ValueError):
            parse_ratio(bad)


def test_parse_ratio_rejects_non_positive():
    for bad in ("0:16", "9:0", "-1:1", "9:-4"):
        with pytest.raises(ValueError):
            parse_ratio(bad)


def test_resolve_output_dimensions_9x16():
    assert resolve_output_dimensions(1080, 9, 16) == (1080, 1920)


def test_resolve_output_dimensions_1x1():
    assert resolve_output_dimensions(1080, 1, 1) == (1080, 1080)


def test_resolve_output_dimensions_4x5():
    assert resolve_output_dimensions(1080, 4, 5) == (1080, 1350)


# --------------------------------------------------------------------------
# 1:1 / 9:16 framing
# --------------------------------------------------------------------------


def test_crop_1to1_is_square():
    calc = CropCalculator(make_config(1, 1))
    crop = calc.calculate(make_face(320, 300), 640, 720)

    w = crop.x2 - crop.x1
    h = crop.y2 - crop.y1
    assert w == h


def test_crop_9x16_keeps_ratio():
    calc = CropCalculator(make_config(9, 16))
    crop = calc.calculate(make_face(320, 300), 640, 720)

    w = crop.x2 - crop.x1
    h = crop.y2 - crop.y1
    assert abs(w / h - 9 / 16) < 0.02


def test_crop_1to1_stays_in_bounds_at_corners():
    calc = CropCalculator(make_config(1, 1))
    for cx, cy in ((10, 10), (630, 700), (10, 700), (630, 10)):
        crop = calc.calculate(make_face(cx, cy), 640, 720)
        assert crop.x1 >= 0 and crop.y1 >= 0
        assert crop.x2 <= 640 and crop.y2 <= 720
        assert abs((crop.x2 - crop.x1) - (crop.y2 - crop.y1)) <= 1


def test_fallback_1to1_is_square():
    calc = CropCalculator(make_config(1, 1))
    calc.calculate(make_face(320, 300), 640, 720)
    crop = calc.calculate(None, 640, 720)  # hold
    calc.reset()  # force fallback path on a fresh scene
    crop = calc.calculate(None, 640, 720)
    w = crop.x2 - crop.x1
    h = crop.y2 - crop.y1
    assert w == h


def test_fallback_9x16_keeps_ratio():
    calc = CropCalculator(make_config(9, 16))
    crop = calc.calculate(None, 640, 720)
    w = crop.x2 - crop.x1
    h = crop.y2 - crop.y1
    assert abs(w / h - 9 / 16) < 0.02


# --------------------------------------------------------------------------
# CLI validation + end-to-end
# --------------------------------------------------------------------------


def test_main_rejects_unsupported_ratio(monkeypatch, caplog):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--input", "missing.mp4", "--analysis-only", "--aspect-ratio", "3:2"],
    )
    import main as main_mod

    with caplog.at_level(logging.ERROR):
        code = main_mod.main()
    assert code == 1
    assert "Unsupported aspect ratio 3:2" in caplog.text


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Small synthetic MP4 without a face (fallback crops)."""
    import numpy as np
    import cv2

    video_path = os.path.join(
        str(tmp_path_factory.mktemp("assets")), "synthetic_input.mp4"
    )
    width, height, fps, duration = 640, 360, 30, 1.0
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for i in range(int(fps * duration)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"f{i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    return video_path


def _run_aspect(synthetic_video, tmp_path, ratio):
    from test_integration import run_main

    out_dir = os.path.join(str(tmp_path), "out_" + ratio.replace(":", "x"))
    code, output = run_main(
        "--input", synthetic_video,
        "--output", os.path.join(out_dir, "out.mp4"),
        "--analysis-only",
        "--aspect-ratio", ratio,
    )
    assert code == 0, output
    timeline_path = os.path.join(out_dir, "decision_timeline.json")
    with open(timeline_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_pipeline_9x16_timeline_metadata_and_crop_shape(synthetic_video, tmp_path):
    timeline = _run_aspect(synthetic_video, tmp_path, "9:16")
    assert timeline["meta"]["target_aspect_ratio"] == "9:16"
    for frame in timeline["frames"]:
        c = frame["crop_coordinates"]
        w = c["x2"] - c["x1"]
        h = c["y2"] - c["y1"]
        assert abs(w / h - 9 / 16) < 0.02


def test_pipeline_1to1_timeline_metadata_and_square_crops(synthetic_video, tmp_path):
    timeline = _run_aspect(synthetic_video, tmp_path, "1:1")
    assert timeline["meta"]["target_aspect_ratio"] == "1:1"
    for frame in timeline["frames"]:
        c = frame["crop_coordinates"]
        assert c["x2"] - c["x1"] == c["y2"] - c["y1"]