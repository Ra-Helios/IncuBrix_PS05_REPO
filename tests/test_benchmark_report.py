"""Benchmark completion.

Adds runtime/realtime-factor timing, the confidence distribution, and crop
bound-violation checks to ``benchmark_results.json``, plus the analysis
resolution in the timeline metadata that makes the bounds checks possible.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import benchmark as bench
from src.timeline import FrameRecord, TimelineBuilder


def _timeline(frames, analysis_width=640, analysis_height=360, strategy="A"):
    return {
        "meta": {
            "comparison_strategy": strategy,
            "analysis_width": analysis_width,
            "analysis_height": analysis_height,
        },
        "frames": frames,
    }


def _frame(idx, confidence=0.9, crop=None, **overrides):
    rec = {
        "frame_idx": idx,
        "timestamp_ms": idx * 66.67,
        "active_speaker_id": "face_1",
        "is_fallback": False,
        "speaker_switch_event": False,
        "fallback_reason": None,
        "crop_coordinates": crop
        or {"x1": 100.0, "y1": 20.0, "x2": 300.0, "y2": 700.0},
        "confidence": confidence,
    }
    rec.update(overrides)
    return rec


# --------------------------------------------------------------------------
# Timing / realtime factor
# --------------------------------------------------------------------------


def test_realtime_factor_computed():
    frames = [_frame(i) for i in range(2)]
    result = bench.evaluate(_timeline(frames), analysis_seconds=30.0, source_duration_seconds=10.0)
    assert result["timing"]["analysis_seconds"] == 30.0
    assert result["timing"]["source_duration_seconds"] == 10.0
    assert result["timing"]["realtime_factor"] == pytest.approx(3.0)


def test_realtime_factor_analysis_faster_than_source():
    frames = [_frame(i) for i in range(2)]
    result = bench.evaluate(_timeline(frames), analysis_seconds=2.0, source_duration_seconds=10.0)
    assert result["timing"]["realtime_factor"] == pytest.approx(0.2)


def test_realtime_factor_none_when_duration_zero():
    frames = [_frame(i) for i in range(2)]
    result = bench.evaluate(_timeline(frames), analysis_seconds=5.0, source_duration_seconds=0.0)
    assert result["timing"]["realtime_factor"] is None


def test_timing_absent_when_not_provided():
    result = bench.evaluate(_timeline([_frame(0)]))
    assert "timing" not in result


# --------------------------------------------------------------------------
# Confidence distribution
# --------------------------------------------------------------------------


def test_confidence_distribution():
    values = [0.2, 0.5, 0.8, 1.0]
    frames = [_frame(i, confidence=v) for i, v in enumerate(values)]
    m = bench.evaluate(_timeline(frames))["metrics"]
    assert m["confidence_distribution"]["min"] == pytest.approx(0.2)
    assert m["confidence_distribution"]["max"] == pytest.approx(1.0)
    assert m["confidence_distribution"]["mean"] == pytest.approx(0.625, abs=1e-3)
    assert m["confidence_distribution"]["p95"] == pytest.approx(1.0, abs=1e-3)


def test_confidence_distribution_empty_when_no_values():
    frames = [_frame(i, confidence=None) for i in range(2)]
    assert bench.evaluate(_timeline(frames))["metrics"]["confidence_distribution"] == {}


# --------------------------------------------------------------------------
# Crop bound violations
# --------------------------------------------------------------------------


def test_out_of_bounds_crop_counted():
    frames = [
        _frame(0, crop={"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 360.0}),
        _frame(1, crop={"x1": 100.0, "y1": 20.0, "x2": 700.0, "y2": 360.0}),
        _frame(2, crop={"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 360.0}),
        _frame(3, crop={"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 360.0}),
    ]
    m = bench.evaluate(_timeline(frames))["metrics"]
    assert m["crop_bound_violations"] == 1
    assert m["crop_in_bounds_proportion"] == pytest.approx(0.75)


def test_degenerate_crop_counted_as_violation():
    frames = [
        _frame(0, crop={"x1": 300.0, "y1": 0.0, "x2": 0.0, "y2": 200.0}),
        _frame(1, crop={"x1": 0.0, "y1": 0.0, "x2": 640.0, "y2": 360.0}),
    ]
    m = bench.evaluate(_timeline(frames))["metrics"]
    assert m["crop_bound_violations"] == 1


def test_no_bounds_check_without_analysis_dims():
    frames = [_frame(0, crop={"x1": 10.0, "y1": 10.0, "x2": 900.0, "y2": 900.0})]
    m = bench.evaluate(_timeline(frames, analysis_width=None, analysis_height=None))["metrics"]
    assert m["crop_bound_violations"] == 0


# --------------------------------------------------------------------------
# Timeline resolution metadata + end-to-end file
# --------------------------------------------------------------------------


def test_timeline_meta_records_analysis_resolution():
    builder = TimelineBuilder("in.mp4", analysis_fps=15, analysis_width=640, analysis_height=360)
    builder.add_record(FrameRecord(frame_idx=0, timestamp_ms=0.0, active_speaker_id=None, confidence=0.0))
    meta = builder.build()["meta"]
    assert meta["analysis_width"] == 640
    assert meta["analysis_height"] == 360


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Small synthetic MP4 without a face."""
    import numpy as np
    import cv2

    video_path = os.path.join(str(tmp_path_factory.mktemp("assets")), "synthetic_input.mp4")
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


def test_cli_benchmark_writes_complete_results(synthetic_video, tmp_path):
    from test_integration import run_main

    out_dir = os.path.join(str(tmp_path), "out")
    code, output = run_main(
        "--input", synthetic_video,
        "--output", os.path.join(out_dir, "out.mp4"),
        "--analysis-only",
        "--benchmark",
        "--strategy", "B",
    )
    assert code == 0, output

    benchmark_path = os.path.join(out_dir, "benchmark_results.json")
    assert os.path.isfile(benchmark_path)
    with open(benchmark_path, "r", encoding="utf-8") as fh:
        result = json.load(fh)

    m = result["metrics"]
    assert "realtime_factor" in result["timing"]
    assert result["timing"]["realtime_factor"] > 0
    assert m["confidence_distribution"] != {}
    assert "crop_bound_violations" in m
    assert "crop_in_bounds_proportion" in m
    # 640x360 analysis frames with fallback crops must have no violations.
    assert m["crop_bound_violations"] == 0
    assert m["crop_in_bounds_proportion"] == pytest.approx(1.0)