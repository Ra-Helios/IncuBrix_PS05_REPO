"""Unit tests for the benchmark reporting.

The metrics are computed from the decision timeline and must be
deterministic so that strategy A vs B runs can be compared in the benchmark.
"""

import json

from src import benchmark as bench
from src.timeline import TimelineBuilder, FrameRecord


def _make_frames(n, crops=None, active=None, reasons=None, switches=None):
    frames = []
    for i in range(n):
        crop = {"x1": i * 10, "y1": 0, "x2": i * 10 + 360, "y2": 640}
        if crops is not None:
            crop = crops[i]
        record = {
            "frame_idx": i,
            "timestamp_ms": i * 66.67,
            "active_speaker_id": "face_1" if active is None or active[i] else None,
            "is_fallback": False if active is None or active[i] else True,
            "speaker_switch_event": False,
            "fallback_reason": None,
            "crop_coordinates": crop,
        }
        if reasons is not None:
            record["fallback_reason"] = reasons[i]
        if switches is not None:
            record["speaker_switch_event"] = switches[i]
        frames.append(record)
    return {"meta": {"comparison_strategy": "A"}, "frames": frames}


def test_evaluate_empty_frames():
    result = bench.evaluate({"meta": {"comparison_strategy": "B"}, "frames": []})
    assert result["schema_version"] == bench.SCHEMA_VERSION
    assert result["meta"]["strategy"] == "B"
    assert result["metrics"]["total_frames"] == 0
    assert result["metrics"]["speaker_proportion"] == 0.0
    assert "timing" not in result


def test_evaluate_basic_metrics():
    active = [True, True, False, True, True, True]
    reasons = [None, None, "no_speech", None, None, None]
    switches = [False, False, False, False, False, True]
    result = bench.evaluate(_make_frames(6, active=active, reasons=reasons, switches=switches))
    m = result["metrics"]
    assert m["total_frames"] == 6
    assert m["active_speaker_frames"] == 5
    assert m["speaker_proportion"] == round(5 / 6, 3)
    assert m["fallback_frames"] == 1
    assert m["speaker_switch_events"] == 1
    assert m["fallback_reason_counts"] == {"no_speech": 1}


def test_timing_included_when_provided():
    result = bench.evaluate(_make_frames(2), analysis_seconds=3.14159)
    assert result["timing"]["analysis_seconds"] == round(3.14159, 3)


def test_crop_jitter_and_velocity():
    crops = [
        {"x1": 0, "y1": 0, "x2": 360, "y2": 640},
        {"x1": 10, "y1": 0, "x2": 370, "y2": 640},
        {"x1": 20, "y1": 0, "x2": 380, "y2": 640},
        {"x1": 30, "y1": 0, "x2": 390, "y2": 640},
    ]
    result = bench.evaluate(_make_frames(4, crops=crops))
    m = result["metrics"]
    assert m["crop_center_jitter_mean_px"] == round(10.0, 3)
    assert m["crop_center_velocity_mean_px_per_frame"] == round(10.0, 3)


def test_write_load_roundtrip(tmp_path):
    result = bench.evaluate(_make_frames(3))
    out = tmp_path / "benchmark_results.json"
    path = bench.write(result, str(out))
    assert path == str(out)
    loaded = bench.load(str(out))
    assert loaded == json.loads(json.dumps(result))
    assert loaded["metrics"]["total_frames"] == 3


def test_benchmark_from_real_timeline_builder():
    builder = TimelineBuilder("in.mp4", strategy="B")
    for i in range(5):
        builder.add_record(
            FrameRecord(
                frame_idx=i,
                timestamp_ms=i * 100.0,
                active_speaker_id="face_1" if i > 0 else None,
                confidence=0.9,
                is_fallback=i == 0,
                fallback_reason=None if i > 0 else "no_speech",
                crop_coordinates={"x1": float(i), "y1": 0.0, "x2": float(i + 100), "y2": 200.0},
            )
        )
    result = bench.evaluate(builder.build(), analysis_seconds=2.0)
    assert result["meta"]["strategy"] == "B"
    assert result["metrics"]["active_speaker_frames"] == 4
    assert result["metrics"]["fallback_reason_counts"] == {"no_speech": 1}
    assert result["timing"]["analysis_seconds"] == 2.0