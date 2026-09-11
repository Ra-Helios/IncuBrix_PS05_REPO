"""Tests for the decision timeline module."""

import json
import os

import pytest

from src.timeline import TimelineBuilder, FrameRecord


def test_timeline_schema(tmp_path):
    """The generated timeline JSON matches the expected schema."""
    builder = TimelineBuilder("some_input.mp4", analysis_fps=15)

    builder.add_record(
        FrameRecord(
            frame_idx=120,
            timestamp_ms=4000.0,
            active_speaker_id="face_1",
            confidence=0.89,
            face_bbox={"x1": 420, "y1": 120, "x2": 960, "y2": 900},
            crop_coordinates={"x1": 420, "y1": 0, "x2": 960, "y2": 960},
            camera_switch_event=False,
            is_fallback=False,
        )
    )

    out_path = os.path.join(str(tmp_path), "decision_timeline.json")
    builder.save(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "meta" in data
    assert data["meta"]["source_file"] == "some_input.mp4"
    assert data["meta"]["target_aspect_ratio"] == "9:16"
    assert data["meta"]["device"] == "cpu"
    assert data["meta"]["strategy"] == "face_tracking_ema"

    assert "frames" in data
    assert len(data["frames"]) == 1

    rec = data["frames"][0]
    assert rec["frame_idx"] == 120
    assert rec["timestamp_ms"] == pytest.approx(4000.0)
    assert rec["active_speaker_id"] == "face_1"
    assert rec["confidence"] == pytest.approx(0.89)
    assert "face_bbox" in rec
    assert "crop_coordinates" in rec
    assert rec["camera_switch_event"] is False
    assert rec["is_fallback"] is False


def test_timeline_records_accumulate():
    """Records accumulate correctly across the analysis pass."""
    builder = TimelineBuilder("input.mp4")
    assert builder.record_count == 0

    builder.add_record(
        FrameRecord(frame_idx=0, timestamp_ms=0.0, active_speaker_id="face_1", confidence=0.5)
    )
    builder.add_record(
        FrameRecord(frame_idx=30, timestamp_ms=1000.0, active_speaker_id="face_1", confidence=0.6)
    )

    assert builder.record_count == 2
    data = builder.build()
    assert len(data["frames"]) == 2


def test_timeline_fallback_flags():
    """Fallback and no-face records are preserved in output."""
    builder = TimelineBuilder("input.mp4")
    builder.add_record(
        FrameRecord(frame_idx=10, timestamp_ms=500.0, active_speaker_id="face_1",
                    confidence=0.0, face_bbox=None,
                    crop_coordinates={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                    is_fallback=True)
    )
    data = builder.build()
    rec = data["frames"][0]
    assert rec["is_fallback"] is True
    assert rec["face_bbox"] is None
    assert rec["crop_coordinates"] is not None
