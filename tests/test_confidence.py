"""Explainable per-frame confidence model.

Unit tests for the documented weighted evidence formula in
``src/confidence`` plus an end-to-end check that a strategy B run records
``confidence_components`` and uses the model value for ``confidence`` while
the timeline remains backward compatible.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ConfidenceConfig, load_config
from src.confidence import build_components, compute_confidence

DEFAULTS = ConfidenceConfig()


def test_default_confidence_weights_loaded():
    config = load_config()
    assert config.confidence.speech_weight == DEFAULTS.speech_weight
    assert config.confidence.mouth_weight == DEFAULTS.mouth_weight
    assert config.confidence.visual_weight == DEFAULTS.visual_weight
    assert config.confidence.track_floor == DEFAULTS.track_floor
    assert config.confidence.track_scale == DEFAULTS.track_scale


def test_no_evidence_is_zero():
    assert compute_confidence(build_components(), DEFAULTS) == 0.0


def test_full_evidence_is_one():
    assert compute_confidence(build_components(1, 1, 1, 1), DEFAULTS) == 1.0


def test_speech_evidence_is_monotonic():
    values = [compute_confidence(build_components(s, 0.5, 0.5, 1.0), DEFAULTS)
              for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_weak_visual_evidence_pulls_confidence_down():
    good = compute_confidence(build_components(0.6, 0.4, 1.0, 1.0), DEFAULTS)
    bad = compute_confidence(build_components(0.6, 0.4, 0.2, 1.0), DEFAULTS)
    assert bad < good


def test_track_continuity_scales_confidence():
    untracked = compute_confidence(build_components(0.9, 0.9, 1.0, 0.0), DEFAULTS)
    tracked = compute_confidence(build_components(0.9, 0.9, 1.0, 1.0), DEFAULTS)
    assert untracked < tracked
    # Default track_floor 0.75 bounds the untracked score (raw = 0.915 here).
    assert untracked == pytest.approx(0.915 * 0.75, abs=1e-4)


def test_components_are_clamped():
    comps = build_components(2.0, -1.0, 0.5, 0.5)
    assert comps["speech"] == 1.0
    assert comps["mouth"] == 0.0
    assert comps["visual"] == comps["track"] == 0.5


def test_documented_formula():
    # raw = 0.6*0.8 + 0.25*0.6 + 0.15*0.4 = 0.69; tracked -> unchanged.
    assert compute_confidence(build_components(0.8, 0.6, 0.4, 1.0), DEFAULTS) == pytest.approx(0.69, abs=1e-4)


def test_timeline_serializes_components():
    from src.timeline import FrameRecord, TimelineBuilder

    builder = TimelineBuilder(source_file="x.mp4", analysis_fps=15, strategy="B")
    record = FrameRecord(
        frame_idx=0,
        timestamp_ms=0.0,
        active_speaker_id="face_1",
        confidence=0.5,
        confidence_components={
            "speech": 0.812345, "mouth": 0.5, "visual": 0.9, "track": 0.75
        },
    )
    builder.add_record(record)
    frame = builder.build()["frames"][0]
    assert frame["confidence_components"]["speech"] == pytest.approx(0.8123, abs=1e-4)
    # original/2 keys remain intact.
    assert "confidence" in frame and "active_speaker_id" in frame


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Small synthetic MP4 without a face (pipeline plumbing check)."""
    import numpy as np
    import cv2

    video_path = os.path.join(str(tmp_path_factory.mktemp("assets")), "synthetic_input.mp4")
    width, height, fps, duration = 640, 360, 30, 1.0
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for i in range(int(fps * duration)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int(width * i / int(fps * duration))
        cv2.rectangle(frame, (x, 100), (x + 80, 220), (0, 150, 255), -1)
        writer.write(frame)
    writer.release()
    return video_path


def test_full_pipeline_records_components_and_strategy_b_uses_model(synthetic_video, tmp_path):
    from test_integration import run_main

    out_dir = os.path.join(str(tmp_path), "out")
    os.makedirs(out_dir, exist_ok=True)
    code, output = run_main(
        "--input", synthetic_video,
        "--output", os.path.join(out_dir, "out.mp4"),
        "--analysis-only",
        "--strategy", "B",
    )
    assert code == 0, output

    timeline_path = os.path.join(out_dir, "decision_timeline.json")
    with open(timeline_path, "r", encoding="utf-8") as fh:
        timeline = json.load(fh)

    frames = timeline["frames"]
    assert frames, "pipeline produced no frames"
    assert all("confidence_components" in f for f in frames)

    config = load_config()
    for frame in frames:
        comps = frame["confidence_components"]
        assert comps["speech"] == pytest.approx(frame["audio_speech_probability"], abs=1e-3)
        expected = compute_confidence(comps, config.confidence)
        assert frame["confidence"] == pytest.approx(expected, abs=2e-3)