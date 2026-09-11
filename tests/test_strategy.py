"""Strategy A/B wiring of the improved behaviors.

Strategy A must pass the baseline config through untouched (exact baseline
reproduction). Strategy B enables the enhanced behavior: smarter framing
(deadband + velocity cap) and the selector track-quality gate, where the
speaker role requires a consistently tracked face. Explicit YAML values
always win over the strategy defaults. Naming/imports mirror the confidence
test module.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.active_speaker import ActiveSpeakerSelector
from src.config import ActiveSpeakerConfig, Config, apply_strategy_defaults, load_config


# --------------------------------------------------------------------------
# apply_strategy_defaults
# --------------------------------------------------------------------------


def test_strategy_a_is_phase2_passthrough():
    cfg = apply_strategy_defaults(Config(), "A")
    assert cfg.framing.deadband == pytest.approx(0.0)
    assert cfg.framing.max_crop_velocity_percent is None
    assert cfg.active_speaker.min_track_quality == pytest.approx(0.0)


def test_strategy_b_enables_enhanced_defaults():
    cfg = apply_strategy_defaults(Config(), "B")
    assert cfg.framing.deadband == pytest.approx(0.02)
    assert cfg.framing.max_crop_velocity_percent == pytest.approx(2.0)
    assert cfg.active_speaker.min_track_quality == pytest.approx(0.5)


def test_strategy_b_respects_explicit_config(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "framing:\n"
        "  deadband: 0.05\n"
        "  max_crop_velocity_percent: 5.0\n"
        "active_speaker:\n"
        "  min_track_quality: 0.3\n"
    )
    cfg = apply_strategy_defaults(load_config(str(p)), "B")
    assert cfg.framing.deadband == pytest.approx(0.05)
    assert cfg.framing.max_crop_velocity_percent == pytest.approx(5.0)
    assert cfg.active_speaker.min_track_quality == pytest.approx(0.3)


# --------------------------------------------------------------------------
# Selector track-quality gate
# --------------------------------------------------------------------------


def make_selector(min_track_quality=0.0, switch_min_frames=3) -> ActiveSpeakerSelector:
    cfg = ActiveSpeakerConfig(
        min_track_quality=min_track_quality,
        speaker_switch_min_frames=switch_min_frames,
    )
    return ActiveSpeakerSelector(cfg)


def lock_speaker(selector, face_id="face_1"):
    """Lock face_1 with high quality and speech at t=0."""
    decision = selector.process(
        0.0,
        1.0,
        {face_id: 0.9},
        track_quality={face_id: 0.9},
    )
    assert decision.active_speaker_id == face_id
    return decision


def test_no_gate_when_floor_zero():
    """Track-quality input is ignored when min_track_quality is 0 (baseline)."""
    sel = make_selector(min_track_quality=0.0, switch_min_frames=1)
    lock_speaker(sel)
    decision = sel.process(
        600.0,
        1.0,
        {"face_1": 0.0, "face_2": 1.0},
        track_quality={"face_1": 0.9, "face_2": 0.1},
    )
    assert decision.active_speaker_id == "face_2"
    assert decision.speaker_switch_event


def test_no_gate_without_quality_map():
    """Even with a floor set, no quality map means the gate is off (baseline)."""
    sel = make_selector(min_track_quality=0.5, switch_min_frames=1)
    lock_speaker(sel)
    decision = sel.process(600.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert decision.active_speaker_id == "face_2"
    assert decision.speaker_switch_event


def test_weak_new_face_cannot_take_floor():
    """A weakly-tracked visitor cannot steal the floor from an established
    speaker under strategy B."""
    sel = make_selector(min_track_quality=0.5)
    lock_speaker(sel)
    for ts in (600.0, 700.0, 800.0):
        decision = sel.process(
            ts,
            1.0,
            {"face_1": 0.2, "face_2": 1.0},
            track_quality={"face_1": 0.9, "face_2": 0.2},
        )
        assert decision.active_speaker_id == "face_1"
        assert not decision.speaker_switch_event


def test_qualified_new_face_still_takes_floor():
    """An established, well-tracked visitor can still switch the speaker."""
    sel = make_selector(min_track_quality=0.5)
    lock_speaker(sel)
    decision = None
    for ts in (600.0, 700.0, 800.0):
        decision = sel.process(
            ts,
            1.0,
            {"face_1": 0.2, "face_2": 1.0},
            track_quality={"face_1": 0.9, "face_2": 0.9},
        )
    assert decision is not None
    assert decision.speaker_switch_event
    assert decision.active_speaker_id == "face_2"


def test_weak_only_faces_never_locked():
    """With only weak tracks visible, no speaker is selected (honest fallback)."""
    sel = make_selector(min_track_quality=0.5)
    decision = sel.process(
        0.0,
        1.0,
        {"face_1": 1.0, "face_2": 1.0},
        track_quality={"face_1": 0.2, "face_2": 0.3},
    )
    assert decision.is_fallback
    assert decision.active_speaker_id is None


def test_locked_face_below_floor_steps_down():
    """If the locked speaker's track quality collapses, it must step down
    immediately rather than keep speaking on weak evidence."""
    sel = make_selector(min_track_quality=0.5)
    lock_speaker(sel)
    decision = sel.process(
        600.0,
        1.0,
        {"face_1": 0.5},
        track_quality={"face_1": 0.3},
    )
    assert decision.is_fallback
    assert decision.fallback_reason == "face_track_lost" or decision.fallback_reason == "no_visible_face"
    assert decision.active_speaker_id is None


# --------------------------------------------------------------------------
# Wiring: CLI strategy B records comparison_strategy and improved config
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Small synthetic MP4 without a face (pipeline plumbing check)."""
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


def test_cli_strategy_b_records_comparison_strategy(synthetic_video, tmp_path):
    from test_integration import run_main

    out_dir = os.path.join(str(tmp_path), "out")
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
    assert timeline["meta"]["comparison_strategy"] == "B"
    assert timeline["meta"]["strategy"] == "face_tracking_ema"


def test_strategy_b_applies_framing_through_cli(monkeypatch):
    """--strategy B must enable the framing improvements in the config
    handed to the pipeline (main() applies the defaults before running)."""
    import importlib
    import main as main_mod
    importlib.reload(main_mod)

    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--input", "missing.mp4", "--analysis-only",
         "--strategy", "B", "--aspect-ratio", "1:1"],
    )
    monkeypatch.setattr(main_mod, "validate_video", lambda _path: None)

    # Spy on run_pipeline so the heavy pipeline is never executed.
    calls = {}
    original = main_mod.run_pipeline

    def spy(input_path, output_path, config, analysis_only=False,
            voice_detector=None, strategy="A", benchmark=False):
        calls["config"] = config
        calls["strategy"] = strategy
        return {"status": "analysis_complete"}

    main_mod.run_pipeline = spy
    try:
        code = main_mod.main()
    finally:
        main_mod.run_pipeline = original

    assert code == 0
    assert calls["strategy"] == "B"
    cfg = calls["config"]
    assert cfg.framing.deadband == pytest.approx(0.02)
    assert cfg.framing.max_crop_velocity_percent == pytest.approx(2.0)
    assert cfg.active_speaker.min_track_quality == pytest.approx(0.5)