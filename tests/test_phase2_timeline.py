"""Unit tests for the baseline timeline fields and metadata."""

from src.timeline import FrameRecord, TimelineBuilder


def test_timeline_meta_keeps_both_strategies():
    builder = TimelineBuilder("in.mp4", analysis_fps=15, target_aspect_ratio="9:16")
    builder.add_record(
        FrameRecord(
            frame_idx=0,
            timestamp_ms=0.0,
            active_speaker_id="face_1",
            confidence=0.9,
        )
    )
    meta = builder.build()["meta"]
    assert meta["strategy"] == "face_tracking_ema"
    assert meta["speaker_strategy"] == "active_speaker_fusion_vad"
    assert meta["device"] == "cpu"


def test_phase2_fields_serialized():
    builder = TimelineBuilder("in.mp4")
    rec = FrameRecord(
        frame_idx=5,
        timestamp_ms=333.33,
        active_speaker_id="face_2",
        confidence=0.8,
        audio_speech_probability=0.91,
        mouth_motion_score=0.42,
        active_speaker_score=0.66,
        active_speaker_confidence=0.83,
        speaker_switch_event=True,
        fallback_reason=None,
    )
    builder.add_record(rec)
    frame = builder.build()["frames"][0]
    assert frame["audio_speech_probability"] == pytest_approx(0.91)
    assert frame["mouth_motion_score"] == pytest_approx(0.42)
    assert frame["active_speaker_score"] == pytest_approx(0.66)
    assert frame["active_speaker_confidence"] == pytest_approx(0.83)
    assert frame["speaker_switch_event"] is True
    assert frame["fallback_reason"] is None
    assert frame["active_speaker_id"] == "face_2"


def test_fallback_fields_serialized():
    builder = TimelineBuilder("in.mp4")
    rec = FrameRecord(
        frame_idx=1,
        timestamp_ms=100.0,
        active_speaker_id=None,
        confidence=0.0,
        is_fallback=True,
        fallback_reason="no_speech",
    )
    builder.add_record(rec)
    frame = builder.build()["frames"][0]
    assert frame["is_fallback"] is True
    assert frame["fallback_reason"] == "no_speech"
    assert frame["active_speaker_id"] is None


def pytest_approx(v):
    return round(v, 4)
