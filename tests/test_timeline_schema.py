"""Unit tests for the timeline extensions.

The extended fields are optional and None by default under strategy A so
the decision_timeline.json schema stays backward compatible.
"""

from src.timeline import FrameRecord, TimelineBuilder


def test_meta_comparison_strategy_default_a():
    builder = TimelineBuilder("in.mp4")
    meta = builder.build()["meta"]
    assert meta["comparison_strategy"] == "A"
    assert meta["strategy"] == "face_tracking_ema"
    assert meta["speaker_strategy"] == "active_speaker_fusion_vad"


def test_meta_comparison_strategy_b():
    builder = TimelineBuilder("in.mp4", strategy="B")
    assert builder.build()["meta"]["comparison_strategy"] == "B"


def test_extended_fields_none_by_default():
    builder = TimelineBuilder("in.mp4")
    builder.add_record(
        FrameRecord(
            frame_idx=0,
            timestamp_ms=0.0,
            active_speaker_id="face_1",
            confidence=0.9,
        )
    )
    frame = builder.build()["frames"][0]
    assert frame["state"] is None
    assert frame["track_quality"] is None
    assert frame["face_visibility"] is None
    assert frame["head_pose_quality"] is None
    assert frame["confidence_components"] is None
    assert frame["scene_id"] is None
    assert frame["scene_change"] is None
    assert frame["transition_reason"] is None
    assert frame["crop_jitter"] is None
    assert frame["crop_velocity"] is None


def test_extended_fields_serialized():
    builder = TimelineBuilder("in.mp4", strategy="B")
    rec = FrameRecord(
        frame_idx=10,
        timestamp_ms=666.66,
        active_speaker_id="face_1",
        confidence=0.9,
        state="SPEAKER_LOCKED",
        track_quality=0.87,
        face_visibility=0.95,
        head_pose_quality=0.6,
        confidence_components={"speech_evidence": 0.4, "visual_evidence": 0.6},
        scene_id=2,
        scene_change=False,
        transition_reason="held_under_grace",
        crop_jitter=12.345678,
        crop_velocity=0.1234567,
    )
    builder.add_record(rec)
    frame = builder.build()["frames"][0]
    assert frame["state"] == "SPEAKER_LOCKED"
    assert frame["track_quality"] == pytest_approx(0.87)
    assert frame["face_visibility"] == pytest_approx(0.95)
    assert frame["head_pose_quality"] == pytest_approx(0.6)
    assert frame["confidence_components"] == {
        "speech_evidence": pytest_approx(0.4),
        "visual_evidence": pytest_approx(0.6),
    }
    assert frame["scene_id"] == 2
    assert frame["scene_change"] is False
    assert frame["transition_reason"] == "held_under_grace"
    assert frame["crop_jitter"] == pytest_approx(12.3457)
    assert frame["crop_velocity"] == pytest_approx(0.1235)


def test_backward_compat_phase2_fields_remain():
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
    assert frame["active_speaker_id"] is None
    assert frame["is_fallback"] is True
    assert frame["fallback_reason"] == "no_speech"
    assert frame["crop_coordinates"] is None
    assert frame["face_bbox"] is None


def pytest_approx(v):
    return round(v, 4)