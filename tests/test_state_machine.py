"""Explicit hysteresis state machine with transition reasons.

Verifies that the active-speaker selector exposes its states and records a
``transition_reason`` every frame, including the SPEAKER_SWITCH_PENDING
confirmation window, hold-grace frames, and fallback paths.
"""

import pytest

from src.active_speaker import (
    REASON_HOLD_GRACE,
    REASON_INITIAL_LOCK,
    REASON_PENDING_SWITCH,
    REASON_STABLE,
    REASON_SWITCH_COMMITTED,
    STATE_FALLBACK,
    STATE_NO_SPEAKER,
    STATE_SPEAKER_LOCKED,
    STATE_SPEAKER_SWITCH_PENDING,
    FACE_TRACK_LOST,
    NO_SPEECH,
    ActiveSpeakerSelector,
)
from src.config import ActiveSpeakerConfig


@pytest.fixture
def selector():
    return ActiveSpeakerSelector(ActiveSpeakerConfig())


def feed(selector, timestamps, speech,
         mouth_a=None, mouth_b=None, mouth_c=None):
    """Feed a sequence of frames; return the list of decisions."""
    decisions = []
    for i, ts in enumerate(timestamps):
        evidence = {}
        if mouth_a is not None:
            evidence["face_1"] = mouth_a[i]
        if mouth_b is not None:
            evidence["face_2"] = mouth_b[i]
        if mouth_c is not None:
            evidence["face_3"] = mouth_c[i]
        decisions.append(selector.process(float(ts), float(speech[i]), evidence))
    return decisions


def test_initial_lock_records_reason_and_state():
    sel = ActiveSpeakerSelector(ActiveSpeakerConfig())
    d = sel.process(0.0, 1.0, {"face_1": 0.9})
    assert d.active_speaker_id == "face_1"
    assert d.state == STATE_SPEAKER_LOCKED
    assert d.transition_reason == REASON_INITIAL_LOCK
    assert not d.is_fallback


def test_stable_reason_on_same_speaker():
    sel = ActiveSpeakerSelector(ActiveSpeakerConfig())
    sel.process(0.0, 1.0, {"face_1": 0.9})
    d = sel.process(100.0, 1.0, {"face_1": 0.9})
    assert d.state == STATE_SPEAKER_LOCKED
    assert d.transition_reason == REASON_STABLE
    assert d.active_speaker_id == "face_1"


def test_pending_state_exposed_before_switch():
    sel = ActiveSpeakerSelector(ActiveSpeakerConfig())
    # Lock face_1 (strong mouth, weak face_2).
    sel.process(0.0, 1.0, {"face_1": 0.9, "face_2": 0.1})
    # Long after the hold period (hold=500ms), face_2 overtakes face_1.
    timestamps = [600.0, 700.0, 800.0]
    mouth_a = [0.1, 0.1, 0.1]
    mouth_b = [0.95, 0.95, 0.95]
    spoken = [1.0, 1.0, 1.0]
    decisions = feed(
        sel, timestamps, spoken, mouth_a=mouth_a, mouth_b=mouth_b
    )
    # First challenge frame opens the confirmation window (pending), the
    # locked speaker still leads the record, then the switch commits after
    # the configured confirm frames (3).
    assert decisions[0].state == STATE_SPEAKER_SWITCH_PENDING
    assert decisions[0].transition_reason == REASON_PENDING_SWITCH
    assert decisions[0].active_speaker_id == "face_1"  # still locked
    assert decisions[1].state == STATE_SPEAKER_SWITCH_PENDING
    assert decisions[2].active_speaker_id == "face_2"
    assert decisions[2].transition_reason == REASON_SWITCH_COMMITTED
    assert decisions[2].speaker_switch_event


def test_hold_grace_records_reason():
    sel = ActiveSpeakerSelector(ActiveSpeakerConfig())
    sel.process(0.0, 1.0, {"face_1": 0.9})
    # Speech/mouth drops below the gate but stays inside the hold window.
    d = sel.process(300.0, 0.0, {"face_1": 0.1})
    assert d.active_speaker_id == "face_1"
    assert d.state == STATE_SPEAKER_LOCKED
    assert d.transition_reason == REASON_HOLD_GRACE
    assert not d.is_fallback


def test_no_speech_fallback_reason():
    sel = ActiveSpeakerSelector(ActiveSpeakerConfig())
    d = sel.process(0.0, 0.0, {"face_1": 0.0})
    assert d.is_fallback
    assert d.state == STATE_NO_SPEAKER
    assert d.transition_reason == NO_SPEECH


def test_face_lost_reason_and_recovery():
    sel = ActiveSpeakerSelector(ActiveSpeakerConfig())
    sel.process(0.0, 1.0, {"face_1": 0.9})
    d = sel.process(100.0, 1.0, {"face_2": 0.1})  # face_1 disappeared
    assert d.is_fallback
    assert d.state == STATE_FALLBACK
    assert d.transition_reason == FACE_TRACK_LOST


def test_state_and_transition_serialized_to_timeline():
    from src.timeline import FrameRecord, TimelineBuilder

    builder = TimelineBuilder(source_file="x.mp4", analysis_fps=15, strategy="B")
    record = FrameRecord(
        frame_idx=0,
        timestamp_ms=0.0,
        active_speaker_id="face_1",
        confidence=0.8,
        state=STATE_SPEAKER_LOCKED,
        transition_reason=REASON_INITIAL_LOCK,
    )
    builder.add_record(record)
    frame = builder.build()["frames"][0]
    assert frame["state"] == STATE_SPEAKER_LOCKED
    assert frame["transition_reason"] == REASON_INITIAL_LOCK