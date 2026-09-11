"""Unit tests for the ActiveSpeakerSelector (fusion + hysteresis + fallback).

These tests drive the selector with synthetic per-frame evidence and verify
the score fusion formula, the confidence gate, speaker locking, hysteresis
hold times, switch margin/margin confirmations, and every fallback reason.
"""

import pytest

from src.active_speaker import (
    FACE_TRACK_LOST,
    LOW_CONFIDENCE,
    NO_SPEECH,
    NO_VISIBLE_FACE,
    ActiveSpeakerSelector,
    aperture_from_visual,
)
from src.config import ActiveSpeakerConfig


def make_selector(**overrides):
    return ActiveSpeakerSelector(ActiveSpeakerConfig(**overrides))


def test_score_fusion_formula():
    """score = audio_weight*speech + mouth_weight*mouth."""
    sel = make_selector(audio_weight=0.4, mouth_weight=0.6)
    assert sel._score_for("face_1", {"face_1": 0.5}, 0.8) == 0.4 * 0.8 + 0.6 * 0.5


def test_minimum_confidence_gate_speech_alone_insufficient():
    """Static face + strong speech alone never reaches candidate (spec)."""
    sel = make_selector(
        audio_weight=0.4, mouth_weight=0.6, minimum_confidence=0.50
    )
    for _ in range(5):
        dec = sel.process(100.0, 0.9, {"face_1": 0.0})
    assert dec.active_speaker_id is None
    assert dec.is_fallback is True
    assert dec.fallback_reason == LOW_CONFIDENCE


def test_locks_speaker_with_mouth_and_speech():
    sel = make_selector(minimum_confidence=0.50)
    dec = sel.process(0.0, 1.0, {"face_1": 1.0})
    assert dec.active_speaker_id == "face_1"
    assert dec.is_fallback is False
    assert dec.state == "SPEAKER_LOCKED"
    # 0.4*1.0 + 0.6*1.0 = 1.0
    assert dec.active_speaker_score == 1.0


def test_no_speech_fallback():
    sel = make_selector()
    dec = sel.process(0.0, 0.0, {"face_1": 0.05})
    assert dec.is_fallback is True
    # Below speech_presence_threshold => no_speech.
    assert dec.fallback_reason == NO_SPEECH
    assert dec.state == "NO_SPEAKER"


def test_no_visible_face_fallback_before_lock():
    sel = make_selector()
    dec = sel.process(0.0, 1.0, {})
    assert dec.is_fallback is True
    assert dec.fallback_reason == NO_VISIBLE_FACE
    assert dec.state == "FALLBACK"


def test_face_track_lost_after_lock():
    sel = make_selector()
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock face_1
    dec = sel.process(100.0, 1.0, {})  # no faces visible
    assert dec.fallback_reason == FACE_TRACK_LOST


def test_face_track_lost_recovery_persists():
    """Once lost, recovery keeps reporting FACE_TRACK_LOST through the gap."""
    sel = make_selector()
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock
    sel.process(100.0, 1.0, {})  # lost
    # Still away => face_track_lost (not no_visible_face).
    dec = sel.process(200.0, 1.0, {})
    assert dec.fallback_reason == FACE_TRACK_LOST


def test_hold_through_brief_no_candidate():
    """Speaker held through a short gap <= minimum_speaker_hold_ms."""
    sel = make_selector(minimum_speaker_hold_ms=500.0)
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock
    # Brief no-candidate dip within hold window keeps speaker locked.
    dec = sel.process(200.0, 0.0, {"face_1": 0.0})
    assert dec.active_speaker_id == "face_1"
    assert dec.is_fallback is False
    # But after the hold window passes without recovery, we fall back.
    dec2 = sel.process(700.0, 0.0, {"face_1": 0.0})
    assert dec2.is_fallback is True


def test_switch_requires_margin_and_hold():
    """A candidate must lead by margin and the current speaker must hold."""
    sel = make_selector(
        minimum_speaker_hold_ms=100.0,
        speaker_switch_margin=0.10,
        speaker_switch_min_frames=1,
    )
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock face_1
    # Candidate face_2 leads strongly, holds enough, margin met => switch.
    dec = sel.process(200.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_2"
    assert dec.speaker_switch_event is True


def test_switch_blocked_before_hold():
    sel = make_selector(
        minimum_speaker_hold_ms=1000.0,
        speaker_switch_margin=0.10,
        speaker_switch_min_frames=1,
    )
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock at t=0
    # Not held long enough yet => stays with face_1.
    dec = sel.process(100.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_1"
    assert dec.speaker_switch_event is False


def test_switch_blocked_below_margin():
    sel = make_selector(
        minimum_speaker_hold_ms=100.0,
        speaker_switch_margin=0.30,
        speaker_switch_min_frames=1,
    )
    sel.process(0.0, 1.0, {"face_1": 1.0})
    # face_2 leads by only 0.2 (below 0.30 margin) => keep face_1.
    dec = sel.process(200.0, 1.0, {"face_1": 0.0, "face_2": 0.2})
    assert dec.active_speaker_id == "face_1"


def test_switch_requires_confirmation_frames():
    sel = make_selector(
        minimum_speaker_hold_ms=100.0,
        speaker_switch_margin=0.10,
        speaker_switch_min_frames=3,
    )
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock face_1

    # Frame 1: face_2 leads but only 1 confirmation collected => no switch.
    dec = sel.process(200.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_1"
    assert dec.speaker_switch_event is False
    # Frame 2: still leading, 2 confirmations => no switch yet.
    dec = sel.process(300.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_1"
    # Frame 3: 3 confirmations => switch commits.
    dec = sel.process(400.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_2"
    assert dec.speaker_switch_event is True


def test_pending_reset_when_candidate_loses_lead():
    """A pending switch is dropped if the candidate stops leading."""
    sel = make_selector(
        minimum_speaker_hold_ms=100.0,
        speaker_switch_margin=0.10,
        speaker_switch_min_frames=3,
    )
    sel.process(0.0, 1.0, {"face_1": 1.0})  # lock face_1
    sel.process(200.0, 1.0, {"face_1": 0.0, "face_2": 1.0})  # pending 1
    # face_2 no longer leads; pending resets.
    dec = sel.process(300.0, 1.0, {"face_1": 1.0, "face_2": 0.0})
    assert dec.active_speaker_id == "face_1"
    # face_2 leads again but pending restarted from 1 => not enough yet.
    dec = sel.process(400.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_1"
    # Two more confirmations now complete the needed 3.
    dec = sel.process(500.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_1"
    dec = sel.process(600.0, 1.0, {"face_1": 0.0, "face_2": 1.0})
    assert dec.active_speaker_id == "face_2"
    assert dec.speaker_switch_event is True


def test_reset_clears_lock():
    sel = make_selector()
    sel.process(0.0, 1.0, {"face_1": 1.0})
    assert sel._locked_id == "face_1"
    sel.reset()
    assert sel._locked_id is None
    assert sel.state == "NO_SPEAKER"


def test_confidence_heuristic():
    sel = make_selector()
    assert sel._confidence(0.5) == pytest.approx(0.625)
    assert sel._confidence(2.0) == 1.0
    assert sel._confidence(-0.5) == 0.0


def test_aperture_from_visual_priority():
    class Fake:
        mouth_aperture = 0.3
        mouth_opening_ratio = 0.9

    assert aperture_from_visual(Fake()) == 0.3


def test_aperture_from_visual_fallback_alias():
    class Fake:
        mouth_opening_ratio = 0.9

    assert aperture_from_visual(Fake()) == 0.9


def test_aperture_from_visual_none():
    assert aperture_from_visual(object()) is None
