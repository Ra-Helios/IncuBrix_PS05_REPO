"""Active-speaker selection: mouth-motion evidence, fusion, and hysteresis.

baseline combines two explainable evidence signals into a per-face
active-speaker score:

    score = audio_weight * speech_evidence + mouth_weight * mouth_evidence

  - speech_evidence comes from Silero VAD (global, same for every face).
  - mouth_evidence is per-face: the |change in mouth aperture| between
    consecutive analysis frames, smoothed and normalized per face track.

A small hysteresis state machine (NO_SPEAKER / SPEAKER_LOCKED /
SPEAKER_SWITCH_PENDING / FALLBACK) plus a minimum hold time, a switch
margin, and confirm frames prevents speaker flicker.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from src.config import ActiveSpeakerConfig

logger = logging.getLogger(__name__)

# Fallback reasons emitted by the selector.
NO_SPEECH = "no_speech"
LOW_CONFIDENCE = "low_active_speaker_confidence"
NO_VISIBLE_FACE = "no_visible_face"
FACE_TRACK_LOST = "face_track_lost"
AUDIO_UNAVAILABLE = "audio_unavailable"

# State names.
STATE_NO_SPEAKER = "NO_SPEAKER"
STATE_SPEAKER_LOCKED = "SPEAKER_LOCKED"
STATE_SPEAKER_SWITCH_PENDING = "SPEAKER_SWITCH_PENDING"
STATE_FALLBACK = "FALLBACK"

# Transition reasons recorded per frame.
REASON_INITIAL_LOCK = "initial_lock"
REASON_STABLE = "stable"
REASON_HOLD_GRACE = "hold_grace"
REASON_PENDING_SWITCH = "pending_switch_confirm"
REASON_SWITCH_COMMITTED = "switch_committed"


@dataclass
class SpeakerDecision:
    """Active-speaker decision for a single analysis frame.

    Attributes:
        timestamp_ms: Video timestamp of the analysis frame.
        active_speaker_id: Selected face ID, or None when no speaker is chosen.
        active_speaker_score: Fused score of the selected speaker.
        audio_speech_probability: VAD speech probability at this timestamp.
        mouth_motion_score: Normalized mouth-motion evidence of the selected
            face (0.0 when no face is selected).
        active_speaker_confidence: Heuristic confidence (see docs; not
            calibrated against ground truth).
        speaker_switch_event: True on the frame a switch is confirmed.
        is_fallback: True when no active speaker was selected.
        fallback_reason: One of the fallback reason strings, or None.
        state: Name of the hysteresis state after processing this frame.
        transition_reason: Why the state/selection is what it is this frame
           ; one of the REASON_* strings or the fallback
            reason.
    """

    timestamp_ms: float
    active_speaker_id: Optional[str] = None
    active_speaker_score: float = 0.0
    audio_speech_probability: float = 0.0
    mouth_motion_score: float = 0.0
    active_speaker_confidence: float = 0.0
    speaker_switch_event: bool = False
    is_fallback: bool = True
    fallback_reason: Optional[str] = None
    state: str = STATE_NO_SPEAKER
    transition_reason: Optional[str] = None


class MouthMotionTracker:
    """Tracks per-face mouth aperture to derive normalized motion evidence.

    mouth_motion[t] = |aperture[t] - aperture[t-1]|, averaged over a short
    smoothing window to keep it stable, then normalized by a scale factor so
    evidence saturates at 1.0 (config.mouth.motion_scale). A face with a
    still mouth produces ~0 evidence, regardless of absolute aperture.
    """

    def __init__(self, smoothing_window: int = 3, motion_scale: float = 0.10) -> None:
        """Initialize the mouth-motion tracker.

        Args:
            smoothing_window: Number of recent |delta| values to average.
            motion_scale: Aperture-delta magnitude that saturates evidence.
        """
        self.smoothing_window = max(2, int(smoothing_window))
        self.motion_scale = max(1e-4, float(motion_scale))
        self._history: Dict[str, deque] = {}

    def update(self, face_id: str, aperture: Optional[float]) -> Optional[float]:
        """Feed one aperture sample for a face and return motion evidence.

        The first sample of a face initializes its history and returns None
        (no motion can be measured without a previous value).

        Args:
            face_id: Tracked face identifier.
            aperture: Current mouth aperture (normalized by mouth width), or
                None if the face is not detected on this frame.

        Returns:
            Normalized smoothed mouth-motion evidence in [0, 1], or None
            when the face has no prior aperture sample yet.
        """
        if face_id not in self._history:
            if aperture is None:
                return None
            self._history[face_id] = deque(
                [aperture], maxlen=self.smoothing_window
            )
            return None

        history = self._history[face_id]
        if aperture is not None:
            history.append(aperture)
        return self._evidence_from_history(history)

    def _evidence_from_history(self, history: deque) -> Optional[float]:
        """Compute normalized motion evidence from a face's history."""
        values = list(history)
        if len(values) < 2:
            return None
        deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        smoothed = float(sum(deltas)) / len(deltas)
        evidence = min(1.0, smoothed / self.motion_scale)
        return round(evidence, 4)

    def reset(self, face_id: Optional[str] = None) -> None:
        """Clear history for one face (or all faces when None)."""
        if face_id is None:
            self._history.clear()
        else:
            self._history.pop(face_id, None)


class ActiveSpeakerSelector:
    """Hysteresis-based active-speaker selector fed with per-frame evidence.

    States:
        NO_SPEAKER             - No candidate reaches minimum confidence yet.
        SPEAKER_LOCKED         - A speaker is locked and actively followed.
        SPEAKER_SWITCH_PENDING - A candidate requests the floor; confirmation
                                 is collected over several frames before the
                                 switch actually commits.
        FALLBACK               - No active speaker; fallback cropping applies.

    The locked speaker is held through short speech gaps (up to
    ``minimum_speaker_hold_ms``) to avoid flicker. A candidate can only take
    the floor after the current speaker has held for at least
    ``minimum_speaker_hold_ms``, when the candidate leads by
    ``speaker_switch_margin``, and after ``speaker_switch_min_frames``
    consecutive confirmations.
    """

    def __init__(self, config: ActiveSpeakerConfig) -> None:
        """Initialize the selector.

        Args:
            config: Active-speaker fusion configuration.
        """
        self.config = config
        self.state = "NO_SPEAKER"
        self._locked_id: Optional[str] = None
        self._locked_since_ms: Optional[float] = None
        self._last_seen_ms: Optional[float] = None
        self._pending_target: Optional[str] = None
        self._pending_count = 0
        self._last_speaker_id: Optional[str] = None
        self._awaiting_recovery = False

    def process(
        self,
        timestamp_ms: float,
        speech_probability: float,
        mouth_evidence: Dict[str, float],
        track_quality: Optional[Dict[str, float]] = None,
    ) -> SpeakerDecision:
        """Select the active speaker for one analysis frame.

        Args:
            timestamp_ms: Frame timestamp in milliseconds.
            speech_probability: VAD speech probability (global evidence).
            mouth_evidence: Normalized mouth-motion evidence per visible face
                ({face_id: 0..1}); only currently visible faces are included.
            track_quality: Optional per-face track quality ({face_id: 0..1}).
                When ``config.min_track_quality`` is > 0 a face below that
                floor is ineligible to take or hold the speaker role. None disables the gate (baseline behavior).

        Returns:
            A SpeakerDecision describing the selection and any events.
        """
        decision = SpeakerDecision(
            timestamp_ms=timestamp_ms,
            audio_speech_probability=round(speech_probability, 4),
        )

        if not mouth_evidence:
            return self._no_visible_face(decision)

        if self._locked_id is not None and self._locked_id not in mouth_evidence:
            return self._locked_face_lost(decision)

        # A locked speaker whose track quality drops below the
        # configured floor must step down immediately (honest weak evidence).
        if (
            self._locked_id is not None
            and self._below_track_floor(self._locked_id, track_quality)
        ):
            return self._locked_face_lost(decision)

        candidate, candidate_score = self._best_candidate(
            mouth_evidence, speech_probability, track_quality
        )

        if candidate is None:
            return self._no_candidate(decision, speech_probability, mouth_evidence)

        decision.mouth_motion_score = round(mouth_evidence.get(candidate, 0.0), 4)
        decision.active_speaker_score = round(candidate_score, 4)

        if self._locked_id is None:
            return self._lock(candidate, candidate_score, decision)

        if self._locked_id == candidate:
            self._last_seen_ms = decision.timestamp_ms
            self._pending_target = None
            self._pending_count = 0
            return self._emit_locked(decision)

        return self._evaluate_switch(
            candidate, candidate_score, mouth_evidence=mouth_evidence,
            speech_probability=speech_probability, decision=decision,
        )

    # ------------------------------------------------------------------
    # state transitions
    # ------------------------------------------------------------------

    def _no_visible_face(self, decision: SpeakerDecision) -> SpeakerDecision:
        """No faces are visible on this frame: fallback."""
        decision.is_fallback = True
        if self._locked_id is not None or self._awaiting_recovery:
            decision.fallback_reason = FACE_TRACK_LOST
            self._awaiting_recovery = True
        else:
            decision.fallback_reason = NO_VISIBLE_FACE
        decision.transition_reason = decision.fallback_reason
        self._drop_pending()
        self._clear_lock()
        self.state = STATE_FALLBACK
        decision.state = self.state
        return decision

    def _locked_face_lost(self, decision: SpeakerDecision) -> SpeakerDecision:
        """The locked speaker is no longer among the visible faces."""
        decision.is_fallback = True
        decision.fallback_reason = FACE_TRACK_LOST
        decision.transition_reason = FACE_TRACK_LOST
        self._awaiting_recovery = True
        self._drop_pending()
        self._clear_lock()
        self.state = STATE_FALLBACK
        decision.state = self.state
        return decision

    def _no_candidate(
        self,
        decision: SpeakerDecision,
        speech_probability: float,
        mouth_evidence: Dict[str, float],
    ) -> SpeakerDecision:
        """No face reaches the confidence gate on this frame."""
        speech_present = speech_probability >= self._speech_presence_threshold()
        reason = LOW_CONFIDENCE if speech_present else NO_SPEECH

        held_under_grace = (
            self._locked_id is not None
            and self._last_seen_ms is not None
            and (decision.timestamp_ms - self._last_seen_ms) <= self.config.minimum_speaker_hold_ms
        )
        if held_under_grace:
            decision.active_speaker_id = self._locked_id
            decision.active_speaker_score = round(
                self._score_for(self._locked_id, mouth_evidence, speech_probability), 4
            )
            decision.mouth_motion_score = round(
                mouth_evidence.get(self._locked_id, 0.0), 4
            )
            decision.active_speaker_confidence = round(
                self._confidence(decision.active_speaker_score), 4
            )
            decision.is_fallback = False
            decision.fallback_reason = None
            decision.state = STATE_SPEAKER_LOCKED
            decision.transition_reason = REASON_HOLD_GRACE
            self.state = STATE_SPEAKER_LOCKED
            return decision

        decision.is_fallback = True
        decision.fallback_reason = reason
        decision.transition_reason = reason
        self._drop_pending()
        self._clear_lock()
        self.state = STATE_FALLBACK if reason == LOW_CONFIDENCE else STATE_NO_SPEAKER
        decision.state = self.state
        return decision

    def _lock(
        self, candidate: str, score: float, decision: SpeakerDecision
    ) -> SpeakerDecision:
        """Lock a speaker when no one is currently locked."""
        self._locked_id = candidate
        self._locked_since_ms = decision.timestamp_ms
        self._last_seen_ms = decision.timestamp_ms
        self._drop_pending()
        self._awaiting_recovery = False
        self.state = STATE_SPEAKER_LOCKED
        decision.state = self.state
        decision.transition_reason = REASON_INITIAL_LOCK
        decision.active_speaker_id = candidate
        decision.active_speaker_confidence = round(self._confidence(score), 4)
        decision.is_fallback = False
        decision.fallback_reason = None
        self._last_speaker_id = candidate
        return decision

    def _emit_locked(self, decision: SpeakerDecision) -> SpeakerDecision:
        """Yield a stable locked-speaker decision (no switch)."""
        assert self._locked_id is not None
        decision.active_speaker_id = self._locked_id
        decision.active_speaker_confidence = round(
            self._confidence(decision.active_speaker_score), 4
        )
        decision.is_fallback = False
        decision.fallback_reason = None
        self.state = STATE_SPEAKER_LOCKED
        decision.state = self.state
        decision.transition_reason = REASON_STABLE
        return decision

    def _evaluate_switch(
        self,
        candidate: str,
        candidate_score: float,
        mouth_evidence: Dict[str, float],
        speech_probability: float,
        decision: SpeakerDecision,
    ) -> SpeakerDecision:
        """Decide whether a visible non-locked candidate takes the floor."""
        locked_score = self._score_for(
            self._locked_id, mouth_evidence, speech_probability
        )
        held_enough = (
            self._locked_since_ms is not None
            and (decision.timestamp_ms - self._locked_since_ms) >= self.config.minimum_speaker_hold_ms
        )

        lead = candidate_score - locked_score
        margin = self.config.speaker_switch_margin

        if self._pending_target == candidate:
            self._pending_count += 1
            if self._pending_count >= self.config.speaker_switch_min_frames:
                return self._switch_to(candidate, candidate_score, decision)
        else:
            self._pending_target = None
            self._pending_count = 0
            if held_enough and lead >= margin:
                self._pending_target = candidate
                self._pending_count = 1
                if self.config.speaker_switch_min_frames <= 1:
                    return self._switch_to(candidate, candidate_score, decision)

        if self._pending_target is not None:
            return self._emit_pending(
                candidate, decision, mouth_evidence, speech_probability
            )
        return self._emit_locked_with_score(
            decision, mouth_evidence, speech_probability
        )

    def _emit_pending(
        self,
        candidate: str,
        decision: SpeakerDecision,
        mouth_evidence: Dict[str, float],
        speech_probability: float,
    ) -> SpeakerDecision:
        """Keep the locked speaker but expose the confirmation window.

        The explicit SPEAKER_SWITCH_PENDING state is recorded
        while a candidate is being confirmed, so the timeline explains why a
        switch is (not yet) happening instead of just showing a locked frame.
        """
        locked_score = self._score_for(
            self._locked_id, mouth_evidence, speech_probability
        )
        decision.active_speaker_id = self._locked_id
        decision.active_speaker_score = round(locked_score, 4)
        decision.mouth_motion_score = round(
            mouth_evidence.get(self._locked_id, 0.0), 4
        )
        decision.active_speaker_confidence = round(
            self._confidence(locked_score), 4
        )
        decision.is_fallback = False
        decision.fallback_reason = None
        self.state = STATE_SPEAKER_SWITCH_PENDING
        decision.state = self.state
        decision.transition_reason = REASON_PENDING_SWITCH
        return decision

    def _switch_to(
        self, target: str, score: float, decision: SpeakerDecision
    ) -> SpeakerDecision:
        """Commit a switch to a new active speaker and record the event."""
        old_id = self._locked_id
        self._locked_id = target
        self._locked_since_ms = decision.timestamp_ms
        self._last_seen_ms = decision.timestamp_ms
        self._pending_target = None
        self._pending_count = 0
        self.state = STATE_SPEAKER_LOCKED
        decision.state = self.state
        decision.transition_reason = REASON_SWITCH_COMMITTED
        decision.active_speaker_id = target
        decision.active_speaker_score = round(score, 4)
        decision.active_speaker_confidence = round(self._confidence(score), 4)
        decision.speaker_switch_event = True
        decision.is_fallback = False
        decision.fallback_reason = None
        self._last_speaker_id = target
        logger.info(
            "Active speaker switch at %.0f ms: %s -> %s (score %.3f)",
            decision.timestamp_ms, old_id, target, score,
        )
        return decision

    def _emit_locked_with_score(
        self,
        decision: SpeakerDecision,
        mouth_evidence: Dict[str, float],
        speech_probability: float,
    ) -> SpeakerDecision:
        """Keep the current speaker while another candidate is weaker."""
        locked_score = self._score_for(
            self._locked_id, mouth_evidence, speech_probability
        )
        decision.active_speaker_score = round(locked_score, 4)
        decision.mouth_motion_score = round(
            mouth_evidence.get(self._locked_id, 0.0), 4
        )
        return self._emit_locked(decision)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the selector to its initial state."""
        self.state = STATE_NO_SPEAKER
        self._clear_lock()
        self._drop_pending()
        self._last_speaker_id = None
        self._awaiting_recovery = False

    def _best_candidate(
        self,
        mouth_evidence: Dict[str, float],
        speech_probability: float,
        track_quality: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Return (best_id, best_score) or (None, 0.0) when below the gate.

        Faces under ``config.min_track_quality`` (when the gate is enabled)
        are ineligible so a transient/weakly-tracked face cannot take the
        speaker role.
        """
        best_id: Optional[str] = None
        best_score = 0.0
        audio_part = self.config.audio_weight * speech_probability
        for face_id, mouth in mouth_evidence.items():
            if self._below_track_floor(face_id, track_quality):
                continue
            score = audio_part + self.config.mouth_weight * mouth
            if score > best_score:
                best_score = score
                best_id = face_id
        if best_id is None or best_score < self.config.minimum_confidence:
            return None, best_score
        return best_id, best_score

    def _below_track_floor(
        self, face_id: str, track_quality: Optional[Dict[str, float]]
    ) -> bool:
        """Whether a face is below the configured track-quality floor.

        The gate is only active when ``min_track_quality`` > 0 and a quality
        map is supplied; otherwise (baseline) every visible face is eligible.

        Args:
            face_id: Candidate/locked face identifier.
            track_quality: Per-face track quality, or None.

        Returns:
            True when the face must be excluded as a speaker.
        """
        floor = float(getattr(self.config, "min_track_quality", 0.0))
        if floor <= 0 or track_quality is None:
            return False
        return track_quality.get(face_id, 0.0) < floor

    def _score_for(
        self,
        face_id: Optional[str],
        mouth_evidence: Dict[str, float],
        speech_probability: float,
    ) -> float:
        if face_id is None:
            return 0.0
        mouth = mouth_evidence.get(face_id, 0.0)
        return self.config.audio_weight * speech_probability + self.config.mouth_weight * mouth

    def _speech_presence_threshold(self) -> float:
        return getattr(self.config, "speech_presence_threshold", 0.5)

    def _confidence(self, score: float) -> float:
        """Heuristic speaker confidence (documented, not calibrated).

        Derived from the raw fusion score: strong scores are trusted more.
        Intentionally simple and explainable.
        """
        return min(1.0, max(0.0, score * 1.25))

    def _clear_lock(self) -> None:
        self._locked_id = None
        self._locked_since_ms = None
        self._last_seen_ms = None

    def _drop_pending(self) -> None:
        self._pending_target = None
        self._pending_count = 0


def aperture_from_visual(face) -> Optional[float]:
    """Return the mouth aperture for a FaceDetection-like object.

    The vision module already computes mouth aperture as the vertical lip
    distance divided by the mouth width; that normalized value is reused.

    Args:
        face: An object with a ``mouth_aperture`` or ``mouth_opening_ratio``.

    Returns:
        Aperture in [0, 1], or None when unavailable.
    """
    value = getattr(face, "mouth_aperture", None)
    if value is None:
        value = getattr(face, "mouth_opening_ratio", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None