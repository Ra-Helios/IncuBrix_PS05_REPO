"""Explainable per-frame confidence model .

The confidence stored for a selected face is a weighted blend of speech,
mouth and visual evidence, scaled by track continuity so that a face in
view for longer (higher ``track_quality``) is trusted more:

    raw        = speech_w * speech + mouth_w * mouth + visual_w * visual
    continuity = track_floor + track_scale * track_quality
    confidence = clamp(raw, 0, 1) * continuity

All weights come from ``ConfidenceConfig``. The formula is deliberately
simple and fully documented in AI_USE.md; no learned model is used. When a
face is not selected, the pipeline records confidence components with
visual/track evidence at 0 so the stored confidence is honestly low.
"""

from typing import Dict

from src.config import ConfidenceConfig


def build_components(
    speech: float = 0.0,
    mouth: float = 0.0,
    visual: float = 0.0,
    track: float = 0.0,
) -> Dict[str, float]:
    """Bundle the four per-frame evidence components.

    Args:
        speech: VAD speech probability at this frame (0-1).
        mouth: Mouth-motion evidence (0-1) from the active-face tracker.
        visual: Combined face usability (0-1) from the weakest-link visual
            quality.
        track: Track continuity (0-1) maintained by the face tracker.

    Returns:
        Dict with the four components clamped to [0, 1].
    """
    return {
        "speech": _clamp01(speech),
        "mouth": _clamp01(mouth),
        "visual": _clamp01(visual),
        "track": _clamp01(track),
    }


def compute_confidence(
    components: Dict[str, float], config: ConfidenceConfig
) -> float:
    """Compute the per-frame confidence from evidence components.

    Args:
        components: Dict with speech/mouth/visual/track keys.
        config: Confidence model weights.

    Returns:
        Confidence in [0, 1].
    """
    speech = _clamp01(float(components.get("speech", 0.0)))
    mouth = _clamp01(float(components.get("mouth", 0.0)))
    visual = _clamp01(float(components.get("visual", 0.0)))
    track = _clamp01(float(components.get("track", 0.0)))

    raw = (
        config.speech_weight * speech
        + config.mouth_weight * mouth
        + config.visual_weight * visual
    )
    continuity = config.track_floor + config.track_scale * track
    return round(max(0.0, min(1.0, raw)) * continuity, 4)


def _clamp01(value: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, float(value)))