"""Machine-readable benchmark reporting for active-speaker reframing.

This module computes a small, deterministic set of
per-run metrics from a decision timeline so runs can be compared
later between strategy A (baseline) and strategy B.

The metrics here are intentionally simple and explainable:
  - how often a speaker was selected (vs. fallback, silence, ...),
  - how stable the speaker selection is (switch count, run lengths),
  - how stable the crop is (center jitter/velocity between samples).
"""

import json
import math
import os
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "benchmark_v1"


def evaluate(
    timeline_data: Dict[str, Any],
    analysis_seconds: Optional[float] = None,
    source_duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute benchmark metrics from a decision_timeline.json dictionary.

    Args:
        timeline_data: A dict as produced by ``TimelineBuilder.build()``
            (with "meta" and "frames" keys).
        analysis_seconds: Optional wall-clock analysis time to report.
        source_duration_seconds: Optional source video duration in seconds;
            enables the realtime factor.

    Returns:
        A machine-readable dict: {"schema_version", "meta", "metrics",
        "timing"}.
    """
    meta = dict(timeline_data.get("meta") or {})
    frames = list(timeline_data.get("frames") or [])

    metrics: Dict[str, Any] = _frame_metrics(
        frames,
        meta.get("analysis_width"),
        meta.get("analysis_height"),
    )

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "source_file": meta.get("source_file"),
            "target_aspect_ratio": meta.get("target_aspect_ratio"),
            "analysis_fps": meta.get("analysis_fps"),
            "strategy": meta.get("comparison_strategy", "A"),
            "device": meta.get("device"),
        },
        "metrics": metrics,
    }

    if analysis_seconds is not None or source_duration_seconds is not None:
        timing: Dict[str, Any] = {}
        if analysis_seconds is not None:
            timing["analysis_seconds"] = round(float(analysis_seconds), 3)
        if source_duration_seconds is not None:
            duration = float(source_duration_seconds)
            timing["source_duration_seconds"] = round(duration, 3)
            if duration > 0 and analysis_seconds is not None:
                timing["realtime_factor"] = round(
                    float(analysis_seconds) / duration, 3
                )
            else:
                timing["realtime_factor"] = None
        result["timing"] = timing

    return result


def write(result: Dict[str, Any], output_path: str) -> str:
    """Write a benchmark result dict to JSON.

    Args:
        result: Dict from ``evaluate()``.
        output_path: Path for the JSON file.

    Returns:
        The output path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return output_path


def load(path: str) -> Dict[str, Any]:
    """Load a benchmark result dict from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frame_metrics(
    frames: List[Dict[str, Any]],
    analysis_width: Optional[int] = None,
    analysis_height: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute aggregate metrics from the frame records.

    Args:
        frames: Frame records from the timeline.
        analysis_width: Analysis frame width when known (for bound checks).
        analysis_height: Analysis frame height when known.

    Returns:
        Metrics dictionary.
    """
    total = len(frames)
    if total == 0:
        return _empty_metrics()

    active = sum(1 for f in frames if f.get("active_speaker_id") is not None)
    falls = sum(1 for f in frames if f.get("is_fallback") is True)
    switches = sum(1 for f in frames if f.get("speaker_switch_event") is True)

    reasons: Dict[str, int] = {}
    for f in frames:
        reason = f.get("fallback_reason")
        if reason is not None:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1

    jitters: List[float] = []
    dt_values: List[float] = []
    prev = None
    prev_idx = None
    for f in frames:
        crop = f.get("crop_coordinates")
        idx = f.get("frame_idx")
        if crop is not None and prev is not None and prev_idx is not None:
            disp = _center_distance(crop, prev)
            jitters.append(disp)
            dt = max(1, int(idx) - int(prev_idx))
            dt_values.append(disp / dt)
        prev = crop
        prev_idx = idx

    runs = _speaker_runs(frames, total)

    # Confidence distribution (per-frame score, strategy
    # dependent) and crop boundary checks against the analysis resolution.
    confidences = [
        float(f["confidence"])
        for f in frames
        if f.get("confidence") is not None
    ]
    bound_violations = sum(
        1
        for f in frames
        if not _crop_in_bounds(
            f.get("crop_coordinates"),
            analysis_width,
            analysis_height,
        )
    )

    confidence: Dict[str, Any] = {}
    if confidences:
        ordered = sorted(confidences)
        confidence = {
            "min": _round3(ordered[0]),
            "mean": _round3(_mean(confidences)),
            "p95": _round3(_p95(confidences)),
            "max": _round3(ordered[-1]),
        }

    return {
        "total_frames": total,
        "active_speaker_frames": active,
        "speaker_proportion": _round3(active / total if total else 0.0),
        "fallback_frames": falls,
        "fallback_proportion": _round3(falls / total if total else 0.0),
        "speaker_switch_events": switches,
        "fallback_reason_counts": dict(
            sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "max_speaker_run_frames": runs["max_run"],
        "mean_speaker_run_frames": _round3(runs["mean_run"]),
        "crop_center_jitter_mean_px": _round3(_mean(jitters)),
        "crop_center_jitter_p95_px": _round3(_p95(jitters)),
        "crop_center_velocity_mean_px_per_frame": _round3(_mean(dt_values)),
        "crop_center_velocity_p95_px_per_frame": _round3(_p95(dt_values)),
        "confidence_distribution": confidence,
        "crop_bound_violations": bound_violations,
        "crop_in_bounds_proportion": _round3(
            (total - bound_violations) / total if total else 0.0
        ),
    }


def _crop_in_bounds(
    crop: Optional[Dict[str, Any]],
    width: Optional[int],
    height: Optional[int],
) -> bool:
    """Whether a crop rectangle is valid and inside the analysis frame.

    Without known analysis dimensions only the degenerate checks apply
    (positive width/height). With dimensions, every corner must fit.

    Args:
        crop: The crop_coordinates dict, or None.
        width: Analysis frame width, or None.
        height: Analysis frame height, or None.

    Returns:
        True when the crop is valid and in bounds.
    """
    if crop is None:
        return False
    try:
        x1, y1, x2, y2 = (float(crop[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if width is not None and height is not None:
        if x1 < 0 or y1 < 0:
            return False
        if x2 > float(width) or y2 > float(height):
            return False
    return True


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_frames": 0,
        "active_speaker_frames": 0,
        "speaker_proportion": 0.0,
        "fallback_frames": 0,
        "fallback_proportion": 0.0,
        "speaker_switch_events": 0,
        "fallback_reason_counts": {},
        "max_speaker_run_frames": 0,
        "mean_speaker_run_frames": 0.0,
        "crop_center_jitter_mean_px": 0.0,
        "crop_center_jitter_p95_px": 0.0,
        "crop_center_velocity_mean_px_per_frame": 0.0,
        "crop_center_velocity_p95_px_per_frame": 0.0,
        "confidence_distribution": {},
        "crop_bound_violations": 0,
        "crop_in_bounds_proportion": 0.0,
    }


def _speaker_runs(frames: List[Dict[str, Any]], total: int) -> Dict[str, float]:
    """Return {max_run, mean_run} of consecutive active-speaker frames."""
    run = 0
    max_run = 0
    runs: List[int] = []
    for f in frames:
        if f.get("active_speaker_id") is not None:
            run += 1
            max_run = max(max_run, run)
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    if not runs:
        return {"max_run": 0, "mean_run": 0.0}
    return {"max_run": max_run, "mean_run": float(sum(runs)) / len(runs)}


def _center_distance(crop: Dict[str, Any], prev: Dict[str, Any]) -> float:
    cx = (float(crop["x1"]) + float(crop["x2"])) / 2.0
    cy = (float(crop["y1"]) + float(crop["y2"])) / 2.0
    px = (float(prev["x1"]) + float(prev["x2"])) / 2.0
    py = (float(prev["y1"]) + float(prev["y2"])) / 2.0
    return float(math.hypot(cx - px, cy - py))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil(0.95 * len(ordered))) - 1
    return float(ordered[max(0, idx)])


def _round3(value: float) -> float:
    return round(float(value), 3)