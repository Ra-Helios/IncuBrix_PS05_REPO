"""Command-line entry point for Active-Speaker Detection and Smart Video Reframing.

baseline: determine which visible tracked face is speaking by fusing Silero VAD
audio speech activity with MediaPipe mouth movement, then drive the reframing
crop with the selected speaker. CPU-only.

Two selector strategies ("A" baseline vs "B" enhanced), benchmark reporting,
robust tracking, confidence evidence, framing control and scene handling.

Usage examples:
    python main.py --input test_assets/single_speaker.mp4 --output output/reframed.mp4
    python main.py --input test_assets/single_speaker.mp4 --output output/reframed.mp4 --analysis-only
    python main.py --input test_assets/single_speaker.mp4 --output output/reframed.mp4 --benchmark
    python main.py --input test_assets/single_speaker.mp4 --output output/reframed.mp4 --strategy B
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional

from src.config import Config, load_config, apply_strategy_defaults
from src.aspect import parse_ratio, resolve_output_dimensions
from src.ingest import iter_analysis_frames, get_video_info, validate_video
from src.vision import FaceDetector
from src.tracking import SimpleTracker, TrackedFace
from src.framing import CropCalculator, CropRect
from src.timeline import TimelineBuilder, FrameRecord
from src import renderer
from src import benchmark as benchmark_mod
from src import audio as audio_mod
from src import confidence as confidence_mod
from src.scene import SceneChangeDetector
from src.vad import SileroVAD, AudioUnavailableVAD
from src.active_speaker import (
    ActiveSpeakerSelector,
    MouthMotionTracker,
    SpeakerDecision,
    aperture_from_visual,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse.Namespace with input, output, aspect_ratio,
        analysis_only, and config options.
    """
    parser = argparse.ArgumentParser(
        description="Active-Speaker Detection and Smart Video Reframing (baseline)"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input MP4 video"
    )
    parser.add_argument(
        "--output", default="output/reframed.mp4", help="Path for the output MP4"
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default=None,
        help="Target aspect ratio as W:H, e.g. '9:16' (default: from config)",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only generate decision_timeline.json, skip rendering",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["A", "B"],
        default=None,
        help="Selector strategy: 'A' (baseline) or 'B' (enhanced). "
        "Default: from config. Explicit YAML values always win.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also write a machine-readable benchmark_results.json next to "
        "the decision timeline (evaluation metrics)",
    )
    args = parser.parse_args()

    if args.aspect_ratio:
        try:
            parse_ratio(args.aspect_ratio)
        except ValueError:
            parser.error("--aspect-ratio must be in W:H format, e.g. '9:16'")

    return args


def report_cpu() -> None:
    """Print a clear CPU execution report."""
    print("Execution device: CPU")
    logger.info("Configured for CPU-only execution (no CUDA/GPU)")


def _build_vad(config: Config, audio_ctx: audio_mod.AudioContext):
    """Build a speech-evidence provider for the pipeline.

    Uses a live Silero VAD when audio is available, otherwise returns an
    audio-unavailable stand-in so the pipeline runs visual-only.

    Args:
        config: Pipeline configuration.
        audio_ctx: Prepared audio context.

    Returns:
        A speech-evidence provider with ``speech_probability_at`` and
        ``speech_intervals`` methods.
    """
    if not audio_ctx.audio_available or not audio_ctx.wav_path:
        logger.info("Audio unavailable; using visual-only speech evidence")
        return AudioUnavailableVAD()
    try:
        vad = SileroVAD(vad_config=config.vad, onnx_backend=True)
        if audio_ctx.wav_path:
            import numpy as np
            import soundfile as sf

            samples, sr = sf.read(audio_ctx.wav_path, dtype="float32")
            audio_ctx.duration_seconds = audio_ctx.duration_seconds or float(
                len(np.asarray(samples)) / sr
            )
            samples = np.asarray(samples).reshape(-1)
            if samples.ndim > 1:
                samples = samples[:, 0]
            vad_start = time.time()
            probs = vad.analyze(samples)
            vad_elapsed = time.time() - vad_start
            logger.info(
                "VAD analysis time: %.3fs over %.1fs of audio (%d chunks)",
                vad_elapsed, audio_ctx.duration_seconds, len(probs),
            )
            n_intervals = len(vad.speech_intervals())
            logger.info(
                "VAD detected %d speech intervals (threshold=%.2f)",
                n_intervals, config.vad.threshold,
            )
        return vad
    except (RuntimeError, ImportError) as exc:
        logger.warning(
            "Silero VAD unavailable (%s); using visual-only speech evidence",
            exc,
        )
        return AudioUnavailableVAD()


def _per_face_mouth_evidence(
    detections: List,
    tracks: Dict[str, TrackedFace],
    mouth_tracker: MouthMotionTracker,
) -> Dict[str, float]:
    """Compute normalized mouth-motion evidence for every active track.

    Only faces currently detected on the frame contribute evidence, so a
    dropped face cannot keep competing for the speaker selection.

    Args:
        detections: Face detections from the current analysis frame.
        tracks: Track dictionary from the tracker.
        mouth_tracker: Mouth-motion feature tracker.

    Returns:
        Map of ``{face_id: evidence_in_[0,1]}`` for currently visible faces.
    """
    aperture_by_center = {
        (det.center_x, det.center_y): det for det in detections
    }
    evidence: Dict[str, float] = {}
    for face_id, track in tracks.items():
        det = aperture_by_center.get(track.last_center)
        if det is None:
            continue
        aperture = aperture_from_visual(det)
        value = mouth_tracker.update(face_id, aperture)
        evidence[face_id] = value if value is not None else 0.0
    return evidence


def _select_crop_face(
    decision: SpeakerDecision, tracks: Dict[str, TrackedFace]
) -> Optional[TrackedFace]:
    """Choose the tracked face to drive framings.

    Prefer the active speaker when a decision was made; otherwise fall back
    to the most-established visible track so the camera keeps following a
    face even during silence, while the decision still flags fallback.

    Args:
        decision: Active-speaker decision for the frame.
        tracks: Track dictionary from the tracker.

    Returns:
        The TrackedFace to center the crop on, or None.
    """
    if decision.active_speaker_id and decision.active_speaker_id in tracks:
        return tracks[decision.active_speaker_id]
    if tracks:
        return max(tracks.values(), key=lambda t: t.total_frames_tracked)
    return None


def run_pipeline(
    input_path: str,
    output_path: str,
    config: Config,
    analysis_only: bool = False,
    voice_detector: object = None,
    strategy: str = "A",
    benchmark: bool = False,
) -> dict:
    """Run the baseline analysis + (optional) rendering pipeline.

    Injectable entry point: the CLI delegates to it and deterministic
    in-process integration tests inject a scripted ``voice_detector``.

    Args:
        input_path: Path to the input video.
        output_path: Path for the output reframed video.
        config: Pipeline configuration.
        analysis_only: If True, skip rendering.
        voice_detector: Optional speech-evidence provider. Defaults to a
            Silero VAD built from ``config.vad`` (or an audio-unavailable
            stand-in when the input has no audio stream).
        strategy: Selector strategy label ("A" or "B"); recorded in the
            timeline and benchmark metadata.
        benchmark: If True, write a machine-readable benchmark_results.json
            next to the decision timeline.

    Returns:
        Dict describing the run: timeline_path, tracked_frames,
        fallback_count, switch_events, status.
    """
    report_cpu()

    info = get_video_info(input_path)
    logger.info("Input: %s", input_path)
    logger.info(
        "Resolution: %dx%d, FPS: %.2f, frames: %d",
        info.width, info.height, info.fps, info.total_frames,
    )
    logger.info(
        "Analysis resolution: %dpx wide, analysis_fps=%d",
        config.analysis_width, config.analysis_fps,
    )

    detector = FaceDetector(
        confidence_threshold=config.detection_confidence,
        max_faces=config.max_faces,
    )
    tracker = SimpleTracker(
        max_gap_frames=config.max_detection_gap_frames,
        match_distance=config.tracking.match_distance,
        iou_threshold=config.tracking.iou_threshold,
        max_track_gap_ms=config.tracking.max_track_gap_ms,
        analysis_fps=float(config.analysis_fps),
    )
    crop_calculator = CropCalculator(config)
    scene_detector = SceneChangeDetector(threshold=config.scene.change_threshold)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    timeline_path = os.path.join(output_dir, "decision_timeline.json")
    # Record the exact analysis resolution so the benchmark can
    # verify every crop stays inside the analysis frame.
    analysis_height = int(round(info.height * config.analysis_width / info.width))
    timeline = TimelineBuilder(
        input_path,
        analysis_fps=config.analysis_fps,
        target_aspect_ratio=f"{config.target_width}:{config.target_height}",
        strategy=strategy,
        analysis_width=config.analysis_width,
        analysis_height=analysis_height,
    )

    audio_ctx = audio_mod.prepare_audio(
        input_path,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
    )
    vad = voice_detector if voice_detector is not None else _build_vad(config, audio_ctx)

    mouth_tracker = MouthMotionTracker(
        smoothing_window=config.mouth.smoothing_window,
        motion_scale=config.mouth.motion_scale,
    )
    selector = ActiveSpeakerSelector(config.active_speaker)

    crop_list: List[CropRect] = []
    crop_frame_indices: List[int] = []
    fallback_count = 0
    tracked_frame_count = 0
    switch_events = 0
    scene_id = 1

    analysis_start = time.time()
    visual_time = 0.0
    speaker_time = 0.0

    for frame_idx, timestamp, analysis_frame in iter_analysis_frames(
        input_path,
        target_width=config.analysis_width,
        target_fps=config.analysis_fps,
    ):
        height, width = analysis_frame.shape[:2]
        timestamp_ms = timestamp * 1000.0

        # Conservative scene-change detection. On a hard cut the
        # transient pipeline state (tracker, framing EMA, mouth history,
        # speaker state) is reset so the new scene starts clean.
        scene_change = scene_detector.detect(analysis_frame)
        if scene_change:
            scene_id += 1
            tracker.reset()
            crop_calculator.reset()
            mouth_tracker.reset()
            selector.reset()
            logger.info(
                "Scene change at frame %d (%.0f ms): resetting transient state",
                frame_idx,
                timestamp_ms,
            )

        v_start = time.time()
        detections = detector.detect(analysis_frame)
        for det in detections:
            det.timestamp_ms = timestamp_ms
        tracks = tracker.update_all(detections, width, timestamp_ms)
        visual_time += time.time() - v_start

        s_start = time.time()
        mouth_evidence = _per_face_mouth_evidence(
            detections, tracks, mouth_tracker
        )
        speech_probability = vad.speech_probability_at(
            timestamp_ms,
            window_ms=config.active_speaker.audio_visual_window_ms,
        )
        # Strategy B hands the selector per-face track quality
        # so it can gate the speaker role on consistently tracked faces;
        # strategy A keeps the baseline call (no gate) exactly.
        track_qualities = (
            {tid: t.track_quality for tid, t in tracks.items()}
            if strategy == "B"
            else None
        )
        decision = selector.process(
            timestamp_ms,
            speech_probability,
            mouth_evidence,
            track_quality=track_qualities,
        )
        speaker_time += time.time() - s_start

        crop_face = _select_crop_face(decision, tracks)
        crop = crop_calculator.calculate(crop_face, width, height)

        if crop_face is not None:
            tracked_frame_count += 1
        if decision.is_fallback:
            fallback_count += 1
        if decision.speaker_switch_event:
            switch_events += 1

        face_bbox = None
        confidence = 0.0
        face_visibility = None
        head_pose_quality = None
        mouth_visibility = None
        visual_quality = 0.0
        track_quality = 0.0
        if crop_face is not None:
            x1, y1, x2, y2 = crop_face.last_bbox
            face_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            confidence = crop_face.last_confidence
            track_quality = crop_face.track_quality
            # The tracking copy stores the detection's bbox verbatim, so the
            # matching detection carries the visual-quality evidence.
            match = next(
                (d for d in detections
                 if (d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2)
                 == (x1, y1, x2, y2)),
                None,
            )
            if match is not None:
                face_visibility = match.face_visibility
                head_pose_quality = match.head_pose_quality
                mouth_visibility = match.mouth_visibility
                visual_quality = match.visual_quality

        # Record explainable confidence components for every frame
        # and (strategy B only) replace the confidence with the model value.
        # Strategy A keeps the baseline confidence exactly.
        components = confidence_mod.build_components(
            speech=decision.audio_speech_probability,
            mouth=decision.mouth_motion_score,
            visual=visual_quality,
            track=track_quality,
        )
        if strategy == "B":
            confidence = confidence_mod.compute_confidence(
                components, config.confidence
            )

        record = FrameRecord(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            active_speaker_id=decision.active_speaker_id,
            confidence=round(confidence, 4),
            face_bbox=face_bbox,
            crop_coordinates=crop.to_dict(),
            camera_switch_event=decision.speaker_switch_event,
            is_fallback=decision.is_fallback,
            audio_speech_probability=round(decision.audio_speech_probability, 4),
            mouth_motion_score=round(decision.mouth_motion_score, 4),
            active_speaker_score=round(decision.active_speaker_score, 4),
            active_speaker_confidence=round(decision.active_speaker_confidence, 4),
            speaker_switch_event=decision.speaker_switch_event,
            fallback_reason=decision.fallback_reason,
            confidence_components=components,
            state=decision.state,
            transition_reason=decision.transition_reason,
            scene_id=scene_id,
            scene_change=scene_change,
            crop_jitter=crop_calculator.last_jitter,
            crop_velocity=crop_calculator.last_velocity,
            track_quality=track_quality,
            face_visibility=face_visibility,
            head_pose_quality=head_pose_quality,
            mouth_visibility=mouth_visibility,
        )
        timeline.add_record(record)
        crop_list.append(crop)
        crop_frame_indices.append(frame_idx)

    analysis_elapsed = time.time() - analysis_start
    logger.info("Total analysis time: %.2fs", analysis_elapsed)
    logger.info("Visual (detect+track) time: %.2fs", visual_time)
    logger.info("Active-speaker time: %.2fs", speaker_time)
    logger.info(
        "Frames with an active face track: %d of %d",
        tracked_frame_count,
        timeline.record_count,
    )
    logger.info("Fallback events: %d", fallback_count)
    logger.info("Speaker switch events: %d", switch_events)

    timeline.save(timeline_path)

    benchmark_path = None
    benchmark_result = None
    if benchmark:
        source_duration_seconds = info.total_frames / info.fps if info.fps > 0 else None
        benchmark_result = benchmark_mod.evaluate(
            timeline.build(),
            analysis_seconds=analysis_elapsed,
            source_duration_seconds=source_duration_seconds,
        )
        if source_duration_seconds:
            realtime = benchmark_result["timing"].get("realtime_factor")
            logger.info("Realtime factor: %.2fx (analysis vs. source)", realtime)
        benchmark_path = os.path.join(output_dir, "benchmark_results.json")
        benchmark_mod.write(benchmark_result, benchmark_path)
        logger.info("Benchmark results written to %s", benchmark_path)

    result = {
        "timeline_path": timeline_path,
        "tracked_frames": tracked_frame_count,
        "fallback_count": fallback_count,
        "switch_events": switch_events,
        "benchmark_path": benchmark_path,
        "benchmark": benchmark_result,
        "status": "analysis_complete",
    }

    vad.close()

    if analysis_only:
        print(f"\nAnalysis complete. Timeline written to {timeline_path}")
        print(f"Processed {timeline.record_count} analysis points in {analysis_elapsed:.2f}s")
        detector.close()
        return result

    logger.info("Starting rendering with original-resolution source...")
    render_start = time.time()

    try:
        renderer.render_video(
            input_path=input_path,
            output_path=output_path,
            crop_frame_indices=crop_frame_indices,
            crops=crop_list,
            analysis_width=config.analysis_width,
            source_width=info.width,
            source_height=info.height,
            source_fps=info.fps,
            output_width=config.output_width,
            output_height=config.output_height,
        )
    except (RuntimeError, ValueError) as exc:
        detector.close()
        result["status"] = "render_failed"
        raise exc

    render_elapsed = time.time() - render_start
    logger.info("Rendering time: %.2fs", render_elapsed)

    detector.close()

    print(f"\nDone. Output: {output_path}")
    print(f"Timeline: {timeline_path}")
    print(f"Analysis: {timeline.record_count} points in {analysis_elapsed:.2f}s")
    print(f"Render: {render_elapsed:.2f}s")
    print("Execution was CPU-only.")

    result["status"] = "complete"
    return result


def main() -> int:
    """Run the baseline pipeline.

    Returns:
        Exit code (0 on success, non-zero on failure).
    """
    args = parse_args()

    config = load_config(args.config)

    strategy = (args.strategy or config.strategy).upper()
    if strategy not in ("A", "B"):
        logger.error("Invalid strategy: %s (expected A or B)", strategy)
        return 1

    # Strategy B enables the improved defaults on
    # top of the loaded config; A passes the config through untouched so the
    # baseline behavior is reproduced exactly.
    config = apply_strategy_defaults(config, strategy)

    # Resolve the target ratio (CLI or config) and validate
    # it against the supported set; derive matching output resolution so the
    # rendered video has exactly the requested W:H ratio.
    selected_ratio = args.aspect_ratio or (
        f"{config.target_width}:{config.target_height}"
    )
    if selected_ratio not in config.aspect.supported_ratios:
        logger.error(
            "Unsupported aspect ratio %s (supported: %s)",
            selected_ratio,
            ", ".join(config.aspect.supported_ratios),
        )
        return 1

    ratio_width, ratio_height = parse_ratio(selected_ratio)
    config.target_width = ratio_width
    config.target_height = ratio_height
    config.output_width, config.output_height = resolve_output_dimensions(
        config.output_width, ratio_width, ratio_height
    )

    if config.target_width <= 0 or config.target_height <= 0:
        logger.error("Invalid aspect ratio: %d:%d", config.target_width, config.target_height)
        return 1

    try:
        validate_video(args.input)
        run_pipeline(
            input_path=args.input,
            output_path=args.output,
            config=config,
            analysis_only=args.analysis_only,
            strategy=strategy,
            benchmark=args.benchmark,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())