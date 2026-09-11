"""Video renderer that applies crop coordinates to the original video.

Renders the final reframed video using the ORIGINAL resolution source,
preserving audio when the input has an audio stream. Uses FFmpeg for the
final muxing/audio handling to avoid unnecessary audio re-encoding. Videos
without an audio track are rendered as video-only without failing.
"""

import logging
import os
import subprocess
from typing import List, Optional

import cv2

from src.framing import CropRect

logger = logging.getLogger(__name__)


def check_ffmpeg() -> str:
    """Check that FFmpeg is available on the system.

    Returns:
        The FFmpeg executable path if found.

    Raises:
        RuntimeError: If FFmpeg is not available.
    """
    for cmd in ("ffmpeg", "ffmpeg.exe"):
        try:
            result = subprocess.run(
                [cmd, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.SubprocessError):
            continue

    raise RuntimeError(
        "FFmpeg is required for rendering but was not found on PATH. "
        "Install FFmpeg and add it to your PATH."
    )


def render_video(
    input_path: str,
    output_path: str,
    crop_frame_indices: List[int],
    crops: List[CropRect],
    analysis_width: int,
    source_width: int,
    source_height: int,
    source_fps: float,
    output_width: int,
    output_height: int,
) -> None:
    """Render the final reframed video from the original-resolution source.

    Strategy:
      1. Write a temporary, video-only MP4 by cropping each ORIGINAL
         frame to the crop rectangle corresponding to its analysis point
         and resizing to the target output resolution (OpenCV).
      2. Probe the input for an audio stream.
      3. If audio is present, extract it and mux it into the final video
         by copying the audio stream to avoid re-encoding. If no audio
         stream exists, the cropped video alone becomes the output and a
         warning is logged (this is not a failure).

    Each crop applies to the source frames between its analysis timestamp
    and the next analysis timestamp (temporal hold). Analysis samples a
    reduced subset of frames, so each crop is stretched over the intervening
    source frames.

    Args:
        input_path: Path to the original source video.
        output_path: Path for the final reframed video.
        crop_frame_indices: The original source frame index for each crop
            (must be the same length as crops, sorted ascending).
        crops: Crop rectangles in analysis-frame coordinates, aligned with
            crop_frame_indices.
        analysis_width: Width of the analysis frames (to scale crops to source).
        source_width: Original video width.
        source_height: Original video height.
        source_fps: Original video FPS.
        output_width: Output frame width.
        output_height: Output frame height.

    Raises:
        RuntimeError: If FFmpeg is unavailable or a step fails.
        ValueError: If crop data is missing or the video cannot be opened.
    """
    check_ffmpeg()

    if not crops or len(crops) != len(crop_frame_indices):
        raise ValueError("Crop data and source indices are missing or mismatched.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    out_dir = os.path.dirname(output_path) or "."
    temp_video = os.path.join(out_dir, "_reframed_video_tmp.mp4")
    temp_audio = os.path.join(out_dir, "_original_audio_tmp.aac")

    try:
        _write_cropped_video(
            input_path=input_path,
            output_path=temp_video,
            crop_frame_indices=crop_frame_indices,
            crops=crops,
            analysis_width=analysis_width,
            source_width=source_width,
            source_height=source_height,
            source_fps=source_fps,
            output_width=output_width,
            output_height=output_height,
        )

        if has_audio_stream(input_path):
            _extract_audio(input_path, temp_audio)
            _mux_video_audio(temp_video, temp_audio, output_path)
        else:
            logger.warning("No audio stream found; output will contain video only.")
            os.replace(temp_video, output_path)
    finally:
        for tmp in (temp_video, temp_audio):
            if os.path.isfile(tmp):
                os.remove(tmp)

    logger.info("Rendered output: %s", output_path)


def _write_cropped_video(
    input_path: str,
    output_path: str,
    crop_frame_indices: List[int],
    crops: List[CropRect],
    analysis_width: int,
    source_width: int,
    source_height: int,
    source_fps: float,
    output_width: int,
    output_height: int,
) -> None:
    """Write a cropped and resized video via OpenCV.

    Reads every ORIGINAL-resolution frame, determines which analysis crop
    applies to it, scales the crop from analysis coordinates to source
    coordinates, crops, and resizes to the output resolution.

    Args:
        input_path: Original source video path.
        output_path: Temporary video-only output path.
        crop_frame_indices: Source frame index for each crop.
        crops: Crop rectangles in analysis-frame coordinates.
        analysis_width: Width of the analysis frames.
        source_width: Original video width.
        source_height: Original video height.
        source_fps: Original video FPS.
        output_width: Output frame width.
        output_height: Output frame height.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open original video for rendering: {input_path}")

    if source_fps <= 0:
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, source_fps, (output_width, output_height)
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create OpenCV video writer.")

    scale_x = source_width / analysis_width
    scale_y = scale_x  # Aspect ratio preserved during downscale

    frame_idx = 0
    crop_pos = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        crop = _crop_for_source_frame(
            crop_frame_indices, crops, frame_idx, crop_pos
        )
        if crop is not None and crop_pos < len(crop_frame_indices):
            if (
                crop_pos + 1 < len(crop_frame_indices)
                and frame_idx >= crop_frame_indices[crop_pos + 1]
            ):
                crop_pos += 1
                crop = crops[crop_pos]

            h, w = frame.shape[:2]

            x1 = int(round(crop.x1 * scale_x))
            y1 = int(round(crop.y1 * scale_y))
            x2 = int(round(crop.x2 * scale_x))
            y2 = int(round(crop.y2 * scale_y))

            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 > x1 and y2 > y1:
                cropped = frame[y1:y2, x1:x2]
                resized = cv2.resize(
                    cropped, (output_width, output_height), interpolation=cv2.INTER_AREA
                )
                writer.write(resized)
                written += 1

        frame_idx += 1

    cap.release()
    writer.release()

    logger.info(
        "Cropped video written: %d frames written, %d source frames read",
        written, frame_idx,
    )


def _crop_for_source_frame(
    crop_frame_indices: List[int],
    crops: List[CropRect],
    frame_idx: int,
    crop_pos: int,
) -> Optional[CropRect]:
    """Return the crop currently in effect for a given source frame.

    Holds the current analysis crop (at crop_pos) until the source frame
    reaches the next analysis index.

    Args:
        crop_frame_indices: Source frame index for each crop.
        crops: Crop rectangles.
        frame_idx: Current source frame index.
        crop_pos: Current position in the analysis timeline.

    Returns:
        The CropRect for the given source frame, or None if out of range.
    """
    if not crops or crop_pos >= len(crops):
        return None
    return crops[crop_pos]


def has_audio_stream(input_path: str) -> bool:
    """Detect whether the input video contains an audio stream using ffprobe.

    Uses ffprobe (part of FFmpeg) to list the stream types in the input
    and checks whether an audio stream is present.

    Args:
        input_path: Path to the input video.

    Returns:
        True if at least one audio stream is present, False otherwise.

    Raises:
        RuntimeError: If ffprobe is unavailable or fails to probe the input.
    """
    check_ffmpeg()

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                input_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"ffprobe (part of FFmpeg) is required for audio detection but was "
            f"not available: {exc}"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"FFprobe failed on {input_path}: {result.stderr[-500:]}"
        )

    stream_types = [
        line.strip().lower()
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return "audio" in stream_types


def _extract_audio(input_path: str, output_audio_path: str) -> None:
    """Extract the audio stream from the original video using FFmpeg.

    Only called when the input is known to contain an audio stream.

    Args:
        input_path: Original video path.
        output_audio_path: Path for the extracted audio file.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn", "-acodec", "copy",
            output_audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed: {result.stderr[-500:]}"
        )
    logger.info("Audio extracted to %s", output_audio_path)


def _mux_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> None:
    """Mux the cropped video with the extracted audio using FFmpeg.

    The audio stream is copied (not re-encoded) to preserve quality.

    Args:
        video_path: Cropped video-only file.
        audio_path: Extracted audio file.
        output_path: Final output path.

    Raises:
        RuntimeError: If FFmpeg muxing fails.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg muxing failed: {result.stderr[-500:]}")
    logger.info("Audio muxed into final video (copied, not re-encoded)")
