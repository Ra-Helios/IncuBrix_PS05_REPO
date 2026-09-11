"""Audio stream handling for active-speaker detection.

Extracts the audio stream of a video into a temporary mono, 16 kHz PCM WAV
file using FFmpeg so that Silero VAD can analyze it. Handles videos without
an audio stream gracefully: the pipeline then relies on visual evidence only.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioContext:
    """Result of preparing the audio track for VAD analysis.

    Attributes:
        source_path: Path to the source video file.
        audio_available: True if an audio stream was found and extracted.
        wav_path: Path to the extracted mono 16 kHz WAV file (if available).
        sample_rate: Sample rate of the extracted WAV.
        channels: Channel count of the extracted WAV.
        duration_seconds: Duration of the extracted audio, if known.
        warning: Human-readable note for the caller when audio is unavailable.
    """

    source_path: str
    audio_available: bool = False
    wav_path: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: Optional[float] = None
    warning: Optional[str] = field(default=None)


def _ffmpeg_binary(name: str) -> Optional[str]:
    """Locate an FFmpeg binary on PATH.

    Args:
        name: Binary name ("ffmpeg" or "ffprobe").

    Returns:
        Absolute path to the binary, or None if not found.
    """
    if shutil.which(name):
        return shutil.which(name)
    logger.debug("Binary %s not found on PATH", name)
    return None


def has_audio_stream(path: str) -> bool:
    """Check whether the file contains at least one audio stream.

    Uses ffprobe to inspect the container. Returns False when FFmpeg is
    unavailable or probing fails, so the pipeline can degrade gracefully.

    Args:
        path: Path to the media file.

    Returns:
        True if an audio stream is present.
    """
    ffprobe = _ffmpeg_binary("ffprobe")
    if ffprobe is None:
        logger.warning("ffprobe not found; cannot inspect audio streams")
        return False

    if not os.path.isfile(path):
        return False

    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffprobe failed on %s: %s", path, exc)
        return False

    if result.returncode != 0:
        logger.debug("ffprobe returned %d for %s", result.returncode, path)
        return False

    types = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "audio" in types


def extract_audio_to_wav(
    input_path: str,
    output_wav_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """Extract the audio track into a mono PCM WAV at the target rate.

    Args:
        input_path: Path to the input video.
        output_wav_path: Path where the WAV file will be written.
        sample_rate: Target sample rate (Hz).
        channels: Target channel count (1 for mono).

    Returns:
        Path to the written WAV file.

    Raises:
        RuntimeError: If FFmpeg is missing or extraction fails.
    """
    ffmpeg = _ffmpeg_binary("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required to extract the audio track but was not found "
            "on PATH. Audio evidence unavailable."
        )

    os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "wav",
        output_wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Audio extraction failed: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg audio extraction failed (returncode {result.returncode}): "
            f"{result.stderr[-500:]}"
        )
    if not os.path.isfile(output_wav_path):
        raise RuntimeError("FFmpeg audio extraction produced no output file")

    logger.info(
        "Audio extracted: %s (%d Hz, %d ch)", output_wav_path, sample_rate, channels
    )
    return output_wav_path


def prepare_audio(
    input_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    work_dir: Optional[str] = None,
) -> AudioContext:
    """Prepare the audio track for VAD analysis.

    Detects the audio stream, and if present, extracts it to a temporary
    mono 16 kHz PCM WAV file. If the file has no audio (or FFmpeg is
    missing), returns a context with audio_available=False so the caller
    can fall back to visual-only active-speaker estimation.

    Args:
        input_path: Path to the input video.
        sample_rate: Target sample rate for the extracted WAV.
        channels: Target channel count for the extracted WAV.
        work_dir: Directory for temporary files. Defaults to a fresh
            temporary directory that the caller should clean up.

    Returns:
        An AudioContext describing the extracted audio (or its absence).
    """
    context = AudioContext(source_path=input_path, sample_rate=sample_rate, channels=channels)

    if not has_audio_stream(input_path):
        context.audio_available = False
        context.warning = "No audio stream found; speakers can only be estimated visually."
        logger.warning("%s", context.warning)
        return context

    try:
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="incubrix_audio_")
        wav_path = os.path.join(work_dir, "audio_16k_mono.wav")
        extract_audio_to_wav(input_path, wav_path, sample_rate=sample_rate, channels=channels)
        context.wav_path = wav_path
        context.audio_available = True
    except RuntimeError as exc:
        context.audio_available = False
        context.warning = f"Audio extraction failed: {exc}"
        logger.warning("%s", context.warning)

    return context


def wav_duration_seconds(wav_path: str) -> Optional[float]:
    """Return the duration (seconds) of a WAV file.

    Uses soundfile when available; returns None if it cannot be read.

    Args:
        wav_path: Path to a WAV file.

    Returns:
        Duration in seconds, or None on failure.
    """
    try:
        import soundfile as sf
        import numpy as np

        data, sample_rate = sf.read(wav_path, dtype="float32")
        return float(len(np.asarray(data)) / sample_rate)
    except Exception as exc:  # noqa: BLE001 - best-effort helper
        logger.debug("Cannot read duration of %s: %s", wav_path, exc)
        return None