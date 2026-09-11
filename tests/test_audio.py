"""Unit tests for the baseline audio preparation module.

These help construct a real config via the top-level ``Config`` dataclass
and validate stream detection, extraction, and graceful fallback when a
video has no audio stream.
"""

import os
import subprocess

import pytest

from src import audio as audio_mod
from src.config import AudioConfig


def _ffprobe_available() -> bool:
    try:
        subprocess.run(
            ["ffprobe", "-version"], capture_output=True, timeout=30
        )
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture()
def silent_video(tmp_path):
    """Create a tiny silent video with an audio stream (ffmpeg anullsrc)."""
    path = os.path.join(str(tmp_path), "silent.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-shortest", "-pix_fmt", "yuv420p", path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture()
def no_audio_video(tmp_path):
    """Create a tiny video with no audio stream."""
    path = os.path.join(str(tmp_path), "noaudio.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
        "-an", "-pix_fmt", "yuv420p", path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_default_audio_config():
    cfg = AudioConfig()
    assert cfg.sample_rate == 16000
    assert cfg.channels == 1


@pytest.mark.skipif(not _ffprobe_available(), reason="ffprobe not found")
def test_has_audio_stream_true(silent_video):
    assert audio_mod.has_audio_stream(silent_video) is True


@pytest.mark.skipif(not _ffprobe_available(), reason="ffprobe not found")
def test_has_audio_stream_false(no_audio_video):
    assert audio_mod.has_audio_stream(no_audio_video) is False


@pytest.mark.skipif(not _ffprobe_available(), reason="ffprobe not found")
def test_prepare_audio_available(silent_video):
    ctx = audio_mod.prepare_audio(
        silent_video, sample_rate=16000, channels=1
    )
    assert ctx.audio_available is True
    assert ctx.wav_path is not None
    assert os.path.isfile(ctx.wav_path)
    dur = audio_mod.wav_duration_seconds(ctx.wav_path)
    assert dur is not None and dur >= 0.9 and dur <= 3.0


@pytest.mark.skipif(not _ffprobe_available(), reason="ffprobe not found")
def test_prepare_audio_no_stream_falls_back(no_audio_video):
    ctx = audio_mod.prepare_audio(
        no_audio_video, sample_rate=16000, channels=1
    )
    assert ctx.audio_available is False
    assert ctx.warning is not None
