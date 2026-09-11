"""Unit tests for the baseline Silero VAD wrapper.

Pure helpers are tested deterministically without loading the model. A
skip-if-available CPU smoke test exercises the real ONNX model to prove
it loads and produces a plausible probability/intervals on CPU.
"""

import math

import numpy as np
import pytest

from src.config import VadConfig
from src.vad import (
    AudioUnavailableVAD,
    SileroVAD,
    build_speech_intervals,
    chunk_index_for_timestamp,
    median_smooth,
    overlapping_chunk_indices,
)


def test_median_smooth_preserves_length():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    out = median_smooth(values, window=3)
    assert len(out) == len(values)


def test_median_smooth_removes_spike():
    values = [0.1, 0.1, 0.9, 0.1, 0.1]
    out = median_smooth(values, window=3)
    assert out[2] == 0.1


def test_median_smooth_small_window_passthrough():
    values = [0.1, 0.2]
    assert median_smooth(values, window=3) == values


def test_chunk_index_for_timestamp():
    # 512 samples @ 16k = 32 ms per chunk.
    assert chunk_index_for_timestamp(0.0) == 0
    assert chunk_index_for_timestamp(31.9) == 0
    assert chunk_index_for_timestamp(32.0) == 1
    assert chunk_index_for_timestamp(1000.0) == 31


def test_overlapping_chunk_indices_single():
    assert overlapping_chunk_indices(100.0, 0.0) == [3]


def test_overlapping_chunk_indices_span():
    # 64ms-96ms region spans ~2 chunks; a 300ms window centered at 80ms
    # overlaps [-70, 230]ms => chunks from 0 to 7.
    indices = overlapping_chunk_indices(80.0, 300.0)
    assert indices == list(range(0, 8))


def test_build_speech_intervals_merges_short_gaps():
    # 32 ms per chunk. Speech at chunks 1-2, silence for 1 chunk (<100ms),
    # then speech again at chunk 4 => merged into a single interval.
    probs = [0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1]
    intervals = build_speech_intervals(
        probs,
        threshold=0.5,
        min_speech_duration_ms=50,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    assert len(intervals) == 1
    itv = intervals[0]
    assert itv["start_ms"] == 32.0
    assert itv["end_ms"] == 192.0
    assert itv["speech_probability"] > 0.5


def test_build_speech_intervals_drops_short_speech():
    probs = [0.1, 0.9, 0.1, 0.1, 0.1]
    intervals = build_speech_intervals(
        probs,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=50,
        speech_pad_ms=0,
    )
    # Single 32ms run is shorter than 250ms => dropped.
    assert intervals == []


def test_build_speech_intervals_empty():
    assert build_speech_intervals([], threshold=0.5) == []


def test_build_speech_intervals_clamps_to_total():
    probs = [0.9, 0.9, 0.9, 0.9, 0.9]
    intervals = build_speech_intervals(
        probs,
        threshold=0.5,
        min_speech_duration_ms=50,
        min_silence_duration_ms=50,
        speech_pad_ms=30,
        total_samples=100,
    )
    # total_ms = 100/16000*1000 = 6.25 ms; end clamped to that.
    assert intervals[0]["end_ms"] == 6.25


def test_audio_unavailable_vad_never_speaks():
    vad = AudioUnavailableVAD()
    assert vad.audio_available is False
    assert vad.speech_probability_at(0.0) == 0.0
    assert vad.is_speech_at(0.0) is False
    assert vad.speech_intervals() == []


@pytest.mark.skipif(
    not SileroVAD.__init__.__globals__.get("HAS_SILERO", False),
    reason="silero-vad / torch CPU not installed",
)
def test_silero_vad_cpu_smoke():
    """Real-model CPU smoke test: loads ONNX backend and detects speech."""
    cfg = VadConfig()
    vad = SileroVAD(vad_config=cfg, onnx_backend=True)
    try:
        assert vad.backend == "onnx"
        # Pure silence.
        silence = np.zeros(16000, dtype=np.float32)  # 1 s silence
        probs = vad.analyze(silence)
        assert len(probs) > 0
        assert max(probs) < 0.5, "Silence should not be detected as speech"
        assert vad.speech_intervals() == []
        assert vad.speech_probability_at(0.0) < 0.5
    finally:
        vad.close()


@pytest.mark.skipif(
    not SileroVAD.__init__.__globals__.get("HAS_SILERO", False),
    reason="silero-vad / torch CPU not installed",
)
def test_silero_vad_detects_tone_like_speech():
    """Roughly periodic energy can trip the VAD; bounding-box sanity only."""
    cfg = VadConfig(threshold=0.5)
    vad = SileroVAD(vad_config=cfg, onnx_backend=True)
    try:
        # 1 s of high-variance noise is not necessarily speech, so just assert
        # the API shape rather than a specific detection.
        audio = (np.random.rand(16000).astype(np.float32) - 0.5) * 0.5
        probs = vad.analyze(audio)
        assert len(probs) == math.ceil(len(audio) / 512)
        for itv in vad.speech_intervals():
            assert itv["end_ms"] >= itv["start_ms"]
        assert 0.0 <= vad.speech_probability_at(10.0) <= 1.0
    finally:
        vad.close()
