"""Silero VAD wrapper for speech-activity evidence aligned to video time.

Loads the Silero VAD model once on CPU and analyzes a mono, 16 kHz audio
waveform into:
  - a per-32 ms speech probability signal,
  - boolean speech intervals with start/end times and mean probability.

The module exposes pure helper functions (median_smooth, build_speech_intervals)
so interval construction and time alignment are independently testable without
invoking the model.
"""

import logging
import statistics
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from src.config import VadConfig

logger = logging.getLogger(__name__)

try:
    import torch
    from silero_vad import load_silero_vad
    HAS_SILERO = True
except ImportError:
    HAS_SILERO = False
    logger.warning(
        "silero-vad (or a PyTorch CPU build) is not installed. "
        "Audio speech evidence unavailable."
    )


def median_smooth(values: Sequence[float], window: int = 3) -> List[float]:
    """Apply a centered median filter to a sequence.

    Median filtering removes isolated spurious spikes in the per-chunk
    VAD probability signal while preserving speech onsets/offsets.

    Args:
        values: Input sequence of probabilities.
        window: Filter window size (odd numbers recommended).

    Returns:
        Smoothed sequence of the same length.
    """
    values = list(values)
    if window <= 1 or len(values) < window:
        return values

    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(statistics.median(values[lo:hi]))
    return out


def chunk_index_for_timestamp(
    timestamp_ms: float,
    window_size_samples: int = 512,
    sample_rate: int = 16000,
) -> int:
    """Return the VAD chunk index covering a timestamp.

    Args:
        timestamp_ms: Timestamp relative to the audio start, in milliseconds.
        window_size_samples: VAD window size in samples.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Index of the chunk (i.e. probability signal) containing the time.
    """
    sample = int(round(timestamp_ms / 1000.0 * sample_rate))
    return max(0, sample // window_size_samples)


def overlapping_chunk_indices(
    timestamp_ms: float,
    window_ms: float,
    window_size_samples: int = 512,
    sample_rate: int = 16000,
) -> List[int]:
    """Return chunk indices overlapping [timestamp-window/2, timestamp+window/2].

    Args:
        timestamp_ms: Center timestamp in milliseconds.
        window_ms: Tolerance window width in milliseconds (0 = nearest chunk).
        window_size_samples: VAD window size in samples.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Sorted list of chunk indices (at least one entry).
    """
    if window_ms is None or window_ms <= 0:
        return [chunk_index_for_timestamp(timestamp_ms, window_size_samples, sample_rate)]

    chunk_ms = window_size_samples / sample_rate * 1000.0
    start_ms = timestamp_ms - window_ms / 2.0
    end_ms = timestamp_ms + window_ms / 2.0
    first = max(0, int(start_ms // chunk_ms))
    last = max(first, int(end_ms // chunk_ms))
    return list(range(first, last + 1))


def build_speech_intervals(
    probs: Sequence[float],
    threshold: float = 0.5,
    window_size_samples: int = 512,
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    total_samples: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Build speech intervals from a per-chunk probability signal.

    Pure function: a chunk counts as speech when its probability is at or
    above the threshold. Runs are merged across silences shorter than
    min_silence_duration_ms and dropped when shorter than
    min_speech_duration_ms. Each interval is padded by speech_pad_ms and
    annotated with the mean speech probability of its detected run.

    Args:
        probs: Per-chunk speech probabilities (one per ~32 ms at 16 kHz).
        threshold: Probability threshold for speech.
        window_size_samples: VAD window size in samples.
        sample_rate: Audio sample rate in Hz.
        min_speech_duration_ms: Minimum speech run duration to keep.
        min_silence_duration_ms: Silences shorter than this are merged.
        speech_pad_ms: Padding added to each interval (ms).
        total_samples: Total audio samples (clamps interval end). Defaults to
            the implied length of `probs`.

    Returns:
        List of {"start_ms", "end_ms", "speech_probability"} dictionaries.
    """
    probs = list(probs)
    n = len(probs)
    if n == 0:
        return []

    chunk_ms = window_size_samples / sample_rate * 1000.0
    if total_samples is None:
        total_ms = n * chunk_ms
    else:
        total_ms = total_samples / sample_rate * 1000.0

    runs: List[tuple] = []
    i = 0
    while i < n:
        if probs[i] >= threshold:
            j = i
            while j < n and probs[j] >= threshold:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    if not runs:
        return []

    merged: List[tuple] = [runs[0]]
    for run in runs[1:]:
        gap_ms = (run[0] - merged[-1][1]) * chunk_ms
        if gap_ms < min_silence_duration_ms:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)

    intervals: List[Dict[str, float]] = []
    for start_idx, end_idx in merged:
        run_start_ms = start_idx * chunk_ms
        run_end_ms = end_idx * chunk_ms
        if (run_end_ms - run_start_ms) < min_speech_duration_ms:
            continue

        start_ms = max(0.0, run_start_ms - speech_pad_ms)
        end_ms = min(total_ms, run_end_ms + speech_pad_ms)
        mean_prob = float(np.mean(probs[start_idx:end_idx]))

        intervals.append(
            {
                "start_ms": round(start_ms, 2),
                "end_ms": round(end_ms, 2),
                "speech_probability": round(mean_prob, 4),
            }
        )
    return intervals


class SileroVAD:
    """CPU-only Silero VAD engine producing a time-aligned probability signal.

    The model is loaded once (typically the ONNX backend, which forces the
    CPU execution provider) and reused for the whole video. `analyze` must be
    called with a mono 16 kHz float waveform before querying probabilities.

    Attributes:
        audio_available: Always True for a live Silero VAD model.
        sample_rate: Sample rate the model expects.
        window_size_samples: VAD window size (512 @ 16 kHz, ~32 ms).
    """

    def __init__(
        self,
        vad_config: VadConfig,
        sample_rate: int = 16000,
        onnx_backend: bool = True,
    ) -> None:
        """Initialize the VAD model on CPU.

        Args:
            vad_config: VAD configuration (threshold, durations, window size).
            sample_rate: Expected audio sample rate.
            onnx_backend: Prefer the ONNX runtime backend (forces CPU).

        Raises:
            RuntimeError: If silero-vad or a CPU PyTorch build is unavailable.
        """
        if not HAS_SILERO:
            raise RuntimeError(
                "silero-vad is required for audio speech evidence. "
                "Install with: pip install silero-vad"
            )

        self.vad_config = vad_config
        self.sample_rate = sample_rate
        self.audio_available = True
        self.window_size_samples = getattr(vad_config, "window_size_samples", 512)
        self.threshold = getattr(vad_config, "threshold", 0.5)
        self.backend: str = ""

        self._model = self._load(onnx_backend=onnx_backend)
        self._probs: List[float] = []
        self._intervals: List[Dict[str, float]] = []
        logger.info("Silero VAD initialized on CPU (backend=%s)", self.backend)

    def _load(self, onnx_backend: bool):
        """Load the Silero model, preferring the CPU-forcing ONNX backend.

        Args:
            onnx_backend: Try the ONNX backend first.

        Returns:
            A loaded Silero VAD model.

        Raises:
            RuntimeError: If neither backend can be loaded.
        """
        last_error: Optional[Exception] = None
        backends = [True] if onnx_backend else [False]
        if onnx_backend:
            backends.append(False)

        for use_onnx in backends:
            try:
                model = load_silero_vad(onnx=use_onnx)
                self.backend = "onnx" if use_onnx else "torch-jit"
                if use_onnx:
                    providers = getattr(getattr(model, "session", None), "get_providers", None)
                    if callable(providers):
                        active = providers()
                        if "CPUExecutionProvider" in active or any("CPU" in p for p in active):
                            logger.info("VAD ONNX execution providers: %s", active)
                model.reset_states()
                return model
            except Exception as exc:  # noqa: BLE001 - try next backend
                last_error = exc
                logger.warning("Silero VAD %s load failed: %s", "onnx" if use_onnx else "jit", exc)

        raise RuntimeError(f"Unable to load Silero VAD model on CPU: {last_error}")

    def analyze(self, audio: Union[np.ndarray, "torch.Tensor"]) -> List[float]:
        """Run VAD over a mono waveform and store the probability signal.

        Args:
            audio: Mono float32 samples at ``self.sample_rate`` Hz.

        Returns:
            The smoothed per-chunk speech probability signal.
        """
        tensor = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(
            np.asarray(audio, dtype=np.float32)
        )
        if tensor.ndim > 1:
            tensor = tensor.flatten()

        self._model.reset_states()
        window = self.window_size_samples
        raw_probs: List[float] = []

        with torch.no_grad():
            for i in range(0, len(tensor), window):
                chunk = tensor[i:i + window]
                if chunk.shape[0] < window:
                    chunk = torch.nn.functional.pad(chunk, (0, window - chunk.shape[0]))
                raw_probs.append(float(self._model(chunk, self.sample_rate).item()))

        self._raw_probs = raw_probs
        self._probs = median_smooth(raw_probs, window=3)
        self._intervals = build_speech_intervals(
            self._probs,
            threshold=self.threshold,
            window_size_samples=self.window_size_samples,
            sample_rate=self.sample_rate,
            min_speech_duration_ms=getattr(self.vad_config, "min_speech_duration_ms", 250),
            min_silence_duration_ms=getattr(self.vad_config, "min_silence_duration_ms", 100),
            speech_pad_ms=getattr(self.vad_config, "speech_pad_ms", 30),
            total_samples=len(tensor),
        )
        return self._probs

    def speech_probability_at(
        self, timestamp_ms: float, window_ms: float = 0.0
    ) -> float:
        """Interpolate the speech probability signal at a video timestamp.

        When ``window_ms > 0`` the maximum over chunks overlapping the
        tolerance window is returned, so a brief utterance close to the
        frame contributes evidence.

        Args:
            timestamp_ms: Video timestamp in milliseconds.
            window_ms: Tolerance window in milliseconds (0 = nearest chunk).

        Returns:
            A probability in [0, 1] (0.0 when no signal is available).
        """
        if not self._probs:
            return 0.0
        indices = overlapping_chunk_indices(
            timestamp_ms,
            window_ms,
            self.window_size_samples,
            self.sample_rate,
        )
        indices = [i for i in indices if i < len(self._probs)]
        if not indices:
            nearest = chunk_index_for_timestamp(
                timestamp_ms, self.window_size_samples, self.sample_rate
            )
            indices = [nearest]
            if nearest >= len(self._probs):
                return 0.0
        return float(max(self._probs[i] for i in indices))

    def is_speech_at(self, timestamp_ms: float, window_ms: float = 0.0) -> bool:
        """Return whether the time point is considered speech.

        Args:
            timestamp_ms: Video timestamp in milliseconds.
            window_ms: Tolerance window in milliseconds.

        Returns:
            True when the signal probability is at or above the threshold.
        """
        return self.speech_probability_at(timestamp_ms, window_ms) >= self.threshold

    def speech_intervals(self) -> List[Dict[str, float]]:
        """Return the detected speech intervals.

        Returns:
            List of {"start_ms", "end_ms", "speech_probability"} dicts.
        """
        return list(self._intervals)

    def close(self) -> None:
        """No-op; kept for API symmetry with the visual detector."""
        return None


class AudioUnavailableVAD:
    """Stand-in speech-evidence provider when the file has no audio stream.

    All speech evidence is reported as absent and no intervals are produced.
    The pipeline then relies purely on mouth-motion evidence.
    """

    audio_available = False
    threshold = 0.5

    def speech_probability_at(self, timestamp_ms: float, window_ms: float = 0.0) -> float:
        return 0.0

    def is_speech_at(self, timestamp_ms: float, window_ms: float = 0.0) -> bool:
        return False

    def speech_intervals(self) -> List[Dict[str, float]]:
        return []

    def close(self) -> None:
        return None