"""Conservative scene-change detection .

Only *hard* scene cuts should reset the reframing pipeline (tracker,
framing EMA, mouth history, speaker state). Gradual motion - a presenter
gesturing, mouth movement, camera shake, a face appearing or disappearing
in a static room - must NOT trip a change.

Scoring:
    For each analysis frame the image is down-scaled to a coarse grid and a
    small grayscale histogram is built per grid cell. The change score is the
    mean L1 distance between the current and previous cell-histogram sets:

        score = sum_cells L1(hist_cur, hist_prev) / (grid*grid*2)

    A hard cut swaps the whole content of every cell and produces a large
    score; localized motion only perturbs a few cells slightly. A change is
    declared only when the current score exceeds both the configured absolute
    threshold AND a multiple of the rolling mean of recent scores (so a
    consistently moving/menacing shot does not creep over the threshold),
    and the exceedance is confirmed over ``confirm_frames`` consecutive
    analysis frames.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SceneChangeDetector:
    """Detects hard scene cuts with a block-histogram content signature.

    Attributes:
        threshold: Absolute change-score threshold (0..1).
        grid: Cells per side of the coarse signature grid (grid*grid cells).
        bins: Grayscale bins per cell histogram.
        confirm_frames: Consecutive frames the candidate exceedance must hold
            before a change is declared.
        relative_factor: The candidate score must also exceed this multiple
            of the rolling mean score.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        grid: int = 8,
        bins: int = 8,
        confirm_frames: int = 2,
        relative_factor: float = 4.0,
    ) -> None:
        self.threshold = float(threshold)
        self.grid = max(2, int(grid))
        self.bins = max(2, int(bins))
        self.confirm_frames = max(1, int(confirm_frames))
        self.relative_factor = float(relative_factor)
        self._reference_sig: Optional[np.ndarray] = None
        self._rolling = 0.05
        self._confirm_count = 0

    def detect(self, frame: np.ndarray) -> bool:
        """Feed one analysis frame; True when a hard scene cut is declared.

        Args:
            frame: BGR analysis frame.

        Returns:
            True exactly on the frame a scene change is confirmed.
        """
        score, changed = self._step(frame)
        if changed:
            logger.info("Scene change declared (score=%.3f)", score)
        return changed

    def score(self, frame: np.ndarray) -> float:
        """Return the raw change score for a frame without declaring cuts.

        Useful for diagnostics/benchmarks.
        """
        score, _ = self._step(frame)
        return round(score, 4)

    def reset(self) -> None:
        """Forget history (e.g., after the pipeline is reset)."""
        self._reference_sig = None
        self._rolling = 0.05
        self._confirm_count = 0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _step(self, frame: np.ndarray) -> tuple:
        """Compare the frame against the current scene reference signature.

        On any non-candidate frame the reference advances (the scene is
        considered to continue), so gradual motion/drift never accumulates.
        A candidate frame (large, outlier content difference) is confirmed
        over two consecutive frames against the *unchanged* reference; only
        then is a hard cut declared and the reference reseeded.
        """
        sig = self._signature(frame)
        if self._reference_sig is None:
            self._reference_sig = sig
            return 0.0, False

        score = float(
            np.sum(np.abs(sig - self._reference_sig))
        ) / (self.grid * self.grid * 2.0)
        self._rolling = 0.9 * self._rolling + 0.1 * score

        candidate = (
            score > self.threshold
            and score > self.relative_factor * self._rolling + 1e-6
        )
        changed = False
        if candidate:
            self._confirm_count += 1
            if self._confirm_count >= self.confirm_frames:
                changed = True
                self._confirm_count = 0
                self._reference_sig = sig
        else:
            self._confirm_count = 0
            self._reference_sig = sig

        return score, changed

    def _signature(self, frame: np.ndarray) -> np.ndarray:
        """Build the block-histogram signature vector of a frame."""
        h, w = frame.shape[:2]
        cell_h = max(1, h // self.grid)
        cell_w = max(1, w // self.grid)
        sig = np.zeros((self.grid * self.grid, self.bins), dtype=np.float32)
        idx = 0
        for cy in range(self.grid):
            for cx in range(self.grid):
                y0, y1 = cy * cell_h, (cy + 1) * cell_h
                x0, x1 = cx * cell_w, (cx + 1) * cell_w
                block = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([block], [0], None, [self.bins], [0, 256])
                hist /= max(1, hist.sum())
                sig[idx] = hist.ravel()
                idx += 1
        return sig