"""Crop calculation and temporal smoothing for 9:16 video reframing.

Computes a 9:16 crop rectangle centered on the tracked face, with padding
above and below. Applies Exponential Moving Average (EMA) for smooth camera
movement and handles face loss with hold-then-fallback behavior.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from src.config import Config
from src.tracking import TrackedFace

logger = logging.getLogger(__name__)


@dataclass
class CropRect:
    """A crop rectangle with integer pixel coordinates.

    Attributes:
        x1: Left coordinate.
        y1: Top coordinate.
        x2: Right coordinate.
        y2: Bottom coordinate.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization.

        Returns:
            Dictionary with x1, y1, x2, y2 keys.
        """
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


class CropCalculator:
    """Calculates and smooths 9:16 crop rectangles for video reframing.

    Uses the tracked face position to compute a crop that keeps the face
    centered with configurable padding. Applies EMA smoothing to prevent
    jarring camera movements.

    Attributes:
        config: Configuration with aspect ratio, padding, and smoothing params.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the crop calculator.

        Args:
            config: Configuration with target aspect ratio, padding, and EMA alpha.
        """
        self.config = config
        self._ema_crop: Optional[Tuple[float, float, float, float]] = None
        self._frames_since_seen: int = 0
        self._is_first_frame: bool = True
        self._last_center: Optional[Tuple[float, float]] = None
        self.last_jitter: float = 0.0
        self.last_velocity: float = 0.0

    def calculate(
        self,
        tracked_face: Optional[TrackedFace],
        frame_width: int,
        frame_height: int,
    ) -> CropRect:
        """Calculate the smoothed 9:16 crop for the current frame.

        If a tracked face is present, computes a crop centered on it with
        vertical padding. If the face is lost, holds the last crop briefly,
        then falls back to a center-crop. The result is EMA-smoothed.

        Args:
            tracked_face: Current tracked face, or None if face is lost.
            frame_width: Width of the analysis frame in pixels.
            frame_height: Height of the analysis frame in pixels.

        Returns:
            A CropRect in analysis-frame pixel coordinates that will be
            scaled to original resolution during rendering.
        """
        self._frames_since_seen += 1

        if tracked_face is not None:
            raw_crop = self._crop_from_face(
                tracked_face, frame_width, frame_height
            )
            self._frames_since_seen = 0
        elif self._frames_since_seen <= self.config.max_detection_gap_frames:
            # Hold last crop during brief face loss
            if self._ema_crop is not None:
                raw_crop = self._ema_crop
            else:
                raw_crop = self._fallback_crop(frame_width, frame_height)
        else:
            # Face lost too long: fall back to center crop
            raw_crop = self._fallback_crop(frame_width, frame_height)

        # Apply EMA smoothing (adaptive alpha and deadband only
        # activate when configured; defaults reproduce the baseline math).
        prev_center = self._last_center
        alpha = self._effective_alpha(raw_crop, frame_width)
        smoothed = self._apply_ema(raw_crop, alpha)
        smoothed = self._apply_deadband(smoothed, frame_width)
        smoothed = self._apply_velocity_limit(smoothed, frame_width)

        # Clamp to frame boundaries
        clamped = self._clamp_to_frame(smoothed, frame_width, frame_height)

        # Record crop-center displacement for the timeline/jitter metrics.
        self._record_motion(clamped, prev_center)

        # Convert to integer CropRect
        crop = CropRect(
            x1=int(round(clamped[0])),
            y1=int(round(clamped[1])),
            x2=int(round(clamped[2])),
            y2=int(round(clamped[3])),
        )

        self._ema_crop = clamped
        self._is_first_frame = False

        return crop

    def _crop_from_face(
        self,
        face: TrackedFace,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float, float]:
        """Compute a 9:16 crop centered on the tracked face.

        The crop is centered horizontally on the face and positioned
        vertically with padding above and below. The aspect ratio is
        always maintained at target_width:target_height.

        Args:
            face: The tracked face with position information.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            A (x1, y1, x2, y2) tuple in float pixel coordinates.
        """
        target_ratio = self.config.target_width / self.config.target_height

        cx, cy = face.last_center
        _, _, _, face_y2 = face.last_bbox
        face_h = face.last_bbox[3] - face.last_bbox[1]

        # Calculate crop dimensions
        # Start with crop height based on face position and padding
        padding_top = face_h * self.config.face_padding_top
        padding_bottom = face_h * self.config.face_padding_bottom

        crop_top = cy - padding_top
        crop_bottom = cy + padding_bottom
        crop_height = crop_bottom - crop_top

        # Adjust width to maintain aspect ratio
        crop_width = crop_height * target_ratio

        # If the crop is too wide, scale it down to fit the frame width.
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = crop_width / target_ratio

        # If the crop is too tall, scale it down to fit the frame height
        # while keeping the target aspect ratio intact. Without this a tall
        # face with generous padding would exceed the analysis frame and get
        # clamped to a non-9:16 region (stretching content at render time).
        if crop_height > frame_height:
            crop_height = frame_height
            crop_width = crop_height * target_ratio
            if crop_width > frame_width:
                crop_width = frame_width

        # Center horizontally on the face
        crop_x1 = cx - crop_width / 2
        crop_x2 = cx + crop_width / 2

        # Recompute vertical to center face area nicely
        crop_y1 = crop_top
        crop_y2 = crop_top + crop_height

        return (crop_x1, crop_y1, crop_x2, crop_y2)

    def _fallback_crop(
        self, frame_width: int, frame_height: int
    ) -> Tuple[float, float, float, float]:
        """Compute a centered fallback crop when no face is detected.

        Produces a 9:16 crop centered in the frame as a safe default.

        Args:
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            A (x1, y1, x2, y2) tuple centered in the frame.
        """
        target_ratio = self.config.target_width / self.config.target_height

        crop_height = frame_height
        crop_width = crop_height * target_ratio

        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = crop_width / target_ratio

        x1 = (frame_width - crop_width) / 2
        y1 = (frame_height - crop_height) / 2

        return (x1, y1, x1 + crop_width, y1 + crop_height)

    def _effective_alpha(
        self, raw_crop: Tuple[float, float, float, float], frame_width: int
    ) -> float:
        """Return the EMA alpha for this frame (adaptive smoothing).

        With the default deadband (0) the configured smoothing alpha is used
        verbatim, reproducing baseline behavior exactly. When a deadband is
        configured, small moves receive extra smoothing so the frame locks in
        place (reducing micro-jitter) while large moves are followed at full
        speed.

        Args:
            raw_crop: The newly calculated (unsmoothed) crop coordinates.
            frame_width: Analysis frame width in pixels.

        Returns:
            EMA alpha in (0, 1].
        """
        base = (
            float(self.config.ema_alpha)
            if self.config.framing.deadband <= 0
            else float(self.config.framing.smoothing_alpha)
        )
        if self.config.framing.deadband <= 0:
            # baseline mode: the top-level ema_alpha is honored verbatim so the
            # default-data path is byte-for-byte identical to baseline.
            return base
        if self._ema_crop is None or self._is_first_frame:
            return base

        prev_center = self._center_of(self._ema_crop)
        raw_center = self._center_of(raw_crop)
        move = math.hypot(
            prev_center[0] - raw_center[0], prev_center[1] - raw_center[1]
        )
        speed = move / max(1.0, float(frame_width))
        if speed < 0.04:
            return base * 0.5
        return base

    def _apply_deadband(
        self, crop: Tuple[float, float, float, float], frame_width: int
    ) -> Tuple[float, float, float, float]:
        """Suppress crop-center motion below the configured deadband.

        When the smoothed center drifted less than ``deadband`` (as a
        fraction of frame width) on either axis, the crop is re-anchored to
        the previous center - eliminating residual jitter while still moving
        freely once the speaker actually shifts.

        Args:
            crop: Smoothed crop coordinates.
            frame_width: Analysis frame width in pixels.

        Returns:
            Possibly deadbanded crop coordinates.
        """
        db = float(self.config.framing.deadband)
        if db <= 0 or self._ema_crop is None or self._is_first_frame:
            return crop

        prev = self._ema_crop
        dx = ((crop[0] + crop[2]) - (prev[0] + prev[2])) / 2
        dy = ((crop[1] + crop[3]) - (prev[1] + prev[3])) / 2
        if abs(dx) < db * frame_width and abs(dy) < db * frame_width:
            pcx, pcy = self._center_of(prev)
            w = crop[2] - crop[0]
            h = crop[3] - crop[1]
            return (pcx - w / 2, pcy - h / 2, pcx + w / 2, pcy + h / 2)
        return crop

    def _apply_velocity_limit(
        self, crop: Tuple[float, float, float, float], frame_width: int
    ) -> Tuple[float, float, float, float]:
        """Limit the per-frame crop-center displacement.

        ``max_crop_velocity_percent`` is the max center movement per analysis
        frame as a percent of frame width; None disables the limit (baseline).

        Args:
            crop: Smoothed crop coordinates.
            frame_width: Analysis frame width in pixels.

        Returns:
            Velocity-limited crop coordinates.
        """
        limit = self.config.framing.max_crop_velocity_percent
        if limit is None or self._ema_crop is None or self._is_first_frame:
            return crop

        max_step = float(limit) / 100.0 * frame_width
        prev = self._ema_crop
        pcx, pcy = self._center_of(prev)
        ccx, ccy = self._center_of(crop)
        dx = ccx - pcx
        dy = ccy - pcy
        dist = math.hypot(dx, dy)
        if dist > max_step and dist > 0:
            scale = max_step / dist
            new_cx = pcx + dx * scale
            new_cy = pcy + dy * scale
            w = crop[2] - crop[0]
            h = crop[3] - crop[1]
            return (new_cx - w / 2, new_cy - h / 2, new_cx + w / 2, new_cy + h / 2)
        return crop

    def _record_motion(
        self,
        clamped: Tuple[float, float, float, float],
        prev_center: Optional[Tuple[float, float]],
    ) -> None:
        """Update jitter/velocity from the emitted crop centers."""
        center = self._center_of(clamped)
        if prev_center is None:
            self.last_jitter = 0.0
        else:
            self.last_jitter = round(
                math.hypot(center[0] - prev_center[0], center[1] - prev_center[1]), 4
            )
        self.last_velocity = self.last_jitter  # px per analysis frame
        self._last_center = center

    @staticmethod
    def _center_of(crop: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Return the center (cx, cy) of a crop rectangle."""
        return ((crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2)

    def _apply_ema(
        self,
        raw_crop: Tuple[float, float, float, float],
        alpha: float,
    ) -> Tuple[float, float, float, float]:
        """Apply Exponential Moving Average to smooth crop movement.

        On the first frame or when no previous crop exists, uses the raw
        crop directly. Otherwise blends the new crop with the previous
        smoothed crop using the given alpha.

        Args:
            raw_crop: The newly calculated (unsmoothed) crop coordinates.
            alpha: EMA smoothing factor.

        Returns:
            Smoothed (x1, y1, x2, y2) crop coordinates.
        """
        if self._ema_crop is None or self._is_first_frame:
            return raw_crop

        smoothed = tuple(
            alpha * r + (1 - alpha) * s
            for r, s in zip(raw_crop, self._ema_crop)
        )
        return smoothed

    @staticmethod
    def _clamp_to_frame(
        crop: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float, float]:
        """Ensure the crop rectangle stays within frame boundaries.

        Shifts the crop if any edge extends outside the frame, while
        maintaining the crop dimensions.

        Args:
            crop: (x1, y1, x2, y2) crop coordinates.
            frame_width: Frame width in pixels.
            frame_height: Frame height in pixels.

        Returns:
            Clamped (x1, y1, x2, y2) that fits within the frame.
        """
        x1, y1, x2, y2 = crop
        crop_w = x2 - x1
        crop_h = y2 - y1

        # Ensure crop doesn't exceed frame dimensions
        if crop_w > frame_width:
            crop_w = frame_width
        if crop_h > frame_height:
            crop_h = frame_height

        # Clamp x
        if x1 < 0:
            x1 = 0
            x2 = crop_w
        elif x2 > frame_width:
            x2 = frame_width
            x1 = x2 - crop_w

        # Clamp y
        if y1 < 0:
            y1 = 0
            y2 = crop_h
        elif y2 > frame_height:
            y2 = frame_height
            y1 = y2 - crop_h

        return (x1, y1, x2, y2)

    def reset(self) -> None:
        """Reset the calculator state for a new video."""
        self._ema_crop = None
        self._frames_since_seen = 0
        self._is_first_frame = True
        self._last_center = None
        self.last_jitter = 0.0
        self.last_velocity = 0.0

    @property
    def frames_since_seen(self) -> int:
        """Return the number of frames since the face was last detected.

        Returns:
            Count of consecutive analysis frames without a detected face.
        """
        return self._frames_since_seen
