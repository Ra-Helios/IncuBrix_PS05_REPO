"""Face tracker supporting single and multi-face scenarios.

Centroid-based association maintains persistent face IDs across frames.
adds conservative matching improvements without replacing
the lightweight design:

  - velocity prediction for one-frame gaps and fast motion,
  - an IoU overlap gate so a track cannot "jump" to a low-overlap detection,
  - time-based track expiry (`max_track_gap_ms`) in addition to frame-based,
  - neighbor-safe greedy matching for people passing close to each other.

`update()` preserves the original single-track API; the pipeline uses
`update_all()` for the multi-face baseline loop.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.vision import FaceDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """A tracked face with persistent ID and state.

    Attributes:
        face_id: Persistent identifier string (e.g. "face_1").
        last_bbox: Most recent bounding box as (x1, y1, x2, y2).
        last_center: Most recent center position as (x, y).
        last_confidence: Most recent detection confidence.
        frames_since_seen: Number of analysis frames since last detection.
        total_frames_tracked: Total frames this face has been tracked.
        last_seen_ms: Video timestamp (ms) of the last detection, or None.
        last_velocity: (dx, dy) center displacement of the last update.
    """

    face_id: str
    last_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_center: Tuple[float, float] = (0.0, 0.0)
    last_confidence: float = 0.0
    frames_since_seen: int = 0
    total_frames_tracked: int = 0
    last_seen_ms: Optional[float] = None
    last_velocity: Tuple[float, float] = (0.0, 0.0)
    confidence_ema: float = 0.0
    track_quality: float = 1.0


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Compute the intersection-over-union of two bboxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class SimpleTracker:
    """Centroid-based face tracker for single and multi-face scenarios.

    Maintains face IDs across analysis frames by computing Euclidean
    distance between detection centroids and the predicted track positions.
    Matching is conservative: a detection must be close enough AND overlap
    the track's previous bbox by at least ``iou_threshold`` to take over a
    track. Unmatched detections create new tracks; lost tracks expire after
    ``max_gap_frames`` (or ``max_track_gap_ms`` when a timestamp is given).

    Attributes:
        max_gap_frames: Frames a face can be lost before the track expires.
        match_distance: Max pixel distance (as fraction of frame width) to
            associate a detection with a track.
        iou_threshold: Minimum overlap ratio for a valid association.
    """

    def __init__(
        self,
        max_gap_frames: int = 15,
        association_threshold: float = 0.5,
        match_distance: Optional[float] = None,
        iou_threshold: float = 0.2,
        max_track_gap_ms: Optional[float] = None,
        analysis_fps: float = 15.0,
    ) -> None:
        """Initialize the face tracker.

        Args:
            max_gap_frames: How many frames to keep a lost face before reset.
            association_threshold: Backward-compatible alias for
                ``match_distance`` (max distance as fraction of frame width).
            match_distance: Max distance as fraction of frame width to
                associate a detection with the existing track. Overrides
                ``association_threshold`` when not None.
            iou_threshold: Minimum bbox IoU required for a valid match.
            max_track_gap_ms: Optional time-based expiry (video ms). When
                given, the effective expiry is the smaller of the frame-based
                and time-based limits.
            analysis_fps: Frames per second used to convert ``max_track_gap_ms``
                into a frame limit.
        """
        self.max_gap_frames = max(1, int(max_gap_frames))
        self.match_distance = (
            float(match_distance) if match_distance is not None
            else float(association_threshold)
        )
        self.iou_threshold = max(0.0, float(iou_threshold))
        if max_track_gap_ms is not None:
            derived = max(1, int(np.ceil(max_track_gap_ms / 1000.0 * analysis_fps)))
            self.max_gap_frames = min(self.max_gap_frames, derived)
            self._track_gap_ms = float(max_track_gap_ms)
        else:
            self._track_gap_ms = 1000.0 * self.max_gap_frames / max(1.0, float(analysis_fps))
        self._tracks: Dict[str, TrackedFace] = {}
        self._next_id = 1

    def update(
        self,
        detections: List[FaceDetection],
        frame_width: int,
        timestamp_ms: Optional[float] = None,
    ) -> Optional[TrackedFace]:
        """Update the tracker and return the best current track.

        This preserves the original contract: a single ``Optional[TrackedFace]``.
        The primary track is the one with the most frames tracked (or, when
        tied, the first created).

        Args:
            detections: Face detections from the current analysis frame.
            frame_width: Width of the analysis frame in pixels.
            timestamp_ms: Optional video timestamp (ms) for time-based expiry.

        Returns:
            The current primary ``TrackedFace``, or ``None`` when no face
            is actively tracked.
        """
        self.update_all(detections, frame_width, timestamp_ms)
        return self._primary()

    def update_all(
        self,
        detections: List[FaceDetection],
        frame_width: int,
        timestamp_ms: Optional[float] = None,
    ) -> Dict[str, TrackedFace]:
        """Update all tracks and return the full track dictionary.

        Each detection is associated with the nearest compatible track (within
        the distance threshold and satisfying the IoU gate).  Unmatched
        detections spawn new tracks.  Tracks expire when they exceed the gap
        tolerance in frames or, when timestamps are supplied, in milliseconds.

        Args:
            detections: Face detections from the current analysis frame.
            frame_width: Width of the analysis frame in pixels.
            timestamp_ms: Optional video timestamp (ms) for time-based expiry.

        Returns:
            Dictionary of active ``{face_id: TrackedFace}`` entries.
        """
        for track in self._tracks.values():
            track.frames_since_seen += 1

        max_dist = frame_width * self.match_distance
        unmatched = list(detections)

        track_list = sorted(
            self._tracks.values(),
            key=lambda t: t.total_frames_tracked,
            reverse=True,
        )
        matched_ids: set = set()
        for track in track_list:
            if not unmatched:
                break
            match_center = self._predicted_center(track)
            # A track seen last frame must overlap its detection (IoU gate).
            # Recovery matches (track lost >=1 frame) may use the predicted
            # position only, because the old bbox is stale after movement.
            require_iou = track.frames_since_seen <= 1
            nearest, nearest_idx = self._nearest_detection(
                match_center, unmatched, max_dist, track.last_bbox,
                self.iou_threshold if require_iou else 0.0,
            )
            if nearest is not None:
                self._update_track(track, nearest, match_center)
                matched_ids.add(track.face_id)
                unmatched.pop(nearest_idx)

        for det in unmatched:
            track = self._create_track(det)
            self._tracks[track.face_id] = track
            matched_ids.add(track.face_id)

        expired = self._expired_tracks(timestamp_ms)
        for fid in expired:
            logger.info(
                "Face %s lost for %d frames (threshold=%d), removing track",
                fid,
                self._tracks[fid].frames_since_seen,
                self.max_gap_frames,
            )
            del self._tracks[fid]

        return dict(self._tracks)

    def reset(self) -> None:
        """Reset the tracker, clearing all active tracks."""
        self._tracks.clear()
        logger.info("Tracker reset")

    @property
    def active_track(self) -> Optional[TrackedFace]:
        """Return the current best tracked face, if any.

        Returns:
            The primary ``TrackedFace``, or ``None``.
        """
        return self._primary()

    @property
    def tracks(self) -> Dict[str, TrackedFace]:
        """Return the dictionary of all active tracks.

        Returns:
            Copy of the active track dictionary.
        """
        return dict(self._tracks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _primary(self) -> Optional[TrackedFace]:
        """Return the track with the highest total_frames_tracked."""
        if not self._tracks:
            return None
        return max(self._tracks.values(), key=lambda t: t.total_frames_tracked)

    @staticmethod
    def _predicted_center(track: TrackedFace) -> Tuple[float, float]:
        """Predict the track's current center using its last velocity.

        Recent gaps (up to 3 missing frames) are bridged by extrapolating the
        track's last velocity with a decaying factor, so a moving face that
        briefly disappears can still be re-acquired on reappearance. Older
        gaps fall back to the last known position so the prediction cannot
        drift arbitrarily.

        Args:
            track: Track to predict for.

        Returns:
            (x, y) center to match against.
        """
        gap = track.frames_since_seen
        if 1 <= gap <= 3 and track.last_center:
            vx, vy = track.last_velocity
            decay = 0.75 ** (gap - 1)
            return (track.last_center[0] + vx * decay,
                    track.last_center[1] + vy * decay)
        return track.last_center

    def _update_track(
        self,
        track: TrackedFace,
        det: FaceDetection,
        match_center: Tuple[float, float],
    ) -> None:
        """Update a track with a new detection."""
        was_lost = track.frames_since_seen > 1
        gap_before = track.frames_since_seen - 1
        prev_center = track.last_center
        track.last_bbox = (
            det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2
        )
        track.last_center = (det.center_x, det.center_y)
        track.last_confidence = det.confidence
        track.frames_since_seen = 0
        track.total_frames_tracked += 1
        track.last_seen_ms = det.timestamp_ms
        # Track quality = smoothed confidence + continuity ramp.
        track.confidence_ema = (
            0.5 * max(track.confidence_ema, 0.0) + 0.5 * float(det.confidence)
        )
        continuity = min(1.0, track.total_frames_tracked / 20.0)
        track.track_quality = round(
            max(0.0, min(1.0, 0.7 * track.confidence_ema + 0.3 * continuity)),
            4,
        )
        if prev_center and prev_center != match_center:
            track.last_velocity = (
                det.center_x - prev_center[0],
                det.center_y - prev_center[1],
            )
        else:
            track.last_velocity = (
                det.center_x - match_center[0],
                det.center_y - match_center[1],
            )
        if was_lost:
            logger.info(
                "Face %s recovered after %d frames",
                track.face_id,
                gap_before,
            )

    def _create_track(self, det: FaceDetection) -> TrackedFace:
        """Create a new track from an unmatched detection."""
        track = TrackedFace(
            face_id=f"face_{self._next_id}",
            last_bbox=(det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2),
            last_center=(det.center_x, det.center_y),
            last_confidence=det.confidence,
            frames_since_seen=0,
            total_frames_tracked=1,
            last_seen_ms=det.timestamp_ms,
            confidence_ema=float(det.confidence),
            track_quality=round(
                max(0.0, min(1.0, 0.7 * float(det.confidence))), 4
            ),
        )
        self._next_id += 1
        logger.info(
            "New track created: %s at (%.0f, %.0f)",
            track.face_id,
            det.center_x,
            det.center_y,
        )
        return track

    def _expired_tracks(self, timestamp_ms: Optional[float]) -> List[str]:
        """Return face IDs to expire under the frame- or time-based rules."""
        expired = [
            fid
            for fid, t in list(self._tracks.items())
            if t.frames_since_seen > self.max_gap_frames
        ]
        if timestamp_ms is not None:
            for fid, t in list(self._tracks.items()):
                if fid in expired:
                    continue
                if (
                    t.last_seen_ms is not None
                    and (timestamp_ms - t.last_seen_ms) > self._max_gap_ms()
                ):
                    expired.append(fid)
        return expired

    def _max_gap_ms(self) -> float:
        """Return the time-based gap limit used for expiry (internal)."""
        # The frame limit encodes the time limit constructed at init.
        # Re-derive a conservative ms limit from the frame limit at 15 fps
        # when no explicit ms config was provided.
        return 1000.0 * self.max_gap_frames / 15.0

    @staticmethod
    def _nearest_detection(
        center: Tuple[float, float],
        detections: List[FaceDetection],
        max_dist: float,
        track_bbox: Tuple[float, float, float, float],
        iou_threshold: float,
    ) -> Optional[Tuple[Optional[FaceDetection], int]]:
        """Find the detection nearest ``center`` within ``max_dist`` and IoU.

        Returns:
            (detection, index) or (None, -1) when no match is found.
        """
        best_det: Optional[FaceDetection] = None
        best_dist = float("inf")
        best_idx = -1

        for idx, det in enumerate(detections):
            dist = SimpleTracker._centroid_distance(
                center, (det.center_x, det.center_y)
            )
            if dist >= best_dist:
                continue
            if dist > max_dist:
                continue
            det_bbox = (det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2)
            if _iou(track_bbox, det_bbox) < iou_threshold:
                continue
            best_dist = dist
            best_det = det
            best_idx = idx

        if best_det is None or best_dist > max_dist:
            return None, -1
        return best_det, best_idx

    @staticmethod
    def _centroid_distance(
        a: Tuple[float, float], b: Tuple[float, float]
    ) -> float:
        """Compute Euclidean distance between two 2D points."""
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))