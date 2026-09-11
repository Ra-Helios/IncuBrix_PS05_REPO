"""Decision timeline generation for auditable processing logs.

Creates a JSON timeline recording per-frame decisions during analysis,
including face positions, crop coordinates, and tracking events.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """A single frame's processing record for the decision timeline.

    Attributes:
        frame_idx: Original frame index in the source video.
        timestamp_ms: Timestamp in milliseconds.
        active_speaker_id: ID of the tracked face (None when no speaker selected).
        confidence: Detection/tracking confidence score.
        face_bbox: Face bounding box as {"x1", "y1", "x2", "y2"}.
        crop_coordinates: Crop rectangle as {"x1", "y1", "x2", "y2"}.
        camera_switch_event: Whether a camera switch occurred.
        is_fallback: Whether fallback crop is being used.
        audio_speech_probability: VAD speech probability at this timestamp.
        mouth_motion_score: Normalized mouth motion evidence of the active face.
        active_speaker_score: Fused audio-visual score of the active face.
        active_speaker_confidence: Heuristic confidence of the selection.
        speaker_switch_event: True when the active speaker changed this frame.
        fallback_reason: Why no speaker was selected (baseline).
        state: Hysteresis state name (optional).
        track_quality: Track continuity/quality in [0,1] (optional).
        face_visibility: Per-frame face visibility in [0,1] (optional).
        head_pose_quality: Head pose evidence in [0,1] (optional).
        mouth_visibility: Fraction of the lips visible in [0,1] (optional).
        confidence_components: Breakdown of the confidence score
            (optional).
        scene_id: Current scene segment id (optional).
        scene_change: True on the frame a scene change is declared
            (optional).
        transition_reason: Reason for the state/decision transition
            (optional).
        crop_jitter: Crop center displacement vs previous frame in px
            (optional).
        crop_velocity: Crop center displacement normalized by frame step
            in px/frame (optional).
    """

    frame_idx: int
    timestamp_ms: float
    active_speaker_id: str
    confidence: float
    face_bbox: Optional[Dict[str, float]] = None
    crop_coordinates: Optional[Dict[str, float]] = None
    camera_switch_event: bool = False
    is_fallback: bool = False
    audio_speech_probability: float = 0.0
    mouth_motion_score: float = 0.0
    active_speaker_score: float = 0.0
    active_speaker_confidence: float = 0.0
    speaker_switch_event: bool = False
    fallback_reason: Optional[str] = None
    state: Optional[str] = None
    track_quality: Optional[float] = None
    face_visibility: Optional[float] = None
    head_pose_quality: Optional[float] = None
    mouth_visibility: Optional[float] = None
    confidence_components: Optional[Dict[str, float]] = None
    scene_id: Optional[int] = None
    scene_change: Optional[bool] = None
    transition_reason: Optional[str] = None
    crop_jitter: Optional[float] = None
    crop_velocity: Optional[float] = None


class TimelineBuilder:
    """Builds and writes the decision_timeline.json file.

    Collects per-frame records during the analysis pass, then writes
    them to a JSON file with metadata.

    Attributes:
        source_file: Path to the source video file.
        analysis_fps: FPS used during analysis.
    """

    def __init__(
        self,
        source_file: str,
        analysis_fps: int = 15,
        target_aspect_ratio: str = "9:16",
        strategy: str = "A",
        analysis_width: Optional[int] = None,
        analysis_height: Optional[int] = None,
    ) -> None:
        """Initialize the timeline builder.

        Args:
            source_file: Path to the source video for metadata.
            analysis_fps: Analysis FPS for metadata.
            target_aspect_ratio: Aspect ratio string (e.g. "9:16").
            strategy: Selector strategy label ("A" or "B") recorded in meta.
            analysis_width: Analysis frame width in pixels.
            analysis_height: Analysis frame height in pixels.
        """
        self.source_file = source_file
        self.analysis_fps = analysis_fps
        self.target_aspect_ratio = target_aspect_ratio
        self.strategy = strategy
        self.analysis_width = analysis_width
        self.analysis_height = analysis_height
        self._records: List[FrameRecord] = []

    def add_record(self, record: FrameRecord) -> None:
        """Add a frame record to the timeline.

        Args:
            record: The FrameRecord to add.
        """
        self._records.append(record)

    def build(self) -> Dict[str, Any]:
        """Build the complete timeline dictionary.

        Returns:
            A dictionary matching the decision_timeline.json schema.
        """
        timeline = {
            "meta": {
                "source_file": os.path.basename(self.source_file),
                "target_aspect_ratio": self.target_aspect_ratio,
                "analysis_fps": self.analysis_fps,
                "strategy": "face_tracking_ema",
                "speaker_strategy": "active_speaker_fusion_vad",
                "comparison_strategy": self.strategy,
                "device": "cpu",
                "analysis_width": self.analysis_width,
                "analysis_height": self.analysis_height,
            },
            "frames": [self._record_to_dict(r) for r in self._records],
        }
        return timeline

    def save(self, output_path: str) -> None:
        """Save the timeline to a JSON file.

        Args:
            output_path: Path where the JSON file will be written.
        """
        timeline = self.build()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)

        logger.info(
            "Timeline saved: %s (%d frames)", output_path, len(self._records)
        )

    @staticmethod
    def _record_to_dict(record: FrameRecord) -> Dict[str, Any]:
        """Convert a FrameRecord to a JSON-serializable dictionary.

        Args:
            record: The FrameRecord to convert.

        Returns:
            Dictionary representation of the record.
        """
        result: Dict[str, Any] = {
            "frame_idx": record.frame_idx,
            "timestamp_ms": round(record.timestamp_ms, 2),
            "active_speaker_id": record.active_speaker_id,
            "confidence": round(record.confidence, 4),
        }

        if record.face_bbox is not None:
            result["face_bbox"] = {
                k: round(v, 2) for k, v in record.face_bbox.items()
            }
        else:
            result["face_bbox"] = None

        if record.crop_coordinates is not None:
            result["crop_coordinates"] = {
                k: round(v, 2) for k, v in record.crop_coordinates.items()
            }
        else:
            result["crop_coordinates"] = None

        result["camera_switch_event"] = record.camera_switch_event
        result["is_fallback"] = record.is_fallback
        result["audio_speech_probability"] = round(record.audio_speech_probability, 4)
        result["mouth_motion_score"] = round(record.mouth_motion_score, 4)
        result["active_speaker_score"] = round(record.active_speaker_score, 4)
        result["active_speaker_confidence"] = round(record.active_speaker_confidence, 4)
        result["speaker_switch_event"] = record.speaker_switch_event
        result["fallback_reason"] = record.fallback_reason

        result["state"] = record.state
        result["track_quality"] = (
            round(record.track_quality, 4) if record.track_quality is not None else None
        )
        result["face_visibility"] = (
            round(record.face_visibility, 4) if record.face_visibility is not None else None
        )
        result["head_pose_quality"] = (
            round(record.head_pose_quality, 4) if record.head_pose_quality is not None else None
        )
        result["mouth_visibility"] = (
            round(record.mouth_visibility, 4) if record.mouth_visibility is not None else None
        )
        if record.confidence_components is not None:
            result["confidence_components"] = {
                k: round(float(v), 4)
                for k, v in record.confidence_components.items()
            }
        else:
            result["confidence_components"] = None
        result["scene_id"] = record.scene_id
        result["scene_change"] = record.scene_change
        result["transition_reason"] = record.transition_reason
        result["crop_jitter"] = (
            round(record.crop_jitter, 4) if record.crop_jitter is not None else None
        )
        result["crop_velocity"] = (
            round(record.crop_velocity, 4) if record.crop_velocity is not None else None
        )

        return result

    @property
    def record_count(self) -> int:
        """Return the number of records collected.

        Returns:
            Number of frame records.
        """
        return len(self._records)
