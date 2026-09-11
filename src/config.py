"""Configuration management for Active-Speaker Detection and Smart Video Reframing.

Loads configuration from YAML file or uses sensible defaults.
All parameters are defined in one place for easy adjustment.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class AudioConfig:
    """Configuration for audio extraction/processing.

    Attributes:
        sample_rate: Target sample rate for the analyzed audio (Hz).
        channels: Channel count used for the extracted audio (mono=1).
    """

    sample_rate: int = 16000
    channels: int = 1


@dataclass
class VadConfig:
    """Configuration for the Silero voice-activity detector.

    Attributes:
        threshold: Probability threshold for treating a chunk as speech.
        version: Silero VAD version string (informational).
        min_speech_duration_ms: Minimum speech run length to keep.
        min_silence_duration_ms: Silences shorter than this are merged.
        speech_pad_ms: Padding added to each detected interval.
        window_size_samples: VAD window size (512 @ 16 kHz ~ 32 ms).
    """

    threshold: float = 0.5
    version: str = "6.2.1"
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    window_size_samples: int = 512


@dataclass
class MouthConfig:
    """Configuration for mouth-motion feature.

    Attributes:
        smoothing_window: Number of recent |delta a| values averaged.
        motion_scale: Aperture delta that saturates mouth evidence at 1.0.
    """

    smoothing_window: int = 3
    motion_scale: float = 0.10


@dataclass
class ActiveSpeakerConfig:
    """Configuration for audio-visual speaker fusion and hysteresis.

    Attributes:
        audio_weight: Weight of VAD evidence in the fused score.
        mouth_weight: Weight of mouth-motion evidence in the fused score.
        audio_visual_window_ms: Tolerance window when aligning audio and
            visual evidence around a frame timestamp.
        minimum_speaker_hold_ms: Minimum time a speaker must hold before a
            switch is allowed (anti-flicker).
        speaker_switch_margin: Minimum score lead a candidate needs to take
            the floor.
        speaker_switch_min_frames: Consecutive frames a candidate must hold
            the lead before the switch commits.
        minimum_confidence: Minimum fused score for a speaker to be selected.
        speech_presence_threshold: VAD probability treated as "speech present"
            when deciding fallback reasons.
        min_track_quality: Minimum per-face track quality
            for a face to be eligible to take or hold the speaker role.
            0 disables the gate (baseline behavior).
    """

    audio_weight: float = 0.4
    mouth_weight: float = 0.6
    audio_visual_window_ms: float = 300.0
    minimum_speaker_hold_ms: float = 500.0
    speaker_switch_margin: float = 0.10
    speaker_switch_min_frames: int = 3
    minimum_confidence: float = 0.50
    speech_presence_threshold: float = 0.5
    min_track_quality: float = 0.0


@dataclass
class TrackingConfig:
    """Configuration for multi-face tracking (extended).

    Defaults reproduce the baseline behavior (state-of-the-art centroid
    tracker with a 15-frame gap tolerance at 15 fps analysis).

    Attributes:
        max_track_gap_ms: How long (in video time) a face may be missing
            before its track expires.
        match_distance: Max distance (fraction of frame width) to associate
            a detection with an existing track.
        iou_threshold: Minimum overlap ratio for future association checks.
    """

    max_track_gap_ms: float = 1000.0
    match_distance: float = 0.5
    iou_threshold: float = 0.2


@dataclass
class FramingConfig:
    """Configuration for crop framing/smoothing (extended).

    Defaults reproduce the baseline behavior (EMA smoothing, no deadband,
    no velocity limit).

    Attributes:
        smoothing_alpha: Exponential moving average factor (0..1); mirrors
            the baseline ``ema_alpha`` default.
        deadband: Center-deadband as fraction of frame width; 0 disables.
        max_crop_velocity_percent: Optional max crop-center displacement per
            frame as percent of frame width; None means no limit.
    """

    smoothing_alpha: float = 0.18
    deadband: float = 0.0
    max_crop_velocity_percent: Optional[float] = None


@dataclass
class SceneConfig:
    """Configuration for scene-change detection (extended).

    Attributes:
        change_threshold: Content-change threshold above which a scene
            change is declared.
    """

    change_threshold: float = 0.5


@dataclass
class ConfidenceConfig:
    """Documented confidence model weights .

    The per-frame confidence is a weighted blend of the speech, mouth and
    visual evidence, scaled by track continuity:

        raw          = speech_w * speech + mouth_w * mouth + visual_w * visual
        confidence   = clamp(raw, 0, 1) * (track_floor + track_scale * track_quality)

    With the defaults below a front-facing, tracked talking face scores
    high while side/partial faces are automatically de-rated by their
    (weakest-link) visual quality. See AI_USE.md for the formula.

    Attributes:
        speech_weight: Weight of the VAD speech probability.
        mouth_weight: Weight of the mouth-motion evidence.
        visual_weight: Weight of the combined visual (face) quality.
        track_floor: Minimum continuity multiplier.
        track_scale: Continuity multiplier range above the floor.
    """

    speech_weight: float = 0.6
    mouth_weight: float = 0.25
    visual_weight: float = 0.15
    track_floor: float = 0.75
    track_scale: float = 0.25


@dataclass
class AspectConfig:
    """Configuration for supported output aspect ratios (extended).

    Attributes:
        supported_ratios: List of supported W:H strings. The configured
            target ratio must be one of these.
    """

    supported_ratios: List[str] = field(default_factory=lambda: ["9:16", "1:1"])


@dataclass
class Config:
    """Configuration for the video reframing pipeline.

    Attributes:
        analysis_width: Width in pixels for downscaled analysis frames.
        analysis_fps: FPS for analysis (original video may be higher).
        target_width: Aspect ratio width component (e.g. 9 for 9:16).
        target_height: Aspect ratio height component (e.g. 16 for 9:16).
        output_width: Output video width in pixels.
        output_height: Output video height in pixels.
        ema_alpha: Exponential moving average factor for smoothing (0-1).
        face_padding_top: Padding above face as multiple of face height.
        face_padding_bottom: Padding below face as multiple of face height.
        max_detection_gap_frames: Frames to hold last crop before fallback.
        detection_confidence: MediaPipe detection confidence threshold.
        audio: Audio extraction configuration.
        vad: Voice-activity detector configuration.
        active_speaker: Active-speaker fusion configuration.
        mouth: Mouth-motion feature configuration.
    """

    analysis_width: int = 640
    analysis_fps: int = 15
    target_width: int = 9
    target_height: int = 16
    output_width: int = 1080
    output_height: int = 1920
    ema_alpha: float = 0.18
    face_padding_top: float = 1.2
    face_padding_bottom: float = 2.0
    max_detection_gap_frames: int = 15
    detection_confidence: float = 0.5
    max_faces: int = 5

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    active_speaker: ActiveSpeakerConfig = field(default_factory=ActiveSpeakerConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    framing: FramingConfig = field(default_factory=FramingConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    aspect: AspectConfig = field(default_factory=AspectConfig)
    strategy: str = "A"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Path to a YAML config file. If None or file does not
            exist, the default Config values are used.

    Returns:
        A Config instance with the loaded or default values.
    """
    config = Config()

    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return config

        if "analysis_width" in data:
            config.analysis_width = int(data["analysis_width"])
        if "analysis_fps" in data:
            config.analysis_fps = int(data["analysis_fps"])
        if "output_width" in data:
            config.output_width = int(data["output_width"])
        if "output_height" in data:
            config.output_height = int(data["output_height"])
        if "ema_alpha" in data:
            config.ema_alpha = float(data["ema_alpha"])
        if "face_padding_top" in data:
            config.face_padding_top = float(data["face_padding_top"])
        if "face_padding_bottom" in data:
            config.face_padding_bottom = float(data["face_padding_bottom"])
        if "max_detection_gap_frames" in data:
            config.max_detection_gap_frames = int(data["max_detection_gap_frames"])
        if "detection_confidence" in data:
            config.detection_confidence = float(data["detection_confidence"])
        if "max_faces" in data:
            config.max_faces = int(data["max_faces"])

        ar = data.get("target_aspect_ratio")
        if ar and isinstance(ar, dict):
            config.target_width = int(ar.get("width", 9))
            config.target_height = int(ar.get("height", 16))

        _apply_nested(data, "audio", config.audio)
        _apply_nested(data, "vad", config.vad)
        _apply_nested(data, "active_speaker", config.active_speaker)
        _apply_nested(data, "mouth", config.mouth)
        _apply_nested(data, "tracking", config.tracking)
        _apply_nested(data, "framing", config.framing)
        _apply_nested(data, "scene", config.scene)
        _apply_nested(data, "confidence", config.confidence)
        _apply_nested(data, "aspect", config.aspect)

        if "strategy" in data:
            config.strategy = str(data["strategy"]).upper()

    return config


def apply_strategy_defaults(config: Config, strategy: str) -> Config:
    """Apply the improved defaults for the given strategy.

    Strategy "A" reproduces baseline exactly: the config is passed through
    untouched (any advanced values the user configured are preserved).

    Strategy "B" enables the enhanced behavior, but only where the user
    has not already configured a value (i.e. only the baseline defaults are
    replaced), so an explicit YAML value always wins:

    - framing: center deadband (jitter suppression) + a per-frame crop
      velocity cap.
    - active_speaker: candidates and the locked speaker must be
      consistently tracked (track-quality gate).

    Args:
        config: The loaded configuration to mutate.
        strategy: Selector strategy label ("A" or "B").

    Returns:
        The same config instance, mutated for the strategy.
    """
    if strategy != "B":
        return config

    if config.framing.deadband == 0.0:
        config.framing.deadband = 0.02
    if config.framing.max_crop_velocity_percent is None:
        config.framing.max_crop_velocity_percent = 2.0
    if config.active_speaker.min_track_quality == 0.0:
        config.active_speaker.min_track_quality = 0.5
    return config


def _apply_nested(data: dict, key: str, target) -> None:
    """Copy scalar fields from a nested YAML section onto a dataclass.

    Unknown keys are ignored; all target fields are scalars (int/float/str),
    coerced to the target attribute's type when possible.

    Args:
        data: Loaded YAML dictionary.
        key: Nested section key name.
        target: Dataclass instance to populate.
    """
    section = data.get(key)
    if not section or not isinstance(section, dict):
        return
    for attr_name, value in section.items():
        if not hasattr(target, attr_name):
            continue
        current = getattr(target, attr_name)
        try:
            if current is None:
                setattr(target, attr_name, float(value))
            elif isinstance(current, bool):
                setattr(target, attr_name, bool(value))
            elif isinstance(current, int):
                setattr(target, attr_name, int(float(value)))
            elif isinstance(current, list):
                if isinstance(value, list):
                    setattr(target, attr_name, list(value))
            else:
                setattr(target, attr_name, type(current)(value))
        except (TypeError, ValueError):
            continue
