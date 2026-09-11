"""Unit tests for the baseline nested configuration loading."""

import os

import pytest

from src.config import Config, load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_config_has_phase2_sections():
    cfg = Config()
    assert cfg.audio.sample_rate == 16000
    assert cfg.vad.threshold == 0.5
    assert cfg.mouth.smoothing_window == 3
    assert cfg.active_speaker.audio_weight == 0.4
    assert cfg.active_speaker.mouth_weight == 0.6
    assert cfg.max_faces == 5


def test_load_project_yaml():
    path = os.path.join(PROJECT_ROOT, "config.yaml")
    cfg = load_config(path)
    assert cfg.audio.sample_rate == 16000
    assert cfg.vad.threshold == 0.5
    assert cfg.vad.version == "6.2.1"
    assert cfg.mouth.smoothing_window == 3
    assert cfg.mouth.motion_scale == pytest.approx(0.10)
    assert cfg.active_speaker.minimum_confidence == pytest.approx(0.50)
    assert cfg.active_speaker.speaker_switch_min_frames == 3
    assert cfg.target_width == 9
    assert cfg.target_height == 16


def test_load_invalid_path_falls_back_to_defaults():
    cfg = load_config("nonexistent.yaml")
    assert cfg.audio.sample_rate == 16000


def test_nested_unknown_keys_ignored(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "audio:\n sample_rate: 44100\n unknown_key: 123\n"
        "vad:\n threshold: 0.3\n"
    )
    cfg = load_config(str(p))
    assert cfg.audio.sample_rate == 44100
    assert cfg.vad.threshold == 0.3
