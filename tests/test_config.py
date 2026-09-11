"""Unit tests for the configuration scaffolding.

These cover the new nested config categories (tracking, framing, scene,
aspect) and the strategy selector key. Defaults must reproduce the
baseline behavior exactly.
"""

import os

import pytest

from src.config import Config, load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_config_has_extended_sections():
    cfg = Config()
    assert cfg.strategy == "A"
    assert cfg.tracking.max_track_gap_ms == pytest.approx(1000.0)
    assert cfg.tracking.match_distance == pytest.approx(0.5)
    assert cfg.tracking.iou_threshold == pytest.approx(0.2)
    assert cfg.framing.smoothing_alpha == pytest.approx(0.18)
    assert cfg.framing.deadband == pytest.approx(0.0)
    assert cfg.framing.max_crop_velocity_percent is None
    assert cfg.scene.change_threshold == pytest.approx(0.5)
    assert cfg.aspect.supported_ratios == ["9:16", "1:1"]


def test_load_project_yaml_extended_defaults():
    path = os.path.join(PROJECT_ROOT, "config.yaml")
    cfg = load_config(path)
    assert cfg.strategy == "A"
    assert cfg.tracking.max_track_gap_ms == pytest.approx(1000.0)
    assert cfg.tracking.match_distance == pytest.approx(0.5)
    assert cfg.framing.smoothing_alpha == pytest.approx(0.18)
    assert cfg.framing.deadband == pytest.approx(0.0)
    assert cfg.framing.max_crop_velocity_percent is None
    assert cfg.scene.change_threshold == pytest.approx(0.5)
    assert cfg.aspect.supported_ratios == ["9:16", "1:1"]


def test_nested_values_loaded(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "strategy: B\n"
        "tracking:\n"
        "  max_track_gap_ms: 500.0\n"
        "  match_distance: 0.3\n"
        "framing:\n"
        "  max_crop_velocity_percent: 2.0\n"
        "scene:\n"
        "  change_threshold: 0.3\n"
        "aspect:\n"
        "  supported_ratios: ['9:16', '1:1', '4:5']\n"
    )
    cfg = load_config(str(p))
    assert cfg.strategy == "B"
    assert cfg.tracking.max_track_gap_ms == pytest.approx(500.0)
    assert cfg.tracking.match_distance == pytest.approx(0.3)
    assert cfg.tracking.iou_threshold == pytest.approx(0.2)
    assert cfg.framing.max_crop_velocity_percent == pytest.approx(2.0)
    assert cfg.scene.change_threshold == pytest.approx(0.3)
    assert cfg.aspect.supported_ratios == ["9:16", "1:1", "4:5"]


def test_strategy_normalized_to_upper(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("strategy: b\n")
    cfg = load_config(str(p))
    assert cfg.strategy == "B"


def test_velocity_none_explicit(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("framing:\n max_crop_velocity_percent: null\n")
    cfg = load_config(str(p))
    assert cfg.framing.max_crop_velocity_percent is None


def test_unknown_nested_keys_ignored(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "tracking:\n"
        "  match_distance: 0.7\n"
        "  unknown_key: 999\n"
        "aspect:\n"
        "  supported_ratios: ['9:16']\n"
        "  bogus: true\n"
    )
    cfg = load_config(str(p))
    assert cfg.tracking.match_distance == pytest.approx(0.7)
    assert cfg.tracking.max_track_gap_ms == pytest.approx(1000.0)
    assert cfg.aspect.supported_ratios == ["9:16"]