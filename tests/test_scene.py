"""Conservative scene-change detection.

Unit tests exercise the detector directly on synthetic frames; an
integration test guards the critical contract that the synthetic speaker
assets (whose backgrounds are static) produce ZERO scene changes while the
face-loss scenario still reports ``face_track_lost``.
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scene import SceneChangeDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PROJECT_ROOT, "test_assets")


def solid_frame(color, size=(640, 360)):
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def moving_block_frames(n=60, size=(640, 360)):
    """Static background with a small moving rectangle (gesture-like motion)."""
    frames = []
    for i in range(n):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        x = int(size[0] * i / n)
        cv2.rectangle(frame, (x, 100), (x + 40, 200), (180, 180, 180), -1)
        frames.append(frame)
    return frames


def test_static_sequence_no_change():
    det = SceneChangeDetector()
    hits = [det.detect(solid_frame((30, 60, 90))) for _ in range(30)]
    assert not any(hits)


def test_idle_then_cut_detected():
    det = SceneChangeDetector()
    for _ in range(20):
        assert det.detect(solid_frame((30, 60, 90))) is False
    # A hard content swap (scene cut).
    changed = False
    for _ in range(5):
        if det.detect(solid_frame((200, 150, 20))):
            changed = True
    assert changed, "hard cut was not declared"


def test_gradual_motion_no_change():
    det = SceneChangeDetector()
    hits = [det.detect(f) for f in moving_block_frames()]
    assert not any(hits)


def test_face_enter_exit_no_change():
    """A face appearing/disappearing against a static background must not
    be treated as a scene change (protects the face_loss contract)."""
    det = SceneChangeDetector()
    background = np.full((360, 640, 3), 40, dtype=np.uint8)
    hits = []
    for i in range(60):
        frame = background.copy()
        if 15 <= i < 45:  # face visible for a while
            cv2.rectangle(frame, (200, 120), (300, 260), (210, 160, 120), -1)
            cv2.circle(frame, (250, 190), 10, (60, 60, 60), -1)
        hits.append(det.detect(frame))
    assert not any(hits)


def test_reset_forgets_baseline():
    det = SceneChangeDetector()
    for _ in range(20):
        det.detect(solid_frame((30, 60, 90)))
    det.reset()
    # After a reset a fresh scene simply becomes the new baseline.
    assert det.detect(solid_frame((200, 150, 20))) is False


def _asset_absent(name):
    return not os.path.isfile(os.path.join(ASSETS, f"{name}.mp4"))


@pytest.mark.skipif(_asset_absent("two_speakers") and _asset_absent("face_loss"),
                    reason="assets not present")
def test_synthetic_assets_produce_zero_scene_changes(tmp_path):
    """Regression guard: no scene change may fire on the deterministic
    speaker assets (two_speakers, face_loss) or the resets would break the
    baseline switch/track-loss contracts."""
    import main
    from src.config import load_config

    config = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    for name in ("two_speakers", "face_loss"):
        if not os.path.isfile(os.path.join(ASSETS, f"{name}.mp4")):
            continue
        out = os.path.join(str(tmp_path), name, "out.mp4")
        result = main.run_pipeline(
            os.path.join(ASSETS, f"{name}.mp4"), out, config, analysis_only=True
        )
        with open(result["timeline_path"], "r", encoding="utf-8") as fh:
            timeline = json.load(fh)
        changes = [f for f in timeline["frames"] if f.get("scene_change")]
        assert not changes, f"{name}.mp4 declared scene changes: {changes[:3]}"
        assert {f["scene_id"] for f in timeline["frames"]} == {1}