"""End-to-end integration test for the original pipeline.

Generates a small synthetic horizontal MP4 with a moving rectangle, then
runs the real main.py pipeline in --analysis-only mode (CPU-only, no GPU)
and verifies the decision_timeline.json output.

The synthetic video contains no real face, so the timeline will use fallback
crops; this validates the full plumbing (ingest -> vision -> tracking ->
framing -> timeline -> CLI) without requiring a real face.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Create a small synthetic horizontal MP4 with a moving rectangle."""
    tmp_path = tmp_path_factory.mktemp("assets")
    video_path = os.path.join(str(tmp_path), "synthetic_input.mp4")

    width, height, fps, duration = 640, 360, 30, 1.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    n_frames = int(fps * duration)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Moving colored rectangle (synthetic "subject")
        x = int(width * i / n_frames)
        cv2.rectangle(frame, (x, 100), (x + 80, 220), (0, 150, 255), -1)
        cv2.putText(frame, f"f{i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    assert os.path.isfile(video_path)
    return video_path


def run_main(*args):
    """Run main.py in a subprocess and return (returncode, output)."""
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "main.py"), *args],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode, result.stdout + result.stderr


def test_cli_reports_cpu(synthetic_video):
    """CLI execution reports CPU as the execution device."""
    code, output = run_main(
        "--input", synthetic_video,
        "--output", "output/test_out.mp4",
        "--analysis-only",
    )
    assert code == 0, output
    assert "Execution device: CPU" in output


def test_metadata_extraction(synthetic_video):
    """Video metadata is extracted correctly."""
    from src.ingest import get_video_info

    info = get_video_info(synthetic_video)
    assert info.width == 640
    assert info.height == 360
    assert info.fps == pytest.approx(30, abs=1)
    assert info.total_frames == 30


def test_invalid_input_handling(tmp_path):
    """Missing input file produces a clear error and non-zero exit code."""
    missing = os.path.join(str(tmp_path), "does_not_exist.mp4")
    code, output = run_main("--input", missing, "--analysis-only")
    assert code != 0
    assert "not found" in output or "ERROR" in output


def test_end_to_end_analysis_only(synthetic_video, tmp_path):
    """Full pipeline runs and generates a valid decision_timeline.json."""
    out_dir = str(tmp_path)
    timeline_path = os.path.join(out_dir, "decision_timeline.json")

    code, output = run_main(
        "--input", synthetic_video,
        "--output", os.path.join(out_dir, "out.mp4"),
        "--analysis-only",
    )
    assert code == 0, output

    assert os.path.isfile(timeline_path)
    with open(timeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["meta"]["device"] == "cpu"
    assert data["meta"]["strategy"] == "face_tracking_ema"
    assert len(data["frames"]) > 0

    for rec in data["frames"]:
        assert "frame_idx" in rec
        assert "timestamp_ms" in rec
        assert "crop_coordinates" in rec
        assert rec["crop_coordinates"]["x1"] < rec["crop_coordinates"]["x2"]
        assert rec["crop_coordinates"]["y1"] < rec["crop_coordinates"]["y2"]


def test_crop_coordinates_stay_in_bounds_from_timeline(synthetic_video, tmp_path):
    """Crop coordinates in the timeline must be within analysis bounds."""
    out_dir = str(tmp_path)
    timeline_path = os.path.join(out_dir, "decision_timeline.json")

    code, output = run_main(
        "--input", synthetic_video,
        "--output", os.path.join(out_dir, "out.mp4"),
        "--analysis-only",
    )
    assert code == 0, output

    with open(timeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rec in data["frames"]:
        c = rec["crop_coordinates"]
        assert c["x1"] >= 0
        assert c["y1"] >= 0
        assert c["x2"] > c["x1"]
        assert c["y2"] > c["y1"]


def test_real_face_tracking_if_asset_present(tmp_path):
    """Real-face demo video: the face must actually be tracked.

    Uses the generated demo asset test_assets/single_speaker.mp4 (a real face
    moving across a wide canvas). If the asset is absent (it is gitignored),
    the test is skipped explicitly rather than silently.
    """
    asset = os.path.join(PROJECT_ROOT, "test_assets", "single_speaker.mp4")
    if not os.path.isfile(asset):
        pytest.skip(
            "test_assets/single_speaker.mp4 not present; "
            "skipping real-face tracking test"
        )

    out_dir = str(tmp_path)
    timeline_path = os.path.join(out_dir, "decision_timeline.json")

    code, output = run_main(
        "--input", asset,
        "--output", os.path.join(out_dir, "out.mp4"),
        "--analysis-only",
    )
    assert code == 0, output

    with open(timeline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with_face = [r for r in data["frames"] if r["face_bbox"]]
    assert len(with_face) >= len(data["frames"]) * 0.9, (
        f"Face should be tracked in most frames, got {len(with_face)}/"
        f"{len(data['frames'])}"
    )

    # baseline: single_speaker.mp4 carries a static face with silent audio, so
    # no speech or mouth motion ever reaches the confidence gate. The face is
    # still tracked and cropped, but no active speaker is ever selected.
    ids = {r["active_speaker_id"] for r in data["frames"]}
    assert ids <= {"face_1", None}, f"Unexpected active speaker ids: {ids}"
    assert None in ids, "Silent single-face asset should never select a speaker"
    for r in data["frames"]:
        assert r["fallback_reason"] == "no_speech" or r["fallback_reason"] is None


def _count_streams(codec_type, path):
    """Count the number of streams of a given codec_type using ffprobe.

    Args:
        codec_type: "video" or "audio".
        path: Media file path.

    Returns:
        Number of matching streams.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"ffprobe failed on {path}: {result.stderr[-500:]}")
    types = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return types.count(codec_type)


def test_render_video_only_input(synthetic_video, tmp_path):
    """Rendering a video WITHOUT an audio stream must not fail.

    The output must contain a video stream, no audio stream, and the
    pipeline must exit successfully with a clear warning in the logs.
    """
    out_path = os.path.join(str(tmp_path), "out_no_audio.mp4")

    code, output = run_main("--input", synthetic_video, "--output", out_path)
    assert code == 0, f"Rendering a video-only input failed:\n{output}"
    assert "No audio stream found; output will contain video only." in output

    assert os.path.isfile(out_path)
    assert _count_streams("video", out_path) == 1
    assert _count_streams("audio", out_path) == 0


def test_render_preserves_audio_when_present(synthetic_video, tmp_path):
    """Rendering an input WITH audio must preserve the audio stream.

    The output must contain both a video and an audio stream (copied, not
    lost) when the source video has audio.
    """
    input_with_audio = os.path.join(str(tmp_path), "with_audio.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-i", synthetic_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            input_with_audio,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"ffmpeg failed to create an audio-bearing input: {result.stderr[-500:]}"
        )

    out_path = os.path.join(str(tmp_path), "out_with_audio.mp4")
    code, output = run_main("--input", input_with_audio, "--output", out_path)
    assert code == 0, output

    assert os.path.isfile(out_path)
    assert _count_streams("video", out_path) == 1
    assert _count_streams("audio", out_path) == 1
    assert "No audio stream found" not in output