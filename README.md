# Active-Speaker Detection and Smart Video Reframing

A CPU-only Python application that reframes a horizontal MP4 of one or more
visible speakers into a vertical **9:16** MP4, *automatically choosing the
active speaker* by fusing Silero VAD audio speech activity with MediaPipe mouth
movement, with smooth face-following cropping, a preserved audio track, and an
auditable `decision_timeline.json`.

This repository implements **audio-visual active-speaker fusion** on top of a
face-tracking baseline, plus **robustness and evaluation** features: robust
multi-face tracking, explainable visual-quality and confidence evidence, an
explicit speaker state machine, conservative scene-change detection, smarter
framing, 1:1 aspect support, strategy comparison (A = baseline behavior,
B = enhanced behavior), and a complete machine-readable benchmark. See §13
for the feature summary.

---

## 1. Project Purpose

Long-form horizontal videos are inefficient on vertical-first platforms.
This tool re-frames a horizontal video into a vertical 9:16 output by:

1. Extracting and analyzing the audio track with Silero Voice Activity Detection (CPU).
2. Analyzing downscaled video frames on the CPU with MediaPipe Face Landmarker.
3. Computing a per-face **mouth-motion evidence** (change in mouth aperture).
4. Fusing VAD speech probability + mouth motion into a per-face active-speaker
   score with hysteresis (hold time, switch margin, confirmation frames).
5. Driving the 9:16 crop to follow the currently speaking face.
6. Rendering the final video from the *original* resolution while preserving audio.

Everything runs on the CPU; there is no CUDA/GPU code anywhere.

## 2. Active-Speaker Scope

- **Active-speaker detection**: speaks "who is talking" by combining global
  Silero VAD evidence with per-face mouth-motion evidence using an explainable
  weighted score.
- **Multi-speaker support**: multiple visible faces are tracked and the crop
  follows the currently speaking one, with hysteresis to avoid flicker.
- **Fallback behavior**: when no speaker can be selected (silence, no visible
  face, face track lost, low confidence, or no audio stream) the pipeline
  reports an auditable `fallback_reason` and falls back to the most
  established visible face for cropping.
- No GPU/CUDA code anywhere in the pipeline.

Not implemented: speaker diarization, or neural lip-sync classifiers (expressly
excluded by the brief along with face recognition).

## 3. Architecture

```
input.mp4
    │
    ├─▶ ffmpeg extract audio → mono 16 kHz WAV ──▶ Silero VAD (CPU) ──▶ speech probability + intervals
    │
    ▼
Downscaled CPU analysis (640px wide, ~15 fps)
    │
    ▼
MediaPipe Face Landmarker (CPU / XNNPACK)
    │
    ▼
Multi-face centroid tracking (face_1, face_2, ...)
    │
    ▼
Mouth aperture  →  |Δ aperture| smoothed  →  per-face mouth-motion evidence
    │                                              │
    ▼                                              ▼
        Active-speaker fusion score  (0.4·speech + 0.6·mouth)
        Hysteresis: lock / hold / switch-margin / confirm frames
                 │
                 ▼
    active_speaker_id + fallback_reason  (auditable per frame)
    │
    ▼
9:16 crop calculation + padding  →  EMA smoothing
    │
    ▼
decision_timeline.json
    │
    ▼
Original-resolution rendering (crop → resize → mux audio)
    │
    ▼
output.mp4 (1080x1920, original audio copied)
```

### Modules

| Module | Responsibility |
|---|---|
| `src/config.py` | Loads `config.yaml` (or defaults) into a typed `Config` incl. audio/vad/mouth/active-speaker/tracking/framing/scene/confidence/aspect sections and the strategy selector. |
| `src/ingest.py` | Validates the video, reads metadata, yields downscaled analysis frames. |
| `src/vision.py` | MediaPipe Face Landmarker, bbox/landmark extraction, mouth aperture metric, per-frame visual-quality evidence. |
| `src/tracking.py` | Multi-face centroid tracker with persistent IDs, time-based gap tolerance, IoU + velocity-predicted recovery. |
| `src/audio.py` | Detects the audio stream and extracts mono 16 kHz WAV via FFmpeg; graceful fallback. |
| `src/vad.py` | Silero VAD wrapper: per-32 ms speech probability, time alignment, speech intervals. |
| `src/active_speaker.py` | Mouth-motion evidence, audio-visual score fusion, hysteresis state machine (incl. an explicit pending-switch state), track-quality speaker gate, fallback reasons. |
| `src/confidence.py` | Documented weighted per-frame confidence model (strategy B). |
| `src/framing.py` | Bounded 9:16/1:1 crop calculation + EMA smoothing, deadband, velocity cap, jitter metrics, safe fallback. |
| `src/scene.py` | Conservative block-histogram scene-change detection with two-frame confirmation. |
| `src/aspect.py` | Supported-ratio parsing/validation and output-resolution derivation. |
| `src/timeline.py` | Builds and writes `decision_timeline.json`. |
| `src/benchmark.py` | Deterministic, machine-readable run metrics (`--benchmark`). |
| `src/renderer.py` | Crops the ORIGINAL video, resizes, muxes original audio via FFmpeg. |
| `main.py` | CLI entry point that wires everything together (`run_pipeline` injectable for tests). |

## 4. Installation

### 4.1 Install FFmpeg

FFmpeg is required for final rendering and audio extraction/handling
(builds with the `flite` filter also let you synthesize test speech).

- **Windows:** download a build from https://www.gyan.dev/ffmpeg/builds/
  (or install via `winget install ffmpeg`), then add the `bin` folder to your PATH.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg` (or your package manager)

Verify with:

```bash
ffmpeg -version
```

### 4.2 Python Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

`silero-vad` also needs a PyTorch CPU build. On Windows install the CPU wheel with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

On first face detection, MediaPipe downloads its `face_landmarker.task` model
and caches it in `~/.mediapipe/models`. Silero VAD is bundled with the package.

## 5. CPU-Only Requirement

- Silero VAD is loaded via the **ONNX runtime backend which forces the
  `CPUExecutionProvider`** (falls back to torch-jit if ONNX is unavailable).
- MediaPipe Face Landmarker runs with its CPU (XNNPACK) delegate.
- The pipeline prints `Execution device: CPU` at startup.

## 6. CLI Usage

```
python main.py --input <input.mp4> [--output <output.mp4>] [options]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--input` | (required) | Path to the horizontal MP4 input. |
| `--output` | `output/reframed.mp4` | Path for the reframed output. |
| `--aspect-ratio` | from config | Target ratio as `W:H`, e.g. `9:16`. |
| `--analysis-only` | off | Generate `decision_timeline.json` only (no rendering). |
| `--config` | `config.yaml` | Path to a YAML config file. |
| `--strategy` | from config | Selector strategy: `A` (baseline) or `B` (enhanced behavior — track-quality speaker gate plus smarter framing defaults). Explicit YAML values always win. |
| `--benchmark` | off | Also write a machine-readable `benchmark_results.json` next to the timeline. |

### Examples

```bash
# Full pipeline (analysis + rendering)
python main.py \
    --input test_assets/single_speaker.mp4 \
    --output output/reframed.mp4

# Analysis only: just write the decision timeline
python main.py \
    --input test_assets/two_speakers.mp4 \
    --output output/reframed.mp4 \
    --analysis-only

# Custom aspect ratio and config (must be in `aspect.supported_ratios`)
python main.py --input input.mp4 --output out.mp4 --aspect-ratio 1:1 --config config.yaml

# Analysis plus a machine-readable benchmark report
python main.py \
    --input test_assets/two_speakers.mp4 \
    --output output/reframed.mp4 \
    --benchmark
```

## 7. Output Description

- `output/reframed.mp4` — vertical 9:16 video (1080×1920 by default; 1080×1080
  when `--aspect-ratio 1:1` is used) rendered from the **original** video, with
  the original audio **copied** (not re-encoded).
- `output/decision_timeline.json` — per-analysis-frame record of the decisions made.

## 8. decision_timeline.json

```json
{
  "meta": {
    "source_file": "input.mp4",
    "target_aspect_ratio": "9:16",
    "analysis_fps": 15,
    "strategy": "face_tracking_ema",
    "speaker_strategy": "active_speaker_fusion_vad",
    "comparison_strategy": "A",
    "device": "cpu",
    "analysis_width": 640,
    "analysis_height": 360
  },
  "frames": [
    {
      "frame_idx": 120,
      "timestamp_ms": 4000.0,
      "active_speaker_id": "face_1",
      "confidence": 0.89,
      "face_bbox": { "x1": 420, "y1": 120, "x2": 960, "y2": 900 },
      "crop_coordinates": { "x1": 420, "y1": 0, "x2": 960, "y2": 960 },
      "camera_switch_event": false,
      "is_fallback": false,
      "audio_speech_probability": 0.99,
      "mouth_motion_score": 0.64,
      "active_speaker_score": 0.78,
      "active_speaker_confidence": 0.97,
      "speaker_switch_event": false,
      "fallback_reason": null,
      "state": "SPEAKER_LOCKED",
      "track_quality": 0.93,
      "face_visibility": 1.0,
      "head_pose_quality": 0.86,
      "mouth_visibility": 1.0,
      "confidence_components": { "speech": 0.99, "mouth": 0.64, "visual": 0.86, "track": 0.93 },
      "scene_id": 1,
      "scene_change": false,
      "transition_reason": "stable",
      "crop_jitter": 2.1,
      "crop_velocity": 2.1
    }
  ]
}
```

baseline field notes:

- `active_speaker_id` — the tracked face currently considered the speaker,
  or `null` when no speaker is selected.
- `active_speaker_score = audio_weight·speech + mouth_weight·mouth`
  (defaults 0.4 / 0.6).
- `mouth_motion_score` — normalized smoothed |Δ aperture| of the active face.
- `active_speaker_confidence` — an explainable heuristic (score·1.25 clipped
  to [0,1]); not calibrated against ground truth.
- `speaker_switch_event` — `true` on the frame a speaker switch commits.
- `camera_switch_event` — mirrors `speaker_switch_event` for compatibility.
- `fallback_reason` — reason no speaker was selected, one of:
  `no_speech`, `low_active_speaker_confidence`, `no_visible_face`,
  `face_track_lost`, `audio_unavailable`.
- `is_fallback` — `true` whenever `active_speaker_id` is `null`.

Extended fields:

- `meta.comparison_strategy` — which selector strategy produced the run
  (`"A"` = baseline, `"B"` = enhanced); `meta.strategy`/`speaker_strategy` keep
  their original values for backward compatibility.
- `meta.analysis_width` / `meta.analysis_height` — exact analysis resolution
  used by the benchmark to verify crop bounds.
- Per-frame fields: `state` (hysteresis state name), `transition_reason`
  (why the state/selection is what it is), `track_quality`,
  `face_visibility`, `head_pose_quality`, `mouth_visibility`,
  `confidence_components` (speech/mouth/visual/track), `scene_id`,
  `scene_change`, `crop_jitter` (px center displacement), `crop_velocity`
  (px/frame). All are recorded for both strategies; under strategy **A**
  `confidence` keeps the baseline value, under **B** it is the documented
  confidence-model value (see AI_USE.md).
- `benchmark_results.json` — written by `--benchmark`; contains
  `schema_version`, `meta` (source, ratio, strategy, device), `metrics`
  (switch/fallback counts, speaker proportion, speaker run lengths, crop
  center jitter/velocity, per-frame confidence distribution, crop
  bound-violation count), and `timing` (`analysis_seconds`,
  `source_duration_seconds`, `realtime_factor`).

## 9. Known Limitations

- The fallback reason is reported in real time; a silent single-face video
  keeps tracking the face but never selects a speaker (correct behavior).
- Face detection is skipped for very small faces (a face smaller than roughly
  100px wide in a 640px analysis frame may not be found).
- The crop is computed only at analysis times (sampled); in between, the last
  crop is held. Fast side-to-side movement may therefore look slightly steppy.
- The audio is remuxed (copied, not re-encoded) but the video track is re-encoded.
- Speaker selection relies on mouth motion; a perfectly still face with speech
  will not be selected (by design, this avoids false attribution).

## 10. Testing

```bash
python -m pytest tests/ -v
```

Unit tests cover: VAD pure helpers + CPU smoke, audio stream detection,
mouth-motion evidence, fuse score/hysteresis/fallback, multi-face tracking,
nested config, core timeline fields. Integration tests run the full pipeline
against deterministic synthetic assets (real detected faces + animated mouths +
synthesized speech):

| Asset | Scenario | Expected |
|---|---|---|
| `single_speaker_speech.mp4` | One face, talking all the time | speaker locked, no switches |
| `two_speakers.mp4` | A → B → A | ≥2 switch events, both speakers selected |
| `silence_speaker.mp4` | Static face, silent audio | always `no_speech`, no speaker |
| `face_loss.mp4` | Speaker leaves for 2s then returns | `face_track_lost`, then re-lock |
| `off_screen_speech.mp4` | Two static faces, offscreen speech | never selects, `no_speech`/`low confidence` |
| `no_face_speech.mp4` | Speech, no visible faces | always `no_visible_face` |

Tests run on CPU and need no GPU.

The suite also covers configuration, timeline extensions, robust tracking,
visual quality, confidence, the state machine, scene detection, smart
framing, aspect ratios, strategy selection and benchmark reporting:
`test_config.py`, `test_timeline_schema.py`, `test_robust_tracking.py`,
`test_vision_quality.py`, `test_confidence.py`, `test_state_machine.py`,
`test_scene.py`, `test_smart_framing.py`, `test_aspect.py`,
`test_strategy.py`, `test_benchmark_report.py`, plus the core benchmark
tests (`test_benchmark.py`).

**Final validation.** The 17-scenario matrix (every synthetic asset
under strategy A and B, plus 1:1 aspect, `--benchmark`, and the
`3_in_a_frame`/stock-video robustness extras) runs the real pipeline and
asserts each expectation. It is gated to keep the everyday suite fast:

```powershell
python scripts/run_validation.py      # runs the matrix, writes VALIDATION_REPORT.md
$env:FULL_VALIDATION = "1"                  # or via pytest:
python -m pytest tests/test_validation.py -o addopts= -q
```

The committed `VALIDATION_REPORT.md` records the current 17/17 pass
result.

## 11. Performance Notes

- Analysis uses 640px-wide frames at ~15 fps (configurable), keeping CPU load low.
- Silero VAD for a 6s clip analyzes in ~50 ms on CPU (ONNX backend).
- The full-resolution video is opened only for the final render pass.
- No multiprocessing is used.

## 12. Configuration

`config.yaml` controls all important parameters, including the core sections:

```yaml
analysis_width: 640
analysis_fps: 15
target_aspect_ratio: { width: 9, height: 16 }
output_width: 1080
output_height: 1920
ema_alpha: 0.18          # lower = smoother/slower camera
face_padding_top: 1.2
face_padding_bottom: 2.0
max_detection_gap_frames: 15   # hold crop this long before fallback
detection_confidence: 0.5

audio:
  sample_rate: 16000
  channels: 1

vad:
  threshold: 0.5
  version: "6.2.1"
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100
  speech_pad_ms: 30
  window_size_samples: 512

mouth:
  smoothing_window: 3      # frames averaged for mouth-motion evidence
  motion_scale: 0.10       # |Δ aperture| that saturates evidence at 1.0

active_speaker:
  audio_weight: 0.4
  mouth_weight: 0.6
  audio_visual_window_ms: 300.0   # alignment tolerance for VAD evidence
  minimum_speaker_hold_ms: 500.0  # anti-flicker hold before a switch
  speaker_switch_margin: 0.10     # required score lead for a switch
  speaker_switch_min_frames: 3    # consecutive confirmations before switching
  minimum_confidence: 0.50        # fused score gate for a speaker candidate
  speech_presence_threshold: 0.5  # "speech present" for fallback reasons
  min_track_quality: 0.0          # speaker gate (0 = off, baseline; B uses 0.5)
```

Additional sections whose defaults reproduce the baseline behavior, plus
a strategy selector:

```yaml
strategy: "A"   # "A" = baseline selector; "B" = enhanced behavior
tracking:
  max_track_gap_ms: 1000.0
  match_distance: 0.5
  iou_threshold: 0.2
framing:
  smoothing_alpha: 0.18   # EMA smoothing; B default changes to use deadband/velocity
  deadband: 0.0           # center deadband as fraction of frame width (0 = off)
  max_crop_velocity_percent: null  # per-frame crop-center velocity cap (None = off)
scene:
  change_threshold: 0.5
confidence:
  speech_weight: 0.6
  mouth_weight: 0.25
  visual_weight: 0.15
  track_floor: 0.75
  track_scale: 0.25
aspect:
  supported_ratios: ["9:16", "1:1"]
```

Strategy **A** passes the loaded config through untouched (exact baseline
behavior). Strategy **B** enables the enhanced behavior — framing
`deadband: 0.02`, `max_crop_velocity_percent: 2.0`, and the speaker
`min_track_quality: 0.5` gate — *only where you have not already set a value*
in the YAML, so explicit configuration always wins.

## 13. Feature summary

All capabilities below are implemented and verified:

- **Configuration:** strategy/config scaffolding, `--strategy` + `--benchmark`
  CLI, `src/benchmark.py`, backward-compatible timeline fields, `AI_USE.md`.
- **Tracking:** robust multi-face tracking (`max_track_gap_ms`,
  `match_distance`, `iou_threshold`, velocity-predicted recovery for brief gaps).
- **Visual evidence:** per-frame visibility / track-quality / head-pose
  evidence; honest weak-evidence reporting for partial/side faces.
- **Confidence:** documented explainable confidence model (weighted
  speech/mouth/visual × track-continuity), recorded components per frame;
  strategy B uses the model value, A keeps the baseline confidence.
- **State machine:** explicit speaker states with a visible
  pending-switch state and per-frame transition reasons.
- **Scene changes:** conservative scene-change detection (two-frame
  confirmation, relative motion baseline) that resets transient pipeline
  state on a cut and never fires on the synthetic assets.
- **Framing:** smarter framing — center deadband, per-frame velocity
  cap, adaptive small-move smoothing, crop jitter/velocity metrics; defaults
  reproduce the baseline exactly.
- **Aspect ratios:** 1:1 + 9:16 validation; `--aspect-ratio` is
  checked against `aspect.supported_ratios` and the output resolution is
  derived to match the ratio.
- **Strategies:** A/B wiring — B enables the track-quality
  speaker gate and smarter-framing defaults; A stays baseline-identical.
- **Benchmarking:** full metrics — realtime factor, confidence
  distribution, crop bound-violation checks in `benchmark_results.json`.
- **Documentation:** this file + `AI_USE.md`.
- **Regression:** the complete test suite passes.

Strategy A vs B summary: both strategies share the same components; A
reproduces the baseline behavior exactly, while B additionally (a) replaces
`confidence` with the documented model value, (b) requires speakers to be
consistently tracked (`min_track_quality` gate), and (c) enables the smarter
framing defaults (deadband + velocity cap).

---

## FAQ / Interview Notes

- *How does the crop stay in bounds?* `CropCalculator._clamp_to_frame` shifts the
  rectangle back into the frame while preserving its W:H dimensions (9:16 or 1:1).
- *How is movement smoothed?* An exponential moving average blends each new crop
  with the previous crop: `smoothed = alpha*new + (1-alpha)*old`.
- *What happens when the face disappears?* The crop is held for
  `max_detection_gap_frames`, then a centered full-frame crop is used — never a
  coordinate outside the video. The active-speaker selector additionally reports
  `face_track_lost` while the locked speaker is away.
- *Why can a still face never be the active speaker?* With `audio_weight=0.4`,
  even perfect speech evidence only contributes 0.4 of the score, below the
  `minimum_confidence` gate of 0.5. Mouth motion is required — this is the
  deliberate audio-visual design.
- *How does VAD evidence get aligned to video time?* Per-32 ms VAD chunks are
  indexed by timestamp; `speech_probability_at(t, window_ms)` takes the maximum
  probability over chunks overlapping `[t-window/2, t+window/2]`
  (`audio_visual_window_ms=300`).
- *What does `silence_speaker.mp4` prove?* A talking *camera* requires both
  speech *and* moving lips; a static face with no speech stays unassigned, and
  the timeline records `no_speech` honestly rather than guessing.