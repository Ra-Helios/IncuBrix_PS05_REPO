# AI Use Disclosure (Candidate Project)

This project is a **candidate assessment submission**. This file documents how
artificial-intelligence tooling was used in its development and operation, so
the submission can be audited honestly.

## 1. Runtime AI models used

The produced software itself uses two **third-party, openly licensed, local,
CPU-only** models. No cloud/paid API is used anywhere in the pipeline.

| Model | Purpose | Origin / License |
|---|---|---|
| MediaPipe Face Landmarker | Face detection and 478 face landmarks (bounding box, mouth aperture) | Google MediaPipe, Apache-2.0 |
| Silero VAD v6.2.1 | Voice-activity probability over 32 ms audio chunks | snakers4/silero-vad, MIT |

All *decision logic* around these models (fusion scoring, hysteresis state
machine, tracking, cropping, fallback behavior, benchmarks) is hand-authored
Python code in this repository and is fully explainable.

## 2. AI coding assistant usage

An AI coding assistant (opencode) was used to write, review, and test the code
in this repository, under continuous human direction.

- The human directed the work and approved each change.
- The assistant authored most of the Python implementation, module structure,
  test suite, and documentation.
- Every implementation step was verified with the local test suite
  (`python -m pytest tests/ -q`) and manual end-to-end runs on synthetic
  assets.
- Development was tracked incrementally with human review at each step. This
  submission is a plain folder copy (no version-control history is included).

## 3. What is NOT done by AI

- No AI-generated media (images/video/audio) is used at runtime.
- Test speech is synthesized offline with the FFmpeg `flite` filter
  (a speech synthesizer), not a generative model.
- No cloud, paid, or GPU-only services are called by the software.
- No personal or user data leaves this machine.

## 4. How to audit the claims

1. **Read the source.** All decisions are produced by readable, deterministic
   Python under `src/`.
2. **Open `decision_timeline.json`.** Every analysis frame records
   `active_speaker_id`, the fused `active_speaker_score`,
   `active_speaker_confidence`, `fallback_reason`, crop coordinates, and
   (extended) state/quality/scene fields. The reasoning is in the file.
3. **Run the benchmark.** `python main.py --input <video> --output out.mp4
   --benchmark` writes a machine-readable `benchmark_results.json`.
4. **Run the tests.** `python -m pytest tests/ -q`.

## 4b. Per-frame confidence model

The `confidence` value stored for a selected face is a deterministic,
documented weighted blend of four per-frame evidence components (all in
[0, 1]) recorded under `confidence_components` in the timeline:

| Component | Source |
|---|---|
| `speech` | Silero VAD speech probability at this frame |
| `mouth` | Mouth-motion evidence of the active face |
| `visual` | Combined face usability — the weakest-link minimum of face/mouth visibility, face size and head-pose frontalness (see the vision module) |
| `track` | Track quality (EMA of detection confidence + continuity ramp) |

The formula (weights from `config.yaml` → `confidence:`):

```
raw        = speech_weight * speech + mouth_weight * mouth + visual_weight * visual
continuity = track_floor   + track_scale * track
confidence = clamp(raw, 0, 1) * continuity
```

Defaults: `speech_weight 0.6`, `mouth_weight 0.25`, `visual_weight 0.15`,
`track_floor 0.75`, `track_scale 0.25`. Consequences, by design:

- No evidence at all → confidence 0.
- A full-evidence frontal, tracked talking face → confidence 1.
- A side or partially visible face is de-rated via its `visual` component
  (the honest low-evidence signal required by the brief).
- A face not yet continuously tracked is scaled down by `continuity`.

Strategy **A** keeps the baseline confidence (tracking detection confidence)
unchanged and only records the components. Strategy **B** sets the frame
`confidence` to the model value above. No learned/model-based confidence
estimator is used.

## 4c. Strategy B wiring and benchmarking

Strategy **A** is the baseline: the configuration is passed through
untouched and every decision path (selector call, framing, confidence) is
identical to the baseline. Strategy **B** additionally enables three explicit,
config-gated improvements:

1. `confidence` is the documented model value (§4b) instead of the detection
   confidence.
2. A track-quality gate (`active_speaker.min_track_quality`, default 0.5 for
   B): a face must be consistently tracked (track quality from the tracker) to take
   or hold the speaker role; a collapsed track steps down immediately. This is
   honest weak-evidence handling, not face recognition.
3. Smarter framing defaults (`framing.deadband` 0.02, `max_crop_velocity_percent`
   2.0): crop-center jitter suppression and a per-frame velocity cap.

An explicit YAML value for any of these always wins over the strategy default,
so B can be tuned or disabled feature-by-feature without code changes.

`--benchmark` writes `benchmark_results.json` with deterministic, explainable
metrics: speaker/fallback/switch proportions, speaker run lengths, crop-center
jitter/velocity, the per-frame confidence distribution, crop bound-violation
counts (validated against the recorded analysis resolution), and timing
including the **realtime factor** (`analysis_seconds / source_duration_seconds`).
These metrics let an auditor compare A vs B runs on the same asset directly.

## 5. Dependency provenance

Third-party software and model licenses are listed in `SOURCES.md`.