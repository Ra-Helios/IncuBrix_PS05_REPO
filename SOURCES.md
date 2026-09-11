# Software Sources and Licenses

This project relies on the following third-party software. Versions listed are
the ones verified for this environment (Python 3.14, Windows, CPU only).

## Runtime dependencies

| Component | Purpose | Version(s) verified | License |
|-----------|---------|---------------------|---------|
| OpenCV (`opencv-python`) | Video I/O, frame rendering, resize | 5.0.0.93 | Apache-2.0 |
| MediaPipe (Python) | Face detection, landmarks, mouth aperture | 1.0.1 | Apache-2.0 |
| MediaPipe Face Landmarker model | `face_landmarker.task` (bundled in wheel, also cached at `~/.mediapipe/models`) | float16 `latest` | Apache-2.0 |
| NumPy | Numerical array handling | 2.5.2 | BSD-3-Clause |
| PyYAML | Configuration parsing | 6.0.3 | MIT |
| PyTorch (CPU) | Runtime for the Silero VAD torch/ONNX model | 2.14.0+cpu | BSD-3-Clause |
| Silero VAD | Voice-activity detection (`silero-vad==6.2.1`) | 6.2.1 | MIT |
| Silero VAD model | `silero_vad.jit` / `silero_vad.onnx` (shipped inside the `silero-vad` wheel) | onboard | MIT |
| soundfile / libsndfile | Reading extracted mono 16 kHz WAV audio | 0.14.0 | BSD-3-Clause / LGPL-2.1 |
| ONNX Runtime | CPU inference provider for the Silero ONNX model | 1.29.0 | MIT |
| pytest | Test runner | 9.1.1 | MIT |

## Model sources

- **MediaPipe Face Landmarker** (CPU delegate):
  `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
  MediaPipe components are under the Apache-2.0 license.

- **Silero VAD** v6.2.1 (bundled in the PyPI wheel, no network download):
  `https://github.com/snakers4/silero-vad`
  MIT license.

## System tools used (not Python packages)

- **FFmpeg / FFprobe** (7.1.1) — audio-stream detection, extraction of the
  audio track to a mono 16 kHz WAV, and audio muxing during rendering.
  LGPL-2.1-or-later/GPL depending on build configuration.
- **ffmpeg `flite` filter** (built into the FFmpeg build) — offline speech
  synthesis used only to generate test assets with real speech. libflite is
  BSD-3-Clause software.

## Notes

- PyTorch is installed from the CPU-only index
  (`--index-url https://download.pytorch.org/whl/cpu`); no CUDA packages are
  installed or required.
- No cloud APIs, STT, diarization, or recognition services are used anywhere
  in the pipeline.