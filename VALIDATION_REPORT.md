# Final Validation Report

Generated: 2026-09-08 20:05:22

- 17 scenarios: **17 passed**, **0 failed**, **0 skipped**

## Results

| # | Scenario | Asset | Strategy | Aspect | Benchmark | Status | Time (s) | Detail |
|---|----------|-------|----------|--------|-----------|--------|----------|--------|
| 1 | s1_single_a | single_speaker_speech | A | 9:16 | - | pass | 5.61 |  |
| 2 | s2_single_b | single_speaker_speech | B | 9:16 | - | pass | 2.5 |  |
| 3 | s3_two_a | two_speakers | A | 9:16 | - | pass | 3.61 |  |
| 4 | s4_two_b | two_speakers | B | 9:16 | - | pass | 3.57 |  |
| 5 | s5_silence_a | silence_speaker | A | 9:16 | - | pass | 2.48 |  |
| 6 | s6_silence_b | silence_speaker | B | 9:16 | - | pass | 2.46 |  |
| 7 | s7_faceloss_a | face_loss | A | 9:16 | - | pass | 2.89 |  |
| 8 | s8_faceloss_b | face_loss | B | 9:16 | - | pass | 2.88 |  |
| 9 | s9_offscreen_a | off_screen_speech | A | 9:16 | - | pass | 2.62 |  |
| 10 | s10_offscreen_b | off_screen_speech | B | 9:16 | - | pass | 2.74 |  |
| 11 | s11_noface_a | no_face_speech | A | 9:16 | - | pass | 1.03 |  |
| 12 | s12_noface_b | no_face_speech | B | 9:16 | - | pass | 1.04 |  |
| 13 | s13_single_1to1 | single_speaker_speech | A | 1:1 | - | pass | 5.29 |  |
| 14 | s14_two_b_benchmark | two_speakers | B | 9:16 | yes | pass | 3.59 |  |
| 15 | s15_silence_1to1 | silence_speaker | A | 1:1 | - | pass | 2.45 |  |
| 16 | s16_three_in_frame | 3_in_a_frame | A | 9:16 | - | pass | 24.97 |  |
| 17 | s17_stock_video | stock_vid | A | 9:16 | - | pass | 8.74 |  |

## Notes

- Every scenario runs the real `main.run_pipeline` (CPU-only) on the
  deterministic synthetic assets; scenarios with an unimplemented
  expectation fail loudly rather than being skipped.
- Strategy A must reproduce the baseline behavior exactly; strategy B
  adds the track-quality speaker gate and smarter framing.
- Crop bound violations are validated against the recorded analysis
  resolution; `confidence` must always stay in [0, 1].
- `--benchmark` outputs `realtime_factor`, the confidence distribution,
  and bound-violation metrics.
