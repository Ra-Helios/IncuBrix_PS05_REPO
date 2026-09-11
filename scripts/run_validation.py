"""Final 17-scenario validation and report.

Runs a fixed 17-scenario matrix through the real pipeline (strategy A/B on
the six synthetic speaker assets, plus aspect-ratio, benchmark and
real-world extra scenarios) and writes ``VALIDATION_REPORT.md``.

Usage:
    python scripts/run_validation.py

Exit code 0 when every scenario passes, 1 otherwise. The same matrix is
available to pytest via ``FULL_VALIDATION=1`` (see
``tests/test_validation.py``).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from src.aspect import resolve_output_dimensions
from src.config import apply_strategy_defaults, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "test_assets")
REPORT_PATH = os.path.join(ROOT, "VALIDATION_REPORT.md")
CONFIG_PATH = os.path.join(ROOT, "config.yaml")


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------


class Scenario:
    """A single validation scenario.

    Attributes:
        name: Unique identifier (used in the report table).
        asset: Test asset basename (without .mp4), or None for skip tests.
        strategy: "A" or "B".
        aspect: "9:16" or "1:1".
        benchmark: Whether to also run --benchmark.
        check: Callable(timeline, result) -> list of failure strings.
    """

    def __init__(self, name, asset, strategy, check, aspect="9:16", benchmark=False):
        self.name = name
        self.asset = asset
        self.strategy = strategy
        self.aspect = aspect
        self.benchmark = benchmark
        self.check = check


def _min_accepted_active_ids(timeline, at_least):
    ids = {f.get("active_speaker_id") for f in timeline["frames"]}
    real = {x for x in ids if x is not None}
    return {None} | real, len(real)


def _frames(timeline):
    return timeline["frames"]


def _bound_violations(timeline):
    width = timeline["meta"].get("analysis_width")
    height = timeline["meta"].get("analysis_height")
    count = 0
    for f in _frames(timeline):
        crop = f.get("crop_coordinates")
        if crop is None:
            count += 1
            continue
        x1, y1, x2, y2 = crop["x1"], crop["y1"], crop["x2"], crop["y2"]
        if x2 <= x1 or y2 <= y1:
            count += 1
            continue
        if width and height and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
            count += 1
    return count


def _reas(timeline, allowed):
    actual = {f.get("fallback_reason") for f in _frames(timeline)}
    return actual - set(allowed)


# ---------------------------------------------------------------------------
# Expectation builders
# ---------------------------------------------------------------------------


def single_speaker_locks(min_proportion=0.7):
    def check(timeline, result):
        failures = []
        frames = _frames(timeline)
        ids = {f["active_speaker_id"] for f in frames}
        if ids - {"face_1", None}:
            failures.append(f"unexpected speaker ids {ids - {'face_1', None}}")
        if result["switch_events"] != 0:
            failures.append(f"expected 0 switches, got {result['switch_events']}")
        locked = sum(1 for f in frames if f["active_speaker_id"] == "face_1")
        if locked < len(frames) * min_proportion:
            failures.append("face_1 never locked for >=70%% of frames")
        return failures

    return check


def two_speakers_switch():
    def check(timeline, result):
        failures = []
        ids = {f["active_speaker_id"] for f in _frames(timeline)}
        if result["switch_events"] < 2:
            failures.append(f"expected >=2 switches, got {result['switch_events']}")
        if "face_1" not in ids or "face_2" not in ids:
            failures.append(f"both speakers not selected: {ids}")
        return failures

    return check


def silence_no_speaker():
    def check(timeline, result):
        failures = []
        for f in _frames(timeline):
            if f["active_speaker_id"] is not None:
                failures.append("active speaker selected during silence")
            if f["fallback_reason"] != "no_speech":
                failures.append(f"expected no_speech, got {f['fallback_reason']}")
        return failures

    return check


def face_loss_recover():
    def check(timeline, result):
        failures = []
        reasons = {f["fallback_reason"] for f in _frames(timeline)}
        if "face_track_lost" not in reasons:
            failures.append("face_track_lost never reported")
        if "no_visible_face" in reasons:
            failures.append("no_visible_face reported after a speaker was locked")
        if not any(f["active_speaker_id"] is not None for f in _frames(timeline)):
            failures.append("speaker never re-locked after the gap")
        return failures

    return check


def off_screen_never_selects():
    def check(timeline, result):
        failures = []
        for f in _frames(timeline):
            if f["active_speaker_id"] is not None:
                failures.append("speaker selected from off-screen speech")
            if f["fallback_reason"] not in ("no_speech", "low_active_speaker_confidence"):
                failures.append(f"unexpected fallback {f['fallback_reason']}")
        return failures

    return check


def no_face_fallback():
    def check(timeline, result):
        failures = []
        for f in _frames(timeline):
            if f["active_speaker_id"] is not None:
                failures.append("speaker selected with no visible face")
            if f["fallback_reason"] != "no_visible_face":
                failures.append(f"expected no_visible_face, got {f['fallback_reason']}")
        return failures

    return check


def always_no_faces_or_no_speech():
    """Generic: fallback reasons must be limited to a safe set, no bounds
    violations, no speaker during 'no face' segments (3_in_a_frame safety)."""

    def check(timeline, result):
        failures = []
        for f in _frames(timeline):
            if f.get("fallback_reason") not in (
                None,
                "no_speech",
                "low_active_speaker_confidence",
                "no_visible_face",
                "face_track_lost",
                "audio_unavailable",
            ):
                failures.append(f"unknown fallback reason {f.get('fallback_reason')}")
        return failures

    return check


def square_crops_and_ratio(ratio="1:1"):
    def check(timeline, result):
        failures = []
        if timeline["meta"]["target_aspect_ratio"] != ratio:
            failures.append(
                f"meta ratio {timeline['meta']['target_aspect_ratio']} != {ratio}"
            )
        for f in _frames(timeline):
            c = f["crop_coordinates"]
            if ratio == "1:1" and c["x2"] - c["x1"] != c["y2"] - c["y1"]:
                failures.append(f"non-square 1:1 crop at frame {f['frame_idx']}")
            if ratio == "9:16":
                w = c["x2"] - c["x1"]
                h = c["y2"] - c["y1"]
                if abs(w / h - 9 / 16) > 0.02:
                    failures.append(f"non-9:16 crop at frame {f['frame_idx']}")
        return failures

    return check


def zero_bound_violations_and_confidence_range():
    def check(timeline, result):
        failures = []
        violations = _bound_violations(timeline)
        if violations:
            failures.append(f"{violations} crops out of bounds / degenerate")
        for f in _frames(timeline):
            conf = f.get("confidence")
            if conf is not None and not (0.0 <= conf <= 1.0):
                failures.append(f"confidence {conf} out of [0,1]")
        return failures

    return check


def strategy_b_metadata():
    def check(timeline, result):
        failures = []
        if timeline["meta"]["comparison_strategy"] != "B":
            failures.append("comparison_strategy != B")
        if timeline["meta"]["device"] != "cpu":
            failures.append("device != cpu")
        return failures

    return check


def benchmark_written():
    def check(timeline, result):
        failures = []
        if result.get("benchmark") is None:
            failures.append("benchmark result missing")
            return failures
        bench = result["benchmark"]
        if "realtime_factor" not in bench.get("timing", {}):
            failures.append("realtime_factor missing")
        elif bench["timing"]["realtime_factor"] is not None and not (
            bench["timing"]["realtime_factor"] > 0
        ):
            failures.append(f"realtime_factor {bench['timing']['realtime_factor']}")
        m = bench["metrics"]
        if "confidence_distribution" not in m:
            failures.append("confidence_distribution missing")
        if m.get("crop_bound_violations", -1) != 0:
            failures.append(f"bound violations in benchmark: {m['crop_bound_violations']}")
        return failures

    return check


def tracks_some_faces():
    def check(timeline, result):
        failures = []
        if result["tracked_frames"] == 0:
            failures.append("no frame had a tracked face")
        return failures

    return check


# ---------------------------------------------------------------------------
# The 17-scenario matrix
# ---------------------------------------------------------------------------


def build_scenarios():
    scenarios = [
        Scenario("s1_single_a", "single_speaker_speech", "A", single_speaker_locks()),
        Scenario("s2_single_b", "single_speaker_speech", "B", single_speaker_locks()),
        Scenario("s3_two_a", "two_speakers", "A", two_speakers_switch()),
        Scenario("s4_two_b", "two_speakers", "B", two_speakers_switch()),
        Scenario("s5_silence_a", "silence_speaker", "A", silence_no_speaker()),
        Scenario("s6_silence_b", "silence_speaker", "B", silence_no_speaker()),
        Scenario("s7_faceloss_a", "face_loss", "A", face_loss_recover()),
        Scenario("s8_faceloss_b", "face_loss", "B", face_loss_recover()),
        Scenario("s9_offscreen_a", "off_screen_speech", "A", off_screen_never_selects()),
        Scenario("s10_offscreen_b", "off_screen_speech", "B", off_screen_never_selects()),
        Scenario("s11_noface_a", "no_face_speech", "A", no_face_fallback()),
        Scenario("s12_noface_b", "no_face_speech", "B", no_face_fallback()),
        Scenario(
            "s13_single_1to1",
            "single_speaker_speech",
            "A",
            _combine(
                single_speaker_locks(),
                square_crops_and_ratio("1:1"),
                zero_bound_violations_and_confidence_range(),
            ),
            aspect="1:1",
        ),
        Scenario(
            "s14_two_b_benchmark",
            "two_speakers",
            "B",
            _combine(
                two_speakers_switch(),
                strategy_b_metadata(),
                benchmark_written(),
                zero_bound_violations_and_confidence_range(),
            ),
            benchmark=True,
        ),
        Scenario(
            "s15_silence_1to1",
            "silence_speaker",
            "A",
            _combine(
                silence_no_speaker(),
                square_crops_and_ratio("1:1"),
                zero_bound_violations_and_confidence_range(),
            ),
            aspect="1:1",
        ),
        Scenario(
            "s16_three_in_frame",
            "3_in_a_frame",
            "A",
            _combine(
                tracks_some_faces(),
                zero_bound_violations_and_confidence_range(),
                always_no_faces_or_no_speech(),
            ),
        ),
        Scenario(
            "s17_stock_video",
            "stock_vid",
            "A",
            _combine(
                zero_bound_violations_and_confidence_range(),
                always_no_faces_or_no_speech(),
            ),
        ),
    ]
    return scenarios


def _combine(*checks):
    def check(timeline, result):
        failures = []
        for c in checks:
            failures.extend(c(timeline, result))
        return failures

    return check


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_scenario(scenario, out_root):
    """Run one scenario and return a result dict."""
    import shutil

    out_dir = os.path.join(out_root, scenario.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    config = apply_strategy_defaults(load_config(CONFIG_PATH), scenario.strategy)
    if scenario.aspect:
        w_str, h_str = scenario.aspect.split(":")
        config.target_width = int(w_str)
        config.target_height = int(h_str)
        config.output_width, config.output_height = resolve_output_dimensions(
            config.output_width, config.target_width, config.target_height
        )

    input_path = os.path.join(ASSETS, f"{scenario.asset}.mp4")
    if not os.path.isfile(input_path):
        return {
            "name": scenario.name,
            "asset": scenario.asset,
            "strategy": scenario.strategy,
            "status": "skip",
            "detail": "asset absent (gitignored)",
            "seconds": 0.0,
        }

    started = time.time()
    failures = ["internal_error"]
    try:
        result = main.run_pipeline(
            input_path=input_path,
            output_path=os.path.join(out_dir, "out.mp4"),
            config=config,
            analysis_only=True,
            strategy=scenario.strategy,
            benchmark=scenario.benchmark,
        )
        with open(result["timeline_path"], "r", encoding="utf-8") as fh:
            timeline = json.load(fh)
        failures = scenario.check(timeline, result) or []
    except Exception as exc:  # noqa: BLE001 - report the traceback + continue
        failures = [f"{type(exc).__name__}: {exc}"]
    elapsed = time.time() - started

    status = "pass" if not failures else "fail"
    return {
        "name": scenario.name,
        "asset": scenario.asset,
        "strategy": scenario.strategy,
        "aspect": scenario.aspect,
        "benchmark": scenario.benchmark,
        "status": status,
        "detail": "; ".join(failures),
        "seconds": round(elapsed, 2),
    }


def run_all(out_root=None):
    """Run the full matrix and produce VALIDATION_REPORT.md."""
    if out_root is None:
        out_root = os.path.join(ROOT, "validation_output")
    results = [run_scenario(s, out_root) for s in build_scenarios()]
    write_report(results)
    return results


def write_report(results):
    """Write the markdown report for the given scenario results."""
    passed = sum(1 for r in results if r["status"] == "pass")
    skipped = sum(1 for r in results if r["status"] == "skip")
    failed = sum(1 for r in results if r["status"] == "fail")

    lines = [
        "# Final Validation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 17 scenarios: **{passed} passed**, **{failed} failed**, "
        f"**{skipped} skipped**",
        "",
        "## Results",
        "",
        "| # | Scenario | Asset | Strategy | Aspect | Benchmark | Status | Time (s) | Detail |",
        "|---|----------|-------|----------|--------|-----------|--------|----------|--------|",
    ]
    for i, r in enumerate(results, 1):
        extra = r.get("aspect", "9:16")
        bm = "yes" if r.get("benchmark") else "-"
        lines.append(
            f"| {i} | {r['name']} | {r['asset']} | {r['strategy']} | "
            f"{extra} | {bm} | {r['status']} | {r['seconds']} | {r['detail']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Every scenario runs the real `main.run_pipeline` (CPU-only) on the",
        "  deterministic synthetic assets; scenarios with an unimplemented",
        "  expectation fail loudly rather than being skipped.",
        "- Strategy A must reproduce the baseline behavior exactly; strategy B",
        "  adds the track-quality speaker gate and smarter framing.",
        "- Crop bound violations are validated against the recorded analysis",
        "  resolution; `confidence` must always stay in [0, 1].",
        "- `--benchmark` outputs `realtime_factor`, the confidence distribution,",
        "  and bound-violation metrics.",
        "",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def cli_main() -> int:
    results = run_all()
    failed = [r for r in results if r["status"] == "fail"]
    for r in results:
        mark = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[r["status"]]
        print(
            f"[{mark}] {r['name']:<24} {r['strategy']} "
            f"{r.get('aspect','9:16'):>4} {r['seconds']:6.1f}s "
            f"{r['detail'] if r['status']=='fail' else ''}"
        )
    print(f"\nReport: {REPORT_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(cli_main())