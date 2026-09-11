"""Final 17-scenario validation matrix (pytest entry).

The full matrix runs the real pipeline over the deterministic synthetic
speaker assets under strategy A and B, plus aspect-ratio, benchmark and
real-world extra scenarios, and asserts every scenario passes.

Running the matrix executes roughly 17 in-process pipeline runs (a few
minutes), so it is gated behind ``FULL_VALIDATION=1``:

    python -m pytest tests/test_validation.py -q -o
        addopts="" --no-header -p no:cacheprovider   # via shell after setting
    $env:FULL_VALIDATION="1"

It is skipped by default so the everyday suite stays fast.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_validation import run_all  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("FULL_VALIDATION") != "1",
    reason="set FULL_VALIDATION=1 to run the 17-scenario validation matrix",
)


def test_17_scenario_matrix(tmp_path):
    results = run_all(out_root=str(tmp_path))
    failures = [r for r in results if r["status"] == "fail"]
    skipped = [r for r in results if r["status"] == "skip"]
    message = "\n".join(
        f"[{r['status']}] {r['name']}: {r['detail']}"
        for r in (failures + skipped)
    )
    assert not failures, f"{len(failures)} scenarios failed:\n{message}"