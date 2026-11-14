"""Runs the demo script the way a presenter would, on the bundled samples, with BM25 ranking."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "demo" / "sample_inputs.json"

pytestmark = pytest.mark.skipif(not SAMPLES.exists(), reason="demo samples have not been exported yet")


def run_demo(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "demo" / "app.py"), "--scorer", "bm25", *args],
                          capture_output=True, text=True, timeout=180, cwd=ROOT)


def test_list_shows_the_samples():
    proc = run_demo("--list")
    assert proc.returncode == 0
    for sample in json.loads(SAMPLES.read_text()):
        assert sample["id"] in proc.stdout


def test_every_sample_runs_and_stays_in_budget():
    for i, sample in enumerate(json.loads(SAMPLES.read_text()), start=1):
        proc = run_demo("--sample", str(i), "--budget", "0.25", "--json")
        assert proc.returncode == 0, proc.stderr[-500:]
        data = json.loads(proc.stdout[proc.stdout.index("{"):])
        assert 0 < data["kept_tokens"] <= data["budget_tokens"] < data["original_tokens"]
        assert len(data["files"]) == len(sample["files"])


def test_bad_inputs_end_with_a_message_not_a_traceback():
    missing = run_demo("--root", str(ROOT / "demo"), "--files", "no_such_file.py", "--issue", "anything")
    assert missing.returncode == 2 and "none of the files could be read" in " ".join(missing.stdout.split())
    assert "Traceback" not in missing.stdout + missing.stderr
    bad_budget = run_demo("--sample", "1", "--budget", "-3")
    assert bad_budget.returncode == 2 and "budget must be above 0" in " ".join(bad_budget.stdout.split())
    out_of_range = run_demo("--sample", "99")
    assert out_of_range.returncode != 0 and "--sample must be between" in out_of_range.stderr + out_of_range.stdout
