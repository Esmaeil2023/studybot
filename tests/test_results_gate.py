"""Quality gate for the committed results.

CI cannot re-run the models (they need the PDFs, a GPU-less hour and an API key), but it can
refuse a commit that replaces the saved results with clearly worse ones, or with broken ones.
The floors are deliberately below the measured values (dev 0.93 / test 0.93), so normal noise
passes and a real regression does not.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANSWER_FLOOR = 0.85


def load(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} is not committed")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("split", ["dev", "test"])
def test_judged_answers_stay_above_the_floor(split):
    summary = load(f"judged_{split}.json")["summary"]
    assert summary["answerable_correct"]["mean"] >= ANSWER_FLOOR
    assert summary["answerable_correct"]["n"] >= 40


@pytest.mark.parametrize("split", ["dev", "test"])
def test_answers_never_cite_chunks_the_model_was_not_shown(split):
    metrics = load(f"answers_{split}.json")["metrics"]
    assert metrics["invalid_citation_rate"] == 0
    assert metrics["answers_without_citation"] in (0, None)


def test_judge_stress_test_has_no_misses():
    summary = load("stress_dev.json")["summary"]
    assert summary["n"] >= 50 and summary["missed"] == []
