"""Answer correctness (Phase 6): an LLM judge, checked against human labels.

    python -m studybot.judge --split dev                  grade results/answers_dev.json
    python -m studybot.judge --split dev --sample 24      write a sheet for you to label by hand
    python -m studybot.judge --validate data/eval/judge_labels.csv

What is graded
    answerable question, bot answered   the judge compares the answer with the gold answer:
                                        correct / partial / wrong
    answerable question, bot abstained  wrong (no model call needed)
    unanswerable question, bot abstained  right (no model call needed)
    unanswerable question, bot answered   listed for manual review, counted as wrong in the
                                        strict number (a few of these are arguable; see README)

Why validate the judge: an LLM judge is itself a model with biases. We label a sample by hand,
measure how often the judge agrees (and Cohen's kappa, which discounts chance agreement), and
report that next to the headline numbers. Confidence intervals come from a bootstrap that
resamples *question pairs* (German and English twin together), because the twins are not
independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

LABELS = ("correct", "partial", "wrong")
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
DEFAULT_JUDGE = "claude-sonnet-5-5"  # a different, stronger model than the answering model
BOOTSTRAP_ROUNDS = 2000
SEED = 13

JUDGE_PROMPT = """You grade answers of a question-answering system about university regulations.
You get a question, the reference answer (the truth) and the system's answer. Compare only the
facts. Ignore wording, language (German or English), length, and citation markers like [C1].

Grade:
- "correct": the system's answer contains every key fact of the reference answer (same numbers,
  deadlines, names) and nothing that contradicts it. Extra correct detail is fine.
- "partial": it contains some of the reference facts but misses a required part (for example only
  one of two requested items), or is right but so vague that a key number is missing.
- "wrong": it contradicts the reference, gives other numbers, says the information is not
  available, or answers a different question.

Reply with one JSON object and nothing else:
{"verdict": "correct" | "partial" | "wrong", "reason": "<one short sentence>"}"""

Complete = Callable[[str, str], str]  # (system, user) -> reply


# ---------------------------------------------------------------- judging


def judge_prompt(question: str, reference: str, answer: str) -> str:
    return f"Question: {question}\nReference answer: {reference}\nSystem answer: {answer}"


def parse_verdict(raw: str) -> tuple[str | None, str]:
    """(verdict, reason) from the judge's reply; verdict None if it cannot be read."""
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start : end + 1])
        verdict = str(data["verdict"]).strip().lower()
    except (ValueError, KeyError, TypeError):
        return None, raw.strip()[:200]
    if verdict not in LABELS:
        return None, raw.strip()[:200]
    return verdict, str(data.get("reason", "")).strip()


def judge_one(record: dict, complete: Complete) -> dict:
    """Add `verdict`, `reason` and `judged_by` to one answer record (see the rules above)."""
    out = {**record}
    if record["gold_answerable"]:
        if not record["answerable"]:
            return {**out, "verdict": "wrong", "reason": "abstained on an answerable question",
                    "judged_by": "rule"}  # fmt: skip
        prompt = judge_prompt(record["question"], record["gold_answer"], record["text"])
        verdict, reason = None, ""
        for _attempt in range(2):  # the model occasionally returns nothing: retry once
            verdict, reason = parse_verdict(complete(JUDGE_PROMPT, prompt))
            if verdict:
                break
        return {**out, "verdict": verdict or "unreadable", "reason": reason, "judged_by": "llm"}
    if not record["answerable"]:
        return {**out, "verdict": "correct", "reason": "abstained on an unanswerable question",
                "judged_by": "rule"}  # fmt: skip
    return {**out, "verdict": "review", "reason": "answered an unanswerable question",
            "judged_by": "rule"}  # fmt: skip


def pair_id(qid: str) -> str:
    """'f01-de' and 'f01-en' belong together."""
    return qid.rsplit("-", 1)[0]


# ---------------------------------------------------------------- statistics


def score_of(verdict: str) -> float:
    return SCORE.get(verdict, 0.0)  # 'review' and 'unreadable' count as wrong (strict)


def bootstrap_ci(
    items: Sequence[tuple[str, float]],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(mean, low, high) of the scores, resampling whole pairs: items are (pair_id, score)."""
    if not items:
        return float("nan"), float("nan"), float("nan")
    groups: dict[str, list[float]] = defaultdict(list)
    for pid, value in items:
        groups[pid].append(value)
    clusters = list(groups.values())
    mean = sum(v for _, v in items) / len(items)
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [rng.choice(clusters) for _ in clusters]
        total = sum(len(c) for c in sample)
        means.append(sum(sum(c) for c in sample) / total)
    means.sort()
    low = means[int((1 - level) / 2 * rounds)]
    high = means[min(rounds - 1, int((1 + level) / 2 * rounds))]
    return mean, low, high


def summarise(judged: Sequence[dict]) -> dict:
    """Headline numbers with 95% CIs, overall and by language."""

    def block(rows: Sequence[dict]) -> dict:
        items = [(pair_id(r["qid"]), 1.0 if r["verdict"] == "correct" else 0.0) for r in rows]
        mean, low, high = bootstrap_ci(items)
        return {"n": len(rows), "mean": mean, "ci_low": low, "ci_high": high}

    answerable = [r for r in judged if r["gold_answerable"]]
    unanswerable = [r for r in judged if not r["gold_answerable"]]
    out = {
        "answerable_correct": block(answerable),
        "answerable_correct_or_partial": block(
            [
                {**r, "verdict": "correct" if r["verdict"] == "partial" else r["verdict"]}
                for r in answerable
            ]
        ),  # fmt: skip
        "unanswerable_handled": block(unanswerable),
        "verdicts": dict(Counter(r["verdict"] for r in judged)),
    }
    for lang in ("de", "en"):
        out[f"answerable_correct_{lang}"] = block([r for r in answerable if r["lang"] == lang])
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two labelers, corrected for the agreement expected by chance."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement_report(pairs: Sequence[tuple[str, str]]) -> dict:
    """pairs = (human, judge). Three-way and 'correct vs not' agreement, plus kappa for both."""
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    binary_h = [h == "correct" for h in human]
    binary_j = [j == "correct" for j in judge]
    n = len(pairs)
    return {
        "n": n,
        "agreement": sum(h == j for h, j in pairs) / n if n else float("nan"),
        "kappa": cohen_kappa(human, judge),
        "agreement_binary": sum(x == y for x, y in zip(binary_h, binary_j, strict=True)) / n
        if n
        else float("nan"),
        "kappa_binary": cohen_kappa([str(x) for x in binary_h], [str(x) for x in binary_j]),
        "disagreements": [{"human": h, "judge": j} for h, j in pairs if h != j],
    }


# ---------------------------------------------------------------- human labelling sheet

SHEET_FIELDS = ["qid", "question", "reference_answer", "system_answer", "human_label"]
LABEL_ALIASES = {"c": "correct", "p": "partial", "w": "wrong"}


def pick_sample(judged: Sequence[dict], size: int, seed: int = SEED) -> list[dict]:
    """A random sample of the answers the judge actually graded (not the rule-based ones)."""
    candidates = [r for r in judged if r["judged_by"] == "llm"]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:size], key=lambda r: r["qid"])


def write_sheet(rows: Sequence[dict], path: str | Path) -> None:
    """The sheet deliberately omits the judge's verdict, so your labels are independent."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "qid": r["qid"],
                    "question": r["question"],
                    "reference_answer": r["gold_answer"],
                    "system_answer": r["text"],
                    "human_label": "",
                }
            )


def read_labels(path: str | Path) -> dict[str, str]:
    """qid -> 'correct' | 'partial' | 'wrong'. Accepts c/p/w. Blank rows are skipped."""
    labels: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("human_label") or "").strip().lower()
            label = LABEL_ALIASES.get(raw, raw)
            if not label:
                continue
            if label not in LABELS:
                raise ValueError(f"{row['qid']}: label must be c, p or w (got {raw!r})")
            labels[row["qid"]] = label
    return labels


# ---------------------------------------------------------------- command line


def format_summary(summary: dict) -> str:
    def line(name: str, key: str) -> str:
        b = summary[key]
        return (
            f"  {name:<46} {b['mean']:.2f}  [{b['ci_low']:.2f}, {b['ci_high']:.2f}]  (n={b['n']})"
        )

    return "\n".join(
        [
            "answers (95% bootstrap CI, resampled by question pair)",
            line("answerable: correct", "answerable_correct"),
            line("answerable: correct (German questions)", "answerable_correct_de"),
            line("answerable: correct (English questions)", "answerable_correct_en"),
            line("unanswerable: refused", "unanswerable_handled"),
            f"  verdicts: {summary['verdicts']}",
        ]
    )


def run_validation(labels_path: str, judged_path: str) -> int:
    labels = read_labels(labels_path)
    judged = {r["qid"]: r for r in json.loads(Path(judged_path).read_text("utf-8"))["answers"]}
    pairs = [(label, judged[qid]["verdict"]) for qid, label in labels.items() if qid in judged]
    if not pairs:
        print("no labelled rows found: fill the human_label column with c, p or w")
        return 2
    report = agreement_report(pairs)
    print(f"judge vs. your labels on {report['n']} answers")
    print(f"  agreement (3 classes)        {report['agreement']:.2f}   kappa {report['kappa']:.2f}")
    print(
        f"  agreement (correct vs not)   {report['agreement_binary']:.2f}   "
        f"kappa {report['kappa_binary']:.2f}"
    )
    for qid, label in labels.items():
        if qid in judged and judged[qid]["verdict"] != label:
            print(
                f"  disagree {qid}: you {label}, judge {judged[qid]['verdict']}"
                f" ({judged[qid]['reason']})"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade answers with an LLM judge")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="model that grades")
    parser.add_argument("--in-dir", default="results")
    parser.add_argument("--sample", type=int, help="write a sheet of N answers to label by hand")
    parser.add_argument("--sheet", default="data/eval/judge_labels.csv")
    parser.add_argument("--validate", metavar="CSV", help="compare the judge with your labels")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        print("The test split is locked. Use --final once, for the reported numbers.")
        return 2
    answers_path = Path(args.in_dir) / f"answers_{args.split}.json"
    judged_path = Path(args.in_dir) / f"judged_{args.split}.json"

    if args.validate:
        if not judged_path.exists():
            print(f"{judged_path} not found: run `python -m studybot.judge --split {args.split}`")
            return 2
        return run_validation(args.validate, str(judged_path))

    if args.sample:
        if not judged_path.exists():
            print(f"{judged_path} not found: run `python -m studybot.judge --split {args.split}`")
            return 2
        judged = json.loads(judged_path.read_text("utf-8"))["answers"]
        sample = pick_sample(judged, args.sample)
        write_sheet(sample, args.sheet)
        print(f"wrote {args.sheet} with {len(sample)} answers: fill human_label with c, p or w")
        return 0

    if not answers_path.exists():
        print(f"{answers_path} not found: run `python -m studybot.answer --batch` first")
        return 2
    from studybot.answer import AnthropicLLM

    llm = AnthropicLLM(args.judge, max_tokens=300)
    records = json.loads(answers_path.read_text("utf-8"))["answers"]
    judged = []
    for i, record in enumerate(records, start=1):
        result = judge_one(record, llm.complete)
        judged.append(result)
        print(f"[{i}/{len(records)}] {result['qid']}: {result['verdict']}")
    summary = summarise(judged)
    print("\n" + format_summary(summary))
    review = [r["qid"] for r in judged if r["verdict"] == "review"]
    if review:
        print(f"\nanswers to an unanswerable question (read these yourself): {review}")
    judged_path.write_text(
        json.dumps({"judge": args.judge, "summary": summary, "answers": judged},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )  # fmt: skip
    print(f"\nwrote {judged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
