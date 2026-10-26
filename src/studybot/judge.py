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
import re
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


# ---------------------------------------------------------------- stress test
#
# The human sample contains almost no wrong answers, so it cannot show whether the judge
# wrongly accepts a bad one. The stress test builds bad answers on purpose: take an answer the
# judge called correct, damage one key fact, and check that the judge no longer says "correct".

_NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
_REFERENCE_NUMBER_PREFIXES = ("§ ", "§§ ", "Abs. ", "Absatz ", "Satz ", "paragraph ", "Nr. ")
_NUMBER_WORDS = {
    "four": "six", "six": "four", "five": "three", "three": "five", "two": "one", "one": "two",
    "vier": "sechs", "sechs": "vier", "fünf": "drei", "drei": "fünf", "einmal": "zweimal",
    "half": "a third", "halb": "ein Drittel", "halbe": "ein Drittel",
}  # fmt: skip
_NAME = re.compile(r"Prof\. Dr\. [A-ZÄÖÜ][\w-]+ [A-ZÄÖÜ][\w-]+")
FAKE_NAME = "Prof. Dr. Anna Berger"
UNAVAILABLE = "The excerpts do not state this."


def _changed(token: str) -> str:
    """A different number with a similar shape: 6.5 -> 7.5, 15 -> 25, 5 -> 7."""
    decimals = len(token.replace(",", ".").split(".")[1]) if re.search(r"[.,]", token) else 0
    value = float(token.replace(",", "."))
    new = value + 1 if decimals else (value + 2 if value < 10 else value + 10)
    return f"{new:.{decimals}f}".replace(".", "," if "," in token else ".")


def swap_number(answer: str, reference: str) -> str | None:
    """Change the first key number (one the reference also has), everywhere it appears; else None.

    Paragraph and sentence numbers ('§ 25 Abs. 1') are left alone."""
    ref_numbers = {m.group() for m in _NUMBER.finditer(reference)}
    target = None
    for m in _NUMBER.finditer(answer):
        if m.group() in ref_numbers and not answer[: m.start()].endswith(
            _REFERENCE_NUMBER_PREFIXES
        ):
            target = m.group()
            break
    if target is None:
        return None

    def replace(m: re.Match) -> str:
        if m.group() != target or answer[: m.start()].endswith(_REFERENCE_NUMBER_PREFIXES):
            return m.group()
        return _changed(target)

    return _NUMBER.sub(replace, answer)


def swap_number_word(answer: str, reference: str) -> str | None:
    """Change a number word ('four' -> 'six') that the reference also contains, else None."""
    for word, other in _NUMBER_WORDS.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pattern.search(reference):
            hit = pattern.search(answer)
            if hit:
                new = other.capitalize() if hit.group()[0].isupper() else other
                return answer[: hit.start()] + new + answer[hit.end() :]
    return None


def swap_name(answer: str, reference: str) -> str | None:
    """Replace a professor's name that the reference also names, else None."""
    ref_names = set(_NAME.findall(reference))
    for m in _NAME.finditer(answer):
        if m.group() in ref_names:
            return answer[: m.start()] + FAKE_NAME + answer[m.end() :]
    return None


def make_stress_cases(judged: Sequence[dict], seed: int = SEED) -> list[dict]:
    """Damaged copies of answers the judge graded 'correct'. Every case should now fail."""
    good = [
        r for r in judged
        if r["gold_answerable"] and r["judged_by"] == "llm" and r["verdict"] == "correct"
    ]  # fmt: skip
    good.sort(key=lambda r: r["qid"])
    cases = []

    def add(row: dict, kind: str, answer: str | None) -> None:
        if answer:
            cases.append(
                {"qid": row["qid"], "kind": kind, "question": row["question"],
                 "reference": row["gold_answer"], "answer": answer, "original": row["text"]}
            )  # fmt: skip

    for row in good:
        ref, text = row["gold_answer"], row["text"]
        add(row, "changed_number", swap_number(text, ref))
        add(row, "changed_number_word", swap_number_word(text, ref))
        add(row, "changed_name", swap_name(text, ref))
        add(row, "not_stated", UNAVAILABLE)
    rng = random.Random(seed)  # an answer to a different question
    for row in good:
        others = [r for r in good if pair_id(r["qid"]) != pair_id(row["qid"])]
        if others:
            add(row, "other_answer", rng.choice(others)["text"])
    return cases


def run_stress(cases: Sequence[dict], complete: Complete) -> list[dict]:
    results = []
    for case in cases:
        prompt = judge_prompt(case["question"], case["reference"], case["answer"])
        verdict = None
        for _attempt in range(2):
            verdict, reason = parse_verdict(complete(JUDGE_PROMPT, prompt))
            if verdict:
                break
        results.append({**case, "verdict": verdict or "unreadable", "reason": reason})
    return results


def summarise_stress(results: Sequence[dict]) -> dict:
    """Per kind of damage: how many cases, and how many the judge refused to call correct."""
    kinds: dict[str, dict] = {}
    for kind in sorted({r["kind"] for r in results}):
        rows = [r for r in results if r["kind"] == kind]
        caught = sum(r["verdict"] != "correct" for r in rows)
        kinds[kind] = {"n": len(rows), "caught": caught, "rate": caught / len(rows)}
    total = len(results)
    caught_all = sum(r["verdict"] != "correct" for r in results)
    return {
        "kinds": kinds,
        "n": total,
        "caught": caught_all,
        "rate": caught_all / total if total else float("nan"),
        "missed": [
            {"qid": r["qid"], "kind": r["kind"], "answer": r["answer"]}
            for r in results
            if r["verdict"] == "correct"
        ],
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


def label_interactively(
    path: str | Path, ask: Callable[[str], str] = input, relabel: Sequence[str] = ()
) -> int:
    """Walk through the sheet in the terminal; saves after every answer, so you can stop anytime.
    `relabel` lists qids to ask again even though they already have a label.
    Answers: c = correct, p = partial, w = wrong, s = skip for now, q = quit."""
    with open(path, newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    todo = [
        i
        for i, row in enumerate(table)
        if row["qid"] in relabel or not (row.get("human_label") or "").strip()
    ]
    for position, i in enumerate(todo, start=1):
        row = table[i]
        print(f"\n--- {position}/{len(todo)}  {row['qid']} ---")
        print(f"QUESTION:  {row['question']}")
        print(f"REFERENCE: {row['reference_answer']}")
        print(f"SYSTEM:    {row['system_answer']}")
        while True:
            choice = (
                ask("c = correct, p = partial, w = wrong, s = skip, q = quit > ").strip().lower()
            )
            if choice in LABEL_ALIASES or choice in ("s", "q"):
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        row["human_label"] = LABEL_ALIASES[choice]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            writer.writerows(table)
    left = sum(1 for row in table if not (row.get("human_label") or "").strip())
    print(f"\n{len(table) - left} labelled, {left} left")
    return 0


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
    parser.add_argument("--stress", action="store_true", help="damage correct answers, re-judge")
    parser.add_argument("--label", action="store_true", help="label the sheet in the terminal")
    parser.add_argument("--relabel", default="", help="with --label: qids to label again (a,b)")
    parser.add_argument("--validate", metavar="CSV", help="compare the judge with your labels")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        print("The test split is locked. Use --final once, for the reported numbers.")
        return 2
    answers_path = Path(args.in_dir) / f"answers_{args.split}.json"
    judged_path = Path(args.in_dir) / f"judged_{args.split}.json"

    if args.stress:
        if not judged_path.exists():
            print(f"{judged_path} not found: run `python -m studybot.judge --split {args.split}`")
            return 2
        from studybot.answer import AnthropicLLM

        llm = AnthropicLLM(args.judge, max_tokens=300)
        judged = json.loads(judged_path.read_text("utf-8"))["answers"]
        results = run_stress(make_stress_cases(judged), llm.complete)
        summary = summarise_stress(results)
        print("stress test: damaged answers the judge should not call correct")
        for kind, k in summary["kinds"].items():
            print(f"  {kind:<20} caught {k['caught']}/{k['n']}")
        print(f"  {'all':<20} caught {summary['caught']}/{summary['n']}  ({summary['rate']:.2f})")
        for m in summary["missed"]:
            print(f"  MISSED {m['qid']} [{m['kind']}]: {m['answer'][:120]}")
        out = Path(args.in_dir) / f"stress_{args.split}.json"
        out.write_text(
            json.dumps({"judge": args.judge, "summary": summary, "cases": results},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )  # fmt: skip
        print(f"\nwrote {out}")
        return 0

    if args.label:
        if not Path(args.sheet).exists():
            print(f"{args.sheet} not found: run with --sample 24 first")
            return 2
        return label_interactively(args.sheet, relabel=[q for q in args.relabel.split(",") if q])

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
