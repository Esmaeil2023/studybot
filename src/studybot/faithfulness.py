"""Faithfulness (Phase 7): does the answer say only what its cited excerpts say?

    python -m studybot.faithfulness --split dev            grade results/answers_dev.json
    python -m studybot.faithfulness --split dev --sample 20    sheet for your own check
    python -m studybot.faithfulness --label                label the sheet in the terminal
    python -m studybot.faithfulness --validate             judge vs. your labels
    python -m studybot.faithfulness --stress --split dev   damaged answers must be flagged

The correctness judge (`judge.py`) asks "is the answer right?" by comparing it with the gold
answer. This one asks a different question that needs no gold answer: "is every statement in the
answer supported by the text it cites?". An answer can be faithful and wrong (the source says
something else than the gold answer) or right and unfaithful (true, but not in the excerpts).

Definition used here: the judge sees the answer and *only the excerpts the answer cites*. It
splits the answer into separate claims and marks each as supported or unsupported. This is
stricter than "supported by anything that was retrieved", and it is exactly what a user who clicks
a citation can check. Refusals are not graded (there is no claim to check).

Reported: the share of claims that are supported, and the share of answers in which *every* claim
is supported ("fully faithful"), with bootstrap confidence intervals over question pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from studybot.ingest import Chunk, load_chunks
from studybot.judge import DEFAULT_JUDGE, SEED, bootstrap_ci, cohen_kappa, pair_id, swap_number

Complete = Callable[[str, str], str]  # (system, user) -> reply

FAITH_PROMPT = """You check whether an answer is supported by the excerpts it cites.
You get numbered excerpts and an answer. Do this:
1. Split the answer into its separate factual claims (one fact each). Ignore citation markers and
   wording; treat a claim in German and in English the same way.
2. For each claim decide: "supported" if the excerpts state it or it follows directly from them
   (for example "four semesters" and "Regelstudienzeit: vier Fachsemester"), or "unsupported" if the
   excerpts do not say it, say something different, or the claim goes beyond them.
Use ONLY the excerpts. Do not use outside knowledge, even if the claim is true in reality.

Reply with one JSON object and nothing else:
{"claims": [{"claim": "<short>", "verdict": "supported" | "unsupported"}]}"""

_MARKER = re.compile(r"\s*\[C\d+\]")
LABELS = ("y", "n")  # y = every claim supported, n = at least one is not
UNSUPPORTED_SENTENCE = "Außerdem beträgt die Studiengebühr 500 Euro pro Semester."


# ---------------------------------------------------------------- grading


def clean_answer(text: str) -> str:
    return _MARKER.sub("", text).strip()


def faith_prompt(excerpts: Sequence[str], answer: str) -> str:
    numbered = "\n\n".join(f"[Excerpt {i}]\n{t}" for i, t in enumerate(excerpts, start=1))
    return f"{numbered}\n\nAnswer: {clean_answer(answer)}"


def parse_claims(raw: str) -> list[dict] | None:
    """[{"claim", "verdict"}] from the judge's reply, or None if it cannot be read."""
    start, end = raw.find("{"), raw.rfind("}")
    try:
        claims = json.loads(raw[start : end + 1])["claims"]
        parsed = [
            {"claim": str(c.get("claim", "")), "verdict": str(c["verdict"]).strip().lower()}
            for c in claims
        ]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not parsed or any(c["verdict"] not in ("supported", "unsupported") for c in parsed):
        return None
    return parsed


def excerpts_of(record: dict, by_id: dict[str, Chunk]) -> list[str]:
    """The texts the answer cites, in citation order."""
    return [by_id[cid].text for cid in record["citations"] if cid in by_id]


def faith_one(record: dict, by_id: dict[str, Chunk], complete: Complete) -> dict | None:
    """Faithfulness of one answer; None for refusals. Adds `claims`, `n_claims`, `n_supported`,
    `faithful` (all supported) and `status` ('graded', 'no_citation' or 'unreadable')."""
    if not record["answerable"]:
        return None
    out = {**record, "claims": [], "n_claims": 0, "n_supported": 0, "faithful": False}
    excerpts = excerpts_of(record, by_id)
    if not excerpts:  # an answer without a citation cannot be checked: counts as unfaithful
        return {**out, "status": "no_citation"}
    prompt = faith_prompt(excerpts, record["text"])
    claims = None
    for _attempt in range(2):  # the model occasionally returns nothing: retry once
        claims = parse_claims(complete(FAITH_PROMPT, prompt))
        if claims:
            break
    if not claims:
        return {**out, "status": "unreadable"}
    supported = sum(c["verdict"] == "supported" for c in claims)
    return {
        **out,
        "claims": claims,
        "n_claims": len(claims),
        "n_supported": supported,
        "faithful": supported == len(claims),
        "status": "graded",
    }


def summarise(results: Sequence[dict]) -> dict:
    """Claim-level and answer-level faithfulness with 95% CIs (resampled by question pair).
    Ungradable answers (no citation, unreadable reply) count as unfaithful and are listed."""
    answers = [r for r in results if r is not None]
    if not answers:
        return {"n_answers": 0}
    ci = bootstrap_ci([(pair_id(r["qid"]), 1.0 if r["faithful"] else 0.0) for r in answers])
    total = sum(r["n_claims"] for r in answers)
    supported = sum(r["n_supported"] for r in answers)
    return {
        "n_answers": len(answers),
        "n_claims": total,
        "claim_support": supported / total if total else float("nan"),
        "fully_faithful": ci[0],
        "fully_faithful_ci": [ci[1], ci[2]],
        "not_graded": [r["qid"] for r in answers if r["status"] != "graded"],
        "unfaithful": [r["qid"] for r in answers if r["status"] == "graded" and not r["faithful"]],
    }


# ---------------------------------------------------------------- stress test


def make_stress_cases(graded: Sequence[dict]) -> list[dict]:
    """Damaged copies of answers judged faithful. Each must now be flagged unfaithful."""
    cases = []
    for r in sorted((g for g in graded if g and g["faithful"]), key=lambda g: g["qid"]):
        text = r["text"]
        cases.append({**r, "kind": "invented_sentence", "text": f"{text} {UNSUPPORTED_SENTENCE}"})
        changed = swap_number(clean_answer(text), clean_answer(text))
        if changed:
            cases.append({**r, "kind": "changed_number", "text": changed})
    return cases


def run_stress(cases: Sequence[dict], by_id: dict[str, Chunk], complete: Complete) -> list[dict]:
    results = []
    for case in cases:
        graded = faith_one(case, by_id, complete)
        results.append({"qid": case["qid"], "kind": case["kind"], "text": case["text"],
                        "faithful": graded["faithful"], "status": graded["status"]})  # fmt: skip
    return results


def summarise_stress(results: Sequence[dict]) -> dict:
    kinds = {}
    for kind in sorted({r["kind"] for r in results}):
        rows = [r for r in results if r["kind"] == kind]
        caught = sum(not r["faithful"] for r in rows)
        kinds[kind] = {"n": len(rows), "caught": caught}
    n = len(results)
    caught = sum(not r["faithful"] for r in results)
    return {
        "kinds": kinds,
        "n": n,
        "caught": caught,
        "missed": [{"qid": r["qid"], "kind": r["kind"]} for r in results if r["faithful"]],
    }


# ---------------------------------------------------------------- human check

SHEET_FIELDS = ["qid", "answer", "excerpts", "human_label"]


def pick_sample(results: Sequence[dict], size: int, seed: int = SEED) -> list[dict]:
    import random

    graded = [r for r in results if r and r["status"] == "graded"]
    rng = random.Random(seed)
    rng.shuffle(graded)
    return sorted(graded[:size], key=lambda r: r["qid"])


def write_sheet(rows: Sequence[dict], by_id: dict[str, Chunk], path: str | Path) -> None:
    """The sheet omits the judge's verdict, so the check is independent."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for r in rows:
            excerpts = "\n\n".join(
                f"[Excerpt {i}] {t}" for i, t in enumerate(excerpts_of(r, by_id), start=1)
            )
            writer.writerow(
                {
                    "qid": r["qid"],
                    "answer": clean_answer(r["text"]),
                    "excerpts": excerpts,
                    "human_label": "",
                }  # fmt: skip
            )


def read_labels(path: str | Path) -> dict[str, str]:
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("human_label") or "").strip().lower()
            if not label:
                continue
            if label not in LABELS:
                raise ValueError(f"{row['qid']}: label must be y or n (got {label!r})")
            labels[row["qid"]] = label
    return labels


def label_interactively(
    path: str | Path, ask: Callable[[str], str] = input, relabel: Sequence[str] = ()
) -> int:
    """y = every statement of the answer is supported by the excerpts, n = at least one is not,
    s = skip, q = quit. Saved after every answer."""
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
        print(f"EXCERPTS:\n{row['excerpts']}\n")
        print(f"ANSWER: {row['answer']}")
        while True:
            choice = ask(
                "every statement supported by the excerpts? y / n / s = skip / q = quit > "
            )
            choice = choice.strip().lower()
            if choice in ("y", "n", "s", "q"):
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        row["human_label"] = choice
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            writer.writerows(table)
    left = sum(1 for row in table if not (row.get("human_label") or "").strip())
    print(f"\n{len(table) - left} labelled, {left} left")
    return 0


def agreement(pairs: Sequence[tuple[str, str]]) -> dict:
    """pairs = (human, judge) with labels y / n."""
    n = len(pairs)
    human = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    return {
        "n": n,
        "agreement": sum(h == j for h, j in pairs) / n if n else float("nan"),
        "kappa": cohen_kappa(human, judge),
        "disagreements": [{"human": h, "judge": j} for h, j in pairs if h != j],
    }


# ---------------------------------------------------------------- command line


def format_summary(s: dict) -> str:
    low, high = s["fully_faithful_ci"]
    return "\n".join(
        [
            "faithfulness (answers only, refusals not graded; CI resampled by question pair)",
            f"  answers graded                              {s['n_answers']}",
            f"  claims supported                            {s['claim_support']:.2f}"
            f"  ({s['n_claims']} claims)",
            f"  answers with every claim supported          {s['fully_faithful']:.2f}"
            f"  [{low:.2f}, {high:.2f}]",
            f"  not gradable (no citation / unreadable)     {len(s['not_graded'])}",
            f"  answers with an unsupported claim           {s['unfaithful']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Is every statement supported by its citations?")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--in-dir", default="results")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--sheet", default="data/eval/faithfulness_labels.csv")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--relabel", nargs="+", default=[], metavar="QID", help="label these again")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        print("The test split is locked. Use --final once, for the reported numbers.")
        return 2
    if args.label or args.relabel:
        if not Path(args.sheet).exists():
            print(f"{args.sheet} not found: run steps 1 and 2 first (grade, then --sample 20)")
            return 2
        return label_interactively(args.sheet, relabel=args.relabel)

    graded_path = Path(args.in_dir) / f"faithfulness_{args.split}.json"
    answers_path = Path(args.in_dir) / f"answers_{args.split}.json"
    by_id = {c.chunk_id: c for c in load_chunks(args.chunks)} if Path(args.chunks).exists() else {}

    if args.validate:
        if not graded_path.exists():
            print(f"{graded_path} not found: run `python -m studybot.faithfulness` first")
            return 2
        judged = {r["qid"]: r for r in json.loads(graded_path.read_text("utf-8"))["answers"]}
        labels = read_labels(args.sheet)
        pairs = [
            (label, "y" if judged[qid]["faithful"] else "n")
            for qid, label in labels.items()
            if qid in judged
        ]
        if not pairs:
            print("no labelled rows: run --label first")
            return 2
        report = agreement(pairs)
        print(f"judge vs. your labels on {report['n']} answers")
        print(f"  agreement {report['agreement']:.2f}   kappa {report['kappa']:.2f}")
        for qid, label in labels.items():
            if qid in judged and ("y" if judged[qid]["faithful"] else "n") != label:
                print(f"  disagree {qid}: you {label}, judge {judged[qid]['claims']}")
        return 0

    if args.sample:
        if not graded_path.exists():
            print(f"{graded_path} not found: run `python -m studybot.faithfulness` first")
            return 2
        answers = json.loads(graded_path.read_text("utf-8"))["answers"]
        sample = pick_sample(answers, args.sample)
        write_sheet(sample, by_id, args.sheet)
        print(f"wrote {args.sheet} with {len(sample)} answers: run --label")
        return 0

    if not answers_path.exists() or not by_id:
        print("missing answers or chunks: run `python -m studybot.answer --batch` first")
        return 2
    from studybot.answer import AnthropicLLM

    llm = AnthropicLLM(args.judge, max_tokens=800)

    if args.stress:
        if not graded_path.exists():
            print(f"{graded_path} not found: run `python -m studybot.faithfulness` first")
            return 2
        graded = json.loads(graded_path.read_text("utf-8"))["answers"]
        results = run_stress(make_stress_cases(graded), by_id, llm.complete)
        summary = summarise_stress(results)
        print("stress test: damaged answers the judge must flag as unfaithful")
        for kind, k in summary["kinds"].items():
            print(f"  {kind:<20} flagged {k['caught']}/{k['n']}")
        print(f"  {'all':<20} flagged {summary['caught']}/{summary['n']}")
        for m in summary["missed"]:
            print(f"  MISSED {m['qid']} [{m['kind']}]")
        out = Path(args.in_dir) / f"faithfulness_stress_{args.split}.json"
        out.write_text(
            json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
        return 0

    records = json.loads(answers_path.read_text("utf-8"))["answers"]
    results = []
    for i, record in enumerate(records, start=1):
        result = faith_one(record, by_id, llm.complete)
        if result is None:
            continue
        results.append(result)
        print(f"[{i}/{len(records)}] {result['qid']}: {result['n_supported']}/{result['n_claims']}")
    summary = summarise(results)
    print("\n" + format_summary(summary))
    graded_path.write_text(
        json.dumps({"judge": args.judge, "summary": summary, "answers": results},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )  # fmt: skip
    print(f"\nwrote {graded_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
