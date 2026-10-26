"""Build the evaluation set (Phase 2): questions, gold evidence mapped to chunks, dev/test split.

    python -m studybot.evalset --review data/eval/review.md

Input (hand-written, reviewed by a human):
    data/eval/facts.yaml         answerable questions, DE + EN, with verbatim evidence spans
    data/eval/unanswerable.yaml  questions the documents cannot answer
    data/processed/chunks.jsonl  the chunks from Phase 1

Output:
    data/eval/questions.jsonl    one row per question (a fact gives two rows: DE and EN)

Why spans and not chunk ids: if the chunking changes later, the questions stay valid. The
evidence is the *text* that answers the question; this script finds the chunks that contain it.
A question is retrieved correctly when at least one gold chunk of each evidence span is found.

Why the split is by `pair_id`: the DE and EN version of a question must land in the same split,
otherwise the dev set would leak the test question in the other language.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from studybot.ingest import Chunk, load_chunks

DEV_FRACTION = 0.4
SPLIT_SEED = 13
MAX_CHUNKS_PER_SPAN = 3  # a span found in more chunks than this is too generic to be evidence
FUZZY_MIN_COVERAGE = 0.8  # share of the span's word trigrams that must occur in the chunk
LANGS = ("de", "en")

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_QUOTES = {
    **dict.fromkeys(map(ord, "„“”‟«»"), '"'),
    **dict.fromkeys(map(ord, "‘’‚‛"), "'"),
}


# ---------------------------------------------------------------- text matching


def normalize(text: str) -> str:
    """Make two spellings of one text comparable: unicode, dashes, quotes, spaces, case."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip().casefold()


_WORD_RE = re.compile(r"\w+(?:-\w+)*")


def _words(text: str) -> list[str]:
    """Words without punctuation, so '120 ECTS-Punkte.' and '120 ECTS-Punkte' are the same."""
    return _WORD_RE.findall(normalize(text))


def _trigrams(words: list[str]) -> set[tuple[str, ...]]:
    if len(words) < 3:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def span_coverage(span: str, chunk_text: str) -> float:
    """Share of the span's word trigrams that also occur in the chunk (1.0 = same words)."""
    span_grams = _trigrams(_words(span))
    if not span_grams:
        return 0.0
    return len(span_grams & _trigrams(_words(chunk_text))) / len(span_grams)


@dataclass
class SpanMatch:
    chunk_ids: list[str]
    method: str  # "exact" | "fuzzy" | "none"
    best_coverage: float = 0.0


def find_span(span: str, chunks: list[Chunk], doc_id: str, section: str | None = None) -> SpanMatch:
    """All chunks of `doc_id` (optionally one section) that contain `span`.

    Exact (after normalisation) first; if no chunk has it exactly, the single best chunk whose
    trigram coverage is at least FUZZY_MIN_COVERAGE.
    """
    pool = [
        c
        for c in chunks
        if c.doc_id == doc_id and (section is None or normalize(c.section) == normalize(section))
    ]
    needle = normalize(span)
    exact = [c.chunk_id for c in pool if needle in normalize(c.text)]
    if exact:
        return SpanMatch(exact, "exact", 1.0)
    scored = sorted(((span_coverage(span, c.text), c.chunk_id) for c in pool), reverse=True)
    if scored and scored[0][0] >= FUZZY_MIN_COVERAGE:
        return SpanMatch([scored[0][1]], "fuzzy", scored[0][0])
    return SpanMatch([], "none", scored[0][0] if scored else 0.0)


# ---------------------------------------------------------------- loading and building


def load_source(path: str | Path, key: str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(data[key])


def map_evidence(fact: dict, chunks: list[Chunk]) -> list[dict]:
    """Evidence spans of one fact, each with the ids of the chunks that contain it."""
    mapped = []
    for ev in fact["evidence"]:
        match = find_span(ev["span"], chunks, ev["doc"], ev.get("section"))
        mapped.append(
            {
                "doc": ev["doc"],
                "section": ev.get("section"),
                "span": ev["span"],
                "chunk_ids": match.chunk_ids,
                "match": match.method,
            }
        )
    return mapped


def assign_splits(
    facts: list[dict], unanswerable: list[dict], seed: int = SPLIT_SEED
) -> dict[str, str]:
    """pair_id -> "dev" | "test". Stratified: facts by document, unanswerable by kind."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for fact in facts:
        groups.setdefault(f"fact:{fact['evidence'][0]['doc']}", []).append(fact["id"])
    for item in unanswerable:
        groups.setdefault(f"unanswerable:{item['kind']}", []).append(item["id"])

    split: dict[str, str] = {}
    for name in sorted(groups):
        ids = sorted(groups[name])
        rng.shuffle(ids)
        n_dev = round(len(ids) * DEV_FRACTION)
        for i, pair_id in enumerate(ids):
            split[pair_id] = "dev" if i < n_dev else "test"
    return split


def build_questions(
    facts: list[dict], unanswerable: list[dict], chunks: list[Chunk], seed: int = SPLIT_SEED
) -> list[dict]:
    split = assign_splits(facts, unanswerable, seed)
    rows: list[dict] = []
    for fact in facts:
        evidence = map_evidence(fact, chunks)
        gold = list(dict.fromkeys(cid for ev in evidence for cid in ev["chunk_ids"]))
        for lang in LANGS:
            rows.append(
                {
                    "qid": f"{fact['id']}-{lang}",
                    "pair_id": fact["id"],
                    "lang": lang,
                    "question": fact[f"q_{lang}"],
                    "answerable": True,
                    "answer": fact[f"a_{lang}"],
                    "category": fact["category"],
                    "split": split[fact["id"]],
                    "evidence": evidence,
                    "gold_chunk_ids": gold,
                }
            )
    for item in unanswerable:
        for lang in LANGS:
            rows.append(
                {
                    "qid": f"{item['id']}-{lang}",
                    "pair_id": item["id"],
                    "lang": lang,
                    "question": item[f"q_{lang}"],
                    "answerable": False,
                    "answer": None,
                    "category": item["kind"],
                    "split": split[item["id"]],
                    "evidence": [],
                    "gold_chunk_ids": [],
                    "why": item["why"],
                }
            )
    return rows


# ---------------------------------------------------------------- validation


def validate(
    facts: list[dict], unanswerable: list[dict], rows: list[dict], chunks: list[Chunk]
) -> list[str]:
    """Everything that would make the evaluation set untrustworthy. Empty list = fine."""
    problems: list[str] = []

    ids = [f["id"] for f in facts] + [u["id"] for u in unanswerable]
    problems += [f"duplicate id {i}" for i, n in Counter(ids).items() if n > 1]

    for fact in facts:
        for field in ("q_de", "q_en", "a_de", "a_en", "category", "evidence"):
            if not fact.get(field):
                problems.append(f"{fact['id']}: missing {field}")
    for item in unanswerable:
        for field in ("q_de", "q_en", "kind", "why"):
            if not item.get(field):
                problems.append(f"{item['id']}: missing {field}")

    questions = [r["question"].strip().casefold() for r in rows]
    problems += [
        f"duplicate question text: {q[:60]}" for q, n in Counter(questions).items() if n > 1
    ]

    for row in rows:
        for ev in row["evidence"]:
            if not ev["chunk_ids"]:
                problems.append(
                    f"{row['qid']}: evidence not found in the chunks: {ev['span'][:70]!r}"
                )
            elif len(ev["chunk_ids"]) > MAX_CHUNKS_PER_SPAN:
                problems.append(
                    f"{row['qid']}: evidence occurs in {len(ev['chunk_ids'])} chunks, "
                    f"make it more specific: {ev['span'][:70]!r}"
                )
            elif ev["match"] == "fuzzy":
                problems.append(f"{row['qid']}: evidence only matched fuzzily: {ev['span'][:70]!r}")
        if row["answerable"] and not row["gold_chunk_ids"]:
            problems.append(f"{row['qid']}: no gold chunk")

    corpus = [(c.chunk_id, normalize(c.text)) for c in chunks]
    for item in unanswerable:
        for pattern in item.get("absent", []):
            regex = re.compile(pattern, re.IGNORECASE)
            hits = [cid for cid, text in corpus if regex.search(text)]
            if hits:
                problems.append(f"{item['id']}: '{pattern}' occurs in {hits[:3]}, not unanswerable")

    by_pair: dict[str, set[str]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], set()).add(row["split"])
    problems += [
        f"{p}: DE and EN are in different splits" for p, s in by_pair.items() if len(s) > 1
    ]
    for lang in LANGS:
        n = sum(1 for r in rows if r["lang"] == lang and r["answerable"])
        m = sum(1 for r in rows if r["lang"] == lang and not r["answerable"])
        if (n, m) != (len(facts), len(unanswerable)):
            problems.append(
                f"{lang}: {n} answerable / {m} unanswerable rows, expected a full pair set"
            )
    return problems


def summary(rows: list[dict]) -> list[str]:
    lines = []
    for split in ("dev", "test"):
        mine = [r for r in rows if r["split"] == split]
        pairs = len({r["pair_id"] for r in mine})
        ans = len({r["pair_id"] for r in mine if r["answerable"]})
        lines.append(
            f"{split}: {pairs} pairs ({ans} answerable, {pairs - ans} unanswerable), "
            f"{len(mine)} questions"
        )
    docs = Counter(r["evidence"][0]["doc"] for r in rows if r["answerable"] and r["lang"] == "en")
    lines.append(
        "answerable pairs per document: " + ", ".join(f"{d}={n}" for d, n in sorted(docs.items()))
    )
    cats = Counter(r["category"] for r in rows if r["lang"] == "en")
    lines.append("categories: " + ", ".join(f"{c}={n}" for c, n in sorted(cats.items())))
    multi = sum(1 for r in rows if r["lang"] == "en" and len(r["gold_chunk_ids"]) > 1)
    lines.append(f"pairs with more than one gold chunk: {multi}")
    return lines


# ---------------------------------------------------------------- review sheet


def review_markdown(rows: list[dict], chunks: list[Chunk]) -> str:
    """A sheet for a human to check every answer against the original PDF."""
    by_id = {c.chunk_id: c for c in chunks}
    out = [
        "# Review sheet for the evaluation set",
        "",
        "For every question: open the PDF at the given page, find the quoted evidence and check",
        "that (1) the evidence says what the answer says, (2) the DE and EN question mean the",
        "same,",
        "(3) the question is clear without extra context. Mark mistakes and send them back.",
        "",
    ]
    pairs = sorted({r["pair_id"] for r in rows})
    for pair_id in pairs:
        de = next(r for r in rows if r["pair_id"] == pair_id and r["lang"] == "de")
        en = next(r for r in rows if r["pair_id"] == pair_id and r["lang"] == "en")
        out.append(f"## {pair_id}  ({de['split']}, {de['category']})")
        out.append(f"- DE: {de['question']}")
        out.append(f"- EN: {en['question']}")
        if de["answerable"]:
            out.append(f"- Answer DE: {de['answer']}")
            out.append(f"- Answer EN: {en['answer']}")
            for ev in de["evidence"]:
                places = []
                for cid in ev["chunk_ids"]:
                    c = by_id[cid]
                    places.append(f"{c.chunk_id} p.{c.page_start}-{c.page_end}")
                out.append(f"- Evidence ({ev['match']}; {', '.join(places)}): “{ev['span']}”")
        else:
            out.append(f"- Expected: no answer in the documents. Why: {de['why']}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- command line


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path: str | Path, split: str | None = None) -> list[dict]:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    return [r for r in rows if split is None or r["split"] == split]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/eval/questions.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--facts", default="data/eval/facts.yaml")
    parser.add_argument("--unanswerable", default="data/eval/unanswerable.yaml")
    parser.add_argument("--out", default="data/eval/questions.jsonl")
    parser.add_argument("--review", help="also write a markdown review sheet to this path")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args(argv)

    if not Path(args.chunks).exists():
        print(f"{args.chunks} not found; run `make chunks` first")
        return 2
    chunks = load_chunks(args.chunks)
    facts = load_source(args.facts, "facts")
    unanswerable = load_source(args.unanswerable, "unanswerable")

    rows = build_questions(facts, unanswerable, chunks, args.seed)
    problems = validate(facts, unanswerable, rows, chunks)
    for line in summary(rows):
        print(line)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        print("\n".join(f"  - {p}" for p in problems))
        return 1

    write_jsonl(rows, Path(args.out))
    print(f"\nwrote {len(rows)} questions to {args.out}")
    if args.review:
        Path(args.review).write_text(review_markdown(rows, chunks), encoding="utf-8")
        print(f"wrote review sheet to {args.review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
