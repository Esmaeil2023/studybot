"""Look at your chunks before you build anything on them.

    python -m studybot.inspect_chunks --chunks data/processed/chunks.jsonl -n 10

Prints per-document statistics, a length histogram, automatic sanity checks and N random chunks
to read by eye. The point of this script is that the checks tell you WHERE to look, but only
reading the chunks tells you whether they are good.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict

from studybot.ingest import Chunk, load_chunks

SHORT_WORDS = 15  # a chunk shorter than this is suspicious
LONG_FACTOR = 1.25  # a chunk longer than max_words * this is suspicious

# ingest turns table-of-contents dot leaders into "…" before a page number ("§ 1 Titel … 3")
_LEADER_RE = re.compile(r"…\s*\d{1,3}\b|\.{4,}")
_ELLIPSIS_WORDS = r"(?:und|oder|bzw|sowie|beziehungsweise|bis|noch|als|wie|and|or)"
_LEFTOVER_HYPHEN_RE = re.compile(rf"\b\w+-\s+(?!{_ELLIPSIS_WORDS}\b)[a-zäöüß]\w*")


def length_histogram(chunks: list[Chunk], bucket: int = 50) -> list[tuple[str, int]]:
    """Counts of chunks per word-count bucket, e.g. ('100-149', 12)."""
    counts: Counter = Counter(c.n_words // bucket for c in chunks)
    return (
        [(f"{b * bucket}-{b * bucket + bucket - 1}", counts[b]) for b in range(max(counts) + 1)]
        if counts
        else []
    )


def numbering_gaps(chunks: list[Chunk]) -> dict[str, list[int]]:
    """For each document: the paragraph numbers missing between its first and last "§ N".

    A gap (e.g. § 4 and § 6 found, § 5 not) usually means a heading was not recognised.
    Sections like "§ 12a" count as 12.
    """
    numbers: dict[str, set[int]] = defaultdict(set)
    for c in chunks:
        match = re.fullmatch(r"§ (\d+)[a-z]?", c.section)
        if match:
            numbers[c.doc_id].add(int(match.group(1)))
    return {
        doc_id: [n for n in range(min(nums), max(nums) + 1) if n not in nums]
        for doc_id, nums in numbers.items()
        if nums
    }


def repeated_labels(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Section labels that occur in more than one separate run, per document.

    Normal for Änderungssatzungen; for a normal regulation it usually means a cross-reference
    was mistaken for a heading.
    """
    result: dict[str, list[str]] = {}
    for doc_id in dict.fromkeys(c.doc_id for c in chunks):
        runs: list[str] = []
        for c in (c for c in chunks if c.doc_id == doc_id):
            if not runs or runs[-1] != c.section:
                runs.append(c.section)
        repeated = sorted({label for label in runs if runs.count(label) > 1})
        if repeated:
            result[doc_id] = repeated
    return result


def find_warnings(chunks: list[Chunk], max_words: int) -> list[str]:
    warnings: list[str] = []
    short = [c.chunk_id for c in chunks if c.n_words < SHORT_WORDS]
    long_ = [c.chunk_id for c in chunks if c.n_words > max_words * LONG_FACTOR]
    leaders = [c.chunk_id for c in chunks if _LEADER_RE.search(c.text)]
    hyphens = [c.chunk_id for c in chunks if _LEFTOVER_HYPHEN_RE.search(c.text)]
    duplicates = [t for t, n in Counter(c.text for c in chunks).items() if n > 1]

    if short:
        warnings.append(f"{len(short)} chunks under {SHORT_WORDS} words, e.g. {short[:5]}")
    if long_:
        warnings.append(
            f"{len(long_)} chunks over {int(max_words * LONG_FACTOR)} words: {long_[:5]}"
        )
    if leaders:
        warnings.append(
            f"{len(leaders)} chunks with table-of-contents leaders ('…'): {leaders[:5]}"
        )
    if hyphens:
        warnings.append(
            f"{len(hyphens)} chunks with a possibly un-joined hyphenation: {hyphens[:5]}"
        )
    if duplicates:
        warnings.append(f"{len(duplicates)} texts occur more than once")
    for doc_id, missing in numbering_gaps(chunks).items():
        if missing:
            warnings.append(f"{doc_id}: no section found for § {missing}")
    for doc_id, labels in repeated_labels(chunks).items():
        warnings.append(f"{doc_id}: section labels in more than one place: {labels}")
    return warnings


def print_report(chunks: list[Chunk], n_samples: int, seed: int, max_words: int) -> None:
    print("== Documents ==")
    for doc_id in dict.fromkeys(c.doc_id for c in chunks):
        mine = [c for c in chunks if c.doc_id == doc_id]
        sections = list(dict.fromkeys(c.section for c in mine))
        words = sum(c.n_words for c in mine)
        print(f"{doc_id}: {len(mine)} chunks, {len(sections)} sections, {words} words")
        print(f"    first sections: {', '.join(sections[:8])}")

    print("\n== Chunk length (words) ==")
    for label, count in length_histogram(chunks):
        print(f"{label:>9} | {'#' * min(count, 60)} {count}")

    print("\n== Checks ==")
    warnings = find_warnings(chunks, max_words)
    print("\n".join(f"WARNING: {w}" for w in warnings) if warnings else "no problems found")

    print(f"\n== {n_samples} random chunks (seed {seed}) ==")
    rng = random.Random(seed)
    for c in rng.sample(chunks, min(n_samples, len(chunks))):
        print(f"\n--- {c.chunk_id}  (pages {c.page_start}-{c.page_end}, {c.n_words} words)")
        print(c.header)
        for line in c.text.split("\n"):
            print(textwrap.fill(line, width=100))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect chunks.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--doc", help="only this doc_id")
    parser.add_argument("-n", type=int, default=10, help="how many random chunks to print")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=350, help="max_words used when chunking")
    args = parser.parse_args(argv)

    chunks = load_chunks(args.chunks)
    if args.doc:
        chunks = [c for c in chunks if c.doc_id == args.doc]
    if not chunks:
        print("no chunks found; run `make chunks` first")
        return 1
    print_report(chunks, args.n, args.seed, args.max_words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
