"""Measure retrieval on the evaluation set (Phase 3).

    python -m studybot.eval_retrieval --system bm25 --split dev
    python -m studybot.eval_retrieval --system dense --split dev
    python -m studybot.eval_retrieval --system hybrid --split dev

Only answerable questions are used here; the unanswerable ones matter once an answer is
generated. For each question the system returns a ranked list of chunks, and we ask where the
first *gold* chunk (a chunk containing the evidence) appears.

Metrics
    hit@k   share of questions with at least one gold chunk in the top k
    full@k  share of questions where *every* evidence span has a gold chunk in the top k
            (differs from hit@k only for the few questions that need several chunks)
    MRR     mean of 1 / rank of the first gold chunk (0 if it is not in the top `depth`)

The test split is locked: it needs `--final`, so that nobody (me included) tunes on it by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from studybot.evalset import load_questions
from studybot.ingest import Chunk, load_chunks
from studybot.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    DenseIndex,
    E5Embedder,
    Reranker,
    rerank,
    rrf_fuse,
)

KS = (1, 3, 5, 10, 20)
DEPTH = 50

Search = Callable[[str, int], list[str]]  # (query, depth) -> chunk ids, best first


# ---------------------------------------------------------------- metrics


def first_gold_rank(ranked: Sequence[str], gold: Sequence[str]) -> int | None:
    """1-based position of the first gold chunk, or None."""
    gold_set = set(gold)
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold_set:
            return position
    return None


def spans_covered(ranked: Sequence[str], evidence: Sequence[dict], k: int) -> tuple[int, int]:
    """(covered, total): how many evidence spans have a gold chunk among the top k."""
    top = set(ranked[:k])
    covered = sum(1 for ev in evidence if top & set(ev["chunk_ids"]))
    return covered, len(evidence)


def score_question(row: dict, ranked: Sequence[str], ks: Sequence[int] = KS) -> dict:
    rank = first_gold_rank(ranked, row["gold_chunk_ids"])
    record = {
        "qid": row["qid"],
        "pair_id": row["pair_id"],
        "lang": row["lang"],
        "doc": row["evidence"][0]["doc"],
        "question": row["question"],
        "first_gold_rank": rank,
        "mrr": 1 / rank if rank else 0.0,
        "top5": list(ranked[:5]),
        "gold": row["gold_chunk_ids"],
    }
    for k in ks:
        record[f"hit@{k}"] = rank is not None and rank <= k
        covered, total = spans_covered(ranked, row["evidence"], k)
        record[f"full@{k}"] = covered == total
    return record


def aggregate(records: Sequence[dict], ks: Sequence[int] = KS) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    out: dict = {"n": n, "mrr": sum(r["mrr"] for r in records) / n}
    for k in ks:
        out[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in records) / n
        out[f"full@{k}"] = sum(r[f"full@{k}"] for r in records) / n
    return out


def evaluate(
    rows: Sequence[dict], search: Search, depth: int = DEPTH, ks: Sequence[int] = KS
) -> list[dict]:
    """Score every answerable question. `rows` come from questions.jsonl."""
    return [
        score_question(row, search(row["question"], depth), ks) for row in rows if row["answerable"]
    ]


def breakdown(records: Sequence[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


# ---------------------------------------------------------------- systems


class Rankings:
    """BM25 and dense rankings of every query, computed once and reused for many fusions."""

    def __init__(self, chunks: list[Chunk], args: argparse.Namespace, need_dense: bool):
        self._texts = {c.chunk_id: c.index_text for c in chunks}
        self._bm25 = BM25Index(chunks)
        self._dense = (
            DenseIndex(chunks, E5Embedder(args.model), cache_dir=args.cache_dir)
            if need_dense
            else None
        )
        self._cache: dict[tuple[str, str], list[str]] = {}

    def bm25(self, query: str, depth: int) -> list[str]:
        key = ("bm25", query)
        if key not in self._cache:
            self._cache[key] = [cid for cid, _ in self._bm25.search(query, depth)]
        return self._cache[key][:depth]

    def dense(self, query: str, depth: int) -> list[str]:
        key = ("dense", query)
        if key not in self._cache:
            self._cache[key] = [cid for cid, _ in self._dense.search(query, depth)]
        return self._cache[key][:depth]

    def reranked(self, reranker: Reranker, candidates: int) -> Search:
        """Dense top `candidates`, reordered by the cross-encoder; the rest keeps dense order."""

        def search(query: str, depth: int) -> list[str]:
            first_stage = self.dense(query, max(depth, candidates))
            head, tail = first_stage[:candidates], first_stage[candidates:]
            key = ("rerank", query)
            if key not in self._cache:
                self._cache[key] = rerank(query, head, self._texts, reranker)
            return (self._cache[key] + tail)[:depth]

        return search

    def hybrid(self, bm25_weight: float, rrf_k: int) -> Search:
        def search(query: str, depth: int) -> list[str]:
            rankings = [self.bm25(query, depth), self.dense(query, depth)]
            return rrf_fuse(rankings, k=rrf_k, weights=[bm25_weight, 1.0])[:depth]

        return search


def build_search(system: str, rankings: Rankings, args: argparse.Namespace) -> Search:
    if system == "bm25":
        return rankings.bm25
    if system == "dense":
        return rankings.dense
    if system == "rerank":
        reranker = CrossEncoderReranker(args.reranker_model)
        return rankings.reranked(reranker, args.candidates)
    return rankings.hybrid(args.bm25_weight, args.rrf_k)


SWEEP_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
SWEEP_KS = (10, 60)


def run_sweep(rows: Sequence[dict], rankings: Rankings) -> list[dict]:
    """Hybrid quality for a small grid of BM25 weights and RRF k (dense weight is always 1).

    Keep the grid small: with 40 dev questions a big grid mostly finds noise.
    """
    results = []
    for k in SWEEP_KS:
        for weight in SWEEP_WEIGHTS:
            records = evaluate(rows, rankings.hybrid(weight, k))
            by_lang = breakdown(records, "lang")
            results.append(
                {
                    "bm25_weight": weight,
                    "rrf_k": k,
                    "hit@10": aggregate(records)["hit@10"],
                    "hit@10_de": by_lang["de"]["hit@10"],
                    "hit@10_en": by_lang["en"]["hit@10"],
                    "hit@1": aggregate(records)["hit@1"],
                    "mrr": aggregate(records)["mrr"],
                }
            )
    return results


def print_sweep(results: Sequence[dict]) -> None:
    print("== hybrid sweep (bm25_weight 0 = dense only) ==")
    print(f"{'k':>4} {'w_bm25':>7} {'hit@1':>7} {'hit@10':>7} {'de@10':>7} {'en@10':>7} {'MRR':>6}")
    for r in results:
        print(
            f"{r['rrf_k']:>4} {r['bm25_weight']:>7.2f} {r['hit@1']:>7.2f} {r['hit@10']:>7.2f} "
            f"{r['hit@10_de']:>7.2f} {r['hit@10_en']:>7.2f} {r['mrr']:>6.2f}"
        )


# ---------------------------------------------------------------- report


def format_table(rows: dict[str, dict], ks: Sequence[int] = KS) -> str:
    header = ["group", "n", *[f"hit@{k}" for k in ks], "full@5", "MRR"]  # hit@20 = recall ceiling
    lines = ["  ".join(f"{h:>8}" if i else f"{h:<18}" for i, h in enumerate(header))]
    for name, m in rows.items():
        cells = [f"{name:<18}", f"{m['n']:>8}"]
        cells += [f"{m[f'hit@{k}']:>8.2f}" for k in ks]
        cells += [f"{m['full@5']:>8.2f}", f"{m['mrr']:>8.2f}"]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def print_report(system: str, split: str, records: list[dict], n_misses: int) -> None:
    print(f"== {system} on {split}: {len(records)} questions ==")
    table = {"all": aggregate(records), **breakdown(records, "lang")}
    print(format_table(table))
    print("\nby document (all languages):")
    print(format_table(breakdown(records, "doc")))
    misses = [r for r in records if not r["hit@10"]]
    print(f"\n{len(misses)} questions without a gold chunk in the top 10", end="")
    print(":" if misses else ".")
    for r in misses[:n_misses]:
        print(f"  {r['qid']}  {r['question'][:80]}")
        print(f"      gold {r['gold']}   top1 {r['top5'][:1]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--system", choices=["bm25", "dense", "hybrid", "rerank"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--cache-dir", default="indexes")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--candidates", type=int, default=20, help="rerank: first-stage top N")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="hybrid: BM25 weight in RRF")
    parser.add_argument("--sweep", action="store_true", help="hybrid: try a small grid, print only")
    parser.add_argument("--misses", type=int, default=10, help="how many misses to print")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.final:
        print("The test split is locked. Tune on dev; use --final once, for the reported numbers.")
        return 2
    for path in (args.chunks, args.questions):
        if not Path(path).exists():
            print(f"{path} not found (run `make chunks` and `make evalset` first)")
            return 2

    chunks = load_chunks(args.chunks)
    rows = load_questions(args.questions, split=args.split)
    rankings = Rankings(chunks, args, need_dense=args.system != "bm25" or args.sweep)
    if args.sweep:
        print_sweep(run_sweep(rows, rankings))
        return 0
    records = evaluate(rows, build_search(args.system, rankings, args))

    print_report(args.system, args.split, records, args.misses)
    out = Path(args.out_dir) / f"retrieval_{args.system}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "system": args.system,
                "split": args.split,
                "model": args.model if args.system != "bm25" else None,
                "reranker": args.reranker_model if args.system == "rerank" else None,
                "candidates": args.candidates if args.system == "rerank" else None,
                "bm25_weight": args.bm25_weight if args.system == "hybrid" else None,
                "rrf_k": args.rrf_k if args.system == "hybrid" else None,
                "overall": aggregate(records),
                "by_lang": breakdown(records, "lang"),
                "by_doc": breakdown(records, "doc"),
                "questions": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
