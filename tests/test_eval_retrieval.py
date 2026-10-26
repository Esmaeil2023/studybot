import json

import pytest

from studybot import eval_retrieval as er
from studybot.evalset import build_questions, write_jsonl
from studybot.ingest import write_jsonl as write_chunks
from test_evalset import CHUNKS, fact


def row(qid="f1-de", gold=("a",), evidence=None, lang="de"):
    return {
        "qid": qid,
        "pair_id": qid[:-3],
        "lang": lang,
        "question": "Frage?",
        "answerable": True,
        "gold_chunk_ids": list(gold),
        "evidence": evidence or [{"doc": "reg", "chunk_ids": list(gold)}],
    }


def test_first_gold_rank():
    assert er.first_gold_rank(["x", "a", "b"], ["b", "a"]) == 2
    assert er.first_gold_rank(["x", "y"], ["a"]) is None
    assert er.first_gold_rank([], ["a"]) is None


def test_spans_covered_needs_one_chunk_per_span():
    evidence = [{"chunk_ids": ["a"]}, {"chunk_ids": ["b", "c"]}]
    assert er.spans_covered(["a", "x", "c"], evidence, 3) == (2, 2)
    assert er.spans_covered(["a", "x", "c"], evidence, 2) == (1, 2)


def test_score_question_hit_full_and_mrr():
    two_spans = row(gold=("a", "b"), evidence=[{"doc": "reg", "chunk_ids": ["a"]},
                                               {"doc": "reg", "chunk_ids": ["b"]}])  # fmt: skip
    r = er.score_question(two_spans, ["x", "a", "y", "z", "b"])
    assert r["first_gold_rank"] == 2 and r["mrr"] == 0.5
    assert r["hit@1"] is False and r["hit@3"] is True
    assert r["full@3"] is False and r["full@5"] is True


def test_score_question_miss():
    r = er.score_question(row(), ["x", "y"])
    assert r["first_gold_rank"] is None and r["mrr"] == 0.0 and not r["hit@10"]


def test_aggregate_and_breakdown():
    records = [
        er.score_question(row("f1-de"), ["a"]),
        er.score_question(row("f2-en", lang="en"), ["x", "a"]),
        er.score_question(row("f3-en", lang="en"), ["x"]),
    ]
    total = er.aggregate(records)
    assert total["n"] == 3
    assert total["hit@1"] == pytest.approx(1 / 3)
    assert total["hit@3"] == pytest.approx(2 / 3)
    assert total["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    by_lang = er.breakdown(records, "lang")
    assert by_lang["de"]["hit@1"] == 1 and by_lang["en"]["n"] == 2
    assert er.aggregate([]) == {"n": 0}


def test_evaluate_skips_unanswerable_questions():
    rows = [row(), {**row("u1-de"), "answerable": False, "gold_chunk_ids": [], "evidence": []}]
    records = er.evaluate(rows, lambda q, depth: ["a"])
    assert [r["qid"] for r in records] == ["f1-de"]


def make_inputs(tmp_path):
    rows = build_questions([fact("f1", "vier Fachsemester")], [], CHUNKS)
    write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    write_jsonl(rows, tmp_path / "q.jsonl")
    return ["--chunks", str(tmp_path / "chunks.jsonl"), "--questions", str(tmp_path / "q.jsonl"),
            "--out-dir", str(tmp_path / "res")]  # fmt: skip


def test_cli_bm25_end_to_end(tmp_path, capsys):
    args = make_inputs(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "q.jsonl").read_text("utf-8").splitlines()]
    split = rows[0]["split"]
    code = er.main(["--system", "bm25", "--split", split, "--final", *args])
    assert code == 0
    out = capsys.readouterr().out
    assert "bm25 on" in out and "hit@5" in out
    saved = json.loads((tmp_path / "res" / f"retrieval_bm25_{split}.json").read_text("utf-8"))
    assert saved["overall"]["n"] == 2 and len(saved["questions"]) == 2


def test_cli_locks_the_test_split(tmp_path, capsys):
    args = make_inputs(tmp_path)
    assert er.main(["--system", "bm25", "--split", "test", *args]) == 2
    assert "locked" in capsys.readouterr().out


def test_cli_reports_missing_files(tmp_path, capsys):
    assert er.main(["--system", "bm25", "--chunks", str(tmp_path / "no.jsonl")]) == 2
    assert "make chunks" in capsys.readouterr().out


class StubRankings(er.Rankings):
    """Rankings without any index: BM25 gives a wrong answer first, dense the right one."""

    def __init__(self):
        self._cache = {}
        self._texts = {"a": "the right answer", "wrong": "something else", "c": "more text"}

    def bm25(self, query, depth):
        return ["wrong", "a"][:depth]

    def dense(self, query, depth):
        return ["a", "wrong"][:depth]


def test_hybrid_weight_moves_the_result_towards_the_trusted_retriever():
    rankings = StubRankings()
    assert rankings.hybrid(0.1, 60)("q", 5)[0] == "a"
    assert rankings.hybrid(0.0, 60)("q", 5)[0] == "a"


def test_run_sweep_covers_the_grid_and_weight_zero_equals_dense():
    rows = [row("f1-de", gold=("a",)), row("f2-en", gold=("a",), lang="en")]
    results = er.run_sweep(rows, StubRankings())
    assert len(results) == len(er.SWEEP_WEIGHTS) * len(er.SWEEP_KS)
    dense_only = [r for r in results if r["bm25_weight"] == 0.0]
    assert all(r["hit@1"] == 1.0 for r in dense_only)
    assert all(set(r) >= {"hit@10_de", "hit@10_en", "mrr"} for r in results)


class OverlapReranker:
    """Scores a passage by how many query words it contains. No model needed."""

    name = "fake/overlap"

    def score(self, query, passages):
        q = set(query.split())
        return [float(len(q & set(p.split()))) for p in passages]


def test_reranked_reorders_only_the_candidates_and_keeps_the_tail():
    class Ranked(StubRankings):
        def dense(self, query, depth):
            return ["wrong", "a", "c"][:depth]

    search = Ranked().reranked(OverlapReranker(), candidates=2)
    # candidates = wrong, a; the query mentions "right answer", so a moves up; c stays last
    assert search("right answer", 5) == ["a", "wrong", "c"]
    assert search("right answer", 1) == ["a"]
