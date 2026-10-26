import json
from collections import Counter

import pytest
import yaml

from studybot import evalset
from studybot.evalset import (
    assign_splits,
    build_questions,
    find_span,
    load_questions,
    map_evidence,
    normalize,
    span_coverage,
    validate,
)
from studybot.ingest import Chunk


def make_chunk(doc_id, section, text, idx=0, page=1):
    return Chunk(
        chunk_id=f"{doc_id}:{section.lower().replace(' ', '').replace('§', 'par')}:{idx}",
        doc_id=doc_id,
        doc_title="t",
        version="v",
        section=section,
        section_title="",
        page_start=page,
        page_end=page,
        header=f"[{doc_id} | {section}]",
        text=text,
        n_words=len(text.split()),
    )


CHUNKS = [
    make_chunk(
        "reg",
        "§ 1",
        "(1) Die Regelstudienzeit beträgt vier Fachsemester. (2) Der Studienumfang "
        "beträgt mindestens 120 ECTS-Punkte.",
    ),
    make_chunk(
        "reg",
        "§ 2",
        "Das Thema kann „nur einmal“ zurückgegeben werden – innerhalb von vier Wochen.",
    ),
    make_chunk(
        "hb", "Modul A", "Frequency: every winter semester\nLecturers: Prof. Dr. Ada Beispiel"
    ),
    make_chunk("hb", "Modul B", "Frequency: every summer semester\nLecturers: Dr. Bob Muster"),
]


def fact(fid, span, doc="reg", section=None, extra=None):
    ev = {"doc": doc, "span": span}
    if section:
        ev["section"] = section
    return {
        "id": fid,
        "category": "number",
        "q_de": f"Frage {fid}?",
        "q_en": f"Question {fid}?",
        "a_de": "Antwort",
        "a_en": "Answer",
        "evidence": [ev, *(extra or [])],
    }


def unans(uid, kind="off-topic", absent=()):
    return {
        "id": uid,
        "kind": kind,
        "q_de": f"Unbeantwortbar {uid}?",
        "q_en": f"Unanswerable {uid}?",
        "absent": list(absent),
        "why": "not covered",
    }


# ---------------------------------------------------------------- matching


def test_normalize_unifies_quotes_dashes_case_and_whitespace():
    assert normalize("  „Nur   einmal“ – Ja\n") == '"nur einmal" - ja'
    assert normalize("S1: A − S2") == "s1: a - s2"


def test_span_coverage():
    assert (
        span_coverage("vier Fachsemester beträgt", "die Regelstudienzeit beträgt vier Fachsemester")
        < 1
    )
    assert (
        span_coverage("beträgt vier Fachsemester", "die Regelstudienzeit beträgt vier Fachsemester")
        == 1
    )
    assert span_coverage("", "text") == 0


def test_find_span_exact_ignores_layout_differences():
    match = find_span('"nur einmal" zurückgegeben werden - innerhalb', CHUNKS, "reg")
    assert (match.method, match.chunk_ids) == ("exact", ["reg:par2:0"])


def test_find_span_fuzzy_takes_the_best_chunk_and_flags_it():
    # the word "und" is not in the chunk, so no exact hit, but 4 of 5 word triples still match
    match = find_span("Der Studienumfang beträgt mindestens 120 ECTS-Punkte und", CHUNKS, "reg")
    assert match.method == "fuzzy"
    assert match.chunk_ids == ["reg:par1:0"]
    assert 0.8 <= match.best_coverage < 1


def test_find_span_none_when_text_is_absent():
    match = find_span("Studiengebühren betragen 500 Euro pro Semester", CHUNKS, "reg")
    assert match.method == "none" and match.chunk_ids == []


def test_find_span_respects_document_and_section():
    assert find_span("Lecturers: Dr. Bob Muster", CHUNKS, "reg").chunk_ids == []
    both = find_span("Frequency: every", CHUNKS, "hb").chunk_ids
    assert len(both) == 2
    only_b = find_span("Frequency: every", CHUNKS, "hb", section="modul b").chunk_ids
    assert only_b == ["hb:modulb:0"]


def test_map_evidence_keeps_all_spans():
    f = fact("f1", "vier Fachsemester", extra=[{"doc": "reg", "span": "120 ECTS-Punkte"}])
    mapped = map_evidence(f, CHUNKS)
    assert [m["chunk_ids"] for m in mapped] == [["reg:par1:0"], ["reg:par1:0"]]
    assert all(m["match"] == "exact" for m in mapped)


# ---------------------------------------------------------------- split


def many_facts(n, doc="reg"):
    return [fact(f"x{doc}{i:02d}", "vier Fachsemester", doc=doc) for i in range(n)]


def test_split_sizes_are_stratified_and_deterministic():
    facts = many_facts(20, "reg") + many_facts(10, "hb")
    unanswerable = [unans(f"u{i}", "off-topic") for i in range(10)] + [
        unans(f"n{i}", "near-miss") for i in range(5)
    ]
    split = assign_splits(facts, unanswerable)

    assert split == assign_splits(facts, unanswerable)
    assert split != assign_splits(facts, unanswerable, seed=999)
    by_group = Counter(
        (
            ("fact", f["evidence"][0]["doc"]) if "q_de" in f and "evidence" in f else None,
            split[f["id"]],
        )
        for f in facts
    )
    assert by_group[(("fact", "reg"), "dev")] == 8
    assert by_group[(("fact", "hb"), "dev")] == 4
    assert sum(1 for u in unanswerable if u["kind"] == "near-miss" and split[u["id"]] == "dev") == 2
    assert Counter(split.values()) == {"dev": 8 + 4 + 4 + 2, "test": 30 + 15 - 18}


def test_build_questions_makes_a_de_en_pair_in_the_same_split():
    rows = build_questions([fact("f1", "vier Fachsemester")], [unans("u1")], CHUNKS)
    assert [r["qid"] for r in rows] == ["f1-de", "f1-en", "u1-de", "u1-en"]
    answerable = rows[0]
    assert answerable["answerable"] and answerable["gold_chunk_ids"] == ["reg:par1:0"]
    assert answerable["question"] == "Frage f1?" and rows[1]["question"] == "Question f1?"
    assert rows[0]["split"] == rows[1]["split"]
    assert rows[2]["answer"] is None and rows[2]["gold_chunk_ids"] == []
    assert rows[2]["category"] == "off-topic"


# ---------------------------------------------------------------- validation


def run_validate(facts, unanswerable):
    rows = build_questions(facts, unanswerable, CHUNKS)
    return validate(facts, unanswerable, rows, CHUNKS)


def test_validate_accepts_a_clean_set():
    assert run_validate([fact("f1", "vier Fachsemester")], [unans("u1", absent=["Mensa"])]) == []


def test_validate_reports_evidence_that_is_not_in_the_chunks():
    problems = run_validate([fact("f1", "Die Gebühr beträgt 500 Euro")], [])
    assert any("evidence not found" in p for p in problems)


def test_validate_reports_fuzzy_matches():
    problems = run_validate(
        [fact("f1", "Der Studienumfang beträgt mindestens 120 ECTS-Punkte und")], []
    )
    assert any("only matched fuzzily" in p for p in problems)


def test_validate_reports_spans_that_are_too_generic():
    chunks = [
        make_chunk("reg", f"§ {i}", "Das gilt entsprechend für alle Fälle.") for i in range(5)
    ]
    facts = [fact("f1", "Das gilt entsprechend")]
    rows = build_questions(facts, [], chunks)
    assert any("make it more specific" in p for p in validate(facts, [], rows, chunks))


def test_validate_catches_an_unanswerable_question_whose_topic_is_in_the_documents():
    problems = run_validate([fact("f1", "vier Fachsemester")], [unans("u1", absent=[r"\bECTS\b"])])
    assert any("u1" in p and "not unanswerable" in p for p in problems)


def test_validate_catches_duplicate_ids_and_missing_fields():
    broken = fact("f1", "vier Fachsemester")
    broken["a_en"] = ""
    problems = run_validate([broken, fact("f1", "120 ECTS-Punkte")], [])
    assert any("duplicate id f1" in p for p in problems)
    assert any("missing a_en" in p for p in problems)


def test_validate_catches_duplicate_question_text():
    a = fact("f1", "vier Fachsemester")
    b = fact("f2", "120 ECTS-Punkte")
    b["q_en"] = a["q_en"]
    assert any("duplicate question text" in p for p in run_validate([a, b], []))


# ---------------------------------------------------------------- command line


def write_inputs(tmp_path, facts, unanswerable):
    from studybot.ingest import write_jsonl

    write_jsonl(CHUNKS, tmp_path / "chunks.jsonl")
    (tmp_path / "facts.yaml").write_text(
        yaml.safe_dump({"facts": facts}, allow_unicode=True), "utf-8"
    )
    (tmp_path / "unanswerable.yaml").write_text(
        yaml.safe_dump({"unanswerable": unanswerable}, allow_unicode=True), "utf-8"
    )
    return [
        "--chunks", str(tmp_path / "chunks.jsonl"),
        "--facts", str(tmp_path / "facts.yaml"),
        "--unanswerable", str(tmp_path / "unanswerable.yaml"),
        "--out", str(tmp_path / "q.jsonl"),
    ]  # fmt: skip


def test_cli_writes_questions_and_review_sheet(tmp_path, capsys):
    args = write_inputs(tmp_path, [fact("f1", "vier Fachsemester")], [unans("u1")])
    code = evalset.main([*args, "--review", str(tmp_path / "review.md")])

    assert code == 0
    rows = load_questions(tmp_path / "q.jsonl")
    assert len(rows) == 4
    assert load_questions(tmp_path / "q.jsonl", split=rows[0]["split"])
    review = (tmp_path / "review.md").read_text("utf-8")
    assert "## f1" in review and "reg:par1:0 p.1-1" in review and "Why: not covered" in review
    assert "pairs" in capsys.readouterr().out


def test_cli_does_not_write_when_validation_fails(tmp_path, capsys):
    args = write_inputs(tmp_path, [fact("f1", "steht nirgends")], [])
    assert evalset.main(args) == 1
    assert not (tmp_path / "q.jsonl").exists()
    assert "evidence not found" in capsys.readouterr().out


def test_cli_reports_missing_chunks_file(tmp_path, capsys):
    assert evalset.main(["--chunks", str(tmp_path / "nope.jsonl")]) == 2
    assert "make chunks" in capsys.readouterr().out


def test_jsonl_roundtrip_keeps_umlauts(tmp_path):
    rows = build_questions([fact("f1", "vier Fachsemester")], [], CHUNKS)
    evalset.write_jsonl(rows, tmp_path / "q.jsonl")
    assert "Fachsemester" in (tmp_path / "q.jsonl").read_text("utf-8")
    assert json.loads((tmp_path / "q.jsonl").read_text("utf-8").splitlines()[0])["qid"] == "f1-de"


# ---------------------------------------------------------------- the real source files


@pytest.fixture(scope="module")
def real_sources():
    return (
        evalset.load_source("data/eval/facts.yaml", "facts"),
        evalset.load_source("data/eval/unanswerable.yaml", "unanswerable"),
    )


def test_real_source_files_are_complete(real_sources):
    facts, unanswerable = real_sources
    assert (len(facts), len(unanswerable)) == (50, 20)
    ids = [x["id"] for x in facts + unanswerable]
    assert len(ids) == len(set(ids))
    for f in facts:
        assert all(f[k].strip() for k in ("q_de", "q_en", "a_de", "a_en")), f["id"]
        assert f["evidence"] and all(e["span"].strip() and e["doc"] for e in f["evidence"]), f["id"]
    for u in unanswerable:
        assert u["kind"] in {"off-topic", "near-miss"} and u["why"], u["id"]


def test_real_questions_are_questions_and_languages_differ(real_sources):
    facts, unanswerable = real_sources
    for item in facts + unanswerable:
        assert item["q_de"].rstrip().endswith("?") and item["q_en"].rstrip().endswith("?"), item[
            "id"
        ]
        assert item["q_de"] != item["q_en"], item["id"]


def test_real_evidence_documents_are_configured(real_sources):
    cfg, docs = __import__("studybot.ingest", fromlist=["load_config"]).load_config(
        "configs/ingest.yaml"
    )
    known = {d.doc_id for d in docs}
    facts, _ = real_sources
    assert {e["doc"] for f in facts for e in f["evidence"]} <= known


def test_real_questions_file_if_present_is_consistent():
    from pathlib import Path

    path = Path("data/eval/questions.jsonl")
    if not path.exists():
        pytest.skip("run `make evalset` first")
    rows = load_questions(path)
    assert len(rows) == 140
    assert Counter(r["split"] for r in rows if r["lang"] == "en") == {"test": 42, "dev": 28}
    assert all(r["gold_chunk_ids"] for r in rows if r["answerable"])
