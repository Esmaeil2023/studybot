from studybot.ingest import Chunk, process_pdf
from studybot.inspect_chunks import (
    find_warnings,
    length_histogram,
    main,
    numbering_gaps,
    repeated_labels,
)


def make_chunk(section, n_words=100, doc_id="d", text=None, i=0):
    text = text if text is not None else " ".join(["wort"] * n_words)
    return Chunk(
        chunk_id=f"{doc_id}:{section}:{i}",
        doc_id=doc_id,
        doc_title="t",
        version="v",
        section=section,
        section_title="",
        page_start=1,
        page_end=1,
        header=f"[{section}]",
        text=text,
        n_words=len(text.split()),
    )


def test_numbering_gaps_finds_missing_sections():
    chunks = [make_chunk(s) for s in ["Vorspann", "§ 1", "§ 2", "§ 4", "§ 5a", "§ 7", "Anlage 1"]]
    assert numbering_gaps(chunks) == {"d": [3, 6]}


def test_numbering_gaps_empty_when_complete():
    chunks = [make_chunk(f"§ {n}") for n in range(1, 6)]
    assert numbering_gaps(chunks) == {"d": []}


def test_repeated_labels_only_counts_separate_runs():
    chunks = [
        make_chunk("§ 1", i=0),
        make_chunk("§ 1", i=1),
        make_chunk("§ 2"),
        make_chunk("§ 1", i=2),
    ]
    assert repeated_labels(chunks) == {"d": ["§ 1"]}
    assert repeated_labels(chunks[:3]) == {}


def test_length_histogram_buckets():
    chunks = [make_chunk("§ 1", n) for n in (10, 60, 70, 160)]
    assert length_histogram(chunks) == [("0-49", 1), ("50-99", 2), ("100-149", 0), ("150-199", 1)]


def test_find_warnings_flags_the_usual_suspects():
    chunks = [
        make_chunk("§ 1", text="zu kurz"),
        make_chunk("§ 2", n_words=500),
        make_chunk("§ 3", text="Geltungsbereich … 3 " + "wort " * 30),
        make_chunk("§ 5", text="Das ist eine Prüfungs- ordnung " + "wort " * 30),
        make_chunk("§ 6", text="Bachelor- und Master " + "wort " * 30),
    ]
    warnings = " | ".join(find_warnings(chunks, max_words=350))
    assert "under 15 words" in warnings
    assert "over 437 words" in warnings
    assert "table-of-contents leaders" in warnings
    assert "un-joined hyphenation" in warnings and "d:§ 5:0" in warnings
    assert "d:§ 6:0" not in warnings  # "Bachelor- und" is a legitimate hyphen
    assert "no section found for § [4]" in warnings


def test_inspect_cli_on_real_pipeline_output(tmp_path, doc, cfg, capsys):
    from studybot.ingest import write_jsonl

    chunks, _ = process_pdf(doc, cfg)
    path = tmp_path / "chunks.jsonl"
    write_jsonl(chunks, path)

    assert main(["--chunks", str(path), "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "== Documents ==" in out
    assert "no problems found" in out or "WARNING" in out
    assert out.count("--- test-fpo:") == 3
