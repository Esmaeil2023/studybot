import json
import re
from collections import Counter

import pytest
import yaml

from sample_regulation import FRONT_MATTER, PARAGRAPHS_6, TOC
from studybot import ingest
from studybot.ingest import (
    ChunkConfig,
    Line,
    Page,
    Piece,
    clean_line,
    dehyphenate,
    extract_pages,
    find_toc_range,
    load_chunks,
    pack_pieces,
    parse_heading,
    process_pdf,
    strip_page_furniture,
)


def chunks_of(chunks, label):
    return [c for c in chunks if c.section == label]


# ---------------------------------------------------------------- small pure functions


@pytest.mark.parametrize(
    ("line", "next_line", "expected"),
    [
        ("§ 1 Geltungsbereich", "", ("§ 1", "Geltungsbereich", False)),
        ("§12a Sonderfälle", "", ("§ 12a", "Sonderfälle", False)),
        ("§ 5", "Wiederholung von Prüfungen", ("§ 5", "Wiederholung von Prüfungen", True)),
        ("§ 5", "(1) Nicht bestandene Prüfungen", ("§ 5", "", False)),
        ("Anlage 2: Studienverlaufsplan", "", ("Anlage 2", "Studienverlaufsplan", False)),
        ("Anlage 1", "Modulübersicht", ("Anlage 1", "Modulübersicht", True)),
        ("Anhang 3: Schwerpunkte im Master", "", ("Anhang 3", "Schwerpunkte im Master", False)),
        ("Präambel", "", ("Präambel", "", False)),
        ("II.", "Abschluss und Modulprüfung", ("Teil II", "Abschluss und Modulprüfung", True)),
        ("IV.", "", None),  # a Roman numeral without a title is not a heading
        ("III.", "weil das so ist.", None),
        ("Ausgefertigt aufgrund des Beschlusses", "", ("Ausfertigung", "", False)),
        # cross-references that happen to start a line are NOT headings
        ("§ 12 Abs. 2 der Allgemeinen Prüfungsordnung bleibt unberührt.", "", None),
        ("§ 7 Satz 3 gilt entsprechend", "", None),
        ("§ 7 und § 8 finden Anwendung.", "", None),
        ("§ 3 Abs. 1", "", None),
        ("§ 5", "gemäß den Vorgaben", ("§ 5", "", False)),  # lower-case next line is not a title
        ("Das Studium beginnt im Winter.", "", None),
        ("Anlage 1 gilt entsprechend", "", None),
        ("Anhang 3 der Studien- und Fachprüfungsordnung für den Bachelor", "", None),
        ("Anhang 2 im Rahmen der Masterarbeit.", "", None),
    ],
)
def test_parse_heading(line, next_line, expected):
    heading = parse_heading(line, next_line)
    if expected is None:
        assert heading is None
    else:
        assert (heading.key, heading.title, heading.title_on_next_line) == expected


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["Prüfungs-", "ordnung gilt."], ["Prüfungsordnung gilt."]),
        (["für Bachelor-", "und Master."], ["für Bachelor- und Master."]),
        (["Studien-", "oder Prüfungsleistungen"], ["Studien- oder Prüfungsleistungen"]),
        (["120 ECTS-", "Punkte"], ["120 ECTS-Punkte"]),
        (
            ["Wahlpflicht-", "module und", "Pflicht-", "module"],
            ["Wahlpflichtmodule und", "Pflichtmodule"],
        ),
        (["Aus-", "ge-", "schlossen"], ["Ausgeschlossen"]),  # chained hyphenation
        (["Ende -", "(2) Neuer Absatz"], ["Ende -", "(2) Neuer Absatz"]),  # no letter before "-"
        (["bis zum 1.", "Oktober"], ["bis zum 1.", "Oktober"]),
    ],
)
def test_dehyphenate(lines, expected):
    result = dehyphenate([Line(1, text) for text in lines])
    assert [line.text for line in result] == expected


def test_dehyphenate_english_keeps_the_hyphen():
    result = dehyphenate([Line(1, "outside third-"), Line(1, "party actors")], keep_hyphen=True)
    assert [line.text for line in result] == ["outside third-party actors"]


def test_dehyphenate_across_a_page_break_keeps_first_page():
    result = dehyphenate([Line(3, "Prüfungs-"), Line(4, "ordnung")])
    assert result == [Line(3, "Prüfungsordnung")]


def test_clean_line_normalises_characters():
    assert clean_line("  Die Prü­fung   ﬁndet  statt ") == "Die Prüfung findet statt"
    assert clean_line("Prüfungs­") == "Prüfungs-"  # a soft hyphen at line end becomes "-"
    assert clean_line("Wahlpflicht‐") == "Wahlpflicht-"


def test_clean_line_collapses_dot_leaders():
    assert clean_line("§ 1 Geltungsbereich " + "." * 90 + " 3") == "§ 1 Geltungsbereich … 3"
    assert clean_line("Wort ... Wort") == "Wort ... Wort"  # three dots are text, not a leader


def test_strip_page_furniture_removes_repeating_header_and_page_numbers():
    pages = [
        Page(n, f"Amtliche Veröffentlichungen Nr. {n}\n§ {n}\nText der Seite {n}.\n- {n} -")
        for n in range(1, 7)
    ]
    cleaned, removed = strip_page_furniture(pages)
    for page in cleaned:
        assert page.text.splitlines()[0].startswith("§ ")  # the "§ n" line survives
        assert "Amtliche" not in page.text
        assert "- " not in page.text.splitlines()[-1]
    assert removed["Amtliche Veröffentlichungen Nr. #"] == 6


def test_strip_page_furniture_needs_enough_pages_to_judge():
    pages = [Page(n, "Kopfzeile\nText\nSeite 1 von 3") for n in range(1, 4)]
    cleaned, _ = strip_page_furniture(pages)
    # 3 pages are too few to call a line "repeating", but page numbers always go
    assert all("Kopfzeile" in p.text for p in cleaned)
    assert all("Seite" not in p.text for p in cleaned)


def _lines(texts):
    return [Line(1, t) for t in texts]


def test_find_toc_range_with_page_numbers_ends_after_the_last_entry():
    texts = [
        "Titel",
        "Inhaltsübersicht",
        "§ 1 Eins … 2",
        "§ 2 Zwei … 3",
        "Aufgrund des Gesetzes erlässt die Universität folgende Satzung:",
        "§ 1",
        "Eins",
    ]
    assert find_toc_range(_lines(texts)) == (1, 4)  # the enacting formula stays outside


def test_find_toc_range_without_page_numbers_ends_at_first_repeated_heading():
    texts = ["Inhaltsverzeichnis", "§ 1 Eins", "§ 2 Zwei", "§ 3 Drei", "§ 1 Eins", "Text"]
    assert find_toc_range(_lines(texts)) == (0, 4)


def test_find_toc_range_accepts_wrapped_appendix_entries_and_part_lines():
    texts = [
        "Inhaltsverzeichnis",
        "I. Allgemeine Regelungen … 3",
        "§ 1 Eins … 3",
        "Anhang 1: Ein langer Titel, der über zwei Zeilen geht",
        "und noch ein Stück … 7",
        "Eingangsformel erlässt folgende Satzung",
        "§ 1",
        "Eins",
    ]
    assert find_toc_range(_lines(texts)) == (0, 5)


def test_find_toc_range_gives_up_instead_of_swallowing_the_document():
    texts = ["Inhaltsverzeichnis", *[f"§ {n} Titel" for n in range(1, 3)], *["Text"] * 500]
    assert find_toc_range(_lines(texts)) is None


def test_find_toc_range_none_without_toc():
    assert find_toc_range(_lines(["§ 1 Eins", "Text"])) is None


def test_pack_pieces_merges_a_tiny_tail():
    pieces = [Piece("a " * 60, 1, 1), Piece("b " * 60, 1, 1), Piece("c " * 5, 1, 1)]
    groups = pack_pieces(pieces, max_words=100, min_words=20)
    assert [sum(p.words for p in g) for g in groups] == [60, 65]  # tail (5) merged into 2nd


def test_pack_pieces_keeps_a_tail_that_would_overshoot():
    # 100 + 30 = 130 words would be more than 120 % of max_words, so the 30-word tail stays alone
    pieces = [Piece("a " * 60, 1, 1), Piece("b " * 100, 1, 1), Piece("c " * 30, 1, 1)]
    groups = pack_pieces(pieces, max_words=100, min_words=40)
    assert [sum(p.words for p in g) for g in groups] == [60, 100, 30]


# ---------------------------------------------------------------- PDF extraction


def test_extract_pages_keeps_page_numbers_and_umlauts(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert [p.number for p in pages] == list(range(1, 9))
    assert "Fachprüfungsordnung" in pages[0].text
    assert "Seite 3 von 8" in pages[2].text


# ---------------------------------------------------------------- regulation pipeline


def test_sections_are_found_in_order_with_titles(doc, cfg):
    chunks, info = process_pdf(doc, cfg)
    labels = list(dict.fromkeys(c.section for c in chunks))
    assert labels == [
        "Eingangsformel",
        "§ 1",
        "§ 2",
        "§ 3",
        "§ 4",
        "§ 5",
        "§ 6",
        "Anhang 1",
        "Ausfertigung",
    ]
    titles = {c.section: c.section_title for c in chunks}
    assert titles["§ 1"] == "Geltungsbereich"
    assert titles["§ 2"] == "Studienumfang und Regelstudienzeit"  # title was on the next line
    assert titles["§ 5"] == "Wiederholung von Prüfungen"
    assert titles["Anhang 1"].startswith("Studienverlaufsplan")
    assert info["sections"] == 9


def test_front_matter_and_table_of_contents_are_dropped(doc, cfg):
    chunks, info = process_pdf(doc, cfg)
    everything = "\n".join(c.text for c in chunks)
    assert info["front_matter_lines_removed"] == len(FRONT_MATTER)
    assert info["toc_removed_lines"] == len(TOC)
    for unwanted in ("Fundstelle", "geändert durch", "Inhaltsverzeichnis", "…"):
        assert unwanted not in everything


def test_front_matter_can_be_kept(doc):
    chunks, info = process_pdf(doc, ChunkConfig(drop_front_matter=False))
    assert chunks[0].section == "Vorspann"
    assert "Fundstelle" in chunks[0].text
    # the table of contents is still skipped
    assert "Inhaltsverzeichnis" not in "\n".join(c.text for c in chunks)
    assert info["front_matter_lines_removed"] == 0


def test_enacting_formula_is_kept_as_its_own_section(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    first = chunks[0]
    assert first.section == "Eingangsformel"
    assert first.text.startswith("Aufgrund des Art. 9 Satz 1")
    assert first.text.endswith("folgende Satzung:")
    assert first.chunk_id == "test-fpo:eingangsformel:0"


def test_roman_part_headings_end_a_section_and_are_not_text(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    everything = "\n".join(c.text for c in chunks)
    assert "Allgemeine Regelungen" not in everything
    assert not re.search(r"(?m)^(?:I|II)\.$", everything)
    assert not chunks_of(chunks, "§ 5")[0].text.rstrip().endswith("Abschluss")


def test_signature_block_is_its_own_section_and_keeps_its_first_line(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    anhang = "\n".join(c.text for c in chunks_of(chunks, "Anhang 1"))
    assert "Ausgefertigt" not in anhang
    assert "4. Semester: Masterarbeit" in anhang
    signature = chunks_of(chunks, "Ausfertigung")[0]
    assert signature.text.startswith("Ausgefertigt aufgrund des Beschlusses")


def test_running_header_and_footer_are_removed(doc, cfg):
    chunks, info = process_pdf(doc, cfg)
    everything = "\n".join(c.text for c in chunks)
    assert "Amtliche Veröffentlichungen" not in everything
    assert "Seite" not in everything
    assert info["furniture_removed"]["Seite # von #"] == 8


def test_hyphenation_is_repaired(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    by_section = {c.section: c.text for c in chunks}
    assert "Allgemeine Prüfungsordnung in ihrer jeweils geltenden Fassung" in by_section["§ 1"]
    assert "Prüfungsordnung für Bachelor- und Masterstudiengänge" in by_section["§ 1"]
    assert "umfasst 120 ECTS-Punkte und ist" in by_section["§ 2"]
    assert "Prüfungsverwaltungssystem" in by_section["§ 4"]
    assert not re.search(r"[a-zäöüß]-\n", by_section["§ 1"])


def test_cross_reference_at_line_start_is_not_a_heading(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    assert "§ 12" not in {c.section for c in chunks}
    text = chunks_of(chunks, "§ 3")[0].text
    assert "§ 12 Abs. 2 der Allgemeinen Prüfungsordnung bleibt unberührt." in text


def test_page_ranges_follow_the_pdf(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    pages = {c.section: (c.page_start, c.page_end) for c in chunks}
    assert pages["§ 1"] == (3, 3)
    assert pages["§ 3"] == (4, 5)  # the section runs over a page break
    assert pages["§ 5"] == (6, 6)
    assert pages["§ 6"] == (7, 7)
    assert pages["Anhang 1"] == (8, 8)


def test_header_prefix_and_metadata(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    c = chunks_of(chunks, "§ 5")[0]
    assert c.header == "[FPO Test | § 5 Wiederholung von Prüfungen]"
    assert c.index_text == f"{c.header}\n{c.text}"
    assert c.doc_id == "test-fpo"
    assert c.version == "2024-10-01"
    assert c.n_words == len(c.text.split())
    assert c.text.startswith("(1) Nicht bestandene Modulprüfungen")


def test_nothing_is_empty_and_ids_are_unique(doc, handbook_doc, cfg, small_cfg):
    for document in (doc, handbook_doc):
        for config in (cfg, small_cfg):
            chunks, _ = process_pdf(document, config)
            assert chunks
            assert all(c.text.strip() for c in chunks)
            assert all(c.n_words > 0 for c in chunks)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids))


def test_output_is_deterministic(doc, handbook_doc, small_cfg):
    for document in (doc, handbook_doc):
        first, _ = process_pdf(document, small_cfg)
        second, _ = process_pdf(document, small_cfg)
        assert first == second


def test_small_limits_split_at_subparagraphs_and_lose_no_words(doc, small_cfg):
    chunks, _ = process_pdf(doc, small_cfg)
    section6 = chunks_of(chunks, "§ 6")

    assert len(section6) >= 3
    assert all(c.n_words <= small_cfg.max_words * 1.2 for c in chunks)
    assert section6[0].text.startswith("(1) ")
    # every sub-paragraph marker starts a line of some chunk: no unit is glued mid-line
    for n in range(1, 7):
        assert any(re.search(rf"(?m)^\({n}\) ", c.text) for c in section6), n
    # nothing lost, nothing duplicated
    expected = " ".join(PARAGRAPHS_6).split()
    actual = " ".join(c.text for c in section6).split()
    assert actual == expected


def test_a_section_that_fits_stays_in_one_chunk(doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    assert len(chunks_of(chunks, "§ 6")) == 1
    assert chunks_of(chunks, "§ 6")[0].chunk_id == "test-fpo:par6:0"


def test_chunk_ids_count_up_within_a_section(doc, small_cfg):
    chunks, _ = process_pdf(doc, small_cfg)
    ids = [c.chunk_id for c in chunks_of(chunks, "§ 6")]
    assert ids == [f"test-fpo:par6:{i}" for i in range(len(ids))]


# ---------------------------------------------------------------- module handbook


def test_handbook_has_one_section_per_module(handbook_doc, cfg):
    chunks, info = process_pdf(handbook_doc, cfg)
    labels = list(dict.fromkeys(c.section for c in chunks))
    assert labels == ["Allgemeines", "Modul TEST-ALGO-M", "Modul TEST-AUTO-M", "Anhang"]
    assert info["sections"] == 4
    algo = chunks_of(chunks, "Modul TEST-ALGO-M")
    title = "Algorithms and Data Structures Algorithmen und Datenstrukturen"
    assert algo[0].section_title == title
    assert algo[0].header == f"[MHB Test | Modul TEST-ALGO-M {title}]"
    assert chunks_of(chunks, "Modul TEST-AUTO-M")[0].section_title == "Automata Theory"
    assert (algo[0].page_start, algo[-1].page_end) == (3, 4)  # a module can span pages


def test_handbook_drops_running_headers_cover_title_and_toc(handbook_doc, cfg):
    chunks, _ = process_pdf(handbook_doc, cfg)
    everything = "\n".join(c.text for c in chunks)
    assert "Test Master Programme 2026" not in everything  # repeated header of every page
    assert "…" not in everything  # table-of-contents leaders
    assert "Table of Contents" not in everything
    assert not re.search(r"(?m)^Module TEST-[A-Z]+-M$", everything)  # running header lines
    # the title line with the ECTS value is content and stays
    assert "Module TEST-ALGO-M Algorithms and 6 ECTS / 180 h" in everything
    # "Module Units" is a field label, not a module
    assert "Module Units" in everything


def test_handbook_keeps_english_compound_hyphens(handbook_doc, cfg):
    chunks, _ = process_pdf(handbook_doc, cfg)
    everything = "\n".join(c.text for c in chunks)
    assert "third-party libraries" in everything


def test_handbook_splits_at_field_labels(handbook_doc):
    chunks, _ = process_pdf(handbook_doc, ChunkConfig(max_words=30, min_words=5))
    algo = chunks_of(chunks, "Modul TEST-ALGO-M")
    assert len(algo) >= 3
    starts = [c.text.split("\n")[0] for c in algo]
    assert any(s.startswith("Contents:") for s in starts)
    assert any(s.startswith("Learning outcomes:") for s in starts)
    # small pieces are packed together, so a label may also sit at a line start inside a chunk
    assert any(re.search(r"(?m)^Examination\b", c.text) for c in algo)


# ---------------------------------------------------------------- files, config, CLI


def test_jsonl_roundtrip_keeps_umlauts(tmp_path, doc, cfg):
    chunks, _ = process_pdf(doc, cfg)
    path = tmp_path / "out" / "chunks.jsonl"
    ingest.write_jsonl(chunks, path)

    raw = path.read_text(encoding="utf-8")
    assert "Prüfungen" in raw  # written as UTF-8, not as ü escapes
    assert json.loads(raw.splitlines()[0])["doc_id"] == "test-fpo"
    assert load_chunks(path) == chunks


def _write_config(tmp_path, doc, **chunking):
    config = {
        "chunking": {"max_words": 80, "min_words": 20, **chunking},
        "documents": [
            {
                "doc_id": doc.doc_id,
                "file": doc.file,
                "title": doc.title,
                "short_title": doc.short_title,
                "version": doc.version,
                "kind": "regulation",
            }
        ],
    }
    path = tmp_path / "ingest.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_cli_end_to_end(tmp_path, doc, capsys):
    config_path = _write_config(tmp_path, doc)
    out = tmp_path / "processed" / "chunks.jsonl"

    code = ingest.main(["--config", str(config_path), "--out", str(out)])

    assert code == 0
    chunks = load_chunks(out)
    assert len(chunks) >= 9
    manifest = json.loads((tmp_path / "processed" / "ingest_manifest.json").read_text("utf-8"))
    assert manifest["chunking"]["max_words"] == 80
    assert len(manifest["documents"][0]["sha256"]) == 64
    assert manifest["documents"][0]["chunks"] == len(chunks)
    assert "test-fpo" in capsys.readouterr().out


def test_cli_reports_missing_pdfs_instead_of_crashing(tmp_path, doc, capsys):
    missing = ingest.DocConfig(**{**doc.__dict__, "file": str(tmp_path / "nope.pdf")})
    config_path = _write_config(tmp_path, missing)

    code = ingest.main(["--config", str(config_path), "--out", str(tmp_path / "x.jsonl")])

    assert code == 2
    assert "nope.pdf" in capsys.readouterr().out
    assert not (tmp_path / "x.jsonl").exists()


@pytest.mark.parametrize("chunking", [{"min_words": 80}, {"min_words": 0}])
def test_config_validation(tmp_path, doc, chunking):
    config_path = _write_config(tmp_path, doc, **chunking)
    with pytest.raises(ValueError):
        ingest.load_config(config_path)


def test_real_config_file_is_valid():
    cfg, docs = ingest.load_config("configs/ingest.yaml")
    assert cfg.max_words == 350
    assert [d.kind for d in docs] == ["regulation", "regulation", "handbook"]
    assert Counter(d.doc_id for d in docs).most_common(1)[0][1] == 1
