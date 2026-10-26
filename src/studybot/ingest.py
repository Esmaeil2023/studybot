"""Phase 1: turn regulation PDFs and the module handbook into clean, section-aware chunks.

The pipeline has one function per stage. Read them top to bottom:

    PDF
     |  extract_pages          text of every page, page numbers kept
     |  strip_page_furniture   remove page numbers and headers/footers that repeat on most pages
     |  pages_to_lines         split into lines, normalise characters, collapse dot leaders
     |  dehyphenate            "Prüfungs-" + "ordnung"  ->  "Prüfungsordnung"
     |
     |  regulations (kind: regulation)         module handbook (kind: handbook)
     |  find_toc_range  skip front matter      handbook_sections  one section per module,
     |                  and table of contents                      found via its running header
     |  parse_sections  cut at "§ 12 Title",
     |                  "Anhang 1", "Teil II"
     |
     |  chunk_section          cut long sections at "(1)", "(2)" (or at the handbook's field
     |                         labels), then at sentences
     v
    data/processed/chunks.jsonl     (one JSON object per line)

Run it from the repo root:

    python -m studybot.ingest --config configs/ingest.yaml --out data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
import yaml

# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Page:
    number: int  # 1-based page number in the PDF
    text: str


@dataclass(frozen=True)
class Line:
    page: int
    text: str


@dataclass(frozen=True)
class DocConfig:
    doc_id: str
    file: str
    title: str
    short_title: str
    version: str
    kind: str = "regulation"  # "regulation" or "handbook"


@dataclass(frozen=True)
class ChunkConfig:
    max_words: int = 350
    min_words: int = 40
    # Regulations only: drop everything before the end of the table of contents (title page,
    # list of amending statutes). Not useful for answering questions, and it contains many URLs.
    drop_front_matter: bool = True


@dataclass
class Section:
    label: str  # "§ 12", "Anhang 1", "Modul GdI-IFP-M", "Eingangsformel", "Vorspann"
    title: str
    page_start: int
    lines: list[Line] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str  # "stufpo-isss-2024:par12:0"
    doc_id: str
    doc_title: str
    version: str
    section: str  # "§ 12"
    section_title: str  # "Wiederholung von Prüfungen"
    page_start: int
    page_end: int
    header: str  # "[StuFPO ISSS 2024 | § 12 Wiederholung von Prüfungen]"
    text: str  # the passage itself, as it stands in the PDF (cleaned)
    n_words: int

    @property
    def index_text(self) -> str:
        """What gets embedded / BM25-indexed from Phase 3 on: header + passage."""
        return f"{self.header}\n{self.text}"


# --------------------------------------------------------------------------------------
# Stage 1: read the PDF
# --------------------------------------------------------------------------------------


def extract_pages(pdf_path: Path) -> list[Page]:
    """Plain text of each page. `sort=True` orders text blocks top-to-bottom."""
    with pymupdf.open(pdf_path) as pdf:
        return [Page(i, page.get_text("text", sort=True)) for i, page in enumerate(pdf, start=1)]


# --------------------------------------------------------------------------------------
# Stage 2: remove running headers, footers and page numbers
# --------------------------------------------------------------------------------------

ZONE = 2  # how many lines at the top and at the bottom of a page can be "furniture"
_PAGE_NUMBER_RE = re.compile(
    r"^(?:Seite\s+)?\d{1,3}(?:\s*(?:von|/)\s*\d{1,3})?$|^-\s*\d{1,3}\s*-$", re.IGNORECASE
)


def _shape(line: str) -> str:
    """Line with digits masked, so 'Seite 3 von 26' and 'Seite 4 von 26' look the same."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def _is_structural(line: str) -> bool:
    """Lines that carry content structure are never furniture, however often they repeat."""
    return line.startswith(("§", "Anlage", "Anhang")) or bool(re.match(r"^\(\d+[a-z]?\)", line))


def _zone_indexes(lines: list[str]) -> set[int]:
    non_empty = [i for i, line in enumerate(lines) if line]
    return set(non_empty[:ZONE] + non_empty[-ZONE:])


def strip_page_furniture(pages: list[Page]) -> tuple[list[Page], Counter]:
    """Drop page numbers and lines that repeat in the header/footer zone of many pages.

    A line is removed only if it sits in the top/bottom `ZONE` lines of its page AND
    (it is a bare page number OR its digit-masked shape appears in that zone on at least
    half of the pages, and at least 4 pages). Returns the cleaned pages and a counter of what
    was removed (shown in the manifest, so you can check it removed the right things).
    """
    per_page = [[line.strip() for line in p.text.splitlines()] for p in pages]

    zone_shapes: Counter = Counter()
    for lines in per_page:
        zone_shapes.update({_shape(lines[i]) for i in _zone_indexes(lines)})
    needed = max(4, math.ceil(0.5 * len(pages)))
    repeating = {s for s, n in zone_shapes.items() if n >= needed and len(s.split()) <= 20}

    removed: Counter = Counter()
    cleaned: list[Page] = []
    for page, lines in zip(pages, per_page, strict=True):
        zone = _zone_indexes(lines)
        kept = []
        for i, line in enumerate(lines):
            is_furniture = (
                i in zone
                and line
                and not _is_structural(line)
                and (_PAGE_NUMBER_RE.match(line) or _shape(line) in repeating)
            )
            if is_furniture:
                removed[_shape(line)] += 1
            else:
                kept.append(line)
        cleaned.append(Page(page.number, "\n".join(kept)))
    return cleaned, removed


# --------------------------------------------------------------------------------------
# Stage 3: lines and character clean-up
# --------------------------------------------------------------------------------------

_CHAR_MAP = str.maketrans(
    {
        "ﬁ": "fi",  # ligatures
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        " ": " ",  # non-breaking space
        "‐": "-",  # typographic hyphens -> plain hyphen
        "‑": "-",
    }
)
_SOFT_HYPHEN = "­"
# Table-of-contents dot leaders ("Geltungsbereich ........ 3") become " … " so that these long
# lines stay short and are easy to recognise.
_LEADER_RE = re.compile(r"\s*\.{4,}\s*")
LEADER = "…"


def clean_line(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw).translate(_CHAR_MAP)
    text = " ".join(text.split())
    if text.endswith(_SOFT_HYPHEN):  # a soft hyphen at the end of a line IS a line-break hyphen
        text = text[:-1] + "-"
    text = text.replace(_SOFT_HYPHEN, "")
    return _LEADER_RE.sub(f" {LEADER} ", text).strip()


def pages_to_lines(pages: list[Page]) -> list[Line]:
    lines = []
    for page in pages:
        for raw in page.text.splitlines():
            text = clean_line(raw)
            if text:
                lines.append(Line(page.number, text))
    return lines


# --------------------------------------------------------------------------------------
# Stage 4: undo line-break hyphenation
# --------------------------------------------------------------------------------------

_HYPHEN_END_RE = re.compile(r"[A-Za-zÄÖÜäöüß]-$")
# After these words a hyphen is a real "elliptical" hyphen: "Bachelor- und Masterstudiengänge".
_ELLIPSIS_WORDS = {"und", "oder", "bzw", "sowie", "beziehungsweise", "bis", "noch"}


def _join_hyphenated(prev: str, nxt: str, keep_hyphen: bool) -> str:
    first_word = nxt.split()[0]
    if keep_hyphen:
        return prev + nxt  # English: "third-" + "party" -> "third-party"
    if first_word[0].islower():
        if first_word.rstrip(".,;:)").lower() in _ELLIPSIS_WORDS:
            return f"{prev} {nxt}"  # keep it: "Bachelor- und Masterstudiengänge"
        return prev[:-1] + nxt  # drop it:  "Prüfungs-" + "ordnung" -> "Prüfungsordnung"
    return prev + nxt  # next word is capitalised: real compound, "ECTS-" + "Punkte"


def dehyphenate(lines: list[Line], keep_hyphen: bool = False) -> list[Line]:
    """Join words that were split with a hyphen at the end of a line.

    German (default). The rule is a heuristic and has to guess whether the hyphen belongs to
    the word:
      * next line starts lower-case  -> the hyphen was only a line break, remove it
      * ...unless it starts with und/oder/bzw./sowie/bis -> "Bachelor- und Master...", keep it
      * next line starts upper-case or a digit -> a real compound hyphen, keep it

    English (`keep_hyphen=True`, used for the handbook). English text is rarely hyphenated by the
    typesetter, so a hyphen at a line end is almost always part of the word ("third-party"):
    the lines are joined and the hyphen stays.
    """
    out: list[Line] = []
    for line in lines:
        if out and _HYPHEN_END_RE.search(out[-1].text) and line.text[0].isalpha():
            prev = out.pop()
            out.append(Line(prev.page, _join_hyphenated(prev.text, line.text, keep_hyphen)))
        else:
            out.append(line)
    return out


# --------------------------------------------------------------------------------------
# Stage 5a (regulations): find headings and cut the document into sections
# --------------------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^§\s*(\d+[a-z]?)(?:\s+(.+))?$")
_APPENDIX_RE = re.compile(r"^(Anlage|Anhang)\s+(\d+[a-z]?)(?:[\s:.–-]+(.+))?$")
_PRAEAMBEL_RE = re.compile(r"^Präambel:?$")
_PART_RE = re.compile(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.$")  # "II." on its own line
_SIGNATURE_RE = re.compile(r"^Ausgefertigt\b")
# A line like "§ 12 Abs. 2 gilt entsprechend" is a cross-reference that happens to start a line.
_REFERENCE_RE = re.compile(r"^(?:Abs|Absatz|Sätze|Satz|Nrn?|Ziffer|Ziff|Buchst|Halbsatz)\b")
_TOC_TITLE_RE = re.compile(
    r"^(?:Inhaltsübersicht|Inhaltsverzeichnis|Table of Contents)$", re.IGNORECASE
)
_TRAILING_PAGE_RE = re.compile(r"\s\d{1,3}$")  # "... Geltungsbereich … 3"
MAX_TOC_LINES = 400  # give up looking for the end of a table of contents after this many lines


@dataclass(frozen=True)
class Heading:
    key: str  # "§ 12" / "Anhang 1" / "Teil II" / "Präambel" / "Ausfertigung"
    title: str
    title_on_next_line: bool = False
    line_is_text: bool = False  # the heading line is also content and stays in the section


def _plausible_title(text: str) -> bool:
    return (
        0 < len(text) <= 100
        and text[0].isupper()
        and text[-1] not in ".,;:"
        and not text.startswith(("Anlage ", "Anhang "))
        and not _REFERENCE_RE.match(text)
    )


def parse_heading(line: str, next_line: str = "") -> Heading | None:
    """Return a Heading if `line` starts a new section, else None.

    Handles "§ 12 Title", "§ 12" + title on the next line, "Anhang 1: Title", "Anlage 1 Title",
    a Roman part number on its own line ("II." + title), "Präambel" and the signature block
    ("Ausgefertigt aufgrund ...").
    Rejects cross-references that wrapped to the start of a line ("§ 12 Abs. 2 gilt ...").
    """
    if _PRAEAMBEL_RE.match(line):
        return Heading("Präambel", "")
    if _SIGNATURE_RE.match(line):
        return Heading("Ausfertigung", "", line_is_text=True)
    if _PART_RE.match(line):
        if next_line and _plausible_title(next_line):
            return Heading(f"Teil {line[:-1]}", next_line, title_on_next_line=True)
        return None
    section = _SECTION_RE.match(line)
    appendix = _APPENDIX_RE.match(line)
    if section:
        key, title = f"§ {section.group(1)}", (section.group(2) or "").strip()
    elif appendix:
        key, title = f"{appendix.group(1)} {appendix.group(2)}", (appendix.group(3) or "").strip()
    else:
        return None
    if title:
        return Heading(key, title) if _plausible_title(title) else None
    if next_line and _plausible_title(next_line):
        return Heading(key, next_line, title_on_next_line=True)
    return Heading(key, "")  # the line is exactly "§ 12" or "Anlage 1"


def find_toc_range(lines: list[Line]) -> tuple[int, int] | None:
    """Locate the table of contents as a half-open index range [start, end), or None.

    It starts at a line "Inhaltsübersicht" / "Inhaltsverzeichnis". Its entries look like
    headings ("§ 1 Geltungsbereich … 3"), so without this step every entry would become a tiny
    chunk. The scan stops where the real text begins, which we recognise by either
      * a heading WITHOUT a trailing page number (when the entries have page numbers), or
      * a heading we have already seen in the table (when they do not).
    When the entries have page numbers, the table ends right after its last entry, so the
    sentence between the table and the first "§" (the enacting formula) is kept.
    If no end is found within MAX_TOC_LINES, nothing is removed (better noisy than lossy).
    """
    start = next((i for i, line in enumerate(lines) if _TOC_TITLE_RE.match(line.text)), None)
    if start is None:
        return None
    seen: set[str] = set()
    numbered_style: bool | None = None
    last_entry: int | None = None

    def end_at(j: int) -> tuple[int, int]:
        return start, (last_entry + 1 if numbered_style and last_entry is not None else j)

    for j in range(start + 1, min(len(lines), start + 1 + MAX_TOC_LINES)):
        text = lines[j].text
        nxt = lines[j + 1].text if j + 1 < len(lines) else ""
        heading = parse_heading(text, nxt)
        has_page_number = bool(_TRAILING_PAGE_RE.search(text))
        if heading is None:
            if LEADER in text and has_page_number:
                last_entry = j  # "I. Allgemeine Regelungen … 5", or the rest of a wrapped entry
            elif len(text) > 120:  # long prose line: the table of contents is over
                return end_at(j)
            continue
        if numbered_style is None:
            numbered_style = has_page_number
        wrapped_entry = LEADER in nxt and bool(_TRAILING_PAGE_RE.search(nxt))
        if heading.key in seen or (numbered_style and not has_page_number and not wrapped_entry):
            return end_at(j)
        seen.add(heading.key)
        if has_page_number:
            last_entry = j
    return None


def parse_sections(
    lines: list[Line], toc_range: tuple[int, int] | None, drop_front_matter: bool = False
) -> list[Section]:
    """Cut the line list into sections.

    With a table of contents and `drop_front_matter`, everything up to the end of the table is
    skipped and the text before the first heading is called "Eingangsformel". Otherwise the
    table of contents alone is skipped and text before the first heading is called "Vorspann".
    """
    begin, label = 0, "Vorspann"
    if toc_range and drop_front_matter:
        begin, label, toc_range = toc_range[1], "Eingangsformel", None
    sections: list[Section] = []
    current = Section(label, "", lines[begin].page if begin < len(lines) else 1)
    i = begin
    while i < len(lines):
        if toc_range and toc_range[0] <= i < toc_range[1]:
            i = toc_range[1]
            continue
        nxt = lines[i + 1].text if i + 1 < len(lines) else ""
        heading = parse_heading(lines[i].text, nxt)
        if heading:
            if current.lines:
                sections.append(current)
            current = Section(heading.key, heading.title, lines[i].page)
            if heading.line_is_text:
                current.lines.append(lines[i])
            i += 2 if heading.title_on_next_line else 1
        else:
            current.lines.append(lines[i])
            i += 1
    if current.lines:
        sections.append(current)
    return sections


# --------------------------------------------------------------------------------------
# Stage 5b (module handbook): one section per module
# --------------------------------------------------------------------------------------

# Every page of a module starts with a running header line "Module GdI-IFP-M"; the first page
# of the module also has "Module GdI-IFP-M <title>   6 ECTS / 180 h".
_MODULE_HEADER_RE = re.compile(r"^Module\s+(\S+)$")
_ECTS_RE = re.compile(r"\s*\d+\s*ECTS\s*/\s*\d+\s*h\b")
_TITLE_STOP_RE = re.compile(r"^(?:\(since|Person responsible|Contents:|PLEASE NOTE)", re.IGNORECASE)
_FIELD_RE = re.compile(
    r"^(?:(?:Contents|Learning outcomes?|Remark|Prerequisites for the module|"
    r"Recommended prior knowledge|Frequency|Description|Literature|Language|Mode of Delivery|"
    r"Lecturers?):|Module Units$|Examination$)",
    re.IGNORECASE,
)


def _module_id(texts: list[str]) -> str | None:
    """The module of a page, from its running header (one of the first two lines)."""
    for text in texts[:2]:
        match = _MODULE_HEADER_RE.match(text)
        # Module ids always contain a hyphen ("GdI-IFP-M"). "Module Handbook" (the cover) and
        # "Module Units" (a field label) are not modules.
        if match and "-" in match.group(1):
            return match.group(1)
    return None


def _module_title(module_id: str, texts: list[str]) -> str:
    """English + German title from the lines after "Module <id> <title> 6 ECTS / 180 h"."""
    start_re = re.compile(rf"^Module\s+{re.escape(module_id)}\s+(.+)$")
    for i, text in enumerate(texts):
        match = start_re.match(text)
        if not match:
            continue
        first = _clean_title(match.group(1))
        parts = [first]
        for nxt in texts[i + 1 : i + 5]:
            if _TITLE_STOP_RE.match(nxt):
                return _dedupe_title(_join_title(parts))[:140]
            parts.append(_clean_title(nxt))
        return first[:140]  # no stop line close by: the title is only the first line
    return ""


_WORKLOAD_RE = re.compile(r"\s*\d+\s*h\s+(?:Präsenzzeit|Selbststudium)\b")


def _clean_title(text: str) -> str:
    """Remove "6 ECTS / 180 h" and "40 h Präsenzzeit" fragments that sit beside the title."""
    return _WORKLOAD_RE.sub("", _ECTS_RE.sub("", text)).strip()


def _join_title(parts: list[str]) -> str:
    """Join title lines with a space, but without one after a hyphen ("Higher-" + "Order")."""
    title = ""
    for part in (p for p in parts if p):
        title = title + part if title.endswith("-") else f"{title} {part}".strip()
    return title


def _dedupe_title(title: str) -> str:
    """Several modules repeat the English title as their German title: keep it once."""
    words = title.split()
    half = len(words) // 2
    if len(words) % 2 == 0 and words[:half] == words[half:]:
        return " ".join(words[:half])
    return title


def handbook_sections(pages: list[Page]) -> list[Section]:
    """One section per module. Pages outside any module become "Allgemeines" / "Anhang"."""
    groups: list[tuple[str | None, list[Line]]] = []
    for page in pages:
        texts = [t for t in (clean_line(raw) for raw in page.text.splitlines()) if t]
        module = _module_id(texts)
        if module:  # drop the running header; it is added back through the chunk header
            texts = [t for t in texts if t != f"Module {module}"]
        lines = [Line(page.number, t) for t in texts]
        if groups and groups[-1][0] == module:
            groups[-1][1].extend(lines)
        else:
            groups.append((module, lines))

    sections: list[Section] = []
    seen_module = False
    for module, lines in groups:
        lines = dehyphenate(lines, keep_hyphen=True)
        if not lines:
            continue
        if module:
            seen_module = True
            title = _module_title(module, [line.text for line in lines])
            sections.append(Section(f"Modul {module}", title, lines[0].page, lines))
            continue
        lines = [ln for ln in lines if LEADER not in ln.text and not _TOC_TITLE_RE.match(ln.text)]
        if lines:
            label = "Anhang" if seen_module else "Allgemeines"
            sections.append(Section(label, "", lines[0].page, lines))
    return sections


# --------------------------------------------------------------------------------------
# Stage 6: cut sections into chunks
# --------------------------------------------------------------------------------------

_SUBPARA_RE = re.compile(r"^\(\d+[a-z]?\)\s")  # "(1) ", "(2a) "
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÄÖÜ(])")


@dataclass(frozen=True)
class Piece:
    """The smallest thing the packer moves around: a sub-paragraph, or part of a long one."""

    text: str
    page_start: int
    page_end: int
    sep: str = "\n"  # separator before this piece when it is not the first in its chunk

    @property
    def words(self) -> int:
        return len(self.text.split())


def _hard_split(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _split_oversize(text: str, max_words: int) -> list[str]:
    """Split one very long sub-paragraph at sentence ends (and, if needed, inside a sentence)."""
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for piece in _hard_split(sentence, max_words):
            n = len(piece.split())
            if current and count + n > max_words:
                parts.append(" ".join(current))
                current, count = [], 0
            current.append(piece)
            count += n
    if current:
        parts.append(" ".join(current))
    return parts


def section_pieces(
    section: Section, max_words: int, unit_start: re.Pattern[str] = _SUBPARA_RE
) -> list[Piece]:
    """Group a section's lines into units: a new one starts at every line matching `unit_start`.

    Regulations start a unit at every "(n) "; the handbook at every field label ("Contents:").
    A unit longer than `max_words` is cut at sentence ends.
    """
    units: list[list[Line]] = []
    for line in section.lines:
        if not units or unit_start.match(line.text):
            units.append([line])
        else:
            units[-1].append(line)

    pieces: list[Piece] = []
    for unit in units:
        text = " ".join(line.text for line in unit)
        first_page, last_page = unit[0].page, unit[-1].page
        if len(text.split()) <= max_words:
            pieces.append(Piece(text, first_page, last_page))
        else:
            for k, part in enumerate(_split_oversize(text, max_words)):
                pieces.append(Piece(part, first_page, last_page, sep="\n" if k == 0 else " "))
    return pieces


def pack_pieces(pieces: list[Piece], max_words: int, min_words: int) -> list[list[Piece]]:
    """Greedy packing: fill a chunk until the next piece would exceed max_words."""
    groups: list[list[Piece]] = []
    current: list[Piece] = []
    count = 0
    for piece in pieces:
        if current and count + piece.words > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(piece)
        count += piece.words
    if current:
        groups.append(current)

    # A tiny last chunk is merged into the one before it, if that stays within 120 % of max_words.
    if len(groups) >= 2:
        tail = sum(p.words for p in groups[-1])
        before = sum(p.words for p in groups[-2])
        if tail < min_words and before + tail <= max_words * 1.2:
            groups[-2].extend(groups.pop())
    return groups


def _render(group: list[Piece]) -> str:
    text = group[0].text
    for piece in group[1:]:
        text += piece.sep + piece.text
    return text.strip()


def _slug(label: str) -> str:
    """'§ 12a' -> 'par12a', 'Anhang 1' -> 'anhang1', 'Modul GdI-IFP-M' -> 'modulgdi-ifp-m'."""
    text = label.lower().replace("§", "par")
    for src, dst in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(src, dst)
    return re.sub(r"[^a-z0-9-]", "", text)


def make_header(doc: DocConfig, label: str, title: str) -> str:
    where = f"{label} {title}".strip()
    return f"[{doc.short_title} | {where}]"


def chunk_section(
    section: Section,
    doc: DocConfig,
    cfg: ChunkConfig,
    counters: Counter,
    unit_start: re.Pattern[str] = _SUBPARA_RE,
) -> list[Chunk]:
    slug = _slug(section.label)
    chunks = []
    pieces = section_pieces(section, cfg.max_words, unit_start)
    for group in pack_pieces(pieces, cfg.max_words, cfg.min_words):
        text = _render(group)
        index = counters[slug]
        counters[slug] += 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{slug}:{index}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                version=doc.version,
                section=section.label,
                section_title=section.title,
                page_start=min(p.page_start for p in group),
                page_end=max(p.page_end for p in group),
                header=make_header(doc, section.label, section.title),
                text=text,
                n_words=len(text.split()),
            )
        )
    return chunks


# --------------------------------------------------------------------------------------
# Putting the stages together
# --------------------------------------------------------------------------------------


def build_chunks(pages: list[Page], doc: DocConfig, cfg: ChunkConfig) -> tuple[list[Chunk], dict]:
    """Everything after PDF extraction. Pure function: easy to test without a PDF."""
    pages, removed = strip_page_furniture(pages)
    info: dict = {
        "pages": len(pages),
        "furniture_removed": dict(removed.most_common(10)),
        "front_matter_lines_removed": 0,
        "toc_removed_lines": 0,
    }
    if doc.kind == "handbook":
        sections = handbook_sections(pages)
        unit_start = _FIELD_RE
    else:
        lines = dehyphenate(pages_to_lines(pages))
        toc_range = find_toc_range(lines)
        if toc_range:
            info["toc_removed_lines"] = toc_range[1] - toc_range[0]
            if cfg.drop_front_matter:
                info["front_matter_lines_removed"] = toc_range[0]
        sections = parse_sections(lines, toc_range, cfg.drop_front_matter)
        unit_start = _SUBPARA_RE

    counters: Counter = Counter()
    chunks = [c for s in sections for c in chunk_section(s, doc, cfg, counters, unit_start)]
    info["sections"] = len(sections)
    info["chunks"] = len(chunks)
    info["words"] = sum(c.n_words for c in chunks)
    return chunks, info


def process_pdf(doc: DocConfig, cfg: ChunkConfig) -> tuple[list[Chunk], dict]:
    path = Path(doc.file)
    chunks, info = build_chunks(extract_pages(path), doc, cfg)
    info["doc_id"] = doc.doc_id
    info["file"] = doc.file
    info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return chunks, info


# --------------------------------------------------------------------------------------
# Config and files
# --------------------------------------------------------------------------------------


def load_config(path: str | Path) -> tuple[ChunkConfig, list[DocConfig]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = ChunkConfig(**raw.get("chunking", {}))
    docs = [DocConfig(**d) for d in raw["documents"]]

    if not 0 < cfg.min_words < cfg.max_words:
        raise ValueError("chunking.min_words must be > 0 and smaller than max_words")
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("documents[].doc_id must be unique")
    for d in docs:
        if d.kind not in {"regulation", "handbook"}:
            raise ValueError(f"{d.doc_id}: kind must be 'regulation' or 'handbook', not {d.kind!r}")
    return cfg, docs


def write_jsonl(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PDFs -> chunks.jsonl (Phase 1)")
    parser.add_argument("--config", default="configs/ingest.yaml")
    parser.add_argument("--out", default="data/processed/chunks.jsonl")
    args = parser.parse_args(argv)

    cfg, docs = load_config(args.config)
    missing = [d for d in docs if not Path(d.file).is_file()]
    if missing:
        print("These PDFs are missing (see data/raw/sources.md for the download links):")
        for d in missing:
            print(f"  {d.file}")
        return 2

    all_chunks: list[Chunk] = []
    manifest_docs = []
    for doc in docs:
        chunks, info = process_pdf(doc, cfg)
        all_chunks.extend(chunks)
        manifest_docs.append(info)
        print(
            f"{doc.doc_id}: {info['pages']} pages -> {info['chunks']} chunks, "
            f"{info['words']} words, {info['sections']} sections; removed "
            f"{info['front_matter_lines_removed']} front-matter and "
            f"{info['toc_removed_lines']} table-of-contents lines"
        )
        print(f"    sha256 {info['sha256']}")

    write_jsonl(all_chunks, args.out)
    manifest_path = Path(args.out).with_name("ingest_manifest.json")
    manifest = {"chunking": asdict(cfg), "documents": manifest_docs}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(all_chunks)} chunks to {args.out} and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
