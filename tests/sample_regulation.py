"""Builds small synthetic PDFs that have the typical problems of the real Bamberg documents.

`build_pdf` writes a German regulation. Each page is a list of physical lines, written exactly
as they should appear in the PDF, so the tests know the expected result. What it contains:

  * a title page with a list of amending statutes and URLs (front matter)
  * a running header and a "Seite N von M" footer on every page
  * a table of contents (Inhaltsverzeichnis) with dot leaders, long lines, a Roman part line,
    and one entry that wraps over two lines
  * the enacting formula between the table of contents and the first part
  * Roman part headings on their own line ("I." + title on the next line)
  * headings with the title on the same line ("§ 1 Geltungsbereich") and on the next line ("§ 2")
  * a cross-reference that starts a line ("§ 12 Abs. 2 der Allgemeinen ...") - NOT a heading
  * line-break hyphenation: "Prüfungs-" / "ordnung", the elliptical "Bachelor-" / "und ...",
    and the real compound hyphen "ECTS-" / "Punkte"
  * a section that continues on the next page (§ 3)
  * a long section with six sub-paragraphs, the last one longer than the small test limit (§ 6)
  * an "Anhang" with numbered list lines, followed by the signature block

`build_handbook_pdf` writes a small English module handbook: a cover, a table of contents,
two modules with a running "Module <id>" header on every page, and an appendix page.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pymupdf

HEADER = "Amtliche Veröffentlichungen der Testuniversität Musterstadt Nr. 81"
LEADER = "." * 80

# (1) .. (6) of § 6, kept as paragraphs so tests can compare the words that come back out.
PARAGRAPHS_6 = [
    "(1) Die Masterarbeit soll zeigen, dass die Studierenden in der Lage sind, innerhalb einer "
    "vorgegebenen Frist ein Problem aus dem Fachgebiet selbstständig nach wissenschaftlichen "
    "Methoden zu bearbeiten.",
    "(2) Das Thema der Masterarbeit wird von einer Betreuerin oder einem Betreuer vergeben. "
    "Die Ausgabe des Themas ist aktenkundig zu machen.",
    "(3) Die Bearbeitungszeit beträgt sechs Monate. Auf begründeten Antrag kann der "
    "Prüfungsausschuss die Frist einmalig um bis zu vier Wochen verlängern.",
    "(4) Die Masterarbeit ist fristgerecht in elektronischer Form und in einem gedruckten "
    "Exemplar beim Prüfungsamt abzugeben. Der Abgabezeitpunkt ist zu vermerken.",
    "(5) Die Arbeit wird von zwei Prüfenden bewertet. Weichen die Noten um mehr als zwei "
    "Notenstufen voneinander ab, wird eine dritte Prüferin oder ein dritter Prüfer bestellt.",
    "(6) Studienleistungen, die an einer ausländischen Hochschule erbracht wurden, werden auf "
    "Antrag der Studierenden anerkannt, sofern keine wesentlichen Unterschiede bestehen. "
    "Die Anerkennung erfolgt durch den Prüfungsausschuss nach Anhörung der zuständigen "
    "Fachvertreterinnen und Fachvertreter. Der Antrag ist spätestens im ersten Fachsemester "
    "nach dem Auslandsaufenthalt beim Prüfungsamt einzureichen. Dem Antrag sind die "
    "Modulbeschreibungen und ein Leistungsnachweis der Gasthochschule beizufügen. Werden Noten "
    "anerkannt, so sind sie zu übernehmen und in die Berechnung der Gesamtnote einzubeziehen. "
    "Ablehnende Entscheidungen sind schriftlich zu begründen und mit einer "
    "Rechtsbehelfsbelehrung zu versehen. Die Studierenden können gegen die Entscheidung "
    "innerhalb eines Monats Widerspruch beim Prüfungsausschuss einlegen. Über den Widerspruch "
    "entscheidet der Prüfungsausschuss nach erneuter Beratung innerhalb von vier Wochen. Die "
    "Entscheidung über den Widerspruch ist ebenfalls schriftlich zu begründen und den "
    "Studierenden zuzustellen.",
]


def _wrap(paragraph: str) -> list[str]:
    return textwrap.wrap(paragraph, width=80, break_on_hyphens=False)


FRONT_MATTER = [
    "Studien- und Fachprüfungsordnung für den Masterstudiengang Beispielinformatik",
    "Vom 1. Oktober 2024",
    "(Fundstelle: https://www.example.de/amtliche-veroeffentlichungen/2024-76.pdf)",
    "geändert durch:",
    "Satzung zur Änderung der Studien- und Fachprüfungsordnung vom 31. März 2025",
]

TOC = [
    "Inhaltsverzeichnis",
    f"I. Allgemeine Regelungen {LEADER} 3",
    f"§ 1 Geltungsbereich {LEADER} 3",
    f"§ 2 Studienumfang und Regelstudienzeit {LEADER} 3",
    f"§ 3 Module und Prüfungsleistungen {LEADER} 3",
    f"§ 4 Anmeldung und Rücktritt {LEADER} 4",
    f"§ 5 Wiederholung von Prüfungen {LEADER} 5",
    f"II. Abschluss {LEADER} 6",
    f"§ 6 Masterarbeit {LEADER} 6",
    "Anhang 1: Studienverlaufsplan des Masterstudiengangs Beispielinformatik",
    f"und seiner Wahlbereiche {LEADER} 7",
]

PAGES: list[list[str]] = [
    # page 1: front matter
    FRONT_MATTER,
    # page 2: table of contents
    TOC,
    # page 3
    [
        "Aufgrund des Art. 9 Satz 1 des Beispielgesetzes erlässt die Testuniversität",
        "Musterstadt folgende Satzung:",
        "I.",
        "Allgemeine Regelungen",
        "§ 1 Geltungsbereich",
        "(1) Diese Ordnung regelt Studium und Prüfungen im Masterstudiengang",
        "Beispielinformatik. Sie ergänzt die Allgemeine Prüfungsordnung für Bachelor-",
        "und Masterstudiengänge der Fakultät.",
        "(2) Soweit diese Ordnung nichts anderes bestimmt, gilt die Allgemeine Prüfungs-",
        "ordnung in ihrer jeweils geltenden Fassung.",
        "§ 2",
        "Studienumfang und Regelstudienzeit",
        "(1) Der Masterstudiengang umfasst 120 ECTS-",
        "Punkte und ist in vier Semestern zu studieren.",
        "(2) Das Studium kann zum Winter- und zum Sommersemester begonnen werden.",
    ],
    # page 4: § 3 starts and continues on page 5
    [
        "§ 3 Module und Prüfungsleistungen",
        "(1) Das Studium gliedert sich in Pflichtmodule, Wahlpflichtmodule und die",
        "Masterarbeit. Jedes Modul schließt mit einer Prüfungsleistung ab.",
        "(2) Prüfungsleistungen sind schriftliche Klausuren, mündliche Prüfungen oder",
        "Projektarbeiten. Die Prüfungsform wird im Modulhandbuch festgelegt.",
        "(3) Die Anmeldung zu Prüfungen ist nur innerhalb der festgelegten Fristen möglich;",
        "§ 12 Abs. 2 der Allgemeinen Prüfungsordnung bleibt unberührt.",
    ],
    # page 5
    [
        "(4) Die Bewertung der Prüfungsleistungen erfolgt nach der Notenskala der",
        "Allgemeinen Prüfungsordnung. Eine Prüfung ist bestanden, wenn sie mindestens mit",
        "der Note 4,0 bewertet wurde.",
        "§ 4 Anmeldung und Rücktritt",
        "(1) Die Anmeldung zu Prüfungen erfolgt elektronisch über das Prüfungsverwaltungs-",
        "system. Eine Abmeldung ist bis eine Woche vor dem Prüfungstermin möglich.",
        "(2) Ein Rücktritt von einer Prüfung ohne Angabe von Gründen ist nur bis zum",
        "genannten Termin zulässig. Bei Krankheit ist unverzüglich ein ärztliches Attest",
        "vorzulegen.",
    ],
    # page 6
    [
        "§ 5",
        "Wiederholung von Prüfungen",
        "(1) Nicht bestandene Modulprüfungen können einmal wiederholt werden. Die",
        "Wiederholung soll spätestens im folgenden Prüfungszeitraum stattfinden.",
        "(2) Eine zweite Wiederholung ist nur in zwei Modulen des Studiengangs möglich.",
        "(3) Die Masterarbeit kann bei Nichtbestehen einmal mit neuem Thema wiederholt werden.",
        "II.",
        "Abschluss",
    ],
    # page 7
    ["§ 6 Masterarbeit", *[line for p in PARAGRAPHS_6 for line in _wrap(p)]],
    # page 8: appendix and signature block
    [
        "Anhang 1: Studienverlaufsplan des Masterstudiengangs Beispielinformatik",
        "1. Semester: Pflichtmodule im Umfang von 30 ECTS-Punkten.",
        "2. Semester: Wahlpflichtmodule im Umfang von 30 ECTS-Punkten.",
        "3. Semester: Projekt und Wahlpflichtmodule im Umfang von 30 ECTS-Punkten.",
        "4. Semester: Masterarbeit im Umfang von 30 ECTS-Punkten.",
        "Ausgefertigt aufgrund des Beschlusses der Hochschulleitung vom 11. März 2024.",
        "Musterstadt, 1. Oktober 2024",
    ],
]


def build_pdf(path: Path, pages: list[list[str]] = PAGES, header: str = HEADER) -> Path:
    """Write `pages` to a PDF: header on top, body lines, "Seite N von M" at the bottom."""
    pdf = pymupdf.open()
    total = len(pages)
    for number, lines in enumerate(pages, start=1):
        page = pdf.new_page(width=595, height=842)
        page.insert_text((60, 40), header, fontsize=8, fontname="helv")
        y = 90
        for line in lines:
            page.insert_text((60, y), line, fontsize=9, fontname="helv")
            y += 14
        page.insert_text((250, 810), f"Seite {number} von {total}", fontsize=8, fontname="helv")
    pdf.save(path)
    pdf.close()
    return path


# ---------------------------------------------------------------------- module handbook

MODULE_A_TEXT = (
    "Contents: Students learn how to design algorithms and analyse their running time. "
    "The course covers sorting, graph search and dynamic programming. Typical exercises "
    "use third-party libraries and real data sets from the automotive industry.",
    "Learning outcomes: Students can choose a suitable data structure for a given problem "
    "and justify their choice. They can prove the correctness of simple algorithms.",
)
MODULE_B_TEXT = (
    "Contents: Introduction to the theory of formal languages and automata.",
    "Learning outcomes: Students can transform automata and apply the pumping lemma.",
)

HANDBOOK_PAGES: list[list[str]] = [
    # cover
    ["Module Handbook", "Test Master Programme", "Valid: 07.09.2026"],
    # table of contents with leaders
    [
        "Table of Contents",
        f"TEST-ALGO-M: Algorithms {LEADER} 3",
        f"TEST-AUTO-M: Automata {LEADER} 5",
    ],
    # module A, page 1
    [
        "Module TEST-ALGO-M",
        "Module TEST-ALGO-M Algorithms and                           6 ECTS / 180 h",
        "Data Structures",
        "Algorithmen und Datenstrukturen",
        "(since WS24/25)",
        "Person responsible for module: Prof. Dr. Test",
        *_wrap(MODULE_A_TEXT[0]),
        *_wrap(MODULE_A_TEXT[1]),
        "Module Units",
        "Frequency: every winter semester",
    ],
    # module A, page 2
    [
        "Module TEST-ALGO-M",
        "Examination",
        "Written examination of 90 minutes.",
        "Description: The exam covers lecture and tutorial contents.",
    ],
    # module B
    [
        "Module TEST-AUTO-M",
        "Module TEST-AUTO-M Automata Theory                          6 ECTS / 180 h",
        "(since SS25)",
        "Person responsible for module: Prof. Dr. Beispiel",
        *_wrap(MODULE_B_TEXT[0]),
        *_wrap(MODULE_B_TEXT[1]),
        "Examination",
        "Oral examination.",
    ],
    # appendix after the last module
    ["List of changes", "The module TEST-AUTO-M was introduced in SS25."],
]


def build_handbook_pdf(path: Path) -> Path:
    return build_pdf(path, HANDBOOK_PAGES, header="Test Master Programme 2026")
