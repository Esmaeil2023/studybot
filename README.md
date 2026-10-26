# StudyBot

Bilingual (German/English) question answering over a university's study and exam regulations.
Answers cite the exact paragraph and page. Retrieval and answer quality are measured on a
hand-written test set.

**Status:** Phase 1 (parsing and chunking). Later phases add the test set, retrieval, reranking,
answer generation, evaluation, the API and the deployment.

## Quickstart

Run everything from the repo root.

**1. Environment** (Python 3.10 or newer)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**2. Check that the tests pass** (they use a synthetic PDF, no downloads needed)

```bash
make test                          # without make: python -m pytest -q
```

**3. Download the three PDFs** into `data/raw/`. The links are in `data/raw/sources.md`.

```bash
cd data/raw
curl -O https://www.uni-bamberg.de/fileadmin/abt-studium/Pruefungs-Studienordnungen/WIAI/ISSS-MA/StuFPO_MA_ISSS_2024-1.pdf
curl -O https://www.uni-bamberg.de/fileadmin/abt-studium/Pruefungs-Studienordnungen/APO/WIAI/APO-BA-MA-WIAI-26.pdf
curl -O https://www.uni-bamberg.de/fileadmin/abt-studium/Modulhandbuecher/WIAI/ISSS-MA/MHB_MA_ISSS_2024.pdf
cd ../..
```

If a link no longer works, download from the landing pages listed in `data/raw/sources.md`
and keep the file names (or change `file:` in `configs/ingest.yaml`).

**4. Build the chunks**

```bash
make chunks      # without make: python -m studybot.ingest --config configs/ingest.yaml --out data/processed/chunks.jsonl
```

It prints one line per document and the SHA-256 of each PDF. Paste the hashes into
`data/raw/sources.md`.

**5. Read the chunks.** This is the most important step of Phase 1.

```bash
make inspect     # without make: python -m studybot.inspect_chunks -n 10
```

Go through this list:

1. Does every document show a plausible number of sections? (The StuFPO and APO should have
   one section per `§`, plus `Anlage` sections.)
2. Under **Checks**: is there a `no section found for § [...]` line? A missing number usually
   means a heading was not recognised.
3. Read all 10 random chunks. Is each one a clean, self-contained passage? Any leftover
   page numbers, headers, table-of-contents lines or broken words?
4. Look at `data/processed/ingest_manifest.json`, key `furniture_removed`. Only headers, footers
   and page numbers should be listed there.
5. Open one PDF next to its chunks and compare one whole `§` word by word.

## How Phase 1 works

`src/studybot/ingest.py` has one function per stage:

| Stage | Function | What it does |
| --- | --- | --- |
| 1 | `extract_pages` | text of each page, page numbers kept (PyMuPDF) |
| 2 | `strip_page_furniture` | removes running headers, footers, page numbers |
| 3 | `pages_to_lines` | splits into lines, normalises spaces, ligatures, soft hyphens |
| 4 | `dehyphenate` | `Prüfungs-` + `ordnung` becomes `Prüfungsordnung`; keeps `Bachelor- und` and `ECTS-Punkte` |
| 5 | `find_toc_range`, `parse_sections` | skips the table of contents; cuts at `§ N`, `Anlage N`, `Präambel` |
| 6 | `chunk_section` | one chunk per `§`; longer ones are cut at `(1)`, `(2)`, then at sentences |

Each line of `data/processed/chunks.jsonl` is one chunk:

| Field | Example |
| --- | --- |
| `chunk_id` | `stufpo-isss-2024:par12:0` |
| `section`, `section_title` | `§ 12`, `Wiederholung von Prüfungen` |
| `page_start`, `page_end` | `9`, `9` |
| `header` | `[StuFPO ISSS 2024 \| § 12 Wiederholung von Prüfungen]` |
| `text` | the passage itself |

From Phase 3 on, the search indexes use `header + "\n" + text` (`Chunk.index_text`), so a passage is
understandable on its own. The evaluation matches gold evidence against `text` only.

## Known limitations

- The parser is tested on synthetic PDFs (regulation and handbook) and was checked by eye on the
  three real PDFs. Other PDFs can differ; the checks in step 5 are there to find out.
- The module handbook is cut per module, using the running "Module <ID>" header of each page.
  Within a module, long text is split at field labels ("Contents:", "Learning outcomes:", ...).
- Front matter (amendment table) and the table of contents are dropped. The enacting formula
  ("Aufgrund des ...") is kept as its own section "Eingangsformel", the signature block as
  "Ausfertigung". Roman part headings ("I. Allgemeine Regelungen") only end a section.
- Hyphen repair is a heuristic: in German, a hyphen at a line end before a lower-case word is
  removed, unless that word is `und`, `oder`, `bzw.`, `sowie`, `bis` or `noch`. In the English
  handbook the hyphen is kept ("third-party").
- Sentence numbers inside a line can stay glued to the text; this is cosmetic.
- Expected, harmless inspector warnings on the real PDFs: one very short chunk (a one-line
  section) and one repeated text (a boilerplate sentence).

## Layout

```
configs/ingest.yaml        which PDFs, chunk size limits
data/raw/                  the PDFs (not committed), sources.md lists them
data/processed/            chunks.jsonl and ingest_manifest.json (generated)
src/studybot/ingest.py     PDFs -> chunks
src/studybot/inspect_chunks.py
tests/                     pytest, with a synthetic regulation PDF
```

## License

MIT, see `LICENSE`.
