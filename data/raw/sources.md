# Sources

The PDFs are not committed to the repo. This file records exactly which documents the
experiments use, so every number in the results table can be traced to a document version.

Download the three files into this folder (`data/raw/`) and keep the file names.
Then fill in the last two columns (the SHA-256 is printed by `make chunks`).

| doc_id | Document | Direct PDF link | Version | Downloaded on | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `stufpo-isss-2024` | Studien- und Fachprüfungsordnung (StuFPO), M.Sc. International Software Systems Science | https://www.uni-bamberg.de/fileadmin/abt-studium/Pruefungs-Studienordnungen/WIAI/ISSS-MA/StuFPO_MA_ISSS_2024-1.pdf | 24.09.2024, zuletzt geändert 31.03.2025 | | |
| `apo-wiai-2026` | Allgemeine Prüfungsordnung (APO) der Fakultät WIAI, konsolidierte Fassung | https://www.uni-bamberg.de/fileadmin/abt-studium/Pruefungs-Studienordnungen/APO/WIAI/APO-BA-MA-WIAI-26.pdf | 31.03.2008, zuletzt geändert 05.08.2026 | | |
| `mhb-isss-2024` | Modulhandbuch M.Sc. International Software Systems Science, Studienbeginn ab WS 2024/25 | https://www.uni-bamberg.de/fileadmin/abt-studium/Modulhandbuecher/WIAI/ISSS-MA/MHB_MA_ISSS_2024.pdf | Studienbeginn ab WS 2024/25 | | |

## Landing pages (use these if a direct link has moved)

- StuFPO: https://www.uni-bamberg.de/abt-studium/aufgaben/pruefungs-studienordnungen/master-studiengaenge/international-software-systems-science/
- APO WIAI: https://www.uni-bamberg.de/abt-studium/aufgaben/pruefungs-studienordnungen/master-studiengaenge/allgemeine-pruefungsordnung-fakultaet-wirtschaftsinformatik-und-angewandte-informatik/
- Modulhandbuch: https://www.uni-bamberg.de/abt-studium/aufgaben/modulhandbuecher/wiai/modulhandbuch-master-international-software-systems-science/

## Which StuFPO version applies?

The StuFPO of 29.09.2025 (`.../amtliche-veroeffentlichungen/2025/2025-81.pdf`) replaces the 2024 version
"subject to transitional rules". The default here is the 2024 version, because it was in force at the
start of the program for the WS 2024/25 cohort. Read the transitional rules (Übergangsvorschriften) at the
end of the 2025 StuFPO to check which version covers your cohort, and swap the `file:` entry in
`configs/ingest.yaml` if needed.
