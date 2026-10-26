PYTHON ?= python

.PHONY: chunks inspect test lint

chunks:  ## PDFs in data/raw -> data/processed/chunks.jsonl
	$(PYTHON) -m studybot.ingest --config configs/ingest.yaml --out data/processed/chunks.jsonl

inspect:  ## stats, warnings and 10 random chunks to read by eye
	$(PYTHON) -m studybot.inspect_chunks --chunks data/processed/chunks.jsonl -n 10

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests
