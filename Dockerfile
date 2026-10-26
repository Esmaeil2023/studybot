# StudyBot service. Build from the project folder AFTER `make chunks` (the image bakes in
# data/processed/chunks.jsonl) and run with the key as an environment variable:
#
#   docker build -t studybot .
#   docker run --rm -p 8080:8080 -e LLM_API_KEY=... studybot      # http://localhost:8080
#
# The two models (about 3 GB) are downloaded at build time, so the container starts without
# network access to Hugging Face. Needs about 4 GB of RAM at run time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch: the default wheel drags in GPU libraries the service never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dense,llm,api]"

# Download the models now, so start-up does not depend on the network.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-base'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY data/processed/chunks.jsonl data/processed/chunks.jsonl
COPY data/cache data/cache

# Embed the chunks once at build time (cached in /app/indexes).
RUN python -c "from studybot.ingest import load_chunks; \
from studybot.retrieval import DenseIndex, E5Embedder; \
DenseIndex(load_chunks('data/processed/chunks.jsonl'), \
E5Embedder('intfloat/multilingual-e5-base'), cache_dir='indexes')"

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8080
# Cloud Run sets $PORT; locally it is 8080.
CMD ["sh", "-c", "uvicorn studybot.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
