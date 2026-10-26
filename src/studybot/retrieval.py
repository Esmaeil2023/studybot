"""Retrieval (Phase 3): keyword search (BM25), dense search (multilingual-e5) and their fusion.

Each retriever answers one question: "given this query, which chunks come first?" and returns a
ranked list of (chunk_id, score). Nothing here calls a language model to write an answer.

    BM25Index(chunks).search(query, k)           words that match, weighted by rarity
    DenseIndex(chunks, embedder).search(query, k)  meaning, via embeddings; works across languages
    rrf_fuse([ranking_a, ranking_b])             combine rankings without comparing their scores

Why both: BM25 finds the exact "TOEFL 90 iBT" or "§ 30" but cannot link the English question
"standard period of study" to the German "Regelstudienzeit". The dense model can, but is less
precise about exact numbers and names. Fusing the two rankings is usually better than either.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import snowballstemmer

from studybot.ingest import Chunk

# ---------------------------------------------------------------- text preparation

_TOKEN_RE = re.compile(r"\w+")

_STOP_DE = frozenset(
    "der die das den dem des ein eine einen einem einer eines und oder aber nicht ist sind "
    "wird werden wurde wurden hat haben kann können muss müssen soll sollen im in am an auf "
    "zu zur zum von vom mit für bei nach aus über als wie wenn auch es sich sie er wir ich man "
    "dass daß dies diese dieser dieses welche welcher welches was wer wo wann wieviel wie".split()
)
_STOP_EN = frozenset(
    "the a an and or but not is are was were be been being has have had can could must shall "
    "should will would may might of in on at to for from by with as it its this that these those "
    "which who what when where how do does did there their they you your i we he she".split()
)

_STEMMERS = {"de": snowballstemmer.stemmer("german"), "en": snowballstemmer.stemmer("english")}


def words(text: str) -> list[str]:
    """Lower-cased word tokens ('ECTS-Punkte' gives 'ects' and 'punkte')."""
    return _TOKEN_RE.findall(text.casefold())


def detect_lang(text: str) -> str:
    """'de' or 'en', by which language's function words occur more often. Good enough for
    telling a German regulation from an English module description."""
    tokens = words(text)
    de = sum(t in _STOP_DE for t in tokens)
    en = sum(t in _STOP_EN for t in tokens)
    return "de" if de > en else "en"


def index_terms(text: str) -> list[str]:
    """Terms of a document chunk: stop words dropped, rest stemmed in the chunk's own language."""
    lang = detect_lang(text)
    stop = _STOP_DE if lang == "de" else _STOP_EN
    return [f"{lang}:{_STEMMERS[lang].stemWord(t)}" for t in words(text) if t not in stop]


def query_terms(query: str) -> list[str]:
    """Terms of a query. The query's language may differ from the chunk's, and a German word
    stemmed the English way (or the reverse) just never matches, so the query is stemmed both
    ways. A number or an abbreviation ('120', 'ects') is the same in both and matches anyway."""
    terms: list[str] = []
    for t in words(query):
        if t in _STOP_DE or t in _STOP_EN:
            continue
        terms += [f"{lang}:{_STEMMERS[lang].stemWord(t)}" for lang in ("de", "en")]
    return list(dict.fromkeys(terms))


# ---------------------------------------------------------------- BM25


class BM25Index:
    """Okapi BM25 over `chunk.index_text` (header + text: '§ 30' and module ids are searchable)."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.k1, self.b = k1, b
        self._tf = [Counter(index_terms(c.index_text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / max(len(self._len), 1)
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str) -> list[float]:
        terms = query_terms(query)
        scores = []
        for tf, length in zip(self._tf, self._len, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return rank(self.chunk_ids, self.score(query), k)


def rank(ids: Sequence[str], scores: Sequence[float], k: int) -> list[tuple[str, float]]:
    """Top k by score. Ties keep document order (stable), chunks with score 0 are dropped."""
    order = sorted(range(len(ids)), key=lambda i: -scores[i])
    return [(ids[i], float(scores[i])) for i in order[:k] if scores[i] > 0]


# ---------------------------------------------------------------- dense retrieval


class Embedder(Protocol):
    """Anything that turns texts into unit-length vectors."""

    name: str

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class E5Embedder:
    """multilingual-e5 via sentence-transformers.

    The e5 models were trained with the prefixes "query: " and "passage: "; leaving them out
    makes retrieval noticeably worse. Needs `pip install -e ".[dense]"` and downloads the model
    (about 1.1 GB for the base model) on first use.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", batch_size: int = 16):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise SystemExit('dense retrieval needs: pip install -e ".[dense]"') from error
        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"query: {t}" for t in texts])

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"passage: {t}" for t in texts])


def _fingerprint(model: str, chunks: Sequence[Chunk]) -> str:
    h = hashlib.sha256(model.encode())
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.index_text.encode())
    return h.hexdigest()


class DenseIndex:
    """Exact nearest-neighbour search over chunk embeddings (cosine similarity).

    With a few hundred chunks a brute-force matrix product is instant and gives exactly the same
    result as FAISS; FAISS only pays off at hundreds of thousands of vectors. The embeddings are
    cached on disk and recomputed only if the chunks or the model change.
    """

    def __init__(
        self, chunks: Sequence[Chunk], embedder: Embedder, cache_dir: str | Path | None = None
    ):
        import numpy as np

        self._np = np
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.embedder = embedder
        fingerprint = _fingerprint(embedder.name, chunks)
        cache = None
        if cache_dir is not None:
            slug = re.sub(r"[^\w.-]+", "_", embedder.name)
            cache = Path(cache_dir) / f"dense-{slug}.npz"
        self.matrix = self._load(cache, fingerprint)
        if self.matrix is None:
            vectors = embedder.embed_passages([c.index_text for c in chunks])
            self.matrix = np.asarray(vectors, dtype="float32")
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, matrix=self.matrix, fingerprint=fingerprint)

    def _load(self, cache: Path | None, fingerprint: str):
        if cache is None or not cache.exists():
            return None
        data = self._np.load(cache, allow_pickle=False)
        if str(data["fingerprint"]) != fingerprint:
            return None
        return data["matrix"]

    def score(self, query: str) -> list[float]:
        vector = self._np.asarray(self.embedder.embed_queries([query])[0], dtype="float32")
        return (self.matrix @ vector).tolist()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]


# ---------------------------------------------------------------- reranking


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher = the passage answers the query better."""

    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A cross-encoder reads the query and the passage *together*, so it can notice that
    "master's thesis grade" is about a regulation paragraph and not about a module description.
    The first-stage retrievers compare two separately computed vectors and cannot do that.
    Slow (one model run per pair), so it only reorders the top candidates of the first stage.

    Default: BAAI/bge-reranker-v2-m3, multilingual (about 2.2 GB download).
    A lighter option: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (about 0.5 GB).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", batch_size: int = 8):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise SystemExit('reranking needs: pip install -e ".[dense]"') from error
        self.name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in scores]


def rerank(
    query: str,
    candidate_ids: Sequence[str],
    texts: dict[str, str],
    reranker: Reranker,
) -> list[str]:
    """`candidate_ids` reordered by reranker score, best first (ties keep the input order)."""
    if not candidate_ids:
        return []
    scores = reranker.score(query, [texts[cid] for cid in candidate_ids])
    order = sorted(range(len(candidate_ids)), key=lambda i: -scores[i])
    return [candidate_ids[i] for i in order]


# ---------------------------------------------------------------- fusion


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    top: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Reciprocal Rank Fusion: score(chunk) = sum over rankings of weight / (k + rank).

    Only ranks are used, never the raw scores, because BM25 scores (0 to ~30) and cosine
    similarities (0.7 to 0.9) cannot be compared. k=60 is the value from the original paper;
    a larger k flattens the difference between rank 1 and rank 5. `weights` (default: all 1)
    lets a trusted ranking count more than a weak one.
    """
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + position)
    fused = sorted(scores, key=lambda cid: -scores[cid])
    return fused[:top] if top else fused
