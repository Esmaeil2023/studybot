import hashlib

import pytest

from studybot.retrieval import (
    BM25Index,
    DenseIndex,
    detect_lang,
    index_terms,
    query_terms,
    rank,
    rerank,
    rrf_fuse,
    words,
)
from test_evalset import make_chunk

np = pytest.importorskip("numpy")

CHUNKS = [
    make_chunk("reg", "§ 1", "Die Regelstudienzeit beträgt vier Fachsemester. Der Studienumfang "
               "beträgt mindestens 120 ECTS-Punkte."),
    make_chunk("reg", "§ 2", "Die Bewerbungsfrist für das Wintersemester endet am 15. Juli."),
    make_chunk("hb", "Modul A", "The module is taught every winter semester by the lecturers."),
    make_chunk("hb", "Modul B", "Students write a thesis and defend it in a colloquium."),
]  # fmt: skip


def test_words_lowercases_and_splits_on_hyphens_and_punctuation():
    assert words("Die ECTS-Punkte (120) gelten.") == ["die", "ects", "punkte", "120", "gelten"]


def test_detect_lang():
    assert detect_lang("Die Prüfung ist nicht bestanden, wenn die Note schlechter ist") == "de"
    assert detect_lang("The exam is passed when the grade is better than four") == "en"


def test_index_terms_stem_in_the_chunks_own_language_and_drop_stop_words():
    de = index_terms("Die Prüfungen werden wiederholt und die Prüfung")
    assert "de:prufung" in de and "de:die" not in de and "de:und" not in de
    en = index_terms("The examinations are repeated by the lecturers")
    assert "en:examin" in en and "en:lectur" in en and "en:the" not in en


def test_query_terms_cover_both_stemmers_and_numbers():
    terms = query_terms("Wie viele ECTS 120?")
    assert "de:120" in terms and "en:120" in terms
    assert "de:ect" in terms or "de:ects" in terms
    assert "de:wie" not in terms  # stop word


def test_bm25_finds_the_chunk_with_the_matching_rare_word():
    index = BM25Index(CHUNKS)
    assert index.search("Bewerbungsfrist Wintersemester", k=3)[0][0] == "reg:par2:0"
    assert index.search("colloquium thesis", k=3)[0][0] == "hb:modulb:0"


def test_bm25_stems_so_inflected_forms_match():
    index = BM25Index(CHUNKS)
    assert index.search("Fachsemestern Regelstudienzeit", k=1)[0][0] == "reg:par1:0"
    assert index.search("lecturer", k=1)[0][0] == "hb:modula:0"


def test_bm25_matches_the_header_too():
    chunks = [make_chunk("reg", "§ 30", "Text ohne Besonderheiten."), CHUNKS[1]]
    assert BM25Index(chunks).search("§ 30", k=1)[0][0] == chunks[0].chunk_id


def test_bm25_cannot_bridge_languages_which_is_the_point_of_the_dense_retriever():
    # "standard period of study" shares no word with "Regelstudienzeit"
    assert BM25Index(CHUNKS).search("standard period of study", k=3) == []


def test_bm25_drops_zero_scores_and_respects_k():
    assert BM25Index(CHUNKS).search("zzzunbekannt", k=5) == []
    assert len(BM25Index(CHUNKS).search("die der", k=5)) == 0
    assert len(BM25Index(CHUNKS).search("semester", k=1)) == 1


def test_rank_is_stable_for_ties():
    assert rank(["a", "b", "c"], [1.0, 1.0, 0.5], 3) == [("a", 1.0), ("b", 1.0), ("c", 0.5)]


def test_rrf_rewards_agreement_over_a_single_first_place():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "a"]])
    assert fused[0] == "b"  # 1/62 + 1/61 beats a's 1/61 + 1/63
    assert set(fused) == {"a", "b", "c"}
    assert rrf_fuse([["a", "b"], ["c"]], top=2) == ["a", "c"]
    assert rrf_fuse([]) == []


def test_rrf_k_changes_the_balance():
    rankings = [["a", "x", "y", "z", "b"], ["b", "q", "r", "s", "a"]]
    assert set(rrf_fuse(rankings, k=60)[:2]) == {"a", "b"}


class FakeEmbedder:
    """Hashed bag of words: texts that share words get similar vectors. No model needed."""

    name = "fake/embedder"
    dim = 64

    def __init__(self):
        self.passage_calls = 0

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]

    def embed_passages(self, texts):
        self.passage_calls += 1
        return [self._vec(t) for t in texts]


def test_dense_index_ranks_by_cosine_similarity():
    index = DenseIndex(CHUNKS, FakeEmbedder())
    hits = index.search("colloquium thesis defend", k=2)
    assert hits[0][0] == "hb:modulb:0"
    assert hits[0][1] > hits[1][1]
    assert len(index.search("anything", k=10)) == len(CHUNKS)


def test_dense_index_caches_embeddings_and_invalidates_on_change(tmp_path):
    embedder = FakeEmbedder()
    first = DenseIndex(CHUNKS, embedder, cache_dir=tmp_path)
    again = DenseIndex(CHUNKS, embedder, cache_dir=tmp_path)
    assert embedder.passage_calls == 1  # second index read the cache
    assert np.allclose(first.matrix, again.matrix)

    changed = [*CHUNKS[:-1], make_chunk("hb", "Modul B", "A completely different text.")]
    DenseIndex(changed, embedder, cache_dir=tmp_path)
    assert embedder.passage_calls == 2  # chunks changed -> recomputed


def test_rrf_weights_let_a_trusted_ranking_win():
    rankings = [["a", "b"], ["b", "a"]]
    assert rrf_fuse(rankings, weights=[1.0, 0.1])[0] == "a"
    assert rrf_fuse(rankings, weights=[0.1, 1.0])[0] == "b"
    assert rrf_fuse(rankings, weights=[0.0, 1.0]) == ["b", "a"]
    with pytest.raises(ValueError):
        rrf_fuse(rankings, weights=[1.0])


class WordOverlap:
    name = "fake/overlap"

    def score(self, query, passages):
        q = set(words(query))
        return [float(len(q & set(words(p)))) for p in passages]


def test_rerank_orders_candidates_by_score_and_is_stable():
    texts = {"a": "alpha beta", "b": "gamma", "c": "alpha beta gamma", "d": "nothing"}
    assert rerank("alpha beta gamma", ["a", "b", "c", "d"], texts, WordOverlap()) == [
        "c", "a", "b", "d",
    ]  # fmt: skip
    assert rerank("zzz", ["d", "b"], texts, WordOverlap()) == ["d", "b"]  # ties keep input order
    assert rerank("alpha", [], texts, WordOverlap()) == []
