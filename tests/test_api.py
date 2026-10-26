from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from studybot import api
from studybot.ingest import Chunk


def chunk(cid="reg:par30:0", text="Die Regelstudienzeit beträgt vier Fachsemester."):
    return Chunk(
        chunk_id=cid, doc_id="reg", doc_title="StuFPO ISSS 2024", version="2024",
        section="§ 30", section_title="Studiendauer", page_start=3, page_end=4,
        header="[StuFPO | § 30]", text=text, n_words=len(text.split()),
    )  # fmt: skip


class FakePipeline:
    def __init__(self, **fields):
        self.fields = {"answerable": True, "text": "Vier Fachsemester [C1].",
                       "citations": ["reg:par30:0"], "error": None, **fields}  # fmt: skip
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        if "boom" in question:
            raise RuntimeError("api down")
        return SimpleNamespace(**self.fields)


def client(pipeline=None, **kwargs):
    pipeline = pipeline or FakePipeline()
    return TestClient(api.create_app(pipeline, [chunk()], **kwargs)), pipeline


def test_ask_returns_answer_with_citation_details():
    c, pipeline = client()
    r = c.post("/ask", json={"question": "  Wie lange dauert das Studium?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"] is True and body["answer"].startswith("Vier")
    cite = body["citations"][0]
    assert cite["document"] == "StuFPO ISSS 2024" and cite["section"] == "§ 30"
    assert (cite["page_start"], cite["page_end"]) == (3, 4)
    assert pipeline.questions == ["Wie lange dauert das Studium?"]  # whitespace trimmed


def test_refusal_has_no_citations():
    c, _ = client(FakePipeline(answerable=False, text="Not in the documents.", citations=[]))
    body = c.post("/ask", json={"question": "Wie hoch sind die Gebühren?"}).json()
    assert body["answerable"] is False and body["citations"] == []


def test_long_excerpts_are_shortened():
    long = chunk(text="wort " * 200)
    cite = api.citation_of(long)
    assert len(cite.excerpt) <= api.EXCERPT_CHARS + 1 and cite.excerpt.endswith("…")


def test_unknown_citation_ids_are_dropped_not_crashed():
    c, _ = client(FakePipeline(citations=["reg:par30:0", "gone:0"]))
    r = c.post("/ask", json={"question": "Wie lange dauert das Studium?"})
    assert [x["chunk_id"] for x in r.json()["citations"]] == ["reg:par30:0"]


@pytest.mark.parametrize("question", ["", "ab", "x" * 501])
def test_bad_questions_are_rejected(question):
    c, _ = client()
    assert c.post("/ask", json={"question": question}).status_code == 422


def test_model_failure_is_a_502_not_a_crash():
    c, _ = client()
    r = c.post("/ask", json={"question": "boom please"})
    assert r.status_code == 502 and "api down" in r.json()["detail"]


def test_unreadable_model_reply_is_a_502():
    c, _ = client(FakePipeline(error="reply was not the requested JSON"))
    assert c.post("/ask", json={"question": "Wie lange dauert das?"}).status_code == 502


def test_api_key_is_enforced_only_when_configured():
    c, _ = client(api_key="secret")
    body = {"question": "Wie lange dauert das Studium?"}
    assert c.post("/ask", json=body).status_code == 401
    assert c.post("/ask", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/ask", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/health").status_code == 200  # health needs no key


def test_rate_limit_returns_429():
    c, _ = client(rate_per_minute=2)
    body = {"question": "Wie lange dauert das Studium?"}
    assert [c.post("/ask", json=body).status_code for _ in range(3)] == [200, 200, 429]


def test_rate_limiter_window_expires():
    now = [0.0]
    limiter = api.RateLimiter(1, window=60, clock=lambda: now[0])
    assert limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")


def test_health_and_demo_page():
    c, _ = client()
    assert c.get("/health").json() == {"status": "ok", "ready": True}
    page = c.get("/")
    assert page.status_code == 200 and "StudyBot" in page.text
    assert "innerHTML" not in page.text  # answers are inserted as text, never as HTML


def test_not_ready_without_pipeline_returns_503():
    app = api.create_app(pipeline=None, chunks=[])  # no lifespan: models not loaded
    c = TestClient(app)
    assert c.get("/health").json()["ready"] is False
    assert c.post("/ask", json={"question": "Wie lange dauert das?"}).status_code == 503


def test_rate_limit_uses_the_forwarded_client_address_behind_a_proxy():
    c, _ = client(rate_per_minute=1)
    body = {"question": "Wie lange dauert das Studium?"}
    one = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}  # the last entry is the proxy's observation
    two = {"X-Forwarded-For": "1.1.1.1, 10.0.0.2"}
    assert c.post("/ask", json=body, headers=one).status_code == 200
    assert c.post("/ask", json=body, headers=one).status_code == 429
    assert c.post("/ask", json=body, headers=two).status_code == 200


def test_load_pipeline_wires_the_real_components(monkeypatch, tmp_path):
    import studybot.answer as answer_module
    import studybot.ingest as ingest_module

    seen = {}
    chunks = [chunk()]
    path = tmp_path / "chunks.jsonl"
    path.write_text("x")
    monkeypatch.setenv("STUDYBOT_CHUNKS", str(path))
    monkeypatch.setenv("STUDYBOT_LLM", "my-model")
    monkeypatch.setattr(ingest_module, "load_chunks", lambda p: chunks)
    monkeypatch.setattr(
        answer_module, "AnthropicLLM", lambda name, **kw: seen.setdefault("llm", name)
    )
    monkeypatch.setattr(
        answer_module, "build_retriever", lambda c, settings: seen.setdefault("settings", settings)
    )
    monkeypatch.setattr(
        answer_module,
        "Pipeline",
        lambda retrieve, c, llm, k: SimpleNamespace(retrieve=retrieve, llm=llm, k=k),
    )
    pipeline, loaded = api.load_pipeline()
    assert loaded == chunks and seen["llm"] == "my-model"
    settings = seen["settings"]  # exactly the fields build_retriever reads
    for field in ("model", "reranker_model", "candidates", "cache_dir", "translations"):
        assert hasattr(settings, field)
    assert settings.no_translate is False


def test_load_pipeline_stops_with_a_clear_message_when_chunks_are_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDYBOT_CHUNKS", str(tmp_path / "nope.jsonl"))
    with pytest.raises(SystemExit, match="make chunks"):
        api.load_pipeline()
