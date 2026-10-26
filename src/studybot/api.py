"""Web service (Phase 7): StudyBot behind an HTTP API and a one-page demo.

    uvicorn studybot.api:app --port 8000        (needs: pip install -e ".[dense,llm,api]")

    GET  /          the demo page
    GET  /health    liveness: always 200 once the process is up
    POST /ask       {"question": "..."} -> answer, citations (document, section, pages), or a
                    refusal ("answerable": false) when the documents do not contain the answer

Settings come from environment variables, so the same image runs locally and in the cloud:

    LLM_API_KEY / ANTHROPIC_API_KEY   key for the model (never baked into the image)
    STUDYBOT_CHUNKS    path to chunks.jsonl            (default data/processed/chunks.jsonl)
    STUDYBOT_LLM       answering model                 (default claude-haiku-5-5)
    STUDYBOT_API_KEY   if set, requests need the header  X-API-Key: <value>
    STUDYBOT_RATE      max /ask calls per minute and client IP   (default 20)

The models are loaded once at start-up (about half a minute), then every request is one
retrieval + one model call. Requests are handled one at a time: the rerank step is the slow part
and sharing a model between threads is not worth the risk for a demo.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from studybot.ingest import Chunk

EXCERPT_CHARS = 280


class AnswerLike(Protocol):
    answerable: bool
    text: str
    citations: list[str]
    error: str | None


class PipelineLike(Protocol):
    def answer(self, question: str) -> AnswerLike: ...


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Citation(BaseModel):
    chunk_id: str
    document: str
    version: str
    section: str
    section_title: str
    page_start: int
    page_end: int
    excerpt: str


class AskResponse(BaseModel):
    answerable: bool
    answer: str
    citations: list[Citation]


class RateLimiter:
    """At most `limit` calls per `window` seconds per client (in memory, one process)."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self._clock = limit, window, clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = self._clock()
        calls = self._calls[client]
        while calls and now - calls[0] > self.window:
            calls.popleft()
        if len(calls) >= self.limit:
            return False
        calls.append(now)
        return True


def client_of(request: Request) -> str:
    """The caller's address. Behind a load balancer (Cloud Run) every request comes from the
    proxy, so use the last X-Forwarded-For entry: the one the proxy itself appended."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def citation_of(chunk: Chunk) -> Citation:
    text = " ".join(chunk.text.split())
    excerpt = text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS].rstrip() + "…"
    return Citation(
        chunk_id=chunk.chunk_id,
        document=chunk.doc_title,
        version=chunk.version,
        section=chunk.section,
        section_title=chunk.section_title,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        excerpt=excerpt,
    )


def load_pipeline() -> tuple[PipelineLike, list[Chunk]]:
    """The real pipeline: dense search + reranker + Claude. Slow; called once at start-up."""
    from studybot.answer import DEFAULT_MODEL, TOP_K, AnthropicLLM, Pipeline, build_retriever
    from studybot.ingest import load_chunks

    path = os.environ.get("STUDYBOT_CHUNKS", "data/processed/chunks.jsonl")
    if not Path(path).exists():
        raise SystemExit(f"{path} not found: run the ingest step (make chunks) first")
    chunks = load_chunks(path)
    settings = SimpleNamespace(
        model="intfloat/multilingual-e5-base",
        reranker_model="BAAI/bge-reranker-v2-m3",
        candidates=20,
        cache_dir=os.environ.get("STUDYBOT_CACHE_DIR", "indexes"),
        translations=os.environ.get("STUDYBOT_TRANSLATIONS", "data/cache/query_translations.json"),
        no_translate=False,
    )
    llm = AnthropicLLM(os.environ.get("STUDYBOT_LLM", DEFAULT_MODEL))
    return Pipeline(build_retriever(chunks, settings), chunks, llm, TOP_K), chunks


def create_app(
    pipeline: PipelineLike | None = None,
    chunks: list[Chunk] | None = None,
    api_key: str | None = None,
    rate_per_minute: int | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake pipeline; production loads the real one at start-up."""
    state: dict = {"pipeline": pipeline, "by_id": {c.chunk_id: c for c in chunks or []}}
    key = api_key if api_key is not None else os.environ.get("STUDYBOT_API_KEY")
    limiter = RateLimiter(rate_per_minute or int(os.environ.get("STUDYBOT_RATE", "20")))
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["pipeline"] is None:
            loaded, loaded_chunks = load_pipeline()
            state["pipeline"] = loaded
            state["by_id"] = {c.chunk_id: c for c in loaded_chunks}
        yield

    app = FastAPI(title="StudyBot", version="1.0", lifespan=lifespan)

    def guard(request: Request, x_api_key: str | None = Header(default=None)) -> None:
        if key and x_api_key != key:
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")
        if not limiter.allow(client_of(request)):
            raise HTTPException(status_code=429, detail="too many requests, try again in a minute")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "ready": state["pipeline"] is not None}

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard)])
    def ask(body: AskRequest) -> AskResponse:
        if state["pipeline"] is None:
            raise HTTPException(status_code=503, detail="still starting up")
        with lock:
            try:
                result = state["pipeline"].answer(body.question.strip())
            except Exception as error:  # the model API can fail (network, quota, overload)
                raise HTTPException(
                    status_code=502, detail=f"model call failed: {error}"
                ) from error
        if result.error:
            raise HTTPException(status_code=502, detail="the model's reply was unreadable, retry")
        cited = [state["by_id"][cid] for cid in result.citations if cid in state["by_id"]]
        return AskResponse(
            answerable=result.answerable,
            answer=result.text,
            citations=[citation_of(c) for c in cited],
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return DEMO_PAGE

    return app


DEMO_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StudyBot</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--muted:#6b6b66;--card:#fff;--line:#e2e2dc;--accent:#1f5fbf}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ececE8;--muted:#9a9a93;
--card:#1e1e1c;--line:#33332f;--accent:#7aa7f0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:1.5rem;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 24px}
form{display:flex;gap:8px}
input{flex:1;padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--card);
color:var(--fg);font:inherit}
button{padding:12px 18px;border:0;border-radius:8px;background:var(--accent);color:#fff;
font:inherit;cursor:pointer}button:disabled{opacity:.6;cursor:wait}
.ex{margin:12px 0 24px;display:flex;flex-wrap:wrap;gap:8px}
.ex button{background:var(--card);color:var(--fg);border:1px solid var(--line);padding:6px 10px;
font-size:.85rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;
margin-top:16px}
.src{font-size:.88rem;color:var(--muted);margin-top:12px;border-top:1px solid var(--line);
padding-top:10px}.src b{color:var(--fg)}
.refuse{border-left:4px solid var(--accent)}
footer{margin-top:32px;color:var(--muted);font-size:.85rem}
</style></head><body><main>
<h1>StudyBot</h1>
<p class="sub">Ask about the ISSS master's regulations (StuFPO, APO WIAI) and module handbook, in
German or English. Answers cite the source or say the documents don't contain it.</p>
<form id="f"><input id="q" required minlength="3" maxlength="500" autocomplete="off"
placeholder="e.g. How long is the standard period of study?" aria-label="Question">
<button id="b">Ask</button></form>
<div class="ex" id="ex"></div><div id="out" aria-live="polite"></div>
<footer>Unofficial student project. Not legal advice: always check the original documents.</footer>
</main><script>
const ex=["How long is the standard period of study?",
"Welche Englischnachweise werden verlangt?","What are the application deadlines?",
"How is the master's thesis graded?"];
const q=document.getElementById("q"),out=document.getElementById("out"),
b=document.getElementById("b");
for(const t of ex){const e=document.createElement("button");e.type="button";e.textContent=t;
e.onclick=()=>{q.value=t;ask()};document.getElementById("ex").append(e)}
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;
if(text!==undefined)e.textContent=text;return e}
async function ask(){b.disabled=true;out.replaceChildren(el("p","sub","Searching…"));
try{const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({question:q.value})});const d=await r.json();
if(!r.ok)throw new Error(d.detail||r.status);
const c=el("div","card"+(d.answerable?"":" refuse"));
c.append(el("div","",(d.answerable?"":"Not in the documents. ")+d.answer));
for(const s of d.citations){const p=el("div","src");
p.append(el("b","",s.document+" · "+s.section+" "+s.section_title+", p. "+s.page_start
+(s.page_end!==s.page_start?"–"+s.page_end:"")),el("div","",s.excerpt));c.append(p)}
out.replaceChildren(c)}catch(e){out.replaceChildren(el("div","card","Error: "+e.message))}
b.disabled=false}
document.getElementById("f").onsubmit=e=>{e.preventDefault();ask()};
</script></body></html>
"""


app = create_app()  # `uvicorn studybot.api:app`; the models load at start-up
