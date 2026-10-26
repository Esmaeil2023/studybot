"""Answer a question from the retrieved chunks, with citations and the option to say "I don't know".

    python -m studybot.answer "How long is the standard period of study?"
    python -m studybot.answer --batch --split dev

The language model sees only the question and the top retrieved chunks, numbered [C1], [C2], ...
It must answer from those excerpts, cite the numbers it used, and answer in the language of the
question. If the excerpts do not contain the answer it must say so instead of guessing. Its
reply is requested as JSON; we map the citation numbers back to chunk ids and pages, and we flag
citations that point to a chunk the model was never shown.

Automatic checks that need no judge (see `answer_metrics`): how often the model abstains on
unanswerable questions (good) and on answerable ones (bad), and whether its citations include a
gold chunk. Whether the answer text is *correct* is judged in the next phase.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from studybot.evalset import load_questions
from studybot.ingest import Chunk, load_chunks

DEFAULT_MODEL = "claude-haiku-5-5"
TOP_K = 5

SYSTEM_PROMPT = """You answer questions about the study and examination regulations of a German \
university and about its module handbook. The excerpts are in German or English.

Rules:
1. Use ONLY the numbered excerpts below. Never use outside knowledge, and never guess.
2. Answer in the language of the question (German question -> German answer, English question \
-> English answer), concisely, in one to three sentences. Copy numbers, deadlines and names \
exactly as the excerpts give them.
3. After each statement, cite the excerpt it comes from, like [C2].
4. If the excerpts do not contain the answer, or only speak about something else (another \
degree programme, another module), do not answer: set "answerable" to false.
5. Reply with one JSON object and nothing else:
{"answerable": true or false, "answer": "<text, or a one-sentence explanation of what is \
missing>", "citations": ["C1", "C3"]}
Use an empty citations list when answerable is false.

Note: module descriptions come from a PDF table that was flattened into text, so a label and \
its value can end up next to each other without a sentence, e.g. "Examination Portfolio" means \
the form of examination is a portfolio. Read such pairs as statements."""

# Version 1.1: the answer is requested as a forced tool call, so the API itself guarantees a
# JSON object with these three fields (v1.0 asked for JSON in plain text and occasionally got
# nothing or malformed text back).
ANSWER_TOOL = {
    "name": "answer",
    "description": "Give the final answer, or say that the excerpts do not contain it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answerable", "answer", "citations"],
    },
}

_CITE_RE = re.compile(r"\bC(\d+)\b")


# ---------------------------------------------------------------- language model


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


def load_env(path: str | Path = ".env") -> None:
    """Put KEY=VALUE lines of a .env file into os.environ (existing variables win)."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class AnthropicLLM:
    """Claude through the Anthropic API. Needs `pip install -e ".[llm]"` and LLM_API_KEY in .env."""

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = 1200, tool: dict | None = None
    ):
        """`tool`: if given, the model must answer through this tool, and `complete` returns the
        tool's input as a JSON string (so the reply is valid JSON by construction)."""
        load_env()
        key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("No API key: copy .env.example to .env and set LLM_API_KEY=...")
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise SystemExit('answer generation needs: pip install -e ".[llm]"') from error
        self.name = model
        self.max_tokens = max_tokens
        self.tool = tool
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> str:
        options = {}
        if self.tool:
            options = {
                "tools": [self.tool],
                "tool_choice": {"type": "tool", "name": self.tool["name"]},
            }
        reply = self._client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **options,
        )
        for block in reply.content:
            if block.type == "tool_use":
                return json.dumps(block.input, ensure_ascii=False)
        return "".join(block.text for block in reply.content if block.type == "text")


# ---------------------------------------------------------------- prompt and parsing


def format_context(chunks: Sequence[Chunk]) -> str:
    """The excerpts as the model sees them: '[C1] <header>, p. N' followed by the text."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        pages = (
            f"p. {c.page_start}"
            if c.page_start == c.page_end
            else f"pp. {c.page_start}-{c.page_end}"
        )
        parts.append(f"[C{i}] {c.header}, {pages}\n{c.text}")
    return "\n\n".join(parts)


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    return f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"


@dataclass
class Answer:
    question: str
    answerable: bool  # the model's decision; False = it abstained
    text: str
    citations: list[str] = field(default_factory=list)  # chunk ids, in citation order
    invalid_citations: list[str] = field(default_factory=list)  # labels the model was not shown
    context_ids: list[str] = field(default_factory=list)  # what the model saw, best first
    error: str | None = None  # set when the reply could not be read
    raw: str = ""


def parse_reply(
    raw: str, context_ids: Sequence[str]
) -> tuple[bool, str, list[str], list[str], str | None]:
    """(answerable, text, cited chunk ids, invalid labels, error) from the model's raw reply."""
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start : end + 1])
        answerable, text = bool(data["answerable"]), str(data["answer"]).strip()
        labels = [str(x) for x in data.get("citations", [])]
    except (ValueError, KeyError, TypeError):
        return False, raw.strip(), [], [], "reply was not the requested JSON"

    labels += [f"C{n}" for n in _CITE_RE.findall(text)]  # citations written inside the text
    cited: list[str] = []
    invalid: list[str] = []
    for label in dict.fromkeys(labels):
        match = _CITE_RE.fullmatch(label.strip("[] "))
        index = int(match.group(1)) if match else 0
        if 1 <= index <= len(context_ids):
            cited.append(context_ids[index - 1])
        else:
            invalid.append(label)
    return answerable, text, ([] if not answerable else cited), invalid, None


class Retriever(Protocol):
    def __call__(self, query: str, depth: int) -> list[str]: ...


class Pipeline:
    """question -> retrieve top chunks -> ask the model -> parsed Answer."""

    def __init__(self, retrieve: Retriever, chunks: Sequence[Chunk], llm: LLM, top_k: int = TOP_K):
        self.retrieve, self.llm, self.top_k = retrieve, llm, top_k
        self._by_id = {c.chunk_id: c for c in chunks}

    def answer(self, question: str) -> Answer:
        context_ids = self.retrieve(question, self.top_k)[: self.top_k]
        context = [self._by_id[cid] for cid in context_ids]
        prompt = build_user_prompt(question, context)
        for _attempt in range(2):  # one retry: replies are sometimes empty or cut off
            raw = self.llm.complete(SYSTEM_PROMPT, prompt)
            answerable, text, cited, invalid, error = parse_reply(raw, context_ids)
            if error is None:
                break
        return Answer(question, answerable, text, cited, invalid, list(context_ids), error, raw)


# ---------------------------------------------------------------- automatic checks


def answer_metrics(records: Sequence[dict]) -> dict:
    """Checks that need no judge. Each record: the Answer fields plus `gold_answerable` and
    `gold_chunk_ids` of the question."""
    answerable = [r for r in records if r["gold_answerable"]]
    unanswerable = [r for r in records if not r["gold_answerable"]]

    def share(items: Sequence[dict], test) -> float | None:
        return sum(1 for r in items if test(r)) / len(items) if items else None

    answered = [r for r in answerable if r["answerable"]]
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "correct_abstention": share(unanswerable, lambda r: not r["answerable"]),
        "false_abstention": share(answerable, lambda r: not r["answerable"]),
        "gold_in_context": share(
            answerable, lambda r: set(r["gold_chunk_ids"]) & set(r["context_ids"])
        ),
        "cites_gold": share(answered, lambda r: set(r["gold_chunk_ids"]) & set(r["citations"])),
        "answers_without_citation": share(answered, lambda r: not r["citations"]),
        "invalid_citation_rate": share(records, lambda r: bool(r["invalid_citations"])),
        "unreadable_replies": share(records, lambda r: r["error"] is not None),
    }


def format_metrics(metrics: dict) -> str:
    labels = {
        "correct_abstention": "abstains on unanswerable questions (want high)",
        "false_abstention": "abstains on answerable questions (want low)",
        "gold_in_context": "answerable: a gold chunk was in the context",
        "cites_gold": "answered: cites a gold chunk",
        "answers_without_citation": "answered without any citation (want 0)",
        "invalid_citation_rate": "cites a chunk it was not shown (want 0)",
        "unreadable_replies": "reply not valid JSON (want 0)",
    }
    lines = [f"answerable: {metrics['n_answerable']}, unanswerable: {metrics['n_unanswerable']}"]
    for key, label in labels.items():
        value = metrics[key]
        lines.append(f"  {label:<52} {'n/a' if value is None else f'{value:.2f}'}")
    return "\n".join(lines)


# ---------------------------------------------------------------- command line


def build_retriever(chunks: list[Chunk], args: argparse.Namespace) -> Retriever:
    """Dense first stage (English questions also searched in German) + cross-encoder reranker."""
    from studybot.retrieval import (
        CrossEncoderReranker,
        DenseIndex,
        E5Embedder,
        rerank,
        rrf_fuse,
    )
    from studybot.translate import default_translator

    translate = None if args.no_translate else default_translator(args.translations)
    dense = DenseIndex(chunks, E5Embedder(args.model), cache_dir=args.cache_dir)
    reranker = CrossEncoderReranker(args.reranker_model)
    texts = {c.chunk_id: c.index_text for c in chunks}

    def retrieve(query: str, depth: int) -> list[str]:
        queries = [query, *(translate(query) if translate else [])]
        rankings = [[cid for cid, _ in dense.search(q, args.candidates)] for q in queries]
        candidates = rrf_fuse(rankings)[: args.candidates] if len(rankings) > 1 else rankings[0]
        return rerank(query, candidates, texts, reranker)[:depth]

    return retrieve


def show(answer: Answer, by_id: dict[str, Chunk]) -> None:
    print(("ANSWER: " if answer.answerable else "NO ANSWER: ") + answer.text)
    for cid in answer.citations:
        c = by_id[cid]
        print(f"  - {c.header}  pp. {c.page_start}-{c.page_end}  ({cid})")
    if answer.invalid_citations:
        print(f"  ! cited chunks it was not shown: {answer.invalid_citations}")
    if answer.error:
        print(f"  ! {answer.error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer questions with citations")
    parser.add_argument("question", nargs="?", help="a single question")
    parser.add_argument("--batch", action="store_true", help="run the evaluation questions")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument("--limit", type=int, help="batch: only the first N questions")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--k", type=int, default=TOP_K, help="chunks shown to the model")
    parser.add_argument("--llm", default=DEFAULT_MODEL)
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--cache-dir", default="indexes")
    parser.add_argument("--translations", default="data/cache/query_translations.json")
    parser.add_argument(
        "--no-translate", action="store_true", help="do not also search English questions in German"
    )
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    if not args.question and not args.batch:
        parser.error("give a question or use --batch")
    if args.batch and args.split == "test" and not args.final:
        print("The test split is locked. Tune on dev; use --final once, for the reported numbers.")
        return 2
    if not Path(args.chunks).exists():
        print(f"{args.chunks} not found (run `make chunks` first)")
        return 2

    chunks = load_chunks(args.chunks)
    pipeline = Pipeline(
        build_retriever(chunks, args), chunks, AnthropicLLM(args.llm, tool=ANSWER_TOOL), args.k
    )
    by_id = {c.chunk_id: c for c in chunks}

    if args.question:
        show(pipeline.answer(args.question), by_id)
        return 0

    rows = load_questions(args.questions, split=args.split)[: args.limit]
    records = []
    for i, row in enumerate(rows, start=1):
        answer = pipeline.answer(row["question"])
        records.append(
            {
                "qid": row["qid"],
                "lang": row["lang"],
                "gold_answerable": row["answerable"],
                "gold_answer": row["answer"],
                "gold_chunk_ids": row["gold_chunk_ids"],
                **asdict(answer),
            }
        )
        print(f"[{i}/{len(rows)}] {row['qid']}: {'answered' if answer.answerable else 'abstained'}")

    metrics = answer_metrics(records)
    print("\n" + format_metrics(metrics))
    out = Path(args.out_dir) / f"answers_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "llm": args.llm,
        "k": args.k,
        "translate": not args.no_translate,
        "metrics": metrics,
        "answers": records,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
