import json
from types import SimpleNamespace

import pytest

from studybot import answer as ans
from studybot.answer import (
    Answer,
    Pipeline,
    answer_metrics,
    build_user_prompt,
    format_context,
    format_metrics,
    load_env,
    parse_reply,
)
from studybot.evalset import build_questions, write_jsonl
from studybot.ingest import write_jsonl as write_chunks
from test_evalset import CHUNKS, fact, make_chunk, unans

IDS = ["reg:par1:0", "reg:par2:0", "hb:modula:0"]


def reply(answerable=True, answer="Vier Fachsemester. [C1]", citations=("C1",)):
    return json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)})


def test_format_context_numbers_chunks_with_header_and_pages():
    a = make_chunk("reg", "§ 1", "Text A", page=3)
    b = make_chunk("reg", "§ 2", "Text B", page=4)
    b.page_end = 5
    text = format_context([a, b])
    assert text.startswith("[C1] [reg | § 1], p. 3\nText A")
    assert "[C2] [reg | § 2], pp. 4-5\nText B" in text
    assert build_user_prompt("Frage?", [a]).endswith("Question: Frage?")


def test_parse_reply_maps_labels_to_chunk_ids():
    answerable, text, cited, invalid, error = parse_reply(reply(), IDS)
    assert (answerable, cited, invalid, error) == (True, ["reg:par1:0"], [], None)
    assert text == "Vier Fachsemester. [C1]"


def test_parse_reply_also_reads_citations_written_in_the_text():
    raw = reply(answer="A [C2], B [C3].", citations=("C2",))
    _, _, cited, _, _ = parse_reply(raw, IDS)
    assert cited == ["reg:par2:0", "hb:modula:0"]  # C3 only appeared in the text, still counted


def test_parse_reply_flags_citations_to_chunks_that_were_not_shown():
    _, _, cited, invalid, _ = parse_reply(reply(citations=("C1", "C9", "C0")), IDS)
    assert cited == ["reg:par1:0"] and invalid == ["C9", "C0"]


def test_parse_reply_abstention_has_no_citations():
    answerable, text, cited, invalid, error = parse_reply(
        reply(False, "Not in the excerpts.", ("C1",)), IDS
    )
    assert (answerable, cited, error) == (False, [], None)


def test_parse_reply_survives_text_around_the_json_and_garbage():
    wrapped = "Sure! ```json\n" + reply() + "\n```"
    assert parse_reply(wrapped, IDS)[0] is True
    answerable, text, cited, invalid, error = parse_reply("I think it is four semesters", IDS)
    assert answerable is False and error and text == "I think it is four semesters"
    assert parse_reply('{"answerable": true}', IDS)[4] is not None  # field missing


class FakeLLM:
    name = "fake-llm"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_pipeline_retrieves_prompts_and_parses():
    llm = FakeLLM([reply()])
    pipeline = Pipeline(
        lambda q, depth: ["reg:par1:0", "reg:par2:0", "hb:modula:0"], CHUNKS, llm, top_k=2
    )
    result = pipeline.answer("Wie lange?")
    assert result.context_ids == ["reg:par1:0", "reg:par2:0"]  # top_k applied
    assert result.citations == ["reg:par1:0"] and result.answerable
    system, user = llm.prompts[0]
    assert "ONLY the numbered excerpts" in system
    assert "[C1]" in user and "[C2]" in user and "[C3]" not in user
    assert "Regelstudienzeit" in user and user.endswith("Question: Wie lange?")


def record(gold_answerable, answerable, gold=("g",), context=("g",), citations=("g",), **kw):
    return {
        "gold_answerable": gold_answerable,
        "answerable": answerable,
        "gold_chunk_ids": list(gold),
        "context_ids": list(context),
        "citations": list(citations) if answerable else [],
        "invalid_citations": kw.get("invalid", []),
        "error": kw.get("error"),
    }


def test_answer_metrics():
    records = [
        record(True, True),  # answered, cites gold
        record(True, True, citations=("x",)),  # answered, cites the wrong chunk
        record(True, False, context=("x",)),  # false abstention, gold not even retrieved
        record(True, True, citations=()),  # answered without citation
        record(False, False, gold=(), context=("x",)),  # correct abstention
        record(False, True, gold=(), context=("x",), citations=("x",), invalid=["C9"]),  # made up
    ]
    m = answer_metrics(records)
    assert (m["n_answerable"], m["n_unanswerable"]) == (4, 2)
    assert m["correct_abstention"] == 0.5
    assert m["false_abstention"] == 0.25
    assert m["gold_in_context"] == 0.75
    assert m["cites_gold"] == pytest.approx(1 / 3)
    assert m["answers_without_citation"] == pytest.approx(1 / 3)
    assert m["invalid_citation_rate"] == pytest.approx(1 / 6)
    assert "abstains on unanswerable" in format_metrics(m)


def test_answer_metrics_handles_missing_groups():
    m = answer_metrics([record(True, True)])
    assert m["correct_abstention"] is None
    assert "n/a" in format_metrics(m)


def test_load_env_does_not_override_existing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nLLM_API_KEY="abc123"\nOTHER=1\n\nBROKEN\n', encoding="utf-8")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "keep")
    load_env(env)
    import os

    assert os.environ["LLM_API_KEY"] == "abc123" and os.environ["OTHER"] == "keep"
    monkeypatch.delenv("LLM_API_KEY")


def test_missing_key_gives_a_helpful_exit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        ans.AnthropicLLM()
    assert ".env" in str(error.value)


def test_cli_batch_with_fakes(tmp_path, monkeypatch, capsys):
    rows = build_questions([fact("f1", "vier Fachsemester")], [unans("u1")], CHUNKS)
    write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    write_jsonl(rows, tmp_path / "q.jsonl")
    split = rows[0]["split"]
    replies = [reply(), reply(False, "no", ()), reply(), reply(False, "no", ())]
    monkeypatch.setattr(
        ans, "build_retriever", lambda chunks, args: lambda q, depth: ["reg:par1:0"]
    )
    monkeypatch.setattr(ans, "AnthropicLLM", lambda model, **kw: FakeLLM(replies))
    base = ["--chunks", str(tmp_path / "chunks.jsonl"), "--questions", str(tmp_path / "q.jsonl"),
            "--out-dir", str(tmp_path / "res")]  # fmt: skip

    assert ans.main(["--batch", "--split", split, "--final", *base]) == 0
    out = capsys.readouterr().out
    assert "abstains on unanswerable" in out
    saved = json.loads((tmp_path / "res" / f"answers_{split}.json").read_text("utf-8"))
    assert len(saved["answers"]) == len([r for r in rows if r["split"] == split])
    assert saved["answers"][0]["gold_chunk_ids"]


def test_cli_single_question_prints_answer_and_source(tmp_path, monkeypatch, capsys):
    write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    monkeypatch.setattr(
        ans, "build_retriever", lambda chunks, args: lambda q, depth: ["reg:par1:0"]
    )
    monkeypatch.setattr(ans, "AnthropicLLM", lambda model, **kw: FakeLLM([reply()]))
    assert ans.main(["Wie lange?", "--chunks", str(tmp_path / "chunks.jsonl")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ANSWER: Vier Fachsemester.") and "[reg | § 1]" in out


def test_cli_locks_test_split_and_needs_a_question(tmp_path, capsys):
    assert ans.main(["--batch", "--split", "test"]) == 2
    assert "locked" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ans.main([])


def test_answer_dataclass_defaults():
    a = Answer("q", True, "t")
    assert a.citations == [] and a.error is None


def test_pipeline_retries_once_on_unreadable_reply():
    llm = FakeLLM(["", reply()])
    pipeline = Pipeline(lambda q, depth: ["reg:par1:0", "reg:par2:0"], CHUNKS, llm, top_k=2)
    answer = pipeline.answer("Wie lange?")
    assert answer.error is None and answer.answerable
    assert len(llm.prompts) == 2


def test_pipeline_gives_up_after_one_retry():
    llm = FakeLLM(["", ""])
    pipeline = Pipeline(lambda q, depth: ["reg:par1:0"], CHUNKS, llm, top_k=1)
    answer = pipeline.answer("Wie lange?")
    assert answer.error is not None and not answer.answerable


class _Block(SimpleNamespace):
    pass


class _FakeClient:
    """Stands in for anthropic.Anthropic().messages.create."""

    def __init__(self, content):
        self.content = content
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content)


def _llm(content, tool=None):
    llm = object.__new__(ans.AnthropicLLM)  # skip the constructor: no key, no network
    llm.name, llm.max_tokens, llm.tool = "m", 100, tool
    llm._client = _FakeClient(content)
    return llm


def test_tool_call_reply_is_returned_as_json_and_parses():
    block = _Block(
        type="tool_use", input={"answerable": True, "answer": "Vier [C1].", "citations": ["C1"]}
    )
    llm = _llm([block], tool=ans.ANSWER_TOOL)
    raw = llm.complete("sys", "user")
    call = llm._client.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "answer"} and call["tools"]
    assert ans.parse_reply(raw, ["reg:par1:0"])[:4] == (True, "Vier [C1].", ["reg:par1:0"], [])


def test_plain_text_reply_still_works_without_a_tool():
    llm = _llm([_Block(type="text", text='{"answerable": false}')])
    assert llm.complete("s", "u") == '{"answerable": false}'
    assert "tools" not in llm._client.calls[0]


def test_tool_schema_requires_the_three_fields():
    assert set(ans.ANSWER_TOOL["input_schema"]["required"]) == {
        "answerable", "answer", "citations"
    }  # fmt: skip
