import json

import pytest

from studybot.translate import QueryTranslator


def test_english_query_is_translated_and_cached(tmp_path):
    calls = []

    def llm(system, user):
        calls.append(user)
        return " Wie lange dauert die Regelstudienzeit? "

    cache = tmp_path / "t.json"
    translator = QueryTranslator(llm, cache)
    q = "How long is the standard period of study in the programme?"
    assert translator(q) == ["Wie lange dauert die Regelstudienzeit?"]
    assert translator(q) == ["Wie lange dauert die Regelstudienzeit?"]
    assert len(calls) == 1
    assert json.loads(cache.read_text(encoding="utf-8"))[q].startswith("Wie lange")


def test_cache_is_reused_without_llm(tmp_path):
    cache = tmp_path / "t.json"
    q = "What is the deadline for the application to the programme?"
    cache.write_text(json.dumps({q: "Bis wann muss man sich bewerben?"}), encoding="utf-8")
    assert QueryTranslator(None, cache)(q) == ["Bis wann muss man sich bewerben?"]


def test_german_query_is_not_translated():
    def llm(system, user):
        raise AssertionError("must not be called")

    german = "Wie lange ist die Regelstudienzeit in dem Studiengang?"
    assert QueryTranslator(llm)(german) == []


def test_missing_translation_without_llm_raises():
    with pytest.raises(KeyError):
        QueryTranslator(None)("What is the deadline for the application to the programme?")


def test_identical_translation_adds_nothing():
    translator = QueryTranslator(lambda s, u: u)
    assert translator("What is the maximum number of ECTS in the programme?") == []


def test_empty_reply_is_retried_once_and_not_cached_as_empty(tmp_path):
    replies = iter(["", "Wie lange dauert das Studium?"])
    translator = QueryTranslator(lambda s, u: next(replies), tmp_path / "t.json")
    q = "How long does the programme take to complete in total?"
    assert translator(q) == ["Wie lange dauert das Studium?"]


def test_cached_empty_translation_is_translated_again(tmp_path):
    q = "How long does the programme take to complete in total?"
    cache = tmp_path / "t.json"
    cache.write_text(json.dumps({q: ""}), encoding="utf-8")
    translator = QueryTranslator(lambda s, u: "Wie lange dauert das Studium?", cache)
    assert translator(q) == ["Wie lange dauert das Studium?"]


def test_two_empty_replies_give_no_extra_query():
    assert QueryTranslator(lambda s, u: "")("How long does the programme take in total?") == []
