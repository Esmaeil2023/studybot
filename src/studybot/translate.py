"""Query translation (Phase 3b): English questions are also searched in German.

The regulations are in German. A multilingual embedding model bridges the gap most of the
time, but a few English questions still miss ("application deadlines" vs. "Bewerbungsfrist").
Translating the question into German and searching with both versions gives the retriever a
second, same-language chance. German questions are left alone.

Translations are cached in a JSON file, so the evaluation is repeatable without an API key and
each question is translated (and paid for) once.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from studybot.retrieval import detect_lang

SYSTEM_PROMPT = (
    "Translate the user's question into German. Keep abbreviations, module names, programme "
    "names and numbers unchanged (ECTS, ISSS, TOEFL, 'Designing Gamified Systems', 15). Use the "
    "terms a German university regulation would use. Reply with the translation only."
)

Complete = Callable[[str, str], str]  # (system, user) -> reply


class QueryTranslator:
    """English query -> German query, with an on-disk cache. `complete` is any LLM call."""

    def __init__(self, complete: Complete | None, cache_path: str | Path | None = None):
        self._complete = complete
        self._path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        if self._path and self._path.exists():
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))

    def __call__(self, query: str) -> list[str]:
        """Extra queries to search with besides the original (empty for German queries)."""
        if detect_lang(query) != "en":
            return []
        if not self._cache.get(query):  # missing, or an earlier empty reply
            if self._complete is None:
                raise KeyError(f"no cached translation and no LLM available for: {query!r}")
            for _attempt in range(2):  # the model occasionally returns nothing: retry once
                reply = self._complete(SYSTEM_PROMPT, query).strip().strip("\"'")
                if reply:
                    self._cache[query] = reply
                    self._save()
                    break
        translation = self._cache.get(query, "")
        return [translation] if translation and translation != query else []

    def _save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=1, sort_keys=True),
                encoding="utf-8",
            )


def default_translator(cache_path: str | Path) -> QueryTranslator:
    """Translator backed by the Anthropic API; the key is only needed for a missing translation."""

    class Lazy:
        llm = None

        def __call__(self, system: str, user: str) -> str:
            if self.llm is None:
                from studybot.answer import AnthropicLLM

                self.llm = AnthropicLLM(max_tokens=200)
            return self.llm.complete(system, user)

    return QueryTranslator(Lazy(), cache_path)
