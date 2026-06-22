"""Проверка галлюцинаций: ghost-цитаты и выдуманные числа."""

from __future__ import annotations

import re

from schema import RAGAnswer, ReviewSentiment


def check_quotes(aspects: list[ReviewSentiment], corpus: str) -> list[tuple[str, str]]:
    """Ghost-цитаты в аспектном анализе: цитата не найдена в корпусе."""
    t = corpus.lower()
    ghosts = []
    for r in aspects:
        for a in r.aspects:
            probe = a.quote.strip().lower()[:40]
            if probe and probe not in t:
                ghosts.append((r.author, a.quote))
    return ghosts


def check_rag_quotes(answer: RAGAnswer, retrieved_docs: list[str]) -> list[str]:
    """Ghost-цитаты в RAG-ответе: цитата не из ретривленного контекста."""
    ctx = "\n".join(retrieved_docs).lower()
    ghosts = []
    for q in answer.quotes:
        probe = q.strip().lower()[:40]
        if probe and probe not in ctx:
            ghosts.append(q)
    return ghosts


def check_issue_quotes(reviews: list[dict], corpus: str) -> list[tuple[str, str]]:
    """Ghost-цитаты в извлечённых issues."""
    t = corpus.lower()
    ghosts = []
    for r in reviews:
        for issue in r.get("issues", []):
            quote = issue.get("quote", "")
            probe = quote.strip().lower()[:40]
            if probe and probe not in t:
                ghosts.append((r.get("author", "?"), quote))
    return ghosts


_NUM_RE = re.compile(r"\d+[.,]?\d*")


def check_number_hallucination(
    answer_text: str, tool_log: list[dict]
) -> list[str]:
    """
    Числа в финальном ответе, которых нет в логе инструментов.
    Возвращает список подозрительных чисел.
    """
    log_text = " ".join(str(e) for e in tool_log).lower()
    suspicious = []
    for m in _NUM_RE.findall(answer_text):
        norm = m.replace(",", ".")
        if norm not in log_text and m not in log_text:
            # пропускаем мелкие целые 1-5 (рейтинги)
            try:
                val = float(norm)
                if 1 <= val <= 5 and val == int(val):
                    continue
            except ValueError:
                pass
            suspicious.append(m)
    return suspicious
