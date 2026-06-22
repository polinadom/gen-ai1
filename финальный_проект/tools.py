"""Инструменты агента: поиск по отзывам и агрегаты."""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
REVIEWS_JSON = BASE_DIR / "output" / "reviews.json"
CORPUS_FILE = BASE_DIR / "input" / "reviews_corpus.txt"

_reviews_cache: list[dict] | None = None
_corpus_cache: str | None = None


def _load_reviews() -> list[dict]:
    global _reviews_cache
    if _reviews_cache is not None:
        return _reviews_cache
    if REVIEWS_JSON.exists():
        _reviews_cache = json.loads(REVIEWS_JSON.read_text(encoding="utf-8"))
        return _reviews_cache
    return []


def set_reviews(reviews: list[dict]) -> None:
    global _reviews_cache
    _reviews_cache = reviews
    REVIEWS_JSON.parent.mkdir(parents=True, exist_ok=True)
    REVIEWS_JSON.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_corpus() -> str:
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = CORPUS_FILE.read_text(encoding="utf-8")
    return _corpus_cache


def search_reviews(query: str, limit: int = 5) -> dict:
    """Ключевой поиск по корпусу отзывов (BM25-подобный по токенам)."""
    corpus = _load_corpus()
    blocks = re.findall(
        r"(=== REVIEW \d+ ===.*?)(?=\n=== REVIEW \d+ ===|\Z)",
        corpus,
        re.DOTALL,
    )
    tokens = set(re.findall(r"[а-яa-zё]{3,}", query.lower()))
    scored = []
    for block in blocks:
        text_lower = block.lower()
        score = sum(1 for t in tokens if t in text_lower)
        if score > 0:
            author_m = re.search(r"Автор:\s*(.+)", block)
            scored.append(
                {
                    "score": score,
                    "author": author_m.group(1).strip() if author_m else "?",
                    "snippet": block.strip()[:400],
                }
            )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "hits": scored[:limit], "total_matched": len(scored)}


def get_service_stats(service: str) -> dict:
    """Средний рейтинг и число отзывов по сервису."""
    reviews = _load_reviews()
    matched = [r for r in reviews if r.get("service") == service]
    if not matched:
        return {"service": service, "count": 0, "avg_rating": None, "error": "нет данных"}
    ratings = [r["rating"] for r in matched if r.get("rating")]
    categories: dict[str, int] = {}
    for r in matched:
        for issue in r.get("issues", []):
            cat = issue.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
    return {
        "service": service,
        "count": len(matched),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "top_issues": sorted(categories.items(), key=lambda x: -x[1])[:5],
    }


def compare_services(service_a: str, service_b: str, aspect: str = "") -> dict:
    """Сравнить два сервиса по рейтингу и жалобам."""
    stats_a = get_service_stats(service_a)
    stats_b = get_service_stats(service_b)
    result = {
        "service_a": stats_a,
        "service_b": stats_b,
        "rating_diff": None,
        "aspect": aspect,
    }
    if stats_a.get("avg_rating") and stats_b.get("avg_rating"):
        result["rating_diff"] = round(stats_a["avg_rating"] - stats_b["avg_rating"], 2)
    if aspect:
        def _aspect_count(stats: dict) -> int:
            return sum(c for cat, c in stats.get("top_issues", []) if cat == aspect)
        result["aspect_complaints_a"] = _aspect_count(stats_a)
        result["aspect_complaints_b"] = _aspect_count(stats_b)
    return result


def calculate(expression: str) -> dict:
    """Безопасный калькулятор для арифметики над числами из инструментов."""
    allowed = set("0123456789+-*/().,% ")
    if not all(ch in allowed for ch in expression):
        return {"error": "недопустимые символы", "expression": expression}
    try:
        expr = expression.replace(",", ".")
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — учебный калькулятор
        return {"expression": expression, "result": float(value)}
    except Exception as e:
        return {"error": str(e), "expression": expression}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Поиск отзывов по ключевым словам в корпусе",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_stats",
            "description": "Статистика отзывов по сервису: Яндекс Еда, Самокат или Купер",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_services",
            "description": "Сравнить два сервиса доставки",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_a": {"type": "string"},
                    "service_b": {"type": "string"},
                    "aspect": {"type": "string", "default": ""},
                },
                "required": ["service_a", "service_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Арифметика над числами из других инструментов",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOLS_IMPL = {
    "search_reviews": search_reviews,
    "get_service_stats": get_service_stats,
    "compare_services": compare_services,
    "calculate": calculate,
}
