"""Локальный парсер корпуса без LLM — для mock-режима и тестов."""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from schema import Issue, Review


def parse_corpus(corpus: str) -> list[Review]:
    blocks = re.findall(
        r"=== REVIEW (\d+) ===(.*?)(?=\n=== REVIEW \d+ ===|\Z)",
        corpus,
        re.DOTALL,
    )
    reviews = []
    for _num, body in blocks:
        author = _field(body, r"Автор:\s*(.+)")
        rating_s = _field(body, r"Рейтинг:\s*(\d+)")
        service = _field(body, r"Сервис:\s*(.+)")
        platform = _field(body, r"Платформа:\s*(.+)")
        date_s = _field(body, r"Дата:\s*(\d{4}-\d{2}-\d{2})")
        text_lines = []
        for line in body.strip().splitlines():
            if line.startswith(("Автор:", "Рейтинг:", "Сервис:", "Платформа:", "Дата:")):
                continue
            text_lines.append(line)
        text = " ".join(text_lines).strip()
        if not author or not service or len(text) < 10:
            continue
        rating = int(rating_s) if rating_s else None
        review_date: Optional[date] = None
        if date_s:
            review_date = date.fromisoformat(date_s)
        issues = _heuristic_issues(text)
        try:
            reviews.append(
                Review(
                    author=author,
                    rating=rating,
                    service=service.strip(),  # type: ignore[arg-type]
                    platform=platform.strip(),  # type: ignore[arg-type]
                    review_date=review_date,
                    text=text,
                    issues=issues,
                    competitor_mentions=_competitors(text, service),
                )
            )
        except Exception:
            continue
    return reviews


def _field(body: str, pattern: str) -> str:
    m = re.search(pattern, body)
    return m.group(1).strip() if m else ""


def _heuristic_issues(text: str) -> list[Issue]:
    rules = [
        (r"опоздан|минут|долг", "delivery_speed"),
        (r"холодн|просроч|порча|вкус|салат", "food_quality"),
        (r"дорог|цен|комисси|руб", "price"),
        (r"приложен|интерфейс|вылет|тормоз|завис", "app_ui"),
        (r"поддержк|чат|тикет|оператор", "support"),
        (r"курьер", "courier"),
    ]
    issues = []
    t_lower = text.lower()
    for pattern, cat in rules:
        m = re.search(pattern, t_lower)
        if m:
            start = max(0, m.start() - 20)
            quote = text[start : m.end() + 40].strip()[:80]
            issues.append(Issue(category=cat, severity=3, quote=quote))
    return issues[:3]


def _competitors(text: str, service: str) -> list[str]:
    names = ["Яндекс Еда", "Самокат", "Купер", "Delivery"]
    found = [n for n in names if n.lower() in text.lower() and n not in service]
    return found
