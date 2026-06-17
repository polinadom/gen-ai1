"""
Расширенная оценка: 10 вопросов
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    # Оригинальные 4 вопроса
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],
        "comment": "Базовый тест — один инструмент, одно число."
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами."
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика."
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Вычисление с формулой: 72 / ставка = годы."
    },
    
    # Новые вопросы - 2 требуют compare_periods
    {
        "id": 5,
        "query": "Во сколько раз выросла инфляция с января 2023 по январь 2024?",
        "expected_tools": ["compare_periods"],
        "must_have": ["раз", "инфляц"],
        "comment": "Использует compare_periods для инфляции"
    },
    {
        "id": 6,
        "query": "На сколько процентов изменилась безработица с марта 2023 по март 2024?",
        "expected_tools": ["compare_periods"],
        "must_have": ["безработиц", "%"],
        "comment": "Использует compare_periods для безработицы"
    },
    
    # Трудные вопросы (2)
    {
        "id": 7,
        "query": "Что больше: инфляция в январе 2024 или безработица в том же месяце?",
        "expected_tools": ["get_inflation", "get_unemployment"],
        "must_have": ["инфляц", "безработиц"],
        "comment": "ТРУДНЫЙ: сравниваются разные метрики, агент может запутаться"
    },
    {
        "id": 8,
        "query": "Какая была реальная ключевая ставка в марте 2022?",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["реальн", "%"],
        "comment": "ТРУДНЫЙ: историческая дата, может быть неоднозначность с датами"
    },
    
    # Реальные макро-вопросы (2)
    {
        "id": 9,
        "query": "Как изменился индекс нищеты (инфляция + безработица) с 2023 по 2024 год?",
        "expected_tools": ["get_inflation", "get_unemployment", "calculate"],
        "must_have": ["индекс", "нищет"],
        "comment": "Реальный макро-вопрос: индекс нищеты"
    },
    {
        "id": 10,
        "query": "Какой был пик безработицы в 2022 году?",
        "expected_tools": ["get_unemployment"],
        "must_have": ["пик", "безработиц"],
        "comment": "Реальный макро-вопрос: анализ безработицы"
    }
]


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match

    print(f"\n  tools used : {used_tools}")
    print(f"  expected    : {case['expected_tools']}  -> {'OK' if tool_match else 'MISS'}")
    print(f"  answer      : {answer[:200]}")
    print(f"  must_have   : {case['must_have']}  -> {'OK' if text_match else 'MISS'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Расширенная оценка макро-агента (10 вопросов)")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Блок 9: общий кэш инструментов на все вопросы"
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="Блок 10: показать токены и стоимость по шагам"
    )
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in CASES]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(CASES)} пройдено")
    for r in results:
        mark = "[OK]  " if r["ok"] else "[FAIL]"
        print(f"  {mark} Q{r['id']} ({r['steps']} шагов) — {r['query'][:60]}")

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(f"\n[кэш] на {len(CASES)} вопросах: {h} попаданий из {h + m} обращений")

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()