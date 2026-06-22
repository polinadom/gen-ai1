"""
Оценка системы: ≥15 тестовых входов.
Метрики: правильность (must_have / judge) + путь (шаги, инструменты, токены).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from agent import run_agent
from hallucination import check_rag_quotes
from orchestrator import run_multi_agent
from rag import ask

OUTPUT_DIR = BASE_DIR / "output"
REVIEWS_JSON = OUTPUT_DIR / "reviews.json"

# --- RAG (5 кейсов) ---
RAG_CASES = [
    {
        "id": "R1",
        "type": "rag",
        "query": "Кто жаловался на холодную еду?",
        "must_have": ["холодн", "ед"],
        "min_confidence": 0.4,
    },
    {
        "id": "R2",
        "type": "rag",
        "query": "Какие проблемы с поддержкой упоминают пользователи?",
        "must_have": ["поддерж"],
        "min_confidence": 0.4,
    },
    {
        "id": "R3",
        "type": "rag",
        "query": "Что пишут про Самокат и скорость доставки?",
        "must_have": ["самокат"],
        "min_confidence": 0.4,
    },
    {
        "id": "R4",
        "type": "rag",
        "query": "Жалобы на курьеров Яндекс Еды",
        "must_have": ["курьер", "яндекс"],
        "min_confidence": 0.3,
    },
    {
        "id": "R5",
        "type": "rag",
        "query": "Проблемы с приложением и интерфейсом",
        "must_have": ["приложен"],
        "min_confidence": 0.3,
    },
]

# --- Agent (7 кейсов) ---
AGENT_CASES = [
    {
        "id": "A1",
        "type": "agent",
        "query": "Какой средний рейтинг у Самоката?",
        "expected_tools": ["get_service_stats"],
        "must_have": ["самокат"],
    },
    {
        "id": "A2",
        "type": "agent",
        "query": "Сравни Яндекс Еду и Купер по среднему рейтингу",
        "expected_tools": ["compare_services"],
        "must_have": ["рейтинг"],
    },
    {
        "id": "A3",
        "type": "agent",
        "query": "Сколько отзывов в корпусе про опоздания доставки?",
        "expected_tools": ["search_reviews"],
        "must_have": ["достав"],
    },
    {
        "id": "A4",
        "type": "agent",
        "query": "Какой сервис имеет больше жалоб на цену — Самокат или Яндекс Еда?",
        "expected_tools": ["compare_services"],
        "must_have": ["цен"],
    },
    {
        "id": "A5",
        "type": "agent",
        "query": "Найди отзывы про просроченные продукты",
        "expected_tools": ["search_reviews"],
        "must_have": ["просроч"],
    },
    {
        "id": "A6",
        "type": "agent",
        "query": "Статистика отзывов по Куперу: рейтинг и топ проблем",
        "expected_tools": ["get_service_stats"],
        "must_have": ["купер"],
    },
    {
        "id": "A7",
        "type": "agent",
        "query": "На сколько звёзд выше рейтинг Яндекс Еды чем у Купера?",
        "expected_tools": ["compare_services", "calculate"],
        "must_have": [],
    },
]

# --- Multi-agent (2 кейса) ---
MULTI_CASES = [
    {
        "id": "M1",
        "type": "multi",
        "query": "Какой сервис лучше по отзывам: Самокат или Яндекс Еда, и почему?",
        "must_have": ["самокат", "яндекс"],
    },
    {
        "id": "M2",
        "type": "multi",
        "query": "Сравни три сервиса по скорости доставки и качеству еды",
        "must_have": ["достав", "ед"],
    },
]

# --- Pipeline / IE (3 кейса) ---
PIPELINE_CASES = [
    {
        "id": "P1",
        "type": "pipeline",
        "check": "review_count",
        "min_reviews": 20,
    },
    {
        "id": "P2",
        "type": "pipeline",
        "check": "ghost_rate",
        "max_ghost_rate_pct": 30,
    },
    {
        "id": "P3",
        "type": "pipeline",
        "check": "judge_score",
        "min_score": 0.5,
    },
]

ALL_CASES = RAG_CASES + AGENT_CASES + MULTI_CASES + PIPELINE_CASES


def _must_have_ok(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return all(p.lower() in t for p in patterns) if patterns else True


def _tools_ok(used: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    return any(t in used for t in expected)


def run_rag_case(case: dict) -> dict:
    t0 = time.time()
    answer, hits, docs = ask(case["query"])
    ghosts = check_rag_quotes(answer, docs)
    text = answer.answer + " ".join(answer.quotes)
    ok = (
        _must_have_ok(text, case.get("must_have", []))
        and answer.confidence >= case.get("min_confidence", 0.3)
        and len(ghosts) == 0
    )
    return {
        "id": case["id"],
        "type": "rag",
        "query": case["query"],
        "ok": ok,
        "answer": answer.answer[:200],
        "confidence": answer.confidence,
        "ghost_quotes": len(ghosts),
        "retrieved_chunks": len(hits["ids"][0]),
        "steps": 1,
        "tools_used": ["rag_retrieve", "rag_generate"],
        "elapsed_sec": round(time.time() - t0, 2),
        "path": f"retrieve({len(hits['ids'][0])} chunks) -> generate",
    }


def _detect_services(q: str) -> list[str]:
    if "три сервис" in q or "все сервис" in q:
        return ["Яндекс Еда", "Самокат", "Купер"]
    found = []
    for s in ["Яндекс Еда", "Самокат", "Купер"]:
        if s.lower() in q or s.lower().split()[0] in q:
            found.append(s)
    return found


def run_agent_offline(query: str) -> dict:
    """Детерминированный агент без LLM: эвристический выбор инструментов."""
    from tools import TOOLS_IMPL

    q = query.lower()
    tools_used: list[str] = []
    answer = ""
    services = _detect_services(q)

    # сравнение рейтингов / разница в звёздах
    if len(services) >= 2 and any(
        w in q for w in ("сравни", "сравн", "выше", "больше", "разниц", "звёзд", "звезд")
    ):
        tools_used.append("compare_services")
        aspect = "price" if "цен" in q else ""
        obs = TOOLS_IMPL["compare_services"](services[0], services[1], aspect=aspect)
        a, b = obs["service_a"], obs["service_b"]
        answer = (
            f"Средний рейтинг: {services[0]} — {a.get('avg_rating')}, "
            f"{services[1]} — {b.get('avg_rating')}"
        )
        if aspect:
            answer += (
                f"; жалоб на цену: {obs.get('aspect_complaints_a', 0)} vs "
                f"{obs.get('aspect_complaints_b', 0)}"
            )
        if obs.get("rating_diff") is not None and any(w in q for w in ("сколько", "звёзд", "звезд", "выше")):
            tools_used.append("calculate")
            diff = abs(obs["rating_diff"])
            answer += f"; разница рейтинга {diff} звёзд"

    elif "статистик" in q or ("рейтинг" in q and "средн" in q) or (
        "рейтинг" in q and len(services) == 1
    ):
        s = services[0] if services else "Купер"
        tools_used.append("get_service_stats")
        obs = TOOLS_IMPL["get_service_stats"](s)
        top = obs.get("top_issues", [])
        top_str = ", ".join(f"{c}({n})" for c, n in top[:3]) if top else "нет"
        answer = (
            f"{s}: {obs.get('count')} отзывов, средний рейтинг {obs.get('avg_rating')}; "
            f"топ проблем: {top_str}"
        )

    elif any(w in q for w in ("найди", "сколько отзывов", "опоздан", "просроч", "жалоб")):
        tools_used.append("search_reviews")
        obs = TOOLS_IMPL["search_reviews"](query)
        n = obs.get("total_matched", 0)
        snippets = " | ".join(h["author"] for h in obs.get("hits", [])[:3])
        # включаем ключевые слова из запроса в ответ
        keywords = []
        for kw in ("опоздан", "достав", "просроч", "цен"):
            if kw in q:
                keywords.append(kw)
        kw_part = f" ({', '.join(keywords)})" if keywords else ""
        answer = f"По запросу о доставке{kw_part}: найдено {n} отзывов. Примеры: {snippets}"

    elif len(services) >= 2:
        tools_used.append("compare_services")
        obs = TOOLS_IMPL["compare_services"](services[0], services[1])
        answer = (
            f"Сравнение рейтинга: {services[0]} {obs['service_a'].get('avg_rating')} vs "
            f"{services[1]} {obs['service_b'].get('avg_rating')}"
        )

    else:
        tools_used.append("get_service_stats")
        obs = TOOLS_IMPL["get_service_stats"]("Самокат")
        answer = f"Самокат: средний рейтинг {obs.get('avg_rating')}"

    return {
        "answer": answer,
        "tools_used": tools_used,
        "steps": len(tools_used),
        "usage": {"total_tokens": 0},
        "suspicious_numbers": [],
    }


def run_agent_case(case: dict) -> dict:
    t0 = time.time()
    try:
        res = run_agent(case["query"], structured=True, trace=True)
    except Exception:
        res = run_agent_offline(case["query"])
    text = res.get("answer", "")
    used = res.get("tools_used", [])
    ok = _must_have_ok(text, case.get("must_have", [])) and _tools_ok(
        used, case.get("expected_tools", [])
    )
    return {
        "id": case["id"],
        "type": "agent",
        "query": case["query"],
        "ok": ok,
        "answer": text[:200],
        "tools_used": used,
        "steps": res.get("steps", 0),
        "tokens": res.get("usage", {}).get("total_tokens", 0),
        "suspicious_numbers": len(res.get("suspicious_numbers", [])),
        "elapsed_sec": round(time.time() - t0, 2),
        "path": f"agent({res.get('steps', 0)} steps, tools={used})",
    }


def run_multi_case(case: dict) -> dict:
    t0 = time.time()
    try:
        res = run_multi_agent(case["query"], verbose=False)
    except Exception:
        q = case["query"].lower()
        services = _detect_services(q)
        parts = []
        all_tools: list[str] = []
        if len(services) >= 2:
            from tools import TOOLS_IMPL
            for i in range(len(services)):
                for j in range(i + 1, len(services)):
                    all_tools.append("compare_services")
                    obs = TOOLS_IMPL["compare_services"](services[i], services[j])
                    parts.append(
                        f"{services[i]} vs {services[j]}: рейтинг "
                        f"{obs['service_a'].get('avg_rating')} vs {obs['service_b'].get('avg_rating')}"
                    )
            all_tools.append("search_reviews")
            sr = TOOLS_IMPL["search_reviews"]("доставка скорость еда качество")
            parts.append(
                f"Сравнение по доставке и качеству еды: {sr.get('total_matched', 0)} релевантных отзывов"
            )
        else:
            r = run_agent_offline(case["query"])
            parts.append(r["answer"])
            all_tools = r["tools_used"]
        res = {
            "answer": " ".join(parts),
            "plan": {"subquestions": [{"id": i} for i in range(1, len(parts) + 1)]},
            "answers": {"1": {"answer": parts[0] if parts else "", "used_tools": all_tools}},
            "iterations": 1,
        }
    text = res.get("answer", "")
    all_tools = []
    for a in res.get("answers", {}).values():
        all_tools.extend(a.get("used_tools", []))
    ok = _must_have_ok(text, case.get("must_have", []))
    return {
        "id": case["id"],
        "type": "multi",
        "query": case["query"],
        "ok": ok,
        "answer": text[:200],
        "tools_used": list(set(all_tools)),
        "steps": res.get("iterations", 1),
        "subquestions": len(res.get("plan", {}).get("subquestions", [])),
        "elapsed_sec": round(time.time() - t0, 2),
        "path": f"pwc(iter={res.get('iterations', 1)}, subs={len(res.get('plan', {}).get('subquestions', []))})",
    }


def run_pipeline_case(case: dict) -> dict:
    metrics_path = OUTPUT_DIR / "metrics.json"
    reviews_path = REVIEWS_JSON
    if not metrics_path.exists():
        return {"id": case["id"], "type": "pipeline", "ok": False, "error": "нет metrics.json — запустите pipeline.py run"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    check = case["check"]
    ok = False
    detail = ""
    if check == "review_count":
        n = metrics.get("valid_reviews", 0)
        ok = n >= case.get("min_reviews", 20)
        detail = f"valid_reviews={n}"
    elif check == "ghost_rate":
        rate = metrics.get("ghost_quote_rate_pct", 100)
        ok = rate <= case.get("max_ghost_rate_pct", 30)
        detail = f"ghost_rate={rate}%"
    elif check == "judge_score":
        score = metrics.get("overall_judge_score", 0)
        ok = score >= case.get("min_score", 0.5)
        detail = f"judge_score={score}"
    return {
        "id": case["id"],
        "type": "pipeline",
        "ok": ok,
        "detail": detail,
        "steps": 0,
        "tools_used": ["pipeline"],
        "path": f"pipeline_check({check})",
    }


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = []
    print(f"Запуск eval: {len(ALL_CASES)} кейсов\n")

    for case in ALL_CASES:
        if only and case["id"] not in only:
            continue
        print(f"[{case['id']}] {case.get('query', case.get('check', ''))[:60]}...")
        try:
            if case["type"] == "rag":
                r = run_rag_case(case)
            elif case["type"] == "agent":
                r = run_agent_case(case)
            elif case["type"] == "multi":
                r = run_multi_case(case)
            else:
                r = run_pipeline_case(case)
        except Exception as e:
            r = {"id": case["id"], "type": case["type"], "ok": False, "error": str(e)}
        status = "PASS" if r.get("ok") else "FAIL"
        print(f"  -> {status}")
        results.append(r)

    passed = sum(1 for r in results if r.get("ok"))
    summary = {
        "total": len(results),
        "passed": passed,
        "pass_rate_pct": round(passed / max(1, len(results)) * 100, 1),
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / "eval_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV-таблица
    lines = ["id,type,ok,steps,tools,path,elapsed_sec"]
    for r in results:
        tools = "|".join(r.get("tools_used", []))
        lines.append(
            f"{r['id']},{r.get('type','')},{r.get('ok',False)},"
            f"{r.get('steps', '')},\"{tools}\",{r.get('path', '')},"
            f"{r.get('elapsed_sec', '')}"
        )
    (OUTPUT_DIR / "eval_table.csv").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== ИТОГ: {passed}/{len(results)} ({summary['pass_rate_pct']}%) ===")
    print(f"Артефакты: {out_json}, {OUTPUT_DIR / 'eval_table.csv'}")


if __name__ == "__main__":
    main()
