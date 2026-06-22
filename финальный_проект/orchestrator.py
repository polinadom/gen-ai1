"""Мультиагент: Планировщик → Исполнители (агент) → Критик."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent import run_agent
from llm_client import get_model, make_client, make_raw_client
from prompts import CRITIC_SYSTEM, PLANNER_SYSTEM
from schema import Plan, SubQuestion, Verdict, WorkerAnswer

BASE_DIR = Path(__file__).parent


def planner(question: str) -> Plan:
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": question},
        ],
        response_model=Plan,
        temperature=0.0,
        max_retries=3,
    )


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]) -> None:
        if node_id in visited:
            return
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])
    return ordered


def worker(sq: SubQuestion, prev_answers: dict[int, WorkerAnswer]) -> WorkerAnswer:
    context = ""
    if prev_answers:
        context = "\nКонтекст предыдущих ответов:\n" + "\n".join(
            f"[{k}] {v.answer}" for k, v in sorted(prev_answers.items())
        )
    query = sq.question + context
    res = run_agent(query, max_iter=6, structured=True, trace=False)
    return WorkerAnswer(
        answer=res["answer"],
        used_tools=res.get("tools_used", []),
        confidence=res.get("structured", {}).get("confidence", 0.7),
    )


def critic(question: str, plan: Plan, answers: dict[int, WorkerAnswer]) -> Verdict:
    client = make_client()
    payload = {
        "question": question,
        "plan": plan.model_dump(),
        "answers": {str(k): v.model_dump() for k, v in answers.items()},
    }
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_model=Verdict,
        temperature=0.0,
        max_retries=2,
    )


def _synthesize(question: str, answers: dict[int, WorkerAnswer]) -> str:
    client = make_raw_client()
    parts = "\n".join(f"{k}. {v.answer}" for k, v in sorted(answers.items()))
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {
                "role": "system",
                "content": "Собери ответы подвопросов в 2-3 связных предложения для пользователя.",
            },
            {"role": "user", "content": f"Вопрос: {question}\n\nОтветы:\n{parts}"},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or parts


def run_multi_agent(
    question: str, *, max_iter: int = 2, verbose: bool = False
) -> dict[str, Any]:
    trace: list[dict] = []
    plan = planner(question)
    trace.append({"kind": "plan", "plan": plan.model_dump()})

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        for sq in _topological_sort(plan.subquestions):
            ans = worker(sq, answers)
            answers[sq.id] = ans
            trace.append({"kind": "worker", "sq_id": sq.id, "answer": ans.model_dump()})
            if verbose:
                print(f"  [{sq.id}] {ans.answer[:80]}... tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append({"kind": "verdict", "verdict": verdict.model_dump()})
        if verdict.ok or verdict.action == "accept":
            final = _synthesize(question, answers)
            return {
                "answer": final,
                "plan": plan.model_dump(),
                "answers": {k: v.model_dump() for k, v in answers.items()},
                "verdict": verdict.model_dump(),
                "trace": trace,
                "iterations": iter_num,
            }
        if verdict.action == "replan":
            plan = planner(question + f"\nУчти замечание критика: {verdict.reason}")
            continue
        # rework: перезапускаем указанные подвопросы
        for rid in verdict.rework_ids:
            sq = next(s for s in plan.subquestions if s.id == rid)
            answers[rid] = worker(sq, answers)

    final = _synthesize(question, answers)
    return {
        "answer": final,
        "plan": plan.model_dump(),
        "answers": {k: v.model_dump() for k, v in answers.items()},
        "trace": trace,
        "iterations": max_iter,
    }
