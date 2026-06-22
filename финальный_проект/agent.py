"""ReAct-агент с инструментами для аналитических запросов по отзывам."""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any

from llm_client import get_model, make_raw_client
from schema import AgentAnswer
from tools import TOOL_SCHEMAS, TOOLS_IMPL

from hallucination import check_number_hallucination

BASE_DIR = Path(__file__).parent
TRACE_FILE = BASE_DIR / "output" / "trace.jsonl"

SYSTEM_PROMPT = """Ты — аналитик отзывов о сервисах доставки еды (Яндекс Еда, Самокат, Купер).
ЧИСЛА НЕ ПРИДУМЫВАЙ — получай через инструменты.

Инструменты:
- search_reviews: поиск по ключевым словам в отзывах
- get_service_stats: средний рейтинг и топ-жалобы по сервису
- compare_services: сравнение двух сервисов
- calculate: арифметика над полученными числами

Когда данных достаточно — вызови submit_answer со структурой (answer, value, unit, sources, confidence).
Формат ответа — кратко, с числами и единицами."""

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Финальный структурированный ответ",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "value": {"type": ["number", "null"]},
                "unit": {"type": ["string", "null"]},
                "sources": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        },
    },
}


def write_trace(run_id: str, entry: dict) -> None:
    TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry["run_id"] = run_id
    entry["ts"] = datetime.datetime.now().isoformat()
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_agent(
    query: str,
    *,
    max_iter: int = 8,
    structured: bool = True,
    trace: bool = True,
) -> dict[str, Any]:
    client = make_raw_client()
    model = get_model()
    run_id = str(uuid.uuid4())
    tools = TOOL_SCHEMAS + ([SUBMIT_SCHEMA] if structured else [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tools_used: list[str] = []
    tool_log: list[dict] = []
    steps = 0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for step in range(1, max_iter + 1):
        steps = step
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        if resp.usage:
            usage_total["prompt_tokens"] += resp.usage.prompt_tokens or 0
            usage_total["completion_tokens"] += resp.usage.completion_tokens or 0
            usage_total["total_tokens"] += resp.usage.total_tokens or 0

        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer_text = msg.content or ""
            if trace:
                write_trace(run_id, {"step": step, "final": answer_text})
            suspicious = check_number_hallucination(answer_text, tool_log)
            return {
                "answer": answer_text,
                "tools_used": tools_used,
                "steps": steps,
                "usage": usage_total,
                "run_id": run_id,
                "suspicious_numbers": suspicious,
            }

        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "submit_answer":
                ans = AgentAnswer.model_validate(args)
                if trace:
                    write_trace(run_id, {"step": step, "final": ans.answer, "structured": ans.model_dump()})
                suspicious = check_number_hallucination(ans.answer, tool_log)
                return {
                    "answer": ans.answer,
                    "structured": ans.model_dump(),
                    "tools_used": tools_used,
                    "steps": steps,
                    "usage": usage_total,
                    "run_id": run_id,
                    "suspicious_numbers": suspicious,
                }

            fn = TOOLS_IMPL.get(name)
            if fn is None:
                obs = {"error": f"неизвестный инструмент: {name}"}
            else:
                tools_used.append(name)
                obs = fn(**args)
            tool_log.append({"call": name, "args": args, "obs": obs})
            if trace:
                write_trace(run_id, {"step": step, "call": name, "args": args, "obs": obs})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs, ensure_ascii=False),
                }
            )

    return {
        "answer": "Превышен лимит шагов",
        "tools_used": tools_used,
        "steps": steps,
        "usage": usage_total,
        "run_id": run_id,
        "suspicious_numbers": [],
    }
