"""Пайплайн: IE → аспекты → Map-Reduce → LLM-as-judge + RAG ingest."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import ValidationError

from hallucination import check_issue_quotes, check_quotes
from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
)
from rag import ingest
from schema import (
    ChunkSummary,
    JudgeReport,
    Review,
    ReviewSentiment,
    ReviewsSummary,
)
from tools import set_reviews

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "input" / "reviews_corpus.txt"
OUTPUT_DIR = BASE_DIR / "output"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = make_client()
    return _client


def load_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_by_review(corpus: str) -> list[str]:
    chunks = re.findall(
        r"=== REVIEW \d+ ===.*?(?=\n=== REVIEW \d+ ===|\Z)",
        corpus,
        re.DOTALL,
    )
    return [c.strip() for c in chunks if c.strip()]


def _call(model_cls, system: str, user: str):
    client = _get_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_model=model_cls,
        temperature=0.0,
        max_retries=3,
    )


def extract_reviews(corpus: str) -> tuple[list[Review], int]:
    raw_items = _call(list[Review], IE_SYSTEM, corpus)
    valid, errors = [], 0
    for item in raw_items:
        try:
            valid.append(Review.model_validate(item.model_dump()))
        except ValidationError:
            errors += 1
    return valid, errors


def extract_aspects(corpus: str) -> list[ReviewSentiment]:
    return _call(list[ReviewSentiment], ASPECTS_SYSTEM, corpus)


def summarize_chunk(chunk: str) -> ChunkSummary:
    return _call(ChunkSummary, CHUNK_SYSTEM, chunk)


def reduce_summaries(summaries: list[ChunkSummary]) -> ReviewsSummary:
    joined = "\n\n".join(
        f"## {s.author} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    return _call(ReviewsSummary, REDUCE_SYSTEM, joined)


def summarize_reviews(corpus: str, workers: int = 4) -> ReviewsSummary:
    chunks = split_by_review(corpus)
    if len(chunks) <= 1:
        chunks = [corpus[i : i + 1200] for i in range(0, len(corpus), 1000)] or [corpus]
    summaries: list[ChunkSummary | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summarize_chunk, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futures):
            summaries[futures[fut]] = fut.result()
    return reduce_summaries([s for s in summaries if s])


def run_judge(reviews: list[dict], summary: dict) -> JudgeReport:
    evidence = "## Рекомендации:\n"
    for i, action in enumerate(summary.get("action_items", []), 1):
        evidence += f"{i}. {action}\n"
    evidence += "\n## Отзывы (issues):\n"
    for r in reviews[:15]:
        for issue in r.get("issues", [])[:2]:
            evidence += f"- {r.get('author')}: [{issue.get('category')}] {issue.get('quote', '')[:80]}\n"
    return _call(JudgeReport, JUDGE_SYSTEM, evidence)


def analyze(input_path: Path | None = None, out_dir: Path | None = None) -> dict:
    inp = input_path or INPUT_FILE
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    corpus = load_corpus(inp)
    n_reviews_in = len(split_by_review(corpus))

    print("-> RAG ingest...")
    n_chunks = ingest(inp)
    print(f"   {n_chunks} чанков проиндексировано")

    print(f"-> IE: извлечение отзывов ({n_reviews_in} блоков)...")
    reviews, val_errors = extract_reviews(corpus)
    reviews_data = [r.model_dump(mode="json") for r in reviews]
    (out / "reviews.json").write_text(
        json.dumps(reviews_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    set_reviews(reviews_data)
    print(f"   {len(reviews)} валидных, {val_errors} ValidationError")

    print("-> Аспектный анализ...")
    aspects = extract_aspects(corpus)
    ghosts_aspect = check_quotes(aspects, corpus)
    ghosts_issue = check_issue_quotes(reviews_data, corpus)
    aspects_data = [a.model_dump() for a in aspects]
    (out / "aspects.json").write_text(
        json.dumps(aspects_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_quotes = sum(len(r.aspects) for r in aspects) + sum(
        len(r.get("issues", [])) for r in reviews_data
    )
    all_ghosts = ghosts_aspect + [(a, q) for a, q in ghosts_issue]
    print(f"   ghost-цитат: {len(all_ghosts)} / {total_quotes}")

    print("-> Map-Reduce...")
    summary = summarize_reviews(corpus)
    summary_dict = json.loads(summary.model_dump_json())
    (out / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-> LLM-as-judge...")
    report = run_judge(reviews_data, summary_dict)
    if report.overall_score < 0.65:
        print(f"   score={report.overall_score:.2f} — повтор REDUCE...")
        summary = summarize_reviews(corpus, workers=2)
        summary_dict = json.loads(summary.model_dump_json())
        (out / "summary.json").write_text(
            json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = run_judge(reviews_data, summary_dict)
    (out / "judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    metrics = {
        "input_reviews": n_reviews_in,
        "valid_reviews": len(reviews),
        "validation_errors": val_errors,
        "ghost_quotes": len(all_ghosts),
        "total_quotes": total_quotes,
        "ghost_quote_rate_pct": round(len(all_ghosts) / max(1, total_quotes) * 100, 1),
        "overall_judge_score": report.overall_score,
        "rag_chunks": n_chunks,
        "elapsed_sec": round(elapsed, 1),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== ИТОГ ===")
    print(summary.headline)
    print(f"Judge score: {report.overall_score:.2f}")
    print(f"Ghost quotes: {len(all_ghosts)}/{total_quotes}")
    print(f"Артефакты: {out}/")
    return metrics


def mock_analyze(input_path: Path | None = None, out_dir: Path | None = None) -> dict:
    """Офлайн-прогон без LLM: парсер + эвристики для воспроизводимых артефактов."""
    from corpus_parser import parse_corpus

    inp = input_path or INPUT_FILE
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(inp)
    n_chunks = ingest(inp)

    reviews = parse_corpus(corpus)
    reviews_data = [r.model_dump(mode="json") for r in reviews]
    (out / "reviews.json").write_text(
        json.dumps(reviews_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    set_reviews(reviews_data)

    ghosts_issue = check_issue_quotes(reviews_data, corpus)
    summary_dict = {
        "headline": "Главные боли: скорость доставки, качество еды и поддержка",
        "key_findings": [
            "Чаще всего жалуются на опоздания и холодную еду",
            "Самокат хвалят за скорость, но ругают за рост цен",
            "Поддержка Купера и Яндекс Еды часто отвечает шаблонами",
        ],
        "action_items": [
            "Улучшить трекинг ETA и компенсации за опоздания",
            "Контроль температуры еды при передаче курьеру",
            "Ускорить эскалацию в живую поддержку",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "aspects.json").write_text("[]", encoding="utf-8")
    (out / "judge_report.json").write_text(
        json.dumps(
            {
                "verdicts": [
                    {
                        "action": summary_dict["action_items"][0],
                        "support": "supported",
                        "evidence": ["опоздан", "минут"],
                        "comment": "множество жалоб на delivery_speed",
                    }
                ],
                "overall_score": 0.78,
                "summary": "Рекомендации в целом подкреплены отзывами",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    total_quotes = sum(len(r.get("issues", [])) for r in reviews_data)
    metrics = {
        "input_reviews": len(split_by_review(corpus)),
        "valid_reviews": len(reviews),
        "validation_errors": 0,
        "ghost_quotes": len(ghosts_issue),
        "total_quotes": total_quotes,
        "ghost_quote_rate_pct": round(len(ghosts_issue) / max(1, total_quotes) * 100, 1),
        "overall_judge_score": 0.78,
        "rag_chunks": n_chunks,
        "mode": "mock",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Mock-прогон: {len(reviews)} отзывов, {n_chunks} чанков, ghost={len(ghosts_issue)}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Пайплайн анализа отзывов о доставке еды")
    parser.add_argument(
        "command",
        choices=["run", "ingest", "analyze", "mock"],
        nargs="?",
        default="run",
        help="run = ingest + analyze; mock = офлайн без LLM",
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.command == "ingest":
        n = ingest(args.input)
        print(f"Индексировано {n} чанков")
    elif args.command == "mock":
        mock_analyze(args.input, args.output)
    elif args.command == "analyze":
        analyze(args.input, args.output)
    else:
        analyze(args.input, args.output)


if __name__ == "__main__":
    main()
