# Финальный проект: анализ отзывов о доставке еды

Конвейер для анализа отзывов о сервисах Яндекс Еда, Самокат и Купер.

## Техники курса

| Техника | Модуль |
|---------|--------|
| RAG (dense + BM25 + RRF) | `rag.py` |
| Структурированный IE + `field_validator` | `pipeline.py`, `schema.py` |
| Map-Reduce сводка | `pipeline.py` |
| LLM-as-judge | `pipeline.py` |
| Агент с инструментами | `agent.py`, `tools.py` |
| Мультиагент (PWC) | `orchestrator.py` |
| Проверка галлюцинаций | `hallucination.py` |

## Быстрый старт

Вы в папке `финальный_проект`

```bash
pip install -r requirements.txt
copy .env.example .env            # для полного LLM-прогона заполните LLM_AUTH_TOKEN

# Офлайн без API (артефакты для проверки структуры):
python pipeline.py mock
python eval.py

# Полный прогон с LLM:
python pipeline.py run
python eval.py
```

Одна команда «всё сразу» (с API):

```bash
python pipeline.py run && python eval.py
```

## Структура

```
финальный_проект/
├── pipeline.py       # IE → аспекты → Map-Reduce → judge + RAG ingest
├── rag.py            # гибридный поиск и Q&A
├── agent.py          # ReAct-агент с инструментами
├── orchestrator.py   # мультиагент PWC
├── eval.py           # 17 eval-кейсов
├── schema.py         # Pydantic + field_validator (дата, рейтинг 1-5)
├── hallucination.py  # ghost-цитаты, подозрительные числа
├── input/
│   └── reviews_corpus.txt   # 25 отзывов (синтетические, авторские)
└── output/           # артефакты прогона
    ├── reviews.json
    ├── aspects.json
    ├── summary.json
    ├── judge_report.json
    ├── metrics.json
    ├── eval_results.json
    └── trace.jsonl
```

## Данные

Корпус `input/reviews_corpus.txt` — 25 синтетических отзывов на русском языке о трёх сервисах доставки. Формат: блоки `=== REVIEW N ===` с метаданными (автор, рейтинг, сервис, платформа, дата).

## Eval

17 кейсов: 5 RAG, 7 agent, 2 multi-agent, 3 pipeline-check. Метрики: pass-rate, шаги агента, использованные инструменты, токены, ghost-цитаты.

Запуск отдельных кейсов: `python eval.py R1 A3`

## Отчёт

См. `отчёт.md`.
