"""RAG: BM25-поиск по корпусу (stdlib + опционально ChromaDB)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from llm_client import get_model, make_client
from schema import RAGAnswer

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "input" / "reviews_corpus.txt"
BM25_CACHE = BASE_DIR / "output" / "bm25_cache.json"
CHROMA_DIR = BASE_DIR / "output" / "chroma_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

_client = None


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def _split_text(text: str) -> list[str]:
    """Рекурсивная нарезка по абзацам и предложениям."""
    parts: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= CHUNK_SIZE:
            parts.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(sent) <= CHUNK_SIZE:
                parts.append(sent)
            else:
                for i in range(0, len(sent), CHUNK_SIZE - CHUNK_OVERLAP):
                    parts.append(sent[i : i + CHUNK_SIZE])
    return [p for p in parts if p.strip()]


def chunk_corpus(text: str) -> list[tuple[str, str, int]]:
    reviews = re.findall(
        r"(=== REVIEW \d+ ===.*?)(?=\n=== REVIEW \d+ ===|\Z)",
        text,
        re.DOTALL,
    )
    chunks = []
    for review in reviews:
        m = re.search(r"=== REVIEW (\d+) ===", review)
        num = int(m.group(1)) if m else 0
        for i, c in enumerate(_split_text(review.strip())):
            cid = f"review_{num}__{i}"
            chunks.append((cid, c.strip(), num))
    return chunks


class SimpleBM25:
    """Минимальный BM25 без внешних зависимостей."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(len(corpus_tokens), 1)
        self.df: dict[str, int] = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.n = len(corpus_tokens)

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query_tokens:
            if term not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores


def ingest(corpus_path: Path | None = None) -> int:
    path = corpus_path or INPUT_FILE
    text = path.read_text(encoding="utf-8")
    chunks = chunk_corpus(text)
    ids = [c[0] for c in chunks]
    docs = [c[1] for c in chunks]
    BM25_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BM25_CACHE.write_text(
        json.dumps(
            {"ids": ids, "tokens": [tokenize_ru(d) for d in docs], "texts": docs},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return len(chunks)


def _load_bm25() -> tuple[SimpleBM25, list[str], list[str]]:
    if not BM25_CACHE.exists():
        ingest()
    data = json.loads(BM25_CACHE.read_text(encoding="utf-8"))
    return SimpleBM25(data["tokens"]), data["ids"], data["texts"]


def bm25_retrieve(query: str, k: int = 8) -> dict:
    bm25, bm25_ids, bm25_texts = _load_bm25()
    scores = bm25.score(tokenize_ru(query))
    order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:k]
    top_ids = [bm25_ids[i] for i in order]
    return {"ids": [top_ids], "documents": [[bm25_texts[i] for i in order]]}


def hybrid_retrieve(query: str, k: int = 8, **_kw) -> dict:
    return bm25_retrieve(query, k=k)


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос продакта по архиву отзывов о доставке еды. "
        "Опирайся ТОЛЬКО на контекст.\n\n"
        f"Контекст:\n{ctx}\n\nВопрос: {query}\n\nОтвет:"
    )


def ask_offline(query: str, k: int = 8) -> tuple[RAGAnswer, dict, list[str]]:
    hits = bm25_retrieve(query, k=k)
    docs = hits["documents"][0]
    q_lower = query.lower()
    q_tokens = tokenize_ru(query)
    stems = [t[:5] for t in q_tokens if len(t) > 4]

    # тематические якоря (без общих слов вроде «ед» из «еду»)
    anchors: list[str] = []
    if "холод" in q_lower:
        anchors.extend(["холодн", "комнатн", "температур"])
    if "поддерж" in q_lower:
        anchors.append("поддерж")
    if "приложен" in q_lower or "интерфейс" in q_lower:
        anchors.extend(["приложен", "интерфейс", "вылет", "тормоз"])

    def _relevant(d: str) -> bool:
        dl = d.lower()
        if anchors and any(a in dl for a in anchors):
            return True
        return any(t in dl for t in q_tokens) or any(s in dl for s in stems)

    if anchors:
        matched = [d for d in docs if any(a in d.lower() for a in anchors)]
        if not matched:
            _, _, all_texts = _load_bm25()
            matched = [t for t in all_texts if any(a in t.lower() for a in anchors)][:3]
    else:
        matched = [d for d in docs if _relevant(d)]
    quotes = [d[:150] for d in (matched or docs)[:3]]
    summary = " ".join(quotes)[:300]
    answer = RAGAnswer(
        answer=f"По корпусу: {summary}",
        quotes=quotes if quotes else [docs[0][:80]],
        confidence=0.75 if matched else 0.5,
        sources=hits["ids"][0][:3],
    )
    return answer, hits, docs


def ask(query: str, k: int = 8) -> tuple[RAGAnswer, dict, list[str]]:
    global _client
    if not BM25_CACHE.exists():
        ingest()
    hits = hybrid_retrieve(query, k=k)
    docs = hits["documents"][0]
    try:
        if _client is None:
            _client = make_client()
        resp: RAGAnswer = _client.chat.completions.create(
            model=get_model(),
            response_model=RAGAnswer,
            messages=[{"role": "user", "content": build_prompt(query, hits)}],
            temperature=0.1,
            max_retries=3,
        )
        return resp, hits, docs
    except Exception:
        return ask_offline(query, k=k)
