"""Pydantic-модели для анализа отзывов о сервисах доставки еды."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ServiceName = Literal["Яндекс Еда", "Самокат", "Купер"]
Platform = Literal["App Store", "Google Play", "RuStore"]
IssueCategory = Literal[
    "delivery_speed", "food_quality", "price", "app_ui", "support", "courier"
]
AspectName = Literal[
    "delivery_speed", "food_quality", "price", "app_ui", "support", "courier"
]

RUSSIAN_TO_ENGLISH = {
    "скорость доставки": "delivery_speed",
    "доставка": "delivery_speed",
    "качество еды": "food_quality",
    "еда": "food_quality",
    "цена": "price",
    "цен": "price",
    "интерфейс": "app_ui",
    "приложение": "app_ui",
    "поддержка": "support",
    "курьер": "courier",
    "курьеры": "courier",
}

ALLOWED_CATEGORIES = [
    "delivery_speed",
    "food_quality",
    "price",
    "app_ui",
    "support",
    "courier",
]


class Issue(BaseModel):
    category: str
    severity: int = Field(ge=1, le=5)
    quote: str

    @field_validator("category", mode="before")
    @classmethod
    def convert_category(cls, v: Any) -> str:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in RUSSIAN_TO_ENGLISH:
                return RUSSIAN_TO_ENGLISH[v_lower]
            if v_lower in ALLOWED_CATEGORIES:
                return v_lower
        return "food_quality"

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in ALLOWED_CATEGORIES:
            return "food_quality"
        return v


class Review(BaseModel):
    author: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    service: ServiceName
    platform: Platform
    review_date: Optional[date] = None
    text: str = Field(min_length=10)
    issues: list[Issue] = Field(default_factory=list)
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def date_not_in_future(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v > date.today():
            raise ValueError("дата отзыва не может быть в будущем")
        return v


class AspectSentiment(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)

    @field_validator("aspect", mode="before")
    @classmethod
    def convert_aspect(cls, v: Any) -> str:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in RUSSIAN_TO_ENGLISH:
                return RUSSIAN_TO_ENGLISH[v_lower]
            if v_lower in ALLOWED_CATEGORIES:
                return v_lower
        return "food_quality"

    @field_validator("aspect")
    @classmethod
    def validate_aspect(cls, v: str) -> str:
        if v not in ALLOWED_CATEGORIES:
            return "food_quality"
        return v


class ReviewSentiment(BaseModel):
    author: str
    aspects: list[AspectSentiment] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    author: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class ReviewsSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str


class RAGAnswer(BaseModel):
    answer: str
    quotes: list[str] = Field(min_length=1, max_length=5)
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    answer: str
    value: Optional[float] = None
    unit: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class SubQuestion(BaseModel):
    id: int
    question: str
    expected_tools: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    reasoning: str
    subquestions: list[SubQuestion]


class WorkerAnswer(BaseModel):
    answer: str
    used_tools: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.8)


class Verdict(BaseModel):
    ok: bool
    action: Literal["accept", "rework", "replan"] = "accept"
    reason: str = ""
    rework_ids: list[int] = Field(default_factory=list)
