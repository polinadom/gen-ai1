"""Pydantic-модели для анализа отзывов на мобильное приложение."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator

IssueCategory = Literal["performance", "design", "support", "price", "ads", "reliability"]
ReviewAspect = Literal["performance", "design", "support", "price", "ads", "reliability"]
Platform = Literal["App Store", "Google Play", "RuStore"]

# Маппинг русских категорий в английские
RUSSIAN_TO_ENGLISH = {
    "производительность": "performance",
    "дизайн": "design", 
    "поддержка": "support",
    "цена": "price",
    "реклама": "ads",
    "надёжность": "reliability",
    "надежность": "reliability"
}


class Issue(BaseModel):
    category: str  # временно str, потом конвертируем
    severity: int = Field(ge=1, le=5)
    quote: str
    
    @field_validator("category", mode="before")
    @classmethod
    def convert_category(cls, v: Any) -> str:
        """Конвертирует русские категории в английские"""
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in RUSSIAN_TO_ENGLISH:
                return RUSSIAN_TO_ENGLISH[v_lower]
            if v_lower in ["performance", "design", "support", "price", "ads", "reliability"]:
                return v_lower
        return "performance"  # значение по умолчанию
    
    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Проверяет, что категория допустима"""
        allowed = ["performance", "design", "support", "price", "ads", "reliability"]
        if v not in allowed:
            return "performance"  # значение по умолчанию
        return v


class Review(BaseModel):
    author: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
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
    aspect: str  # временно str, потом конвертируем
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)
    
    @field_validator("aspect", mode="before")
    @classmethod
    def convert_aspect(cls, v: Any) -> str:
        """Конвертирует русские названия аспектов в английские"""
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower in RUSSIAN_TO_ENGLISH:
                return RUSSIAN_TO_ENGLISH[v_lower]
            if v_lower in ["performance", "design", "support", "price", "ads", "reliability"]:
                return v_lower
        return "performance"
    
    @field_validator("aspect")
    @classmethod
    def validate_aspect(cls, v: str) -> str:
        allowed = ["performance", "design", "support", "price", "ads", "reliability"]
        if v not in allowed:
            return "performance"
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

class MultiDocSummary(BaseModel):
    """Сравнение нескольких банков"""
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_bank: dict[str, list[str]]
    overall_headline: str