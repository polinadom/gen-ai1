
import json
import re
from pathlib import Path

import pandas as pd
from llm_client import make_client
from schema import Review, MultiDocSummary

client = make_client()

# Промпт для Multi-doc анализа
MULTI_DOC_SYSTEM = """Ты — аналитик. Перед тобой отзывы о 5 разных банках.

Проанализируй и верни JSON:
{
  "common_themes": ["общая проблема 1", "общая проблема 2"],
  "unique_per_bank": {"bank1": ["уникальная проблема"], "bank2": [...]},
  "overall_headline": "Общий вывод"
}

ОТВЕЧАЙ ТОЛЬКО JSON."""


def extract_simple_reviews(text: str, bank_name: str) -> list[dict]:
    """Извлекает простые отзывы из транскрипта"""
    # Ищем высказывания участников
    names = re.findall(r'^([А-ЯЁ][а-яё]+):', text, re.MULTILINE)
    unique_names = list(set(names))[:4]
    
    reviews = []
    for name in unique_names:
        pattern = rf'{name}:(.*?)(?=\n[А-ЯЁ][а-яё]+:|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            quote = matches[0].strip()[:300]
            # Определяем тональность по ключевым словам
            sentiment = "positive" if any(w in quote.lower() for w in ["нравится", "хорошо", "удобно"]) else "negative"
            reviews.append({
                "bank": bank_name,
                "author": name,
                "text": quote,
                "sentiment": sentiment,
                "rating": 4 if sentiment == "positive" else 2
            })
    return reviews


def analyze_all_banks():
    """Анализирует все 5 банков и делает Multi-doc сравнение"""
    print("\n" + "="*60)
    print("РАУНД 7: Многодокументный анализ (5 банков)")
    print("="*60)
    
    transcripts_dir = Path("starter/transcripts")
    all_reviews = []
    
    # Собираем отзывы со всех банков
    for txt_file in transcripts_dir.glob("*.txt"):
        bank_name = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        reviews = extract_simple_reviews(text, bank_name)
        all_reviews.extend(reviews)
        print(f"  ✅ {bank_name}: {len(reviews)} отзывов")
    
    # Создаём DataFrame для анализа
    df = pd.DataFrame(all_reviews)
    print(f"\n📊 Всего отзывов: {len(df)}")
    
    # Сводная таблица по банкам
    pivot = pd.crosstab(df['bank'], df['sentiment'])
    print("\n📊 Сводка по банкам:")
    print(pivot)
    
    # Формируем текст для LLM
    summary_text = ""
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        summary_text += f"\n=== {bank.upper()} ===\n"
        for _, row in bank_df.head(2).iterrows():
            summary_text += f"- {row['author']}: {row['text'][:100]}...\n"
    
    # Multi-doc анализ через LLM
    print("\n🔄 Запуск Multi-doc анализа через LLM...")
    result = client.create(
        messages=[
            {"role": "system", "content": MULTI_DOC_SYSTEM},
            {"role": "user", "content": f"Проанализируй эти отзывы о 5 банках:\n{summary_text}"}
        ],
        response_model=MultiDocSummary,
        temperature=0.0
    )
    
    # Сохраняем результаты
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Сохраняем Multi-doc результат
    (output_dir / "multi_doc_summary.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    
    # Сохраняем сводную таблицу
    pivot.to_csv(output_dir / "cross_bank_table.csv", encoding="utf-8")
    
    # Выводим результат
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ MULTI-DOC АНАЛИЗА")
    print("="*60)
    print(f"\n🏷️ {result.overall_headline}")
    
    print("\n🌐 Общие проблемы (встречаются у большинства банков):")
    for theme in result.common_themes:
        print(f"   • {theme}")
    
    print("\n🪙 Уникальные проблемы по банкам:")
    for bank, problems in result.unique_per_bank.items():
        print(f"   • {bank}: {', '.join(problems[:2])}")
    
    print("\n✅ Сохранено:")
    print("   - output/multi_doc_summary.json")
    print("   - output/cross_bank_table.csv")
    
    return result


if __name__ == "__main__":
    analyze_all_banks()