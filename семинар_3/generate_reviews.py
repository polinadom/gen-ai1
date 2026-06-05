"""Генерация синтетических отзывов на мобильное приложение через LLM."""

import json
import random
from pathlib import Path
from llm_client import make_client, get_model

client = make_client()
MODEL = get_model()

# Шаблоны для разнообразия
AUTHORS = [
    "Анна", "Максим", "Елена", "Дмитрий", "Ольга", "Сергей", "Татьяна", "Алексей",
    "Мария", "Иван", "Наталья", "Владимир", "Юлия", "Андрей", "Светлана", "Павел",
    "Екатерина", "Михаил", "Ирина", "Николай", "Александра", "Виктор", "Кристина", "Роман"
]

PLATFORMS = ["Google Play", "App Store", "RuStore"]

RATINGS = [1, 2, 3, 4, 5]
RATING_WEIGHTS = [0.15, 0.10, 0.15, 0.30, 0.30]  # больше 4 и 5 звёзд

def generate_review(i: int) -> dict:
    """Сгенерировать один отзыв через LLM"""
    
    author = random.choice(AUTHORS)
    platform = random.choice(PLATFORMS)
    rating = random.choices(RATINGS, weights=RATING_WEIGHTS)[0]
    
    prompt = f"""Напиши реалистичный отзыв на мобильное банковское приложение "МойБанк" от пользователя {author}.

Характеристики отзыва:
- Платформа: {platform}
- Оценка: {rating}⭐ (из 5)
- Дата: февраль-март 2026

Требования к отзыву:
1. Длина: 1-3 предложения
2. Упомяни хотя бы одну из проблем: производительность, дизайн, поддержка, цена, реклама, надёжность
3. Если оценка 1-2⭐ — отзыв негативный
4. Если оценка 4-5⭐ — позитивный или нейтральный
5. Можешь упомянуть конкурента (Тинькофф, Сбер, ВТБ, Альфа)

Верни ТОЛЬКО JSON в формате:
{{"text": "текст отзыва"}}

Никаких пояснений, только JSON."""

    try:
        result = client.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=None,
            temperature=0.9
        )
        return {
            "author": author,
            "rating": rating,
            "platform": platform,
            "review_date": f"2026-{random.randint(2,3):02d}-{random.randint(1,28):02d}",
            "text": result.get("text", "").strip()
        }
    except Exception as e:
        print(f"  Ошибка генерации: {e}")
        # Фолбэк - случайный отзыв
        return {
            "author": author,
            "rating": rating,
            "platform": platform,
            "review_date": f"2026-02-{random.randint(1,28):02d}",
            "text": f"{'👍' if rating >= 4 else '👎'} Приложение {'отличное' if rating >= 4 else 'так себе'}. " + (
                "Переводы работают быстро." if rating >= 4 else "Часто вылетает и тормозит."
            )
        }

def main():
    """Генерация 30 отзывов"""
    print(f"🚀 Генерация отзывов через модель: {MODEL}")
    print("="*50)
    
    n_reviews = 30
    reviews = []
    
    for i in range(1, n_reviews + 1):
        print(f"  Генерация отзыва {i}/{n_reviews}...")
        review = generate_review(i)
        reviews.append(review)
    
    # Форматируем как в примере
    output = []
    for i, r in enumerate(reviews, 1):
        output.append(f"=== REVIEW {i} ===")
        output.append(f"Автор: {r['author']}")
        output.append(f"Платформа: {r['platform']}")
        output.append(f"Рейтинг: {r['rating']}")
        output.append(f"Дата: {r['review_date']}")
        output.append(r['text'])
        output.append("")
    
    # Сохраняем
    output_text = "\n".join(output)
    Path("input/reviews.txt").write_text(output_text, encoding="utf-8")
    
    print(f"\n✅ Сгенерировано {n_reviews} отзывов")
    print(f"✅ Сохранено в input/reviews.txt")
    
    # Показываем пример
    print("\n📝 Пример сгенерированного отзыва:")
    print("="*50)
    print(output_text[:500])
    print("...")

if __name__ == "__main__":
    main()