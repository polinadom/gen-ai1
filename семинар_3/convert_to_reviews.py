"""Конвертация транскриптов в формат отзывов для Multi-doc."""
import re
from pathlib import Path

def convert_transcript_to_reviews(transcript_path: Path) -> str:
    """Превращает диалог в несколько отзывов"""
    text = transcript_path.read_text(encoding="utf-8")
    
    # Извлекаем имена участников
    names = re.findall(r'^([А-ЯЁ][а-яё]+):', text, re.MULTILINE)
    unique_names = list(set(names))[:4]
    
    reviews = []
    review_id = 1
    
    for name in unique_names:
        # Ищем высказывания этого участника
        pattern = rf'{name}:(.*?)(?=\n[А-ЯЁ][а-яё]+:|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            quote = matches[0].strip()[:200]
            rating = "3"  # средняя оценка
            reviews.append(f"""=== REVIEW {review_id} ===
Автор: {name}
Платформа: App Store
Рейтинг: {rating}
Дата: 2026-03-15
Текст: {quote}
""")
            review_id += 1
    
    return "\n".join(reviews)

def main():
    input_dir = Path("starter/transcripts")
    output_dir = Path("input")
    output_dir.mkdir(exist_ok=True)
    
    all_reviews = []
    for txt_file in input_dir.glob("*.txt"):
        print(f"  Конвертация {txt_file.name}...")
        reviews = convert_transcript_to_reviews(txt_file)
        all_reviews.append(reviews)
    
    # Сохраняем все отзывы в один файл
    combined = "\n\n".join(all_reviews)
    output_path = output_dir / "all_banks_reviews.txt"
    output_path.write_text(combined, encoding="utf-8")
    
    print(f"\n✅ Сохранено {output_path}")
    print(f"   Количество отзывов: {combined.count('=== REVIEW')}")

if __name__ == "__main__":
    main()