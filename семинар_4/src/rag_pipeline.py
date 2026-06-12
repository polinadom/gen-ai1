import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 1. ЗАГРУЗКА ТЕКСТОВ
# ============================================================================

TEXTS_DIR = "starter/data"
documents = []
doc_names = []

my_files = ['1', '2', '3', '4', '5', '6', '7']

for filename in sorted(os.listdir(TEXTS_DIR)):
    name, ext = os.path.splitext(filename)
    # Загружаем только те файлы, которые в списке my_files
    if name in my_files and ext.lower() in ['.txt', '.md']:
        with open(os.path.join(TEXTS_DIR, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            documents.append(text)
            doc_names.append(f"doc_{name}")  # будет doc_1, doc_2...
        print(f"   Загружен: doc_{name}")

print(f"\n✅ Загружено {len(documents)} документов")
print(f"   Документы: {doc_names}")

# ============================================================================
# 2. ЧАНКИНГ
# ============================================================================

def chunk_fixed_size(text, chunk_size=2000, overlap=0):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def chunk_recursive(text, chunk_size=400, overlap=80):
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)
    except ImportError:
        print("⚠️ langchain не установлен, используем fixed-size")
        return chunk_fixed_size(text, chunk_size, overlap)

# Создаём чанки
chunks_fixed = []
chunks_recursive = []
chunk_to_doc_fixed = []
chunk_to_doc_recursive = []

for doc_idx, doc in enumerate(documents):
    fixed_chunks = chunk_fixed_size(doc)
    for chunk in fixed_chunks:
        chunks_fixed.append(chunk)
        chunk_to_doc_fixed.append(doc_idx)
    
    rec_chunks = chunk_recursive(doc)
    for chunk in rec_chunks:
        chunks_recursive.append(chunk)
        chunk_to_doc_recursive.append(doc_idx)

print(f"\n📊 Стратегия A (fixed-size): {len(chunks_fixed)} чанков")
print(f"📊 Стратегия B (recursive): {len(chunks_recursive)} чанков")

# ============================================================================
# 3. ИНДЕКСАЦИЯ
# ============================================================================

print("\n🔍 Загрузка модели эмбеддингов...")
model = SentenceTransformer('intfloat/multilingual-e5-small')

print("🔍 Индексация fixed-size...")
embeddings_fixed = model.encode(chunks_fixed, show_progress_bar=True)

print("🔍 Индексация recursive...")
embeddings_recursive = model.encode(chunks_recursive, show_progress_bar=True)

# ============================================================================
# 4. ПОИСК
# ============================================================================

def retrieve(query, embeddings, chunks, chunk_to_doc, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx][:200] + "...",
            'doc_idx': chunk_to_doc[idx],
            'score': float(similarities[idx])
        })
    return results

# ============================================================================
# 5. EVAL
# ============================================================================

def evaluate(gold_path, embeddings, chunks, chunk_to_doc):
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    hits = 0
    details = []
    
    for item in gold_data:
        question = item['question']
        gold_docs = set(item['gold_sources'])
        
        results = retrieve(question, embeddings, chunks, chunk_to_doc)
        retrieved_docs = set([f"doc_{r['doc_idx']+1}" for r in results])
        
        is_hit = len(gold_docs & retrieved_docs) > 0
        if is_hit:
            hits += 1
        
        details.append({
            'id': item['id'],
            'question': question,
            'gold': list(gold_docs),
            'retrieved': list(retrieved_docs),
            'hit': is_hit
        })
    
    return hits / len(gold_data) if gold_data else 0, details

GOLD_PATH = "gold/gold.json"

if os.path.exists(GOLD_PATH):
    hit_rate_fixed, details_fixed = evaluate(GOLD_PATH, embeddings_fixed, chunks_fixed, chunk_to_doc_fixed)
    hit_rate_recursive, details_recursive = evaluate(GOLD_PATH, embeddings_recursive, chunks_recursive, chunk_to_doc_recursive)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ EVALUATION (hit-rate@5)")
    print("="*60)
    print(f"Стратегия A (fixed-size):   {hit_rate_fixed:.2%}")
    print(f"Стратегия B (recursive):    {hit_rate_recursive:.2%}")
    
    # Сохраняем результаты
    with open('results/evaluation.json', 'w', encoding='utf-8') as f:
        json.dump({
            'fixed_size': {'hit_rate': hit_rate_fixed, 'details': details_fixed},
            'recursive': {'hit_rate': hit_rate_recursive, 'details': details_recursive}
        }, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Результаты сохранены в results/evaluation.json")
else:
    print(f"\n⚠️ Файл {GOLD_PATH} не найден!")