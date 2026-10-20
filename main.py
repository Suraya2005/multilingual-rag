import time
from pathlib import Path
from langdetect import detect, LangDetectException

# Import your custom modules
from src.document_loader import load_pdf_document
from src.chunking import chunk_text
from src.embedding import Embedder
from src.vector_store import FaissVectorStore
from src.prompt_constructor import construct_prompt
from src.lm import LM

# ==========================================
# 1. SETUP & INDEXING
# ==========================================
print("--- Starting Pipeline ---")

# NOTE: If VRAM is tight, pass force_cpu=True to the Embedder
embedder = Embedder(force_cpu=False) 
lm = LM()

md_files = sorted(Path("data").glob("*.md"))
all_chunks = []

for md_path in md_files:
    if md_path.name.endswith("_qwen.md"):
        continue
    chunks = chunk_text(str(md_path))
    for c in chunks:
        c["metadata"]["source_file"] = md_path.name.replace(".md", ".pdf")
    all_chunks.extend(chunks)

print(f"📚 Total chunks: {len(all_chunks)}")
print(f"🔍 Chunks with 'idx': {sum(1 for c in all_chunks if 'idx' in c['text'].lower())}")

embedded_chunks = embedder.embed_chunks(all_chunks)
embeddings = [c["embedding"] for c in embedded_chunks]

store = FaissVectorStore(len(embeddings[0]))
store.add(embeddings)
print("✅ FAISS index built.\n")

# ==========================================
# 2. THE QUERY FUNCTION
# ==========================================
LANG_MAP = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "zh-cn": "Chinese", "zh-tw": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "ru": "Russian", "ta": "Tamil",
}

def detect_language(text):
    tl = text.lower()
    if any(w in tl for w in ["¿qué", "¿cómo", "qué es"]): return "Spanish"
    if any(w in tl for w in ["qu'est", "comment", "quel"]): return "French"
    if any(w in tl for w in ["was ist", "wie"]): return "German"
    if any('\u4e00' <= c <= '\u9fff' for c in text): return "Chinese"
    if any('\u3040' <= c <= '\u30ff' for c in text): return "Japanese"
    if any('\u0600' <= c <= '\u06ff' for c in text): return "Arabic"
    if any('\u0900' <= c <= '\u097f' for c in text): return "Hindi"
    if any('\u0b80' <= c <= '\u0bff' for c in text): return "Tamil"
    if any('\u0400' <= c <= '\u04ff' for c in text): return "Russian"
    if any(w in tl for w in ["what", "how", "which", "list"]): return "English"
    try:
        return LANG_MAP.get(detect(text), "English")
    except LangDetectException:
        return "English"

def ask(query: str, top_k: int = 2, max_tokens: int = 100):
    """Asks a question. Automatically filters to paper7 for DR queries."""
    
    target_lang = detect_language(query)
    qe = embedder.embed_query(query)
    raw = store.search(qe, top_k=20)

    # If the question is about Diabetic Retinopathy, filter to paper7
    dr_keywords = ['retinopathy', 'dr', 'idx', 'eyeart', 'lesion', 'cnn', 'retina', 'screening', 'fda']
    is_dr = any(kw in query.lower() for kw in dr_keywords)

    if is_dr:
        rc = [(i, s) for i, s in raw if embedded_chunks[i]['metadata'].get('source_file') == 'paper7.pdf'][:top_k]
    else:
        rc = raw[:top_k]

    p = construct_prompt(query, rc, embedded_chunks)
    
    start = time.time()
    answer = lm.generate_text(p, desired_output_tokens=max_tokens)
    
    # Optional: Translate output if target_lang != English
    if target_lang != "English":
        translate_p = f"Translate to {target_lang}. Output only the translation.\n\n{answer}"
        answer = lm.generate_text(translate_p, desired_output_tokens=max_tokens)
        
    elapsed = time.time() - start
    sources = sorted(set(embedded_chunks[i]['metadata'].get('source_file') for i, _ in rc))
    
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"Language: {target_lang} | Sources: {sources}")
    print(f"⏱️  {elapsed:.2f}s")
    print(f"A: {answer.strip()}")
    return answer

# ==========================================
# 3. RUN TESTS
# ==========================================
ask("What is IDx-DR?")
ask("What is diabetic retinopathy?")
ask("¿Qué es la retinopatía diabética?")
ask("What is the capital of France?")