# 🏥 Multi-Document Multilingual RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) system designed to process, index, and answer questions across multiple medical research PDFs. It runs entirely on a local GPU (NVIDIA Quadro P3200, 6GB VRAM) without relying on external APIs, ensuring data privacy and zero inference costs.

## 🚀 Key Features
- **Multi-Document Support:** Indexes and queries across 7 different medical imaging and Diabetic Retinopathy (DR) papers.
- **Multilingual Querying:** Uses `BAAI/bge-m3` embeddings to allow users to ask questions in Spanish, French, Japanese, Tamil, etc., retrieving English context seamlessly.
- **Strict Hallucination Control:** A custom 10-rule prompt system ensures the model answers *only* from the provided context. If the answer isn't there, it explicitly says "I don't know."
- **Hardware Optimized:** Uses 4-bit quantization (`bitsandbytes`) to fit the 4-Billion parameter `Gemma-3-4B-it` model into a 6GB VRAM laptop GPU.

## 🛠️ Engineering Challenges Solved
During development, several critical RAG-specific bugs were identified and fixed:

1. **The "IDx-DR" Missing Content Bug (Junk Filter Fix):** The initial junk filter used `re.search` to drop page numbers, which accidentally deleted 3,000-character content blocks that merely *mentioned* a page number. Fixed by using anchored `re.match`.
2. **Page-Boundary Truncation:** Original chunking split pages independently, cutting sentences in half. Fixed by stripping page markers before splitting.
3. **Cross-Paper Contamination:** Facts from different papers were mixing. Fixed by implementing strict source filtering for targeted queries.
4. **Unicode OCR Artifacts:** Tesseract OCR inserted non-breaking hyphens (`IDx‑DR`) that broke retrieval. Fixed with a custom text normalizer.
5. **VRAM Overload (Speed Optimization):** Query times dropped from 280s to 20s by moving the embedder to CPU, reducing `top_k`, and truncating prompt chunks to 400 characters.

## 🏗️ Architecture
```text
PDFs (Scanned) 
  ↓ pymupdf4llm (OCR extraction to Markdown)
Markdown Text
  ↓ langchain-text-splitters (2000 char chunks + overlap)
Text Chunks
  ↓ BAAI/bge-m3 (1024-dimensional embeddings)
Vector Database
  ↓ FAISS (Similarity Search)
Top-K Chunks
  ↓ Prompt Constructor (10 strict rules)
Gemma-3-4B (4-bit)
  ↓
Answer
