from pathlib import Path
from pymupdf4llm import to_markdown
from typing import List, Dict

def load_pdf_document(pdf_path: str, output_path: str) -> List[Dict]:
    """
    Extracts text from PDF and saves it as Markdown.
    Note: This uses Tesseract OCR for scanned PDFs.
    """
    # Extract text, keeping page boundaries and ignoring images to save processing time
    pages = to_markdown(
        pdf_path,
        page_chunks=True,
        write_images=False
    )

    documents = []
    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": pdf_path,
                "page_number": None, # Filled during chunking
                "chunk_index": None
            }
        })

    # Save to markdown file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(documents, start=1):
            f.write(f"## Page {i}\n\n")
            f.write(doc["text"])
            f.write("\n\n")

    return documents
