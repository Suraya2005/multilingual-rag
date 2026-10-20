import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def normalize_text(text: str) -> str:
    """Fixes special Unicode characters (like non-breaking hyphens) that break matching."""
    replacements = {
        "\u2011": "-", "\u2010": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def is_reference_chunk(text: str) -> bool:
    """Detects bibliography chunks by checking if >40% of lines start with citations."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return False

    citation_starts = 0
    for line in lines:
        # Matches patterns like "Author et al (2023)" or "[1]"
        if re.match(r'^[-*]?\s*[A-Z][a-zA-Z]+.*\(\d{4}[a-z]?\)', line) or re.match(r'^\[\d+\]', line):
            citation_starts += 1

    return citation_starts / len(lines) > 0.4

def is_junk_chunk(text: str) -> bool:
    """Detects headers, footers, and short chunks."""
    text_clean = text.strip()

    if len(text_clean) < 100:
        return True

    # FIXED: Uses re.match with anchors to ONLY filter if the ENTIRE chunk is a page marker.
    # Old code used re.search, which wrongly deleted content that mentioned a page number.
    if re.match(r'^#*\s*Page \d+ of \d+\s*$', text_clean):
        return True

    if re.match(r'^\s*(Fig\.|Figure|Table)\s+\d+[:\s]', text_clean) and len(text_clean) < 200:
        return True

    if text_clean.startswith("**Keywords**") or text_clean.startswith("Keywords:"):
        return True

    return False

def chunk_text(md_path: str) -> list:
    """Splits text into chunks, filtering out junk and references."""
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 1. Normalize Unicode (fixes IDx-DR hyphens)
    markdown_text = normalize_text(markdown_text)
    
    # 2. Remove page markers so chunks span across pages
    full_text = re.sub(r"\n## Page \d+\n", "\n\n", markdown_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        is_separator_regex=False,
        length_function=len,
    )

    raw_chunks = splitter.split_text(full_text)

    results = []
    for i, chunk in enumerate(raw_chunks):
        if is_reference_chunk(chunk) or is_junk_chunk(chunk):
            continue
            
        results.append({
            "text": chunk,
            "metadata": {
                "source": md_path,
                "page_number": None,
                "chunk_index": i,
            }
        })
    return results