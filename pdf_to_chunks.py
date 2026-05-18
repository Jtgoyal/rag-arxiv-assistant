# pdf_to_chunks.py
# Day 3 deliverable: load a PDF, extract text, split into chunks.

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path: str) -> str:
    """
    Load a PDF and return its full text content as a single string.

    PyMuPDF (imported as `fitz`) is fast and handles complex layouts
    like multi-column papers and embedded equations better than
    alternatives like PyPDF2.
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        # get_text() extracts text in reading order.
        # Other modes available: "blocks", "html", "dict" — we just need plain text.
        text += page.get_text()
    doc.close()  # always close PDF handles
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks suitable for embedding.

    RecursiveCharacterTextSplitter is smarter than naive splitting:
    it tries to split on natural boundaries (paragraphs, sentences, words)
    BEFORE falling back to mid-word splits. The 'separators' list defines
    the priority order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on these, in priority order:
        # 1. Double newline (paragraph boundary) — best split point
        # 2. Single newline (line boundary)
        # 3. Period + space (sentence boundary)
        # 4. Single space (word boundary)
        # 5. Empty string (character-level, last resort)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,  # measure chunks in characters (simple). Could use a tokenizer for token-accurate sizing.
    )
    chunks = splitter.split_text(text)
    return chunks


# ====================
# MAIN — for testing
# ====================
if __name__ == "__main__":
    # Load
    print("Loading PDF...")
    text = load_pdf("test_paper.pdf")
    print(f"Extracted {len(text):,} characters from PDF.\n")

    # Show a preview
    print("--- First 500 chars ---")
    print(text[:500])
    print("--- End preview ---\n")

    # Chunk
    print("Chunking...")
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print(f"Created {len(chunks)} chunks.\n")

    # Inspect chunks
    print(f"Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
    print(f"Min chunk length: {min(len(c) for c in chunks)} chars")
    print(f"Max chunk length: {max(len(c) for c in chunks)} chars\n")

    # Show first 2 chunks to verify overlap
    print("--- Chunk 0 (last 150 chars) ---")
    print("..." + chunks[0][-150:])
    print()
    print("--- Chunk 1 (first 150 chars) ---")
    print(chunks[1][:150] + "...")
    print()
    print("Notice: the end of chunk 0 should partially overlap with the start of chunk 1.")


    # ====================
    # EXPERIMENT: try different chunk sizes
    # ====================
    print("\n=== Chunk Size Experiment ===\n")
    for size in [300, 500, 1000, 2000]:
        chunks_exp = chunk_text(text, chunk_size=size, chunk_overlap=int(size * 0.2))
        avg_len = sum(len(c) for c in chunks_exp) / len(chunks_exp)
        print(f"Size {size:5d}: {len(chunks_exp):4d} chunks, avg {avg_len:6.0f} chars")