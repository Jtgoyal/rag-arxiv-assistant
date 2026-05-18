# paper_pipeline.py
# Day 4: Combine ArXiv fetching + PDF chunking into one pipeline.
# Output: list of dicts with chunk text + source metadata.

from fetch_arxiv import fetch_papers, download_paper
from pdf_to_chunks import load_pdf, chunk_text


def fetch_and_chunk(topic: str, max_papers: int = 3, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Full pipeline: fetch papers from ArXiv, extract text, chunk each.

    Returns a list of dicts:
    [
        {
            "text": "...chunk content...",
            "paper_title": "Paper Title",
            "paper_url": "http://arxiv.org/abs/...",
            "chunk_id": "papertitle__chunk_0"
        },
        ...
    ]

    Each chunk knows which paper it came from — critical for citation in RAG.
    """
    print(f"Fetching {max_papers} papers on '{topic}'...")
    papers = fetch_papers(topic, max_results=max_papers)
    print(f"Found {len(papers)} papers.\n")

    all_chunks = []

    for i, paper in enumerate(papers, 1):
        print(f"[{i}/{len(papers)}] Processing: {paper.title[:80]}...")

        # Download the PDF
        try:
            pdf_path = download_paper(paper)
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            continue

        # Extract text
        try:
            text = load_pdf(pdf_path)
        except Exception as e:
            print(f"  ❌ PDF parse failed: {e}")
            continue

        # Chunk
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"  ✓ {len(chunks)} chunks created")

        # Attach metadata to each chunk
        for j, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "paper_title": paper.title,
                "paper_url": paper.entry_id,
                "chunk_id": f"{paper.entry_id.split('/')[-1]}__chunk_{j}",
            })

    print(f"\nTotal chunks across all papers: {len(all_chunks)}")
    return all_chunks


# ====================
# MAIN — test the pipeline
# ====================
if __name__ == "__main__":
    chunks = fetch_and_chunk("retrieval augmented generation", max_papers=2)

    # Show a sample chunk
    if chunks:
        print("\n--- Sample chunk (index 5) ---")
        sample = chunks[5] if len(chunks) > 5 else chunks[0]
        print(f"From paper: {sample['paper_title']}")
        print(f"URL: {sample['paper_url']}")
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"\nText:\n{sample['text']}")