# fetch_arxiv.py
# Day 4: Fetch papers from ArXiv by topic, download PDFs, extract metadata.

import arxiv
import os
import time
from urllib.error import HTTPError

def fetch_papers(topic: str, max_results: int = 5):
    """
    Fetch papers from ArXiv matching a topic query.

    Returns a list of arxiv.Result objects. Each has:
    - .title (str)
    - .summary (str) — the abstract
    - .authors (list of arxiv.Author)
    - .published (datetime)
    - .pdf_url (str) — direct PDF download link
    - .entry_id (str) — ArXiv URL like 'http://arxiv.org/abs/2005.11401v4'
    """
    # Build the search query.
    # SortCriterion.SubmittedDate = newest first.
    # SortCriterion.Relevance = best matches first (default).
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # arxiv.Client is the recommended way to fetch.
    # The older `search.results()` is deprecated.
    client = arxiv.Client(
    page_size=10,
    delay_seconds=3,
    num_retries=5,
    )
    results = list(client.results(search))
    return results



def download_paper(paper, output_dir: str = "papers", max_retries: int = 3) -> str:
    """
    Download a paper's PDF to local disk, with retry on rate-limit.
    Returns the path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)

    arxiv_id = paper.entry_id.split("/")[-1]
    filename = f"{arxiv_id}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Skip if cached
    if os.path.exists(filepath):
        print(f"  Already cached: {filename}")
        return filepath

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            paper.download_pdf(dirpath=output_dir, filename=filename)
            time.sleep(3)  # be polite to ArXiv between successful downloads
            return filepath
        except HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 10  # 10s, 20s, 30s
                print(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise  # other HTTP errors → fail fast

    raise RuntimeError(f"Failed to download {arxiv_id} after {max_retries} retries")


# ====================
# MAIN — test it
# ====================
if __name__ == "__main__":
    topic = "retrieval augmented generation"
    print(f"Searching ArXiv for: '{topic}'\n")

    papers = fetch_papers(topic, max_results=3)
    print(f"Found {len(papers)} papers.\n")

    for i, p in enumerate(papers, 1):
        print(f"--- Paper {i} ---")
        print(f"Title: {p.title}")
        print(f"Authors: {', '.join(a.name for a in p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"Published: {p.published.strftime('%Y-%m-%d')}")
        print(f"URL: {p.entry_id}")
        print(f"Abstract (first 200 chars): {p.summary[:200]}...")

        # Download the PDF
        print("Downloading PDF...")
        path = download_paper(p)
        print(f"Saved to: {path}\n")