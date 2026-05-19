# rag_pipeline.py
# Day 5: End-to-end RAG. Takes a topic + question, returns a cited answer.

import os
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from paper_pipeline import fetch_and_chunk, load_and_chunk_local

load_dotenv()

# ====================
# CONFIG (centralize all magic numbers here — easy to tune later)
# ====================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # MiniLM output dimension
TOP_K = 5         # how many chunks to retrieve per query
LLM_MODEL = "llama-3.1-8b-instant"


# ====================
# INDEX BUILDING
# ====================
class PaperIndex:
    """
    A FAISS index over paper chunks, with metadata kept in parallel.

    Design choice: FAISS stores only vectors, not text. So we keep the
    chunks list in parallel: chunks[i] corresponds to vector i in the index.
    This is simpler than a full vector DB (Chroma, Pinecone) and adequate
    at our scale.
    """

    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        print(f"Loading embedding model: {embed_model_name}...")
        self.embed_model = SentenceTransformer(embed_model_name)
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks: List[Dict] = []  # parallel list — chunks[i] ↔ vector i

    def build(self, chunk_dicts: List[Dict]):
        """
        Build the index from a list of chunk dicts (output of fetch_and_chunk).
        Each chunk dict has: text, paper_title, paper_url, chunk_id.
        """
        if not chunk_dicts:
            raise ValueError("No chunks to index.")

        print(f"Embedding {len(chunk_dicts)} chunks...")
        texts = [c["text"] for c in chunk_dicts]
        embeddings = self.embed_model.encode(
            texts,
            show_progress_bar=True,  # nice for long ingestion runs
            batch_size=32,
        ).astype("float32")

        self.index.add(embeddings)
        self.chunks = chunk_dicts
        print(f"Index built: {self.index.ntotal} vectors stored.\n")

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """
        Given a query string, return the top-k most similar chunks
        with their similarity scores attached.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty — call build() first.")

        # Embed the query (must be wrapped in a list)
        q_emb = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, k)

        # Build result list with chunk + distance score
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when there aren't enough vectors
                continue
            chunk = dict(self.chunks[idx])  # copy so we don't mutate stored data
            chunk["distance"] = float(dist)
            results.append(chunk)

        return results
    
# ====================
# GENERATION
# ====================
def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Construct a citation-aware prompt.

    Key design choices:
    1. Numbered citations [1], [2], ... map to retrieved chunks.
    2. Instructions are explicit: "use ONLY the context".
    3. We append a legend showing which paper each number refers to,
       so the user can verify.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = f"[{i}] From '{chunk['paper_title']}':\n{chunk['text']}"
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the context below.
Cite sources inline using bracket notation like [1], [2], etc.
If the context does not contain enough information to answer, say "I cannot answer this from the loaded papers."

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Call Groq's Llama 3 with the constructed prompt.
    Returns the answer text (raw, with [1]/[2]/... citations).
    """
    prompt = build_prompt(question, retrieved_chunks)
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1)
    # temperature=0.1 keeps answers more deterministic.
    # For pure retrieval-grounded tasks, low temperature reduces hallucination.

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def format_response(answer: str, retrieved_chunks: List[Dict]) -> str:
    """
    Format the final response: answer + a 'Sources' section
    showing what each [N] refers to.
    """
    sources_lines = ["\n--- Sources ---"]
    for i, chunk in enumerate(retrieved_chunks, start=1):
        sources_lines.append(
            f"[{i}] {chunk['paper_title']}\n"
            f"    URL: {chunk['paper_url']}\n"
            f"    Distance: {chunk['distance']:.3f}"
        )
    return answer + "\n" + "\n".join(sources_lines)


# ====================
# MAIN — interactive CLI
# ====================
if __name__ == "__main__":
    # Mode selector — use local PDFs while ArXiv is rate-limiting.
    mode = input("Mode? [local / arxiv] (default: local): ").strip().lower()
    if not mode:
        mode = "local"

    if mode == "arxiv":
        topic = input("Research topic: ").strip() or "retrieval augmented generation"
        chunks = fetch_and_chunk(topic, max_papers=3)
    else:
        chunks = load_and_chunk_local(folder="local_papers")

    if not chunks:
        print("No chunks were loaded. Exiting.")
        exit(1)

    # Build the index
    index = PaperIndex()
    index.build(chunks)

    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("Index ready. Ask questions about these papers.")
    print("Type 'quit' or 'exit' to leave.")
    print("=" * 60 + "\n")

    while True:
        question = input("\nQ: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if not question:
            continue

        retrieved = index.retrieve(question, k=TOP_K)
        print("\nGenerating answer...\n")
        answer = generate_answer(question, retrieved)
        print(format_response(answer, retrieved))