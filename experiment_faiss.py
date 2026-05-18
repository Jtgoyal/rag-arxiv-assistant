# experiment_faiss.py
# Day 2 — Part 2: Build a tiny FAISS index, do a query.

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Our "knowledge base" — a list of sentences we want to search over.
knowledge_base = [
    "Retrieval augmented generation combines an LLM with a search system over external documents.",
    "FAISS is a library for efficient similarity search of dense vectors, built by Facebook AI.",
    "Embeddings convert text into numerical vectors that capture semantic meaning.",
    "Python is one of the most popular languages for machine learning and data science.",
    "The Streamlit framework lets you build web apps in pure Python without HTML or JavaScript.",
    "Llama 3 is an open-weight large language model released by Meta.",
    "Tokyo is the capital city of Japan and one of the most populous cities in the world.",
    "Cricket is a popular sport in India, England, and Australia.",
]

# Step 2: Embed all knowledge base entries.
print(f"Embedding {len(knowledge_base)} sentences...")
kb_embeddings = model.encode(knowledge_base).astype('float32')
# FAISS requires float32. sentence-transformers returns float32 by default,
# but the .astype('float32') is good defensive practice.

print(f"Embeddings shape: {kb_embeddings.shape}")  # (8, 384)

# Step 3: Build the FAISS index.
# IndexFlatL2: simplest possible — stores all vectors, brute-force searches at query time.
# Good for <100k vectors. For bigger scale you'd use IVF or HNSW (approximate).
dimension = 384  # MiniLM's output dimension
index = faiss.IndexFlatL2(dimension)
index.add(kb_embeddings)
print(f"Index now contains {index.ntotal} vectors.\n")

# Step 4: Query the index.
queries = [
    "What is RAG?",
    "How do I search vectors quickly?",
    "Tell me about cricket.",
    "What is the meaning of life?",  # nothing relevant in KB
]

for q in queries:
    # Embed the query (must be a list, even with one item)
    q_emb = model.encode([q]).astype('float32')

    # Search: get the top 3 most similar vectors
    k = 3
    distances, indices = index.search(q_emb, k)

    print(f"Query: {q}")
    print(f"  Top {k} matches:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"    {rank+1}. (distance={dist:.3f}) {knowledge_base[idx]}")
    print()