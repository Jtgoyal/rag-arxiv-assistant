# experiment_embeddings.py
# Day 2 experiment: see embeddings in action.
# Goal: prove to ourselves that semantic similarity works.

from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Load the embedding model.
# all-MiniLM-L6-v2: 22M parameters, ~80MB, outputs 384-dim vectors.
# First time you run this, it downloads the model — be patient (~30s).
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

# Step 2: Pick sentences that test semantic understanding.
# We'll deliberately use synonyms and unrelated topics to test the model.
sentences = [
    "How does retrieval augmented generation work?",
    "Can you explain retrieval augmented generation?",       # paraphrase — should be HIGH
    "What is semantic search using embeddings?",             # related concept
    "I want to bake a cake.",                                # unrelated
    "The price of bitcoin is dropping.",                     # unrelated
]

# Step 3: Convert sentences to vectors.
# .encode() returns a numpy array of shape (n_sentences, 384).
embeddings = model.encode(sentences)
print(f"Shape of embeddings: {embeddings.shape}")
print(f"First embedding (first 5 dims only): {embeddings[0][:5]}")
print(f"Magnitude (length) of first embedding: {np.linalg.norm(embeddings[0]):.4f}\n")

# Step 4: Define cosine similarity.
# cosine(a, b) = (a · b) / (|a| × |b|)
# For normalized vectors (|a|=|b|=1), this is just a · b.
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Step 5: Compare sentence 0 to all others.
print(f"Comparing '{sentences[0]}' to others:\n")
for i, s in enumerate(sentences):
    sim = cos_sim(embeddings[0], embeddings[i])
    print(f"  Sim to '{s}': {sim:.3f}")