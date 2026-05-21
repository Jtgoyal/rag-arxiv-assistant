# evaluation/eval_set.py
# Manual evaluation set for the RAG ArXiv Assistant.
# Each entry includes: question, expected answer keywords, expected paper(s), category.

# Categories:
#   "direct"    — single-paper, single-fact question
#   "compare"   — needs information from multiple papers
#   "out_of_scope" — should trigger refusal

EVAL_QUESTIONS = [
    # ===== DIRECT QUESTIONS (8) =====
    {
        "id": 1,
        "question": "What is RAG?",
        "category": "direct",
        "expected_papers": ["rag_paper"],
        "expected_keywords": ["retrieval", "augmented", "generation", "non-parametric"],
        "notes": "Should retrieve the abstract or intro of the original RAG paper."
    },
    {
        "id": 2,
        "question": "What are the two main components of RAG?",
        "category": "direct",
        "expected_papers": ["rag_paper"],
        "expected_keywords": ["parametric", "non-parametric", "retriever", "generator"],
        "notes": "Looking for the parametric+non-parametric architecture description."
    },
    {
        "id": 3,
        "question": "What is the role of the retriever in RAG?",
        "category": "direct",
        "expected_papers": ["rag_paper"],
        "expected_keywords": ["retrieve", "documents", "context", "knowledge"],
        "notes": "Concept check."
    },
    {
        "id": 4,
        "question": "What is Self-RAG?",
        "category": "direct",
        "expected_papers": ["self_rag"],
        "expected_keywords": ["self-reflection", "reflection tokens", "critique"],
        "notes": "Should retrieve from self_rag paper, not original rag_paper."
    },
    {
        "id": 5,
        "question": "What are reflection tokens?",
        "category": "direct",
        "expected_papers": ["self_rag"],
        "expected_keywords": ["reflection", "token", "critique", "retrieve"],
        "notes": "Specific to Self-RAG."
    },
    {
        "id": 6,
        "question": "What is REALM?",
        "category": "direct",
        "expected_papers": ["realm"],
        "expected_keywords": ["retrieval", "language model", "pre-training", "masked"],
        "notes": "Should retrieve from REALM paper."
    },
    {
        "id": 7,
        "question": "How is REALM pre-trained?",
        "category": "direct",
        "expected_papers": ["realm"],
        "expected_keywords": ["masked language modeling", "retrieve", "salient spans"],
        "notes": "Specific REALM concept."
    },
    {
        "id": 8,
        "question": "What problems does RAG solve?",
        "category": "direct",
        "expected_papers": ["rag_paper"],
        "expected_keywords": ["hallucination", "knowledge", "factual"],
        "notes": "Multi-fact answer; expect multiple cited chunks."
    },

    # ===== COMPARISON QUESTIONS (4) =====
    {
        "id": 9,
        "question": "How does Self-RAG differ from standard RAG?",
        "category": "compare",
        "expected_papers": ["self_rag", "rag_paper"],
        "expected_keywords": ["self-reflection", "critique", "adaptive retrieval"],
        "notes": "Should pull from both papers."
    },
    {
        "id": 10,
        "question": "What is the difference between RAG and REALM?",
        "category": "compare",
        "expected_papers": ["rag_paper", "realm"],
        "expected_keywords": ["pre-training", "fine-tuning", "encoder"],
        "notes": "Architectural comparison."
    },
    {
        "id": 11,
        "question": "Compare the retrieval mechanisms in REALM and Self-RAG.",
        "category": "compare",
        "expected_papers": ["realm", "self_rag"],
        "expected_keywords": ["retriever", "encoder", "adaptive", "reflection"],
        "notes": "Specific comparison across two non-default papers."
    },
    {
        "id": 12,
        "question": "Which paper introduced reflection tokens?",
        "category": "compare",
        "expected_papers": ["self_rag"],
        "expected_keywords": ["Self-RAG", "reflection tokens"],
        "notes": "Disambiguation question."
    },

    # ===== OUT-OF-SCOPE QUESTIONS (5) =====
    {
        "id": 13,
        "question": "What is the capital of France?",
        "category": "out_of_scope",
        "expected_papers": [],
        "expected_keywords": ["cannot answer", "not", "loaded papers"],
        "notes": "Obvious off-topic. Should trigger Layer 1 (distance > threshold)."
    },
    {
        "id": 14,
        "question": "How do I bake a chocolate cake?",
        "category": "out_of_scope",
        "expected_papers": [],
        "expected_keywords": ["cannot answer"],
        "notes": "Obvious off-topic, Layer 1 expected."
    },
    {
        "id": 15,
        "question": "How do I train a neural network from scratch?",
        "category": "out_of_scope",
        "expected_papers": [],
        "expected_keywords": ["cannot answer"],
        "notes": "ML-adjacent but not in papers. May trigger Layer 1 or Layer 2."
    },
    {
        "id": 16,
        "question": "What is the loss function for training a transformer?",
        "category": "out_of_scope",
        "expected_papers": [],
        "expected_keywords": ["cannot answer"],
        "notes": "Specifically off-topic. Tests Layer 2 (since chunks might be vectorally close)."
    },
    {
        "id": 17,
        "question": "Who won the FIFA World Cup in 2022?",
        "category": "out_of_scope",
        "expected_papers": [],
        "expected_keywords": ["cannot answer"],
        "notes": "Totally unrelated; tests Layer 1 firing on obvious off-topic."
    },

    # ===== EDGE / TRICKY QUESTIONS (3) =====
    {
        "id": 18,
        "question": "How many parameters does RAG have?",
        "category": "direct",
        "expected_papers": ["rag_paper"],
        "expected_keywords": ["parameters", "billion", "BART"],
        "notes": "Specific technical detail; tests precise retrieval."
    },
    {
        "id": 19,
        "question": "What datasets are used in the RAG paper experiments?",
        "category": "direct",
        "expected_papers": ["rag_paper"],
        "expected_keywords": ["Natural Questions", "TriviaQA", "Jeopardy"],
        "notes": "Concrete fact extraction."
    },
    {
        "id": 20,
        "question": "Explain the math behind dense retrieval.",
        "category": "direct",
        "expected_papers": ["rag_paper", "realm"],
        "expected_keywords": ["dot product", "embedding", "vector"],
        "notes": "Borderline — may or may not be in retrieved chunks."
    },
]