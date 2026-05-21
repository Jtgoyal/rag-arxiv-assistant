# evaluation/run_eval.py
# Runs the evaluation set against the RAG pipeline and outputs metrics.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from paper_pipeline import load_and_chunk_local
from rag_pipeline import PaperIndex, generate_answer_with_guard, validate_and_clean_citations
from eval_set import EVAL_QUESTIONS


def keyword_match(text: str, keywords: list) -> tuple[int, list]:
    """
    Returns (matches_found, matched_keywords).
    Case-insensitive substring match.
    """
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    return len(matched), matched


def evaluate_question(q: dict, index: PaperIndex) -> dict:
    """Run one question and collect metrics."""
    retrieved = index.retrieve(q["question"], k=5)
    answer, refusal_reason = generate_answer_with_guard(q["question"], retrieved)

    # Citation validation (only for non-refusal answers)
    if refusal_reason == "answered":
        cleaned_answer, cited_indices = validate_and_clean_citations(answer, retrieved)
    else:
        cleaned_answer = answer
        cited_indices = []

    # Retrieval accuracy: did expected paper appear in top-5?
    expected_papers = set(q["expected_papers"])
    retrieved_papers = {chunk["paper_title"] for chunk in retrieved}
    if not expected_papers:
        retrieval_correct = (refusal_reason != "answered")  # for out_of_scope, "correct" means refused
    else:
        retrieval_correct = bool(expected_papers & retrieved_papers)

    # Keyword match on answer
    keyword_count, matched_kws = keyword_match(cleaned_answer, q["expected_keywords"])
    keyword_recall = keyword_count / max(len(q["expected_keywords"]), 1)

    # Refusal correctness
    if q["category"] == "out_of_scope":
        # Should have refused
        refusal_correct = (refusal_reason != "answered")
    else:
        # Should have answered
        refusal_correct = (refusal_reason == "answered")

    # Top distance for context
    top_distance = retrieved[0]["distance"] if retrieved else float("inf")

    return {
        "id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "refusal_reason": refusal_reason,
        "top_distance": round(top_distance, 3),
        "retrieved_papers": list(retrieved_papers),
        "expected_papers": list(expected_papers),
        "retrieval_correct": retrieval_correct,
        "keyword_recall": round(keyword_recall, 2),
        "matched_keywords": matched_kws,
        "refusal_correct": refusal_correct,
        "cited_indices": cited_indices,
        "answer_preview": cleaned_answer[:200] + ("..." if len(cleaned_answer) > 200 else ""),
    }


def main():
    print("Loading papers...")
    chunks = load_and_chunk_local(folder="sample_papers")
    print(f"Loaded {len(chunks)} chunks.\n")

    print("Building index...")
    index = PaperIndex()
    index.build(chunks)
    print()

    print(f"Running {len(EVAL_QUESTIONS)} eval questions...\n")
    results = []
    for q in EVAL_QUESTIONS:
        print(f"  [{q['id']:2d}] {q['category']:12s} | {q['question'][:60]}...")
        result = evaluate_question(q, index)
        results.append(result)

    # ==== Compute aggregate metrics ====
    n = len(results)
    direct = [r for r in results if r["category"] == "direct"]
    compare = [r for r in results if r["category"] == "compare"]
    oos = [r for r in results if r["category"] == "out_of_scope"]

    retrieval_acc = sum(r["retrieval_correct"] for r in results) / n
    refusal_acc = sum(r["refusal_correct"] for r in results) / n
    avg_keyword_recall_answered = (
        sum(r["keyword_recall"] for r in results if r["category"] != "out_of_scope")
        / max(len([r for r in results if r["category"] != "out_of_scope"]), 1)
    )

    oos_refusal_rate = sum(r["refusal_correct"] for r in oos) / max(len(oos), 1)

    # ==== Print summary ====
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:              {n}")
    print(f"  Direct:                     {len(direct)}")
    print(f"  Compare:                    {len(compare)}")
    print(f"  Out-of-scope:               {len(oos)}")
    print()
    print(f"Retrieval accuracy (top-5):   {retrieval_acc:.0%}  ({sum(r['retrieval_correct'] for r in results)}/{n})")
    print(f"Refusal correctness:          {refusal_acc:.0%}  ({sum(r['refusal_correct'] for r in results)}/{n})")
    print(f"Keyword recall (answered):    {avg_keyword_recall_answered:.0%}")
    print(f"Out-of-scope refusal rate:    {oos_refusal_rate:.0%}  ({sum(r['refusal_correct'] for r in oos)}/{len(oos)})")
    print()

    # Save raw results
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()