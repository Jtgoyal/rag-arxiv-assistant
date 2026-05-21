# Failure Analysis — Day 11 Evaluation Run

Evaluation set: 20 questions across direct, compare, and out-of-scope categories.
Aggregate metrics: 95% retrieval, 85% refusal correctness, 80% OOS refusal, 132 invalid citations dropped.

3 of 20 questions failed. Pattern: two over-refusals, one hallucination slip-through.

---

## Q11 (compare): "Compare the retrieval mechanisms in REALM and Self-RAG"

**Result:** Retrieval correct (right papers in top-5) but LLM refused (over-refusal).

**Why it failed:**
- Each paper contains its own retrieval discussion, but no chunk discusses both methods comparatively.
- Layer 2 (the LLM) saw chunks describing REALM retrieval and Self-RAG retrieval *separately*, and judged the comparison context insufficient.

**Failure category:** Over-refusal on compound/comparison questions.

**Planned fix:** Multi-query retrieval — split compound questions into sub-queries with an LLM ("What is REALM's retrieval mechanism?" + "What is Self-RAG's retrieval mechanism?"), retrieve for each, merge results, then generate.

---

## Q16 (out_of_scope): "What is the loss function for training a transformer?"

**Result:** OOS question that should have been refused, but the LLM produced an answer (hallucination slip-through).

**Why it failed:**
- Papers contain transformer-related vocabulary (encoder, training, attention) → top retrieval distances were under the 1.5 threshold, so Layer 1 didn't refuse.
- Layer 2 (LLM) saw chunks with ML jargon and reached to construct an answer rather than refusing.
- This is the dangerous failure mode — a real hallucination, not just an answer/refusal mismatch.

**Failure category:** Hallucination on ML-adjacent OOS queries.

**Planned fix:**
- Tighter Layer 2 prompt that emphasizes refusal over partial answers.
- Or: introduce a reranker (cross-encoder) that scores topical relevance after retrieval — chunks that are vectorally close but topically off get filtered before LLM.

---

## Q20 (direct): "Explain the math behind dense retrieval"

**Result:** Retrieval correct, LLM refused (over-refusal again).

**Why it failed:**
- Dense retrieval math (dot product, embedding similarity) is described in papers but spread across multiple chunks and notation-heavy passages.
- The LLM judged any single chunk insufficient to explain the math, so it refused.

**Failure category:** Same as Q11 — concept exists in papers but not packaged into a single self-contained chunk.

**Planned fix:** Same as Q11 — multi-query retrieval, or larger retrieval k for math/explanation questions.

---

## Aggregate insight: the precision/recall tension

Two over-refusals + one hallucination = the two ends of the same precision/recall dial.

| Setting | Over-refusal | Hallucination | When to use |
|---|---|---|---|
| Tighter guards (lower distance threshold, stricter refusal prompt) | More | Less | High-stakes domains: medical, legal |
| Looser guards (higher threshold, lenient prompt) | Less | More | Exploratory research / brainstorming |

My current calibration leans toward over-refusal — appropriate for a research assistant where hallucination is worse than "I don't know." If I were deploying for content generation, I'd loosen.

---

## What the eval ALSO revealed (the citation hallucinations)

Beyond the 3 question-level failures: the citation validator dropped **132 invalid `[N]` references** across the 20 questions. The LLM frequently hallucinated citation numbers like `[6]`, `[7]`, `[8]` when only 5 chunks were retrieved.

Without the Day 8 validator, users would have seen these fabricated citations. This is the kind of failure that's invisible without measurement — it's why eval matters.