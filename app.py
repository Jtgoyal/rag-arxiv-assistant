# app.py
# Day 6: Streamlit UI for the RAG pipeline.
import logging
import warnings
import re

# Silence the noisy transformers/sentence-transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from paper_pipeline import fetch_and_chunk, load_and_chunk_local
from rag_pipeline import PaperIndex, generate_answer_with_guard, TOP_K, DISTANCE_THRESHOLD, validate_and_clean_citations

# ====================
# PAGE CONFIG
# ====================
st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ArXiv Research Assistant")
st.caption("Ask questions about research papers and get cited answers.")


# ====================
# SESSION STATE INIT
# ====================
# Session state persists across reruns. This is how we keep the FAISS index
# alive without rebuilding it every time the user types a question.
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "papers_info" not in st.session_state:
    st.session_state.papers_info = None


# ====================
# SIDEBAR — paper loading
# ====================
with st.sidebar:
    st.header("Load Papers")

    mode = st.radio(
        "Source:",
        ["Local PDFs", "ArXiv (live)"],
        help="Local mode uses pre-downloaded PDFs (fast, reliable). ArXiv mode fetches the latest papers (may be rate-limited).",
    )

    if mode == "ArXiv (live)":
        topic = st.text_input("Topic:", value="retrieval augmented generation")
        max_papers = st.slider("Number of papers:", 1, 5, 3)
    else:
        st.info("Loading PDFs from `sample_papers/` folder.")
        topic = None
        max_papers = None

    load_button = st.button("Load Papers", type="primary")

    if load_button:
        # Pre-validate input before doing anything expensive
        if mode == "ArXiv (live)":
            topic_clean = (topic or "").strip()
            if not topic_clean:
                st.error("⚠️ Please enter a research topic before fetching from ArXiv.")
                st.stop()
            if len(topic_clean) < 3:
                st.error("⚠️ Topic is too short. Please enter at least 3 characters.")
                st.stop()

        with st.spinner("Loading and processing papers..."):
            try:
                if mode == "ArXiv (live)":
                    chunks = fetch_and_chunk(topic, max_papers=max_papers)
                    if not chunks:
                        st.warning(
                            f"No papers found on ArXiv for topic '{topic}'. "
                            "Try a different topic, or switch to **Local PDFs** for the pre-loaded demo papers."
                        )
                        st.stop()
                else:
                    chunks = load_and_chunk_local(folder="sample_papers")
                    if not chunks:
                        st.error(
                            "Couldn't load any papers from `sample_papers/`. "
                            "Make sure the folder exists and contains PDF files."
                        )
                        st.stop()
            except Exception as e:
                st.error(
                    f"Couldn't fetch papers from ArXiv. This is usually a rate-limit or "
                    f"transient network issue. Try again in a few minutes, or switch to "
                    f"**Local PDFs** mode for the pre-loaded demo papers.\n\n"
                    f"_Details: `{type(e).__name__}`_"
                )
                st.stop()

            # Build the FAISS index
            index = PaperIndex()
            index.build(chunks)

            st.session_state.index = index
            st.session_state.chunks = chunks

            unique_papers = {}
            for c in chunks:
                title = c["paper_title"]
                if title not in unique_papers:
                    unique_papers[title] = c["paper_url"]
            st.session_state.papers_info = unique_papers

            st.success(f"Loaded {len(unique_papers)} papers, {len(chunks)} chunks indexed.")

    # Show what's loaded
    if st.session_state.papers_info:
        st.divider()
        st.subheader("Loaded Papers")
        for title, url in st.session_state.papers_info.items():
            st.markdown(f"- **{title}**")

        if st.session_state.chunks:
            st.caption(f"📊 {len(st.session_state.chunks)} chunks indexed across {len(st.session_state.papers_info)} papers.")


# ====================
# MAIN AREA — Q&A
# ====================
if st.session_state.index is None:
    st.info("👈 Load papers from the sidebar to start asking questions.")
else:
    question = st.text_input(
        "Ask a question about the loaded papers:",
        placeholder="e.g., What is retrieval-augmented generation?",
    )

    # K-slider (advanced control)
    with st.expander("⚙️ Retrieval settings"):
        k = st.slider("Number of chunks to retrieve (top-k):", 1, 10, TOP_K)

    if question:
        stripped = question.strip()

        if len(stripped) < 5:
            st.info("Please enter a more complete question (at least a few words).")
        elif len(stripped) > 1000:
            st.warning(
                "Question is unusually long — more focused questions give cleaner, "
                "more reliable answers. Try breaking it into 2-3 separate questions."
            )
        else:
            with st.spinner("Retrieving and generating..."):
                retrieved = st.session_state.index.retrieve(question, k=k)
                raw_answer, refusal_reason = generate_answer_with_guard(question, retrieved)

                if refusal_reason == "answered":
                    answer, cited_indices = validate_and_clean_citations(raw_answer, retrieved)
                else:
                    answer = raw_answer
                    cited_indices = []

            # Display the answer
            st.markdown("### Answer")

            # Three-way display based on what happened
            if refusal_reason == "distance_threshold":
                st.error(answer)
                st.caption("🛑 Layer 1 guard: top retrieval distance exceeded threshold. The LLM was not called.")
            elif refusal_reason == "llm_refusal":
                st.warning(answer)
                st.caption("⚠️ Layer 2 guard: retrieval distances were OK, but the LLM judged the context insufficient.")
            else:
                # Style citations: replace [N] with bold
                styled = re.sub(r"\[(\d+)\]", r"**[\1]**", answer)
                st.markdown(styled)

                if cited_indices:
                    cited_str = ", ".join(f"[{n}]" for n in cited_indices)
                    st.caption(f"✅ Cited sources: {cited_str} — see below for full chunks.")
                else:
                    st.caption("⚠️ No source citations detected in the answer. Verify carefully against the sources below.")

            # Display sources — cited ones first and expanded, uncited collapsed
            st.markdown("### Sources")
            st.caption("Cited sources are expanded by default. Uncited sources are collapsed (the retriever found them but the LLM didn't use them).")

            for i, chunk in enumerate(retrieved, start=1):
                is_cited = i in cited_indices
                is_above_threshold = chunk["distance"] > DISTANCE_THRESHOLD
                flag = "✅" if is_cited else ("🚫" if is_above_threshold else "○")
                label = f"{flag} [{i}] {chunk['paper_title']} — distance {chunk['distance']:.3f}"

                with st.expander(label, expanded=is_cited):
                    st.markdown(f"**URL:** `{chunk['paper_url']}`")
                    st.markdown(f"**Citation status:** {'Cited by LLM' if is_cited else 'Retrieved but not cited'}")
                    st.markdown("**Chunk text:**")
                    st.text(chunk["text"])


# ====================
# FOOTER
# ====================
st.divider()
st.caption(
    "Built by [Jatin Goyal](https://github.com/Jtgoyal) · "
    "Code on [GitHub](https://github.com/Jtgoyal/rag-arxiv-assistant) · "
    "Two-layer hallucination guard + citation validation · "
    "Evaluated on 20 questions: 95% retrieval@5, 132 invalid citations dropped"
)