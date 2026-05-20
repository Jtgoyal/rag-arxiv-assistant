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
from rag_pipeline import PaperIndex, generate_answer, format_response, TOP_K


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
        st.info("Loading PDFs from `local_papers/` folder.")
        topic = None
        max_papers = None

    load_button = st.button("Load Papers", type="primary")

    if load_button:
        with st.spinner("Loading and processing papers..."):
            try:
                if mode == "ArXiv (live)":
                    chunks = fetch_and_chunk(topic, max_papers=max_papers)
                else:
                    chunks = load_and_chunk_local(folder="local_papers")

                if not chunks:
                    st.error("No chunks were created. Check your sources.")
                else:
                    # Build the FAISS index
                    index = PaperIndex()
                    index.build(chunks)

                    # Persist in session_state
                    st.session_state.index = index
                    st.session_state.chunks = chunks

                    # Compute unique papers for display
                    unique_papers = {}
                    for c in chunks:
                        title = c["paper_title"]
                        if title not in unique_papers:
                            unique_papers[title] = c["paper_url"]
                    st.session_state.papers_info = unique_papers

                    st.success(f"Loaded {len(unique_papers)} papers, {len(chunks)} chunks indexed.")
            except Exception as e:
                st.error(f"Failed to load papers: {e}")

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
        with st.spinner("Retrieving and generating..."):
            retrieved = st.session_state.index.retrieve(question, k=k)
            raw_answer = generate_answer(question, retrieved)

            # Validate citations: drop any [N] that doesn't correspond to a real chunk
            from rag_pipeline import validate_and_clean_citations
            answer, cited_indices = validate_and_clean_citations(raw_answer, retrieved)

        # Display the answer
        st.markdown("### Answer")

        refusal_indicators = ["cannot answer", "not enough information", "not in the loaded papers", "insufficient context"]
        is_refusal = any(phrase in answer.lower() for phrase in refusal_indicators)

        if is_refusal:
            st.warning(answer)
            st.caption("The system couldn't find relevant information in the loaded papers. Try a different question or load more papers.")
        else:
            # Style citations: replace [N] with a visually-prominent badge
            styled = re.sub(
                r"\[(\d+)\]",
                r"**[\1]**",  # bold the citation
                answer,
            )
            st.markdown(styled)

            # If the LLM cited specific sources, surface a clear hint
            if cited_indices:
                cited_str = ", ".join(f"[{n}]" for n in cited_indices)
                st.caption(f"Cited sources: {cited_str} — see below for full chunks.")
            else:
                st.caption("⚠️ No source citations detected in the answer. Verify carefully against the sources below.")

        # Display sources — cited ones first and expanded, uncited collapsed
        st.markdown("### Sources")
        st.caption("Cited sources are expanded by default. Uncited sources are collapsed (the retriever found them but the LLM didn't use them).")

        # Show cited sources first, expanded
        for i, chunk in enumerate(retrieved, start=1):
            is_cited = i in cited_indices if not is_refusal else False
            label = f"{'✅' if is_cited else '○'} [{i}] {chunk['paper_title']} — distance {chunk['distance']:.3f}"

            with st.expander(label, expanded=is_cited):
                st.markdown(f"**URL:** `{chunk['paper_url']}`")
                st.markdown(f"**Citation status:** {'Cited by LLM' if is_cited else 'Retrieved but not cited'}")
                st.markdown("**Chunk text:**")
                st.text(chunk["text"])