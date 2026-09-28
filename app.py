import html
import math
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Semantic Search Engine", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------------
# Styling — dark "lab console" theme. Palette, type, and the relevance-meter
# component are all keyed to the subject: vector similarity search.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0C1116;
    --surface: #141B22;
    --surface-raised: #1A232C;
    --border: #26323C;
    --text: #E7EDF2;
    --text-muted: #7F93A0;
    --signal: #3ECF8E;
    --signal-dim: rgba(62, 207, 142, 0.16);
    --web: #E8A33D;
    --web-dim: rgba(232, 163, 61, 0.16);
    --answer: #6C8EEF;
    --answer-dim: rgba(108, 142, 239, 0.14);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }

/* Hero */
.ss-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--signal);
    margin-bottom: 0.4rem;
}
.ss-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: var(--text);
    margin: 0 0 0.35rem 0;
    line-height: 1.15;
}
.ss-subtitle {
    color: var(--text-muted);
    font-size: 0.98rem;
    margin-bottom: 1.1rem;
}
.ss-stats { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
.ss-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.32rem 0.7rem;
}
.ss-chip b { color: var(--text); font-weight: 600; }
.ss-chip.on { color: var(--signal); border-color: rgba(62,207,142,0.35); }
.ss-chip.off { color: var(--text-muted); }

hr.ss-rule { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0 1.4rem 0; }

/* Result cards */
.ss-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.15rem 0.9rem 1.15rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.ss-card:hover { border-color: #35434F; transform: translateY(-1px); }
.ss-card-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
.ss-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    background: var(--surface-raised);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.1rem 0.45rem;
    min-width: 1.6rem;
    text-align: center;
}
.ss-card-title { font-weight: 600; color: var(--text); font-size: 0.98rem; flex: 1; }
.ss-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.16rem 0.5rem;
    border-radius: 4px;
    white-space: nowrap;
}
.ss-badge.doc { color: var(--signal); background: var(--signal-dim); }
.ss-badge.web { color: var(--web); background: var(--web-dim); }
.ss-card-body { color: var(--text); font-size: 0.92rem; line-height: 1.55; margin: 0.15rem 0 0.65rem 0; }
.ss-card-body a { color: var(--web); }

/* Relevance meter — the signature element: real similarity scores as a readout */
.ss-meter-row { display: flex; align-items: center; gap: 0.55rem; margin-top: 0.3rem; }
.ss-meter-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    width: 4.6rem;
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.ss-meter-track {
    flex: 1;
    height: 5px;
    background: var(--surface-raised);
    border-radius: 3px;
    overflow: hidden;
}
.ss-meter-fill { height: 100%; border-radius: 3px; background: var(--signal); }
.ss-meter-fill.rerank { background: var(--answer); }
.ss-meter-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    width: 3.2rem;
    text-align: right;
    flex-shrink: 0;
}

/* Answer card */
.ss-answer {
    background: var(--answer-dim);
    border: 1px solid rgba(108,142,239,0.35);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
}
.ss-answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--answer);
    margin-bottom: 0.5rem;
}
.ss-answer-text { color: var(--text); font-size: 0.98rem; line-height: 1.6; white-space: pre-wrap; }

/* Section labels */
.ss-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.1rem 0 0.6rem 0;
}

/* Empty state */
.ss-empty {
    border: 1px dashed var(--border);
    border-radius: 10px;
    padding: 1.6rem;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.92rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-family: 'Space Grotesk', sans-serif; }

/* Accessibility: visible focus ring */
button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 2px solid var(--signal) !important; outline-offset: 2px; }

@media (prefers-reduced-motion: reduce) {
    .ss-card { transition: none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def score_bar(label, value, css_class="", kind="cosine"):
    """
    Renders a relevance-meter row. `value` is shown exactly as computed.
    Bar width is a visualization aid only:
      - cosine similarity is already ~[0,1] for relevant matches -> clamp directly
      - cross-encoder rerank scores are unbounded logits -> squash with a sigmoid
    """
    if kind == "cosine":
        pct = max(0.0, min(1.0, value))
    else:
        pct = 1 / (1 + math.exp(-value))
    st.markdown(
        f"""<div class="ss-meter-row">
            <div class="ss-meter-label">{esc(label)}</div>
            <div class="ss-meter-track"><div class="ss-meter-fill {css_class}" style="width:{pct*100:.1f}%"></div></div>
            <div class="ss-meter-value">{value:.3f}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_doc_card(rank, r):
    st.markdown(
        f"""<div class="ss-card">
            <div class="ss-card-head">
                <div class="ss-rank">{rank:02d}</div>
                <div class="ss-card-title">{esc(r['title'])}</div>
                <div class="ss-badge doc">Document</div>
            </div>
            <div class="ss-card-body">{esc(r['text'])}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        score_bar("semantic", r["semantic_score"], kind="cosine")
    with c2:
        if r.get("rerank_score") is not None:
            score_bar("rerank", r["rerank_score"], "rerank", kind="logit")


def render_web_card(rank, w):
    href = esc(w.get("href", ""))
    st.markdown(
        f"""<div class="ss-card">
            <div class="ss-card-head">
                <div class="ss-rank">{rank:02d}</div>
                <div class="ss-card-title"><a href="{href}" target="_blank" style="color:inherit; text-decoration:none;">{esc(w['title'])}</a></div>
                <div class="ss-badge web">Web</div>
            </div>
            <div class="ss-card-body">{esc(w['body'])}</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--text-muted);">{href}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    return RAGEngine()


try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to start the RAG engine: {e}")
    st.stop()

docs = engine.list_documents()
total_chunks = sum(d["chunks"] for d in docs)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="ss-eyebrow">Retrieval-Augmented Search</div>', unsafe_allow_html=True)
st.markdown('<div class="ss-title">Semantic Search Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ss-subtitle">Query your own documents by meaning, not keywords — with optional live web search and grounded, cited answers.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Knowledge Base")
    uploaded = st.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    col_a, col_b = st.columns(2)
    with col_a:
        index_clicked = st.button("Index files", use_container_width=True)
    with col_b:
        clear_clicked = st.button("Clear index", use_container_width=True)

    if index_clicked:
        if not uploaded:
            st.warning("Upload at least one document.")
        else:
            with st.spinner("Indexing..."):
                count, errors = engine.add_uploaded_files(uploaded)
            if count:
                st.success(f"Indexed {count} new chunks.")
            if errors:
                for name, msg in errors:
                    st.warning(f"**{name}**: {msg}")
            if not count and not errors:
                st.info("Nothing new to index.")
            st.rerun()

    if clear_clicked:
        engine.clear_index()
        st.success("Index cleared.")
        st.rerun()

    if docs:
        st.divider()
        st.caption(f"INDEXED DOCUMENTS ({len(docs)})")
        for d in docs:
            st.caption(f"📄 {d['title']} — {d['chunks']} chunks")
    else:
        st.divider()
        st.caption("No documents indexed yet.")

    st.divider()
    top_k = st.slider("Retrieved chunks", 1, 10, 5)
    rerank = st.checkbox("Use cross-encoder re-ranking", value=True)
    use_web = st.checkbox("Search the internet", value=False)
    web_k = st.slider("Web results", 1, 8, 4) if use_web else 4
    use_rag = st.checkbox("Generate RAG answer", value=False)
    answer_length = st.radio("Answer length", ["Concise (5-7 lines)", "Detailed"], horizontal=True) if use_rag else "Concise (5-7 lines)"
    st.caption("RAG mode needs a free Groq key or paid OpenAI key. Retrieval and web search work without one.")

# ---------------------------------------------------------------------------
# Stat chips
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="ss-stats">
        <div class="ss-chip"><b>{len(docs)}</b> documents</div>
        <div class="ss-chip"><b>{total_chunks}</b> chunks indexed</div>
        <div class="ss-chip {'on' if use_web else 'off'}">web search {'ON' if use_web else 'OFF'}</div>
        <div class="ss-chip {'on' if use_rag else 'off'}">RAG answer {'ON' if use_rag else 'OFF'}</div>
    </div>""",
    unsafe_allow_html=True,
)
st.markdown('<hr class="ss-rule">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
query = st.text_input("Ask a question", placeholder="What are the eligibility requirements for a software engineering job?", label_visibility="collapsed")

if query.strip():
    has_docs = bool(docs)
    if not has_docs and not use_web:
        st.markdown(
            '<div class="ss-empty">No documents indexed yet. Upload and index a document in the sidebar, or turn on <b>Search the internet</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        doc_results = []
        if has_docs:
            with st.spinner("Searching documents..."):
                doc_results = engine.search(query, top_k=top_k, rerank=rerank)["results"]

        web_results, web_error = [], None
        if use_web:
            with st.spinner("Searching the web..."):
                web_results, web_error = engine.web_search(query, max_results=web_k)
            if web_error:
                st.warning(web_error)

        if not doc_results and not web_results:
            st.markdown('<div class="ss-empty">No matching results found.</div>', unsafe_allow_html=True)
        elif use_rag:
            with st.spinner("Generating answer..."):
                length_key = "concise" if answer_length.startswith("Concise") else "detailed"
                answer = engine.generate_answer(query, doc_results, web_results=web_results, length=length_key)
            st.markdown(
                f"""<div class="ss-answer">
                    <div class="ss-answer-label">Synthesized Answer</div>
                    <div class="ss-answer-text">{esc(answer)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if doc_results:
                st.markdown('<div class="ss-section">Document Sources</div>', unsafe_allow_html=True)
                for i, r in enumerate(doc_results, 1):
                    render_doc_card(i, r)
            if web_results:
                st.markdown('<div class="ss-section">Web Sources</div>', unsafe_allow_html=True)
                for i, w in enumerate(web_results, 1):
                    render_web_card(i, w)
        else:
            if doc_results:
                st.markdown('<div class="ss-section">Semantic Search Results — Your Documents</div>', unsafe_allow_html=True)
                for i, r in enumerate(doc_results, 1):
                    render_doc_card(i, r)
            if web_results:
                st.markdown('<div class="ss-section">Web Search Results</div>', unsafe_allow_html=True)
                for i, w in enumerate(web_results, 1):
                    render_web_card(i, w)
else:
    st.markdown(
        '<div class="ss-empty">Enter a natural-language question to search the knowledge base and/or the web.</div>',
        unsafe_allow_html=True,
    )

with st.expander("System architecture"):
    st.code("""Documents → extraction → chunking → embeddings → FAISS
Query → embedding → semantic retrieval → re-ranking → context
Context + Query → RAG/LLM → grounded answer + sources""")
