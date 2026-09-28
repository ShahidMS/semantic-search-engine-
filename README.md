# RAG-Enhanced Semantic AI Search Engine

A Streamlit app that indexes your documents (PDF/DOCX/TXT), retrieves the most
relevant passages for a natural-language question using semantic (embedding)
search plus optional cross-encoder re-ranking, and can optionally generate a
grounded, cited answer using an LLM (Retrieval-Augmented Generation).

## Features
- Dark technical-console UI (custom theme, not default Streamlit look) with a relevance-meter readout on every result showing the actual semantic/rerank scores
- PDF / DOCX / TXT ingestion (with per-file error handling — one bad file won't crash a batch upload)
- Word-based chunking with overlap
- Sentence-Transformer embeddings (`all-MiniLM-L6-v2`)
- FAISS vector similarity search
- Top-K retrieval with cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`)
- Duplicate-upload protection (re-indexing the same file is a no-op)
- Sidebar view of indexed documents + one-click "Clear index"
- **Live internet search** (DuckDuckGo, no API key required) — blends with document results
- Optional RAG answer generation with source citations, grounded in your documents *and* the web when both are enabled (works without an API key too — retrieval still functions)
- Streamlit UI

## Project structure
```
.
├── app.py                 # Streamlit UI (custom dark theme)
├── rag_engine.py           # Ingestion, chunking, embedding, FAISS search, web search, RAG
├── requirements.txt
├── .streamlit/config.toml   # theme colors
├── .env.example             # copy to .env to set an OpenAI key (optional)
├── sample_data.txt          # sample content for a quick demo
├── tests/
│   └── test_rag_engine.py   # smoke test, no UI/API key required
└── index/                   # created at runtime — FAISS index + metadata
```

## Setup & run

Recommended: Python 3.10 or 3.11, in a virtual environment.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding and re-ranking models (~100 MB total)
from Hugging Face, so it needs an internet connection once. After that they're
cached locally and startup is fast.

The app opens at `http://localhost:8501`.

## Demo script (for your submission walkthrough)
1. Run the app.
2. In the sidebar, upload `sample_data.txt` (included in this repo) and click **Index uploaded documents**.
3. Ask: *"What are the eligibility requirements for a software engineering job?"*
4. Toggle **Use cross-encoder re-ranking** on/off and compare the ranking of results.
5. Try uploading the same file again to show the duplicate-upload guard (0 new chunks, no duplication).
6. Click **Clear index** to reset, if you want to demo ingestion from scratch again.
7. (Optional) Set up an LLM key as below, enable **Generate RAG answer**, and ask the same question — the app now returns a synthesized, cited answer instead of raw passages.

## Internet search
Check **Search the internet** in the sidebar to also pull live results from
DuckDuckGo (no API key needed). It works standalone (even with zero documents
indexed) or alongside your document search — when both are on and **Generate
RAG answer** is enabled, the model is given both sets of context and cites
them separately as `[Source N]` (your documents) and `[Web N]` (the internet).

## Optional: enabling RAG answer generation
Retrieval, re-ranking, and web search all work with no API key. To also
generate a synthesized answer, set up **one** of the two providers below in
a `.env` file (copy `.env.example` to `.env` and fill it in), then restart
the app and check **Generate RAG answer** in the sidebar.

**Option A — Groq (recommended, free tier):**
1. Get a free key at https://console.groq.com/keys
2. In `.env`:
   ```
   GROQ_API_KEY=gsk_...
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

**Option B — OpenAI (paid, no free tier):**
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
```

If `GROQ_API_KEY` is set, Groq is used; otherwise it falls back to OpenAI if
that key is set. Groq exposes an OpenAI-compatible API, so no extra library
is needed — the app just points the existing `openai` client at Groq's
endpoint.

Without a key, the app is explicit that no LLM answer was generated instead of
pretending otherwise — retrieval results are still shown.

## Running the smoke test
Confirms indexing, duplicate handling, search relevance, and index-clearing
without needing the Streamlit UI:
```bash
python -m pytest tests/ -v
# or
python tests/test_rag_engine.py
```

## Architecture
```
Documents → extraction → chunking → embeddings → FAISS
Query → embedding → semantic retrieval → re-ranking → context
Context + Query → RAG/LLM → grounded answer + sources
```
See `advanced_architecture.png` for the diagram version.

## Troubleshooting
- **`ModuleNotFoundError`** — you're likely not in the virtual environment where you ran `pip install -r requirements.txt`. Re-activate it.
- **Model download fails / hangs on first run** — check your internet connection; Hugging Face Hub must be reachable. Retry `streamlit run app.py`.
- **Scanned/image PDFs return no text** — this app extracts text directly from the PDF; it doesn't run OCR. Convert to a text-based PDF or paste the content into a `.txt` file.
- **Re-uploading a file adds nothing** — that's expected: identical content is detected and skipped so the index doesn't fill with duplicates. Use **Clear index** if you want to rebuild from scratch.
- **RAG answer says no LLM key configured** — set `GROQ_API_KEY` (free) or `OPENAI_API_KEY` (paid) in `.env` (see above) and restart the app.
