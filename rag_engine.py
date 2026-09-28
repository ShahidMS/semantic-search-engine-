import os, re, json, tempfile, hashlib, shutil
from pathlib import Path
import numpy as np
import faiss

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sentence_transformers import SentenceTransformer, CrossEncoder


class RAGEngine:
    def __init__(self, index_dir="index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)

        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            raise RuntimeError(
                "Could not load the embedding/re-ranking models. "
                "Check your internet connection (first run downloads them from "
                f"Hugging Face) and try again. Original error: {e}"
            )

        self.records = []          # one entry per chunk
        self.doc_hashes = set()    # sha256 of raw text, used to skip duplicate uploads
        self.index = None
        self._load()

    # ---------- persistence ----------

    def _load(self):
        meta = self.index_dir / "metadata.json"
        idx = self.index_dir / "faiss.index"
        if meta.exists() and idx.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            self.records = data.get("records", [])
            self.doc_hashes = set(data.get("doc_hashes", []))
            self.index = faiss.read_index(str(idx))

    def _save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        payload = {"records": self.records, "doc_hashes": list(self.doc_hashes)}
        (self.index_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def clear_index(self):
        """Wipe the vector index and all indexed documents."""
        self.index = None
        self.records = []
        self.doc_hashes = set()
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(exist_ok=True)

    def list_documents(self):
        """Return [{'title':..., 'source':..., 'chunks': n}, ...] for the sidebar."""
        counts = {}
        for r in self.records:
            key = (r["title"], r["source"])
            counts[key] = counts.get(key, 0) + 1
        return [
            {"title": t, "source": s, "chunks": n}
            for (t, s), n in counts.items()
        ]

    # ---------- ingestion ----------

    def _chunk(self, text, size=500, overlap=80):
        words = re.findall(r"\S+", text)
        chunks = []
        step = max(1, size - overlap)
        for i in range(0, len(words), step):
            c = " ".join(words[i:i + size]).strip()
            if c:
                chunks.append(c)
            if i + size >= len(words):
                break
        return chunks

    def add_text(self, title, text, source="manual"):
        if not text or not text.strip():
            return 0

        text_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if text_hash in self.doc_hashes:
            return 0  # already indexed, skip silently (prevents duplicate chunks
                      # if the same file is uploaded/indexed twice)

        chunks = self._chunk(text)
        if not chunks:
            return 0

        emb = self.embedder.encode(chunks, normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

        start = len(self.records)
        self.records += [
            {"id": start + i, "title": title, "source": source, "text": c}
            for i, c in enumerate(chunks)
        ]
        self.doc_hashes.add(text_hash)
        self._save()
        return len(chunks)

    def add_uploaded_files(self, uploaded):
        """
        Returns (total_chunks_added, errors) where errors is a list of
        (filename, message) tuples for files that failed to parse.
        """
        total = 0
        errors = []
        for f in uploaded:
            suffix = Path(f.name).suffix.lower()
            try:
                if suffix == ".txt":
                    text = f.read().decode("utf-8", errors="ignore")
                elif suffix == ".pdf":
                    from pypdf import PdfReader
                    import io
                    data = f.read()
                    reader = PdfReader(io.BytesIO(data))
                    text = "\n".join((p.extract_text() or "") for p in reader.pages)
                    if not text.strip():
                        errors.append((f.name, "No extractable text found (likely a scanned/image-only PDF)."))
                        continue
                elif suffix == ".docx":
                    from docx import Document
                    data = f.read()
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
                    tmp.write(data)
                    tmp.close()
                    try:
                        d = Document(tmp.name)
                        text = "\n".join(p.text for p in d.paragraphs)
                    finally:
                        os.unlink(tmp.name)
                else:
                    errors.append((f.name, f"Unsupported file type: {suffix or 'unknown'}"))
                    continue

                added = self.add_text(f.name, text, f.name)
                total += added
                if added == 0 and text.strip():
                    errors.append((f.name, "Already indexed — skipped duplicate."))

            except Exception as e:
                errors.append((f.name, f"Failed to process: {e}"))

        return total, errors

    # ---------- search ----------

    def search(self, query, top_k=5, rerank=True):
        if self.index is None or not self.records:
            return {"results": []}

        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        k = min(max(top_k * 4, top_k), len(self.records))
        scores, ids = self.index.search(q, k)

        results = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            r = dict(self.records[int(i)])
            r["semantic_score"] = float(score)
            r["rerank_score"] = None
            results.append(r)

        if rerank and results:
            pairs = [(query, r["text"]) for r in results]
            rs = self.reranker.predict(pairs)
            for r, s in zip(results, rs):
                r["rerank_score"] = float(s)
            results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        else:
            results = sorted(results, key=lambda x: x["semantic_score"], reverse=True)[:top_k]

        return {"results": results}

    # ---------- web search ----------

    def web_search(self, query, max_results=5):
        """
        Live internet search via DuckDuckGo (no API key required).
        Returns (results, error) where results is a list of
        {'title', 'body', 'href'} and error is None or a message string.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # older package name, fallback
            except ImportError:
                return [], ("Web search requires the 'ddgs' package. "
                             "Install it with: pip install ddgs")
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            results = [
                {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
                for r in raw
            ]
            return results, None
        except Exception as e:
            return [], f"Web search failed: {e}"

    # ---------- answer generation ----------

    def _llm_provider(self):
        """
        Picks a provider from environment variables. Groq is preferred when
        both keys are set since it has a free tier; Groq exposes an
        OpenAI-compatible endpoint so the same 'openai' client library works
        for both, just pointed at a different base_url.
        Returns (client_kwargs, model, provider_name) or (None, None, None).
        """
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if groq_key:
            return (
                {"api_key": groq_key, "base_url": "https://api.groq.com/openai/v1"},
                os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "Groq",
            )
        if openai_key:
            return {"api_key": openai_key}, os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "OpenAI"
        return None, None, None

    def generate_answer(self, query, results, web_results=None, length="concise"):
        """
        results: retrieved document chunks (from self.search)
        web_results: optional list of {'title','body','href'} from self.web_search
        length: 'concise' (~5-7 sentences) or 'detailed' (thorough, multi-paragraph)
        """
        client_kwargs, model, provider = self._llm_provider()
        if not provider:
            return ("RAG retrieval is ready, but no LLM API key is configured. "
                    "The retrieved context is shown below. Set GROQ_API_KEY (free) "
                    "or OPENAI_API_KEY (paid) in .env to enable generated answers.")
        if not results and not web_results:
            return "No relevant context was retrieved, so no grounded answer can be generated."
        try:
            from openai import OpenAI
            client = OpenAI(**client_kwargs)

            context_parts = []
            for i, r in enumerate(results):
                context_parts.append(f"[Source {i+1}] {r['title']}\n{r['text']}")
            if web_results:
                for j, w in enumerate(web_results):
                    context_parts.append(f"[Web {j+1}] {w['title']} ({w['href']})\n{w['body']}")
            context = "\n\n".join(context_parts)

            length_instruction = (
                "Answer in 5-7 sentences maximum — a tight, well-organized summary, not an exhaustive breakdown."
                if length == "concise"
                else "Answer thoroughly, covering all relevant points from the context in full detail, using multiple paragraphs or a structured list where helpful."
            )

            prompt = f"""Answer the question using only the supplied context. Context items labelled [Source N] come from the user's uploaded documents; items labelled [Web N] come from a live internet search. If the context does not contain the answer, say that it is not available. Cite sources inline as [Source 1], [Web 2], etc., and prefer document sources over web sources when both agree. {length_instruction}
Question: {query}
Context:
{context}"""
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"LLM generation failed ({provider}): {e}. Retrieved context is still available in the Sources section."
