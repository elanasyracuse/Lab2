# lab4_simple.py — Lab 4 Part A (ChromaDB + PDF ingestion + quick search)
import os, glob
import streamlit as st
from pypdf import PdfReader

# ---------- SQLite shim (must be before chromadb import) ----------
try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# ------------------------------------------------------------------

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ================================
# Page setup & keys
# ================================
st.set_page_config(page_title="Lab 4 — ChromaDB (Simple)", page_icon="📄", layout="centered")
st.markdown("## Lab 4 📒 ChromaDB")
st.caption("PDFs are read from `pdfs/*.pdf`")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_KEY:
    st.warning("OpenAI API key missing. Set `OPENAI_API_KEY` in secrets or env.", icon="⚠️")

# ================================
# Helpers
# ================================
def read_pdf_text(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        pieces = []
        for p in reader.pages:
            try:
                pieces.append(p.extract_text() or "")
            except Exception:
                pieces.append("")
        return "\n".join(pieces).strip()

def chunk_text(text: str, max_chars: int = 1400, overlap: int = 150):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + "\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
                tail = cur[-overlap:] if overlap > 0 else ""
                cur = (tail + "\n" + p).strip()
            else:
                chunks.append(p[:max_chars])
                cur = p[max(0, max_chars - overlap):max_chars - overlap + max_chars]
    if cur:
        chunks.append(cur)
    return chunks

def get_client(persist_dir: str = ".chromadb"):
    return chromadb.Client(Settings(persist_directory=persist_dir))

def get_collection(client, name: str, api_key: str):
    embedder = OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
    return client.get_or_create_collection(name=name, embedding_function=embedder)

def build_collection(pdf_glob: str = "pdfs/*.pdf"):
    client = get_client()
    coll = get_collection(client, "Lab4Collection", OPENAI_KEY)

    pdf_paths = sorted(glob.glob(pdf_glob))
    if not pdf_paths:
        return coll, 0, "No PDFs found in `pdfs/`."

    ids, docs, metas = [], [], []
    for path in pdf_paths:
        fname = os.path.basename(path)
        text = read_pdf_text(path)
        if not text:
            continue
        for i, ch in enumerate(chunk_text(text)):
            ids.append(f"{fname}::chunk-{i}")
            docs.append(ch)
            metas.append({"source": fname, "chunk": i})

    added = 0
    for start in range(0, len(ids), 256):
        end = start + 256
        try:
            coll.add(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end])
            added += (end - start)
        except Exception:
            for j in range(start, end):
                try:
                    coll.add(ids=[ids[j]], documents=[docs[j]], metadatas=[metas[j]])
                    added += 1
                except Exception:
                    pass

    try:
        client.persist()
    except Exception:
        pass

    return coll, added, None

# --- NEW: keep the collection once per app run in session_state ---
def ensure_collection_in_session():
    if "Lab4_vectorDB" not in st.session_state:
        client = get_client()
        st.session_state["Lab4_vectorDB"] = get_collection(client, "Lab4Collection", OPENAI_KEY)
    return st.session_state["Lab4_vectorDB"]

# ================================
# UI
# ================================
col_build, col_info = st.columns([1, 2])
with col_build:
    if st.button("📚 Build / Refresh collection", use_container_width=True, type="primary"):
        coll, added, msg = build_collection()
        if msg:
            st.info(msg)
        # store for this run
        st.session_state["Lab4_vectorDB"] = coll
        st.success(f"Collection ready. Added {added} chunks.")

with col_info:
    coll = ensure_collection_in_session()
    try:
        st.write(f"**Current vector count:** {coll.count()}")
    except Exception:
        st.write("**Current vector count:** 0")

st.divider()
st.markdown("### Quick search (prints the top 3 document names)")

q = st.text_input("Enter a query", value="data mining")
if st.button("🔎 Search", use_container_width=True) and q.strip():
    coll = ensure_collection_in_session()
    try:
        # ask for a few more results to dedupe down to 3 unique filenames
        res = coll.query(query_texts=[q.strip()], n_results=10)
        metas = res.get("metadatas", [[]])[0]
    except Exception as e:
        metas = []
        st.error(f"Search failed: {e}")

    # keep first occurrence of each filename in ranked order
    ordered_unique_files = []
    seen = set()
    for m in metas:
        name = m.get("source", "unknown")
        if name not in seen:
            ordered_unique_files.append(name)
            seen.add(name)
        if len(ordered_unique_files) == 3:
            break

    if not ordered_unique_files:
        st.info("No matches yet. If this is a fresh run, click **Build / Refresh collection**.")
    else:
        st.markdown("**Top 3 documents**")
        for i, fname in enumerate(ordered_unique_files, start=1):
            st.write(f"{i}. {fname}")
