# lab4.py  — Lab 4 Part A (ChromaDB + PDF ingestion + quick search)

import os
import glob
import textwrap
from io import BytesIO

import streamlit as st
from pypdf import PdfReader

# ---------- SQLite shim (MUST be before importing chromadb) ----------
try:
    import pysqlite3  # ships a modern SQLite >= 3.35
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # If the shim isn't available, Chroma will raise a clear error later.
    pass
# --------------------------------------------------------------------

# Vector DB
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ================================
# Config / Keys
# ================================
st.set_page_config(
    page_title="Lab 4 ▢ ChromaDB (Part A)",
    page_icon=":material/folder:",
    layout="wide",
)
st.title("Lab 4 ▢ Part A: Build a ChromaDB Collection")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not OPENAI_KEY:
    st.warning(
        "OpenAI API key missing. Add OPENAI_API_KEY to .streamlit/secrets.toml or your environment.",
        icon="⚠️",
    )

# ================================
# Helper functions
# ================================
def read_pdf_text(path: str) -> str:
    """Extract text from a PDF using pypdf. Empty pages are handled gracefully."""
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
    """
    Simple paragraph-based chunker to keep embedding calls small.
    """
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
                # A single mega-paragraph—hard cut
                chunks.append(p[:max_chars])
                cur = p[max(0, max_chars - overlap):max_chars - overlap + max_chars]
    if cur:
        chunks.append(cur)
    return chunks

def get_client(persist_dir: str = ".chromadb"):
    """Create a Chroma client that persists to disk so hot-reloads don’t wipe data."""
    return chromadb.Client(Settings(persist_directory=persist_dir))

def get_collection(client, name: str, api_key: str):
    embedder = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection(
        name=name,
        embedding_function=embedder,
    )

def build_or_refresh_collection(pdf_glob: str = "pdfs/*.pdf"):
    """
    Build (or refresh) the collection from all PDFs under pdf_glob.
    Safe to click multiple times—duplicates are avoided with stable IDs.
    """
    client = get_client()
    coll = get_collection(client, "Lab4Collection", OPENAI_KEY)

    pdf_paths = sorted(glob.glob(pdf_glob))
    if not pdf_paths:
        st.info("No PDFs found. Put your files in the `pdfs/` folder.", icon="ℹ️")
        return coll, 0

    to_add_ids, to_add_docs, to_add_metas = [], [], []

    for path in pdf_paths:
        fname = os.path.basename(path)
        text = read_pdf_text(path)
        if not text:
            continue
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            # Deterministic, file-based id so re-runs won’t duplicate
            doc_id = f"{fname}::chunk-{idx}"
            to_add_ids.append(doc_id)
            to_add_docs.append(ch)
            to_add_metas.append({"source": fname, "chunk": idx})

    # Add in batches; if duplicates exist, fall back to per-item add.
    added = 0
    batch = 512
    for start in range(0, len(to_add_ids), batch):
        end = start + batch
        try:
            coll.add(
                ids=to_add_ids[start:end],
                documents=to_add_docs[start:end],
                metadatas=to_add_metas[start:end],
            )
            added += (end - start)
        except Exception:
            for i in range(start, end):
                try:
                    coll.add(
                        ids=[to_add_ids[i]],
                        documents=[to_add_docs[i]],
                        metadatas=[to_add_metas[i]],
                    )
                    added += 1
                except Exception:
                    pass

    try:
        client.persist()
    except Exception:
        pass

    return coll, added

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.subheader("Ingestion")
    st.caption("PDFs are read from `pdfs/*.pdf`.")
    if st.button("Build / Refresh collection", type="primary", use_container_width=True):
        coll, added = build_or_refresh_collection()
        st.session_state["Lab4_coll_ready"] = True
        st.success(f"Collection ready. Added {added} chunks.")
    st.divider()
    st.subheader("Debug")
    if st.button("Count vectors"):
        coll = get_collection(get_client(), "Lab4Collection", OPENAI_KEY)
        st.info(f"Vector count: {coll.count()}")

# ================================
# Main UI: quick search
# ================================
st.header("Quick test query")
query = st.text_input("Enter a query", value="data mining")
k = st.slider("Results", 1, 10, 3)

if st.button("Search"):
    coll = get_collection(get_client(), "Lab4Collection", OPENAI_KEY)
    res = coll.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    st.subheader("Top matches")
    if not docs:
        st.write("No matches yet—try building the collection from the sidebar.")
    else:
        for i, (d, m) in enumerate(zip(docs, metas), start=1):
            source = m.get("source", "unknown")
            chunk_i = m.get("chunk", "?")
            with st.expander(f"{i}. {source}  (chunk {chunk_i})", expanded=False):
                st.write(textwrap.shorten(d, width=900, placeholder=" …"))
