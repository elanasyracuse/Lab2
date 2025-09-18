# lab4b_chatbot.py — Lab 4B (Course information chatbot with RAG)
import os, glob
import streamlit as st
from pypdf import PdfReader

# --- SQLite shim ---
try:
    import pysqlite3, sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
# -------------------

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# ================================
# Page setup & keys
# ================================
st.set_page_config(page_title="Lab 4B — Course Chatbot", page_icon="🤖", layout="centered")
st.markdown("## Lab 4B 🤖 Course Information Chatbot")

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
client_llm = OpenAI(api_key=OPENAI_KEY)

# ================================
# Helpers
# ================================
def read_pdf_text(path: str) -> str:
    with open(path, "rb") as f:
        from pypdf import PdfReader
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
    try:
        coll.add(ids=ids, documents=docs, metadatas=metas)
    except Exception:
        pass
    try:
        client.persist()
    except Exception:
        pass
    return coll

def ensure_collection_in_session():
    if "Lab4_vectorDB" not in st.session_state:
        client = get_client()
        st.session_state["Lab4_vectorDB"] = get_collection(client, "Lab4Collection", OPENAI_KEY)
    return st.session_state["Lab4_vectorDB"]

def rag_answer(question: str, k: int = 3):
    coll = ensure_collection_in_session()
    res = coll.query(query_texts=[question], n_results=k)
    docs = res.get("documents", [[]])[0]
    if not docs:
        return "I couldn’t find anything in the course PDFs. Answering from general knowledge."

    # Combine retrieved chunks
    context = "\n\n".join(docs)
    prompt = f"""
You are a helpful course information assistant.

User question: {question}

Relevant course material:
{context}

Answer the user clearly. If you used the retrieved course material, say so.
If not enough info is found, be clear about that.
"""
    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",   # or "gpt-5-mini"
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ================================
# UI
# ================================
if st.button("📚 Build / Refresh collection", type="primary"):
    coll = build_collection()
    st.session_state["Lab4_vectorDB"] = coll
    st.success("Collection built/refreshed.")

st.divider()
st.markdown("### Chat with the Course Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input
user_q = st.chat_input("Ask me about the course…")
if user_q:
    st.session_state.chat_history.append(("user", user_q))
    with st.chat_message("user"):
        st.write(user_q)

    answer = rag_answer(user_q)
    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)
