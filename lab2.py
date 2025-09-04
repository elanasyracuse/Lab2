import os
import io
import streamlit as st
from typing import Optional
from openai import OpenAI

# Optional: load .env for local dev (no-op in prod if .env is absent)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- helpers ---------
def get_api_key() -> Optional[str]:
    # 1) Prefer env/.env (works locally & in Codespaces)
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # 2) Try Streamlit secrets ONLY if configured
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

def read_pdf(file_obj) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz) if available,
    otherwise fall back to PyPDF2. No hard dependency—imports are inside.
    """
    data = file_obj.read()  # read bytes once

    # Try PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        doc.close()
        return "\n".join(pages).strip()
    except Exception:
        pass

    # Try PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        pass

    return ""  # if neither works or extraction fails

def clear_doc_state():
    for k in ("document_text", "document_name"):
        if k in st.session_state:
            del st.session_state[k]

def summarize_doc(client: OpenAI, model: str, document_text: str, style: str):
    """
    Call the Chat Completions API to summarize the document using the chosen style.
    `style` is one of: '100_words', 'two_paragraphs', 'five_bullets'
    """
    if style == "100_words":
        instructions = (
            "Summarize the document in ~100 words (strict cap: 110 words max). "
            "Write concise, plain English. No preamble, no title."
        )
    elif style == "two_paragraphs":
        instructions = (
            "Summarize the document in exactly two connected paragraphs. "
            "Paragraph 1: core ideas. Paragraph 2: implications or next steps. "
            "No title or bullet points."
        )
    else:  # five_bullets
        instructions = (
            "Summarize the document in exactly five bullet points. "
            "Each bullet should be a single sentence. No intro/outro, no title."
        )

    messages = [
        {"role": "system", "content": "You are a careful, concise summarizer."},
        {"role": "user", "content": f"{instructions}\n\n---\nDOCUMENT:\n{document_text}"},
    ]

    # Stream the response (no max_tokens to avoid param mismatch on some models)
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

# ---------- UI ----------
st.title("📘 Lab 2: Summarization (Default)")
st.caption("🔐 API key source preference: env/.env if present; otherwise Streamlit secrets (if set).")

# Sidebar controls for Lab 2c
with st.sidebar:
    st.header("Summary Options")
    summary_choice = st.radio(
        "Choose a summary type:",
        [
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points",
        ],
        index=0,
        help="The type of summary is passed directly to the LLM.",
    )

    st.divider()
    st.header("Model")
    # Base mini models to choose from; default to 4o-mini for best $/quality
    base_model = st.selectbox(
        "Base model (mini):",
        ["gpt-4o","gpt-4o-mini"],
        index=0,
        help="Pick an inexpensive default model. You can override to 4o with the checkbox below.",
    )
    use_advanced = st.checkbox("Use Advanced Model (4o)", value=False)

    # Lab 2d: explanation about default model
    with st.expander("Why is this the default model? (Lab 2d)"):
        st.markdown(
            "- **Default = gpt-4o-mini**: It delivers strong quality at significantly lower cost than 4o, "
            "making it ideal for frequent summarization.\n"
            "- **Advanced = gpt-4o**: Use when the source text is long/nuanced and you want the most robust reasoning. "
            "It typically costs more than mini models.\n\n"
            "Choosing a cheaper default keeps your average cost low, while the checkbox lets you selectively pay for "
            "higher quality only when needed."
        )

# Resolve final model
final_model = "gpt-4o" if use_advanced else base_model

api_key = get_api_key()
if not api_key:
    st.error(
        "Missing `OPENAI_API_KEY`. Add it to `.env` for local runs, "
        "or to **.streamlit/secrets.toml** when deployed."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# Validate key with a tiny 'ping' (no max_tokens to avoid 400s on some models)
try:
    _ = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}],
        temperature=0,
    )
    st.success("API key loaded and validated ✅")
except Exception as e:
    st.error("OpenAI error during validation. See details below.")
    st.exception(e)
    st.stop()

# File upload (no question box; we will directly show the chosen summary)
uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

# If user removed the file, purge all doc data immediately
if not uploaded_file:
    clear_doc_state()

# If a file is present, read it fresh each time (no lingering access)
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == "txt":
        document = uploaded_file.read().decode(errors="ignore")
        st.session_state["document_text"] = document
        st.session_state["document_name"] = uploaded_file.name
    elif file_extension == "pdf":
        document = read_pdf(uploaded_file)
        if not document:
            st.error(
                "Couldn't extract text from this PDF. "
                "Install PyMuPDF (fitz) or PyPDF2 in your environment and try again."
            )
            clear_doc_state()
        else:
            st.session_state["document_text"] = document
            st.session_state["document_name"] = uploaded_file.name
    else:
        st.error("Unsupported file type.")
        clear_doc_state()

# If we have a doc, summarize it immediately based on sidebar choice
if "document_text" in st.session_state:
    st.subheader("Summary")
    style_key = (
        "100_words"
        if summary_choice.startswith("Summarize the document in 100 words")
        else "two_paragraphs"
        if "2 connecting paragraphs" in summary_choice
        else "five_bullets"
    )
    try:
        stream = summarize_doc(client, final_model, st.session_state["document_text"], style_key)
        st.write_stream(stream)
        st.caption(f"Model: `{final_model}` · Mode: **{summary_choice}**")
    except Exception as e:
        st.error("OpenAI error while generating the summary. See details below.")
        st.exception(e)
else:
    st.info("Upload a .txt or .pdf to generate a summary.")
