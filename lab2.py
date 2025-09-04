import os
import streamlit as st
from openai import OpenAI

# Optional: allow .env locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- helpers ---------
def get_api_key() -> str | None:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return os.getenv("OPENAI_API_KEY")

def read_pdf(file_obj) -> str:
    data = file_obj.read()
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        pages = [p.get_text("text") for p in doc]
        doc.close()
        return "\n".join(pages).strip()
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        pass
    return ""

def clear_doc_state():
    for k in ("document_text", "document_name"):
        if k in st.session_state:
            del st.session_state[k]

# ---------- UI ----------
st.title("📘 Lab 2: Document question answering (Default)")
st.caption("API key is loaded from Streamlit secrets or your environment (.env).")

openai_api_key = get_api_key()
if not openai_api_key:
    st.error(
        "Missing `OPENAI_API_KEY`. Add it to **.streamlit/secrets.toml** on deploy "
        "or set it in a local `.env` / environment variable."
    )
    st.stop()

model = st.selectbox(
    "Choose a model",
    ["gpt-3.5-turbo", "gpt-4.1", "gpt-5-chat-latest", "gpt-5-nano"],
    index=1,
)

client = OpenAI(api_key=openai_api_key)
key_ok = False
try:
    _ = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    st.success("API key loaded and validated ✅")
    key_ok = True
except Exception as e:
    st.error(f"API key or billing issue: {e}")

if key_ok:
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    if not uploaded_file:
        clear_doc_state()

    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            document = uploaded_file.read().decode(errors="ignore")
            st.session_state["document_text"] = document
            st.session_state["document_name"] = uploaded_file.name
        elif file_extension == 'pdf':
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

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder='Example: "Is this course hard?"',
        disabled=("document_text" not in st.session_state),
    )

    if ("document_text" in st.session_state) and question:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Here's a document: {st.session_state['document_text']}\n\n---\n\n{question}"
                ),
            }
        ]
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)
