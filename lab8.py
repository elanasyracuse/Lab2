import streamlit as st
import PyPDF2
# or if you're using specific classes:
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < text_length:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

def get_embedding(text, client):
    """Get embedding for text using OpenAI"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:8000]  # Limit token length
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

def retrieve_relevant_chunks(query, chunks, embeddings, client, top_k=5):
    """Retrieve most relevant chunks using cosine similarity"""
    query_embedding = get_embedding(query, client)
    
    if query_embedding is None:
        return []
    
    # Calculate cosine similarities
    query_embedding = np.array(query_embedding).reshape(1, -1)
    embeddings_array = np.array(embeddings)
    
    similarities = cosine_similarity(query_embedding, embeddings_array)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return chunks with their similarity scores
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx],
            'index': idx
        })
    
    return results

def rerank_chunks(query, retrieved_chunks, client):
    """Re-rank chunks using LLM-based semantic similarity"""
    if not retrieved_chunks:
        return []
    
    # Use LLM to score relevance
    reranked = []
    
    for item in retrieved_chunks:
        prompt = f"""On a scale of 0-10, rate how relevant this text chunk is to answering the query.
Query: {query}

Text Chunk: {item['chunk'][:500]}...

Respond with ONLY a number between 0-10."""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a relevance scoring assistant. Respond only with a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            # Extract number from response
            score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_text)))
            score = min(max(score, 0), 10)  # Clamp between 0-10
            
            item['rerank_score'] = score
            reranked.append(item)
        except:
            item['rerank_score'] = item['similarity'] * 10  # Fallback to embedding score
            reranked.append(item)
    
    # Sort by rerank score
    reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
    return reranked

def generate_answer(query, relevant_chunks, client, company_name):
    """Generate answer using LLM with retrieved context"""
    # Prepare context from chunks
    context = "\n\n---\n\n".join([
        f"[Source {i+1}, Relevance: {chunk['rerank_score']:.2f}/10]\n{chunk['chunk'][:800]}" 
        for i, chunk in enumerate(relevant_chunks[:3])
    ])
    
    prompt = f"""You are a financial analyst assistant analyzing SEC 10-Q filings for {company_name}.

Based on the following excerpts from the 10-Q filing, answer the user's question.

Context from 10-Q Filing:
{context}

User Question: {query}

Instructions:
1. Provide a clear, comprehensive answer based on the context
2. Cite which sources you used (e.g., "According to Source 1...")
3. If the context doesn't fully answer the question, acknowledge what's missing
4. Use financial terminology appropriately
5. Be objective and analytical

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert in SEC filings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(page_title="SEC 10-Q RAG Chatbot", page_icon="📊", layout="wide")
    
    st.title("📊 SEC 10-Q Financial Reports Chatbot")
    st.markdown("**Retrieval-Augmented Generation (RAG) with Re-Ranking**")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Upload 10-Q Filings")
        
        uploaded_files = st.file_uploader(
            "Upload SEC 10-Q PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload Amazon and Apple 10-Q PDF files"
        )
        
        st.divider()
        
        st.header("⚙️ RAG Settings")
        max_chunks = st.slider(
            "Max chunks to retrieve",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of document chunks to retrieve before re-ranking"
        )
        
        use_reranking = st.checkbox(
            "Enable Re-Ranking",
            value=True,
            help="Use LLM-based re-ranking to improve relevance"
        )
        
        st.divider()
        
        st.markdown("""
        ### 📖 What is RAG?
        
        **Retrieval-Augmented Generation** combines:
        
        1. **Retrieval**: Find relevant document chunks using embeddings
        2. **Re-Ranking**: Score and reorder by semantic relevance
        3. **Generation**: Use LLM to synthesize an answer
        
        This approach grounds AI responses in actual document content!
        """)
    
    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing documents..."):
            client = get_openai_client()
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.documents:
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        # Chunk text
                        chunks = chunk_text(text, chunk_size=1000, overlap=200)
                        
                        # Generate embeddings
                        embeddings = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, chunk in enumerate(chunks):
                            status_text.text(f"Embedding {uploaded_file.name}: {i+1}/{len(chunks)}")
                            embedding = get_embedding(chunk, client)
                            if embedding:
                                embeddings.append(embedding)
                            progress_bar.progress((i + 1) / len(chunks))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Store in session state
                        st.session_state.documents[uploaded_file.name] = {
                            'chunks': chunks[:len(embeddings)],  # Only keep chunks with embeddings
                            'embeddings': embeddings,
                            'company': 'Amazon' if 'amazon' in uploaded_file.name.lower() else 
                                      'Apple' if 'apple' in uploaded_file.name.lower() else 'Company'
                        }
                        
                        st.success(f"✅ Processed {uploaded_file.name}: {len(embeddings)} chunks")
    
    # Main chat interface
    if st.session_state.documents:
        st.success(f"📚 {len(st.session_state.documents)} document(s) loaded and ready!")
        
        # Document selector
        selected_doc = st.selectbox(
            "Select a document to analyze:",
            options=list(st.session_state.documents.keys()),
            format_func=lambda x: f"{st.session_state.documents[x]['company']} - {x}"
        )
        
        st.divider()
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Financial Performance"):
                st.session_state.query = "Summarize the financial performance this quarter"
        
        with col2:
            if st.button("⚠️ Risk Factors"):
                st.session_state.query = "What are the main risks mentioned?"
        
        with col3:
            if st.button("💰 Cash Flow"):
                st.session_state.query = "Explain the company's cash flow situation"
        
        # Query input
        query = st.text_input(
            "Ask a question about the 10-Q filing:",
            value=st.session_state.get('query', ''),
            placeholder="e.g., What are the key revenue drivers this quarter?"
        )
        
        if query:
            client = get_openai_client()
            doc_data = st.session_state.documents[selected_doc]
            
            with st.spinner("🔍 Retrieving and analyzing..."):
                # Step 1: Retrieve relevant chunks
                st.markdown("### 🔍 Step 1: Retrieval")
                retrieved = retrieve_relevant_chunks(
                    query,
                    doc_data['chunks'],
                    doc_data['embeddings'],
                    client,
                    top_k=max_chunks
                )
                
                st.info(f"Retrieved {len(retrieved)} chunks using embedding similarity")
                
                # Step 2: Re-rank (optional)
                if use_reranking:
                    st.markdown("### 🎯 Step 2: Re-Ranking")
                    reranked = rerank_chunks(query, retrieved, client)
                    st.info(f"Re-ranked {len(reranked)} chunks using semantic scoring")
                    final_chunks = reranked
                else:
                    final_chunks = retrieved
                
                # Step 3: Generate answer
                st.markdown("### 💬 Step 3: Generation")
                answer = generate_answer(query, final_chunks, client, doc_data['company'])
                
            # Display results
            st.markdown("---")
            st.markdown("## 📝 Analysis Results")
            st.markdown(answer)
            
            # Show sources
            st.markdown("---")
            st.markdown("## 📚 Sources Used")
            
            for i, chunk in enumerate(final_chunks[:3]):
                with st.expander(f"Source {i+1} - Relevance: {chunk.get('rerank_score', chunk['similarity']*10):.2f}/10"):
                    st.markdown(f"**Embedding Similarity:** {chunk['similarity']:.4f}")
                    if use_reranking:
                        st.markdown(f"**Re-Rank Score:** {chunk['rerank_score']:.2f}/10")
                    st.markdown(f"**Chunk Index:** {chunk['index']}")
                    st.text_area(
                        "Content:",
                        chunk['chunk'][:1000] + "..." if len(chunk['chunk']) > 1000 else chunk['chunk'],
                        height=200,
                        key=f"source_{i}"
                    )
    else:
        st.info("👆 Please upload SEC 10-Q PDF files in the sidebar to get started!")
        
        st.markdown("""
        ### 📚 About This Chatbot
        
        This chatbot demonstrates **Retrieval-Augmented Generation (RAG)** with **Re-Ranking**:
        
        #### How it works:
        
        1. **Document Processing**: PDFs are split into chunks and converted to embeddings
        2. **Retrieval**: When you ask a question, the system finds relevant chunks using cosine similarity
        3. **Re-Ranking**: Chunks are re-scored using an LLM for semantic relevance
        4. **Generation**: Top chunks are used as context for the LLM to generate an answer
        
        #### Why Re-Ranking?
        
        - Embedding similarity (cosine) is fast but not always semantically perfect
        - Re-ranking uses an LLM to better understand query intent and context
        - Improves answer quality by prioritizing truly relevant information
        
        #### Try It:
        
        Upload Amazon and Apple 10-Q filings and compare their financial performance!
        """)

if __name__ == "__main__":
    main()