"""
Standalone Streamlit UI for RAG Digest Dashboard.
This file contains all necessary logic, including a mocked database and
digest generation, to run independently.

To run this file:
1. Save it as my_rag_dashboard.py
2. Run in your terminal: streamlit run my_rag_dashboard.py
"""
import streamlit as st
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# --- Configuration and Setup ---
st.set_page_config(layout="wide", page_title="My Standalone RAG Dashboard", initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Keywords (normally loaded from config.json)
ALL_KEYWORDS = ["RAG", "LLM", "Knowledge Graph", "Vector DB", "Fine-Tuning", "Transformer"]

# --- Database Mock (Replaces database_manager.py) ---

class MockDatabaseManager:
    """Mocks the core database manager for standalone operation."""
    def __init__(self, *args, **kwargs):
        logging.warning("Using MockDatabaseManager for standalone operation.")
        self._mock_papers = self._initialize_mock_papers()
        
    def _initialize_mock_papers(self) -> List[Dict[str, Any]]:
        """Creates a fixed set of mock papers for demonstration."""
        now = datetime.now()
        
        # Paper 1: Highly relevant to RAG and Knowledge Graph (high score potential)
        paper_1 = {
            'arxiv_id': '2407.001A',
            'title': 'The Fusion of RAG and Knowledge Graphs for Factual Accuracy',
            'abstract': 'Investigating hybrid RAG systems that utilize structured knowledge graphs to filter noisy retrieved chunks.',
            'authors': json.dumps(['Dr. Elana K.', 'Amaan Z.']), 
            'categories': json.dumps(['cs.CL', 'cs.AI']),
            'pdf_url': 'https://arxiv.org/pdf/2407.001A',
            'published_date': (now - timedelta(days=2)).isoformat(),
            'summary': json.dumps({
                'key_insights': 'Knowledge Graph integration reduced context noise by 40% in zero-shot QA.',
                'methodology': 'Used Graph Neural Networks to score retrieved chunks before LLM input.',
                'results': 'Achieved state-of-the-art results on fact-checking benchmarks.',
            }),
            'rank_score': 0 
        }
        
        # Paper 2: Focused on LLM and Fine-Tuning
        paper_2 = {
            'arxiv_id': '2407.002B',
            'title': 'Parameter-Efficient Fine-Tuning for Domain-Specific LLMs',
            'abstract': 'A study on LoRA and QLoRA techniques for adapting large language models to medical data.',
            'authors': json.dumps(['Dr. Nikita T.']), 
            'categories': json.dumps(['cs.LG']),
            'pdf_url': 'https://arxiv.org/pdf/2407.002B',
            'published_date': (now - timedelta(days=5)).isoformat(),
            'summary': json.dumps({
                'key_insights': 'QLoRA proved most efficient, reducing training time by 60% with minimal performance loss.',
                'methodology': 'Compared various PEFT techniques across three different LLM architectures.',
                'results': 'Maintained 98% of full fine-tuning accuracy.',
            }),
            'rank_score': 0
        }

        # Paper 3: General RAG, less relevant to specific preferences (low score potential)
        paper_3 = {
            'arxiv_id': '2407.003C',
            'title': 'Optimizing Document Loading in Vector Databases',
            'abstract': 'Improving throughput of document ingestion into vector databases using parallel processing.',
            'authors': json.dumps(['Dr. Jane Doe']), 
            'categories': json.dumps(['cs.DB']),
            'pdf_url': 'https://arxiv.org/pdf/2407.003C',
            'published_date': (now - timedelta(days=9)).isoformat(), # Outside weekly window
            'summary': json.dumps({
                'key_insights': 'A 5x speed improvement in ingestion achieved using Ray.',
                'methodology': 'Benchmarking different parallelization frameworks.',
                'results': 'Focus on infrastructure, not RAG-specific retrieval mechanics.',
            }),
            'rank_score': 0
        }
        
        return [paper_1, paper_2, paper_3]

    def get_papers_for_digest(self, start_date_str: str, end_date_str: str) -> List[Dict[str, Any]]:
        """Mocks fetching papers within a date range."""
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
        
        # Filter mock papers based on the date range
        filtered_papers = []
        for paper in self._mock_papers:
            pub_date = datetime.fromisoformat(paper['published_date'])
            if start_date <= pub_date <= end_date:
                filtered_papers.append(paper)
        
        return filtered_papers
        
    def get_stats(self) -> Dict:
        """Mocks pipeline statistics."""
        return {
            'total_papers': 150, 
            'processed_papers': len(self._mock_papers), 
            'papers_with_embeddings': len(self._mock_papers), 
            'total_chunks': 500
        }
            
db = MockDatabaseManager() # Initialize the mock database

# --- Email Digest Bot (Replaces email_digest_bot.py) ---

class EmailDigestBot:
    """
    Generates personalized weekly RAG research digests.
    """
    def __init__(self, db_instance: MockDatabaseManager):
        self.db = db_instance

    def _rank_papers(self, papers: List[Dict[str, Any]], preferences: List[str]) -> List[Dict[str, Any]]:
        """Ranks papers based on keyword matching."""
        if not papers: return []
            
        ranked_papers = []
        preference_set = {p.lower() for p in preferences}
        
        for paper in papers:
            rank_score = 0
            
            parsed_paper = self._parse_paper_json_fields(paper)
            
            # Combine relevant fields for scoring
            text_to_score = f"{parsed_paper.get('title', '')} {parsed_paper.get('abstract', '')}"
            summary = parsed_paper.get('summary', {})
            for value in summary.values():
                 if isinstance(value, str):
                    text_to_score += f" {value}"
            
            # Check for preference keywords
            for pref in preference_set:
                if pref in text_to_score.lower():
                    rank_score += 1
            
            parsed_paper['rank_score'] = rank_score
            ranked_papers.append(parsed_paper)

        # Sort by rank score (descending)
        ranked_papers.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        return ranked_papers

    def _parse_paper_json_fields(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Parses JSON string fields from the mock data."""
        parsed_paper = dict(paper) 
        for field in ['authors', 'categories', 'summary']:
            value = parsed_paper.get(field)
            if isinstance(value, str):
                try:
                    parsed_paper[field] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_paper[field] = [] if field in ['authors', 'categories'] else {}
            elif value is None:
                parsed_paper[field] = [] if field in ['authors', 'categories'] else {}
                
        return parsed_paper

    def _generate_paper_card_html(self, paper: Dict[str, Any]) -> str:
        """Generates the HTML snippet for a single paper card."""
        authors = paper.get('authors', [])
        summary = paper.get('summary', {})
        key_insights = summary.get('key_insights', paper.get('abstract', 'No key insights provided.'))
        rank_score_display = f"{paper.get('rank_score', 0) * 10} / 10" # Scale rank score for display

        return f"""
        <!-- Individual Paper Card -->
        <div style="
            background-color: #ffffff; 
            padding: 20px; 
            margin-bottom: 25px; 
            border-radius: 8px; 
            border-top: 5px solid #2980b9; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        ">
            <h3 style="
                font-family: 'Inter', sans-serif; 
                color: #2c3e50; 
                margin-top: 0; 
                margin-bottom: 8px; 
                font-size: 1.35em;
            ">
                <a href="{paper.get('pdf_url', '#')}" target="_blank" style="color: #2980b9; text-decoration: none; font-weight: 700;">
                    {paper.get('title', 'Untitled Paper')}
                </a>
            </h3>
            <p style="
                font-family: 'Inter', sans-serif; 
                color: #7f8c8d; 
                font-size: 0.9em; 
                margin-bottom: 15px;
            ">
                By: {', '.join(authors)} | **Relevance Score:** <span style="color: #f39c12; font-weight: bold;">{rank_score_display}</span>
            </p>
            
            <div style="border-left: 4px solid #27ae60; padding-left: 15px;">
                <p style="
                    font-family: 'Inter', sans-serif; 
                    color: #34495e; 
                    font-size: 1.0em; 
                    line-height: 1.5;
                    margin: 0;
                ">
                    <strong>Key Insight:</strong> {key_insights}
                </p>
            </div>
            
            <p style="
                font-family: 'Inter', sans-serif; 
                font-size: 0.9em; 
                margin-top: 15px;
            ">
                <a href="{paper.get('pdf_url', '#')}" target="_blank" style="
                    display: inline-block;
                    padding: 8px 15px;
                    background-color: #2980b9;
                    color: white;
                    text-decoration: none; 
                    font-weight: bold;
                    border-radius: 4px;
                ">
                    View Full PDF &rarr;
                </a>
            </p>
        </div>
        """

    def generate_digest_html(self, preferences: List[str]) -> str:
        """Generates the personalized HTML digest."""
        
        # 1. Define the weekly window (last 7 days)
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)
        start_date_str = one_week_ago.isoformat()
        end_date_str = today.isoformat()

        # 2. Fetch processed papers from the mock database
        papers = self.db.get_papers_for_digest(start_date_str, end_date_str)
        
        if not papers:
            return self._generate_empty_digest_html()
            
        # 3. Personalize and rank the papers
        ranked_papers = self._rank_papers(papers, preferences)
        
        # Show the top 5 relevant papers for a digest email
        top_papers = ranked_papers[:5]
        
        # 4. Generate the HTML body
        paper_cards_html = "".join(
            self._generate_paper_card_html(p) for p in top_papers
        )
        
        digest_date = today.strftime('%B %d, %Y')
        
        # 5. Full HTML Template
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG Digest - Weekly Research Update</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
                body, table, td, div, p {{ 
                    font-family: 'Inter', sans-serif; 
                    font-size: 16px;
                    line-height: 1.6;
                    color: #34495e;
                }}
                .container {{
                    max-width: 650px;
                    margin: 20px auto;
                    padding: 25px;
                    background-color: #ecf0f1;
                    border-radius: 12px;
                    box-shadow: 0 0 30px rgba(0, 0, 0, 0.15);
                }}
                h1 {{ 
                    color: #2c3e50; 
                    font-weight: 700; 
                    font-size: 2.0em; 
                    border-bottom: 4px solid #f39c12;
                    padding-bottom: 12px;
                    margin-top: 0;
                }}
                .header-meta {{ color: #7f8c8d; font-size: 0.95em; }}
                @media only screen and (max-width: 680px) {{
                    .container {{
                        width: 95% !important;
                        margin: 10px auto !important;
                        padding: 15px !important;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>
                    <span style="color: #2980b9;">🤖</span> RAG Digest: Your Weekly Research Update
                </h1>
                <p class="header-meta">
                    Delivered on: **{digest_date}** | Prioritized Focus: **{', '.join(preferences) if preferences else 'No Specific Filter'}**
                </p>
                
                <p style="margin-bottom: 20px;">
                    Greetings! Here are the **Top {len(top_papers)}** most relevant Retrieval-Augmented Generation (RAG) and LLM papers processed by our pipeline this week, tailored to your interests.
                </p>
                
                {paper_cards_html}

                <!-- Footer -->
                <div style="text-align: center; margin-top: 40px; padding-top: 25px; border-top: 1px solid #bdc3c7;">
                    <p style="color: #7f8c8d; font-size: 0.85em; margin-bottom: 5px;">
                        This automated digest helps you stay ahead of the curve.
                    </p>
                    <p style="color: #7f8c8d; font-size: 0.85em; margin: 0;">
                        Manage preferences and dive deeper into all indexed papers in the full Streamlit Dashboard.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template

    def _generate_empty_digest_html(self) -> str:
        """Fallback for when no papers are found within the time window."""
        return """
        <div style="
            font-family: 'Inter', sans-serif; 
            max-width: 600px; 
            margin: 20px auto; 
            padding: 40px; 
            background-color: #fff0f0; 
            text-align: center; 
            border-radius: 10px;
            border: 1px solid #e74c3c;
        ">
            <h1 style="color: #e74c3c;">No Digest Content</h1>
            <p style="color: #e74c3c; font-size: 1.1em; font-weight: bold;">
                ⚠️ No new relevant RAG/LLM papers were found or processed this week. 
            </p>
            <p style="color: #34495e;">
                Check back next week! The paper pipeline may be currently running or there were no recent arXiv submissions matching the criteria.
            </p>
        </div>
        """
        
bot = EmailDigestBot(db_instance=db) # Initialize the standalone bot

# --- Streamlit UI Core Logic ---

def display_dashboard():
    """Renders the main Streamlit interface."""
    
    st.title("🚀 My Standalone RAG Research Dashboard")
    st.markdown("A completely self-contained Streamlit application for research tracking and digest generation.")
    
    st.markdown("""
        <style>
            /* Custom Streamlit component styling */
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 1.1rem;
            }
            .stButton>button {
                background-color: #2980b9;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: #2c3e50;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # 1. Status and Stats
    stats = db.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers Indexed", stats.get('total_papers', 0))
    col2.metric("Papers Processed", stats.get('processed_papers', 0))
    col3.metric("Papers with Embeddings", stats.get('papers_with_embeddings', 0))
    col4.metric("Total Chunks", stats.get('total_chunks', 0))

    st.markdown("---")

    # 2. Main Tabs
    tab1, tab2, tab3 = st.tabs(["📧 Digest Preview", "⚙️ User Preferences", "📚 Paper Browser"])

    with tab2:
        # --- User Preferences Management ---
        st.header("Manage Research Interests")
        st.write("Select keywords to prioritize the papers featured in your weekly RAG Digest.")

        # Initialize session state for preferences if it doesn't exist
        if 'user_preferences' not in st.session_state:
            st.session_state['user_preferences'] = ['RAG', 'Knowledge Graph'] 

        selected_preferences = st.multiselect(
            'Your Focus Areas (Keywords are matched against Title, Abstract, and Summary)',
            options=ALL_KEYWORDS,
            default=st.session_state['user_preferences'],
            key='preference_selector'
        )

        if st.button('Save Preferences', key='save_prefs'):
            st.session_state['user_preferences'] = selected_preferences
            st.success(f"Preferences saved! Digest will now prioritize: **{', '.join(selected_preferences) or 'All Topics'}**")
            
    with tab1:
        # --- Weekly Email Digest Preview ---
        st.header("Weekly Digest Preview")
        st.write("Generate and view the HTML email content based on your current preferences and the latest mock papers.")

        current_preferences = st.session_state.get('user_preferences', [])
        
        st.info(f"Digest is generated using keywords: **{', '.join(current_preferences) or 'None (Showing top ranked available papers)'}**")
        
        if st.button('Generate & Preview Personalized Digest', key='generate_preview'):
            with st.spinner('Generating HTML digest...'):
                html_digest = bot.generate_digest_html(current_preferences)
            
            st.subheader("HTML Email Preview (Top Ranked Papers)")
            st.components.v1.html(
                html_digest, 
                height=700, 
                scrolling=True
            )
            
            with st.expander("View Raw HTML Source"):
                st.code(html_digest, language='html') 

    with tab3:
        # --- Basic Paper Browser ---
        st.header("All Processed Papers (Mock Data)")
        st.write("Browse all mock RAG/LLM papers that have been fetched and processed with summaries.")
        
        # Fetch papers without date constraint for the mock browser display
        # We use a wide date range to ensure all mock papers are included for display
        papers_list = db.get_papers_for_digest("1970-01-01", datetime.now().isoformat())
        
        if papers_list:
            display_data = []
            # We use the bot's ranking and parsing logic for a better display
            ranked_papers = bot._rank_papers(papers_list, st.session_state.get('user_preferences', []))
            
            # Use st.column_config.TextColumn if available, but avoid MarkdownColumn
            column_config_available = hasattr(st.column_config, 'TextColumn')
            
            for parsed_p in ranked_papers:
                abstract_text = parsed_p['abstract']
                key_insight = parsed_p.get('summary', {}).get('key_insights', abstract_text)
                
                display_data.append({
                    'ID': parsed_p['arxiv_id'],
                    'Title': parsed_p['title'],
                    'Relevance Score': parsed_p['rank_score'],
                    'Authors': ', '.join(parsed_p.get('authors', [])),
                    'Key Insight': key_insight[:150] + '...' if len(key_insight) > 150 else key_insight,
                    'Date': parsed_p['published_date'].split('T')[0],
                    'Link': f"[PDF]({parsed_p['pdf_url']})", # Markdown link string
                })

            if column_config_available:
                # Use column_config only if TextColumn (introduced at 1.25.0 too, but safer than MarkdownColumn) is present
                st.dataframe(display_data,
                    column_config={
                        # We use TextColumn for the Key Insight to add a help tooltip
                        "Key Insight": st.column_config.TextColumn("Key Insight", help="Summary Insight from Fine-Tuned LLM"),
                    },
                    hide_index=True)
            else:
                # Fallback for very old versions: remove column_config entirely
                st.dataframe(display_data, hide_index=True)
                st.warning("Note: Your Streamlit version may not support advanced column configurations.")
        else:
            st.info("No mock papers loaded. This indicates an issue with the MockDatabaseManager initialization.")


if __name__ == "__main__":
    display_dashboard()