import json
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Mock Keywords (can be loaded from config in a real scenario)
ALL_KEYWORDS = ["RAG", "LLM", "Knowledge Graph", "Vector DB", "Fine-Tuning", "Transformer"]

class EmailDigestBot:
    """
    Generates personalized weekly RAG research digests (HTML content only).
    Relies on a DatabaseManager instance to fetch paper data.
    """
    def __init__(self, db_instance):
        self.db = db_instance

    def _parse_paper_json_fields(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Parses JSON string fields from the mock data (authors, categories, summary)."""
        parsed_paper = dict(paper) 
        for field in ['authors', 'categories', 'summary']:
            value = parsed_paper.get(field)
            if isinstance(value, str):
                try:
                    # In a real system, the DB should return JSON data here
                    parsed_paper[field] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_paper[field] = [] if field in ['authors', 'categories'] else {}
            elif value is None:
                parsed_paper[field] = [] if field in ['authors', 'categories'] else {}
                
        return parsed_paper
        
    def _rank_papers(self, papers: List[Dict[str, Any]], preferences: List[str]) -> List[Dict[str, Any]]:
        """Ranks papers based on keyword matching against user preferences."""
        if not papers: return []
            
        ranked_papers = []
        # Convert preferences to a case-insensitive set for quick lookup
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
            
            # Check for preference keywords in the text
            for pref in preference_set:
                if pref in text_to_score.lower():
                    rank_score += 1
            
            parsed_paper['rank_score'] = rank_score
            ranked_papers.append(parsed_paper)

        # Sort by rank score (descending)
        ranked_papers.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
        return ranked_papers

    def _generate_paper_card_html(self, paper: Dict[str, Any]) -> str:
        """Generates the HTML snippet for a single paper card."""
        authors = paper.get('authors', [])
        summary = paper.get('summary', {})
        key_insights = summary.get('key_insights', paper.get('abstract', 'No key insights provided.'))
        
        max_keywords = len(ALL_KEYWORDS)
        normalized_score = min(100, (paper.get('rank_score', 0) / max_keywords) * 100) if max_keywords > 0 else 0
        rank_score_display = f"{int(normalized_score)}%"

        # Note: Tailwind is not available in emails, so inline CSS is used for styling consistency.
        return f"""
        <div style="
            background-color: #ffffff; 
            padding: 20px; 
            margin-bottom: 25px; 
            border-radius: 8px; 
            border-top: 5px solid #2980b9; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            font-family: 'Inter', sans-serif;
        ">
            <h3 style="
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
                color: #7f8c8d; 
                font-size: 0.9em; 
                margin-bottom: 15px;
            ">
                By: {', '.join(authors)} | **Relevance Score:** <span style="color: #f39c12; font-weight: bold;">{rank_score_display}</span>
            </p>
            
            <div style="border-left: 4px solid #27ae60; padding-left: 15px;">
                <p style="
                    color: #34495e; 
                    font-size: 1.0em; 
                    line-height: 1.5;
                    margin: 0;
                ">
                    <strong>Key Insight:</strong> {key_insights}
                </p>
            </div>
            
            <p style="
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

    def generate_digest_html(self, preferences: List[str], max_papers: int = 5) -> str:
        """Generates the personalized HTML digest for a given user's preferences."""
        
        # 1. Define the weekly window (last 7 days for the demo)
        today = datetime.now()
        one_week_ago = today - timedelta(days=7)
        start_date_str = one_week_ago.isoformat()
        end_date_str = today.isoformat()

        # 2. Fetch processed papers from the database
        # Note: This relies on the real DatabaseManager now
        papers = self.db.get_papers_for_digest(start_date_str, end_date_str)
        
        if not papers:
            return self._generate_empty_digest_html()
            
        # 3. Personalize and rank the papers
        ranked_papers = self._rank_papers(papers, preferences)
        
        top_papers = ranked_papers[:max_papers]
        
        # 4. Generate the HTML body
        paper_cards_html = "".join(
            self._generate_paper_card_html(p) for p in top_papers
        )
        
        digest_date = today.strftime('%B %d, %Y')
        
        # 5. Full HTML Template (Using inline CSS for email compatibility)
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