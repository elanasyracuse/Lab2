import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Core database manager for all RAG bot data"""
    
    def __init__(self, db_path="./data/ragbot.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _create_tables(self):
        """Create all necessary tables"""
        
        # Main papers table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            authors TEXT,  -- JSON array stored as text
            published_date DATETIME,
            categories TEXT,  -- JSON array
            pdf_url TEXT,
            pdf_downloaded BOOLEAN DEFAULT 0,
            full_text TEXT,
            sections TEXT,  -- JSON object
            processed BOOLEAN DEFAULT 0,
            embedding_created BOOLEAN DEFAULT 0,
            summary_generated BOOLEAN DEFAULT 0,
            fetched_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Embeddings table
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT,
            embedding BLOB,  -- Stored as a pickled NumPy array
            chunk_type TEXT, -- e.g., 'intro', 'section_1', 'full_text'
            FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
        )
        """)
        
        # Pipeline logs
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time DATETIME,
            end_time DATETIME,
            papers_fetched INTEGER,
            papers_processed INTEGER,
            status TEXT, -- 'SUCCESS', 'FAILED', 'PARTIAL'
            error_message TEXT
        )
        """)
        
        # NEW: Users/Subscribers table for the email digest
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            preferences TEXT, -- JSON array of keywords
            is_active BOOLEAN DEFAULT 1,
            join_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.commit()

    # --- Existing Paper Methods (omitted for brevity, assume they are still here) ---

    def insert_paper(self, paper_data: Dict) -> None:
        """Insert or ignore a new paper."""
        try:
            # ... existing insert logic ...
            pass
        except sqlite3.IntegrityError:
            logger.info(f"Paper already exists: {paper_data.get('arxiv_id')}")
    
    def get_paper(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch a single paper by its ID."""
        # ... existing get_paper logic ...
        return None 
    
    def update_paper_content(self, arxiv_id: str, full_text: str, sections: Dict) -> None:
        """Update a paper with parsed text and sections."""
        # ... existing update_paper_content logic ...
        pass

    def get_unprocessed_papers(self, limit: int = 50) -> List[Dict]:
        """Fetch papers not yet parsed."""
        # ... existing get_unprocessed_papers logic ...
        return []

    def get_papers_for_digest(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch processed papers within a date range."""
        # This is used by the digest generation
        self.cursor.execute("""
        SELECT * FROM papers WHERE published_date BETWEEN ? AND ? AND processed = 1 AND summary_generated = 1
        """, (start_date, end_date))
        return [dict(row) for row in self.cursor.fetchall()]

    def get_stats(self) -> Dict:
        # ... existing get_stats logic ...
        return {}
        
    # --- NEW User Management Methods ---

    def add_or_update_user(self, email: str, preferences: List[str]):
        """Add a new user or update an existing user's preferences."""
        pref_json = json.dumps(preferences)
        self.cursor.execute("""
        INSERT INTO users (email, preferences, is_active) 
        VALUES (?, ?, 1)
        ON CONFLICT(email) DO UPDATE SET 
            preferences = excluded.preferences, 
            is_active = excluded.is_active
        """, (email, pref_json))
        self.conn.commit()
        logger.info(f"User preferences updated for: {email}")

    def get_all_subscribers(self) -> List[Dict]:
        """Retrieve all active subscribers with their preferences."""
        self.cursor.execute("SELECT email, preferences FROM users WHERE is_active = 1")
        subscribers = []
        for row in self.cursor.fetchall():
            try:
                preferences = json.loads(row['preferences'])
            except json.JSONDecodeError:
                preferences = []
            subscribers.append({'email': row['email'], 'preferences': preferences})
        return subscribers
    
    def close(self):
        """Close database connection."""
        self.conn.close()

# Example usage (for testing the new table)
if __name__ == "__main__":
    manager = DatabaseManager()
    
    # 1. Add some mock users
    manager.add_or_update_user("amaan@example.com", ["RAG", "LLM", "Knowledge Graph"])
    manager.add_or_update_user("jane.doe@example.com", ["Fine-Tuning", "Transformer"])
    
    # 2. Retrieve subscribers
    subscribers = manager.get_all_subscribers()
    print("\nActive Subscribers:")
    for sub in subscribers:
        print(f"- {sub['email']} | Prefs: {sub['preferences']}")
        
    manager.close()