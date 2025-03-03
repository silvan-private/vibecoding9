"""Speaker database management module."""
import sqlite3
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class SpeakerDatabase:
    """Manages speaker voice prints and transcribed segments."""
    
    def __init__(self, db_path: str):
        """Initialize database connection and tables.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(db_path)
        # Enable foreign key support
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()
        logger.info(f"Initialized database at {db_path}")
        
    def _create_tables(self):
        """Initialize database tables if they don't exist."""
        with self.conn:
            # Create speakers table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS speakers (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    voiceprint BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create segments table with foreign key constraint
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY,
                    speaker_id INTEGER NOT NULL,
                    source_url TEXT,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
                )
            """)
            
            # Create indices for better query performance
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_speaker_name 
                ON speakers(name)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_segment_speaker 
                ON segments(speaker_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_segment_time 
                ON segments(start, end)
            """)
    
    def add_speaker(self, name: str, voiceprint: Optional[bytes] = None) -> int:
        """Add a new speaker to the database.
        
        Args:
            name: Unique name for the speaker
            voiceprint: Optional binary voice print data
            
        Returns:
            int: ID of the newly created speaker
            
        Raises:
            sqlite3.IntegrityError: If speaker name already exists
        """
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO speakers (name, voiceprint) VALUES (?, ?)",
                (name, voiceprint)
            )
            return cursor.lastrowid
    
    def add_segment(self, speaker_id: int, start: float, end: float, 
                   text: str, source_url: Optional[str] = None,
                   confidence: float = 1.0) -> int:
        """Add a transcribed segment for a speaker.
        
        Args:
            speaker_id: ID of the speaker
            start: Start time in seconds
            end: End time in seconds
            text: Transcribed text
            source_url: Optional source URL
            confidence: Confidence score (0-1)
            
        Returns:
            int: ID of the newly created segment
            
        Raises:
            sqlite3.IntegrityError: If speaker_id doesn't exist
        """
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO segments 
                (speaker_id, start, end, text, source_url, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (speaker_id, start, end, text, source_url, confidence))
            return cursor.lastrowid
    
    def get_speaker(self, speaker_id: int) -> Tuple[int, str, Optional[bytes]]:
        """Get speaker information by ID.
        
        Args:
            speaker_id: ID of the speaker to retrieve
            
        Returns:
            Tuple of (id, name, voiceprint)
            
        Raises:
            sqlite3.Error: If speaker not found
        """
        cursor = self.conn.execute(
            "SELECT id, name, voiceprint FROM speakers WHERE id = ?",
            (speaker_id,)
        )
        result = cursor.fetchone()
        if not result:
            raise sqlite3.Error(f"Speaker {speaker_id} not found")
        return result
    
    def get_segments(self, speaker_id: Optional[int] = None) -> List[tuple]:
        """Get transcribed segments, optionally filtered by speaker.
        
        Args:
            speaker_id: Optional speaker ID to filter by
            
        Returns:
            List of (id, speaker_id, start, end, text, confidence) tuples
        """
        query = """
            SELECT id, speaker_id, start, end, text, confidence 
            FROM segments
        """
        params = []
        if speaker_id is not None:
            query += " WHERE speaker_id = ?"
            params.append(speaker_id)
        query += " ORDER BY start"
        
        return self.conn.execute(query, params).fetchall()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def verify_database(db_path: str):
    """Verify database structure and basic functionality.
    
    Args:
        db_path: Path to database file to verify
        
    Raises:
        AssertionError: If verification fails
    """
    try:
        # Create/connect to database
        db = SpeakerDatabase(db_path)
        
        # Check tables exist
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        
        assert 'speakers' in table_names, "speakers table not found"
        assert 'segments' in table_names, "segments table not found"
        
        # Verify table structure
        speaker_cols = db.conn.execute("PRAGMA table_info(speakers)").fetchall()
        segment_cols = db.conn.execute("PRAGMA table_info(segments)").fetchall()
        
        assert len(speaker_cols) >= 3, "speakers table missing columns"
        assert len(segment_cols) >= 6, "segments table missing columns"
        
        # Test basic operations
        test_speaker_id = db.add_speaker("Test Speaker")
        assert test_speaker_id > 0, "failed to add speaker"
        
        test_segment_id = db.add_segment(
            test_speaker_id, 0.0, 1.0, "Test text"
        )
        assert test_segment_id > 0, "failed to add segment"
        
        # Clean up test data
        with db.conn:
            db.conn.execute("DELETE FROM segments WHERE id = ?", 
                          (test_segment_id,))
            db.conn.execute("DELETE FROM speakers WHERE id = ?", 
                          (test_speaker_id,))
        
        db.close()
        logger.info(f"Successfully verified database at {db_path}")
        
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        raise 