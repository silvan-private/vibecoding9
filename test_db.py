import sqlite3
import os
from pathlib import Path

def test_database_creation():
    """Test basic database creation and structure"""
    db_path = "speaker_knowledge.db"
    
    print("\nTesting SQLite database creation...")
    
    # Create database connection
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        
        # Create speakers table
        c.execute('''
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                reference_url TEXT,
                reference_start_time REAL,
                reference_end_time REAL,
                voice_print_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create segments table
        c.execute('''
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY,
                speaker_id INTEGER,
                source_url TEXT,
                start_time REAL,
                end_time REAL,
                text TEXT,
                match_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id)
            )
        ''')
        
        # Commit changes
        conn.commit()
    
    # Show database info
    db_path = os.path.abspath(db_path)
    print(f"\nDatabase file created at: {db_path}")
    print(f"Database file size: {os.path.getsize(db_path)} bytes")
    
    # Show database structure
    print("\nDatabase structure:")
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        
        # List all tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            
            # Get column info
            c.execute(f"PRAGMA table_info({table_name})")
            columns = c.fetchall()
            
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
    
    print("\n✓ Database created successfully!")
    print("You can find the database file at:", db_path)

if __name__ == "__main__":
    test_database_creation() 