#!/usr/bin/env python3
"""
SQLite Database Initialization Script for Intellichat
This script initializes the SQLite database with proper schema and indexes.
"""

import sqlite3
import os
import sys

def init_database():
    """Initialize SQLite database with required tables."""
    db_path = 'intellichat.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_text TEXT,
                keywords TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        ''')
        
        # Create chat_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_keywords ON chunks(keywords)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_document_id ON chat_history(document_id)')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized successfully: {db_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        return False

def check_database():
    """Check if database exists and is accessible."""
    db_path = 'intellichat.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        expected_tables = ['documents', 'chunks', 'chat_history']
        existing_tables = [table[0] for table in tables]
        
        missing_tables = [table for table in expected_tables if table not in existing_tables]
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        conn.close()
        print(f"✅ Database is accessible and properly configured")
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Intellichat SQLite Database Initialization")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Check existing database
        check_database()
    else:
        # Initialize new database
        success = init_database()
        if success:
            print("\n🎉 Database setup complete!")
            print("You can now run: streamlit run app.py")
        else:
            print("\n❌ Database setup failed!")
            sys.exit(1)
