import sqlite3
import json
import os
from datetime import datetime
import uuid
import numpy as np

class NaNEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)

class DBManager:
    def __init__(self, db_path="database.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Drop existing table if it exists
                cursor.execute('DROP TABLE IF EXISTS pdfs')
                
                # Create table with proper schema
                cursor.execute('''
                    CREATE TABLE pdfs (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        is_indexed BOOLEAN DEFAULT FALSE
                    )
                ''')
                conn.commit()
                print("Database initialized successfully")
                
        except sqlite3.Error as e:
            print(f"Error initializing database: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error initializing database: {str(e)}")
            raise

    def save_pdf(self, name, file_path, metadata):
        """Save PDF information to database."""
        pdf_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Use custom encoder to handle NaN values
            metadata_json = json.dumps(metadata, cls=NaNEncoder)
            cursor.execute(
                'INSERT INTO pdfs (id, name, timestamp, file_path, metadata, is_indexed) VALUES (?, ?, ?, ?, ?, ?)',
                (pdf_id, name, datetime.now().isoformat(), file_path, metadata_json, False)
            )
            conn.commit()
        return pdf_id

    def get_history(self):
        """Get all PDF history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM pdfs ORDER BY timestamp DESC LIMIT 10')
                rows = cursor.fetchall()
                
                if not rows:
                    return []
                    
                return [{
                    'id': row[0],
                    'name': row[1],
                    'timestamp': row[2],
                    'file_path': row[3],
                    'result': json.loads(row[4]) if row[4] else {},
                    'is_indexed': bool(row[5]) if len(row) > 5 else False
                } for row in rows]
                
        except sqlite3.Error as e:
            print(f"Database error in get_history: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON decode error in get_history: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error in get_history: {str(e)}")
            raise

    def remove_pdf(self, pdf_id):
        """Remove PDF from database and file system."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM pdfs WHERE id = ?', (pdf_id,))
            row = cursor.fetchone()
            if row:
                file_path = row[0]
                if os.path.exists(file_path):
                    os.remove(file_path)
                cursor.execute('DELETE FROM pdfs WHERE id = ?', (pdf_id,))
                conn.commit()

    def clear_history(self):
        """Clear all history and remove PDF files."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM pdfs')
            rows = cursor.fetchall()
            for row in rows:
                file_path = row[0]
                if os.path.exists(file_path):
                    os.remove(file_path)
            cursor.execute('DELETE FROM pdfs')
            conn.commit()

    def update_index_status(self, pdf_id: str, is_indexed: bool):
        """Update the indexing status of a PDF."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE pdfs SET is_indexed = ? WHERE id = ?',
                (is_indexed, pdf_id)
            )
            conn.commit()

    def get_pdf_path(self, pdf_id: str) -> str:
        """Get the file path for a PDF by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM pdfs WHERE id = ?', (pdf_id,))
            row = cursor.fetchone()
            return row[0] if row else None 