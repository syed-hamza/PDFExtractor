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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pdfs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    is_indexed BOOLEAN DEFAULT FALSE
                )
            ''')
            conn.commit()

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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM pdfs ORDER BY timestamp DESC LIMIT 10')
            rows = cursor.fetchall()
            return [{
                'id': row[0],
                'name': row[1],
                'timestamp': row[2],
                'file_path': row[3],
                'result': json.loads(row[4]),
                'is_indexed': bool(row[5])
            } for row in rows]

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