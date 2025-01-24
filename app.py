from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from Libraries.pdf_processor import PDFProcessor
from Libraries.db_manager import DBManager
from Libraries.rag_manager import RAGManager
import shutil
import sqlite3
import openai
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STORAGE_FOLDER'] = 'storage'
app.config['INDICES_FOLDER'] = 'storage/indices'

# Ensure required folders exist with proper permissions
for folder in [app.config['UPLOAD_FOLDER'], app.config['STORAGE_FOLDER'], app.config['INDICES_FOLDER']]:
    try:
        os.makedirs(folder, exist_ok=True)
        # Ensure directory has write permissions
        os.chmod(folder, 0o755)
        logger.info(f"Directory {folder} is ready")
    except Exception as e:
        logger.error(f"Error setting up directory {folder}: {str(e)}")

ALLOWED_EXTENSIONS = {'pdf'}
db_manager = DBManager()

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key:
            return jsonify({'error': 'No API key provided'}), 401
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@require_api_key
def chat():
    try:
        api_key = request.headers.get('X-API-KEY')
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        # Get document ID if provided
        doc_id = data.get('doc_id')
        
        if doc_id:
            # Use RAG for document-specific queries
            rag_manager = RAGManager(api_key)
            response = rag_manager.query_document(doc_id, data['message'])
            if response:
                return jsonify({'success': True, 'response': response})
            else:
                return jsonify({'error': 'Failed to query document'}), 500
        else:
            # Use regular OpenAI chat for general queries
            openai.api_key = api_key
            messages = [
                {"role": "system", "content": "You are a helpful assistant that helps users understand PDF documents."}
            ]
            messages.append({"role": "user", "content": data['message']})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            return jsonify({'success': True, 'response': assistant_response})
        
    except openai.error.AuthenticationError:
        return jsonify({'error': 'Invalid API key'}), 401
    except openai.error.RateLimitError:
        return jsonify({'error': 'Rate limit exceeded'}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Upload endpoint called")
    if 'file' not in request.files:
        logger.error("No file in request")
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'success': False, 'error': 'Invalid file type. Only PDF files are allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Saving file to {filepath}")
        file.save(filepath)
        
        if not PDFProcessor.is_valid_pdf(filepath):
            logger.error(f"Invalid PDF file: {filename}")
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Invalid or corrupted PDF file'}), 400
        
        # Process the PDF
        logger.debug(f"Processing PDF: {filename}")
        result = PDFProcessor.extract_text(filepath)
        
        # Clean up temporary file
        os.remove(filepath)
        
        if not result['success']:
            logger.error(f"PDF processing failed: {result.get('error', 'Unknown error')}")
            return jsonify({'success': False, 'error': result['error']}), 400
            
        logger.debug("PDF processed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.exception("Error in upload_file")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    logger.debug("Save PDF endpoint called")
    if 'file' not in request.files:
        logger.error("No file in request")
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.debug(f"Saving file temporarily to {temp_path}")
        file.save(temp_path)
        
        # Process the PDF
        logger.debug(f"Processing PDF: {filename}")
        result = PDFProcessor.extract_text(temp_path)
        if not result['success']:
            logger.error(f"PDF processing failed: {result.get('error', 'Unknown error')}")
            os.remove(temp_path)
            return jsonify({'success': False, 'error': result['error']}), 400
        
        # Save to permanent storage
        storage_path = os.path.join(app.config['STORAGE_FOLDER'], filename)
        logger.debug(f"Moving file to permanent storage: {storage_path}")
        shutil.move(temp_path, storage_path)
        
        # Save to database
        logger.debug("Saving to database")
        pdf_id = db_manager.save_pdf(filename, storage_path, result)
        
        # Add id to result
        result['id'] = pdf_id
        result['is_indexed'] = False
        
        logger.debug("PDF saved successfully")
        return jsonify({
            'success': True,
            'id': pdf_id,
            'pdfUrl': f'/pdf/{pdf_id}',
            'result': result
        })
        
    except Exception as e:
        logger.exception("Error in save_pdf")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/pdf/<pdf_id>')
def serve_pdf(pdf_id):
    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM pdfs WHERE id = ?', (pdf_id,))
            row = cursor.fetchone()
            if row:
                return send_file(row[0], mimetype='application/pdf')
        return 'PDF not found', 404
    except Exception as e:
        return str(e), 500

@app.route('/history')
def get_history():
    try:
        history = db_manager.get_history()
        if not history:
            return jsonify([])
            
        return jsonify([{
            'id': item['id'],
            'name': item['name'],
            'timestamp': item['timestamp'],
            'result': item['result'],
            'pdfUrl': f'/pdf/{item["id"]}',
            'is_indexed': item.get('is_indexed', False)  # Add indexing status
        } for item in history])
        
    except sqlite3.Error as e:
        print(f"Database error in get_history: {str(e)}")
        return jsonify({'success': False, 'error': 'Database error occurred'}), 500
    except Exception as e:
        print(f"Unexpected error in get_history: {str(e)}")
        return jsonify({'success': False, 'error': 'An unexpected error occurred'}), 500

@app.route('/remove_pdf/<pdf_id>', methods=['DELETE'])
def remove_pdf(pdf_id):
    try:
        db_manager.remove_pdf(pdf_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        db_manager.clear_history()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/index_document/<pdf_id>', methods=['POST'])
@require_api_key
def index_document(pdf_id):
    try:
        api_key = request.headers.get('X-API-KEY')
        
        # Get PDF path
        pdf_path = db_manager.get_pdf_path(pdf_id)
        if not pdf_path:
            return jsonify({'error': 'PDF not found'}), 404
        
        # Create RAG manager and index document
        rag_manager = RAGManager(api_key)
        if rag_manager.index_document(pdf_path, pdf_id):
            # Update indexing status in database
            db_manager.update_index_status(pdf_id, True)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Failed to index document'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_index/<pdf_id>', methods=['GET'])
@require_api_key
def check_index(pdf_id):
    try:
        api_key = request.headers.get('X-API-KEY')
        rag_manager = RAGManager(api_key)
        is_indexed = rag_manager.is_indexed(pdf_id)
        return jsonify({'success': True, 'is_indexed': is_indexed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File is too large. Maximum size is 16MB'}), 413

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug=False) 