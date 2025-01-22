import PyPDF2
from typing import Dict, Any, List
import os
import fitz  # PyMuPDF
import io
import base64
import tabula
import pandas as pd
import numpy as np
import re

def clean_table_data(df):
    """Clean table data by replacing NaN values and converting to native Python types."""
    if isinstance(df, pd.DataFrame):
        df = df.replace({np.nan: None})
        records = df.to_dict('records')
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (np.int64, np.int32)):
                    record[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    record[key] = float(value)
        return records
    return []

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving intentional line breaks and spacing."""
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Preserve paragraph breaks but remove excessive line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Remove spaces at the start of lines
    text = re.sub(r'(?m)^[ \t]+', '', text)
    # Remove spaces at the end of lines
    text = re.sub(r'(?m)[ \t]+$', '', text)
    return text.strip()

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove unnecessary double backslashes while preserving actual line breaks
    text = text.replace('\\\\', '\n')
    
    # Fix common OCR issues
    replacements = {
        'ﬁ': 'fi',
        '−': '-',
        '\u2028': ' ',  # Line separator
        '\u2029': '\n',  # Paragraph separator
        '  ': ' ',      # Double spaces
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        # Skip empty lines
        if line:
            cleaned_lines.append(line)
    
    # Join lines with single newlines
    text = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive newlines
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    
    return text.strip()

def extract_text_from_page(page) -> str:
    """Extract text from a page while preserving layout."""
    text_blocks = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            block_text = []
            for line in block["lines"]:
                line_text = []
                for span in line["spans"]:
                    # Preserve any intentional spacing at the start
                    space_before = ' ' * int(span.get("space_before", 0))
                    line_text.append(space_before + span["text"])
                block_text.append(''.join(line_text))
            # Join lines with appropriate spacing
            text_blocks.append('\n'.join(block_text))
    
    # Join blocks with double newlines to separate paragraphs
    return '\n\n'.join(text_blocks)

def format_math_expressions(text: str) -> str:
    """Format mathematical expressions for better display."""
    # Common math symbols and their LaTeX equivalents
    math_symbols = {
        '±': r'\pm',
        '×': r'\times',
        '÷': r'\div',
        '≤': r'\leq',
        '≥': r'\geq',
        '≠': r'\neq',
        '∞': r'\infty',
        '∑': r'\sum',
        '∏': r'\prod',
        '∫': r'\int',
        '√': r'\sqrt',
        'α': r'\alpha',
        'β': r'\beta',
        'γ': r'\gamma',
        'δ': r'\delta',
        'π': r'\pi',
        'μ': r'\mu',
        'σ': r'\sigma',
        'θ': r'\theta',
        'Δ': r'\Delta',
        '∂': r'\partial',
        'φ': r'\phi',
        'Φ': r'\Phi',
        '→': r'\rightarrow',
        '←': r'\leftarrow',
        '↔': r'\leftrightarrow',
        '⇒': r'\Rightarrow',
        '⇐': r'\Leftarrow',
        '⇔': r'\Leftrightarrow',
    }

    def process_math_content(content: str) -> str:
        """Process content that has been identified as mathematical."""
        # Replace math symbols
        for symbol, latex in math_symbols.items():
            content = content.replace(symbol, latex)
        
        # Handle subscripts and superscripts
        content = re.sub(r'([a-zA-Z])_(\d+|[a-zA-Z])', r'\1_{\2}', content)
        content = re.sub(r'([a-zA-Z\d])(\^)(\d+|[a-zA-Z])', r'\1^{\3}', content)
        
        # Handle fractions
        content = re.sub(r'(\d+)\/(\d+)', r'\\frac{\1}{\2}', content)
        
        # Handle parentheses in equations
        content = re.sub(r'\(([^)]+)\)', r'\\left(\1\\right)', content)
        
        return content

    def is_math_content(text: str) -> bool:
        """Determine if text is likely mathematical content."""
        math_patterns = [
            r'[=+\-*/^_]',  # Basic operators
            r'[α-ωΑ-Ω]',    # Greek letters
            r'\d+\/\d+',    # Fractions
            r'[a-zA-Z]_\d', # Subscripts
            r'[a-zA-Z]\^',  # Superscripts
            r'\\[a-zA-Z]+', # LaTeX commands
        ]
        return any(re.search(pattern, text) for pattern in math_patterns)

    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
            
        if is_math_content(paragraph):
            # Process as math block
            lines = paragraph.split('\n')
            if len(lines) > 1:
                # Multi-line equation
                processed_lines = [process_math_content(line.strip()) for line in lines if line.strip()]
                formatted_paragraphs.append('\\[\n' + ' \\\\ '.join(processed_lines) + '\n\\]')
            else:
                # Single line equation
                processed = process_math_content(paragraph.strip())
                formatted_paragraphs.append('\\(' + processed + '\\)')
        else:
            # Regular text - preserve as is
            formatted_paragraphs.append(paragraph)

    return '\n\n'.join(formatted_paragraphs)

class PDFProcessor:
    @staticmethod
    def extract_text(filepath: str) -> dict:
        """Extract text and metadata from a PDF file."""
        try:
            doc = fitz.open(filepath)
            content = []
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # Extract text with better layout preservation
                text = page.get_text("text")
                text = PDFProcessor.clean_text(text)
                
                # Extract tables if any
                tables = PDFProcessor.extract_tables(page)
                
                # Extract images if any
                images = PDFProcessor.extract_images(page)
                
                content.append({
                    'page': page_num + 1,
                    'content': text,
                    'tables': tables,
                    'images': images
                })
            
            # Extract metadata
            metadata = doc.metadata
            
            return {
                'success': True,
                'content': content,
                'total_pages': total_pages,
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if 'doc' in locals():
                doc.close()
            
    @staticmethod
    def is_valid_pdf(file_path: str) -> bool:
        """Check if the file is a valid PDF."""
        try:
            with open(file_path, 'rb') as file:
                PyPDF2.PdfReader(file)
                return True
        except:
            return False 