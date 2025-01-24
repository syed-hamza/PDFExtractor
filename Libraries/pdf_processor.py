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
from PIL import Image

class PDFProcessor:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        # Fix common OCR issues
        replacements = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '−': '-',
            '\u2028': ' ',  # Line separator
            '\u2029': '\n',  # Paragraph separator
            '…': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix spacing issues
        text = re.sub(r'(?<=\w)(?=[A-Z])', ' ', text)  # Add space between words stuck together
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize line breaks
        
        # Split into lines and clean each line
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Fix cases where words are stuck together
                line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    @staticmethod
    def extract_text_from_page(page) -> str:
        """Extract text from a page while preserving layout."""
        # Get text with better layout preservation
        text = page.get_text("dict")
        
        blocks = []
        for block in text["blocks"]:
            if "lines" in block:
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        # Add space before if needed
                        if span.get("space_before", 0) > 0:
                            line_text.append(" ")
                        line_text.append(span["text"])
                    block_text.append("".join(line_text))
                blocks.append(" ".join(block_text))
        
        return "\n\n".join(blocks)

    @staticmethod
    def is_math_content(text: str) -> bool:
        """Determine if text is likely mathematical content."""
        # Count math-related characters
        math_chars = set('+-*/=≠<>≤≥∫∑∏√∂∇∆∈∉⊂⊃∪∩αβγδθλμπσφω')
        math_char_count = sum(1 for c in text if c in math_chars)
        
        # Check for specific patterns that indicate math content
        math_patterns = [
            r'\$.*\$',  # LaTeX math delimiters
            r'\\[a-zA-Z]+{',  # LaTeX commands
            r'\d+/\d+',  # Fractions
            r'[a-zA-Z]_\d',  # Subscripts
            r'[a-zA-Z]\^',  # Superscripts
            r'\\begin{equation}',  # LaTeX equation environments
            r'\\[\(\)]',  # LaTeX inline math
        ]
        
        # If there are too few math characters and no math patterns, it's probably not math
        if math_char_count < 2 and not any(re.search(pattern, text) for pattern in math_patterns):
            return False
        
        # Check the ratio of math characters to text length
        math_ratio = math_char_count / len(text) if text else 0
        return math_ratio > 0.1 or any(re.search(pattern, text) for pattern in math_patterns)

    @staticmethod
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

        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            if PDFProcessor.is_math_content(paragraph):
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
                text = PDFProcessor.extract_text_from_page(page)
                text = PDFProcessor.clean_text(text)
                
                # Only format as math if it's actually mathematical content
                if PDFProcessor.is_math_content(text):
                    text = PDFProcessor.format_math_expressions(text)
                
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
    def extract_tables(page) -> List[List[List[str]]]:
        """Extract tables from a page."""
        try:
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save temporary image
            temp_img_path = "temp_page.png"
            img.save(temp_img_path)
            
            # Extract tables using tabula
            tables = tabula.read_pdf(temp_img_path, pages=1, multiple_tables=True)
            
            # Clean up
            os.remove(temp_img_path)
            
            # Convert tables to list format and clean data
            cleaned_tables = []
            for table in tables:
                if not table.empty:
                    # Convert DataFrame to list of lists
                    table_data = table.values.tolist()
                    # Clean the data
                    cleaned_table = [[str(cell).strip() if pd.notna(cell) else None for cell in row] for row in table_data]
                    cleaned_tables.append(cleaned_table)
            
            return cleaned_tables
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []

    @staticmethod
    def extract_images(page) -> List[Dict[str, Any]]:
        """Extract images from a page."""
        images = []
        try:
            for img_index, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    
                    if base_image:
                        image_data = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Convert to base64
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        
                        images.append({
                            'data': f'data:image/{image_ext};base64,{base64_data}',
                            'type': image_ext
                        })
                except Exception as e:
                    print(f"Error processing image {img_index}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error extracting images: {str(e)}")
            
        return images

    @staticmethod
    def is_valid_pdf(file_path: str) -> bool:
        """Check if the file is a valid PDF."""
        try:
            # Try to open with PyMuPDF
            doc = fitz.open(file_path)
            is_valid = doc.is_pdf
            doc.close()
            return is_valid
        except Exception:
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    PyPDF2.PdfReader(file)
                return True
            except Exception:
                return False 