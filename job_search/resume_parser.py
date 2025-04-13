"""
Resume parsing utilities for extracting text from PDF and DOCX files.
"""

import os
import pdfplumber
import docx2txt

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        str: Extracted text or None if failed
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found!")
        return None
    
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip() if text else None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file.
    
    Args:
        docx_path (str): Path to DOCX file
        
    Returns:
        str: Extracted text or None if failed
    """
    if not os.path.exists(docx_path):
        print(f"Error: File '{docx_path}' not found!")
        return None
    
    try:
        return docx2txt.process(docx_path).strip()
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None

def get_resume_text(pdf_path=None, docx_path=None):
    """Get text from resume, prioritizing PDF over DOCX.
    
    Args:
        pdf_path (str, optional): Path to PDF file
        docx_path (str, optional): Path to DOCX file
        
    Returns:
        str: Extracted text or None if failed
    """
    if pdf_path:
        text = extract_text_from_pdf(pdf_path)
        if text:
            return text
            
    if docx_path:
        return extract_text_from_docx(docx_path)
        
    return None
