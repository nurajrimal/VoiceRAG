"""
pdf_loader.py
Handles PDF upload, text extraction, and chunking.
"""

import os
import pdfplumber
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    Tries pdfplumber first (better for complex layouts), falls back to pypdf.
    """
    text = ""

    # Try pdfplumber first (handles tables and columns better)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        if text.strip():
            print(f"[PDF Loader] Extracted {len(text)} characters using pdfplumber.")
            return text
    except Exception as e:
        print(f"[PDF Loader] pdfplumber failed: {e}. Trying pypdf...")

    # Fallback to pypdf
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        print(f"[PDF Loader] Extracted {len(text)} characters using pypdf.")
    except Exception as e:
        print(f"[PDF Loader] pypdf also failed: {e}")
        raise RuntimeError(f"Could not extract text from PDF: {pdf_path}")

    return text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:
    """
    Split extracted text into overlapping chunks for embedding.

    Args:
        text: Full document text
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"[PDF Loader] Split into {len(chunks)} chunks.")
    return chunks


def get_pdf_metadata(pdf_path: str) -> dict:
    """Extract basic metadata from the PDF."""
    meta = {"filename": os.path.basename(pdf_path), "pages": 0, "title": "Unknown"}
    try:
        reader = PdfReader(pdf_path)
        meta["pages"] = len(reader.pages)
        info = reader.metadata
        if info and info.title:
            meta["title"] = info.title
    except Exception:
        pass
    return meta


def load_and_process_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Full pipeline: load PDF → extract text → split into chunks.

    Returns:
        (chunks, metadata)
    """
    print(f"[PDF Loader] Processing: {pdf_path}")
    metadata = get_pdf_metadata(pdf_path)
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF. It may be a scanned image.")

    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    return chunks, metadata