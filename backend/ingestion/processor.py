"""
Document processor for ingesting new filings into ChromaDB.
"""
import re
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
import PyPDF2

from backend.core.database import get_collection, embed_texts
from backend.core.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_html(file_path: Path) -> str:
    """Extract text from HTML/HTM filing."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    soup = BeautifulSoup(content, "lxml")
    
    # Remove scripts and styles
    for element in soup(["script", "style", "meta", "link"]):
        element.decompose()
    
    text = soup.get_text(separator=" ", strip=True)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF filing."""
    text_parts = []
    
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""
    
    return " ".join(text_parts)


def extract_text(file_path: Path) -> str:
    """Extract text from a filing based on file type."""
    suffix = file_path.suffix.lower()
    
    if suffix in [".htm", ".html", ".xml"]:
        return extract_text_from_html(file_path)
    elif suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        # Try HTML extraction as default
        return extract_text_from_html(file_path)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Target words per chunk
        overlap: Words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        
        if chunk.strip():
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks


async def process_filing(
    file_path: Path,
    ticker: str,
    company_name: str,
    filing_type: str,
    filing_date: str,
) -> int:
    """
    Process a single filing and add to ChromaDB.
    
    Returns:
        Number of chunks added
    """
    # Extract text
    text = extract_text(file_path)
    
    if not text or len(text) < 100:
        print(f"No text extracted from {file_path}")
        return 0
    
    # Chunk text
    chunks = chunk_text(text)
    
    if not chunks:
        return 0
    
    # Prepare for ChromaDB
    collection = get_collection()
    
    # Generate embeddings
    embeddings = embed_texts(chunks)
    
    # Prepare IDs and metadata
    ids = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{ticker}_{filing_type}_{filing_date}_{i}"
        ids.append(chunk_id)
        metadatas.append({
            "company": ticker.upper() if ticker in ["FLEX", "JBL", "CLS", "BHE", "SANM"] else ticker,
            "company_name": company_name,
            "filing_type": filing_type,
            "fiscal_year": filing_date[:4],
            "quarter": "",
            "source_file": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })
    
    # Map ticker to company name for display
    company_display_map = {
        "FLEX": "Flex",
        "JBL": "Jabil", 
        "CLS": "Celestica",
        "BHE": "Benchmark",
        "SANM": "Sanmina",
    }
    
    for meta in metadatas:
        if meta["company"] in company_display_map:
            meta["company"] = company_display_map[meta["company"]]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    
    print(f"Added {len(chunks)} chunks from {file_path.name}")
    return len(chunks)


async def process_new_filings(filings: list[dict]) -> int:
    """
    Process a list of new filings and add to ChromaDB.
    
    Args:
        filings: List of filing dicts with local_path, ticker, etc.
        
    Returns:
        Total chunks added
    """
    total_chunks = 0
    
    for filing in filings:
        local_path = filing.get("local_path")
        if not local_path:
            continue
        
        path = Path(local_path)
        if not path.exists():
            continue
        
        chunks = await process_filing(
            file_path=path,
            ticker=filing["ticker"],
            company_name=filing["company"],
            filing_type=filing["form"],
            filing_date=filing["filing_date"],
        )
        total_chunks += chunks
    
    return total_chunks
