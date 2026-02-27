#!/usr/bin/env python3
"""Rebuild ChromaDB for a single company. Uses PyPDF2 for extraction to avoid memory issues."""
import sys
import os
import re
import gc
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, ".")

COMPANY = sys.argv[1] if len(sys.argv) > 1 else None
if not COMPANY:
    print("Usage: python3 rebuild_single.py <Company>")
    print("  e.g. python3 rebuild_single.py Benchmark")
    sys.exit(1)

import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import PyPDF2

SCRIPT_DIR = Path("Vector Database")
BASE = Path(__file__).resolve().parent
DB_PATH = str(BASE / "chromadb_store")

print(f"Rebuilding: {COMPANY}")
print(f"DB path: {DB_PATH}")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="capex_docs", metadata={"hnsw:space": "cosine"})
print(f"Existing docs: {collection.count()}")

print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")
print("Model loaded")

# --- Simple extractors (low memory) ---
def extract_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for el in soup(["script", "style", "head", "meta"]):
        el.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def extract_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f"  PDF error: {e}")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def extract(path):
    if path.suffix.lower() in (".html", ".htm"):
        return extract_html(path)
    elif path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    return ""

# --- SEC section detection ---
SEC_SECTIONS = [
    (r"(?i)consolidated\s+statements?\s+of\s+cash\s+flows?", "Cash Flow Statement"),
    (r"(?i)consolidated\s+balance\s+sheets?", "Balance Sheet"),
    (r"(?i)consolidated\s+statements?\s+of\s+(?:operations|income)", "Income Statement"),
    (r"(?:^|\n)\s*ITEM\s+7[\.:]\s*MANAGEMENT", "Item 7. MD&A"),
    (r"(?:^|\n)\s*ITEM\s+8[\.:]\s*FINANCIAL", "Item 8. Financial Statements"),
]

def detect_section(text):
    for pattern, name in SEC_SECTIONS:
        if re.search(pattern, text[:500]):
            return name
    return ""

def detect_table_type(text):
    t = text.lower()
    if any(k in t for k in ["cash flows from", "investing activities", "operating activities"]):
        return "cash_flow"
    if any(k in t for k in ["total assets", "total liabilities"]):
        return "balance_sheet"
    if any(k in t for k in ["net revenue", "gross profit", "operating income"]):
        return "income_statement"
    return ""

# --- Chunking ---
def chunk_text(text, chunk_words=250, overlap_words=50):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_words]).strip()
        if len(chunk) > 50:
            section = detect_section(chunk)
            chunks.append((section, chunk))
        i += chunk_words - overlap_words
    return chunks

# --- Filing type detection ---
def detect_filing_type(path, text=""):
    name = path.name.lower()
    if re.search(r"20-?f|20f", name): return "10-K"
    if re.search(r"6-?k|6k", name): return "10-Q"
    if "10-k" in name or "10k" in name: return "10-K"
    if "10-q" in name or "10q" in name: return "10-Q"
    if "8-k" in name or "8k" in name: return "8-K"
    if "annual" in name and "report" in name: return "10-K"
    if "earnings" in name and "presentation" in name: return "Earnings Presentation"
    if "earnings" in name and "release" in name: return "Press Release"
    if "press" in name or "release" in name: return "Press Release"
    return "Other"

def detect_fy(path, text=""):
    name = path.name
    m = re.search(r"(\d{4})", name)
    if m:
        return f"FY{m.group(1)[-2:]}"
    if text:
        m = re.search(r"fiscal\s*year\s*ended\s*.*?(20\d{2})", text[:10000], re.IGNORECASE)
        if m:
            return f"FY{m.group(1)[-2:]}"
    return "Unknown"

def detect_quarter(path):
    name = path.name
    m = re.search(r"[Qq](\d)", name)
    if m:
        return f"Q{m.group(1)}"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        mo = int(m.group(2))
        if mo in (1,2,3): return "Q1"
        if mo in (4,5,6): return "Q2"
        if mo in (7,8,9): return "Q3"
        if mo in (10,11,12): return "Q4"
    return ""

# --- Company file sources ---
SOURCES = {
    "Benchmark": [
        ("benchmark/benchmark_filings", None),
        ("benchmark/Annual Report", "10-K"),
        ("benchmark/Earnings Presentation", "Earnings Presentation"),
        ("benchmark/Press Release", "Press Release"),
    ],
    "Sanmina": [
        ("Sanmina/10K", "10-K"),
        ("Sanmina/10Q", "10-Q"),
        ("Sanmina/sanmina_8k", "8-K"),
        ("Sanmina/SanminaEarningsPresentations", "Earnings Presentation"),
        ("Sanmina/SanminaPressReleases", "Press Release"),
    ],
}

if COMPANY not in SOURCES:
    print(f"Unknown company: {COMPANY}. Available: {list(SOURCES.keys())}")
    sys.exit(1)

# --- Discover files ---
files = []
for subdir, ftype in SOURCES[COMPANY]:
    dirpath = BASE / subdir
    if not dirpath.is_dir():
        print(f"  Skipping {subdir} (not found)")
        continue
    for f in sorted(dirpath.iterdir()):
        if f.is_file() and f.suffix.lower() in (".html", ".htm", ".pdf"):
            actual_type = ftype if ftype else detect_filing_type(f)
            files.append((f, COMPANY, actual_type))

print(f"Files found: {len(files)}")

# --- Process ---
total = 0
for filepath, company, filing_type in files:
    print(f"  {filepath.name[:62]:<64}", end="", flush=True)

    text = extract(filepath)
    if not text or len(text.strip()) < 100:
        print("skip")
        continue

    if not filing_type or filing_type == "Other":
        filing_type = detect_filing_type(filepath, text)
    fy = detect_fy(filepath, text)
    q = detect_quarter(filepath)

    unit_header = ""
    um = re.search(r"\((?:in|amounts?\s+in)\s+(thousands|millions|billions)\)", text[:5000], re.IGNORECASE)
    if um:
        unit_header = um.group(1).lower()

    structured = chunk_text(text)
    if not structured:
        print("no chunks")
        continue

    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", filepath.stem)
    parent_id = f"{company}_{safe_stem}"

    ids, texts, metas = [], [], []
    for i, (section, chunk) in enumerate(structured):
        ids.append(f"{company}_{safe_stem}_chunk{i:04d}")
        texts.append(chunk)
        metas.append({
            "company": company,
            "source_file": filepath.name,
            "filing_type": filing_type,
            "fiscal_year": fy,
            "quarter": q,
            "chunk_index": i,
            "total_chunks": len(structured),
            "section_header": section,
            "table_type": detect_table_type(chunk),
            "table_context": f"in {unit_header}" if unit_header else "",
            "parent_id": parent_id,
        })

    for start in range(0, len(texts), 64):
        emb = model.encode(texts[start:start+64], show_progress_bar=False)
        collection.upsert(
            ids=ids[start:start+64],
            embeddings=emb.tolist(),
            documents=texts[start:start+64],
            metadatas=metas[start:start+64],
        )

    total += len(structured)
    print(f"{len(structured)} chunks")

    del text, structured, ids, texts, metas
    gc.collect()

print(f"\n{COMPANY} complete: {total} chunks")
print(f"Total in DB: {collection.count()}")
