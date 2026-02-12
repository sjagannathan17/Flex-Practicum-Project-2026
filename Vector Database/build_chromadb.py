#!/usr/bin/env python3
"""
CapEx Intelligence — Multi-Company ChromaDB Vector Embedding Pipeline
ChromaDB 1.x | sentence-transformers all-mpnet-base-v2 (768-dim)

Companies: Flex, Jabil, Celestica, Benchmark, Sanmina

Run:
    cd "Vector Database"
    pip install chromadb sentence-transformers beautifulsoup4 lxml PyPDF2
    python build_chromadb.py
"""

import re
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from bs4 import BeautifulSoup
    import PyPDF2
except ImportError:
    import subprocess, sys
    for p in ["chromadb", "sentence-transformers", "beautifulsoup4", "lxml", "PyPDF2"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", p, "-q"])
    import chromadb
    from sentence_transformers import SentenceTransformer
    from bs4 import BeautifulSoup
    import PyPDF2

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent  # SCU Flex Practicum 2026/
DB_PATH = str(BASE / "chromadb_store")

# Company configurations: folder name -> (subfolder, filing_type) pairs
# filing_type=None means auto-detect from filename
SOURCES = {
    "Flex": [
        ("annual_10K", "10-K"),
        ("quarterly_10Q", "10-Q"),
        ("flex_8k_press_releases", "8-K"),
        ("flex_transcripts", "Earnings Transcript"),
        ("Earnings Presentation", "Earnings Presentation"),
        ("Press Releases", "Press Release"),
    ],
    "Jabil": [
        ("10K", "10-K"),
        ("10Q", "10-Q"),
        ("8K", "8-K"),
        ("Earnings Call", "Earnings Transcript"),
        ("Earnings Presentation", "Earnings Presentation"),
        ("Press Release", "Press Release"),
    ],
    "Celestica/Celestica": [  # Nested folder
        ("10-K", "10-K"),
        ("10-Q", "10-Q"),
        ("Earnings Calls/Transcript", "Earnings Transcript"),
        ("Earnings Presentation ", "Earnings Presentation"),  # Note trailing space
        ("ShareholderLetter", "Shareholder Letter"),
    ],
    "benchmark": [
        ("Annual Report", "10-K"),
        ("benchmark_filings", None),  # Auto-detect 10-K/10-Q from filename
        ("Earnings Presentation/2022", "Earnings Presentation"),
        ("Earnings Presentation/2023", "Earnings Presentation"),
        ("Earnings Presentation/2024", "Earnings Presentation"),
        ("Earnings Presentation/2025", "Earnings Presentation"),
        ("Press Release/2022", "Press Release"),
        ("Press Release/2023", "Press Release"),
        ("Press Release/2024", "Press Release"),
        ("Press Release/2025", "Press Release"),
        ("Sidoti September Small-Cap Virtual Conference", "Investor Presentation"),
    ],
    "Sanmina": [
        ("10K", "10-K"),
        ("10Q", "10-Q"),
        ("sanmina_8k", "8-K"),
        ("SanminaEarningsPresentations", None),  # Mixed - auto-detect
        ("SanminaPressReleases", "Press Release"),
    ],
}

# Display name normalization
COMPANY_DISPLAY = {
    "Flex": "Flex",
    "Jabil": "Jabil",
    "Celestica/Celestica": "Celestica",
    "benchmark": "Benchmark",
    "Sanmina": "Sanmina",
}

# Company CIKs for SEC lookup
COMPANY_CIKS = {
    "Flex": "0000866374",
    "Jabil": "0000898293",
    "Celestica": "0001030894",
    "Benchmark": "0001080020",
    "Sanmina": "0000897723",
}

# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_html(path):
    """Extract text from HTML/HTM files."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n", strip=True)


def extract_pdf(path):
    """Extract text from PDF files."""
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f" ⚠️  PDF error {path.name}: {e}")
    return text


def extract_txt(path):
    """Extract text from TXT/MD files."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f" ⚠️  TXT error {path.name}: {e}")
        return ""


def extract(path):
    """Extract text from any supported file type."""
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return extract_html(path)
    elif suffix == ".pdf":
        return extract_pdf(path)
    elif suffix in (".txt", ".md"):
        return extract_txt(path)
    return ""


# ---------------------------------------------------------------------------
# FILING TYPE DETECTION
# ---------------------------------------------------------------------------
def detect_filing_type(path):
    """Auto-detect filing type from filename."""
    name = path.name.lower()
    
    # SEC filings
    if "10-k" in name or "10k" in name or "20f" in name:
        return "10-K"
    if "10-q" in name or "10q" in name or "6k" in name:
        return "10-Q"
    if "8-k" in name or "8k" in name:
        return "8-K"
    
    # Sanmina mixed folder detection
    if "presentation" in name or "slide" in name:
        return "Earnings Presentation"
    if "press" in name or "release" in name:
        return "Press Release"
    if "transcript" in name or "call" in name:
        return "Earnings Transcript"
    
    return "Other"


# ---------------------------------------------------------------------------
# FISCAL QUARTER EXTRACTION
# ---------------------------------------------------------------------------
def get_fiscal_quarter(path, company):
    """Extract fiscal year and quarter from filename."""
    name = path.name
    
    # Pattern 1: FY22Q3 / FY2022Q3 / fy24q2
    m = re.search(r"[Ff][Yy](\d{2,4})[_\-]?[Qq](\d)", name)
    if m:
        fy = m.group(1)[-2:]
        return f"FY{fy}", f"Q{m.group(2)}"
    
    # Pattern 2: Q3-FY26 / Q1_FY25
    m = re.search(r"[Qq](\d)[_\-]?[Ff][Yy](\d{2,4})", name)
    if m:
        fy = m.group(2)[-2:]
        return f"FY{fy}", f"Q{m.group(1)}"
    
    # Pattern 3: JBL_2023_10Q_Q2 / Company_2025_EarningsCall_Q3
    m = re.search(r"(\d{4}).*[Qq](\d)", name)
    if m:
        return f"FY{m.group(1)[-2:]}", f"Q{m.group(2)}"
    
    # Pattern 4: 25Q2 / 23Q3
    m = re.search(r"(\d{2})[Qq](\d)", name)
    if m:
        return f"FY{m.group(1)}", f"Q{m.group(2)}"
    
    # Pattern 5: Q1-2025 / Q2_2024
    m = re.search(r"[Qq](\d)[_\-](\d{4})", name)
    if m:
        return f"FY{m.group(2)[-2:]}", f"Q{m.group(1)}"
    
    # Pattern 6: Q1_22 / Q3_24
    m = re.search(r"[Qq](\d)[_\-](\d{2})(?!\d)", name)
    if m:
        return f"FY{m.group(2)}", f"Q{m.group(1)}"
    
    # Pattern 7: Benchmark format Q1-2023 in folder
    m = re.search(r"[Qq](\d)[_\-]?20(\d{2})", name)
    if m:
        return f"FY{m.group(2)}", f"Q{m.group(1)}"
    
    # Pattern 8: Date format YYYY-MM-DD (Flex 10-Q/10-K)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        
        if company == "Flex":
            # Flex FY ends March
            fy = y + 1 if mo >= 4 else y
            if mo in (7, 8):
                q = "Q1"
            elif mo in (10, 11):
                q = "Q2"
            elif mo in (1, 2):
                q = "Q3"
            elif mo in (5, 6):
                q = "Q4"
            else:
                q = ""
            return f"FY{str(fy)[-2:]}", q
        
        elif company == "Benchmark":
            # Benchmark = calendar year
            if mo in (1, 2, 3):
                q = "Q1"
            elif mo in (4, 5, 6):
                q = "Q2"
            elif mo in (7, 8, 9):
                q = "Q3"
            elif mo in (10, 11, 12):
                q = "Q4"
            else:
                q = ""
            return f"FY{str(y)[-2:]}", q
    
    # Pattern 9: Year only from filename like _2024_ or -2023-
    m = re.search(r"[_\-]?(20\d{2})[_\-]?", name)
    if m:
        return f"FY{m.group(1)[-2:]}", ""
    
    return "Unknown", ""


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_words=400, overlap_words=40):
    """Split text into overlapping chunks."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_words]).strip()
        if len(chunk) > 50:  # Minimum chunk size
            chunks.append(chunk)
        i += chunk_words - overlap_words
    
    return chunks


# ---------------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------------
def discover_all_files():
    """Walk the SOURCES map and return list of (path, company_display, filing_type)."""
    results = []
    
    for company_folder, subdir_list in SOURCES.items():
        company_path = BASE / company_folder
        if not company_path.is_dir():
            print(f"  ⚠️  Skipping {company_folder}/ — not found")
            continue
        
        display = COMPANY_DISPLAY[company_folder]
        file_count = 0
        
        for subdir_name, ftype in subdir_list:
            subdir_path = company_path / subdir_name
            if not subdir_path.is_dir():
                print(f"  ⚠️  {company_folder}/{subdir_name}/ not found, skipping")
                continue
            
            # Get all supported files (recursive for nested folders)
            extensions = (".html", ".htm", ".pdf", ".txt", ".md")
            files = sorted([
                f for f in subdir_path.rglob("*")
                if f.is_file() and f.suffix.lower() in extensions
                and not f.name.startswith(".")
            ])
            
            for f in files:
                actual_ftype = ftype if ftype else detect_filing_type(f)
                results.append((f, display, actual_ftype))
                file_count += 1
        
        print(f"  📂 {display:<12} → {file_count} files")
    
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def build_db():
    print("=" * 70)
    print("  CAPEX INTELLIGENCE — CHROMADB EMBEDDING PIPELINE")
    print("  Companies: Flex, Jabil, Celestica, Benchmark, Sanmina")
    print("=" * 70)
    
    # --- ChromaDB ---
    print(f"\n📁 DB path: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(name="capex_docs")
        print("   Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name="capex_docs",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   Collection: capex_docs | Fresh start")
    
    # --- Embedding model ---
    print("\n🔄 Loading embedding model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    print("   ✓ Model loaded (768-dim vectors)")
    
    # --- Discover files ---
    print("\n📂 Scanning company folders...")
    all_files = discover_all_files()
    print(f"\n   Total files to process: {len(all_files)}")
    
    if not all_files:
        print("\n❌ No files found. Check folder structure.")
        return
    
    # --- Process ---
    total_chunks = 0
    stats = defaultdict(lambda: {"files": 0, "chunks": 0})
    company_stats = defaultdict(lambda: {"files": 0, "chunks": 0})
    
    for filepath, company, filing_type in all_files:
        fy, q = get_fiscal_quarter(filepath, company)
        
        print(f"  📄 [{company:<11}] {filepath.name[:55]:<55} ", end="", flush=True)
        
        text = extract(filepath)
        if not text or len(text.strip()) < 100:
            print("→ empty/too short")
            continue
        
        chunks = chunk_text(text)
        if not chunks:
            print("→ no chunks")
            continue
        
        print(f"→ {len(chunks):>3} chunks", end="", flush=True)
        
        # Build batch
        ids, texts, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", filepath.stem)
            doc_id = f"{company}_{safe_stem}_chunk{i:04d}"
            
            ids.append(doc_id)
            texts.append(chunk)
            metadatas.append({
                "company": company,
                "source_file": filepath.name,
                "filing_type": filing_type,
                "fiscal_year": fy,
                "quarter": q,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        
        # Embed and upsert in batches of 64
        for start in range(0, len(texts), 64):
            b_ids = ids[start:start + 64]
            b_txt = texts[start:start + 64]
            b_meta = metadatas[start:start + 64]
            
            embeddings = model.encode(b_txt, show_progress_bar=False)
            collection.add(
                ids=b_ids,
                embeddings=embeddings.tolist(),
                documents=b_txt,
                metadatas=b_meta,
            )
        
        total_chunks += len(chunks)
        stats[filing_type]["files"] += 1
        stats[filing_type]["chunks"] += len(chunks)
        company_stats[company]["files"] += 1
        company_stats[company]["chunks"] += len(chunks)
        print(" ✓")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("  EMBEDDING COMPLETE")
    print("=" * 70)
    print(f"\n  Total docs in collection: {collection.count()}")
    print(f"  Total chunks embedded:    {total_chunks}\n")
    
    print("  BY COMPANY:")
    print(f"  {'Company':<14} {'Files':<8} {'Chunks'}")
    print(f"  {'-' * 34}")
    for co in sorted(company_stats.keys()):
        print(f"  {co:<14} {company_stats[co]['files']:<8} {company_stats[co]['chunks']}")
    
    print(f"\n  BY FILING TYPE:")
    print(f"  {'Filing Type':<28} {'Files':<8} {'Chunks'}")
    print(f"  {'-' * 48}")
    for ftype in sorted(stats.keys()):
        print(f"  {ftype:<28} {stats[ftype]['files']:<8} {stats[ftype]['chunks']}")
    
    print(f"\n  DB stored at: {DB_PATH}")
    
    # --- Smoke tests ---
    print("\n" + "=" * 70)
    print("  SMOKE TESTS")
    print("=" * 70)
    
    test_queries = [
        ("Flex CapEx", "Flex capital expenditure property plant equipment"),
        ("Jabil AI", "Jabil AI data center investment manufacturing"),
        ("Celestica", "Celestica capital investment revenue growth"),
        ("Benchmark", "Benchmark Electronics capital expenditure"),
        ("Sanmina", "Sanmina capital investment facilities expansion"),
        ("Cross-company", "AI data center liquid cooling infrastructure investment"),
    ]
    
    for label, query in test_queries:
        print(f"\n  🔍 [{label}] \"{query[:50]}...\"")
        q_emb = model.encode([query])
        results = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                sim = round(1 - dist, 3)
                file_name = meta['source_file'][:40] if len(meta['source_file']) > 40 else meta['source_file']
                print(f"     sim={sim:.3f}  [{meta['company']:<11}] {file_name}")
        else:
            print("     No results found")
    
    print("\n✅ ChromaDB ready! Next: Start the FastAPI backend.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_db()
