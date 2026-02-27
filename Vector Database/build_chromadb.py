#!/usr/bin/env python3
"""
CapEx Intelligence — Multi-Company ChromaDB Vector Embedding Pipeline
ChromaDB 1.4.x | sentence-transformers all-mpnet-base-v2 (768-dim)

Run from the project root:
    cd Flex-Practicum-Project-2026
    python "Vector Database/build_chromadb.py"
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

try:
    import pdfplumber
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
    import pdfplumber

# ---------------------------------------------------------------------------
# CONFIG — auto-detect project root from this script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent                          # project root
DB_PATH = str(BASE / "chromadb_store")

# Company → list of (subfolder, filing_type) pairs.
# Paths are relative to BASE / company_folder.
SOURCES = {
    "Flex": [
        ("annual_10K",             "10-K"),
        ("quarterly_10Q",          "10-Q"),
        ("flex_8k_press_releases", "8-K"),
        ("flex_transcripts",       "Earnings Transcript"),
        ("Earnings Presentation",  "Earnings Presentation"),
        ("Press Releases",         "Press Release"),
    ],
    "Jabil": [
        ("10K",                    "10-K"),
        ("10Q",                    "10-Q"),
        ("8K",                     "8-K"),
        ("Earnings Call",          "Earnings Transcript"),
        ("Earnings Presentation",  "Earnings Presentation"),
        ("Press Release",          "Press Release"),
    ],
    "Celestica/Celestica": [
        ("10-K",                   "10-K"),
        ("10-Q",                   "10-Q"),
        ("Earnings Calls",         "Earnings Transcript"),
        ("Earnings Presentation ", "Earnings Presentation"),
        ("News Releases",          "Press Release"),
        ("ShareholderLetter",      "Shareholder Letter"),
    ],
    "benchmark": [
        ("benchmark_filings",      None),   # None = auto-detect from filename
        ("Annual Report",          "10-K"),
        ("Earnings Presentation",  "Earnings Presentation"),
        ("Press Release",          "Press Release"),
    ],
    "Sanmina": [
        ("10K",                    "10-K"),
        ("10Q",                    "10-Q"),
        ("sanmina_8k",             "8-K"),
        ("SanminaEarningsPresentations", "Earnings Presentation"),
        ("SanminaPressReleases",   "Press Release"),
    ],
}

COMPANY_DISPLAY = {
    "Flex": "Flex",
    "Jabil": "Jabil",
    "Celestica/Celestica": "Celestica",
    "benchmark": "Benchmark",
    "Sanmina": "Sanmina",
}

# ---------------------------------------------------------------------------
# TEXT POST-PROCESSING (Person 1 fixes)
# ---------------------------------------------------------------------------
def _collapse_char_spaced(text):
    """Collapse character-spaced text like 'S t o c k  T r a d i n g' -> 'Stock Trading'."""
    return re.sub(
        r'(?<!\S)((?:\S ){3,}\S)(?!\S)',
        lambda m: m.group(1).replace(" ", ""),
        text,
    )

def _fix_word_boundaries(text):
    """Insert spaces at camelCase / punctuation boundaries.
    e.g. 'theCompany' -> 'the Company', 'operations.The' -> 'operations. The'
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    return text

def _clean_extracted_text(text):
    """Apply all post-processing fixes to extracted text."""
    text = _collapse_char_spaced(text)
    text = _fix_word_boundaries(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()

# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for el in soup(["script", "style", "head", "meta"]):
        el.decompose()
    return _clean_extracted_text(soup.get_text(separator="\n", strip=True))

def _extract_page_with_tables(page):
    """Extract text from a single pdfplumber page, rendering tables as Markdown."""
    parts = []
    tables = page.find_tables()
    table_bboxes = [t.bbox for t in tables]

    # Extract non-table text
    if table_bboxes:
        non_table_page = page
        for bbox in table_bboxes:
            clipped = (
                max(0, bbox[0]), max(0, bbox[1]),
                min(page.width, bbox[2]), min(page.height, bbox[3]),
            )
            try:
                non_table_page = non_table_page.outside_bbox(clipped)
            except Exception:
                pass
        text = non_table_page.extract_text() or ""
        if text.strip():
            parts.append(text)
    else:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)

    # Extract tables as Markdown
    for table in tables:
        rows = table.extract()
        if not rows or len(rows) < 2:
            continue
        header = [cell.strip() if cell else "" for cell in rows[0]]
        md_lines = ["| " + " | ".join(header) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            cells = [cell.strip() if cell else "" for cell in row]
            if any(cells):
                md_lines.append("| " + " | ".join(cells) + " |")
        parts.append("\n".join(md_lines))

    return "\n\n".join(parts)

def _extract_page_words(page):
    """Word-level extraction for multi-column pages (correct reading order)."""
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    if not words:
        return ""
    # Group by y-position (lines), then sort each line by x
    lines = {}
    for w in words:
        y_key = round(w["top"] / 3) * 3
        lines.setdefault(y_key, []).append(w)
    sorted_lines = sorted(lines.items())
    result = []
    for _, line_words in sorted_lines:
        line_words.sort(key=lambda w: w["x0"])
        result.append(" ".join(w["text"] for w in line_words))
    return "\n".join(result)

def extract_pdf(path):
    """Extract text from PDF using pdfplumber (with table + multi-column support).
    Falls back to PyPDF2 if pdfplumber fails entirely."""
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = _extract_page_with_tables(page)
                if not page_text or len(page_text.strip()) < 20:
                    page_text = _extract_page_words(page)
                if page_text:
                    text += page_text + "\n\n"
        if text.strip():
            return _clean_extracted_text(text)
    except Exception as e:
        print(f" ⚠️  pdfplumber error, falling back to PyPDF2: {e}")

    # PyPDF2 fallback
    text = ""
    try:
        with open(path, "rb") as f:
            for page in PyPDF2.PdfReader(f).pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        print(f" ⚠️  PDF error: {e}")
    return _clean_extracted_text(text)


def extract_pdf_structured(path):
    """Extract text from PDF with per-page structure and table tracking.
    Returns list of dicts: [{page_num, text, tables, unit_header}]."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = []
                table_objs = page.find_tables()

                for table in table_objs:
                    rows = table.extract()
                    if not rows or len(rows) < 2:
                        continue
                    header = [cell.strip() if cell else "" for cell in rows[0]]
                    data_rows = []
                    for row in rows[1:]:
                        cells = [cell.strip() if cell else "" for cell in row]
                        if any(cells):
                            data_rows.append(cells)
                    if data_rows:
                        page_tables.append({"header": header, "rows": data_rows})

                page_text = _extract_page_with_tables(page)
                if not page_text or len(page_text.strip()) < 20:
                    page_text = _extract_page_words(page)
                page_text = _clean_extracted_text(page_text or "")

                # Detect unit header on this page
                unit = ""
                unit_match = re.search(r"\((?:in|amounts?\s+in)\s+(thousands|millions|billions)\)", page_text, re.IGNORECASE)
                if unit_match:
                    unit = unit_match.group(1).lower()

                pages.append({
                    "page_num": page_num,
                    "text": page_text,
                    "tables": page_tables,
                    "unit_header": unit,
                })
    except Exception as e:
        print(f" ⚠️  structured PDF extraction failed: {e}")
    return pages


def extract(path):
    if path.suffix.lower() in (".html", ".htm"):
        return extract_html(path)
    elif path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    return ""


# ---------------------------------------------------------------------------
# TABLE SERIALIZATION — linearized text for better embedding
# ---------------------------------------------------------------------------
def serialize_table_for_embedding(header, rows, table_context=""):
    """Convert a table to linearized text for embedding.
    e.g. 'Row 1: Revenue = $5,000M, Growth = 15%'"""
    lines = []
    if table_context:
        lines.append(table_context)
    for row in rows:
        pairs = []
        for h, v in zip(header, row):
            if h and v:
                pairs.append(f"{h} = {v}")
        if pairs:
            lines.append("; ".join(pairs))
    return "\n".join(lines)


def serialize_table_as_markdown(header, rows):
    """Convert a table to Markdown format for display."""
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# SEC SECTION DETECTION
# ---------------------------------------------------------------------------
SEC_SECTION_PATTERNS = [
    (r"(?:^|\n)\s*ITEM\s+1[\.:]\s*BUSINESS", "Item 1. Business"),
    (r"(?:^|\n)\s*ITEM\s+1A[\.:]\s*RISK\s*FACTORS", "Item 1A. Risk Factors"),
    (r"(?:^|\n)\s*ITEM\s+7[\.:]\s*MANAGEMENT", "Item 7. MD&A"),
    (r"(?:^|\n)\s*ITEM\s+8[\.:]\s*FINANCIAL\s*STATEMENTS", "Item 8. Financial Statements"),
    (r"(?i)consolidated\s+statements?\s+of\s+cash\s+flows?", "Cash Flow Statement"),
    (r"(?i)consolidated\s+balance\s+sheets?", "Balance Sheet"),
    (r"(?i)consolidated\s+statements?\s+of\s+(?:operations|income)", "Income Statement"),
    (r"(?i)notes?\s+to\s+(?:consolidated\s+)?financial\s+statements", "Notes to Financial Statements"),
]

def _detect_section(text):
    """Detect the SEC section a chunk of text belongs to."""
    for pattern, section_name in SEC_SECTION_PATTERNS:
        if re.search(pattern, text[:500]):
            return section_name
    return ""

def _detect_table_type(text):
    """Detect if a chunk contains a specific financial table type."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["cash flows from", "investing activities", "operating activities"]):
        return "cash_flow"
    if any(kw in text_lower for kw in ["total assets", "total liabilities", "stockholders"]):
        return "balance_sheet"
    if any(kw in text_lower for kw in ["net revenue", "cost of", "gross profit", "operating income"]):
        return "income_statement"
    return ""

# ---------------------------------------------------------------------------
# CONTENT-BASED DOC TYPE PATTERNS (for ambiguous filenames)
# ---------------------------------------------------------------------------
CONTENT_DOC_PATTERNS = {
    "10-K": [
        r"ANNUAL\s*REPORT", r"FORM\s*10-K", r"FORM\s*20-F",
        r"pursuant\s*to\s*section\s*13\s*or\s*15\(d\)",
        r"fiscal\s*year\s*ended",
        r"annual\s*report\s*on\s*form\s*10-k",
        r"annual\s*report\s*on\s*form\s*20-f",
    ],
    "10-Q": [
        r"QUARTERLY\s*REPORT", r"FORM\s*10-Q", r"FORM\s*6-K",
        r"quarterly\s*report\s*on\s*form\s*10-q",
        r"quarter\s*ended", r"quarterly\s*period\s*ended",
        r"report\s*of\s*foreign\s*private\s*issuer",
    ],
}

# ---------------------------------------------------------------------------
# FILING TYPE: auto-detect with foreign issuer + content fallback
# ---------------------------------------------------------------------------
def detect_filing_type(path, content=""):
    name = path.name.lower()

    # Foreign issuer forms (Celestica: 20-F = annual, 6-K = quarterly)
    if re.search(r"20-?f|20f", name):       return "10-K"
    if re.search(r"6-?k|6k", name):         return "10-Q"

    # Standard US forms
    if name.startswith("10-k") or "10k" in name:   return "10-K"
    if name.startswith("10-q") or "10q" in name:   return "10-Q"
    if name.startswith("8-k")  or "8k"  in name:   return "8-K"
    if "annual" in name and "report" in name:       return "10-K"

    # Content-based fallback for ambiguous filenames
    if content:
        sample = content[:5000]
        scores = {}
        for dtype, patterns in CONTENT_DOC_PATTERNS.items():
            scores[dtype] = sum(1 for p in patterns if re.search(p, sample, re.IGNORECASE))
        best = max(scores, key=scores.get) if scores else None
        if best and scores[best] >= 2:
            return best

    return "Other"

# ---------------------------------------------------------------------------
# FISCAL QUARTER EXTRACTION
# ---------------------------------------------------------------------------
def get_fiscal_quarter(path, company, content=""):
    name = path.name

    # --- Pattern 1: FY22Q3 / FY2022Q3 / fy24q2 ---
    m = re.search(r"[Ff][Yy](\d{2,4})[_\-]?[Qq](\d)", name)
    if m:
        fy = m.group(1)[-2:]
        return f"FY{fy}", f"Q{m.group(2)}"

    # --- Pattern 2: Q3-FY26 / Q1_FY25 ---
    m = re.search(r"[Qq](\d)[_\-]?[Ff][Yy](\d{2,4})", name)
    if m:
        fy = m.group(2)[-2:]
        return f"FY{fy}", f"Q{m.group(1)}"

    # --- Pattern 3: JBL_2023_10Q_Q2 / JBL_2025_EarningsCall_Q3 ---
    m = re.search(r"(\d{4}).*[Qq](\d)", name)
    if m:
        return f"FY{m.group(1)[-2:]}", f"Q{m.group(2)}"

    # --- Pattern 4: 25Q2 / 23Q3 ---
    m = re.search(r"(\d{2})[Qq](\d)", name)
    if m:
        return f"FY{m.group(1)}", f"Q{m.group(2)}"

    # --- Pattern 5: Samsara 8K filenames like 8K_0824.pdf  (MMYY) ---
    # and 10Q like 10Q_0824.pdf
    m = re.search(r"(?:8[Kk]|10[Kk]|10[Qq])_(\d{2})(\d{2})", name)
    if m:
        mm, yy = int(m.group(1)), m.group(2)
        # Samsara FY ends late Jan/early Feb → Feb-Jan fiscal year
        # Q1=Feb-Apr, Q2=May-Jul, Q3=Aug-Oct, Q4=Nov-Jan
        if   mm in (2,3,4):     q = "Q1"
        elif mm in (5,6,7):     q = "Q2"
        elif mm in (8,9,10):    q = "Q3"
        elif mm in (11,12,1):   q = "Q4"
        else:                   q = ""
        # FY = calendar year of the end month (Jan belongs to prior FY end)
        fy_year = int(yy) if mm != 1 else int(yy) - 1
        return f"FY{yy}", q

    # --- Pattern 6: Benchmark/Flex HTML with date: 10-Q_2023-09-30 / Flex_10-Q_2024-07-26 ---
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))

        if company in ("Flex",):
            # Flex FY ends March. Filing months: Jul=Q1, Oct=Q2, Jan=Q3, May=Q4/10-K
            fy = y + 1 if mo >= 4 else y
            if   mo in (7,8):     q = "Q1"
            elif mo in (10,11):   q = "Q2"
            elif mo in (1,2):     q = "Q3"
            elif mo in (5,6):     q = "Q4"
            else:                 q = ""
            return f"FY{str(fy)[-2:]}", q

        elif company in ("benchmark",):
            # Benchmark = calendar year. Date in filename IS the period-end date.
            # 2023-09-30 → Q3 FY23
            if   mo in (1,2,3):     q = "Q1"
            elif mo in (4,5,6):     q = "Q2"
            elif mo in (7,8,9):     q = "Q3"
            elif mo in (10,11,12):  q = "Q4"
            else:                   q = ""
            return f"FY{str(y)[-2:]}", q

    # --- Pattern 7: Q1-2025 (calendar) ---
    m = re.search(r"[Qq](\d)[_\-](\d{4})", name)
    if m:
        return f"FY{m.group(2)[-2:]}", f"Q{m.group(1)}"

    # --- Content-based fallback (for UUID filenames like Sanmina) ---
    if content:
        sample = content[:10000]
        fy_patterns = [
            r"fiscal\s*year\s*ended\s*.*?(20\d{2})",
            r"year\s*ended\s*(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{1,2}\s*,?\s*(20\d{2})",
            r"for\s*the\s*year\s*ended\s*.*?(20\d{2})",
            r"quarterly\s*(?:report|period)\s*ended\s*.*?(20\d{2})",
        ]
        for pat in fy_patterns:
            m = re.search(pat, sample, re.IGNORECASE)
            if m:
                return f"FY{m.group(1)[-2:]}", ""

    return "Unknown", ""

# ---------------------------------------------------------------------------
# CHUNKING — structure-aware with table preservation
# ---------------------------------------------------------------------------
MIN_CHUNK_WORDS = 60
MAX_CHUNK_WORDS = 350
TARGET_CHUNK_WORDS = 250
OVERLAP_WORDS = 50

def _split_by_sections(text):
    """Split text by SEC section headers. Returns list of (section_name, content)."""
    section_breaks = []
    for pattern, name in SEC_SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            section_breaks.append((m.start(), name))

    if not section_breaks:
        return [("", text)]

    section_breaks.sort(key=lambda x: x[0])
    sections = []

    # Text before first section
    if section_breaks[0][0] > 0:
        pre = text[:section_breaks[0][0]].strip()
        if pre:
            sections.append(("", pre))

    for i, (pos, name) in enumerate(section_breaks):
        end = section_breaks[i + 1][0] if i + 1 < len(section_breaks) else len(text)
        content = text[pos:end].strip()
        if content:
            sections.append((name, content))

    return sections


def _chunk_section(text, section_name=""):
    """Chunk a section, keeping tables intact and respecting paragraph boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        # If this paragraph contains a table (has | characters), keep it intact
        is_table = para.count("|") >= 4

        if is_table:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_words = 0
            # Table as its own chunk (even if large)
            chunks.append(para)
            continue

        # If adding this paragraph would exceed max, flush
        if current_words + para_words > MAX_CHUNK_WORDS and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep overlap: last paragraph carries over
            overlap_text = current_chunk[-1] if current_chunk else ""
            current_chunk = [overlap_text] if len(overlap_text.split()) <= OVERLAP_WORDS else []
            current_words = len(" ".join(current_chunk).split())

        current_chunk.append(para)
        current_words += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS // 2 or c.count("|") >= 4]


def chunk_text_structured(text):
    """Structure-aware chunking: splits by SEC sections, preserves tables, respects paragraphs."""
    sections = _split_by_sections(text)
    all_chunks = []
    for section_name, content in sections:
        section_chunks = _chunk_section(content, section_name)
        for chunk in section_chunks:
            all_chunks.append((section_name, chunk))
    return all_chunks


def chunk_text(text, chunk_words=250, overlap_words=50):
    """Fallback fixed-word chunker (used when structure-aware chunking is not needed)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_words]).strip())
        i += chunk_words - overlap_words
    return [c for c in chunks if len(c) > 50]

# ---------------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------------
def discover_all_files():
    """
    Walk the explicit SOURCES map. Returns list of (path, company_display, filing_type).
    """
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

            # Grab all HTML and PDF files (non-recursive — one level only)
            files = sorted(
                [f for f in subdir_path.iterdir()
                 if f.is_file() and f.suffix.lower() in (".html", ".htm", ".pdf")]
            )

            for f in files:
                # For benchmark, ftype is None → auto-detect
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
    print("  MULTI-COMPANY CAPEX — CHROMADB EMBEDDING PIPELINE")
    print("=" * 70)

    # --- ChromaDB ---
    print(f"\n📁 DB path: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="capex_docs",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   Collection: capex_docs | Existing docs: {collection.count()}")

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
    stats        = defaultdict(lambda: {"files": 0, "chunks": 0})
    company_stats = defaultdict(lambda: {"files": 0, "chunks": 0})

    for filepath, company, filing_type in all_files:
        print(f"  📄 [{company:<11}] {filepath.name:<65} ", end="", flush=True)

        text = extract(filepath)
        if not text or len(text.strip()) < 100:
            print("→ empty")
            continue

        # Use content-based detection for filing type and fiscal quarter
        if not filing_type or filing_type == "Other":
            filing_type = detect_filing_type(filepath, text)
        fy, q = get_fiscal_quarter(filepath, company, text)

        # Detect unit header from full text
        unit_header = ""
        unit_match = re.search(r"\((?:in|amounts?\s+in)\s+(thousands|millions|billions)\)", text[:5000], re.IGNORECASE)
        if unit_match:
            unit_header = unit_match.group(1).lower()

        # Structure-aware chunking
        structured_chunks = chunk_text_structured(text)
        if not structured_chunks:
            print("→ no chunks")
            continue

        print(f"→ {len(structured_chunks)} chunks", end="", flush=True)

        # Build batch with rich metadata
        ids, texts, metadatas = [], [], []
        safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", filepath.stem)
        parent_id = f"{company}_{safe_stem}"

        for i, (section_name, chunk) in enumerate(structured_chunks):
            doc_id = f"{company}_{safe_stem}_chunk{i:04d}"

            table_type = _detect_table_type(chunk)
            if not section_name:
                section_name = _detect_section(chunk)

            ids.append(doc_id)
            texts.append(chunk)
            metadatas.append({
                "company":        company,
                "source_file":    filepath.name,
                "filing_type":    filing_type,
                "fiscal_year":    fy,
                "quarter":        q,
                "chunk_index":    i,
                "total_chunks":   len(structured_chunks),
                "section_header": section_name,
                "table_type":     table_type,
                "table_context":  f"in {unit_header}" if unit_header else "",
                "parent_id":      parent_id,
            })

        # Embed + upsert in batches of 64
        for start in range(0, len(texts), 64):
            b_ids  = ids[start:start + 64]
            b_txt  = texts[start:start + 64]
            b_meta = metadatas[start:start + 64]

            embeddings = model.encode(b_txt, show_progress_bar=False)
            collection.upsert(
                ids=b_ids,
                embeddings=embeddings.tolist(),
                documents=b_txt,
                metadatas=b_meta,
            )

        total_chunks += len(chunks)
        stats[filing_type]["files"]  += 1
        stats[filing_type]["chunks"] += len(chunks)
        company_stats[company]["files"]  += 1
        company_stats[company]["chunks"] += len(chunks)
        print(" ✓")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  EMBEDDING COMPLETE")
    print("=" * 70)
    print(f"\n  Total docs in collection: {collection.count()}")
    print(f"  Total chunks embedded:    {total_chunks}\n")

    print(f"  BY COMPANY:")
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
        ("Single company",  "Flex capital expenditure property equipment"),
        ("Cross-company",   "capital expenditure purchases property equipment manufacturing"),
        ("Competitor",      "Jabil capital investment facility expansion"),
        ("Transcript",      "AI data center liquid cooling investment outlook"),
    ]

    for label, query in test_queries:
        print(f"\n  🔍 [{label}] \"{query}\"")
        q_emb = model.encode([query])
        results = collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            sim = round(1 - dist, 3)
            print(f"     sim={sim}  [{meta['company']:<11}] {meta['source_file']:<50} {meta['filing_type']} | {meta['fiscal_year']} {meta['quarter']}")
            print(f"     {doc[:120]}...")

    print("\n✅ ChromaDB ready for all companies. Next: RAG query layer.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_db()
