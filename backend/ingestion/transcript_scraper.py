"""
Earnings call transcript and IR press release scraper.

Sources:
  1. Motley Fool — free public earnings transcripts
  2. Company IR websites — press releases filtered by relevance keywords

Documents are ingested directly into the per-company ChromaDB collections
using the same embedding model as the rest of the pipeline.
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from backend.core.config import DATA_DIR
from backend.core.database import get_company_collection, embed_texts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

TICKERS = ["FLEX", "JBL", "CLS", "BHE", "SANM"]

# Maps API ticker → ChromaDB company name (used by get_company_collection)
TICKER_TO_COMPANY: dict[str, str] = {
    "FLEX": "Flex",
    "JBL": "Jabil",
    "CLS": "Celestica",
    "BHE": "Benchmark",
    "SANM": "Sanmina",
}

# Motley Fool slug per ticker
COMPANY_FOOL_NAMES: dict[str, str] = {
    "FLEX": "flex",
    "JBL": "jabil",
    "CLS": "celestica",
    "BHE": "benchmark-electronics",
    "SANM": "sanmina",
}

# IR news-release pages per ticker
IR_PAGES: dict[str, str] = {
    "FLEX": "https://ir.flex.com/news-releases",
    "JBL": "https://investors.jabil.com/news-releases",
    "CLS": "https://ir.celestica.com/news-releases",
    "BHE": "https://ir.bench.com/news-releases",
    "SANM": "https://ir.sanmina.com/news-releases",
}

RELEVANT_KEYWORDS = [
    "earnings", "revenue", "quarterly", "results", "guidance",
    "AI", "data center", "artificial intelligence", "hyperscaler",
    "capacity", "facility", "expansion", "partnership", "contract",
    "liquid cooling", "power", "infrastructure",
]

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# CHROMADB INGESTION
# ---------------------------------------------------------------------------

def _ingest_document(doc: dict) -> int:
    """
    Chunk and upsert a single document into the appropriate per-company
    ChromaDB collection.  Returns number of chunks written.

    doc must contain: ticker, source, type, text
    doc may contain: date, url, title
    """
    text = doc.get("text", "").strip()
    if len(text) < 100:
        return 0

    ticker = doc["ticker"]
    company = TICKER_TO_COMPANY.get(ticker)
    if not company:
        logger.warning(f"[transcript_scraper] Unknown ticker: {ticker}")
        return 0

    collection = get_company_collection(company)
    chunks = _chunk_text(text)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(
            f"{ticker}_{doc['source']}_{i}_{chunk[:50]}".encode()
        ).hexdigest()
        ids.append(uid)
        documents.append(chunk)
        metadatas.append({
            "ticker": ticker,
            "company": company,
            "source": doc["source"],
            "filing_type": doc.get("type", "unknown"),
            "date": doc.get("date", "unknown"),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "chunk_index": i,
        })

    # Embed with the project-wide model and upsert
    embeddings = embed_texts(documents)
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    return len(chunks)


# ---------------------------------------------------------------------------
# SOURCE 2 — MOTLEY FOOL
# ---------------------------------------------------------------------------

def fetch_motleyfool_transcript(ticker: str, quarters_back: int = 4) -> list[dict]:
    """
    Fetch the last *quarters_back* earnings transcripts for *ticker* from
    Motley Fool.  Returns list of document dicts ready for ingestion.
    """
    company_slug = COMPANY_FOOL_NAMES.get(ticker)
    if not company_slug:
        return []

    search_url = (
        "https://www.fool.com/search/solr.aspx?"
        f"q={company_slug}+earnings+transcript&"
        "filter=ArticleTypes%3AEarningsCallTranscript"
    )

    results: list[dict] = []

    try:
        resp = requests.get(search_url, headers=_REQUEST_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            if "earnings/call-transcripts" in href and company_slug in href:
                links.append(href)

        links = list(dict.fromkeys(links))  # deduplicate, preserve order

        for link in links[:quarters_back]:
            try:
                full_url = (
                    link if link.startswith("http")
                    else f"https://www.fool.com{link}"
                )
                page = requests.get(full_url, headers=_REQUEST_HEADERS, timeout=15)
                page_soup = BeautifulSoup(page.text, "html.parser")

                article = page_soup.find("div", class_="article-body") or page_soup.find("article")
                if not article:
                    continue

                text = article.get_text(separator="\n", strip=True)
                if len(text) < 200:
                    continue

                date_tag = page_soup.find("time")
                date = date_tag.get("datetime", "unknown") if date_tag else "unknown"

                title_tag = page_soup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else ""

                results.append({
                    "ticker": ticker,
                    "source": "motleyfool",
                    "type": "earnings_transcript",
                    "date": date,
                    "text": text,
                    "url": full_url,
                    "title": title,
                })

                logger.info(f"[Motley Fool] {ticker}: fetched transcript {full_url}")
                time.sleep(2)

            except Exception as e:
                logger.warning(f"[Motley Fool] Failed to fetch {link}: {e}")

    except Exception as e:
        logger.warning(f"[Motley Fool] Search failed for {ticker}: {e}")

    return results


# ---------------------------------------------------------------------------
# SOURCE 3 — IR PRESS RELEASES
# ---------------------------------------------------------------------------

def fetch_ir_press_releases(ticker: str, max_items: int = 10) -> list[dict]:
    """
    Scrape recent relevant press releases from the company's IR page.
    Returns list of document dicts ready for ingestion.
    """
    url = IR_PAGES.get(ticker)
    if not url:
        return []

    results: list[dict] = []

    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        base_url = "/".join(url.split("/")[:3])

        for a in soup.find_all("a", href=True)[:60]:
            if len(results) >= max_items:
                break

            title_text = a.get_text(strip=True)
            title_lower = title_text.lower()

            if not any(kw.lower() in title_lower for kw in RELEVANT_KEYWORDS):
                continue

            link: str = a["href"]
            if not link.startswith("http"):
                link = base_url + link

            try:
                page = requests.get(link, headers=_REQUEST_HEADERS, timeout=15)
                page_soup = BeautifulSoup(page.text, "html.parser")

                content = (
                    page_soup.find("div", class_="press-release-body")
                    or page_soup.find("main")
                    or page_soup.find("article")
                )
                if not content:
                    continue

                text = content.get_text(separator="\n", strip=True)
                if len(text) < 200:
                    continue

                date_tag = page_soup.find("time")
                date = date_tag.get("datetime", "unknown") if date_tag else "unknown"

                results.append({
                    "ticker": ticker,
                    "source": "ir_website",
                    "type": "press_release",
                    "date": date,
                    "text": text,
                    "url": link,
                    "title": title_text,
                })

                logger.info(f"[IR] {ticker}: fetched press release: {title_text}")
                time.sleep(1)

            except Exception as e:
                logger.warning(f"[IR] Failed to fetch {link}: {e}")

    except Exception as e:
        logger.warning(f"[IR] Failed to load IR page for {ticker}: {e}")

    return results


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_transcript_ingestion() -> int:
    """
    Full weekly pipeline: fetch Motley Fool transcripts + IR press releases
    for all 5 companies and ingest into ChromaDB.

    Returns total chunks added.
    """
    logger.info("[transcript_scraper] Starting full transcript ingestion...")
    total_chunks = 0

    for ticker in TICKERS:
        logger.info(f"\n[transcript_scraper] ── {ticker} ──")

        # Motley Fool transcripts
        logger.info(f"[transcript_scraper] {ticker}: fetching Motley Fool transcripts...")
        transcripts = fetch_motleyfool_transcript(ticker, quarters_back=4)
        for doc in transcripts:
            n = _ingest_document(doc)
            total_chunks += n
            logger.info(f"[transcript_scraper] {ticker}: transcript → {n} chunks")

        # IR press releases
        logger.info(f"[transcript_scraper] {ticker}: fetching IR press releases...")
        press_releases = fetch_ir_press_releases(ticker, max_items=10)
        for doc in press_releases:
            n = _ingest_document(doc)
            total_chunks += n
        logger.info(
            f"[transcript_scraper] {ticker}: {len(press_releases)} press releases → "
            f"{sum(0 for _ in press_releases)} chunks (varies by length)"
        )

        time.sleep(3)  # polite pause between companies

    _persist_ingestion_metadata(total_chunks)
    logger.info(f"[transcript_scraper] Done. Total chunks added: {total_chunks}")
    return total_chunks


def run_ir_only_ingestion() -> int:
    """
    Quick daily pipeline: fetch only IR press releases (no transcripts).
    Used for the weekday 8 AM job to catch overnight announcements.

    Returns total chunks added.
    """
    logger.info("[transcript_scraper] Starting daily IR press release check...")
    total_chunks = 0

    for ticker in TICKERS:
        press_releases = fetch_ir_press_releases(ticker, max_items=5)
        for doc in press_releases:
            n = _ingest_document(doc)
            total_chunks += n
        if press_releases:
            logger.info(f"[transcript_scraper] {ticker}: {len(press_releases)} IR items")
        time.sleep(2)

    logger.info(f"[transcript_scraper] Daily IR check done. Chunks added: {total_chunks}")
    return total_chunks


def _persist_ingestion_metadata(chunks_added: int) -> None:
    """Write a timestamp file so the KB status endpoint can display last_indexed."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        meta_file = DATA_DIR / "kb_last_indexed.json"
        existing: dict = {}
        if meta_file.exists():
            try:
                existing = json.loads(meta_file.read_text())
            except Exception:
                pass
        existing.update({
            "last_indexed": datetime.now(timezone.utc).isoformat(),
            "chunks_added": chunks_added,
            "source": "transcript_scraper",
        })
        meta_file.write_text(json.dumps(existing, indent=2))
    except Exception as e:
        logger.warning(f"[transcript_scraper] Could not persist metadata: {e}")
