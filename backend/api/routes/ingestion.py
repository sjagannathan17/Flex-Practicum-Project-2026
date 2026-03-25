"""
API routes for data ingestion management.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.core.config import BASE_DIR, DATA_DIR, COMPANIES
from backend.ingestion.sec_downloader import SECDownloader
from backend.ingestion.scheduler import (
    start_scheduler,
    stop_scheduler,
    get_scheduler_status,
    run_manual_check,
)
from backend.ingestion.processor import process_new_filings, process_filing

router = APIRouter()

# ── Knowledge-base job state (module-level, survives across requests) ──────
_reindex_job: dict = {
    "status": "idle",        # idle | running | done | error
    "current_file": "",
    "processed": 0,
    "total": 0,
    "chunks_added": 0,
    "started_at": None,
    "completed_at": None,
    "error": "",
}
_KB_META_FILE = DATA_DIR / "kb_last_indexed.json"

# Ticker → display name
_TICKER_TO_NAME = {info["name"].split()[0]: info["name"].split()[0] for info in COMPANIES.values()}
_TICKER_MAP = {
    "FLEX": "Flex",
    "JBL": "Jabil",
    "CLS": "Celestica",
    "BHE": "Benchmark",
    "SANM": "Sanmina",
}
# Company directory name → display name
_DIR_TO_COMPANY = {
    "Flex": "Flex",
    "Jabil": "Jabil",
    "Celestica": "Celestica",
    "Benchmark": "Benchmark",
    "Sanmina": "Sanmina",
}


def _infer_filing_type(path: Path) -> str:
    s = (str(path) + path.stem).lower()
    if "10-k" in s or "10k" in s or "10_k" in s:
        return "10-K"
    if "10-q" in s or "10q" in s or "10_q" in s:
        return "10-Q"
    if "8-k" in s or "8k" in s:
        return "8-K"
    return "OTHER"


def _scan_all_filings() -> list[dict]:
    """Return list of dicts {filepath, company, filing_type} for all local filings."""
    results = []
    seen: set[str] = set()

    # 1. Company-specific directories in project root
    for dir_name, company in _DIR_TO_COMPANY.items():
        company_dir = BASE_DIR / dir_name
        if not company_dir.exists():
            continue
        for fp in company_dir.rglob("*"):
            if fp.suffix.lower() in (".html", ".htm", ".pdf") and str(fp) not in seen:
                seen.add(str(fp))
                results.append({
                    "filepath": fp,
                    "company": company,
                    "filing_type": _infer_filing_type(fp),
                    "fiscal_year": "Unknown",
                    "quarter": "",
                })

    # 2. EDGAR-downloaded files in data/sec_filings/{TICKER}/
    sec_dir = DATA_DIR / "sec_filings"
    if sec_dir.exists():
        for ticker_dir in sec_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            company = _TICKER_MAP.get(ticker_dir.name.upper(), ticker_dir.name)
            for fp in ticker_dir.rglob("*"):
                if fp.suffix.lower() in (".html", ".htm", ".pdf") and str(fp) not in seen:
                    seen.add(str(fp))
                    results.append({
                        "filepath": fp,
                        "company": company,
                        "filing_type": _infer_filing_type(fp),
                        "fiscal_year": "Unknown",
                        "quarter": "",
                    })

    return results


def _run_reindex():
    """Background thread: re-index all local filings into ChromaDB."""
    global _reindex_job
    _reindex_job["status"] = "running"
    _reindex_job["started_at"] = datetime.now(timezone.utc).isoformat()
    _reindex_job["error"] = ""
    _reindex_job["chunks_added"] = 0
    _reindex_job["processed"] = 0

    try:
        filings = _scan_all_filings()
        _reindex_job["total"] = len(filings)

        total_chunks = 0
        for i, filing in enumerate(filings, 1):
            fp: Path = filing["filepath"]
            _reindex_job["current_file"] = fp.name
            _reindex_job["processed"] = i - 1
            try:
                chunks = process_filing(
                    filepath=fp,
                    company=filing["company"],
                    filing_type=filing["filing_type"],
                    fiscal_year=filing["fiscal_year"],
                    quarter=filing["quarter"],
                )
                total_chunks += chunks
            except Exception as e:
                print(f"[reindex] Error on {fp.name}: {e}")

        _reindex_job["chunks_added"] = total_chunks
        _reindex_job["processed"] = len(filings)
        _reindex_job["current_file"] = ""
        _reindex_job["status"] = "done"
        _reindex_job["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Persist metadata
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _KB_META_FILE.write_text(json.dumps({
            "last_indexed": _reindex_job["completed_at"],
            "chunks_added": total_chunks,
            "files_processed": len(filings),
        }))

    except Exception as e:
        _reindex_job["status"] = "error"
        _reindex_job["error"] = str(e)
        _reindex_job["completed_at"] = datetime.now(timezone.utc).isoformat()


# ── Existing routes ────────────────────────────────────────────────────────

class FilingCheckRequest(BaseModel):
    """Request body for filing check."""
    days_back: int = 30
    filing_types: list[str] = ["10-K", "10-Q", "8-K"]


@router.get("/ingestion/status")
async def get_ingestion_status():
    """Get current ingestion scheduler status."""
    scheduler_status = get_scheduler_status()
    downloader = SECDownloader()
    download_stats = downloader.get_download_stats()
    return {
        "scheduler": scheduler_status,
        "downloads": download_stats,
    }


@router.post("/ingestion/start-scheduler")
async def api_start_scheduler():
    try:
        start_scheduler()
        return {"status": "started", "scheduler": get_scheduler_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingestion/stop-scheduler")
async def api_stop_scheduler():
    try:
        stop_scheduler()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingestion/check-filings")
async def check_filings(
    background_tasks: BackgroundTasks,
    request: FilingCheckRequest,
):
    """Manually trigger a check for new SEC filings (background)."""
    async def check_and_process():
        downloader = SECDownloader()
        new_filings = await downloader.check_and_download_new_filings(
            filing_types=request.filing_types,
            days_back=request.days_back,
        )
        if new_filings:
            await process_new_filings(new_filings)

    background_tasks.add_task(check_and_process)
    return {
        "status": "checking",
        "message": f"Checking for filings from last {request.days_back} days",
        "filing_types": request.filing_types,
    }


@router.get("/ingestion/filings")
async def get_available_filings(ticker: Optional[str] = None, days_back: int = 90):
    downloader = SECDownloader()
    if ticker:
        filings = await downloader.get_company_filings(ticker.upper(), days_back=days_back)
        return {"ticker": ticker, "filings": filings}
    all_filings = {}
    for company_ticker in COMPANIES.keys():
        filings = await downloader.get_company_filings(company_ticker, days_back=days_back)
        all_filings[company_ticker] = filings
    return {"filings": all_filings}


@router.post("/ingestion/download-filing")
async def download_specific_filing(
    ticker: str,
    form: str,
    filing_date: str,
    background_tasks: BackgroundTasks,
):
    downloader = SECDownloader()
    filings = await downloader.get_company_filings(ticker.upper(), days_back=365)
    target = next(
        (f for f in filings if f["form"] == form and f["filing_date"] == filing_date),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail=f"Filing not found: {ticker} {form} {filing_date}")
    if target["already_downloaded"]:
        return {"status": "already_downloaded", "filing": target}

    async def download_and_process():
        path = await downloader.download_filing(target)
        if path:
            target["local_path"] = str(path)
            await process_new_filings([target])

    background_tasks.add_task(download_and_process)
    return {"status": "downloading", "filing": target}


# ── Knowledge-base endpoints ───────────────────────────────────────────────

@router.get("/ingestion/kb-status")
async def get_kb_status():
    """Return knowledge-base metadata: doc count, last indexed, total local files."""
    try:
        import chromadb
        from backend.core.config import CHROMADB_PATH
        client = chromadb.PersistentClient(path=CHROMADB_PATH)
        doc_count = sum(c.count() for c in client.list_collections())
    except Exception:
        doc_count = 0

    last_indexed: Optional[str] = None
    chunks_at_last_index: int = 0
    if _KB_META_FILE.exists():
        try:
            meta = json.loads(_KB_META_FILE.read_text())
            last_indexed = meta.get("last_indexed")
            chunks_at_last_index = meta.get("chunks_added", 0)
        except Exception:
            pass

    total_files = len(_scan_all_filings())

    return {
        "doc_count": doc_count,
        "last_indexed": last_indexed,
        "total_local_files": total_files,
        "chunks_at_last_index": chunks_at_last_index,
    }


@router.post("/ingestion/reindex-all")
async def reindex_all():
    """Start a full re-index of all local filing files into ChromaDB."""
    if _reindex_job["status"] == "running":
        raise HTTPException(status_code=409, detail="Re-index already in progress")
    t = threading.Thread(target=_run_reindex, daemon=True)
    t.start()
    return {"status": "started", "total_files": len(_scan_all_filings())}


@router.get("/ingestion/reindex-progress")
async def get_reindex_progress():
    """Poll current re-index job progress."""
    return dict(_reindex_job)
