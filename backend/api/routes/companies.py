"""
Company API endpoints.
"""
from fastapi import APIRouter, HTTPException
from backend.core.config import COMPANIES, COMPANY_NAME_TO_TICKER
from backend.core.database import get_all_collections_stats

router = APIRouter()


@router.get("/companies")
async def list_companies():
    """
    List all tracked companies with their metadata.
    """
    stats = get_all_collections_stats()
    companies_data = []
    
    for ticker, info in COMPANIES.items():
        company_name = info["name"].split()[0]  # Get first word (Flex, Jabil, etc.)
        doc_count = stats.get("companies", {}).get(company_name, 0)
        
        companies_data.append({
            "ticker": ticker,
            "name": info["name"],
            "cik": info["cik"],
            "fiscal_year_end": info["fiscal_year_end"],
            "headquarters": info["headquarters"],
            "color": info["color"],
            "document_count": doc_count,
        })
    
    return {
        "companies": companies_data,
        "total": len(companies_data),
    }


@router.get("/companies/{ticker}")
async def get_company(ticker: str):
    """
    Get detailed information about a specific company.
    """
    ticker = ticker.upper()
    
    if ticker not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    
    info = COMPANIES[ticker]
    stats = get_all_collections_stats()

    # Get document count for this company
    company_name = info["name"].split()[0]
    doc_count = stats.get("companies", {}).get(company_name, 0)
    
    return {
        "ticker": ticker,
        **info,
        "document_count": doc_count,
    }


@router.get("/companies/{ticker}/filings")
async def get_company_filings(ticker: str, filing_type: str = None, limit: int = 20):
    """
    Get filings for a specific company from ChromaDB.
    """
    ticker = ticker.upper()
    
    if ticker not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    
    company_name = COMPANIES[ticker]["name"].split()[0]
    
    # Get documents from ChromaDB (per-company collection)
    from backend.core.database import get_company_collection
    collection = get_company_collection(company_name)

    where_filter = {}
    if filing_type:
        where_filter = {"filing_type": filing_type}

    get_kwargs: dict = {"include": ["metadatas"], "limit": limit * 10}
    if where_filter:
        get_kwargs["where"] = where_filter

    results = collection.get(**get_kwargs)
    
    # Deduplicate by source file
    seen_files = set()
    filings = []
    
    for meta in results.get("metadatas", []):
        source = meta.get("source_file", "")
        if source not in seen_files:
            seen_files.add(source)
            filings.append({
                "source_file": source,
                "filing_type": meta.get("filing_type", "Unknown"),
                "fiscal_year": meta.get("fiscal_year", "Unknown"),
                "quarter": meta.get("quarter", ""),
            })
        
        if len(filings) >= limit:
            break
    
    return {
        "company": company_name,
        "ticker": ticker,
        "filings": filings,
        "total": len(filings),
    }


@router.get("/companies/compare/{tickers}")
async def compare_companies(tickers: str):
    """
    Get comparison data for multiple companies.
    
    Args:
        tickers: Comma-separated list of tickers (e.g., "FLEX,JBL,CLS")
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    invalid = [t for t in ticker_list if t not in COMPANIES]
    if invalid:
        raise HTTPException(status_code=404, detail=f"Companies not found: {invalid}")
    
    stats = get_all_collections_stats()

    comparison = []
    for ticker in ticker_list:
        info = COMPANIES[ticker]
        company_name = info["name"].split()[0]
        doc_count = stats.get("companies", {}).get(company_name, 0)
        
        comparison.append({
            "ticker": ticker,
            "name": info["name"],
            "headquarters": info["headquarters"],
            "fiscal_year_end": info["fiscal_year_end"],
            "color": info["color"],
            "document_count": doc_count,
        })
    
    return {
        "companies": comparison,
        "total": len(comparison),
    }
