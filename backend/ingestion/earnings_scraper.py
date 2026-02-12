"""
Earnings transcript and presentation scraper.
Scrapes from publicly available sources.
"""
import re
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from backend.core.config import COMPANIES, DATA_DIR


class EarningsCalendar:
    """
    Tracks earnings dates and provides calendar functionality.
    """
    
    # Known earnings calendar (could be updated from external API)
    EARNINGS_SCHEDULE = {
        "FLEX": {
            "fiscal_year_end": "March",
            "typical_quarters": ["Q1 July", "Q2 October", "Q3 January", "Q4 April"],
        },
        "JBL": {
            "fiscal_year_end": "August", 
            "typical_quarters": ["Q1 December", "Q2 March", "Q3 June", "Q4 September"],
        },
        "CLS": {
            "fiscal_year_end": "December",
            "typical_quarters": ["Q1 April", "Q2 July", "Q3 October", "Q4 February"],
        },
        "BHE": {
            "fiscal_year_end": "December",
            "typical_quarters": ["Q1 April", "Q2 July", "Q3 October", "Q4 February"],
        },
        "SANM": {
            "fiscal_year_end": "September",
            "typical_quarters": ["Q1 January", "Q2 April", "Q3 July", "Q4 October"],
        },
    }
    
    @classmethod
    def get_company_schedule(cls, ticker: str) -> dict:
        """Get earnings schedule for a company."""
        return cls.EARNINGS_SCHEDULE.get(ticker, {})
    
    @classmethod
    def get_all_schedules(cls) -> dict:
        """Get all company earnings schedules."""
        return cls.EARNINGS_SCHEDULE


class TranscriptScraper:
    """
    Scrapes earnings call transcripts from public sources.
    Note: In production, you'd use a paid API like Seeking Alpha or similar.
    """
    
    def __init__(self):
        self.data_dir = DATA_DIR / "earnings_transcripts"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    
    async def search_company_news(self, ticker: str, query: str = "earnings") -> list[dict]:
        """
        Search for company earnings news using web search.
        Returns links to potential transcript sources.
        """
        from backend.rag.web_search import search_web
        
        search_query = f"{ticker} {COMPANIES.get(ticker, {}).get('name', '')} {query} transcript call"
        results = await search_web(search_query, count=10)
        
        # Filter for likely transcript sources
        transcript_sources = []
        for result in results:
            url_lower = result["url"].lower()
            title_lower = result["title"].lower()
            
            # Look for transcript-related content
            if any(term in url_lower or term in title_lower for term in 
                   ["transcript", "earnings", "call", "conference"]):
                transcript_sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "description": result["description"],
                    "source": self._identify_source(result["url"]),
                })
        
        return transcript_sources
    
    def _identify_source(self, url: str) -> str:
        """Identify the source website."""
        if "seekingalpha" in url.lower():
            return "Seeking Alpha"
        elif "fool" in url.lower():
            return "Motley Fool"
        elif "yahoo" in url.lower():
            return "Yahoo Finance"
        elif "bloomberg" in url.lower():
            return "Bloomberg"
        elif "reuters" in url.lower():
            return "Reuters"
        elif "cnbc" in url.lower():
            return "CNBC"
        else:
            return "Other"
    
    async def get_investor_relations_url(self, ticker: str) -> Optional[str]:
        """Get the investor relations page URL for a company."""
        ir_urls = {
            "FLEX": "https://investors.flex.com/",
            "JBL": "https://investors.jabil.com/",
            "CLS": "https://www.celestica.com/about-us/investor-relations/",
            "BHE": "https://ir.bench.com/",
            "SANM": "https://www.sanmina.com/investors/",
        }
        return ir_urls.get(ticker)
    
    async def fetch_press_releases(self, ticker: str) -> list[dict]:
        """
        Fetch recent press releases from investor relations.
        """
        ir_url = await self.get_investor_relations_url(ticker)
        if not ir_url:
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(ir_url, headers=self.headers, timeout=15.0)
                response.raise_for_status()
        except Exception as e:
            print(f"Error fetching IR page for {ticker}: {e}")
            return []
        
        # Parse the page
        soup = BeautifulSoup(response.text, "lxml")
        
        # Look for press release links
        releases = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            
            # Filter for earnings-related content
            if any(term in text.lower() for term in 
                   ["earnings", "quarter", "results", "q1", "q2", "q3", "q4"]):
                full_url = href if href.startswith("http") else f"{ir_url.rstrip('/')}/{href.lstrip('/')}"
                releases.append({
                    "title": text,
                    "url": full_url,
                    "ticker": ticker,
                })
        
        return releases[:10]  # Return top 10


class EarningsDataManager:
    """
    Manages earnings data collection and storage.
    """
    
    def __init__(self):
        self.scraper = TranscriptScraper()
        self.calendar = EarningsCalendar()
    
    async def get_company_earnings_info(self, ticker: str) -> dict:
        """Get comprehensive earnings info for a company."""
        company = COMPANIES.get(ticker, {})
        schedule = self.calendar.get_company_schedule(ticker)
        
        # Search for recent transcripts
        transcript_sources = await self.scraper.search_company_news(ticker)
        
        return {
            "ticker": ticker,
            "company_name": company.get("name", "Unknown"),
            "fiscal_year_end": schedule.get("fiscal_year_end", "Unknown"),
            "typical_quarters": schedule.get("typical_quarters", []),
            "recent_transcript_sources": transcript_sources[:5],
            "ir_url": await self.scraper.get_investor_relations_url(ticker),
        }
    
    async def get_all_earnings_info(self) -> list[dict]:
        """Get earnings info for all tracked companies."""
        results = []
        for ticker in COMPANIES.keys():
            info = await self.get_company_earnings_info(ticker)
            results.append(info)
        return results


# Convenience functions
async def search_earnings_transcripts(ticker: str) -> list[dict]:
    """Search for earnings transcripts for a company."""
    scraper = TranscriptScraper()
    return await scraper.search_company_news(ticker)


async def get_earnings_calendar() -> dict:
    """Get the earnings calendar for all companies."""
    return EarningsCalendar.get_all_schedules()


async def get_upcoming_earnings() -> list[dict]:
    """Get upcoming earnings dates based on typical schedule patterns."""
    from datetime import datetime
    
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    upcoming = []
    
    for ticker, schedule in EarningsCalendar.EARNINGS_SCHEDULE.items():
        company = COMPANIES.get(ticker, {})
        quarters = schedule.get("typical_quarters", [])
        
        for q in quarters:
            # Parse quarter info (e.g., "Q1 July")
            parts = q.split()
            if len(parts) >= 2:
                quarter = parts[0]
                month_name = parts[1]
                
                # Map month name to number
                month_map = {
                    "January": 1, "February": 2, "March": 3, "April": 4,
                    "May": 5, "June": 6, "July": 7, "August": 8,
                    "September": 9, "October": 10, "November": 11, "December": 12
                }
                month_num = month_map.get(month_name, 0)
                
                # Check if this quarter's earnings is upcoming (within next 3 months)
                if month_num:
                    diff = month_num - current_month
                    if diff < 0:
                        diff += 12
                    
                    if diff <= 3:
                        year = current_year if month_num >= current_month else current_year + 1
                        upcoming.append({
                            "ticker": ticker,
                            "company_name": company.get("name", ticker),
                            "quarter": quarter,
                            "expected_month": month_name,
                            "expected_year": year,
                            "days_until": diff * 30,  # Rough estimate
                        })
    
    # Sort by days until
    upcoming.sort(key=lambda x: x["days_until"])
    
    return upcoming


async def scrape_earnings_highlights(ticker: str) -> dict:
    """Get earnings highlights from recent filings and web search."""
    from backend.rag.web_search import search_web
    from backend.rag.retriever import search_documents
    
    company = COMPANIES.get(ticker, {})
    company_name = company.get("name", ticker)
    
    # Search for recent earnings news
    search_query = f"{company_name} earnings results 2024 2025"
    web_results = await search_web(search_query, count=5)
    
    # Get earnings-related documents from our database
    doc_results = search_documents(
        query=f"{company_name} quarterly earnings results revenue profit",
        company_filter=company_name.split()[0],  # First word of company name
        n_results=5
    )
    
    highlights = {
        "ticker": ticker,
        "company_name": company_name,
        "web_news": [
            {
                "title": r["title"],
                "url": r["url"],
                "description": r["description"][:200] if r.get("description") else ""
            }
            for r in web_results
        ],
        "document_highlights": [
            {
                "source": d.get("source", "Unknown"),
                "fiscal_year": d.get("fiscal_year", "Unknown"),
                "preview": d["content"][:300] + "..."
            }
            for d in doc_results
        ]
    }
    
    return highlights
