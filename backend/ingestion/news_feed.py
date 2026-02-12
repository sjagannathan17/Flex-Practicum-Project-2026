"""
News feed integration for competitive intelligence.
Aggregates news from multiple sources for tracked companies.
"""
import asyncio
from datetime import datetime
from typing import Optional
from collections import defaultdict

from backend.core.config import COMPANIES
from backend.rag.web_search import search_web


class NewsFeed:
    """
    Aggregates and manages company news from multiple sources.
    """
    
    # News categories for filtering (all lowercase for case-insensitive matching)
    CATEGORIES = {
        "earnings": ["earnings", "quarterly", "revenue", "profit", "results", "guidance", "eps", "beat", "miss", "forecast", "outlook", "fiscal"],
        "ai": ["artificial intelligence", "ai", "machine learning", "data center", "gpu", "nvidia", "hyperscale", "cloud computing", "generative", "llm", "semiconductor", "chip"],
        "capex": ["capital expenditure", "capex", "investment", "expansion", "factory", "facility", "million", "billion", "spending", "infrastructure"],
        "strategy": ["acquisition", "merger", "partnership", "restructuring", "strategy", "deal", "agreement", "acquire", "divest"],
        "operations": ["manufacturing", "supply chain", "production", "capacity", "logistics", "assembly", "plant"],
    }
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour cache
    
    async def get_company_news(self, ticker: str, category: Optional[str] = None, count: int = 10) -> dict:
        """
        Get recent news for a specific company.
        
        Args:
            ticker: Company ticker symbol
            category: Optional category filter (earnings, ai, capex, strategy, operations)
            count: Number of results to return
        """
        company = COMPANIES.get(ticker, {})
        company_name = company.get("name", ticker)
        
        # Build search query
        search_terms = [company_name]
        
        if category and category in self.CATEGORIES:
            search_terms.extend(self.CATEGORIES[category][:2])  # Add category terms
        
        query = " ".join(search_terms) + " news"
        
        # Search web for news
        results = await search_web(query, count=count * 2)
        
        # Process and categorize results
        news_items = []
        for result in results:
            categories = self._categorize_content(result["title"] + " " + result.get("description", ""))
            
            # If category filter is specified, add it to categories if search terms match
            # This ensures we return relevant results even if categorization misses them
            if category and category not in categories:
                content_lower = (result["title"] + " " + result.get("description", "")).lower()
                if any(kw.lower() in content_lower for kw in self.CATEGORIES.get(category, [])):
                    categories.append(category)
            
            item = {
                "title": result["title"],
                "url": result["url"],
                "description": result.get("description", ""),
                "source": self._extract_source(result["url"]),
                "categories": categories,
                "relevance_score": self._calculate_relevance(result, company_name),
            }
            
            news_items.append(item)
        
        # Sort by relevance
        news_items.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "category_filter": category,
            "news": news_items[:count],
            "total_found": len(news_items),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def get_industry_news(self, count: int = 15) -> dict:
        """
        Get industry-wide EMS/electronics manufacturing news.
        """
        queries = [
            "EMS electronics manufacturing services news",
            "contract manufacturing electronics industry",
            "data center manufacturing supply chain",
        ]
        
        all_results = []
        for query in queries:
            results = await search_web(query, count=5)
            all_results.extend(results)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result["url"] not in seen_urls:
                seen_urls.add(result["url"])
                unique_results.append({
                    "title": result["title"],
                    "url": result["url"],
                    "description": result.get("description", ""),
                    "source": self._extract_source(result["url"]),
                    "categories": self._categorize_content(result["title"] + " " + result.get("description", "")),
                })
        
        return {
            "news": unique_results[:count],
            "total_found": len(unique_results),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def get_competitor_comparison_news(self) -> dict:
        """
        Get comparative news mentioning multiple competitors.
        """
        company_names = [c["name"].split()[0] for c in COMPANIES.values()]
        
        # Search for comparative coverage
        query = " OR ".join(company_names) + " EMS comparison analysis"
        results = await search_web(query, count=10)
        
        # Find articles mentioning multiple companies
        comparative_news = []
        for result in results:
            content = (result["title"] + " " + result.get("description", "")).lower()
            mentioned = [name for name in company_names if name.lower() in content]
            
            if len(mentioned) >= 2:
                comparative_news.append({
                    "title": result["title"],
                    "url": result["url"],
                    "description": result.get("description", ""),
                    "source": self._extract_source(result["url"]),
                    "companies_mentioned": mentioned,
                })
        
        return {
            "comparative_news": comparative_news,
            "total_found": len(comparative_news),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def get_all_companies_news(self, count_per_company: int = 3) -> dict:
        """
        Get news for all tracked companies.
        """
        all_news = {}
        
        for ticker in COMPANIES.keys():
            news = await self.get_company_news(ticker, count=count_per_company)
            all_news[ticker] = news
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        return {
            "companies": all_news,
            "total_companies": len(all_news),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _extract_source(self, url: str) -> str:
        """Extract source name from URL."""
        domain_sources = {
            "reuters.com": "Reuters",
            "bloomberg.com": "Bloomberg",
            "wsj.com": "Wall Street Journal",
            "ft.com": "Financial Times",
            "cnbc.com": "CNBC",
            "yahoo.com": "Yahoo Finance",
            "seekingalpha.com": "Seeking Alpha",
            "fool.com": "Motley Fool",
            "marketwatch.com": "MarketWatch",
            "barrons.com": "Barron's",
            "businesswire.com": "Business Wire",
            "prnewswire.com": "PR Newswire",
            "globenewswire.com": "GlobeNewswire",
        }
        
        url_lower = url.lower()
        for domain, source in domain_sources.items():
            if domain in url_lower:
                return source
        
        # Extract domain as fallback
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return "Unknown"
    
    def _categorize_content(self, content: str) -> list[str]:
        """Categorize content based on keywords."""
        content_lower = content.lower()
        categories = []
        
        for category, keywords in self.CATEGORIES.items():
            # Check if any keyword matches (case-insensitive)
            if any(kw.lower() in content_lower for kw in keywords):
                categories.append(category)
        
        # Also check for common patterns that indicate specific categories
        if not categories:
            # Check for earnings-related content
            if any(term in content_lower for term in ['q1', 'q2', 'q3', 'q4', 'quarter', 'fiscal', 'fy2', 'fy2025', 'fy2024', 'eps', 'beat', 'miss']):
                categories.append('earnings')
            # Check for AI/tech content
            if any(term in content_lower for term in ['nvidia', 'gpu', 'hyperscale', 'cloud', 'generative', 'llm', 'chip', 'semiconductor']):
                categories.append('ai')
            # Check for investment content
            if any(term in content_lower for term in ['million', 'billion', 'invest', 'expand', 'new facility', 'build']):
                categories.append('capex')
            # Check for M&A/strategy
            if any(term in content_lower for term in ['acquire', 'deal', 'agreement', 'partner', 'announce']):
                categories.append('strategy')
        
        return categories if categories else ["general"]
    
    def _calculate_relevance(self, result: dict, company_name: str) -> float:
        """Calculate relevance score for a result."""
        score = 0.0
        content = (result["title"] + " " + result.get("description", "")).lower()
        company_lower = company_name.lower()
        
        # Company name mentioned
        if company_lower in content:
            score += 0.5
        
        # Recent news indicators
        if any(term in content for term in ["today", "announces", "reports", "2024", "2025"]):
            score += 0.2
        
        # Business relevance
        if any(term in content for term in ["earnings", "revenue", "investment", "strategy"]):
            score += 0.3
        
        return min(score, 1.0)


# API routes
from fastapi import APIRouter, HTTPException

router = APIRouter()
_news_feed = NewsFeed()


@router.get("/news/company/{ticker}")
async def get_company_news(ticker: str, category: Optional[str] = None, count: int = 10):
    """Get news for a specific company."""
    ticker_upper = ticker.upper()
    if ticker_upper not in COMPANIES:
        raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
    
    return await _news_feed.get_company_news(ticker_upper, category, count)


@router.get("/news/industry")
async def get_industry_news(count: int = 15):
    """Get industry-wide EMS news."""
    return await _news_feed.get_industry_news(count)


@router.get("/news/comparative")
async def get_comparative_news():
    """Get news comparing multiple companies."""
    return await _news_feed.get_competitor_comparison_news()


@router.get("/news/all")
async def get_all_news(count_per_company: int = 3):
    """Get news for all tracked companies."""
    return await _news_feed.get_all_companies_news(count_per_company)


# Convenience functions
async def get_company_news(ticker: str, category: Optional[str] = None) -> dict:
    """Get news for a company."""
    return await _news_feed.get_company_news(ticker, category)


async def get_latest_industry_news() -> dict:
    """Get latest industry news."""
    return await _news_feed.get_industry_news()
