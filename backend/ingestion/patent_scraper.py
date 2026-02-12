"""
USPTO Patent Scraper for competitive intelligence.
Tracks patent filings for EMS companies to identify innovation trends.
"""
import re
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import httpx

from backend.core.config import COMPANIES
from backend.rag.web_search import search_web


# Patent categories relevant to EMS industry
PATENT_CATEGORIES = {
    "ai_ml": ["artificial intelligence", "machine learning", "neural network", "deep learning", "AI system"],
    "automation": ["robotic", "automation", "automated assembly", "pick and place", "automated inspection"],
    "manufacturing": ["PCB", "circuit board", "soldering", "surface mount", "electronics manufacturing"],
    "power_systems": ["power supply", "power management", "battery", "charging", "energy storage"],
    "connectivity": ["5G", "wireless", "antenna", "RF", "signal processing"],
    "sensors": ["sensor", "IoT", "monitoring", "detection", "measurement"],
    "thermal": ["thermal management", "cooling", "heat sink", "heat dissipation"],
    "materials": ["advanced materials", "composite", "coating", "substrate"],
}

# Company assignee names for USPTO search
COMPANY_ASSIGNEES = {
    "Flex": ["Flex Ltd", "Flextronics", "Flex International"],
    "Jabil": ["Jabil Inc", "Jabil Circuit"],
    "Celestica": ["Celestica Inc", "Celestica International"],
    "Benchmark": ["Benchmark Electronics"],
    "Sanmina": ["Sanmina Corporation", "Sanmina-SCI"],
}


class PatentScraper:
    """Scrapes and analyzes patent data for EMS companies."""
    
    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 3600  # 1 hour cache
    
    def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached data if still valid."""
        if key in self._cache:
            if datetime.now().timestamp() - self._cache_time.get(key, 0) < self._cache_ttl:
                return self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: dict):
        """Cache data."""
        self._cache[key] = data
        self._cache_time[key] = datetime.now().timestamp()
    
    async def search_patents(self, company: str, category: Optional[str] = None, 
                            days_back: int = 365) -> dict:
        """
        Search for patents filed by a company.
        Uses web search to find patent information.
        """
        cache_key = f"patents_{company}_{category}_{days_back}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        assignees = COMPANY_ASSIGNEES.get(company, [company])
        
        # Build search query
        search_terms = []
        for assignee in assignees[:2]:  # Limit to avoid too long queries
            if category and category in PATENT_CATEGORIES:
                keywords = PATENT_CATEGORIES[category][:3]
                search_terms.append(f'"{assignee}" patent {" OR ".join(keywords)}')
            else:
                search_terms.append(f'"{assignee}" patent USPTO')
        
        all_patents = []
        
        for query in search_terms:
            try:
                results = await search_web(query, count=10)
                
                for result in results.get('results', []):
                    patent_info = self._parse_patent_result(result, company)
                    if patent_info:
                        all_patents.append(patent_info)
            except Exception as e:
                print(f"Patent search error for {company}: {e}")
        
        # Deduplicate by title
        seen_titles = set()
        unique_patents = []
        for patent in all_patents:
            title_key = patent.get('title', '')[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_patents.append(patent)
        
        # Categorize patents
        categorized = self._categorize_patents(unique_patents)
        
        result = {
            "company": company,
            "total_patents": len(unique_patents),
            "patents": unique_patents[:20],  # Limit results
            "by_category": categorized,
            "search_date": datetime.now().isoformat(),
            "period_days": days_back,
        }
        
        self._set_cache(cache_key, result)
        return result
    
    def _parse_patent_result(self, result: dict, company: str) -> Optional[dict]:
        """Parse a search result into patent info."""
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        url = result.get('url', '')
        
        # Check if it's likely a patent
        is_patent = any(term in title.lower() or term in url.lower() 
                       for term in ['patent', 'uspto', 'patentscope', 'espacenet'])
        
        if not is_patent:
            return None
        
        # Extract patent number if present
        patent_num = None
        num_match = re.search(r'US\s*(\d{7,})', title + ' ' + snippet)
        if num_match:
            patent_num = f"US{num_match.group(1)}"
        
        # Determine category
        category = "general"
        for cat, keywords in PATENT_CATEGORIES.items():
            if any(kw.lower() in (title + snippet).lower() for kw in keywords):
                category = cat
                break
        
        return {
            "title": title,
            "snippet": snippet,
            "url": url,
            "patent_number": patent_num,
            "category": category,
            "company": company,
            "source": "web_search",
        }
    
    def _categorize_patents(self, patents: list) -> dict:
        """Categorize patents by type."""
        by_category = defaultdict(list)
        for patent in patents:
            cat = patent.get('category', 'general')
            by_category[cat].append(patent)
        
        return {
            cat: {
                "count": len(patents),
                "patents": patents[:5]  # Limit per category
            }
            for cat, patents in by_category.items()
        }
    
    async def compare_patent_activity(self) -> dict:
        """Compare patent activity across all companies."""
        comparison = {}
        
        for ticker, config in COMPANIES.items():
            company_name = config['name'].split()[0]
            try:
                patents = await self.search_patents(company_name)
                comparison[company_name] = {
                    "total": patents.get('total_patents', 0),
                    "by_category": {
                        cat: data.get('count', 0) 
                        for cat, data in patents.get('by_category', {}).items()
                    },
                }
            except Exception as e:
                comparison[company_name] = {"total": 0, "error": str(e)}
        
        # Determine leaders
        sorted_companies = sorted(
            comparison.items(),
            key=lambda x: x[1].get('total', 0),
            reverse=True
        )
        
        return {
            "companies": comparison,
            "leader": sorted_companies[0][0] if sorted_companies else None,
            "total_patents_tracked": sum(c.get('total', 0) for c in comparison.values()),
            "analysis_date": datetime.now().isoformat(),
        }
    
    def get_innovation_score(self, patents: dict) -> dict:
        """Calculate innovation score based on patent categories."""
        by_category = patents.get('by_category', {})
        
        # Weight different categories
        weights = {
            "ai_ml": 2.0,
            "automation": 1.5,
            "power_systems": 1.3,
            "connectivity": 1.3,
            "sensors": 1.2,
            "manufacturing": 1.0,
            "thermal": 1.0,
            "materials": 1.0,
            "general": 0.5,
        }
        
        weighted_score = 0
        total_count = 0
        
        for cat, data in by_category.items():
            count = data.get('count', 0)
            weight = weights.get(cat, 1.0)
            weighted_score += count * weight
            total_count += count
        
        # Normalize to 0-100
        max_expected = 50  # Assume 50 patents is max
        normalized = min(100, (weighted_score / max_expected) * 100)
        
        # Calculate focus areas
        focus_areas = []
        if by_category.get('ai_ml', {}).get('count', 0) > 0:
            focus_areas.append("AI/ML")
        if by_category.get('automation', {}).get('count', 0) > 0:
            focus_areas.append("Automation")
        if by_category.get('power_systems', {}).get('count', 0) > 0:
            focus_areas.append("Power Systems")
        
        return {
            "innovation_score": round(normalized, 1),
            "total_patents": total_count,
            "weighted_score": round(weighted_score, 1),
            "focus_areas": focus_areas[:3],
            "ai_focus": by_category.get('ai_ml', {}).get('count', 0) > 0,
        }


# Singleton instance
_patent_scraper = PatentScraper()


async def search_company_patents(company: str, category: Optional[str] = None) -> dict:
    """Search patents for a company."""
    return await _patent_scraper.search_patents(company, category)


async def compare_all_patents() -> dict:
    """Compare patent activity across all companies."""
    return await _patent_scraper.compare_patent_activity()


def get_innovation_score(patents: dict) -> dict:
    """Get innovation score for patent data."""
    return _patent_scraper.get_innovation_score(patents)


def get_patent_categories() -> dict:
    """Get available patent categories."""
    return {
        cat: {"keywords": keywords[:5], "description": f"{cat.replace('_', ' ').title()} patents"}
        for cat, keywords in PATENT_CATEGORIES.items()
    }
