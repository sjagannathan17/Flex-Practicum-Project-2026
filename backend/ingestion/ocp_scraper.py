"""
Open Compute Project (OCP) integration for competitive intelligence.
Tracks company involvement in open-source data center hardware initiatives.
"""
import asyncio
from datetime import datetime
from typing import Optional
from collections import defaultdict
import httpx

from backend.core.config import COMPANIES
from backend.rag.web_search import search_web


# OCP Project categories relevant to EMS companies
OCP_CATEGORIES = {
    "server": {
        "name": "Server",
        "keywords": ["server", "compute", "motherboard", "chassis"],
        "relevance": "high",
    },
    "rack": {
        "name": "Rack & Power",
        "keywords": ["rack", "power", "ups", "pdu", "cooling"],
        "relevance": "high",
    },
    "storage": {
        "name": "Storage",
        "keywords": ["storage", "ssd", "nvme", "hdd", "jbod"],
        "relevance": "medium",
    },
    "networking": {
        "name": "Networking",
        "keywords": ["networking", "switch", "nic", "router"],
        "relevance": "medium",
    },
    "hardware_management": {
        "name": "Hardware Management",
        "keywords": ["bmc", "firmware", "management", "openbmc"],
        "relevance": "medium",
    },
    "sustainability": {
        "name": "Sustainability",
        "keywords": ["sustainability", "circular", "recycling", "carbon"],
        "relevance": "medium",
    },
}

# Known OCP member companies and their involvement
OCP_MEMBER_INFO = {
    "Flex": {
        "member_status": "Solution Provider",
        "focus_areas": ["Manufacturing", "Design Services", "System Integration"],
        "known_contributions": [
            "Data center hardware manufacturing",
            "OCP-compliant server production",
            "Custom rack solutions",
        ],
    },
    "Jabil": {
        "member_status": "Solution Provider", 
        "focus_areas": ["Manufacturing", "Supply Chain", "Design"],
        "known_contributions": [
            "Server chassis manufacturing",
            "Thermal solutions",
            "Custom enclosures",
        ],
    },
    "Celestica": {
        "member_status": "Solution Provider",
        "focus_areas": ["Manufacturing", "Hardware Design", "Testing"],
        "known_contributions": [
            "OCP server manufacturing",
            "Switch manufacturing",
            "Hardware validation",
        ],
    },
    "Benchmark": {
        "member_status": "Contributor",
        "focus_areas": ["Precision Manufacturing", "Aerospace/Defense"],
        "known_contributions": [
            "High-reliability components",
            "Specialty manufacturing",
        ],
    },
    "Sanmina": {
        "member_status": "Solution Provider",
        "focus_areas": ["Manufacturing", "PCB", "Systems Integration"],
        "known_contributions": [
            "PCB manufacturing for OCP designs",
            "System assembly",
            "Supply chain management",
        ],
    },
}


class OCPScraper:
    """Scrapes and aggregates OCP-related information for tracked companies."""
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour cache
    
    def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached data if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).seconds < self._cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: dict):
        """Cache data with timestamp."""
        self._cache[key] = (data, datetime.now())
    
    async def get_company_ocp_involvement(self, company: str) -> dict:
        """
        Get OCP involvement details for a company.
        
        Args:
            company: Company name
            
        Returns:
            Dictionary with OCP membership and contribution details
        """
        cache_key = f"ocp_involvement_{company.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Get static member info
        member_info = OCP_MEMBER_INFO.get(company, {
            "member_status": "Unknown",
            "focus_areas": [],
            "known_contributions": [],
        })
        
        # Search for recent OCP-related news
        ocp_news = await self._search_ocp_news(company)
        
        # Search for OCP contributions
        contributions = await self._search_ocp_contributions(company)
        
        # Calculate OCP engagement score
        engagement_score = self._calculate_engagement_score(
            member_info, ocp_news, contributions
        )
        
        result = {
            "company": company,
            "member_status": member_info.get("member_status", "Unknown"),
            "focus_areas": member_info.get("focus_areas", []),
            "known_contributions": member_info.get("known_contributions", []),
            "recent_news": ocp_news[:5],
            "recent_contributions": contributions[:5],
            "engagement_score": engagement_score,
            "ocp_categories": self._identify_category_involvement(company, ocp_news),
            "timestamp": datetime.now().isoformat(),
        }
        
        self._set_cache(cache_key, result)
        return result
    
    async def _search_ocp_news(self, company: str) -> list:
        """Search for OCP-related news for a company."""
        query = f"{company} Open Compute Project OCP data center"
        results = await search_web(query, count=10)
        
        ocp_news = []
        for result in results:
            content = (result.get("title", "") + " " + result.get("description", "")).lower()
            
            # Filter for OCP-relevant content
            if any(term in content for term in ["open compute", "ocp", "data center", "hyperscale"]):
                ocp_news.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("description", ""),
                    "relevance": self._calculate_relevance(content),
                })
        
        return sorted(ocp_news, key=lambda x: x["relevance"], reverse=True)
    
    async def _search_ocp_contributions(self, company: str) -> list:
        """Search for OCP contributions by a company."""
        query = f"site:opencompute.org {company}"
        results = await search_web(query, count=5)
        
        contributions = []
        for result in results:
            contributions.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "type": self._identify_contribution_type(result.get("title", "")),
            })
        
        return contributions
    
    def _calculate_relevance(self, content: str) -> float:
        """Calculate relevance score for OCP content."""
        score = 0.0
        
        high_value_terms = ["open compute", "ocp", "hyperscale", "data center hardware"]
        medium_value_terms = ["server", "rack", "infrastructure", "manufacturing"]
        
        for term in high_value_terms:
            if term in content:
                score += 0.3
        
        for term in medium_value_terms:
            if term in content:
                score += 0.1
        
        return min(score, 1.0)
    
    def _identify_contribution_type(self, title: str) -> str:
        """Identify the type of OCP contribution."""
        title_lower = title.lower()
        
        if "specification" in title_lower or "spec" in title_lower:
            return "Specification"
        elif "design" in title_lower:
            return "Design Contribution"
        elif "project" in title_lower:
            return "Project"
        elif "member" in title_lower or "partner" in title_lower:
            return "Membership"
        else:
            return "General"
    
    def _identify_category_involvement(self, company: str, news: list) -> list:
        """Identify which OCP categories a company is involved in."""
        involved_categories = []
        
        all_content = " ".join([
            n.get("title", "") + " " + n.get("description", "")
            for n in news
        ]).lower()
        
        for cat_id, cat_info in OCP_CATEGORIES.items():
            if any(kw in all_content for kw in cat_info["keywords"]):
                involved_categories.append({
                    "id": cat_id,
                    "name": cat_info["name"],
                    "relevance": cat_info["relevance"],
                })
        
        return involved_categories
    
    def _calculate_engagement_score(
        self, 
        member_info: dict, 
        news: list, 
        contributions: list
    ) -> dict:
        """Calculate overall OCP engagement score."""
        score = 0.0
        
        # Member status weight
        status_weights = {
            "Platinum": 1.0,
            "Gold": 0.8,
            "Solution Provider": 0.6,
            "Contributor": 0.4,
            "Community": 0.2,
            "Unknown": 0.1,
        }
        score += status_weights.get(member_info.get("member_status", "Unknown"), 0.1) * 30
        
        # News activity weight
        score += min(len(news) * 5, 30)
        
        # Contributions weight
        score += min(len(contributions) * 8, 40)
        
        # Normalize to 0-100
        score = min(score, 100)
        
        level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
        
        return {
            "score": round(score, 1),
            "level": level,
            "factors": {
                "member_status": member_info.get("member_status", "Unknown"),
                "news_activity": len(news),
                "contributions": len(contributions),
            },
        }
    
    async def compare_ocp_involvement(self) -> dict:
        """Compare OCP involvement across all tracked companies."""
        results = {}
        
        for ticker, config in COMPANIES.items():
            company = config["name"].split()[0]
            involvement = await self.get_company_ocp_involvement(company)
            results[company] = {
                "member_status": involvement["member_status"],
                "engagement_score": involvement["engagement_score"]["score"],
                "engagement_level": involvement["engagement_score"]["level"],
                "focus_areas": involvement["focus_areas"],
                "categories": [c["name"] for c in involvement.get("ocp_categories", [])],
            }
            await asyncio.sleep(0.5)  # Rate limiting
        
        # Rank by engagement score
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]["engagement_score"],
            reverse=True
        )
        
        return {
            "companies": results,
            "rankings": [{"company": c, "score": d["engagement_score"]} for c, d in ranked],
            "most_engaged": ranked[0][0] if ranked else None,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_ocp_categories(self) -> dict:
        """Get all OCP categories."""
        return OCP_CATEGORIES


# Global instance
_ocp_scraper = OCPScraper()


async def get_company_ocp_data(company: str) -> dict:
    """Get OCP involvement for a company."""
    return await _ocp_scraper.get_company_ocp_involvement(company)


async def compare_ocp_involvement() -> dict:
    """Compare OCP involvement across companies."""
    return await _ocp_scraper.compare_ocp_involvement()


def get_ocp_categories() -> dict:
    """Get OCP category definitions."""
    return _ocp_scraper.get_ocp_categories()


def get_ocp_member_info() -> dict:
    """Get known OCP member information."""
    return OCP_MEMBER_INFO
