"""
Workday ATS adapter — fetches job postings directly from a company's Workday
tenant instead of going through generic web search.

Output shape matches JobScraper.search_jobs() so this can be swapped in behind
search_company_jobs() without route changes.
"""
import asyncio
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import httpx

from backend.ingestion.job_scraper import (
    JOB_CATEGORIES,
    LOCATION_REGIONS,
)


# US state abbreviations — matched case-sensitive with word boundaries
# against locationsText so "MI" won't false-match "Miami" or similar.
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR",  # PR = Puerto Rico (US territory)
}
_US_STATE_RE = re.compile(r"\b(" + "|".join(sorted(US_STATES)) + r")\b")

# ISO-3166 alpha-2 country codes → region. Used when the location string ends
# in a country code (e.g. SuccessFactors emits "City, State, IN" / "City, CN").
# Only consulted when the string has 3+ comma-separated parts so US "City, ST"
# is never misread (e.g. "Florence, KY" must not become Cayman Islands).
COUNTRY_CODE_TO_REGION = {
    "US": "americas", "MX": "americas", "CA": "americas", "BR": "americas",
    "CN": "asia_pacific", "MY": "asia_pacific", "SG": "asia_pacific",
    "IN": "asia_pacific", "VN": "asia_pacific", "TH": "asia_pacific",
    "TW": "asia_pacific", "JP": "asia_pacific", "KR": "asia_pacific",
    "PH": "asia_pacific", "ID": "asia_pacific",
    "DE": "europe", "UK": "europe", "GB": "europe", "PL": "europe",
    "HU": "europe", "CZ": "europe", "IE": "europe", "NL": "europe",
    "CH": "europe", "RO": "europe", "UA": "europe", "HR": "europe",
    "SE": "europe", "IT": "europe", "ES": "europe", "FR": "europe",
}

# City → region for postings that arrive without a country (common on Jabil's
# Workday tenant). Lowercase keys; longest match wins.
CITY_TO_REGION = {
    # Americas (Mexico / Brazil / Caribbean / Central America)
    "guadalajara": "americas", "chihuahua": "americas", "monterrey": "americas",
    "baja": "americas", "tijuana": "americas", "reynosa": "americas",
    "manaus": "americas", "torres": "americas", "sorocaba": "americas",
    "betim": "americas",
    "santo domingo": "americas", "san cristobal": "americas",
    # Asia-Pacific
    "penang": "asia_pacific", "ranjangaon": "asia_pacific",
    "sanchong": "asia_pacific", "bandung": "asia_pacific",
    "shenzhen": "asia_pacific", "shanghai": "asia_pacific", "beijing": "asia_pacific",
    "tianjin": "asia_pacific", "suzhou": "asia_pacific", "wuxi": "asia_pacific",
    "guangzhou": "asia_pacific",
    "ho chi minh": "asia_pacific", "hanoi": "asia_pacific", "dong nai": "asia_pacific",
    "sungai petani": "asia_pacific",
    # Europe
    "uzhhorod": "europe", "kwidzyn": "europe", "osijek": "europe",
    "tczew": "europe", "bydgoszcz": "europe",
    "dublin": "europe", "bray": "europe",  # Ireland
    "bettlach": "europe", "grenchen": "europe",  # Switzerland
    "tiszaujvaros": "europe",  # Hungary
}


def detect_region(location: str) -> str:
    """
    Classify a location string into americas / asia_pacific / europe / unknown.
    Shared across Workday + SuccessFactors adapters; ordered by precision.
    """
    if not location:
        return "unknown"
    loc_lc = location.lower()
    parts = [p.strip() for p in location.split(",") if p.strip()]

    # 1. Trailing ISO country code (only when 3+ parts so "Florence, KY" isn't
    #    misread as Cayman Islands).
    if len(parts) >= 3:
        tail = parts[-1].upper()
        if len(tail) == 2 and tail in COUNTRY_CODE_TO_REGION:
            return COUNTRY_CODE_TO_REGION[tail]

    # 2. Country name appears anywhere in the string.
    for reg, locations in LOCATION_REGIONS.items():
        if any(loc.lower() in loc_lc for loc in locations):
            return reg

    # 3. US state abbreviation (case-sensitive, word boundary).
    if _US_STATE_RE.search(location):
        return "americas"

    # 4. Known international city.
    for city in sorted(CITY_TO_REGION, key=len, reverse=True):
        if city in loc_lc:
            return CITY_TO_REGION[city]

    return "unknown"


# Per-company Workday tenant config.
# host = the wd*.myworkdayjobs.com subdomain
# tenant = the URL path segment after /wday/cxs/
# site = the careers site name (e.g. "Careers", "External")
WORKDAY_TENANTS: dict[str, dict[str, str]] = {
    "Flex": {
        "host": "flextronics.wd1.myworkdayjobs.com",
        "tenant": "flextronics",
        "site": "Careers",
    },
    "Jabil": {
        "host": "jabil.wd5.myworkdayjobs.com",
        "tenant": "jabil",
        "site": "Jabil_Careers",
    },
    # Celestica runs on SAP SuccessFactors (careers.celestica.com), not Workday —
    # it needs a separate adapter and is intentionally excluded here.
}

PAGE_SIZE = 20          # Workday caps list responses around 20
MAX_JOBS = 2000         # safety cap per company per fetch (Flex ~1.3k, Jabil ~1.9k)
REQUEST_TIMEOUT = 20.0
MAX_CONCURRENT_PAGES = 8  # parallel page fetches per tenant


class WorkdayJobScraper:
    """Fetches and analyzes job postings from Workday-hosted career sites."""

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._cache_time: dict[str, float] = {}
        self._cache_ttl = 1800  # 30 min

    def _get_cached(self, key: str) -> Optional[dict]:
        if key in self._cache:
            if datetime.now().timestamp() - self._cache_time.get(key, 0) < self._cache_ttl:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, data: dict):
        self._cache[key] = data
        self._cache_time[key] = datetime.now().timestamp()

    def supports(self, company: str) -> bool:
        return company in WORKDAY_TENANTS

    async def search_jobs(self, company: str, category: Optional[str] = None) -> dict:
        """Fetch jobs from the company's Workday tenant. Same output shape as JobScraper.search_jobs."""
        if not self.supports(company):
            raise ValueError(f"No Workday tenant configured for '{company}'")

        cache_key = f"workday_{company}_{category}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        cfg = WORKDAY_TENANTS[company]
        url = f"https://{cfg['host']}/wday/cxs/{cfg['tenant']}/{cfg['site']}/jobs"

        # Workday accepts a free-text searchText; use category keywords if provided.
        search_text = ""
        if category and category in JOB_CATEGORIES:
            search_text = JOB_CATEGORIES[category][0]

        raw_postings = await self._fetch_all_pages(url, search_text)

        jobs = [self._parse_posting(p, company, cfg["host"]) for p in raw_postings]
        jobs = [j for j in jobs if j]

        # Optional category filter (post-fetch, since Workday searchText is fuzzy)
        if category:
            jobs = [j for j in jobs if j.get("category") == category]

        analysis = self._analyze_jobs(jobs)

        result = {
            "company": company,
            "total_jobs": len(jobs),
            "jobs": jobs[:25],
            "analysis": analysis,
            "search_date": datetime.now().isoformat(),
            "source": "workday",
        }
        self._set_cache(cache_key, result)
        return result

    async def _fetch_all_pages(self, url: str, search_text: str) -> list[dict]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        async def fetch(client: httpx.AsyncClient, offset: int) -> Optional[dict]:
            payload = {
                "appliedFacets": {},
                "limit": PAGE_SIZE,
                "offset": offset,
                "searchText": search_text,
            }
            try:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError) as e:
                print(f"Workday fetch error at offset={offset}: {e}")
                return None

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Fetch offset=0 first to learn the total job count.
            first = await fetch(client, 0)
            if not first:
                return []
            first_page = first.get("jobPostings", []) or []
            total = first.get("total") or 0

            # If we already have everything, or no total reported, return what we have.
            if total <= PAGE_SIZE or len(first_page) < PAGE_SIZE:
                return first_page

            target = min(total, MAX_JOBS)
            offsets = list(range(PAGE_SIZE, target, PAGE_SIZE))

            # Fetch remaining pages in parallel, capped by a semaphore.
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

            async def bounded(offset: int) -> Optional[dict]:
                async with semaphore:
                    return await fetch(client, offset)

            pages = await asyncio.gather(*[bounded(o) for o in offsets])

        postings = list(first_page)
        for p in pages:
            if p:
                postings.extend(p.get("jobPostings", []) or [])
        return postings

    def _parse_posting(self, p: dict, company: str, host: str) -> Optional[dict]:
        title = (p.get("title") or "").strip()
        if not title:
            return None
        location = (p.get("locationsText") or "").strip()
        ext = p.get("externalPath") or ""
        url = f"https://{host}{ext}" if ext else ""
        posted = p.get("postedOn") or ""
        job_id = (p.get("bulletFields") or [""])[0]

        title_lc = title.lower()
        loc_lc = location.lower()

        category = "general"
        for cat, keywords in JOB_CATEGORIES.items():
            if any(kw.lower() in title_lc for kw in keywords):
                category = cat
                break

        region = self._detect_region(location, loc_lc)

        seniority = "mid"
        if any(t in title_lc for t in ["senior", "sr.", "lead", "principal", "staff"]):
            seniority = "senior"
        elif any(t in title_lc for t in ["junior", "jr.", "entry", "associate", "intern"]):
            seniority = "junior"
        elif any(t in title_lc for t in ["director", " vp", "head ", "chief", "manager"]):
            seniority = "leadership"

        return {
            "title": title,
            "snippet": f"{location} · Posted {posted}".strip(" ·"),
            "url": url,
            "category": category,
            "region": region,
            "seniority": seniority,
            "company": company,
            "job_id": job_id,
            "location": location,
            "posted_on": posted,
        }

    def _detect_region(self, location: str, loc_lc: str) -> str:
        return detect_region(location)

    def _analyze_jobs(self, jobs: list) -> dict:
        by_category: dict[str, int] = defaultdict(int)
        by_region: dict[str, int] = defaultdict(int)
        by_seniority: dict[str, int] = defaultdict(int)

        for j in jobs:
            by_category[j.get("category", "general")] += 1
            by_region[j.get("region", "unknown")] += 1
            by_seniority[j.get("seniority", "mid")] += 1

        total = len(jobs) or 1
        ai_focus = (by_category.get("ai_ml", 0) + by_category.get("data_center", 0)) / total
        tech_focus = (by_category.get("software", 0) + by_category.get("hardware", 0)) / total

        sorted_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
        top_categories = [c for c, _ in sorted_categories[:3]]

        return {
            "by_category": dict(by_category),
            "by_region": dict(by_region),
            "by_seniority": dict(by_seniority),
            "ai_hiring_focus": round(ai_focus * 100, 1),
            "tech_hiring_focus": round(tech_focus * 100, 1),
            "top_categories": top_categories,
            "is_hiring_ai": ai_focus > 0.1,
            "is_expanding": len(jobs) > 10,
        }


_workday_scraper = WorkdayJobScraper()


async def search_company_jobs_workday(company: str, category: Optional[str] = None) -> dict:
    return await _workday_scraper.search_jobs(company, category)


# ---------------------------------------------------------------------------
# Demo: python -m backend.ingestion.workday_scraper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo():
        result = await search_company_jobs_workday("Flex")
        print(f"Company: {result['company']} | Source: {result['source']}")
        print(f"Total jobs fetched: {result['total_jobs']}")
        print(f"Top categories: {result['analysis']['top_categories']}")
        print(f"By region: {result['analysis']['by_region']}")
        print(f"AI hiring focus: {result['analysis']['ai_hiring_focus']}%")
        print("\nFirst 5 postings:")
        for j in result["jobs"][:5]:
            print(f"  - [{j['category']:12s}] {j['title']}  ({j['location']})")

    asyncio.run(_demo())
