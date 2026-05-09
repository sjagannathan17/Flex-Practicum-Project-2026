"""
SAP SuccessFactors career-site adapter.

Unlike Workday, SuccessFactors typically doesn't expose a public JSON API per
tenant — the careers site is server-rendered HTML. We paginate the public
search page (?startrow=0, 25, 50, ...) and parse rows out of the markup.

Output shape matches JobScraper.search_jobs() so this slots in behind
search_company_jobs() like the Workday adapter.
"""
import asyncio
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from backend.ingestion.job_scraper import JOB_CATEGORIES
from backend.ingestion.workday_scraper import detect_region


# Per-company SuccessFactors tenant config.
SF_TENANTS: dict[str, dict[str, str]] = {
    "Celestica": {
        "host": "careers.celestica.com",
        "company_param": "Celestica",
        "locale": "en_US",
    },
    # Other SF-hosted companies can be added here as their host/company param
    # is identified.
}

PAGE_SIZE = 25
MAX_JOBS = 1500
REQUEST_TIMEOUT = 20.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# "Results 1 – 25 of <b>1051</b>" → captures total count.
_TOTAL_RE = re.compile(r"of\s*<b>\s*(\d+)\s*</b>")


class SuccessFactorsJobScraper:
    """Fetches and analyzes job postings from SuccessFactors career sites."""

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
        return company in SF_TENANTS

    async def search_jobs(self, company: str, category: Optional[str] = None) -> dict:
        if not self.supports(company):
            raise ValueError(f"No SuccessFactors tenant configured for '{company}'")

        cache_key = f"sf_{company}_{category}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        cfg = SF_TENANTS[company]
        base = f"https://{cfg['host']}"
        keyword = JOB_CATEGORIES[category][0] if category and category in JOB_CATEGORIES else ""

        jobs = await self._fetch_all_pages(base, cfg["company_param"], cfg["locale"], keyword, company)

        # Optional category filter (post-fetch, since SF keyword search is fuzzy).
        if category:
            jobs = [j for j in jobs if j.get("category") == category]

        analysis = self._analyze_jobs(jobs)

        result = {
            "company": company,
            "total_jobs": len(jobs),
            "jobs": jobs[:25],
            "analysis": analysis,
            "search_date": datetime.now().isoformat(),
            "source": "successfactors",
        }
        self._set_cache(cache_key, result)
        return result

    async def _fetch_all_pages(
        self, base: str, company_param: str, locale: str, keyword: str, company_name: str
    ) -> list[dict]:
        jobs: list[dict] = []
        seen_hrefs: set[str] = set()
        offset = 0
        total: Optional[int] = None

        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            follow_redirects=True,
        ) as client:
            while offset < MAX_JOBS:
                params = {
                    "company": company_param,
                    "locale": locale,
                    "startrow": offset,
                }
                if keyword:
                    params["q"] = keyword
                try:
                    r = await client.get(f"{base}/search/", params=params)
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"SuccessFactors fetch error at offset={offset}: {e}")
                    break

                if total is None:
                    m = _TOTAL_RE.search(r.text)
                    if m:
                        total = int(m.group(1))

                page_jobs = self._parse_page(r.text, base, company_name)
                if not page_jobs:
                    break

                new_in_page = 0
                for j in page_jobs:
                    if j["url"] in seen_hrefs:
                        continue
                    seen_hrefs.add(j["url"])
                    jobs.append(j)
                    new_in_page += 1

                if new_in_page == 0:
                    break

                offset += PAGE_SIZE
                if total is not None and offset >= total:
                    break

        return jobs

    def _parse_page(self, html: str, base: str, company: str) -> list[dict]:
        soup = BeautifulSoup(html, "lxml")
        out: list[dict] = []
        seen: set[str] = set()
        for a in soup.select("a.jobTitle-link"):
            href = a.get("href", "")
            if not href or href in seen:
                continue
            seen.add(href)
            tr = a.find_parent("tr")
            loc_el = tr.select_one(".jobLocation") if tr else None
            location = loc_el.get_text(strip=True) if loc_el else ""
            title = a.get_text(strip=True)
            url = href if href.startswith("http") else f"{base}{href}"
            job_id_match = re.search(r"/(\d{6,})/?$", href)
            job_id = job_id_match.group(1) if job_id_match else ""
            out.append(self._build_job(title, location, url, job_id, company))
        return out

    def _build_job(self, title: str, location: str, url: str, job_id: str, company: str) -> dict:
        title_lc = title.lower()

        category = "general"
        for cat, keywords in JOB_CATEGORIES.items():
            if any(kw.lower() in title_lc for kw in keywords):
                category = cat
                break

        seniority = "mid"
        if any(t in title_lc for t in ["senior", "sr.", "lead", "principal", "staff"]):
            seniority = "senior"
        elif any(t in title_lc for t in ["junior", "jr.", "entry", "associate", "intern"]):
            seniority = "junior"
        elif any(t in title_lc for t in ["director", " vp", "head ", "chief", "manager"]):
            seniority = "leadership"

        return {
            "title": title,
            "snippet": location,
            "url": url,
            "category": category,
            "region": detect_region(location),
            "seniority": seniority,
            "company": company,
            "job_id": job_id,
            "location": location,
        }

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


_sf_scraper = SuccessFactorsJobScraper()


async def search_company_jobs_sf(company: str, category: Optional[str] = None) -> dict:
    return await _sf_scraper.search_jobs(company, category)


# ---------------------------------------------------------------------------
# Demo: python -m backend.ingestion.successfactors_scraper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo():
        result = await search_company_jobs_sf("Celestica")
        a = result["analysis"]
        print(f"Company: {result['company']} | Source: {result['source']}")
        print(f"Total jobs fetched: {result['total_jobs']}")
        print(f"Top categories: {a['top_categories']}")
        print(f"By region: {a['by_region']}")
        print(f"AI hiring focus: {a['ai_hiring_focus']}%")
        print("\nFirst 5 postings:")
        for j in result["jobs"][:5]:
            print(f"  - [{j['category']:12s}] [{j['region']:12s}] {j['title']}  ({j['location']})")

    asyncio.run(_demo())
