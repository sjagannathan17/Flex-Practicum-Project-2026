"""
Automated SEC filing downloader using EDGAR API.
Downloads new 10-K, 10-Q, and 8-K filings for tracked companies.
Also fetches 8-K exhibit attachments (earnings call transcripts).
"""
import httpx
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from backend.core.config import COMPANIES, SEC_USER_AGENT, DATA_DIR


# SEC EDGAR API endpoints
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"


class SECDownloader:
    """Downloads SEC filings from EDGAR."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        }
        self.download_dir = DATA_DIR / "sec_filings"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_file = self.download_dir / "downloaded_filings.json"
        self.downloaded = self._load_tracking()
    
    def _load_tracking(self) -> dict:
        """Load tracking of already downloaded filings."""
        if self.tracking_file.exists():
            with open(self.tracking_file) as f:
                return json.load(f)
        return {}
    
    def _save_tracking(self):
        """Save tracking of downloaded filings."""
        with open(self.tracking_file, "w") as f:
            json.dump(self.downloaded, f, indent=2)
    
    def _format_cik(self, cik: str) -> str:
        """Format CIK to 10 digits with leading zeros."""
        return cik.lstrip("0").zfill(10)
    
    async def get_company_filings(
        self,
        ticker: str,
        filing_types: list[str] = ["10-K", "10-Q", "8-K"],
        days_back: int = 90,
    ) -> list[dict]:
        """
        Get recent filings for a company.
        
        Args:
            ticker: Company ticker
            filing_types: Types of filings to look for
            days_back: How many days back to check
            
        Returns:
            List of filing metadata
        """
        company = COMPANIES.get(ticker)
        if not company:
            print(f"Unknown ticker: {ticker}")
            return []
        
        cik = self._format_cik(company["cik"])
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            print(f"Error fetching filings for {ticker}: {e}")
            return []
        
        # Parse recent filings
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        if not recent:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        
        for i in range(len(forms)):
            form = forms[i]
            if form not in filing_types:
                continue
            
            filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
            if filing_date < cutoff_date:
                continue
            
            accession = accessions[i].replace("-", "")
            filing_id = f"{ticker}_{form}_{dates[i]}_{accession}"
            
            filings.append({
                "ticker": ticker,
                "company": company["name"],
                "cik": cik,
                "form": form,
                "filing_date": dates[i],
                "accession": accession,
                "accession_formatted": accessions[i],
                "primary_doc": primary_docs[i],
                "description": descriptions[i] if i < len(descriptions) else "",
                "filing_id": filing_id,
                "already_downloaded": filing_id in self.downloaded,
            })
        
        return filings
    
    async def download_filing(self, filing: dict) -> Optional[Path]:
        """
        Download a filing document.
        
        Args:
            filing: Filing metadata dict
            
        Returns:
            Path to downloaded file, or None if failed
        """
        if filing["filing_id"] in self.downloaded:
            print(f"Already downloaded: {filing['filing_id']}")
            return Path(self.downloaded[filing["filing_id"]])
        
        # Create company directory
        company_dir = self.download_dir / filing["ticker"]
        company_dir.mkdir(exist_ok=True)
        
        # Build download URL
        url = EDGAR_FILING_URL.format(
            cik=filing["cik"].lstrip("0"),
            accession=filing["accession"],
            filename=filing["primary_doc"],
        )
        
        # Determine output filename
        ext = Path(filing["primary_doc"]).suffix or ".htm"
        filename = f"{filing['form']}_{filing['filing_date']}{ext}"
        output_path = company_dir / filename
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, timeout=60.0)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    f.write(response.content)
            
            # Track download
            self.downloaded[filing["filing_id"]] = str(output_path)
            self._save_tracking()
            
            print(f"Downloaded: {filing['ticker']} {filing['form']} ({filing['filing_date']})")
            return output_path
            
        except Exception as e:
            print(f"Error downloading {filing['filing_id']}: {e}")
            return None
    
    async def check_and_download_new_filings(
        self,
        filing_types: list[str] = ["10-K", "10-Q", "8-K"],
        days_back: int = 30,
    ) -> list[dict]:
        """
        Check all tracked companies for new filings and download them.
        
        Returns:
            List of newly downloaded filings
        """
        new_filings = []
        
        for ticker in COMPANIES.keys():
            print(f"\nChecking {ticker} for new filings...")
            filings = await self.get_company_filings(ticker, filing_types, days_back)
            
            for filing in filings:
                if not filing["already_downloaded"]:
                    path = await self.download_filing(filing)
                    if path:
                        filing["local_path"] = str(path)
                        new_filings.append(filing)

                # For 8-K filings, always check for new transcript exhibits
                # (catches exhibits even when the primary document was already downloaded)
                if filing["form"] == "8-K":
                    exhibit_paths = await self.download_8k_exhibits(filing)
                    for ep in exhibit_paths:
                        new_filings.append({
                            **filing,
                            "form": "8-K-EX",
                            "local_path": str(ep),
                            "filing_id": f"{filing['filing_id']}_exhibit_{ep.name}",
                            "description": "8-K Exhibit (transcript)",
                        })

        return new_filings
    
    async def get_8k_exhibits(self, cik: str, accession: str) -> list[dict]:
        """
        Fetch exhibit list from an 8-K filing index page and return entries
        that are likely earnings call transcripts (by description keyword or
        file size > 50 KB).
        """
        raw_cik = cik.lstrip("0")
        index_url = (
            f"https://data.sec.gov/Archives/edgar/data/{raw_cik}/{accession}/"
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(index_url, headers=self.headers, timeout=30.0)
                resp.raise_for_status()
        except Exception as e:
            print(f"[8-K exhibits] Error fetching index {index_url}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        transcript_keywords = ["transcript", "earnings call", "conference call"]
        exhibits = []

        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            description = cells[1].text.strip().lower()
            filename = cells[2].text.strip()

            if not filename or "." not in filename:
                continue

            ext = Path(filename).suffix.lower()
            if ext not in (".htm", ".html", ".txt"):
                continue

            is_transcript_desc = any(kw in description for kw in transcript_keywords)

            # Try to parse size from the last table cell (bytes on EDGAR)
            size_kb = 0.0
            if len(cells) >= 5:
                try:
                    size_kb = int("".join(c for c in cells[4].text if c.isdigit())) / 1024
                except Exception:
                    pass

            if is_transcript_desc or size_kb > 50:
                exhibits.append(
                    {
                        "url": f"https://data.sec.gov/Archives/edgar/data/{raw_cik}/{accession}/{filename}",
                        "description": description,
                        "filename": filename,
                    }
                )

        return exhibits

    async def download_8k_exhibits(self, filing: dict) -> list[Path]:
        """
        Download transcript-likely exhibits attached to an 8-K filing.
        Returns list of paths to successfully downloaded files.
        """
        cik = filing["cik"]
        accession = filing["accession"]
        ticker = filing["ticker"]
        filing_date = filing["filing_date"]

        exhibits = await self.get_8k_exhibits(cik, accession)
        if not exhibits:
            return []

        company_dir = self.download_dir / ticker
        company_dir.mkdir(exist_ok=True)

        downloaded: list[Path] = []
        async with httpx.AsyncClient() as client:
            for exhibit in exhibits:
                exhibit_id = (
                    f"{ticker}_8K_exhibit_{accession}_{exhibit['filename']}"
                )

                if exhibit_id in self.downloaded:
                    existing = Path(self.downloaded[exhibit_id])
                    if existing.exists():
                        downloaded.append(existing)
                    continue

                safe_name = exhibit["filename"].replace("/", "_")
                output_path = company_dir / f"8K_exhibit_{filing_date}_{safe_name}"

                try:
                    resp = await client.get(
                        exhibit["url"], headers=self.headers, timeout=60.0
                    )
                    resp.raise_for_status()

                    output_path.write_bytes(resp.content)
                    self.downloaded[exhibit_id] = str(output_path)
                    self._save_tracking()
                    downloaded.append(output_path)
                    print(
                        f"[8-K exhibit] Downloaded: {ticker} {filing_date} "
                        f"{exhibit['filename']} ({len(resp.content)//1024} KB)"
                    )

                except Exception as e:
                    print(
                        f"[8-K exhibit] Error downloading {exhibit['filename']}: {e}"
                    )

        return downloaded

    def get_download_stats(self) -> dict:
        """Get statistics about downloaded filings."""
        stats = {
            "total_downloaded": len(self.downloaded),
            "by_company": {},
            "by_form": {},
        }
        
        for filing_id in self.downloaded:
            parts = filing_id.split("_")
            if len(parts) >= 2:
                ticker, form = parts[0], parts[1]
                stats["by_company"][ticker] = stats["by_company"].get(ticker, 0) + 1
                stats["by_form"][form] = stats["by_form"].get(form, 0) + 1
        
        return stats


# Sync wrapper functions for use in scheduler
def check_new_filings_sync(
    filing_types: list[str] = ["10-K", "10-Q", "8-K"],
    days_back: int = 30,
) -> list[dict]:
    """Synchronous wrapper for checking and downloading new filings."""
    import asyncio
    
    downloader = SECDownloader()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            downloader.check_and_download_new_filings(filing_types, days_back)
        )
    finally:
        loop.close()


async def download_all_recent_filings():
    """Download all recent filings for all companies."""
    downloader = SECDownloader()
    return await downloader.check_and_download_new_filings(days_back=365)
