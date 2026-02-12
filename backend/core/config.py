"""
Configuration and settings for the CapEx Intelligence Platform.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent  # SCU Flex Practicum 2026/
BACKEND_DIR = BASE_DIR / "backend"
CHROMADB_PATH = str(BASE_DIR / "chromadb_store")
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ---------------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIM = 768
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# RAG SETTINGS
# ---------------------------------------------------------------------------
CHUNK_SIZE = 400  # words per chunk
CHUNK_OVERLAP = 40  # overlap words
TOP_K_RESULTS = 10  # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # minimum similarity score

# ---------------------------------------------------------------------------
# COMPANIES
# ---------------------------------------------------------------------------
COMPANIES = {
    "FLEX": {
        "name": "Flex Ltd.",
        "cik": "0000866374",
        "ticker": "FLEX",
        "fiscal_year_end": "March",
        "industry": "Electronics Manufacturing Services",
        "headquarters": "Singapore",
        "color": "#00A0E3",  # Flex blue
    },
    "JBL": {
        "name": "Jabil Inc.",
        "cik": "0000898293",
        "ticker": "JBL",
        "fiscal_year_end": "August",
        "industry": "Electronics Manufacturing Services",
        "headquarters": "St. Petersburg, FL, USA",
        "color": "#1E4D2B",  # Jabil green
    },
    "CLS": {
        "name": "Celestica Inc.",
        "cik": "0001030894",
        "ticker": "CLS",
        "fiscal_year_end": "December",
        "industry": "Electronics Manufacturing Services",
        "headquarters": "Toronto, Canada",
        "color": "#003366",  # Celestica blue
    },
    "BHE": {
        "name": "Benchmark Electronics, Inc.",
        "cik": "0001080020",
        "ticker": "BHE",
        "fiscal_year_end": "December",
        "industry": "Electronics Manufacturing Services",
        "headquarters": "Tempe, AZ, USA",
        "color": "#B8860B",  # Benchmark gold
    },
    "SANM": {
        "name": "Sanmina Corporation",
        "cik": "0000897723",
        "ticker": "SANM",
        "fiscal_year_end": "September",
        "industry": "Electronics Manufacturing Services",
        "headquarters": "San Jose, CA, USA",
        "color": "#C41E3A",  # Sanmina red
    },
}

# Company name to ticker mapping
COMPANY_NAME_TO_TICKER = {
    "Flex": "FLEX",
    "Jabil": "JBL",
    "Celestica": "CLS",
    "Benchmark": "BHE",
    "Sanmina": "SANM",
}

# ---------------------------------------------------------------------------
# FILING TYPES
# ---------------------------------------------------------------------------
FILING_TYPES = [
    "10-K",
    "10-Q",
    "8-K",
    "Earnings Transcript",
    "Earnings Presentation",
    "Press Release",
    "Shareholder Letter",
    "Investor Presentation",
]

# ---------------------------------------------------------------------------
# SEC SETTINGS
# ---------------------------------------------------------------------------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "CapExIntel/1.0 (contact@example.com)")
SEC_BASE_URL = "https://www.sec.gov"
SEC_EDGAR_URL = "https://data.sec.gov"

# ---------------------------------------------------------------------------
# SCHEDULER SETTINGS
# ---------------------------------------------------------------------------
INGESTION_SCHEDULE = os.getenv("INGESTION_SCHEDULE", "0 16 * * 1-5")  # 4 PM ET weekdays

# ---------------------------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------------------------
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
WEB_SEARCH_RESULTS = 5  # number of web results to include

# ---------------------------------------------------------------------------
# ALERTS
# ---------------------------------------------------------------------------
ALERT_EMAIL = os.getenv("ALERT_FROM_EMAIL", "alerts@capexintel.com")
ANOMALY_THRESHOLD = 0.2  # 20% change triggers anomaly alert
SENTIMENT_SHIFT_THRESHOLD = 0.3  # sentiment change threshold
