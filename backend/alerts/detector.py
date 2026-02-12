"""
Alert detector for competitive intelligence.
Monitors for significant events and triggers alerts.
"""
import re
from datetime import datetime
from typing import Optional
from collections import defaultdict

from backend.core.config import COMPANIES, ANOMALY_THRESHOLD, SENTIMENT_SHIFT_THRESHOLD
from backend.rag.retriever import search_documents, get_company_documents
from backend.analytics.sentiment import analyze_company_sentiment
from backend.analytics.anomaly import detect_capex_anomalies, detect_sentiment_shifts, detect_ai_investment_changes
from backend.alerts.alert_manager import create_alert


class AlertDetector:
    """
    Monitors for various alert-worthy events across companies.
    """
    
    # Alert type configurations
    ALERT_TYPES = {
        "new_filing": {
            "severity": "low",
            "description": "New SEC filing detected",
        },
        "capex_spike": {
            "severity": "high",
            "description": "Significant CapEx increase detected",
        },
        "capex_drop": {
            "severity": "medium",
            "description": "Significant CapEx decrease detected",
        },
        "sentiment_positive": {
            "severity": "low",
            "description": "Positive sentiment shift detected",
        },
        "sentiment_negative": {
            "severity": "high",
            "description": "Negative sentiment shift detected",
        },
        "ai_investment_surge": {
            "severity": "medium",
            "description": "Increased AI/Data Center investment focus",
        },
        "competitor_announcement": {
            "severity": "medium",
            "description": "Major competitor announcement",
        },
        "earnings_upcoming": {
            "severity": "low",
            "description": "Earnings report upcoming",
        },
        "strategic_change": {
            "severity": "high",
            "description": "Strategic direction change detected",
        },
    }
    
    # Keywords for strategic change detection
    STRATEGIC_KEYWORDS = [
        "acquisition", "merger", "restructuring", "divestiture", 
        "strategic review", "transformation", "pivot", "exit",
        "new ceo", "leadership change", "layoff", "workforce reduction",
    ]
    
    # Keywords for major announcements
    ANNOUNCEMENT_KEYWORDS = [
        "announces", "reports", "launches", "expands", "opens",
        "partners with", "signs agreement", "wins contract", "secures",
    ]
    
    def __init__(self):
        self._last_check = {}
        self._detected_alerts = []
    
    def detect_new_filings(self, company: str) -> list[dict]:
        """
        Detect new SEC filings for a company.
        """
        alerts = []
        company_title = company.title() if company.upper() not in COMPANIES else company
        
        # Get recent documents
        docs = get_company_documents(company_title, limit=10)
        
        for doc in docs:
            # Check if this is a new filing (would need timestamp comparison in production)
            filing_type = doc.get("metadata", {}).get("filing_type", "Unknown")
            fiscal_year = doc.get("metadata", {}).get("fiscal_year", "Unknown")
            
            # For 8-K filings, always create an alert
            if "8-K" in filing_type:
                alerts.append({
                    "type": "new_filing",
                    "company": company_title,
                    "title": f"New {filing_type} Filing",
                    "message": f"{company_title} filed a new {filing_type} for {fiscal_year}",
                    "severity": "medium" if "8-K" in filing_type else "low",
                    "metadata": {
                        "filing_type": filing_type,
                        "fiscal_year": fiscal_year,
                    },
                })
        
        return alerts[:3]  # Limit to 3 filing alerts
    
    def detect_capex_alerts(self, company: str) -> list[dict]:
        """
        Detect CapEx anomalies for a company.
        """
        alerts = []
        company_title = company.title() if company.upper() not in COMPANIES else company
        
        anomalies = detect_capex_anomalies(company_title)
        
        if anomalies.get("has_anomalies"):
            for anomaly in anomalies.get("anomalies", []):
                alert_type = "capex_spike" if anomaly["direction"] == "spike" else "capex_drop"
                
                alerts.append({
                    "type": alert_type,
                    "company": company_title,
                    "title": f"CapEx {anomaly['direction'].title()} Detected",
                    "message": f"{company_title} shows {anomaly['pct_change_from_mean']:.1f}% change from mean in {anomaly['period']}",
                    "severity": anomaly.get("severity", "medium"),
                    "metadata": {
                        "period": anomaly["period"],
                        "value": anomaly.get("value"),
                        "pct_change": anomaly["pct_change_from_mean"],
                    },
                })
        
        return alerts
    
    def detect_sentiment_alerts(self, company: str) -> list[dict]:
        """
        Detect sentiment shifts for a company.
        """
        alerts = []
        company_title = company.title() if company.upper() not in COMPANIES else company
        
        sentiment_data = detect_sentiment_shifts(company_title)
        
        if sentiment_data.get("has_significant_shift"):
            shift = sentiment_data.get("shift", 0)
            alert_type = "sentiment_positive" if shift > 0 else "sentiment_negative"
            
            alerts.append({
                "type": alert_type,
                "company": company_title,
                "title": f"Sentiment {'Improvement' if shift > 0 else 'Decline'} Detected",
                "message": f"{company_title} sentiment shifted by {abs(shift):.1%}",
                "severity": "high" if abs(shift) > SENTIMENT_SHIFT_THRESHOLD else "medium",
                "metadata": {
                    "shift": shift,
                    "current_score": sentiment_data.get("current_score"),
                    "previous_score": sentiment_data.get("previous_score"),
                },
            })
        
        return alerts
    
    def detect_ai_investment_alerts(self, company: str) -> list[dict]:
        """
        Detect changes in AI investment focus.
        """
        alerts = []
        company_title = company.title() if company.upper() not in COMPANIES else company
        
        ai_changes = detect_ai_investment_changes(company_title)
        
        if ai_changes.get("trend") == "increasing":
            alerts.append({
                "type": "ai_investment_surge",
                "company": company_title,
                "title": "AI Investment Focus Increasing",
                "message": f"{company_title} is showing increased focus on AI/Data Center investments",
                "severity": "medium",
                "metadata": {
                    "trend": ai_changes.get("trend"),
                    "current_focus": ai_changes.get("latest_ai_focus", 0),
                },
            })
        
        return alerts
    
    def detect_strategic_changes(self, company: str) -> list[dict]:
        """
        Detect strategic changes from document content.
        """
        alerts = []
        company_title = company.title() if company.upper() not in COMPANIES else company
        
        # Search for strategic keywords in recent documents
        for keyword in self.STRATEGIC_KEYWORDS:
            docs = search_documents(
                query=f"{company_title} {keyword}",
                company_filter=company_title,
                n_results=3
            )
            
            for doc in docs:
                content_lower = doc["content"].lower()
                if keyword in content_lower:
                    # Extract a snippet around the keyword
                    idx = content_lower.find(keyword)
                    start = max(0, idx - 50)
                    end = min(len(content_lower), idx + len(keyword) + 100)
                    snippet = doc["content"][start:end].strip()
                    
                    alerts.append({
                        "type": "strategic_change",
                        "company": company_title,
                        "title": f"Strategic Change: {keyword.title()}",
                        "message": f"...{snippet}...",
                        "severity": "high",
                        "metadata": {
                            "keyword": keyword,
                            "source": doc.get("source", "Unknown"),
                        },
                    })
                    break  # One alert per keyword
        
        return alerts[:3]  # Limit strategic alerts
    
    def detect_all_alerts(self, company: Optional[str] = None) -> list[dict]:
        """
        Run all alert detections for one or all companies.
        """
        all_alerts = []
        
        companies_to_check = [company] if company else list(COMPANIES.keys())
        
        for ticker in companies_to_check:
            company_name = COMPANIES.get(ticker, {}).get("name", ticker).split()[0]
            
            # Run all detectors
            all_alerts.extend(self.detect_capex_alerts(company_name))
            all_alerts.extend(self.detect_sentiment_alerts(company_name))
            all_alerts.extend(self.detect_ai_investment_alerts(company_name))
            # Strategic changes are expensive, run less frequently
            # all_alerts.extend(self.detect_strategic_changes(company_name))
        
        # Deduplicate and sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        all_alerts.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return all_alerts
    
    def create_alerts_from_detections(self, detections: list[dict]) -> list[dict]:
        """
        Create persistent alerts from detected events.
        """
        created_alerts = []
        
        for detection in detections:
            alert = create_alert(
                alert_type=detection["type"],
                company=detection["company"],
                title=detection["title"],
                message=detection["message"],
                severity=detection["severity"],
                metadata=detection.get("metadata"),
            )
            created_alerts.append(alert)
        
        return created_alerts


# Singleton detector instance
_detector = AlertDetector()


def run_alert_detection(company: Optional[str] = None) -> dict:
    """
    Run alert detection and create alerts.
    """
    detections = _detector.detect_all_alerts(company)
    created = _detector.create_alerts_from_detections(detections)
    
    return {
        "detected": len(detections),
        "created": len(created),
        "alerts": created,
    }


def get_alert_detector() -> AlertDetector:
    """Get the singleton detector instance."""
    return _detector
