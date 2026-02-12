"""
Alert management system for competitive intelligence.
Manages alerts for anomalies, sentiment shifts, and significant changes.
"""
import json
from typing import Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from backend.analytics.anomaly import get_all_anomalies
from backend.core.config import ANOMALY_THRESHOLD, SENTIMENT_SHIFT_THRESHOLD, BASE_DIR


# In-memory alert storage (would use a database in production)
_alerts: list[dict] = []
_alert_id_counter = 0

# Alerts file for persistence
ALERTS_FILE = BASE_DIR / "data" / "alerts.json"


def load_alerts():
    """Load alerts from file."""
    global _alerts, _alert_id_counter
    
    try:
        if ALERTS_FILE.exists():
            with open(ALERTS_FILE, 'r') as f:
                data = json.load(f)
                _alerts = data.get('alerts', [])
                _alert_id_counter = data.get('counter', 0)
    except Exception:
        _alerts = []
        _alert_id_counter = 0


def save_alerts():
    """Save alerts to file."""
    try:
        ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERTS_FILE, 'w') as f:
            json.dump({
                'alerts': _alerts,
                'counter': _alert_id_counter,
            }, f, indent=2)
    except Exception:
        pass


def create_alert(
    alert_type: str,
    severity: str,
    company: str,
    title: str,
    description: str,
    data: Optional[dict] = None,
) -> dict:
    """
    Create a new alert.
    
    Args:
        alert_type: Type of alert (capex_anomaly, sentiment_shift, ai_investment_change, etc.)
        severity: Severity level (low, medium, high, critical)
        company: Company name
        title: Alert title
        description: Alert description
        data: Additional data/context
        
    Returns:
        Created alert object
    """
    global _alert_id_counter
    
    _alert_id_counter += 1
    
    alert = {
        "id": _alert_id_counter,
        "type": alert_type,
        "severity": severity,
        "company": company,
        "title": title,
        "description": description,
        "data": data or {},
        "created_at": datetime.utcnow().isoformat(),
        "read": False,
        "dismissed": False,
    }
    
    _alerts.append(alert)
    save_alerts()
    
    return alert


def get_alerts(
    company: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 50,
) -> list[dict]:
    """
    Get alerts with optional filters.
    """
    filtered = _alerts
    
    if company:
        filtered = [a for a in filtered if a["company"] == company]
    
    if alert_type:
        filtered = [a for a in filtered if a["type"] == alert_type]
    
    if severity:
        filtered = [a for a in filtered if a["severity"] == severity]
    
    if unread_only:
        filtered = [a for a in filtered if not a["read"] and not a["dismissed"]]
    
    # Sort by created_at descending
    filtered = sorted(filtered, key=lambda x: x["created_at"], reverse=True)
    
    return filtered[:limit]


def mark_alert_read(alert_id: int) -> bool:
    """Mark an alert as read."""
    for alert in _alerts:
        if alert["id"] == alert_id:
            alert["read"] = True
            save_alerts()
            return True
    return False


def dismiss_alert(alert_id: int) -> bool:
    """Dismiss an alert."""
    for alert in _alerts:
        if alert["id"] == alert_id:
            alert["dismissed"] = True
            save_alerts()
            return True
    return False


def get_alert_summary() -> dict:
    """Get summary of current alerts."""
    active_alerts = [a for a in _alerts if not a["dismissed"]]
    unread_alerts = [a for a in active_alerts if not a["read"]]
    
    by_severity = defaultdict(int)
    by_type = defaultdict(int)
    by_company = defaultdict(int)
    
    for alert in active_alerts:
        by_severity[alert["severity"]] += 1
        by_type[alert["type"]] += 1
        by_company[alert["company"]] += 1
    
    return {
        "total_active": len(active_alerts),
        "unread": len(unread_alerts),
        "by_severity": dict(by_severity),
        "by_type": dict(by_type),
        "by_company": dict(by_company),
        "has_critical": by_severity.get("critical", 0) > 0,
        "has_high": by_severity.get("high", 0) > 0,
    }


def check_and_generate_alerts() -> list[dict]:
    """
    Check for anomalies and generate new alerts.
    This should be run periodically (e.g., after new filings are ingested).
    
    Returns:
        List of newly created alerts
    """
    new_alerts = []
    
    try:
        # Get all anomalies
        anomalies = get_all_anomalies()
        
        # Process CapEx anomalies
        for anomaly_data in anomalies.get("capex_anomalies", []):
            company = anomaly_data.get("company")
            for anomaly in anomaly_data.get("anomalies", []):
                # Check if we already have this alert
                existing = [
                    a for a in _alerts 
                    if a["company"] == company 
                    and a["type"] == "capex_anomaly"
                    and a["data"].get("period") == anomaly.get("period")
                    and not a["dismissed"]
                ]
                
                if not existing:
                    severity = "high" if anomaly.get("severity") == "high" else "medium"
                    direction = anomaly.get("direction", "change")
                    
                    alert = create_alert(
                        alert_type="capex_anomaly",
                        severity=severity,
                        company=company,
                        title=f"CapEx {direction.title()} Detected - {company}",
                        description=f"Unusual {direction} in capital expenditure for {anomaly.get('period')}. "
                                   f"{anomaly.get('pct_change_from_mean', 0):.1f}% deviation from mean.",
                        data=anomaly,
                    )
                    new_alerts.append(alert)
        
        # Process sentiment shifts
        for shift_data in anomalies.get("sentiment_shifts", []):
            company = shift_data.get("company")
            for shift in shift_data.get("shifts", []):
                existing = [
                    a for a in _alerts 
                    if a["company"] == company 
                    and a["type"] == "sentiment_shift"
                    and a["data"].get("to_period") == shift.get("to_period")
                    and not a["dismissed"]
                ]
                
                if not existing:
                    severity = "high" if shift.get("severity") == "high" else "medium"
                    direction = shift.get("direction", "changing")
                    
                    alert = create_alert(
                        alert_type="sentiment_shift",
                        severity=severity,
                        company=company,
                        title=f"Sentiment {direction.title()} - {company}",
                        description=f"Significant sentiment shift from {shift.get('from_period')} to {shift.get('to_period')}. "
                                   f"Change: {shift.get('change', 0):.3f}",
                        data=shift,
                    )
                    new_alerts.append(alert)
        
        # Process AI investment changes
        for ai_data in anomalies.get("ai_investment_changes", []):
            company = ai_data.get("company")
            for change in ai_data.get("changes", []):
                existing = [
                    a for a in _alerts 
                    if a["company"] == company 
                    and a["type"] == "ai_investment_change"
                    and a["data"].get("to_period") == change.get("to_period")
                    and not a["dismissed"]
                ]
                
                if not existing:
                    direction = change.get("direction", "changing")
                    severity = "medium" if abs(change.get("pct_change", 0)) > 50 else "low"
                    
                    alert = create_alert(
                        alert_type="ai_investment_change",
                        severity=severity,
                        company=company,
                        title=f"AI Focus {direction.title()} - {company}",
                        description=f"AI investment focus {direction} from {change.get('from_period')} to {change.get('to_period')}. "
                                   f"Change: {change.get('pct_change', 0):.1f}%",
                        data=change,
                    )
                    new_alerts.append(alert)
    
    except Exception as e:
        # Log error but don't fail
        print(f"Error generating alerts: {e}")
    
    return new_alerts


def clear_all_alerts() -> int:
    """Clear all alerts. Returns count of cleared alerts."""
    global _alerts, _alert_id_counter
    count = len(_alerts)
    _alerts = []
    _alert_id_counter = 0
    save_alerts()
    return count


# Load alerts on module import
load_alerts()
