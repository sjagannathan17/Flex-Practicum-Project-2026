"""
Alerts module for competitive intelligence notifications.
"""
from .alert_manager import (
    create_alert,
    get_alerts,
    mark_alert_read,
    dismiss_alert,
    get_alert_summary,
    check_and_generate_alerts,
    clear_all_alerts,
)
from .detector import run_alert_detection, get_alert_detector
from .email_sender import send_alert_notification, send_alert_digest, get_email_sender
from .slack_client import send_slack_alert, send_slack_digest, get_slack_client

__all__ = [
    # Alert Manager
    "create_alert",
    "get_alerts",
    "mark_alert_read",
    "dismiss_alert",
    "get_alert_summary",
    "check_and_generate_alerts",
    "clear_all_alerts",
    # Detector
    "run_alert_detection",
    "get_alert_detector",
    # Email
    "send_alert_notification",
    "send_alert_digest",
    "get_email_sender",
    # Slack
    "send_slack_alert",
    "send_slack_digest",
    "get_slack_client",
]
