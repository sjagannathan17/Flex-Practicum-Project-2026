"""
API routes for alert management.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.alerts import (
    get_alerts,
    mark_alert_read,
    dismiss_alert,
    get_alert_summary,
    check_and_generate_alerts,
    clear_all_alerts,
    run_alert_detection,
    send_alert_notification,
    send_alert_digest,
    send_slack_alert,
    send_slack_digest,
    get_email_sender,
    get_slack_client,
)

router = APIRouter()


class NotificationRequest(BaseModel):
    email: Optional[str] = None
    slack_channel: Optional[str] = None


class DigestRequest(BaseModel):
    email: Optional[str] = None
    slack_channel: Optional[str] = None
    period: str = "daily"


@router.get("/alerts")
async def list_alerts(
    company: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 50,
):
    """
    Get alerts with optional filters.
    """
    try:
        alerts = get_alerts(
            company=company,
            alert_type=alert_type,
            severity=severity,
            unread_only=unread_only,
            limit=limit,
        )
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/summary")
async def alerts_summary():
    """
    Get summary of current alerts.
    """
    try:
        return get_alert_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/check")
async def check_for_alerts():
    """
    Check for new anomalies and generate alerts.
    """
    try:
        new_alerts = check_and_generate_alerts()
        return {
            "new_alerts": len(new_alerts),
            "alerts": new_alerts,
            "message": f"Generated {len(new_alerts)} new alerts",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/detect")
async def run_detection(company: Optional[str] = None):
    """
    Run full alert detection for one or all companies.
    Uses the enhanced detector with multiple alert types.
    """
    try:
        result = run_alert_detection(company)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/read")
async def read_alert(alert_id: int):
    """
    Mark an alert as read.
    """
    try:
        success = mark_alert_read(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "success", "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_single_alert(alert_id: int):
    """
    Dismiss an alert.
    """
    try:
        success = dismiss_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "success", "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts")
async def clear_alerts():
    """
    Clear all alerts.
    """
    try:
        count = clear_all_alerts()
        return {"status": "success", "cleared": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/notify")
async def send_notification(alert_id: int, request: NotificationRequest):
    """
    Send notification for a specific alert via email and/or Slack.
    """
    try:
        alerts = get_alerts(limit=1000)
        alert = next((a for a in alerts if a.get("id") == alert_id), None)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        results = {}
        
        if request.email:
            results["email"] = send_alert_notification(request.email, alert)
        
        if request.slack_channel:
            results["slack"] = await send_slack_alert(alert, request.slack_channel)
        
        if not request.email and not request.slack_channel:
            raise HTTPException(status_code=400, detail="Must provide email or slack_channel")
        
        return {
            "status": "success",
            "alert_id": alert_id,
            "notifications": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/notify/digest")
async def send_digest(request: DigestRequest):
    """
    Send a digest of recent alerts via email and/or Slack.
    """
    try:
        alerts = get_alerts(limit=100)
        
        if not alerts:
            return {"status": "success", "message": "No alerts to send"}
        
        results = {}
        
        if request.email:
            results["email"] = send_alert_digest(request.email, alerts, request.period)
        
        if request.slack_channel:
            results["slack"] = await send_slack_digest(alerts, request.slack_channel, request.period)
        
        if not request.email and not request.slack_channel:
            raise HTTPException(status_code=400, detail="Must provide email or slack_channel")
        
        return {
            "status": "success",
            "alert_count": len(alerts),
            "notifications": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/config")
async def get_notification_config():
    """
    Get current notification configuration status.
    """
    email_sender = get_email_sender()
    slack_client = get_slack_client()
    
    return {
        "email": {
            "enabled": email_sender.enabled,
            "has_api_key": bool(email_sender.api_key),
            "from_email": email_sender.from_email,
        },
        "slack": {
            "enabled": slack_client.enabled,
            "has_webhook": bool(slack_client.webhook_url),
            "has_bot_token": bool(slack_client.bot_token),
            "default_channel": slack_client.default_channel,
        },
    }


@router.post("/alerts/test/email")
async def test_email_notification(email: str):
    """
    Send a test email notification.
    """
    test_alert = {
        "type": "test",
        "company": "Test Company",
        "title": "Test Alert Notification",
        "message": "This is a test alert to verify email notifications are working correctly.",
        "severity": "low",
    }
    
    result = send_alert_notification(email, test_alert)
    return result


@router.post("/alerts/test/slack")
async def test_slack_notification(channel: Optional[str] = None):
    """
    Send a test Slack notification.
    """
    test_alert = {
        "type": "test",
        "company": "Test Company",
        "title": "Test Alert Notification",
        "message": "This is a test alert to verify Slack notifications are working correctly.",
        "severity": "low",
    }
    
    result = await send_slack_alert(test_alert, channel)
    return result
