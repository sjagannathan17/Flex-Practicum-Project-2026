"""
Slack integration for alert notifications.
Uses Slack webhooks or API for message delivery.
"""
import os
import json
from datetime import datetime
from typing import Optional
import httpx

from backend.core.config import BASE_DIR


class SlackClient:
    """
    Handles Slack notifications for alerts.
    """
    
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN", "")
        self.default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "#competitive-intel")
        self.enabled = bool(self.webhook_url) or bool(self.bot_token)
        self._message_log = []
    
    async def send_alert(
        self,
        alert: dict,
        channel: Optional[str] = None,
    ) -> dict:
        """
        Send an alert to Slack.
        
        Args:
            alert: Alert data to send
            channel: Optional channel override
        """
        blocks = self._build_alert_blocks(alert)
        
        result = {
            "success": False,
            "channel": channel or self.default_channel,
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.webhook_url:
            result = await self._send_via_webhook(blocks, result)
        elif self.bot_token:
            result = await self._send_via_api(blocks, channel, result)
        else:
            # Log message instead of sending
            self._message_log.append({
                "blocks": blocks,
                "channel": channel or self.default_channel,
                "timestamp": result["timestamp"],
                "alert": alert,
            })
            result["logged"] = True
            result["success"] = True
            print(f"[SLACK LOG] Channel: {channel or self.default_channel}, Alert: {alert.get('title', 'Unknown')}")
        
        return result
    
    async def _send_via_webhook(self, blocks: list, result: dict) -> dict:
        """Send message via webhook."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json={"blocks": blocks},
                    timeout=10.0
                )
                result["success"] = response.status_code == 200
                result["status_code"] = response.status_code
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    async def _send_via_api(self, blocks: list, channel: Optional[str], result: dict) -> dict:
        """Send message via Slack API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                    json={
                        "channel": channel or self.default_channel,
                        "blocks": blocks,
                    },
                    timeout=10.0
                )
                data = response.json()
                result["success"] = data.get("ok", False)
                if not result["success"]:
                    result["error"] = data.get("error", "Unknown error")
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _build_alert_blocks(self, alert: dict) -> list:
        """Build Slack blocks for an alert."""
        severity = alert.get("severity", "low")
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji} {alert.get('title', 'Alert')}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Company:*\n{alert.get('company', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Type:*\n{alert.get('type', 'Unknown').replace('_', ' ').title()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Details:*\n{alert.get('message', 'No details available')}"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "📊 <http://localhost:3000/alerts|View in Dashboard> • Flex Competitive Intelligence Platform"
                    }
                ]
            }
        ]
        
        return blocks
    
    async def send_digest(
        self,
        alerts: list[dict],
        channel: Optional[str] = None,
        period: str = "daily"
    ) -> dict:
        """
        Send a digest of alerts to Slack.
        """
        if not alerts:
            return {"success": True, "message": "No alerts to send"}
        
        # Group by severity
        high_count = len([a for a in alerts if a.get("severity") == "high"])
        medium_count = len([a for a in alerts if a.get("severity") == "medium"])
        low_count = len([a for a in alerts if a.get("severity") == "low"])
        
        # Group by company
        companies = list(set(a.get("company", "Unknown") for a in alerts))
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 {period.title()} Alert Digest",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{len(alerts)} alerts* detected across *{len(companies)} companies*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"🔴 *High:* {high_count}"},
                    {"type": "mrkdwn", "text": f"🟡 *Medium:* {medium_count}"},
                    {"type": "mrkdwn", "text": f"🟢 *Low:* {low_count}"},
                    {"type": "mrkdwn", "text": f"🏢 *Companies:* {', '.join(companies[:3])}{'...' if len(companies) > 3 else ''}"}
                ]
            },
            {
                "type": "divider"
            }
        ]
        
        # Add top 5 alerts
        for alert in alerts[:5]:
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(alert.get("severity", "low"), "⚪")
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{severity_emoji} *{alert.get('company', 'Unknown')}* - {alert.get('title', 'Alert')}\n_{alert.get('message', '')[:100]}..._"
                }
            })
        
        if len(alerts) > 5:
            blocks.append({
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"_...and {len(alerts) - 5} more alerts_"}
                ]
            })
        
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View All Alerts", "emoji": True},
                    "url": "http://localhost:3000/alerts"
                }
            ]
        })
        
        result = {
            "success": False,
            "channel": channel or self.default_channel,
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.webhook_url:
            result = await self._send_via_webhook(blocks, result)
        elif self.bot_token:
            result = await self._send_via_api(blocks, channel, result)
        else:
            self._message_log.append({
                "blocks": blocks,
                "channel": channel or self.default_channel,
                "timestamp": result["timestamp"],
                "type": "digest",
            })
            result["logged"] = True
            result["success"] = True
            print(f"[SLACK LOG] Digest: {len(alerts)} alerts to {channel or self.default_channel}")
        
        return result
    
    def get_message_log(self) -> list[dict]:
        """Get logged messages (when Slack is not configured)."""
        return self._message_log


# Singleton client instance
_slack_client = SlackClient()


async def send_slack_alert(alert: dict, channel: Optional[str] = None) -> dict:
    """Send a single alert to Slack."""
    return await _slack_client.send_alert(alert, channel)


async def send_slack_digest(alerts: list[dict], channel: Optional[str] = None, period: str = "daily") -> dict:
    """Send a digest of alerts to Slack."""
    return await _slack_client.send_digest(alerts, channel, period)


def get_slack_client() -> SlackClient:
    """Get the singleton Slack client instance."""
    return _slack_client
