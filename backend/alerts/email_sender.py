"""
Email notification system for alerts.
Uses SendGrid for email delivery (or fallback to console logging).
"""
import os
from datetime import datetime
from typing import Optional
from collections import defaultdict

from backend.core.config import BASE_DIR

# Try to import SendGrid
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    HAS_SENDGRID = True
except ImportError:
    HAS_SENDGRID = False
    print("SendGrid not installed. Email notifications will be logged only.")


class EmailSender:
    """
    Handles email notifications for alerts.
    """
    
    def __init__(self):
        self.api_key = os.getenv("SENDGRID_API_KEY", "")
        self.from_email = os.getenv("ALERT_FROM_EMAIL", "alerts@flexintel.ai")
        self.enabled = bool(self.api_key) and HAS_SENDGRID
        self._email_log = []
        
    def send_alert_email(
        self,
        to_email: str,
        subject: str,
        alert_data: dict,
        html: bool = True
    ) -> dict:
        """
        Send an alert notification email.
        
        Args:
            to_email: Recipient email address
            subject: Email subject line
            alert_data: Alert information to include
            html: Whether to send HTML email
        """
        if html:
            content = self._generate_html_content(alert_data)
            content_type = "text/html"
        else:
            content = self._generate_text_content(alert_data)
            content_type = "text/plain"
        
        result = {
            "success": False,
            "to": to_email,
            "subject": subject,
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.enabled:
            try:
                message = Mail(
                    from_email=Email(self.from_email),
                    to_emails=To(to_email),
                    subject=subject,
                    html_content=Content(content_type, content) if html else None,
                    plain_text_content=content if not html else None,
                )
                
                sg = SendGridAPIClient(self.api_key)
                response = sg.send(message)
                
                result["success"] = response.status_code in [200, 201, 202]
                result["status_code"] = response.status_code
                
            except Exception as e:
                result["error"] = str(e)
        else:
            # Log email instead of sending
            self._email_log.append({
                "to": to_email,
                "subject": subject,
                "content": content,
                "timestamp": result["timestamp"],
            })
            result["logged"] = True
            result["success"] = True  # Consider logged as success
            print(f"[EMAIL LOG] To: {to_email}, Subject: {subject}")
        
        return result
    
    def send_digest_email(
        self,
        to_email: str,
        alerts: list[dict],
        period: str = "daily"
    ) -> dict:
        """
        Send a digest email with multiple alerts.
        """
        if not alerts:
            return {"success": True, "message": "No alerts to send"}
        
        subject = f"Flex Intel {period.title()} Alert Digest - {len(alerts)} Alerts"
        
        digest_data = {
            "type": "digest",
            "period": period,
            "alert_count": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat(),
        }
        
        return self.send_alert_email(to_email, subject, digest_data)
    
    def _generate_html_content(self, alert_data: dict) -> str:
        """Generate HTML email content."""
        if alert_data.get("type") == "digest":
            return self._generate_digest_html(alert_data)
        
        severity_colors = {
            "high": "#EF4444",
            "medium": "#F59E0B",
            "low": "#10B981",
        }
        
        severity = alert_data.get("severity", "low")
        color = severity_colors.get(severity, "#6B7280")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f3f4f6; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 24px; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .severity-badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: {color}; color: white; margin-top: 8px; }}
                .content {{ padding: 24px; }}
                .alert-title {{ font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 8px; }}
                .alert-company {{ color: #6b7280; font-size: 14px; margin-bottom: 16px; }}
                .alert-message {{ background: #f9fafb; border-radius: 8px; padding: 16px; color: #374151; line-height: 1.6; }}
                .footer {{ background: #f9fafb; padding: 16px 24px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #9ca3af; }}
                .btn {{ display: inline-block; background: #3b82f6; color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; margin-top: 16px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔔 Flex Intelligence Alert</h1>
                    <span class="severity-badge">{severity.upper()} PRIORITY</span>
                </div>
                <div class="content">
                    <div class="alert-title">{alert_data.get('title', 'Alert')}</div>
                    <div class="alert-company">📊 {alert_data.get('company', 'Unknown Company')}</div>
                    <div class="alert-message">{alert_data.get('message', '')}</div>
                    <a href="http://localhost:3000/alerts" class="btn">View in Dashboard →</a>
                </div>
                <div class="footer">
                    Flex Competitive Intelligence Platform • AI Powered<br>
                    Sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_digest_html(self, digest_data: dict) -> str:
        """Generate HTML for digest emails."""
        alerts = digest_data.get("alerts", [])
        
        # Group alerts by company
        by_company = defaultdict(list)
        for alert in alerts:
            by_company[alert.get("company", "Unknown")].append(alert)
        
        company_sections = ""
        for company, company_alerts in by_company.items():
            alerts_html = ""
            for alert in company_alerts[:5]:  # Limit per company
                severity = alert.get("severity", "low")
                color = {"high": "#EF4444", "medium": "#F59E0B", "low": "#10B981"}.get(severity, "#6B7280")
                alerts_html += f"""
                <div style="border-left: 3px solid {color}; padding: 8px 12px; margin: 8px 0; background: #f9fafb; border-radius: 0 8px 8px 0;">
                    <div style="font-weight: 600; color: #1f2937;">{alert.get('title', 'Alert')}</div>
                    <div style="font-size: 13px; color: #6b7280; margin-top: 4px;">{alert.get('message', '')[:100]}...</div>
                </div>
                """
            
            company_sections += f"""
            <div style="margin-bottom: 24px;">
                <h3 style="color: #1f2937; margin-bottom: 12px;">🏢 {company}</h3>
                {alerts_html}
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f3f4f6; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 24px; }}
                .content {{ padding: 24px; }}
                .stats {{ display: flex; gap: 16px; margin-bottom: 24px; }}
                .stat {{ background: #f3f4f6; padding: 16px; border-radius: 8px; flex: 1; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: 700; color: #1f2937; }}
                .stat-label {{ font-size: 12px; color: #6b7280; }}
                .footer {{ background: #f9fafb; padding: 16px 24px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #9ca3af; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 {digest_data.get('period', 'Daily').title()} Alert Digest</h1>
                    <p style="margin: 8px 0 0 0; opacity: 0.9;">Flex Competitive Intelligence Platform</p>
                </div>
                <div class="content">
                    <div class="stats">
                        <div class="stat">
                            <div class="stat-value">{len(alerts)}</div>
                            <div class="stat-label">Total Alerts</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{len([a for a in alerts if a.get('severity') == 'high'])}</div>
                            <div class="stat-label">High Priority</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{len(by_company)}</div>
                            <div class="stat-label">Companies</div>
                        </div>
                    </div>
                    {company_sections}
                    <a href="http://localhost:3000/alerts" style="display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none;">View All Alerts →</a>
                </div>
                <div class="footer">
                    Flex Competitive Intelligence Platform • AI Powered<br>
                    Digest for {datetime.now().strftime('%Y-%m-%d')}
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _generate_text_content(self, alert_data: dict) -> str:
        """Generate plain text email content."""
        lines = [
            "=" * 50,
            "FLEX COMPETITIVE INTELLIGENCE ALERT",
            "=" * 50,
            "",
            f"Type: {alert_data.get('type', 'Unknown')}",
            f"Company: {alert_data.get('company', 'Unknown')}",
            f"Severity: {alert_data.get('severity', 'low').upper()}",
            "",
            f"Title: {alert_data.get('title', 'Alert')}",
            "",
            "Message:",
            alert_data.get('message', ''),
            "",
            "-" * 50,
            f"Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "View in dashboard: http://localhost:3000/alerts",
        ]
        return "\n".join(lines)
    
    def get_email_log(self) -> list[dict]:
        """Get logged emails (when SendGrid is not configured)."""
        return self._email_log


# Singleton sender instance
_email_sender = EmailSender()


def send_alert_notification(to_email: str, alert: dict) -> dict:
    """Send a single alert notification."""
    subject = f"[{alert.get('severity', 'INFO').upper()}] {alert.get('title', 'Alert')} - {alert.get('company', '')}"
    return _email_sender.send_alert_email(to_email, subject, alert)


def send_alert_digest(to_email: str, alerts: list[dict], period: str = "daily") -> dict:
    """Send a digest of multiple alerts."""
    return _email_sender.send_digest_email(to_email, alerts, period)


def get_email_sender() -> EmailSender:
    """Get the singleton email sender instance."""
    return _email_sender
