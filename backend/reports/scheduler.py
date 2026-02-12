"""
Report scheduler for automated report generation.
Schedules and manages weekly/monthly competitive intelligence reports.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from backend.core.config import COMPANIES, BASE_DIR
from backend.analytics.sentiment import analyze_company_sentiment
from backend.analytics.classifier import compare_investment_focus
from backend.analytics.trends import compare_company_trends
from backend.analytics.anomaly import get_all_anomalies
from backend.exports.excel import generate_comparison_excel
from backend.exports.pdf import generate_html_preview


# Storage for scheduled reports
REPORTS_DIR = BASE_DIR / "data" / "reports"
SCHEDULE_FILE = BASE_DIR / "data" / "report_schedule.json"


class ReportScheduler:
    """Manages scheduled report generation."""
    
    def __init__(self):
        self._schedules = {}
        self._generated_reports = []
        self._load_schedules()
    
    def _load_schedules(self):
        """Load saved schedules from disk."""
        try:
            if SCHEDULE_FILE.exists():
                with open(SCHEDULE_FILE, 'r') as f:
                    self._schedules = json.load(f)
        except Exception as e:
            print(f"Failed to load schedules: {e}")
            self._schedules = {}
    
    def _save_schedules(self):
        """Save schedules to disk."""
        try:
            SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SCHEDULE_FILE, 'w') as f:
                json.dump(self._schedules, f, indent=2)
        except Exception as e:
            print(f"Failed to save schedules: {e}")
    
    def schedule_report(self, report_type: str, frequency: str, 
                       recipients: list = None, options: dict = None) -> dict:
        """
        Schedule a recurring report.
        
        Args:
            report_type: 'competitive_analysis', 'sentiment', 'anomaly', 'comprehensive'
            frequency: 'weekly', 'monthly', 'daily'
            recipients: List of email addresses
            options: Additional options (format, companies, etc.)
        """
        schedule_id = f"{report_type}_{frequency}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate next run time
        now = datetime.now()
        if frequency == 'daily':
            next_run = now.replace(hour=6, minute=0, second=0) + timedelta(days=1)
        elif frequency == 'weekly':
            # Monday 6 AM
            days_until_monday = (7 - now.weekday()) % 7 or 7
            next_run = now.replace(hour=6, minute=0, second=0) + timedelta(days=days_until_monday)
        else:  # monthly
            # First of next month
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=6, minute=0, second=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=6, minute=0, second=0)
        
        schedule = {
            "id": schedule_id,
            "report_type": report_type,
            "frequency": frequency,
            "recipients": recipients or [],
            "options": options or {},
            "created_at": now.isoformat(),
            "next_run": next_run.isoformat(),
            "last_run": None,
            "run_count": 0,
            "active": True,
        }
        
        self._schedules[schedule_id] = schedule
        self._save_schedules()
        
        return schedule
    
    def get_schedules(self) -> list:
        """Get all scheduled reports."""
        return list(self._schedules.values())
    
    def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel a scheduled report."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id]['active'] = False
            self._save_schedules()
            return True
        return False
    
    def run_report(self, report_type: str, options: dict = None) -> dict:
        """
        Generate a report immediately.
        """
        options = options or {}
        
        report = {
            "type": report_type,
            "generated_at": datetime.now().isoformat(),
            "data": {},
        }
        
        if report_type == 'competitive_analysis':
            report['data'] = self._generate_competitive_analysis()
        elif report_type == 'sentiment':
            report['data'] = self._generate_sentiment_report()
        elif report_type == 'anomaly':
            report['data'] = self._generate_anomaly_report()
        elif report_type == 'comprehensive':
            report['data'] = self._generate_comprehensive_report()
        else:
            report['data'] = {'error': f'Unknown report type: {report_type}'}
        
        # Store generated report
        self._generated_reports.append(report)
        
        # Save to file if requested
        if options.get('save_to_file'):
            self._save_report_to_file(report)
        
        return report
    
    def _generate_competitive_analysis(self) -> dict:
        """Generate competitive analysis report."""
        investment_data = compare_investment_focus()
        trends_data = compare_company_trends()
        
        return {
            "title": "Competitive Investment Analysis",
            "period": f"{datetime.now().strftime('%B %Y')}",
            "investment_comparison": investment_data,
            "trends_comparison": trends_data,
            "key_insights": self._extract_key_insights(investment_data, trends_data),
        }
    
    def _generate_sentiment_report(self) -> dict:
        """Generate sentiment analysis report."""
        sentiments = {}
        for ticker, config in COMPANIES.items():
            company = config['name'].split()[0]
            sentiments[company] = analyze_company_sentiment(company)
        
        # Find extremes
        sorted_by_sentiment = sorted(
            sentiments.items(),
            key=lambda x: x[1].get('sentiment_score', 0),
            reverse=True
        )
        
        return {
            "title": "Sentiment Analysis Report",
            "period": f"{datetime.now().strftime('%B %Y')}",
            "company_sentiments": sentiments,
            "most_positive": sorted_by_sentiment[0][0] if sorted_by_sentiment else None,
            "most_cautious": sorted_by_sentiment[-1][0] if sorted_by_sentiment else None,
        }
    
    def _generate_anomaly_report(self) -> dict:
        """Generate anomaly detection report."""
        anomalies = get_all_anomalies()
        
        # Count anomalies by type
        anomaly_counts = {}
        total_anomalies = 0
        
        for company, company_anomalies in anomalies.items():
            for anomaly_type, anomaly_list in company_anomalies.items():
                if isinstance(anomaly_list, list):
                    count = len(anomaly_list)
                    anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + count
                    total_anomalies += count
        
        return {
            "title": "Anomaly Detection Report",
            "period": f"{datetime.now().strftime('%B %Y')}",
            "total_anomalies": total_anomalies,
            "by_type": anomaly_counts,
            "by_company": anomalies,
            "alert_level": "high" if total_anomalies > 10 else "medium" if total_anomalies > 5 else "low",
        }
    
    def _generate_comprehensive_report(self) -> dict:
        """Generate comprehensive monthly report."""
        return {
            "title": "Comprehensive Competitive Intelligence Report",
            "period": f"{datetime.now().strftime('%B %Y')}",
            "competitive_analysis": self._generate_competitive_analysis(),
            "sentiment_analysis": self._generate_sentiment_report(),
            "anomaly_detection": self._generate_anomaly_report(),
            "executive_summary": self._generate_executive_summary(),
        }
    
    def _extract_key_insights(self, investment_data: dict, trends_data: dict) -> list:
        """Extract key insights from data."""
        insights = []
        
        # Investment leaders
        if investment_data.get('ai_leaders'):
            insights.append(f"AI Investment Leaders: {', '.join(investment_data['ai_leaders'][:2])}")
        
        # Trend observations
        companies = trends_data.get('companies', {})
        positive_outlook = [c for c, d in companies.items() if d.get('outlook') == 'positive']
        if positive_outlook:
            insights.append(f"Positive outlook: {', '.join(positive_outlook[:2])}")
        
        return insights
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary."""
        now = datetime.now()
        return f"""
Executive Summary - {now.strftime('%B %Y')}

This comprehensive report analyzes the competitive landscape of the EMS 
(Electronics Manufacturing Services) industry, focusing on Flex, Jabil, 
Celestica, Benchmark, and Sanmina.

Key areas covered:
- AI and Data Center investment trends
- Sentiment analysis from SEC filings
- Anomaly detection in financial metrics
- Geographic footprint analysis

Report generated: {now.strftime('%Y-%m-%d %H:%M')}
        """.strip()
    
    def _save_report_to_file(self, report: dict):
        """Save report to file."""
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{report['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = REPORTS_DIR / filename
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            report['saved_to'] = str(filepath)
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    def get_generated_reports(self, limit: int = 10) -> list:
        """Get recently generated reports."""
        return self._generated_reports[-limit:]


# Singleton instance
_scheduler = ReportScheduler()


def schedule_weekly_report(report_type: str = 'comprehensive', 
                          recipients: list = None) -> dict:
    """Schedule a weekly report."""
    return _scheduler.schedule_report(report_type, 'weekly', recipients)


def schedule_monthly_report(report_type: str = 'comprehensive',
                           recipients: list = None) -> dict:
    """Schedule a monthly report."""
    return _scheduler.schedule_report(report_type, 'monthly', recipients)


def get_scheduled_reports() -> list:
    """Get all scheduled reports."""
    return _scheduler.get_schedules()


def run_scheduled_report(report_type: str, options: dict = None) -> dict:
    """Run a report immediately."""
    return _scheduler.run_report(report_type, options)


def cancel_scheduled_report(schedule_id: str) -> bool:
    """Cancel a scheduled report."""
    return _scheduler.cancel_schedule(schedule_id)


def get_generated_reports(limit: int = 10) -> list:
    """Get recently generated reports."""
    return _scheduler.get_generated_reports(limit)
