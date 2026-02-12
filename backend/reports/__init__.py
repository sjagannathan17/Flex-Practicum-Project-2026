"""
Reports module for auto-generated reports and scheduling.
"""
from .auto_summarizer import (
    summarize_new_filing,
    get_company_summaries,
    generate_intelligence_brief,
)
from .scheduler import (
    schedule_weekly_report,
    schedule_monthly_report,
    get_scheduled_reports,
    run_scheduled_report,
)
from .calendar import (
    get_earnings_calendar,
    sync_earnings_to_calendar,
    get_upcoming_events,
)

__all__ = [
    # Auto-summarizer
    "summarize_new_filing",
    "get_company_summaries",
    "generate_intelligence_brief",
    # Scheduler
    "schedule_weekly_report",
    "schedule_monthly_report",
    "get_scheduled_reports",
    "run_scheduled_report",
    # Calendar
    "get_earnings_calendar",
    "sync_earnings_to_calendar",
    "get_upcoming_events",
]
