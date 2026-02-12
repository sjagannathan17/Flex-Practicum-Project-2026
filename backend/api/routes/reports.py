"""
API routes for reports and calendar.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from backend.reports.auto_summarizer import (
    get_company_summaries,
    generate_intelligence_brief,
)
from backend.reports.scheduler import (
    schedule_weekly_report,
    schedule_monthly_report,
    get_scheduled_reports,
    run_scheduled_report,
    cancel_scheduled_report,
    get_generated_reports,
)
from backend.reports.calendar import (
    get_earnings_calendar,
    sync_earnings_to_calendar,
    get_upcoming_events,
    get_calendar_summary,
    get_company_calendar,
    add_calendar_event,
    confirm_earnings,
    export_ical,
)

router = APIRouter()


# ============== Auto-Summarizer Routes ==============

@router.get("/summaries/{company}")
async def get_summaries(company: str, limit: int = 3):
    """Get auto-generated summaries for a company's latest filings."""
    try:
        return await get_company_summaries(company.title(), limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brief/{company}")
async def get_brief(company: str):
    """Get comprehensive intelligence brief for a company."""
    try:
        return await generate_intelligence_brief(company.title())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Report Scheduler Routes ==============

class ScheduleRequest(BaseModel):
    report_type: str = "comprehensive"
    recipients: Optional[list] = None


class RunReportRequest(BaseModel):
    report_type: str
    save_to_file: Optional[bool] = False


@router.get("/schedules")
async def list_schedules():
    """Get all scheduled reports."""
    return {
        "schedules": get_scheduled_reports(),
        "total": len(get_scheduled_reports()),
    }


@router.post("/schedules/weekly")
async def create_weekly_schedule(request: ScheduleRequest):
    """Schedule a weekly report."""
    try:
        schedule = schedule_weekly_report(request.report_type, request.recipients)
        return {
            "success": True,
            "schedule": schedule,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedules/monthly")
async def create_monthly_schedule(request: ScheduleRequest):
    """Schedule a monthly report."""
    try:
        schedule = schedule_monthly_report(request.report_type, request.recipients)
        return {
            "success": True,
            "schedule": schedule,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Cancel a scheduled report."""
    success = cancel_scheduled_report(schedule_id)
    if success:
        return {"success": True, "message": f"Schedule {schedule_id} cancelled"}
    raise HTTPException(status_code=404, detail="Schedule not found")


@router.post("/run")
async def run_report(request: RunReportRequest):
    """Generate a report immediately."""
    try:
        report = run_scheduled_report(request.report_type, {
            "save_to_file": request.save_to_file,
        })
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generated")
async def list_generated_reports(limit: int = 10):
    """Get recently generated reports."""
    return {
        "reports": get_generated_reports(limit),
        "total": len(get_generated_reports(limit)),
    }


# ============== Calendar Routes ==============

@router.get("/calendar")
async def get_calendar(year: Optional[int] = None):
    """Get earnings calendar for a year."""
    try:
        if year is None:
            year = datetime.now().year
        events = get_earnings_calendar(year)
        return {
            "year": year,
            "events": events,
            "total": len(events),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calendar/sync")
async def sync_calendar():
    """Sync/refresh earnings calendar."""
    try:
        result = sync_earnings_to_calendar()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/upcoming")
async def get_upcoming(days: int = 30):
    """Get upcoming calendar events."""
    try:
        events = get_upcoming_events(days)
        return {
            "days": days,
            "events": events,
            "total": len(events),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/summary")
async def calendar_summary():
    """Get calendar summary statistics."""
    try:
        return get_calendar_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/company/{company}")
async def get_company_events(company: str):
    """Get calendar events for a specific company."""
    try:
        events = get_company_calendar(company.title())
        return {
            "company": company.title(),
            "events": events,
            "total": len(events),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AddEventRequest(BaseModel):
    company: str
    event_type: str
    date: str
    description: Optional[str] = None


@router.post("/calendar/event")
async def add_event(request: AddEventRequest):
    """Add a custom calendar event."""
    try:
        event = add_calendar_event(
            request.company,
            request.event_type,
            request.date,
            request.description
        )
        return {"success": True, "event": event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConfirmEventRequest(BaseModel):
    date: str
    time: Optional[str] = None


@router.post("/calendar/event/{event_id}/confirm")
async def confirm_event(event_id: str, request: ConfirmEventRequest):
    """Confirm an earnings date."""
    try:
        event = confirm_earnings(event_id, request.date, request.time)
        if event:
            return {"success": True, "event": event}
        raise HTTPException(status_code=404, detail="Event not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar/export/ical")
async def export_calendar_ical():
    """Export calendar to iCal format."""
    try:
        ical_content = export_ical()
        return Response(
            content=ical_content,
            media_type="text/calendar",
            headers={
                "Content-Disposition": "attachment; filename=earnings_calendar.ics"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
