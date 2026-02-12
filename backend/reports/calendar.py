"""
Earnings calendar management and sync.
Tracks earnings dates for all monitored companies.
"""
import json
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from backend.core.config import COMPANIES, BASE_DIR


# Typical earnings schedule patterns (month, week)
# Most EMS companies report quarterly, typically 3-4 weeks after quarter end
TYPICAL_EARNINGS_SCHEDULE = {
    "Q1": {"month": 4, "week": 3},   # Late April
    "Q2": {"month": 7, "week": 3},   # Late July
    "Q3": {"month": 10, "week": 3},  # Late October
    "Q4": {"month": 1, "week": 3},   # Late January (next year)
}

# Storage
CALENDAR_FILE = BASE_DIR / "data" / "earnings_calendar.json"


class EarningsCalendar:
    """Manages earnings calendar and event tracking."""
    
    def __init__(self):
        self._events = []
        self._load_calendar()
    
    def _load_calendar(self):
        """Load saved calendar from disk."""
        try:
            if CALENDAR_FILE.exists():
                with open(CALENDAR_FILE, 'r') as f:
                    self._events = json.load(f)
        except Exception as e:
            print(f"Failed to load calendar: {e}")
            self._events = []
    
    def _save_calendar(self):
        """Save calendar to disk."""
        try:
            CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CALENDAR_FILE, 'w') as f:
                json.dump(self._events, f, indent=2)
        except Exception as e:
            print(f"Failed to save calendar: {e}")
    
    def generate_earnings_calendar(self, year: int = None) -> list:
        """
        Generate estimated earnings dates for all companies.
        Based on typical quarterly reporting patterns.
        """
        if year is None:
            year = datetime.now().year
        
        events = []
        
        for ticker, config in COMPANIES.items():
            company_name = config['name'].split()[0]
            
            for quarter, schedule in TYPICAL_EARNINGS_SCHEDULE.items():
                month = schedule['month']
                week = schedule['week']
                
                # Calculate estimated date (third week of the month)
                event_year = year if quarter != 'Q4' else year + 1
                first_day = datetime(event_year, month, 1)
                # Find the Monday of the target week
                days_to_monday = (7 - first_day.weekday()) % 7
                target_date = first_day + timedelta(days=days_to_monday + (week - 1) * 7)
                
                # Adjust Q4 reporting year label
                fiscal_year = year if quarter != 'Q4' else year
                
                events.append({
                    "id": f"{ticker}_{quarter}_{fiscal_year}",
                    "company": company_name,
                    "ticker": ticker,
                    "quarter": quarter,
                    "fiscal_year": fiscal_year,
                    "estimated_date": target_date.strftime('%Y-%m-%d'),
                    "time": "Before Market" if ticker in ['FLEX', 'JBL'] else "After Market",
                    "event_type": "earnings",
                    "confirmed": False,
                    "status": "estimated",
                })
        
        # Sort by date
        events.sort(key=lambda x: x['estimated_date'])
        
        return events
    
    def get_upcoming_events(self, days: int = 30) -> list:
        """Get events in the next N days."""
        today = datetime.now()
        cutoff = today + timedelta(days=days)
        
        # Generate calendar if empty
        if not self._events:
            self._events = self.generate_earnings_calendar()
            self._save_calendar()
        
        upcoming = []
        for event in self._events:
            event_date = datetime.strptime(event['estimated_date'], '%Y-%m-%d')
            if today <= event_date <= cutoff:
                # Add days until
                event['days_until'] = (event_date - today).days
                upcoming.append(event)
        
        return upcoming
    
    def get_events_by_month(self, year: int, month: int) -> list:
        """Get all events for a specific month."""
        if not self._events:
            self._events = self.generate_earnings_calendar(year)
            self._save_calendar()
        
        return [
            event for event in self._events
            if event['estimated_date'].startswith(f"{year}-{month:02d}")
        ]
    
    def get_company_events(self, company: str) -> list:
        """Get all events for a specific company."""
        if not self._events:
            self._events = self.generate_earnings_calendar()
            self._save_calendar()
        
        return [
            event for event in self._events
            if event['company'].lower() == company.lower()
        ]
    
    def add_custom_event(self, company: str, event_type: str, 
                        date: str, description: str = None) -> dict:
        """Add a custom calendar event."""
        event = {
            "id": f"custom_{company}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "company": company,
            "event_type": event_type,
            "date": date,
            "description": description,
            "confirmed": True,
            "status": "custom",
            "created_at": datetime.now().isoformat(),
        }
        
        self._events.append(event)
        self._save_calendar()
        
        return event
    
    def update_event(self, event_id: str, updates: dict) -> Optional[dict]:
        """Update an existing event."""
        for i, event in enumerate(self._events):
            if event.get('id') == event_id:
                self._events[i].update(updates)
                self._events[i]['updated_at'] = datetime.now().isoformat()
                self._save_calendar()
                return self._events[i]
        return None
    
    def confirm_earnings_date(self, event_id: str, confirmed_date: str,
                             confirmed_time: str = None) -> Optional[dict]:
        """Confirm an earnings date."""
        updates = {
            "confirmed": True,
            "confirmed_date": confirmed_date,
            "status": "confirmed",
        }
        if confirmed_time:
            updates['confirmed_time'] = confirmed_time
        
        return self.update_event(event_id, updates)
    
    def get_calendar_summary(self) -> dict:
        """Get calendar summary statistics."""
        if not self._events:
            self._events = self.generate_earnings_calendar()
        
        today = datetime.now()
        
        # Count by status
        confirmed = sum(1 for e in self._events if e.get('confirmed'))
        upcoming_30 = len(self.get_upcoming_events(30))
        upcoming_7 = len(self.get_upcoming_events(7))
        
        # Next event
        upcoming = self.get_upcoming_events(365)
        next_event = upcoming[0] if upcoming else None
        
        return {
            "total_events": len(self._events),
            "confirmed_events": confirmed,
            "upcoming_30_days": upcoming_30,
            "upcoming_7_days": upcoming_7,
            "next_event": next_event,
            "companies_tracked": len(COMPANIES),
        }
    
    def export_to_ical(self) -> str:
        """Export calendar to iCal format."""
        lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Flex Competitive Intelligence//Earnings Calendar//EN",
            "CALSCALE:GREGORIAN",
            "METHOD:PUBLISH",
        ]
        
        for event in self._events:
            date_str = event.get('confirmed_date') or event.get('estimated_date')
            if not date_str:
                continue
            
            # Convert to iCal date format
            event_date = datetime.strptime(date_str, '%Y-%m-%d')
            ical_date = event_date.strftime('%Y%m%d')
            
            lines.extend([
                "BEGIN:VEVENT",
                f"UID:{event.get('id', 'unknown')}@flex-intel",
                f"DTSTART;VALUE=DATE:{ical_date}",
                f"SUMMARY:{event.get('company', 'Unknown')} {event.get('quarter', '')} Earnings",
                f"DESCRIPTION:{event.get('event_type', 'Earnings Report')}",
                f"STATUS:{'CONFIRMED' if event.get('confirmed') else 'TENTATIVE'}",
                "END:VEVENT",
            ])
        
        lines.append("END:VCALENDAR")
        
        return "\n".join(lines)


# Singleton instance
_calendar = EarningsCalendar()


def get_earnings_calendar(year: int = None) -> list:
    """Get earnings calendar for a year."""
    return _calendar.generate_earnings_calendar(year)


def sync_earnings_to_calendar() -> dict:
    """Sync/refresh earnings calendar."""
    year = datetime.now().year
    events = _calendar.generate_earnings_calendar(year)
    _calendar._events = events
    _calendar._save_calendar()
    return {
        "synced": True,
        "events_count": len(events),
        "year": year,
    }


def get_upcoming_events(days: int = 30) -> list:
    """Get upcoming calendar events."""
    return _calendar.get_upcoming_events(days)


def get_calendar_summary() -> dict:
    """Get calendar summary."""
    return _calendar.get_calendar_summary()


def get_company_calendar(company: str) -> list:
    """Get calendar for a specific company."""
    return _calendar.get_company_events(company)


def add_calendar_event(company: str, event_type: str, 
                      date: str, description: str = None) -> dict:
    """Add a custom calendar event."""
    return _calendar.add_custom_event(company, event_type, date, description)


def confirm_earnings(event_id: str, date: str, time: str = None) -> Optional[dict]:
    """Confirm an earnings date."""
    return _calendar.confirm_earnings_date(event_id, date, time)


def export_ical() -> str:
    """Export calendar to iCal format."""
    return _calendar.export_to_ical()
