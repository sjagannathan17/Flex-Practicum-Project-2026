'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  CalendarDays,
  Clock,
  RefreshCw,
  Download,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Filter,
  AlertCircle,
  LayoutList,
  Calendar,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const COMPANY_COLORS: Record<string, string> = {
  Flex: '#3B82F6',
  Jabil: '#10B981',
  Celestica: '#6366F1',
  Benchmark: '#F59E0B',
  Sanmina: '#EF4444',
};

interface CalendarEvent {
  id: string;
  company: string;
  ticker: string;
  quarter: string;
  fiscal_year: number;
  estimated_date: string;
  time: string;
  event_type: string;
  confirmed: boolean;
  status: string;
  days_until?: number;
}

interface CalendarSummary {
  total_events: number;
  confirmed_events: number;
  upcoming_30_days: number;
  upcoming_7_days: number;
  next_event: CalendarEvent | null;
  companies_tracked: number;
}

// Static watch list per company
const WATCH_LIST: Record<string, string[]> = {
  Sanmina: [
    'Server ecosystem revenue as % of total',
    'New hyperscaler qualification announcements',
    'Networking segment margin trend',
    'Any commentary on AI infrastructure demand',
  ],
  Flex: [
    'AI/DC segment revenue guidance update',
    'Liquid cooling and power module order trajectory',
    'Hyperscaler customer concentration',
    'FY2026 CapEx commitment vs competitors',
  ],
  Benchmark: [
    'HPC vs traditional segment revenue split',
    'Any new AI-adjacent customer wins',
    'Medical/defense segment stability',
    'Management tone on AI opportunity sizing',
  ],
  Celestica: [
    'AI networking platform revenue (AMD partnership)',
    'Hyperscaler customer count and pipeline',
    'Gross margin expansion in data center segment',
    'FY2026 capacity investment guidance',
  ],
  Jabil: [
    'AI server revenue growth rate (track vs +32% YoY baseline)',
    'New hyperscaler supplier qualifications',
    'AI vs non-AI revenue mix shift',
    'Any commentary on Flex as competitive reference',
  ],
};

// Static strip data (ordered by date ascending)
const WATCH_STRIP = [
  { company: 'Sanmina', quarter: 'Q2 FY2026', date: 'Mar 29, 2026', days: 6, confirmed: false },
  { company: 'Flex', quarter: 'Q4 FY2026', date: 'Mar 31, 2026', days: 8, confirmed: false },
  { company: 'Benchmark', quarter: 'Q1 FY2026', date: 'Mar 31, 2026', days: 8, confirmed: false },
  { company: 'Celestica', quarter: 'Q1 FY2026', date: 'Apr 20, 2026', days: 27, confirmed: false },
  { company: 'Jabil', quarter: 'Q3 FY2026', date: 'May 31, 2026', days: 68, confirmed: false },
];

// Post-earnings intel mock data
const POST_EARNINGS = [
  {
    company: 'Jabil',
    quarter: 'Q2 FY2026',
    date: 'Feb 2026',
    takeaways: [
      'AI-related revenue +32% YoY, above guidance',
      'Added 2 new hyperscaler customers',
      'Raised FY2026 revenue guidance by 4%',
    ],
    flexImpact: 'Jabil accelerating faster than Flex in AI server segment',
    impactLevel: 'warn' as const,
  },
  {
    company: 'Celestica',
    quarter: 'Q4 FY2025',
    date: 'Jan 2026',
    takeaways: [
      'Data center revenue now 45% of total (vs 32% prior year)',
      'AMD partnership contributing measurable revenue',
      'Gross margin expanded 80bps YoY',
    ],
    flexImpact: 'Direct overlap in hyperscaler hardware — Celestica gaining share',
    impactLevel: 'high' as const,
  },
  {
    company: 'Benchmark',
    quarter: 'Q4 FY2025',
    date: 'Feb 2026',
    takeaways: [
      'HPC segment grew 18% but from small base',
      'Medical/defense stable, no AI guidance provided',
      'Conservative FY2026 outlook',
    ],
    flexImpact: 'Low competitive threat — not accelerating in AI/DC',
    impactLevel: 'low' as const,
  },
];

function CountdownBadge({ days }: { days: number }) {
  if (days <= 7)
    return (
      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-600 border border-red-300">
        {days} days
      </span>
    );
  if (days <= 14)
    return (
      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-orange-100 text-orange-600 border border-orange-300">
        {days} days
      </span>
    );
  return (
    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">
      {days} days
    </span>
  );
}

function StatusTag({ confirmed }: { confirmed: boolean }) {
  return confirmed ? (
    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-green-100 text-green-700">
      Confirmed
    </span>
  ) : (
    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500">
      Estimated
    </span>
  );
}

function ImpactLine({ level, text }: { level: 'warn' | 'high' | 'low'; text: string }) {
  if (level === 'high')
    return (
      <p className="text-xs font-medium text-red-600 border-l-2 border-red-300 pl-2 mt-2">
        {text}
      </p>
    );
  if (level === 'warn')
    return (
      <p className="text-xs font-medium text-orange-600 border-l-2 border-orange-300 pl-2 mt-2">
        {text}
      </p>
    );
  return (
    <p className="text-xs font-medium text-green-600 border-l-2 border-green-300 pl-2 mt-2">
      {text}
    </p>
  );
}

export default function CalendarPage() {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [upcoming, setUpcoming] = useState<CalendarEvent[]>([]);
  const [summary, setSummary] = useState<CalendarSummary | null>(null);
  const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth());
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [selectedCompany, setSelectedCompany] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [viewMode, setViewMode] = useState<'list' | 'calendar'>('list');
  const [expandedStrip, setExpandedStrip] = useState<string | null>(null);
  const [showAllWatch, setShowAllWatch] = useState(false);

  const companies = Object.keys(COMPANY_COLORS);
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
  ];

  useEffect(() => {
    fetchCalendarData();
  }, [selectedYear]);

  const fetchCalendarData = async () => {
    setLoading(true);
    try {
      const [calendarRes, upcomingRes, summaryRes] = await Promise.all([
        fetch(`${API_URL}/api/calendar?year=${selectedYear}`),
        fetch(`${API_URL}/api/calendar/upcoming?days=60`),
        fetch(`${API_URL}/api/calendar/summary`),
      ]);
      if (calendarRes.ok) {
        const data = await calendarRes.json();
        setEvents(data.events || []);
      }
      if (upcomingRes.ok) {
        const data = await upcomingRes.json();
        setUpcoming(data.events || []);
      }
      if (summaryRes.ok) {
        setSummary(await summaryRes.json());
      }
    } catch (err) {
      console.error('Failed to fetch calendar:', err);
    } finally {
      setLoading(false);
    }
  };

  const syncCalendar = async () => {
    setSyncing(true);
    try {
      await fetch(`${API_URL}/api/calendar/sync`, { method: 'POST' });
      await fetchCalendarData();
    } catch (err) {
      console.error('Failed to sync:', err);
    } finally {
      setSyncing(false);
    }
  };

  const downloadIcal = () => {
    window.open(`${API_URL}/api/calendar/export/ical`, '_blank');
  };

  const getMonthEvents = () => {
    const monthStr = `${selectedYear}-${String(selectedMonth + 1).padStart(2, '0')}`;
    return events.filter((e) => {
      const matchesMonth = e.estimated_date.startsWith(monthStr);
      const matchesCompany = selectedCompany === 'all' || e.company === selectedCompany;
      return matchesMonth && matchesCompany;
    });
  };

  const getFilteredUpcoming = () => {
    if (selectedCompany === 'all') return upcoming;
    return upcoming.filter((e) => e.company === selectedCompany);
  };

  const prevMonth = () => {
    if (selectedMonth === 0) { setSelectedMonth(11); setSelectedYear(selectedYear - 1); }
    else setSelectedMonth(selectedMonth - 1);
  };

  const nextMonth = () => {
    if (selectedMonth === 11) { setSelectedMonth(0); setSelectedYear(selectedYear + 1); }
    else setSelectedMonth(selectedMonth + 1);
  };

  const getDaysInMonth = (year: number, month: number) => new Date(year, month + 1, 0).getDate();
  const getFirstDayOfMonth = (year: number, month: number) => new Date(year, month, 1).getDay();

  const renderCalendarGrid = () => {
    const daysInMonth = getDaysInMonth(selectedYear, selectedMonth);
    const firstDay = getFirstDayOfMonth(selectedYear, selectedMonth);
    const monthEvents = getMonthEvents();
    const days = [];
    for (let i = 0; i < firstDay; i++) {
      days.push(<div key={`empty-${i}`} className="h-24 bg-slate-50 dark:bg-slate-800/30" />);
    }
    for (let day = 1; day <= daysInMonth; day++) {
      const dateStr = `${selectedYear}-${String(selectedMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      const dayEvents = monthEvents.filter((e) => e.estimated_date === dateStr);
      const isToday = new Date().toISOString().split('T')[0] === dateStr;
      days.push(
        <div
          key={day}
          className={`h-24 border border-slate-100 dark:border-slate-700 p-1 ${isToday ? 'bg-blue-50 dark:bg-blue-900/20 ring-2 ring-blue-500' : 'bg-white dark:bg-slate-800/20'}`}
        >
          <div className={`text-sm font-medium ${isToday ? 'text-blue-600' : 'text-slate-600 dark:text-slate-400'}`}>{day}</div>
          <div className="mt-1 space-y-1 overflow-y-auto max-h-16">
            {dayEvents.map((event, idx) => (
              <div
                key={idx}
                className="text-xs p-1 rounded truncate"
                style={{
                  backgroundColor: COMPANY_COLORS[event.company] + '20',
                  borderLeft: `3px solid ${COMPANY_COLORS[event.company]}`,
                }}
                title={`${event.company} ${event.quarter} Earnings`}
              >
                {event.company} {event.quarter}
              </div>
            ))}
          </div>
        </div>
      );
    }
    return days;
  };

  // Group list-view events by month
  const groupedListEvents = () => {
    const filtered = (upcoming.length > 0 ? upcoming : WATCH_STRIP.map((s) => ({
      id: s.company,
      company: s.company,
      ticker: s.company.toUpperCase(),
      quarter: s.quarter.split(' ')[0],
      fiscal_year: parseInt(s.quarter.split('FY')[1]),
      estimated_date: s.date,
      time: 'TBD',
      event_type: 'earnings',
      confirmed: s.confirmed,
      status: s.confirmed ? 'Confirmed' : 'Estimated',
      days_until: s.days,
    }))).filter((e) => selectedCompany === 'all' || e.company === selectedCompany);

    const groups: Record<string, typeof filtered> = {};
    filtered.forEach((e) => {
      const d = new Date(e.estimated_date);
      const key = isNaN(d.getTime()) ? e.estimated_date : `${months[d.getMonth()]} ${d.getFullYear()}`;
      if (!groups[key]) groups[key] = [];
      groups[key].push(e);
    });
    return groups;
  };

  // Use static strip if API data unavailable
  const stripData = WATCH_STRIP;
  const watchListVisible = showAllWatch ? stripData : stripData.slice(0, 3);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-200 border-t-blue-600 mx-auto" />
          <p className="text-slate-600 dark:text-slate-400 mt-4 font-medium">Loading earnings intelligence...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 p-6">

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="bg-gradient-to-br from-indigo-600 to-purple-700 p-3 rounded-xl shadow-lg shadow-indigo-500/20">
            <CalendarDays className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold text-slate-900 dark:text-white">Earnings Intelligence</h1>
              <button
                onClick={downloadIcal}
                className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-medium"
              >
                Export iCal
              </button>
            </div>
            <p className="text-slate-500 dark:text-slate-400 mt-0.5 text-sm">
              Monitor competitor earnings windows and key signals to watch
            </p>
          </div>
        </div>
        <button
          onClick={syncCalendar}
          disabled={syncing}
          className="inline-flex items-center justify-center h-9 w-9 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors disabled:opacity-50"
          title="Sync calendar"
        >
          <RefreshCw className={`h-4 w-4 text-slate-500 ${syncing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Earnings Watch Strip */}
      <div className="mb-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
          Earnings Watch Strip
        </p>
        <div className="flex gap-3 overflow-x-auto pb-2">
          {stripData.map((item) => {
            const color = COMPANY_COLORS[item.company] || '#64748B';
            const isExpanded = expandedStrip === item.company;
            return (
              <div
                key={item.company}
                className="flex-shrink-0 w-52 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/60 overflow-hidden shadow-sm"
              >
                <div className="flex">
                  <div className="w-1 flex-shrink-0" style={{ backgroundColor: color }} />
                  <div className="flex-1 p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm text-slate-900 dark:text-white">{item.company}</span>
                      <button
                        onClick={() => setExpandedStrip(isExpanded ? null : item.company)}
                        className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                      >
                        {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                      </button>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400">{item.quarter}</p>
                    <p className="text-xs font-medium text-slate-700 dark:text-slate-300 mt-0.5">{item.date}</p>
                    <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                      <CountdownBadge days={item.days} />
                      <StatusTag confirmed={item.confirmed} />
                    </div>
                  </div>
                </div>
                {isExpanded && (
                  <div className="px-3 pb-3 border-t border-slate-100 dark:border-slate-700 pt-2">
                    <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5">
                      📋 Key signals to watch:
                    </p>
                    <ul className="space-y-1">
                      {(WATCH_LIST[item.company] || []).map((point, i) => (
                        <li key={i} className="text-xs text-slate-600 dark:text-slate-400 flex gap-1.5">
                          <span className="text-slate-300 dark:text-slate-600 mt-0.5">•</span>
                          <span>{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Two-column main layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT: Calendar / List view */}
        <div className="lg:col-span-2">
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader>
              <div className="flex items-center justify-between flex-wrap gap-3">
                {/* Month nav + company filter */}
                <div className="flex items-center gap-3 flex-wrap">
                  <div className="flex items-center gap-1">
                    <button onClick={prevMonth} className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors">
                      <ChevronLeft className="h-5 w-5 text-slate-600 dark:text-slate-400" />
                    </button>
                    <span className="font-semibold text-slate-900 dark:text-white min-w-[130px] text-center">
                      {months[selectedMonth]} {selectedYear}
                    </span>
                    <button onClick={nextMonth} className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-colors">
                      <ChevronRight className="h-5 w-5 text-slate-600 dark:text-slate-400" />
                    </button>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Filter className="h-4 w-4 text-slate-400" />
                    <select
                      value={selectedCompany}
                      onChange={(e) => setSelectedCompany(e.target.value)}
                      className="text-sm border border-slate-200 dark:border-slate-600 rounded-lg px-3 py-1.5 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="all">All Companies</option>
                      {companies.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                </div>

                {/* View toggle */}
                <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-700 rounded-lg p-1">
                  <button
                    onClick={() => setViewMode('list')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      viewMode === 'list'
                        ? 'bg-white dark:bg-slate-600 text-slate-900 dark:text-white shadow-sm'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                    }`}
                  >
                    <LayoutList className="h-3.5 w-3.5" />
                    List
                  </button>
                  <button
                    onClick={() => setViewMode('calendar')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      viewMode === 'calendar'
                        ? 'bg-white dark:bg-slate-600 text-slate-900 dark:text-white shadow-sm'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
                    }`}
                  >
                    <Calendar className="h-3.5 w-3.5" />
                    Calendar
                  </button>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              {viewMode === 'list' ? (
                /* LIST VIEW */
                <div className="space-y-6">
                  {Object.entries(groupedListEvents()).length > 0 ? (
                    Object.entries(groupedListEvents()).map(([monthLabel, evts]) => (
                      <div key={monthLabel}>
                        <p className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2 border-b border-slate-100 dark:border-slate-700 pb-1">
                          {monthLabel}
                        </p>
                        <div className="space-y-2">
                          {evts.map((event, idx) => {
                            const color = COMPANY_COLORS[event.company] || '#64748B';
                            return (
                              <div key={idx} className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                                <div className="h-3 w-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                                <span className="font-semibold text-sm text-slate-900 dark:text-white w-24 flex-shrink-0">{event.company}</span>
                                <span className="text-sm text-slate-500 dark:text-slate-400 flex-1">{event.quarter} FY{event.fiscal_year}</span>
                                <span className="text-sm text-slate-600 dark:text-slate-300">{event.estimated_date}</span>
                                {event.days_until !== undefined && (
                                  <CountdownBadge days={event.days_until} />
                                )}
                                <StatusTag confirmed={event.confirmed} />
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))
                  ) : (
                    /* Fallback: show static strip data as list */
                    (() => {
                      const marchItems = stripData.filter(s => s.date.includes('Mar'));
                      const aprilItems = stripData.filter(s => s.date.includes('Apr'));
                      const mayItems = stripData.filter(s => s.date.includes('May'));
                      const sections = [
                        { label: 'MARCH 2026', items: marchItems },
                        { label: 'APRIL 2026', items: aprilItems },
                        { label: 'MAY 2026', items: mayItems },
                      ].filter(s => s.items.length > 0);
                      return sections.map(({ label, items }) => (
                        <div key={label}>
                          <p className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2 border-b border-slate-100 dark:border-slate-700 pb-1">
                            {label}
                          </p>
                          <div className="space-y-2">
                            {items.map((item) => {
                              const color = COMPANY_COLORS[item.company] || '#64748B';
                              return (
                                <div key={item.company} className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                                  <div className="h-3 w-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                                  <span className="font-semibold text-sm text-slate-900 dark:text-white w-24 flex-shrink-0">{item.company}</span>
                                  <span className="text-sm text-slate-500 dark:text-slate-400 flex-1">{item.quarter}</span>
                                  <span className="text-sm text-slate-600 dark:text-slate-300">{item.date}</span>
                                  <CountdownBadge days={item.days} />
                                  <StatusTag confirmed={item.confirmed} />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ));
                    })()
                  )}
                </div>
              ) : (
                /* CALENDAR VIEW */
                <>
                  <div className="grid grid-cols-7 mb-2">
                    {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day) => (
                      <div key={day} className="text-center text-sm font-medium text-slate-500 dark:text-slate-400 py-2">
                        {day}
                      </div>
                    ))}
                  </div>
                  <div className="grid grid-cols-7 gap-px bg-slate-200 dark:bg-slate-700 rounded-lg overflow-hidden">
                    {renderCalendarGrid()}
                  </div>
                  <div className="flex flex-wrap gap-3 mt-4 pt-4 border-t border-slate-100 dark:border-slate-700">
                    {companies.map((company) => (
                      <div key={company} className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ backgroundColor: COMPANY_COLORS[company] }} />
                        <span className="text-xs text-slate-600 dark:text-slate-400">{company}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* RIGHT: Watch List + Post-Earnings Intel */}
        <div className="space-y-5">

          {/* Sub-section A: Upcoming Watch List */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Clock className="h-4 w-4 text-blue-500" />
                Upcoming — Watch List
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {watchListVisible.map((item) => {
                  const color = COMPANY_COLORS[item.company] || '#64748B';
                  const watchPoints = (WATCH_LIST[item.company] || []).slice(0, 3);
                  return (
                    <div
                      key={item.company}
                      className="pb-4 border-b border-slate-100 dark:border-slate-700 last:border-0 last:pb-0"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className="inline-flex items-center justify-center h-5 w-5 rounded-full text-[10px] font-bold text-white flex-shrink-0"
                          style={{ backgroundColor: color }}
                        >
                          {item.company[0]}
                        </span>
                        <span className="font-bold text-sm text-slate-900 dark:text-white">{item.company}</span>
                        <StatusTag confirmed={item.confirmed} />
                        <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">{item.date}</span>
                        <CountdownBadge days={item.days} />
                      </div>
                      <ul className="space-y-1 pl-7">
                        {watchPoints.map((pt, i) => (
                          <li key={i} className="text-xs text-slate-500 dark:text-slate-400 flex gap-1.5">
                            <span className="text-slate-300 dark:text-slate-600 mt-0.5">•</span>
                            <span>{pt}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </div>
              {!showAllWatch && stripData.length > 3 && (
                <button
                  onClick={() => setShowAllWatch(true)}
                  className="mt-3 text-xs text-blue-500 hover:text-blue-700 font-medium"
                >
                  View all ({stripData.length - 3} more) →
                </button>
              )}
              {showAllWatch && (
                <button
                  onClick={() => setShowAllWatch(false)}
                  className="mt-3 text-xs text-slate-400 hover:text-slate-600 font-medium"
                >
                  Show less ↑
                </button>
              )}
            </CardContent>
          </Card>

          {/* Sub-section B: Post-Earnings Intel */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">📊 Post-Earnings Intel</CardTitle>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                Last 90 days — key takeaways and Flex implications
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-5">
                {POST_EARNINGS.map((card) => {
                  const color = COMPANY_COLORS[card.company] || '#64748B';
                  return (
                    <div
                      key={card.company + card.quarter}
                      className="pb-5 border-b border-slate-100 dark:border-slate-700 last:border-0 last:pb-0"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className="inline-flex items-center justify-center h-5 w-5 rounded-full text-[10px] font-bold text-white flex-shrink-0"
                          style={{ backgroundColor: color }}
                        >
                          {card.company[0]}
                        </span>
                        <span className="font-bold text-sm text-slate-900 dark:text-white">{card.company}</span>
                        <span className="text-xs text-slate-400 dark:text-slate-500">{card.quarter}</span>
                        <span className="text-xs text-slate-400 dark:text-slate-500 ml-auto">{card.date}</span>
                      </div>
                      <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1 pl-7">
                        Key Takeaways:
                      </p>
                      <ul className="space-y-0.5 pl-7 mb-2">
                        {card.takeaways.map((t, i) => (
                          <li key={i} className="text-xs text-slate-500 dark:text-slate-400 flex gap-1.5">
                            <span className="text-slate-300 dark:text-slate-600 mt-0.5">•</span>
                            <span>{t}</span>
                          </li>
                        ))}
                      </ul>
                      <div className="pl-7">
                        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Flex Impact: </span>
                        <ImpactLine level={card.impactLevel} text={card.flexImpact} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

        </div>
      </div>
    </div>
  );
}
