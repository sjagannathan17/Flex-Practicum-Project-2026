'use client';

import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import {
  Building2,
  Target,
  Lightbulb,
  RefreshCw,
  ArrowUpRight,
  Cpu,
  CalendarDays,
  AlertTriangle,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// ── Interfaces ────────────────────────────────────────────────────────────────

interface CompetitorInvestment {
  company: string;
  investment_focus: string[];
  guidance_outlook: string;
  recent_highlights: string[];
  ai_growth_pct: number;
}

interface HyperscalerDemand {
  outlook: string;
  drivers: string[];
  beneficiaries: string[];
}

interface CompetitorData {
  as_of?: string;
  growth_definition?: string;
  growth_period?: string;
  competitors: CompetitorInvestment[];
  hyperscaler_demand: HyperscalerDemand;
}

interface CompanySentiment {
  company: string;
  documents_analyzed: number;
  sentiment_score: number;
  ai_mentions: number;
}

interface EarningsCalendarRow {
  company: string;
  q1: string;
  q2: string;
  q3: string;
  q4: string;
  fy: string;
}

// ── Static config ─────────────────────────────────────────────────────────────

const COMPANY_COLORS: Record<string, string> = {
  Flex: '#3B82F6',
  Jabil: '#10B981',
  Celestica: '#6366F1',
  Benchmark: '#F59E0B',
  Sanmina: '#EF4444',
};

const ESTIMATED_EARNINGS_2026: EarningsCalendarRow[] = [
  { company: 'Flex',      q1: '2026/06/28', q2: '2026/09/27', q3: '2026/12/31', q4: '2026/03/31', fy: '2027/03/31' },
  { company: 'Benchmark', q1: '2026/03/31', q2: '2026/06/30', q3: '2026/09/30', q4: '2026/12/31', fy: '2026/12/31' },
  { company: 'Jabil',     q1: '2026/11/30', q2: '2026/03/18', q3: '2026/05/31', q4: '2026/08/31', fy: '2026/08/31' },
  { company: 'Celestica', q1: '2026/03/31', q2: '2026/06/30', q3: '2026/09/30', q4: '2026/12/31', fy: '2026/12/31' },
  { company: 'Sanmina',   q1: '2026/12/27', q2: '2026/03/29', q3: '2026/06/28', q4: '2026/09/27', fy: '2026/09/27' },
];

type OutlookType = 'AI-Heavy' | 'Diversified' | 'Traditional';
type OutlookTrend = 'Accelerating' | 'Steady' | 'Stable' | 'Cautious';
type GrowthArrow = '↑↑' | '↑' | '→';
type ThreatLevel = 'HIGH' | 'MID' | 'LOW' | 'SELF';

interface CompanyMeta {
  outlookType: OutlookType;
  outlookTrend: OutlookTrend;
  growthArrow: GrowthArrow;
  threat: ThreatLevel;
  vsFlexNote?: string;
  guidanceOverride: string;
  focusOverride: string[];
  highlightsOverride: string[];
  sparkline: number[];
  aiMentions: number;
  vsLastQuarter: string;
  keyTopic: string;
}

const COMPANY_META: Record<string, CompanyMeta> = {
  Celestica: {
    outlookType: 'AI-Heavy', outlookTrend: 'Accelerating', growthArrow: '↑↑', threat: 'HIGH',
    vsFlexNote: 'Growing 23% faster than Flex in AI/DC segment, direct overlap risk',
    guidanceOverride: 'Very bullish — AI networking platform driving accelerated growth',
    focusOverride: ['AI networking', 'AMD partnership', 'Hyperscaler hardware'],
    highlightsOverride: [
      'Expanded AMD AI networking platform partnership',
      'Q4 2025 revenue beat driven by data center segment',
      'New hyperscaler qualifications for AI switching hardware',
    ],
    sparkline: [3, 4, 6, 8, 10], aiMentions: 10, vsLastQuarter: '+67%', keyTopic: 'AI networking, AMD partnership',
  },
  Jabil: {
    outlookType: 'AI-Heavy', outlookTrend: 'Accelerating', growthArrow: '↑↑', threat: 'HIGH',
    vsFlexNote: "AI server revenue growing 32% YoY vs Flex's 15% in same segment",
    guidanceOverride: 'Strong — AI server demand accelerating beyond initial guidance',
    focusOverride: ['AI server assembly', 'Hyperscaler supply', 'Compute hardware'],
    highlightsOverride: [
      'AI-related revenue +32% YoY in latest quarter',
      'Added 2 new hyperscaler customers in AI server segment',
      'Expanding capacity in existing US and Mexico sites',
    ],
    sparkline: [2, 2, 3, 4, 4], aiMentions: 4, vsLastQuarter: '+33%', keyTopic: 'AI server assembly, hyperscaler',
  },
  Flex: {
    outlookType: 'Diversified', outlookTrend: 'Steady', growthArrow: '↑', threat: 'SELF',
    guidanceOverride: 'Positive — expecting AI/DC to reach 35% of revenue by FY25',
    focusOverride: ['Liquid cooling', 'Power modules', 'AI server assembly'],
    highlightsOverride: [
      'Expanded liquid cooling production capacity in Mexico',
      'New AI server assembly line in Malaysia operational',
      'Won major hyperscaler contract for GPU server racks',
    ],
    sparkline: [2, 3, 4, 4, 4], aiMentions: 4, vsLastQuarter: '+0%', keyTopic: 'Liquid cooling, power modules',
  },
  Sanmina: {
    outlookType: 'Diversified', outlookTrend: 'Stable', growthArrow: '↑', threat: 'MID',
    vsFlexNote: 'Similar diversification profile but stronger in server networking',
    guidanceOverride: 'Stable — server ecosystem wins offsetting traditional segment softness',
    focusOverride: ['Server ecosystem', 'Networking hardware', 'Vertical integration'],
    highlightsOverride: [
      '3 new hyperscaler qualifications in Q1 2026',
      'Vertical integration strategy showing margin improvement',
      'Communications segment stable with new contract wins',
    ],
    sparkline: [1, 1, 2, 2, 2], aiMentions: 2, vsLastQuarter: '+0%', keyTopic: 'Server ecosystem, networking',
  },
  Benchmark: {
    outlookType: 'Traditional', outlookTrend: 'Cautious', growthArrow: '→', threat: 'LOW',
    vsFlexNote: 'Lower AI exposure than Flex, less competitive threat in AI/DC',
    guidanceOverride: 'Cautious — maintaining traditional segment focus, selective AI exposure',
    focusOverride: ['HPC computing', 'Medical/defense', 'Precision manufacturing'],
    highlightsOverride: [
      'HPC computing segment showing early AI-adjacent growth',
      'Medical and aerospace segments providing stable revenue base',
      'Selective investment in high-performance computing capabilities',
    ],
    sparkline: [2, 2, 2, 2, 3], aiMentions: 3, vsLastQuarter: '+50%', keyTopic: 'HPC, medical/defense balance',
  },
};

// Sparklines displayed in the order below
const SIGNAL_ORDER = ['Celestica', 'Flex', 'Jabil', 'Sanmina', 'Benchmark'];

// ── Sub-components ────────────────────────────────────────────────────────────

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const W = 80;
  const H = 24;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * W},${H - 4 - ((v - min) / range) * (H - 8)}`)
    .join(' ');
  return (
    <svg width={W} height={H} style={{ overflow: 'visible', display: 'block' }}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      {data.map((v, i) => (
        <circle
          key={i}
          cx={(i / (data.length - 1)) * W}
          cy={H - 4 - ((v - min) / range) * (H - 8)}
          r="2.5"
          fill={color}
        />
      ))}
    </svg>
  );
}

function ThreatBadge({ level }: { level: ThreatLevel }) {
  if (level === 'SELF') return <span className="text-slate-400 text-sm font-medium">—</span>;
  const styles: Record<Exclude<ThreatLevel, 'SELF'>, string> = {
    HIGH: 'bg-red-100 text-red-600 border border-red-300',
    MID:  'bg-orange-100 text-orange-600 border border-orange-300',
    LOW:  'bg-green-100 text-green-600 border border-green-300',
  };
  const labels: Record<Exclude<ThreatLevel, 'SELF'>, string> = {
    HIGH: 'HIGH',
    MID:  'MEDIUM',
    LOW:  'LOW',
  };
  return (
    <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold whitespace-nowrap ${styles[level]}`}>
      {labels[level]}
    </span>
  );
}

function OutlookTag({ type, trend }: { type: OutlookType; trend: OutlookTrend }) {
  const typeStyle: Record<OutlookType, string> = {
    'AI-Heavy':    'bg-violet-100 text-violet-700 dark:bg-violet-500/20 dark:text-violet-300',
    'Diversified': 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300',
    'Traditional': 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
  };
  const trendStyle: Record<OutlookTrend, string> = {
    'Accelerating': 'bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-300',
    'Steady':       'bg-sky-100 text-sky-700 dark:bg-sky-500/20 dark:text-sky-300',
    'Stable':       'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300',
    'Cautious':     'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300',
  };
  const trendArrow: Record<OutlookTrend, string> = {
    'Accelerating': '↑',
    'Steady':       '→',
    'Stable':       '→',
    'Cautious':     '↓',
  };
  return (
    <div className="flex flex-col gap-0.5">
      <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold leading-tight ${typeStyle[type]}`}>{type}</span>
      <span className={`rounded px-1.5 py-0.5 text-[10px] font-semibold leading-tight ${trendStyle[trend]}`}>
        {trendArrow[trend]} {trend}
      </span>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function CompetitorInvestmentsPage() {
  const [data, setData] = useState<CompetitorData | null>(null);
  const [sentiment, setSentiment] = useState<CompanySentiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedCompany, setSelectedCompany] = useState<CompetitorInvestment | null>(null);
  const [earningsView, setEarningsView] = useState<'next' | 'full'>('next');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async (forceRefresh = false) => {
    try {
      setLoading(true);
      const forceParam = forceRefresh ? '?force_refresh=true' : '';
      const [competitorRes, sentimentRes] = await Promise.all([
        fetch(`${API_URL}/api/intelligence/competitor-investments${forceParam}`),
        fetch(`${API_URL}/api/sentiment/compare`),
      ]);

      if (competitorRes.ok) {
        const json = await competitorRes.json();
        setData(json);
        // Default: select Flex
        const flex = json.competitors?.find((c: CompetitorInvestment) => c.company === 'Flex');
        setSelectedCompany(flex || json.competitors?.[0] || null);
      }

      if (sentimentRes.ok) {
        const sentimentJson = await sentimentRes.json();
        setSentiment(sentimentJson.comparison || []);
      }
    } catch (err) {
      console.error('Failed to fetch competitor data:', err);
    } finally {
      setLoading(false);
    }
  };

  // ── Earnings calendar helpers ──────────────────────────────────────────────

  const parseCalendarDate = (value: string): Date | null => {
    if (!value || value === '—') return null;
    const parsed = new Date(value.replace(/\//g, '-') + 'T00:00:00');
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  };

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const nextReleaseRows = useMemo(() => {
    return ESTIMATED_EARNINGS_2026.map((row) => {
      const rawEvents = [
        { label: 'Q1', date: row.q1 },
        { label: 'Q2', date: row.q2 },
        { label: 'Q3', date: row.q3 },
        { label: 'Q4', date: row.q4 },
        { label: 'FY', date: row.fy },
      ]
        .map((item) => ({ ...item, parsed: parseCalendarDate(item.date) }))
        .filter((item) => item.parsed !== null) as Array<{ label: string; date: string; parsed: Date }>;

      rawEvents.sort((a, b) => a.parsed.getTime() - b.parsed.getTime());

      const dedupByDate = new Map<string, { date: string; parsed: Date; labels: string[] }>();
      for (const event of rawEvents) {
        if (!dedupByDate.has(event.date)) {
          dedupByDate.set(event.date, { date: event.date, parsed: event.parsed, labels: [event.label] });
        } else {
          dedupByDate.get(event.date)!.labels.push(event.label);
        }
      }
      const mergedEvents = Array.from(dedupByDate.values()).sort((a, b) => a.parsed.getTime() - b.parsed.getTime());
      const nextEvent = mergedEvents.find((e) => e.parsed.getTime() >= today.getTime()) || mergedEvents[0];
      const daysLeft = nextEvent
        ? Math.ceil((nextEvent.parsed.getTime() - today.getTime()) / (1000 * 60 * 60 * 24))
        : null;

      return {
        company: row.company,
        nextDate: nextEvent?.date || '—',
        nextLabel: nextEvent ? nextEvent.labels.join(' + ') : '—',
        daysLeft,
      };
    }).sort((a, b) => {
      if (a.nextDate === '—' && b.nextDate === '—') return 0;
      if (a.nextDate === '—') return 1;
      if (b.nextDate === '—') return -1;
      const ad = parseCalendarDate(a.nextDate)?.getTime() || Number.MAX_SAFE_INTEGER;
      const bd = parseCalendarDate(b.nextDate)?.getTime() || Number.MAX_SAFE_INTEGER;
      return ad - bd;
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const nextEarnings = useMemo(
    () => nextReleaseRows.find((r) => r.daysLeft !== null && r.daysLeft > 0) || null,
    [nextReleaseRows],
  );

  // ── Loading / error states ─────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 flex items-center justify-center dark:from-slate-950 dark:to-slate-950">
        <div className="text-slate-500">Loading competitor intelligence data...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 flex items-center justify-center dark:from-slate-950 dark:to-slate-950">
        <div className="text-red-500">Failed to load data</div>
      </div>
    );
  }

  // Enrich each competitor with static meta
  const enriched = data.competitors.map((c) => ({
    ...c,
    meta: COMPANY_META[c.company] || null,
  }));

  const sel = selectedCompany;
  const selMeta = sel ? COMPANY_META[sel.company] : null;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-4 dark:from-slate-950 dark:via-slate-950 dark:to-slate-950">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-3">
              <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-1.5 rounded-xl">
                <Target className="h-5 w-5 text-white" />
              </div>
              Competitive Intelligence
            </h1>
            <p className="text-slate-500 mt-0.5 text-sm dark:text-slate-400">
              AI/DC momentum, threat levels, and strategic moves across EMS peers
            </p>
          </div>
          <button
            onClick={() => fetchData(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-white rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-all shadow-sm dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            <RefreshCw className="h-4 w-4" />
            Re-analyze
          </button>
        </div>
      </div>

      <Card className="border-0 shadow-xl dark:bg-slate-900">
        <CardContent className="p-4 lg:p-5">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.45fr_1fr] xl:h-[calc(100vh-165px)]">

            {/* ── LEFT PANEL ──────────────────────────────────────────────── */}
            <section className="min-h-0 flex flex-col gap-4 overflow-y-auto xl:[&::-webkit-scrollbar]:w-1.5 xl:[&::-webkit-scrollbar-thumb]:rounded-full xl:[&::-webkit-scrollbar-thumb]:bg-slate-300 dark:xl:[&::-webkit-scrollbar-thumb]:bg-slate-700 xl:[&::-webkit-scrollbar-track]:bg-transparent">

              {/* Section 1: Competitive Snapshot Table */}
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900/50">
                <h2 className="mb-1 flex items-center gap-2 text-base font-semibold text-slate-900 dark:text-slate-100">
                  <Cpu className="h-4 w-4 text-purple-600" />
                  Competitive Snapshot
                </h2>
                <p className="mb-3 text-[11px] text-slate-500 dark:text-slate-400">
                  Growth metric: {data.growth_definition || 'Composite'}
                  {data.growth_period ? ` · period ${data.growth_period}` : ''}
                  {data.as_of ? ` · as of ${data.as_of}` : ''}
                </p>

                {/* Table header */}
                <div className="grid grid-cols-[1.4fr_0.65fr_1.05fr_0.85fr] gap-2 border-b border-slate-200 pb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500 dark:border-slate-700 dark:text-slate-400 px-3">
                  <span>Company</span>
                  <span>Growth</span>
                  <span>Outlook</span>
                  <span className="text-right">Threat to Flex</span>
                </div>

                {/* Table rows */}
                <div className="mt-2 space-y-2">
                  {enriched.map(({ company, ai_growth_pct, meta }) => {
                    const isSelected = selectedCompany?.company === company;
                    const arrow = meta?.growthArrow || '↑';
                    const arrowColor = arrow === '↑↑' ? 'text-green-600' : arrow === '↑' ? 'text-emerald-500' : 'text-slate-400';
                    const companyData = data.competitors.find(c => c.company === company)!;
                    return (
                      <button
                        key={company}
                        type="button"
                        onClick={() => setSelectedCompany(companyData)}
                        className={`grid w-full grid-cols-[1.4fr_0.65fr_1.05fr_0.85fr] items-center gap-2 rounded-xl border px-3 py-2.5 text-left transition ${
                          isSelected
                            ? 'border-l-4 border-l-indigo-500 border-t-indigo-200 border-r-indigo-200 border-b-indigo-200 bg-indigo-50 shadow-md dark:bg-indigo-950/40 dark:border-l-indigo-400 dark:border-t-indigo-800 dark:border-r-indigo-800 dark:border-b-indigo-800'
                            : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900/60 dark:hover:border-slate-600 dark:hover:bg-slate-900'
                        }`}
                      >
                        {/* Company */}
                        <div className="flex items-center gap-2 min-w-0">
                          <div
                            className="h-7 w-7 rounded-lg flex items-center justify-center text-xs font-bold text-white shrink-0"
                            style={{ backgroundColor: COMPANY_COLORS[company] }}
                          >
                            {company.charAt(0)}
                          </div>
                          <p className="truncate font-semibold text-slate-900 dark:text-slate-100">{company}</p>
                        </div>

                        {/* Growth + arrow */}
                        <div>
                          <p className="font-semibold text-slate-900 dark:text-slate-100">
                            +{ai_growth_pct}%{' '}
                            <span className={`font-bold ${arrowColor}`}>{arrow}</span>
                          </p>
                          <p className="text-[10px] text-slate-500 dark:text-slate-400">Composite</p>
                        </div>

                        {/* Two-part outlook tag */}
                        {meta ? (
                          <OutlookTag type={meta.outlookType} trend={meta.outlookTrend} />
                        ) : (
                          <span className="text-xs text-slate-500">—</span>
                        )}

                        {/* Threat badge */}
                        <div className="flex justify-end">
                          {meta ? <ThreatBadge level={meta.threat} /> : <span className="text-slate-400">—</span>}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Section 2: AI/DC Momentum Signals */}
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900/50">
                <h2 className="mb-0.5 flex items-center gap-2 text-base font-semibold text-slate-900 dark:text-slate-100">
                  <Cpu className="h-4 w-4 text-orange-500" />
                  AI/DC Momentum Signals
                </h2>
                <p className="mb-3 text-[11px] text-slate-500 dark:text-slate-400">
                  Frequency of AI and data center topics in earnings calls — last 5 quarters
                </p>

                <div className="rounded-xl border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900 overflow-hidden">
                  {/* Table header */}
                  <div className="grid grid-cols-[1.1fr_88px_0.7fr_0.65fr_1.4fr] gap-2 bg-slate-50 dark:bg-slate-800 px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700">
                    <span>Company</span>
                    <span>Trend</span>
                    <span>Mentions</span>
                    <span>vs Last Qtr</span>
                    <span>Key Topic</span>
                  </div>
                  <div className="divide-y divide-slate-100 dark:divide-slate-800">
                    {SIGNAL_ORDER.map((name) => {
                      const meta = COMPANY_META[name];
                      if (!meta) return null;
                      const color = COMPANY_COLORS[name] || '#64748B';
                      const isUp = meta.vsLastQuarter.startsWith('+') && meta.vsLastQuarter !== '+0%';
                      const isFlat = meta.vsLastQuarter === '+0%';
                      const trendColor = isUp ? 'text-green-600' : isFlat ? 'text-slate-500' : 'text-red-500';
                      const trendArrow = isUp ? '↑' : isFlat ? '→' : '↓';
                      return (
                        <div key={name} className="grid grid-cols-[1.1fr_88px_0.7fr_0.65fr_1.4fr] gap-2 items-center px-3 py-2.5">
                          <div className="flex items-center gap-2">
                            <div className="h-5 w-5 rounded flex items-center justify-center text-[9px] font-bold text-white shrink-0" style={{ backgroundColor: color }}>
                              {name.charAt(0)}
                            </div>
                            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">{name}</span>
                          </div>
                          <Sparkline data={meta.sparkline} color={color} />
                          <span className="text-xs font-bold text-slate-900 dark:text-slate-100">{meta.aiMentions}</span>
                          <span className={`text-xs font-semibold ${trendColor}`}>
                            {meta.vsLastQuarter} {trendArrow}
                          </span>
                          <span className="text-[10px] text-slate-500 dark:text-slate-400 truncate">{meta.keyTopic}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Callout */}
                <div className="mt-3 flex items-start gap-2 rounded-xl border border-orange-200 bg-orange-50 px-3 py-2.5 dark:border-orange-700/40 dark:bg-orange-500/10">
                  <AlertTriangle className="h-4 w-4 text-orange-500 shrink-0 mt-0.5" />
                  <p className="text-[11px] text-orange-800 dark:text-orange-300 leading-relaxed">
                    <span className="font-semibold">Celestica&rsquo;s AI/DC mentions grew 67% last quarter</span> — fastest acceleration among EMS peers
                  </p>
                </div>
              </div>
            </section>

            {/* ── RIGHT PANEL ─────────────────────────────────────────────── */}
            {sel && (
              <section className="min-h-0 rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900/50 overflow-y-auto xl:[&::-webkit-scrollbar]:w-1.5 xl:[&::-webkit-scrollbar-thumb]:rounded-full xl:[&::-webkit-scrollbar-thumb]:bg-slate-300 dark:xl:[&::-webkit-scrollbar-thumb]:bg-slate-700 xl:[&::-webkit-scrollbar-track]:bg-transparent">
                <h2 className="mb-3 flex items-center gap-2 text-base font-semibold text-slate-900 dark:text-slate-100">
                  <Building2 className="h-4 w-4 text-blue-600" />
                  <div
                    className="h-6 w-6 rounded-md flex items-center justify-center text-xs font-bold text-white"
                    style={{ backgroundColor: COMPANY_COLORS[sel.company] }}
                  >
                    {sel.company.charAt(0)}
                  </div>
                  {sel.company} Strategy Detail
                  {selMeta && selMeta.threat !== 'SELF' && (
                    <ThreatBadge level={selMeta.threat} />
                  )}
                </h2>

                <div className="space-y-3">
                  {/* Guidance Outlook */}
                  <div className="rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
                    <h4 className="mb-1 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
                      <Lightbulb className="h-4 w-4 text-amber-500" />
                      Guidance Outlook
                    </h4>
                    <p className="text-sm text-slate-700 dark:text-slate-200">
                      {selMeta?.guidanceOverride || sel.guidance_outlook}
                    </p>
                    {selMeta?.vsFlexNote && (
                      <div className="mt-2 rounded-lg bg-indigo-50 dark:bg-indigo-900/30 px-2.5 py-1.5">
                        <p className="text-xs text-indigo-700 dark:text-indigo-300">
                          <span className="font-semibold">vs Flex:</span> {selMeta.vsFlexNote}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Focus Areas */}
                  <div className="rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
                    <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
                      <Target className="h-4 w-4 text-indigo-500" />
                      Focus Areas
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {(selMeta?.focusOverride || sel.investment_focus).map((focus, idx) => (
                        <span key={idx} className="rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200">
                          {focus}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Recent Highlights */}
                  <div className="rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
                    <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
                      <ArrowUpRight className="h-4 w-4 text-green-500" />
                      Recent Highlights
                    </h4>
                    <ul className="space-y-1.5 text-sm text-slate-700 dark:text-slate-200">
                      {(selMeta?.highlightsOverride || sel.recent_highlights).map((highlight, idx) => (
                        <li key={idx} className="rounded-lg bg-green-50 px-2.5 py-2 dark:bg-emerald-900/35 dark:text-emerald-100">
                          • {highlight}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Earnings Calendar */}
                  <div className="rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-900">
                    <div className="mb-2 flex items-center justify-between">
                      <h4 className="flex items-center gap-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
                        <CalendarDays className="h-4 w-4 text-indigo-500" />
                        Earnings Calendar
                      </h4>
                      <span className="rounded-full bg-orange-100 text-orange-700 dark:bg-orange-500/20 dark:text-orange-300 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wide">
                        2026 Projected
                      </span>
                    </div>

                    {/* Next earnings countdown banner */}
                    {nextEarnings && (
                      <div className="mb-2 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700/40 px-2.5 py-1.5">
                        <p className="text-[11px] text-blue-700 dark:text-blue-300">
                          <span className="font-semibold">Next earnings:</span>{' '}
                          {nextEarnings.company} in{' '}
                          <span className="font-bold">{nextEarnings.daysLeft} day{nextEarnings.daysLeft !== 1 ? 's' : ''}</span>{' '}
                          ({nextEarnings.nextDate})
                        </p>
                      </div>
                    )}

                    {/* Toggle */}
                    <div className="mb-2 inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1 dark:border-slate-700 dark:bg-slate-800">
                      {(['next', 'full'] as const).map((v) => (
                        <button
                          key={v}
                          type="button"
                          onClick={() => setEarningsView(v)}
                          className={`rounded-md px-2 py-1 text-[11px] font-semibold transition ${
                            earningsView === v
                              ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-100'
                              : 'text-slate-500 dark:text-slate-300'
                          }`}
                        >
                          {v === 'next' ? 'Next Releases' : 'Full Schedule'}
                        </button>
                      ))}
                    </div>

                    {earningsView === 'next' ? (
                      <div className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
                        <div className="grid grid-cols-[1.05fr_0.95fr_0.9fr_0.7fr] gap-1 bg-slate-50 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-300">
                          <span>Company</span><span>Next Date</span><span>Next Event</span><span>Days</span>
                        </div>
                        <div className="divide-y divide-slate-200 dark:divide-slate-700">
                          {nextReleaseRows.map((row) => (
                            <div key={row.company} className="grid grid-cols-[1.05fr_0.95fr_0.9fr_0.7fr] gap-1 px-2 py-1.5 text-[11px] text-slate-700 dark:text-slate-200">
                              <span className="font-semibold">{row.company}</span>
                              <span>{row.nextDate}</span>
                              <span className="font-medium">{row.nextLabel}</span>
                              <span>{row.daysLeft === null ? '—' : row.daysLeft <= 0 ? 'Today' : `${row.daysLeft}d`}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
                        <div className="grid grid-cols-[1.15fr_0.8fr_0.8fr_0.8fr_0.8fr_0.8fr] gap-1 bg-slate-50 px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-300">
                          <span>Company</span><span>Q1</span><span>Q2</span><span>Q3</span><span>Q4</span><span>FY</span>
                        </div>
                        <div className="divide-y divide-slate-200 dark:divide-slate-700">
                          {ESTIMATED_EARNINGS_2026.map((row) => (
                            <div key={row.company} className="grid grid-cols-[1.15fr_0.8fr_0.8fr_0.8fr_0.8fr_0.8fr] gap-1 px-2 py-1.5 text-[11px] text-slate-700 dark:text-slate-200">
                              <span className="font-semibold">{row.company}</span>
                              <span>{row.q1}</span><span>{row.q2}</span><span>{row.q3}</span><span>{row.q4}</span>
                              <span className="font-medium">{row.fy}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    <p className="mt-2 text-[10px] text-slate-500 dark:text-slate-400">
                      Projected from the historical quarter-end pattern you provided.
                    </p>
                  </div>
                </div>
              </section>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
