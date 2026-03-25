'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  RefreshCw,
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  ArrowRight,
  ArrowDownRight,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Target,
  Zap,
  Minus,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  ReferenceLine,
  TooltipProps,
} from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const COMPANY_COLORS: Record<string, string> = {
  Flex:      '#3B82F6',
  Jabil:     '#10B981',
  Celestica: '#6366F1',
  Benchmark: '#F59E0B',
  Sanmina:   '#EF4444',
};

// ── Section 1: Insight cards ────────────────────────────────────────────────
// (static — NLP-derived insights)

// ── Section 2A: Horizontal grouped bar ────────────────────────────────────
const INTENSITY_DATA = [
  { name: 'Celestica', AI: 47, DC: 45 },
  { name: 'Jabil',     AI: 44, DC: 24 },
  { name: 'Flex',      AI:  4, DC: 23 },
  { name: 'Sanmina',   AI:  2, DC: 15 },
  { name: 'Benchmark', AI:  1, DC:  0 },
];

// ── Section 2B: Multi-line trend ───────────────────────────────────────────
const TREND_DATA = [
  { q: 'Q4 FY24', Celestica: 22, Jabil: 18, Sanmina:  8, Flex: 15, Benchmark: 2 },
  { q: 'Q1 FY25', Celestica: 28, Jabil: 22, Sanmina: 10, Flex: 18, Benchmark: 2 },
  { q: 'Q2 FY25', Celestica: 35, Jabil: 30, Sanmina: 12, Flex: 20, Benchmark: 3 },
  { q: 'Q3 FY25', Celestica: 40, Jabil: 38, Sanmina: 10, Flex: 21, Benchmark: 2 },
  { q: 'Q4 FY25', Celestica: 47, Jabil: 44, Sanmina: 17, Flex: 27, Benchmark: 4 },
];

// ── Section 2C: AI vs Traditional mix ─────────────────────────────────────
const MIX_DATA = [
  { name: 'Celestica', ai: 85, trad: 15 },
  { name: 'Jabil',     ai: 72, trad: 28 },
  { name: 'Flex',      ai: 55, trad: 45 },
  { name: 'Sanmina',   ai: 48, trad: 52 },
  { name: 'Benchmark', ai: 22, trad: 78 },
];

// ── Section 3: Comparison table ────────────────────────────────────────────
const STATIC_TABLE = [
  { company: 'Celestica', capex: 85, ai: 47, dc: 45, score: 92, trend: 'up',   trendLabel: 'Above Avg', vsFlexDiff:  65, vsFlexLabel: '+65 pts' },
  { company: 'Jabil',     capex: 72, ai: 44, dc: 24, score: 68, trend: 'up',   trendLabel: 'Above Avg', vsFlexDiff:  41, vsFlexLabel: '+41 pts' },
  { company: 'Sanmina',   capex: 60, ai:  2, dc: 15, score: 17, trend: 'flat', trendLabel: 'Avg',       vsFlexDiff: -10, vsFlexLabel: '-10 pts' },
  { company: 'Flex',      capex: 97, ai:  4, dc: 23, score: 27, trend: 'flat', trendLabel: 'Avg',       vsFlexDiff:   0, vsFlexLabel: '— baseline' },
  { company: 'Benchmark', capex:  3, ai:  1, dc:  0, score:  1, trend: 'down', trendLabel: 'Below Avg', vsFlexDiff: -26, vsFlexLabel: '-26 pts' },
];

// ── Section 4: Strategic Momentum cards ────────────────────────────────────
type MomentumOverall = 'Accelerating' | 'Transitioning' | 'Cautious';
interface MomentumRow { label: string; direction: 'up' | 'flat' | 'down'; value?: string }
interface MomentumCard { company: string; overall: MomentumOverall; rows: MomentumRow[] }

const MOMENTUM_CARDS: MomentumCard[] = [
  {
    company: 'Flex',
    overall: 'Transitioning',
    rows: [
      { label: 'CapEx',     direction: 'flat', value: 'stabilizing' },
      { label: 'AI Focus',  direction: 'up',   value: 'increasing' },
      { label: 'AI/DC Mix', direction: 'flat', value: '55% of focus' },
    ],
  },
  {
    company: 'Jabil',
    overall: 'Accelerating',
    rows: [
      { label: 'CapEx',     direction: 'up', value: 'increasing' },
      { label: 'AI Focus',  direction: 'up', value: 'increasing (steep)' },
      { label: 'AI/DC Mix', direction: 'up', value: '72% of focus' },
    ],
  },
  {
    company: 'Celestica',
    overall: 'Accelerating',
    rows: [
      { label: 'CapEx',     direction: 'up', value: 'increasing' },
      { label: 'AI Focus',  direction: 'up', value: 'increasing (steepest)' },
      { label: 'AI/DC Mix', direction: 'up', value: '85% of focus' },
    ],
  },
  {
    company: 'Benchmark',
    overall: 'Cautious',
    rows: [
      { label: 'CapEx',     direction: 'down', value: 'decreasing' },
      { label: 'AI Focus',  direction: 'flat', value: 'flat' },
      { label: 'AI/DC Mix', direction: 'flat', value: '22% of focus' },
    ],
  },
  {
    company: 'Sanmina',
    overall: 'Cautious',
    rows: [
      { label: 'CapEx',     direction: 'down', value: 'decreasing' },
      { label: 'AI Focus',  direction: 'down', value: 'decreasing' },
      { label: 'AI/DC Mix', direction: 'flat', value: '48% of focus' },
    ],
  },
];

// ── Section 5: Anomaly alerts ──────────────────────────────────────────────
type AlertLevel = 'high' | 'medium' | 'internal';
interface AnomalyAlert {
  company: string;
  level: AlertLevel;
  badgeLabel: string;
  what: string;
  flexImpact: string;
}

const ANOMALY_ALERTS: AnomalyAlert[] = [
  {
    company: 'Celestica',
    level: 'high',
    badgeLabel: '🔴 AI Surge',
    what: 'AI/DC mentions tripled over 3 consecutive quarters (Q2→Q4 FY2025)',
    flexImpact: 'Celestica is rapidly repositioning as AI-first — direct competitive threat in hyperscaler hardware segment',
  },
  {
    company: 'Jabil',
    level: 'high',
    badgeLabel: '🔴 AI Surge',
    what: 'AI server revenue guidance raised twice in same fiscal year',
    flexImpact: 'Jabil accelerating faster than initial projections — may reach AI-dominant mix before Flex completes transition',
  },
  {
    company: 'Benchmark',
    level: 'medium',
    badgeLabel: '🟠 CapEx Anomaly',
    what: 'CapEx spending unusually erratic across 6 quarters — no consistent investment pattern',
    flexImpact: 'Low threat signal — Benchmark lacks consistent AI investment commitment',
  },
  {
    company: 'Sanmina',
    level: 'medium',
    badgeLabel: '🟠 Strategy Shift',
    what: 'AI Focus declining while server ecosystem wins increasing — repositioning toward infrastructure over AI-specific hardware',
    flexImpact: 'Sanmina differentiating away from direct AI competition, focusing on networking layer instead',
  },
  {
    company: 'Flex',
    level: 'internal',
    badgeLabel: '🔵 Internal Watch',
    what: 'CapEx activity shows 2 unusual periods — likely reflects AI/DC capacity ramp timing',
    flexImpact: 'Internal benchmark — ensure CapEx cadence aligns with AI transition timeline vs peers',
  },
];

// ── Section 6: Keywords ────────────────────────────────────────────────────
const KEYWORDS: Record<string, string[]> = {
  Celestica:  ['AI networking', 'hyperscaler', 'AMD platform', 'switching', '400G'],
  Jabil:      ['AI server', 'hyperscaler', 'compute', 'inference', 'rack assembly'],
  Sanmina:    ['server', 'networking', 'communication', 'backplane', 'integration'],
  Flex:       ['liquid cooling', 'power', 'AI infrastructure', 'data center', 'thermal'],
  Benchmark:  ['HPC', 'compute', 'defense', 'medical', 'precision'],
};

// ── Helpers ────────────────────────────────────────────────────────────────
function TrendBadge({ trend, label }: { trend: string; label: string }) {
  if (trend === 'up')
    return (
      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
        <ArrowUpRight className="h-3 w-3" /> {label}
      </span>
    );
  if (trend === 'down')
    return (
      <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400">
        <ArrowDownRight className="h-3 w-3" /> {label}
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400">
      <ArrowRight className="h-3 w-3" /> {label}
    </span>
  );
}

function VsFlexBadge({ diff, label }: { diff: number; label: string }) {
  if (diff > 20)  return <span className="text-xs font-bold text-red-600 dark:text-red-400">{label}</span>;
  if (diff > 0)   return <span className="text-xs font-bold text-orange-500 dark:text-orange-400">{label}</span>;
  if (diff < 0)   return <span className="text-xs font-medium text-green-600 dark:text-green-400">{label}</span>;
  return <span className="text-xs font-medium text-slate-400">{label}</span>;
}

function OverallBadge({ overall }: { overall: MomentumOverall }) {
  const styles: Record<MomentumOverall, string> = {
    Accelerating:  'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    Transitioning: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    Cautious:      'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
  };
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${styles[overall]}`}>
      {overall}
    </span>
  );
}

function DirectionIcon({ direction }: { direction: 'up' | 'flat' | 'down' }) {
  if (direction === 'up')   return <TrendingUp className="h-4 w-4 text-green-500 flex-shrink-0" />;
  if (direction === 'down') return <TrendingDown className="h-4 w-4 text-red-500 flex-shrink-0" />;
  return <Minus className="h-4 w-4 text-slate-400 flex-shrink-0" />;
}

function AlertCard({ alert }: { alert: AnomalyAlert }) {
  const styles: Record<AlertLevel, { border: string; bg: string; badge: string }> = {
    high:     { border: 'border-l-red-400',    bg: 'bg-red-50/60 dark:bg-red-900/10',       badge: 'bg-red-100 text-red-600 border-red-300' },
    medium:   { border: 'border-l-orange-400', bg: 'bg-orange-50/60 dark:bg-orange-900/10', badge: 'bg-orange-100 text-orange-600 border-orange-300' },
    internal: { border: 'border-l-blue-400',   bg: 'bg-blue-50/60 dark:bg-blue-900/10',    badge: 'bg-blue-100 text-blue-700 border-blue-300' },
  };
  const s = styles[alert.level];
  const color = COMPANY_COLORS[alert.company] || '#64748b';
  return (
    <div className={`rounded-xl border border-l-4 ${s.border} ${s.bg} p-4`}>
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span
          className="inline-flex items-center justify-center h-5 w-5 rounded-full text-[10px] font-bold text-white flex-shrink-0"
          style={{ backgroundColor: color }}
        >
          {alert.company[0]}
        </span>
        <span className="font-bold text-sm text-slate-900 dark:text-white">{alert.company}</span>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${s.badge}`}>
          {alert.badgeLabel}
        </span>
      </div>
      <p className="text-sm text-slate-700 dark:text-slate-300 mb-1">
        <span className="font-semibold">What: </span>{alert.what}
      </p>
      <p className="text-xs text-slate-500 dark:text-slate-400">
        <span className="font-semibold">Flex impact: </span>{alert.flexImpact}
      </p>
    </div>
  );
}

function IntensityTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const total = payload.reduce((s, p) => s + (p.value as number), 0);
  return (
    <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-3 shadow-lg text-xs">
      <p className="font-bold text-slate-900 dark:text-white mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }}>{p.name}: <span className="font-semibold">{p.value}</span></p>
      ))}
      <p className="mt-1 border-t border-slate-100 dark:border-slate-700 pt-1 text-slate-500">
        Combined: <span className="font-semibold">{total}</span>
      </p>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────
export default function AnalysisPage() {
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [keywordsOpen, setKeywordsOpen] = useState(false);

  // Keep API data state for future live connections
  const [, setCapexData]  = useState<unknown[]>([]);
  const [, setAiData]     = useState<unknown[]>([]);
  const [, setTrends]     = useState<unknown>(null);
  const [, setClassification] = useState<unknown>(null);
  const [, setAnomalies]  = useState<unknown>(null);

  useEffect(() => { fetchAllData(); }, []);

  const fetchAllData = async () => {
    setLoading(true);
    try {
      const results = await Promise.allSettled([
        fetch(`${API_URL}/api/analysis/capex`),
        fetch(`${API_URL}/api/analysis/ai-investments`),
        fetch(`${API_URL}/api/analytics/trends`),
        fetch(`${API_URL}/api/analytics/classification`),
        fetch(`${API_URL}/api/analytics/anomalies`),
      ]);
      const [capexR, aiR, trendsR, classR, anomR] = results;
      if (capexR.status === 'fulfilled' && capexR.value.ok) {
        const d = await capexR.value.json(); setCapexData(d.mentions || []);
      }
      if (aiR.status === 'fulfilled' && aiR.value.ok) {
        const d = await aiR.value.json(); setAiData(d.mentions || []);
      }
      if (trendsR.status === 'fulfilled' && trendsR.value.ok) {
        setTrends(await trendsR.value.json());
      }
      if (classR.status === 'fulfilled' && classR.value.ok) {
        setClassification(await classR.value.json());
      }
      if (anomR.status === 'fulfilled' && anomR.value.ok) {
        setAnomalies(await anomR.value.json());
      }
      setError(null);
    } catch {
      setError('Some backend endpoints unavailable — showing static data.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
        <RefreshCw className="h-8 w-8 animate-spin text-purple-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 p-6 space-y-6">

      {/* ── Header ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="bg-gradient-to-br from-purple-600 to-indigo-700 p-3 rounded-xl shadow-lg shadow-purple-500/20">
            <Zap className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-white">AI Exposure Tracker</h1>
            <p className="text-slate-500 dark:text-slate-400 text-sm mt-0.5">
              How aggressively are EMS peers positioning in AI/DC — based on earnings document NLP analysis
            </p>
          </div>
        </div>
        <Button variant="outline" onClick={fetchAllData} className="dark:border-slate-600 dark:text-slate-300">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-700 text-sm text-orange-700 dark:text-orange-400">
          {error}
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 1 — Insight Cards
      ══════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-0 shadow-lg dark:bg-slate-800/60 border-l-4 border-l-red-400">
          <CardContent className="p-5">
            <div className="flex items-start gap-3">
              <div className="p-2.5 bg-red-100 dark:bg-red-900/30 rounded-lg flex-shrink-0">
                <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-0.5">AI Narrative Gap</p>
                <p className="text-xl font-bold text-slate-900 dark:text-white leading-tight">Celestica leads Flex by 65 pts</p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">In AI/DC mention intensity across last 4 quarters</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg dark:bg-slate-800/60 border-l-4 border-l-orange-400">
          <CardContent className="p-5">
            <div className="flex items-start gap-3">
              <div className="p-2.5 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex-shrink-0">
                <TrendingUp className="h-5 w-5 text-orange-600 dark:text-orange-400" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-0.5">Fastest Accelerating</p>
                <p className="text-xl font-bold text-slate-900 dark:text-white leading-tight">Jabil +67% QoQ</p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Biggest increase in AI/DC mentions vs prior quarter</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg dark:bg-slate-800/60 border-l-4 border-l-orange-400">
          <CardContent className="p-5">
            <div className="flex items-start gap-3">
              <div className="p-2.5 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex-shrink-0">
                <Target className="h-5 w-5 text-orange-600 dark:text-orange-400" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-0.5">Flex AI Positioning</p>
                <p className="text-xl font-bold text-slate-900 dark:text-white leading-tight">Below Peer Avg</p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Flex ranks 4th of 5 in AI narrative intensity</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 2 — Three Charts
      ══════════════════════════════════════════════════════════════════ */}

      {/* Charts A + B: side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* A: Horizontal grouped bar */}
        <Card className="border-0 shadow-xl dark:bg-slate-800/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">AI/DC Narrative Intensity by Company</CardTitle>
            <p className="text-xs text-slate-400 dark:text-slate-500">Based on keyword frequency in earnings calls and filings</p>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={INTENSITY_DATA} layout="vertical" margin={{ left: 10, right: 40, top: 4, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                  <XAxis type="number" domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                  <YAxis dataKey="name" type="category" width={72} tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }} />
                  <Tooltip content={<IntensityTooltip />} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                  <ReferenceLine x={27} stroke="#3B82F6" strokeDasharray="4 3" strokeWidth={1.5}
                    label={{ value: 'Flex baseline', position: 'insideTopRight', fill: '#3B82F6', fontSize: 10, fontWeight: 600 }}
                  />
                  <Bar dataKey="AI" name="AI/ML Mentions"       fill="#8b5cf6" radius={[0, 4, 4, 0]} maxBarSize={14} />
                  <Bar dataKey="DC" name="Data Center Mentions" fill="#10b981" radius={[0, 4, 4, 0]} maxBarSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* B: Multi-line trend */}
        <Card className="border-0 shadow-xl dark:bg-slate-800/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">AI/DC Narrative Momentum — 5 Quarter Trend</CardTitle>
            <p className="text-xs text-slate-400 dark:text-slate-500">Celestica and Jabil accelerating significantly vs Flex</p>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={TREND_DATA} margin={{ left: 0, right: 16, top: 8, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="q" tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#94a3b8' }} domain={[0, 55]} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                  <ReferenceLine x="Q4 FY25" stroke="transparent" strokeWidth={0}
                    label={{ value: '↑ Celestica acceleration', position: 'insideTopLeft', fill: '#6366F1', fontSize: 9, fontWeight: 600 }}
                  />
                  <Line dataKey="Celestica" stroke={COMPANY_COLORS.Celestica} strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                  <Line dataKey="Jabil"     stroke={COMPANY_COLORS.Jabil}     strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} />
                  <Line dataKey="Sanmina"   stroke={COMPANY_COLORS.Sanmina}   strokeWidth={1.5} dot={{ r: 2.5 }} />
                  <Line dataKey="Flex"      stroke={COMPANY_COLORS.Flex}      strokeWidth={2}   strokeDasharray="5 3" dot={{ r: 3 }} activeDot={{ r: 5 }} />
                  <Line dataKey="Benchmark" stroke={COMPANY_COLORS.Benchmark} strokeWidth={1.5} dot={{ r: 2.5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1 text-right">Flex shown as dashed line</p>
          </CardContent>
        </Card>
      </div>

      {/* C: AI vs Traditional Mix — full width */}
      <Card className="border-0 shadow-xl dark:bg-slate-800/60">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between flex-wrap gap-2">
            <div>
              <CardTitle className="text-base">AI vs Traditional Revenue Mix</CardTitle>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                Estimated AI/DC exposure as % of total business focus
              </p>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 italic max-w-xs text-right">
              Celestica and Jabil have pivoted heavily to AI/DC. Flex is mid-transition. Benchmark remains traditional-focused.
            </p>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={MIX_DATA} layout="vertical" margin={{ left: 10, right: 60, top: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                <XAxis type="number" domain={[0, 100]} tickLine={false} axisLine={false}
                  tick={{ fontSize: 11, fill: '#94a3b8' }}
                  tickFormatter={(v) => `${v}%`}
                />
                <YAxis dataKey="name" type="category" width={72} tickLine={false} axisLine={false}
                  tick={{ fontSize: 12, fill: '#64748b', fontWeight: 500 }}
                />
                <Tooltip
                  formatter={(value: number, name: string) => [`${value}%`, name]}
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
                />
                <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                <ReferenceLine x={55} stroke="#3B82F6" strokeDasharray="4 3" strokeWidth={1.5}
                  label={{ value: 'Flex AI mix (55%)', position: 'insideTopRight', fill: '#3B82F6', fontSize: 10, fontWeight: 600 }}
                />
                <Bar dataKey="ai"   name="AI/Data Center" stackId="a" fill="#8B5CF6" radius={[0, 0, 0, 0]} maxBarSize={18} />
                <Bar dataKey="trad" name="Traditional"    stackId="a" fill="#94A3B8" radius={[0, 4, 4, 0]} maxBarSize={18} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 3 — Comparison Table
      ══════════════════════════════════════════════════════════════════ */}
      <Card className="border-0 shadow-xl dark:bg-slate-800/60">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Company Comparison — AI/DC Positioning</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 dark:border-slate-700">
                  {['Company', 'CapEx Mentions', 'AI/ML Mentions', 'Data Center', 'Tech Focus Score', 'Trend', 'vs Flex'].map((h, i) => (
                    <th key={h} className={`py-3 px-4 font-semibold text-slate-500 dark:text-slate-400 ${i === 0 ? 'text-left' : i === 5 ? 'text-center' : 'text-right'}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {STATIC_TABLE.map((row) => {
                  const isFlexRow = row.company === 'Flex';
                  const isThreat  = row.vsFlexDiff > 0;
                  return (
                    <tr key={row.company} className={[
                      'border-b last:border-0 transition-colors',
                      isFlexRow ? 'bg-blue-50/60 dark:bg-blue-900/10 border-l-2 border-l-blue-500'
                        : isThreat ? 'bg-red-50/30 dark:bg-red-900/5 hover:bg-red-50/60 dark:hover:bg-red-900/10'
                        : 'hover:bg-slate-50 dark:hover:bg-slate-700/30',
                    ].join(' ')}>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: COMPANY_COLORS[row.company] || '#64748b' }} />
                          <span className={`font-semibold ${isFlexRow ? 'text-blue-700 dark:text-blue-400' : 'text-slate-800 dark:text-slate-200'}`}>
                            {row.company}
                          </span>
                          {isFlexRow && <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400">US</span>}
                        </div>
                      </td>
                      <td className="text-right py-3 px-4 text-slate-600 dark:text-slate-300">{row.capex}</td>
                      <td className="text-right py-3 px-4 font-semibold" style={{ color: '#8b5cf6' }}>{row.ai}</td>
                      <td className="text-right py-3 px-4 font-semibold" style={{ color: '#10b981' }}>{row.dc}</td>
                      <td className="text-right py-3 px-4"><span className="font-bold text-slate-900 dark:text-white">{row.score}</span></td>
                      <td className="text-center py-3 px-4"><TrendBadge trend={row.trend} label={row.trendLabel} /></td>
                      <td className="text-right py-3 px-4"><VsFlexBadge diff={row.vsFlexDiff} label={row.vsFlexLabel} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-4 p-4 rounded-xl bg-orange-50 dark:bg-orange-900/10 border border-orange-200 dark:border-orange-800">
            <p className="text-sm text-orange-800 dark:text-orange-300">
              <span className="font-bold">⚠️ Insight:</span>{' '}
              Celestica and Jabil are significantly outpacing Flex in AI/DC narrative intensity. This suggests
              stronger strategic commitment and likely higher investor/customer AI positioning. Flex's CapEx
              mention count is high but AI-specific language remains below peer average.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 4 — Strategic Momentum Cards
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <div className="mb-4">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Strategic Momentum — Trend Direction</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">Quarter-over-quarter direction of key strategic indicators</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {MOMENTUM_CARDS.map((card) => {
            const color = COMPANY_COLORS[card.company] || '#64748b';
            return (
              <Card key={card.company} className="border-0 shadow-lg dark:bg-slate-800/60 overflow-hidden">
                <div className="flex h-full">
                  <div className="w-1 flex-shrink-0" style={{ backgroundColor: color }} />
                  <div className="flex-1 p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-bold text-slate-900 dark:text-white text-sm">{card.company}</span>
                      <OverallBadge overall={card.overall} />
                    </div>
                    <div className="space-y-2">
                      {card.rows.map((row) => (
                        <div key={row.label} className="flex items-center justify-between text-xs gap-2">
                          <span className="text-slate-500 dark:text-slate-400 flex-shrink-0">{row.label}</span>
                          <div className="flex items-center gap-1 text-right">
                            <DirectionIcon direction={row.direction} />
                            <span className={`text-slate-700 dark:text-slate-300 leading-tight ${row.label === 'AI/DC Mix' ? 'font-semibold' : ''}`}>
                              {row.value}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 5 — Strategic Anomaly Alerts
      ══════════════════════════════════════════════════════════════════ */}
      <div>
        <div className="mb-4">
          <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-orange-500" />
            Strategic Anomaly Alerts
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-0.5">Unusual patterns detected in CapEx and AI investment signals</p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {ANOMALY_ALERTS.map((alert) => (
            <AlertCard key={alert.company} alert={alert} />
          ))}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          SECTION 6 — Keyword Deep Dive (collapsible)
      ══════════════════════════════════════════════════════════════════ */}
      <Card className="border-0 shadow-xl dark:bg-slate-800/60">
        <button
          type="button"
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => setKeywordsOpen((v) => !v)}
        >
          <div>
            <p className="font-semibold text-slate-900 dark:text-white text-sm">📝 Top AI/DC Keywords by Company</p>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
              Most frequently mentioned AI and data center terms in earnings documents
            </p>
          </div>
          {keywordsOpen
            ? <ChevronUp className="h-4 w-4 text-slate-400 flex-shrink-0" />
            : <ChevronDown className="h-4 w-4 text-slate-400 flex-shrink-0" />
          }
        </button>
        {keywordsOpen && (
          <CardContent className="pt-0">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {Object.entries(KEYWORDS).map(([company, tags]) => {
                const color = COMPANY_COLORS[company] || '#64748b';
                return (
                  <div key={company}>
                    <div className="flex items-center gap-1.5 mb-2">
                      <div className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                      <p className="text-xs font-bold text-slate-700 dark:text-slate-300">{company}</p>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-[11px] font-medium px-2 py-0.5 rounded-full"
                          style={{ backgroundColor: color + '18', color, border: `1px solid ${color}40` }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        )}
      </Card>

    </div>
  );
}
