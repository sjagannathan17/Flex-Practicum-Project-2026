'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ChartDescription } from '@/components/ui/chart-description';
import {
  DollarSign,
  TrendingUp,
  Building2,
  Cpu,
  Zap,
  Globe,
  RefreshCw,
  Server,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  ArrowUpRight,
  ArrowDownRight,
  Newspaper,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

interface Big5Company {
  name: string;
  ticker: string;
  capex_2026_billions: number;
  capex_2025_billions: number;
  yoy_growth_pct: number;
  ai_focus_areas: string[];
  key_metrics: Record<string, number>;
  recent_announcements: string[];
  color: string;
}

interface StargateProject {
  total_investment_billions: number;
  timeline: string;
  partners: string[];
  initial_deployment_billions: number;
  planned_capacity_gw: number;
  locations: string[];
}

interface Big5Data {
  last_updated: string;
  source: string;
  total_2026_capex_billions: number;
  companies: Big5Company[];
  stargate_project: StargateProject;
}

interface NewsItem {
  title: string;
  url?: string;
  description?: string;
  source?: string;
  timestampLabel?: string;
  categories?: string[];
}

function formatAISubdomainLabel(area: string): string {
  const raw = area.trim();
  if (!raw) return 'AI (General)';
  const noLeadingAI = raw.replace(/^ai[\s/-]+/i, '');
  const noTrailingAI = noLeadingAI.replace(/[\s/-]+ai$/i, '');
  const aiParenMatch = noTrailingAI.match(/^(.*)\(ai\)\s*$/i);
  if (aiParenMatch && aiParenMatch[1]?.trim()) {
    const sub = aiParenMatch[1].trim();
    const titledSub = sub.split(' ').filter(Boolean).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    return `AI (${titledSub})`;
  }
  const leadingStyleMatch = noTrailingAI.match(/^ai\s*\((.+)\)$/i);
  if (leadingStyleMatch?.[1]) {
    return `AI (${leadingStyleMatch[1].trim()})`;
  }
  const titled = noTrailingAI.split(' ').filter(Boolean).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  return `AI (${titled})`;
}

const DEMAND_SIGNALS = [
  {
    company: 'AWS',
    color: '#FF9900',
    headline: 'CapEx guidance doubled YoY',
    dataPoint: '$200B',
    dataDetail: '+100%',
    direction: 'up' as const,
    opportunity: 'HIGH' as const,
    implication: 'AI server, power systems, and liquid cooling demand surge expected',
  },
  {
    company: 'Meta',
    color: '#1877F2',
    headline: 'Highest YoY growth in Big Five',
    dataPoint: '+212% YoY',
    dataDetail: '$125B',
    direction: 'up' as const,
    opportunity: 'HIGH' as const,
    implication: 'Rapid infrastructure scale-up creates new supplier qualification windows',
  },
  {
    company: 'Alphabet',
    color: '#34A853',
    headline: 'Sustained AI infrastructure investment',
    dataPoint: '$180B',
    dataDetail: '+140%',
    direction: 'up' as const,
    opportunity: 'HIGH' as const,
    implication: 'Custom networking and compute hardware demand expanding',
  },
  {
    company: 'Microsoft',
    color: '#00A4EF',
    headline: 'Steady growth, Azure-focused',
    dataPoint: '$120B',
    dataDetail: '+50%',
    direction: 'up' as const,
    opportunity: 'MEDIUM' as const,
    implication: 'More predictable demand curve, good for long-term contracts',
  },
  {
    company: 'Oracle',
    color: '#F80000',
    headline: 'Stargate anchor, GPU clusters',
    dataPoint: '$50B',
    dataDetail: '+136%',
    direction: 'up' as const,
    opportunity: 'MEDIUM' as const,
    implication: 'Niche but fast-growing; focus on compute rack opportunities',
  },
];

const PRIORITY_TABLE = [
  { rank: 1, company: 'Amazon (AWS)', ticker: 'AMZN', capex: '$200B', yoy: '+100%', aiFocus: 'AI Compute, Cloud Infra', opportunity: 'HIGH' as const, color: '#FF9900' },
  { rank: 2, company: 'Meta', ticker: 'META', capex: '$125B', yoy: '+212%', aiFocus: 'AI Compute, Custom Silicon', opportunity: 'HIGH' as const, color: '#1877F2' },
  { rank: 3, company: 'Alphabet', ticker: 'GOOGL', capex: '$180B', yoy: '+140%', aiFocus: 'AI Infra, Custom Chips', opportunity: 'HIGH' as const, color: '#34A853' },
  { rank: 4, company: 'Microsoft', ticker: 'MSFT', capex: '$120B', yoy: '+50%', aiFocus: 'Azure AI, Cloud', opportunity: 'MEDIUM' as const, color: '#00A4EF' },
  { rank: 5, company: 'Oracle', ticker: 'ORCL', capex: '$50B', yoy: '+136%', aiFocus: 'GPU Clusters, Stargate', opportunity: 'MEDIUM' as const, color: '#F80000' },
];

const INFRA_KEYWORDS = [
  'data center', 'ai infrastructure', 'gpu cluster', 'liquid cooling',
  'power capacity', 'facility', 'construction', 'hyperscaler', 'expansion',
  'aws', 'azure', 'google cloud', 'meta ai', 'oracle cloud',
];

const HYPERSCALER_COMPANIES = ['amazon', 'aws', 'meta', 'alphabet', 'google', 'microsoft', 'oracle'];

function OpportunityBadge({ level }: { level: 'HIGH' | 'MEDIUM' | 'WATCH' }) {
  if (level === 'HIGH')   return <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-600 border border-red-300">HIGH</span>;
  if (level === 'MEDIUM') return <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-orange-100 text-orange-600 border border-orange-300">MEDIUM</span>;
  return <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-600 border border-green-300">WATCH</span>;
}

export default function HyperscalerDemandPage() {
  const [data, setData] = useState<Big5Data | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedRowTicker, setSelectedRowTicker] = useState<string | null>(null);
  const [chartsExpanded, setChartsExpanded] = useState(false);
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);

  useEffect(() => {
    fetchData();
    fetchNews();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/intelligence/big5-capex`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (err) {
      console.error('Failed to fetch Big 5 data:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchNews = async () => {
    try {
      const res = await fetch(`${API_URL}/api/news/feed`);
      if (res.ok) {
        const json = await res.json();
        const allItems: NewsItem[] = json.items || json.news || json || [];
        const filtered = allItems.filter((item: NewsItem) => {
          const text = `${item.title} ${item.description || ''}`.toLowerCase();
          const matchesKeyword = INFRA_KEYWORDS.some((kw) => text.includes(kw));
          const matchesCompany = HYPERSCALER_COMPANIES.some((c) => text.includes(c));
          return matchesKeyword || matchesCompany;
        });
        setNewsItems(filtered.slice(0, 6));
      }
    } catch {
      // leave empty — placeholder shown
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 flex items-center justify-center dark:from-slate-950 dark:to-slate-950">
        <div className="text-slate-500">Loading demand intelligence data...</div>
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

  const chartData = data.companies.map((c) => ({
    name: c.name.split(' ')[0],
    capex_2026: c.capex_2026_billions,
    capex_2025: c.capex_2025_billions,
    color: c.color,
  }));

  const pieData = data.companies.map((c) => ({
    name: c.name.split(' ')[0],
    value: c.capex_2026_billions,
    color: c.color,
  }));

  const totalCapex = data.companies.reduce((sum, c) => sum + c.capex_2026_billions, 0);
  const avgGrowth = Math.round(data.companies.reduce((sum, c) => sum + c.yoy_growth_pct, 0) / data.companies.length);

  // Map ticker → company detail for expandable rows
  const companyByTicker: Record<string, Big5Company> = {};
  for (const c of data.companies) {
    companyByTicker[c.ticker] = c;
  }
  // Also map by first-name match for Priority Table
  const companyByFirstName: Record<string, Big5Company> = {};
  for (const c of data.companies) {
    companyByFirstName[c.name.split(' ')[0].toLowerCase()] = c;
  }

  function getDetailCompany(row: typeof PRIORITY_TABLE[0]): Big5Company | null {
    return companyByTicker[row.ticker] || companyByFirstName[row.company.split(' ')[0].toLowerCase()] || null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-4 dark:from-slate-950 dark:via-slate-950 dark:to-slate-950">
      {/* Header */}
      <div className="mb-5">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-3">
              <div className="bg-gradient-to-br from-orange-500 to-red-600 p-1.5 rounded-xl">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              Hyperscaler Demand Intelligence
            </h1>
            <p className="text-slate-500 mt-0.5 text-sm">
              AWS · Alphabet · Microsoft · Meta · Oracle — FY2026 Capital Expenditure Outlook — Flex Business Opportunity View
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Badge className="bg-green-100 text-green-700">Updated: {data.last_updated}</Badge>
            <button
              onClick={fetchData}
              className="flex items-center gap-2 px-3 py-1.5 bg-white rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-all shadow-sm dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-5">

        {/* ── SECTION 1: Demand Signals ── */}
        <Card className="border-0 shadow-xl dark:bg-slate-900 dark:text-slate-100">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <ArrowUpRight className="h-5 w-5 text-orange-500" />
              Demand Signals
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {DEMAND_SIGNALS.map((signal) => (
                <div
                  key={signal.company}
                  className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800/60"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-sm shrink-0"
                        style={{ backgroundColor: signal.color }}
                      >
                        {signal.company.charAt(0)}
                      </div>
                      <div className="min-w-0">
                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">{signal.company}</span>
                        <p className="font-semibold text-slate-900 dark:text-slate-100 leading-tight">{signal.headline}</p>
                      </div>
                    </div>
                    <OpportunityBadge level={signal.opportunity} />
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <span className="text-2xl font-bold" style={{ color: signal.color }}>{signal.dataPoint}</span>
                    <span className="text-sm text-slate-500 dark:text-slate-400">{signal.dataDetail}</span>
                    {signal.direction === 'up'
                      ? <ArrowUpRight className="h-5 w-5 text-green-500 ml-auto" />
                      : <ArrowDownRight className="h-5 w-5 text-red-500 ml-auto" />}
                  </div>
                  <p className="mt-2 text-xs text-slate-500 dark:text-slate-400 border-t border-slate-100 dark:border-slate-700 pt-2">
                    Likely drives {signal.implication}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* ── SECTION 2: Customer Priority Ranking ── */}
        <Card className="border-0 shadow-xl dark:bg-slate-900 dark:text-slate-100">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2">
              <Building2 className="h-5 w-5 text-blue-500" />
              Customer Priority Ranking
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="text-left py-2 px-3 text-slate-500 dark:text-slate-400 font-medium w-16">Priority</th>
                    <th className="text-left py-2 px-3 text-slate-500 dark:text-slate-400 font-medium">Company</th>
                    <th className="text-left py-2 px-3 text-slate-500 dark:text-slate-400 font-medium">2026 CapEx</th>
                    <th className="text-left py-2 px-3 text-slate-500 dark:text-slate-400 font-medium">YoY Growth</th>
                    <th className="text-left py-2 px-3 text-slate-500 dark:text-slate-400 font-medium">AI Focus</th>
                    <th className="text-left py-2 px-3 text-slate-500 dark:text-slate-400 font-medium">Flex Opportunity</th>
                  </tr>
                </thead>
                <tbody>
                  {PRIORITY_TABLE.map((row, idx) => {
                    const isExpanded = selectedRowTicker === row.ticker;
                    const detail = getDetailCompany(row);
                    return (
                      <React.Fragment key={row.ticker}>
                        <tr
                          key={row.ticker}
                          onClick={() => setSelectedRowTicker(isExpanded ? null : row.ticker)}
                          className={`cursor-pointer transition-colors border-b border-slate-100 dark:border-slate-800 ${
                            idx % 2 === 0
                              ? 'bg-white dark:bg-slate-900'
                              : 'bg-slate-50/60 dark:bg-slate-800/40'
                          } hover:bg-blue-50 dark:hover:bg-blue-950/30`}
                        >
                          <td className="py-3 px-3">
                            <div className="flex items-center gap-2">
                              <div
                                className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold"
                                style={{ backgroundColor: row.color }}
                              >
                                {row.rank}
                              </div>
                            </div>
                          </td>
                          <td className="py-3 px-3">
                            <div className="flex items-center gap-2">
                              <div
                                className="w-7 h-7 rounded-lg flex items-center justify-center text-white font-bold text-xs shrink-0"
                                style={{ backgroundColor: row.color }}
                              >
                                {row.company.charAt(0)}
                              </div>
                              <div>
                                <p className="font-semibold text-slate-900 dark:text-slate-100">{row.company}</p>
                                <p className="text-xs text-slate-400">{row.ticker}</p>
                              </div>
                            </div>
                          </td>
                          <td className="py-3 px-3 font-bold text-slate-900 dark:text-slate-100">{row.capex}</td>
                          <td className="py-3 px-3 font-semibold text-green-600">{row.yoy}</td>
                          <td className="py-3 px-3 text-slate-600 dark:text-slate-300">{row.aiFocus}</td>
                          <td className="py-3 px-3">
                            <div className="flex items-center justify-between gap-2">
                              <OpportunityBadge level={row.opportunity} />
                              {isExpanded ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
                            </div>
                          </td>
                        </tr>
                        {isExpanded && detail && (
                          <tr className="bg-blue-50/60 dark:bg-blue-950/20">
                            <td colSpan={6} className="px-4 pb-4 pt-2">
                              <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-white dark:bg-slate-900 p-4">
                                <h4 className="font-semibold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
                                  <div className="w-6 h-6 rounded-md flex items-center justify-center text-white text-xs font-bold" style={{ backgroundColor: row.color }}>
                                    {row.company.charAt(0)}
                                  </div>
                                  {detail.name} — AI Investment Details
                                </h4>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                  <div>
                                    <h5 className="mb-2 font-semibold text-slate-700 dark:text-slate-200 text-sm">Key Metrics</h5>
                                    <div className="space-y-1.5">
                                      <div className="flex justify-between rounded-lg bg-slate-50 p-2 text-xs dark:bg-slate-800">
                                        <span className="text-slate-500 dark:text-slate-400">2026 CapEx</span>
                                        <span className="font-bold text-slate-900 dark:text-slate-100">${detail.capex_2026_billions}B</span>
                                      </div>
                                      <div className="flex justify-between rounded-lg bg-slate-50 p-2 text-xs dark:bg-slate-800">
                                        <span className="text-slate-500 dark:text-slate-400">2025 CapEx</span>
                                        <span className="font-bold text-slate-900 dark:text-slate-100">${detail.capex_2025_billions}B</span>
                                      </div>
                                      <div className="flex justify-between rounded-lg bg-green-50 p-2 text-xs dark:bg-emerald-900/30">
                                        <span className="text-slate-600 dark:text-slate-300">YoY Growth</span>
                                        <span className="font-bold text-green-600">+{detail.yoy_growth_pct}%</span>
                                      </div>
                                    </div>
                                  </div>
                                  <div>
                                    <h5 className="mb-2 font-semibold text-slate-700 dark:text-slate-200 text-sm">AI Focus Subdomains</h5>
                                    <div className="space-y-1.5">
                                      {detail.ai_focus_areas.map((area, i) => (
                                        <div key={i} className="flex items-center gap-2 rounded-lg bg-purple-50 p-2 dark:bg-violet-900/30">
                                          <Cpu className="h-3 w-3 text-purple-600 dark:text-violet-300 shrink-0" />
                                          <span className="text-xs text-slate-700 dark:text-slate-100">{formatAISubdomainLabel(area)}</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                  <div>
                                    <h5 className="mb-2 font-semibold text-slate-700 dark:text-slate-200 text-sm">Recent Announcements</h5>
                                    <div className="space-y-1.5">
                                      {detail.recent_announcements.map((ann, i) => (
                                        <div key={i} className="rounded-lg bg-blue-50 p-2 dark:bg-sky-900/20">
                                          <p className="text-xs text-slate-700 dark:text-slate-100">{ann}</p>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* ── SECTION 3: CapEx Overview (collapsible) ── */}
        <Card className="border-0 shadow-xl dark:bg-slate-900 dark:text-slate-100">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <DollarSign className="h-5 w-5 text-orange-500" />
                CapEx Overview
              </CardTitle>
              <button
                onClick={() => setChartsExpanded(!chartsExpanded)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 text-sm transition-colors"
              >
                {chartsExpanded ? (
                  <><ChevronUp className="h-4 w-4" /> Hide CapEx Charts</>
                ) : (
                  <><ChevronDown className="h-4 w-4" /> Show CapEx Charts</>
                )}
              </button>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            {/* Summary stat cards — always visible */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <Card className="border-0 shadow-md bg-gradient-to-br from-orange-500 to-red-600 text-white">
                <CardContent className="p-3">
                  <p className="text-orange-100 text-xs">Total 2026 CapEx</p>
                  <div className="mt-1 flex items-center gap-2">
                    <p className="text-xl font-bold">${totalCapex}B</p>
                    <DollarSign className="h-4 w-4 text-orange-200" />
                  </div>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-md dark:border dark:border-slate-700 dark:bg-slate-800">
                <CardContent className="p-3">
                  <p className="text-slate-500 dark:text-slate-400 text-xs">YoY Growth (Avg)</p>
                  <div className="mt-1 flex items-center gap-2">
                    <p className="text-xl font-bold text-green-600">+{avgGrowth}%</p>
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  </div>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-md dark:border dark:border-slate-700 dark:bg-slate-800">
                <CardContent className="p-3">
                  <p className="text-slate-500 dark:text-slate-400 text-xs">Stargate Project</p>
                  <div className="mt-1 flex items-center gap-2">
                    <p className="text-xl font-bold text-blue-600">${data.stargate_project.total_investment_billions}B</p>
                    <Server className="h-4 w-4 text-blue-500" />
                  </div>
                </CardContent>
              </Card>
              <Card className="border-0 shadow-md dark:border dark:border-slate-700 dark:bg-slate-800">
                <CardContent className="p-3">
                  <p className="text-slate-500 dark:text-slate-400 text-xs">Planned Capacity</p>
                  <div className="mt-1 flex items-center gap-2">
                    <p className="text-xl font-bold text-purple-600">{data.stargate_project.planned_capacity_gw}GW</p>
                    <Zap className="h-4 w-4 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Collapsible charts + Stargate detail */}
            {chartsExpanded && (
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <Card className="border border-slate-200 dark:border-slate-700 shadow-none dark:bg-slate-800/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center gap-2">
                        <Building2 className="h-4 w-4 text-blue-600" />
                        2025 vs 2026 CapEx Comparison
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={chartData} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                          <XAxis type="number" unit="B" />
                          <YAxis type="category" dataKey="name" width={80} />
                          <Tooltip
                            formatter={(value) => [`$${value}B`, '']}
                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 40px rgba(0,0,0,0.1)' }}
                          />
                          <Bar dataKey="capex_2025" name="2025 CapEx" fill="#A78BFA" radius={[0, 4, 4, 0]} />
                          <Bar dataKey="capex_2026" name="2026 CapEx" radius={[0, 4, 4, 0]}>
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <ChartDescription
                        description="Year-over-year capital expenditure comparison. 2026 projections show near-doubling of investment levels."
                        source={data.source}
                        lastUpdated={data.last_updated}
                      />
                    </CardContent>
                  </Card>

                  <Card className="border border-slate-200 dark:border-slate-700 shadow-none dark:bg-slate-800/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center gap-2">
                        <Cpu className="h-4 w-4 text-purple-600" />
                        2026 CapEx Distribution
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={220}>
                        <PieChart margin={{ top: 8, right: 28, left: 28, bottom: 8 }}>
                          <Pie
                            data={pieData}
                            cx="54%"
                            cy="50%"
                            outerRadius={88}
                            innerRadius={46}
                            paddingAngle={2}
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip formatter={(value) => [`$${value}B`, 'CapEx']} />
                        </PieChart>
                      </ResponsiveContainer>
                      <ChartDescription
                        description="Market share of planned AI infrastructure spending. Amazon leads with ~30% of total planned investment."
                        source="Company Earnings Reports"
                      />
                    </CardContent>
                  </Card>
                </div>

                {/* Stargate Project detail */}
                <div className="rounded-xl border border-slate-200 bg-gradient-to-r from-slate-50 to-slate-100 p-4 dark:border-slate-700 dark:from-slate-900 dark:to-slate-800">
                  <h3 className="mb-3 flex items-center gap-2 font-semibold text-slate-900 dark:text-white">
                    <Globe className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                    Stargate Project — $500B AI Infrastructure Initiative
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                    <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200 dark:bg-white/10 dark:ring-0">
                      <p className="text-xs text-slate-500 dark:text-slate-400">Total Investment</p>
                      <p className="text-xl font-bold text-slate-900 dark:text-white">${data.stargate_project.total_investment_billions}B</p>
                    </div>
                    <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200 dark:bg-white/10 dark:ring-0">
                      <p className="text-xs text-slate-500 dark:text-slate-400">Timeline</p>
                      <p className="text-xl font-bold text-slate-900 dark:text-white">{data.stargate_project.timeline}</p>
                    </div>
                    <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200 dark:bg-white/10 dark:ring-0">
                      <p className="text-xs text-slate-500 dark:text-slate-400">Initial Deployment</p>
                      <p className="text-xl font-bold text-slate-900 dark:text-white">${data.stargate_project.initial_deployment_billions}B</p>
                    </div>
                    <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200 dark:bg-white/10 dark:ring-0">
                      <p className="text-xs text-slate-500 dark:text-slate-400">Planned Capacity</p>
                      <p className="text-xl font-bold text-slate-900 dark:text-white">{data.stargate_project.planned_capacity_gw}GW</p>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-4">
                    <div>
                      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">Partners</p>
                      <div className="flex flex-wrap gap-2">
                        {data.stargate_project.partners.map((p) => (
                          <Badge key={p} className="bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300">{p}</Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">Locations</p>
                      <div className="flex flex-wrap gap-2">
                        {data.stargate_project.locations.map((l) => (
                          <Badge key={l} className="bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-300">{l}</Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 border-t border-slate-200 dark:border-white/10 pt-3">
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Source: Futurum Research - AI Capex 2026 Report
                      <a
                        href="https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-2 inline-flex items-center gap-1 text-blue-400 hover:text-blue-300"
                      >
                        Read Full Report <ExternalLink className="h-3 w-3" />
                      </a>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── SECTION 4: Infrastructure News ── */}
        <Card className="border-0 shadow-xl dark:bg-slate-900 dark:text-slate-100">
          <CardHeader className="pb-3">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Newspaper className="h-5 w-5 text-slate-500" />
                Hyperscaler Infrastructure News
              </CardTitle>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                AWS · Alphabet · Microsoft · Meta · Oracle — Data Center &amp; AI Build-out
              </p>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            {newsItems.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                {newsItems.map((item, idx) => {
                  const text = `${item.title} ${item.description || ''}`.toLowerCase();
                  const relevanceLevel = (['data center', 'ai infrastructure', 'gpu cluster', 'liquid cooling'].some((k) => text.includes(k)))
                    ? '🔴'
                    : (['aws', 'azure', 'google cloud', 'expansion', 'hyperscaler'].some((k) => text.includes(k)))
                    ? '🟠'
                    : '🟢';
                  return (
                    <div key={idx} className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/60 p-3 flex flex-col gap-2">
                      <div className="flex items-start gap-2">
                        <span className="text-base">{relevanceLevel}</span>
                        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100 leading-snug flex-1">
                          {item.url ? (
                            <a href={item.url} target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 dark:hover:text-blue-400">
                              {item.title}
                            </a>
                          ) : item.title}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <span>{item.source || 'News'}</span>
                        {item.timestampLabel && <><span>·</span><span>{item.timestampLabel}</span></>}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-xl border border-dashed border-slate-300 dark:border-slate-600 p-8 text-center text-slate-400 dark:text-slate-500">
                Infrastructure news feed — connect news data source to activate
              </div>
            )}
          </CardContent>
        </Card>

      </div>
    </div>
  );
}
