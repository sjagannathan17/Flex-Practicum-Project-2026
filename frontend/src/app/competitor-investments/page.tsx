'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ChartDescription } from '@/components/ui/chart-description';
import {
  TrendingUp,
  Building2,
  Target,
  Lightbulb,
  RefreshCw,
  ArrowUpRight,
  Cpu,
  DollarSign,
  ChevronRight,
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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

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
  competitors: CompetitorInvestment[];
  hyperscaler_demand: HyperscalerDemand;
}

const COMPANY_COLORS: Record<string, string> = {
  'Flex': '#3B82F6',
  'Jabil': '#10B981',
  'Celestica': '#6366F1',
  'Benchmark': '#F59E0B',
  'Sanmina': '#EF4444',
};

const OUTLOOK_COLORS: Record<string, string> = {
  'Very bullish': 'bg-green-100 text-green-700 border-green-300',
  'Strong': 'bg-emerald-100 text-emerald-700 border-emerald-300',
  'Positive': 'bg-blue-100 text-blue-700 border-blue-300',
  'Stable': 'bg-slate-100 text-slate-700 border-slate-300',
  'Cautious': 'bg-amber-100 text-amber-700 border-amber-300',
};

function getOutlookStyle(outlook: string): string {
  for (const [key, value] of Object.entries(OUTLOOK_COLORS)) {
    if (outlook.toLowerCase().includes(key.toLowerCase())) {
      return value;
    }
  }
  return 'bg-slate-100 text-slate-700 border-slate-300';
}

export default function CompetitorInvestmentsPage() {
  const [data, setData] = useState<CompetitorData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCompany, setSelectedCompany] = useState<CompetitorInvestment | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/intelligence/competitor-investments`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setSelectedCompany(json.competitors?.[0] || null);
      }
    } catch (err) {
      console.error('Failed to fetch competitor data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 flex items-center justify-center">
        <div className="text-slate-500">Loading competitor investment data...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 flex items-center justify-center">
        <div className="text-red-500">Failed to load data</div>
      </div>
    );
  }

  const aiGrowthData = data.competitors.map((c) => ({
    name: c.company,
    growth: c.ai_growth_pct,
    fill: COMPANY_COLORS[c.company] || '#64748B',
  }));

  const radarData = data.competitors.map((c) => ({
    company: c.company,
    aiGrowth: c.ai_growth_pct,
    focusAreas: c.investment_focus.length * 25,
    highlights: c.recent_highlights.length * 30,
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
              <div className="bg-gradient-to-br from-indigo-500 to-purple-600 p-2 rounded-xl">
                <Target className="h-6 w-6 text-white" />
              </div>
              Competitor Investment Monitor
            </h1>
            <p className="text-slate-500 mt-1">
              EMS Company Investment Plans, Guidance & AI Strategy
            </p>
          </div>
          <button
            onClick={fetchData}
            className="flex items-center gap-2 px-4 py-2 bg-white rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-all shadow-sm"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Hyperscaler Demand Banner */}
      <Card className="border-0 shadow-xl bg-gradient-to-r from-blue-600 to-indigo-700 mb-8">
        <CardContent className="p-6">
          <div className="flex items-start gap-6">
            <div className="bg-white/20 p-4 rounded-xl">
              <DollarSign className="h-8 w-8 text-white" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h3 className="text-xl font-bold text-white">Hyperscaler Demand Outlook</h3>
                <Badge className="bg-green-400 text-green-900">{data.hyperscaler_demand.outlook}</Badge>
              </div>
              <div className="space-y-2 mb-4">
                {data.hyperscaler_demand.drivers.map((driver, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-blue-100">
                    <ChevronRight className="h-4 w-4" />
                    <span>{driver}</span>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-blue-200 text-sm">Primary Beneficiaries:</span>
                {data.hyperscaler_demand.beneficiaries.map((company) => (
                  <Badge
                    key={company}
                    className="text-white"
                    style={{ backgroundColor: COMPANY_COLORS[company] || '#64748B' }}
                  >
                    {company}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Growth Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card className="border-0 shadow-xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
              AI Revenue Growth (YoY)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={aiGrowthData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis type="number" unit="%" domain={[0, 80]} />
                <YAxis type="category" dataKey="name" width={80} />
                <Tooltip
                  formatter={(value) => [`${value}%`, 'AI Growth']}
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 40px rgba(0,0,0,0.1)' }}
                />
                <Bar dataKey="growth" radius={[0, 8, 8, 0]}>
                  {aiGrowthData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <ChartDescription
              description="Year-over-year AI/Data Center revenue growth rates for each EMS company. Celestica leads with 68% growth driven by HPS segment expansion."
              source="Company Earnings Reports"
            />
          </CardContent>
        </Card>

        <Card className="border-0 shadow-xl">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5 text-purple-600" />
              Investment Profile Comparison
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#E2E8F0" />
                <PolarAngleAxis dataKey="company" tick={{ fill: '#64748B', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                {data.competitors.map((c) => (
                  <Radar
                    key={c.company}
                    name={c.company}
                    dataKey="aiGrowth"
                    stroke={COMPANY_COLORS[c.company]}
                    fill={COMPANY_COLORS[c.company]}
                    fillOpacity={0.2}
                  />
                ))}
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
            <ChartDescription
              description="Radar visualization comparing AI growth rates, investment focus areas, and recent activity across EMS competitors."
              source="Company Analysis"
            />
          </CardContent>
        </Card>
      </div>

      {/* Company Detail Cards */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <Building2 className="h-5 w-5 text-blue-500" />
          Company Investment Profiles
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
          {data.competitors.map((company) => (
            <button
              key={company.company}
              onClick={() => setSelectedCompany(company)}
              className={`text-left p-4 rounded-xl border-2 transition-all ${
                selectedCompany?.company === company.company
                  ? 'border-indigo-500 bg-indigo-50 shadow-lg'
                  : 'border-slate-200 bg-white hover:border-slate-300'
              }`}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold mb-2"
                style={{ backgroundColor: COMPANY_COLORS[company.company] }}
              >
                {company.company.charAt(0)}
              </div>
              <p className="font-semibold text-slate-900">{company.company}</p>
              <div className="mt-2">
                <p className="text-lg font-bold text-green-600">+{company.ai_growth_pct}%</p>
                <p className="text-xs text-slate-500">AI Growth</p>
              </div>
            </button>
          ))}
        </div>

        {/* Selected Company Details */}
        {selectedCompany && (
          <Card className="border-0 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold"
                  style={{ backgroundColor: COMPANY_COLORS[selectedCompany.company] }}
                >
                  {selectedCompany.company.charAt(0)}
                </div>
                {selectedCompany.company} - Investment Strategy
                <Badge className={`ml-auto ${getOutlookStyle(selectedCompany.guidance_outlook)}`}>
                  {selectedCompany.guidance_outlook.split('-')[0].trim()}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Guidance Outlook */}
                <div>
                  <h4 className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
                    <Lightbulb className="h-4 w-4 text-amber-500" />
                    Guidance Outlook
                  </h4>
                  <div className="p-4 bg-slate-50 rounded-xl">
                    <p className="text-slate-700">{selectedCompany.guidance_outlook}</p>
                  </div>
                </div>

                {/* Investment Focus */}
                <div>
                  <h4 className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
                    <Target className="h-4 w-4 text-indigo-500" />
                    Investment Focus Areas
                  </h4>
                  <div className="space-y-2">
                    {selectedCompany.investment_focus.map((focus, idx) => (
                      <div key={idx} className="flex items-center gap-2 p-2 bg-indigo-50 rounded-lg">
                        <div className="w-2 h-2 rounded-full bg-indigo-500"></div>
                        <span className="text-slate-700">{focus}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recent Highlights */}
                <div>
                  <h4 className="font-semibold text-slate-700 mb-3 flex items-center gap-2">
                    <ArrowUpRight className="h-4 w-4 text-green-500" />
                    Recent Highlights
                  </h4>
                  <div className="space-y-2">
                    {selectedCompany.recent_highlights.map((highlight, idx) => (
                      <div key={idx} className="p-3 bg-green-50 rounded-lg">
                        <p className="text-sm text-slate-700">{highlight}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Investment Comparison Table */}
      <Card className="border-0 shadow-xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5 text-slate-600" />
            Investment Comparison Matrix
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-3 px-4 font-semibold text-slate-600">Company</th>
                  <th className="text-left py-3 px-4 font-semibold text-slate-600">AI Growth</th>
                  <th className="text-left py-3 px-4 font-semibold text-slate-600">Focus Areas</th>
                  <th className="text-left py-3 px-4 font-semibold text-slate-600">Outlook</th>
                </tr>
              </thead>
              <tbody>
                {data.competitors.map((company) => (
                  <tr key={company.company} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-6 h-6 rounded flex items-center justify-center text-white text-xs font-bold"
                          style={{ backgroundColor: COMPANY_COLORS[company.company] }}
                        >
                          {company.company.charAt(0)}
                        </div>
                        <span className="font-medium">{company.company}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className="bg-green-100 text-green-700">+{company.ai_growth_pct}%</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1">
                        {company.investment_focus.slice(0, 2).map((focus, idx) => (
                          <span key={idx} className="text-xs px-2 py-1 bg-slate-100 rounded-full text-slate-600">
                            {focus}
                          </span>
                        ))}
                        {company.investment_focus.length > 2 && (
                          <span className="text-xs text-slate-400">+{company.investment_focus.length - 2}</span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className={getOutlookStyle(company.guidance_outlook)}>
                        {company.guidance_outlook.split('-')[0].trim()}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ChartDescription
            description="Summary of EMS competitor investment strategies derived from earnings calls, guidance, and public disclosures. Focus areas indicate primary technology investment directions."
            source="Company Earnings Calls & SEC Filings"
          />
        </CardContent>
      </Card>
    </div>
  );
}
