'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Newspaper,
  Building2,
  Globe,
  ExternalLink,
  RefreshCw,
  Server,
  Rss,
  Clock,
  Tag,
  Filter,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

interface PressRelease {
  company: string;
  title: string;
  date: string;
  url: string;
  summary: string;
  category: string;
}

interface OCPNews {
  title: string;
  date: string;
  url: string;
  relevance: string;
  companies_mentioned: string[];
}

interface IndustryNews {
  title: string;
  source: string;
  date: string;
  url: string;
  summary: string;
  relevance: string;
}

interface NewsData {
  press_releases: PressRelease[];
  ocp_news: OCPNews[];
  industry_news: IndustryNews[];
}

const COMPANY_COLORS: Record<string, string> = {
  'Flex': '#3B82F6',
  'Jabil': '#10B981',
  'Celestica': '#6366F1',
  'Benchmark': '#F59E0B',
  'Sanmina': '#EF4444',
};

export default function NewsMonitorPage() {
  const [data, setData] = useState<NewsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'all' | 'press' | 'ocp' | 'industry'>('all');
  const [companyFilter, setCompanyFilter] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/intelligence/news/all`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
      }
    } catch (err) {
      console.error('Failed to fetch news:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 flex items-center justify-center">
        <div className="text-slate-500">Loading news feed...</div>
      </div>
    );
  }

  const tabs = [
    { id: 'all', label: 'All News', icon: Newspaper },
    { id: 'press', label: 'Press Releases', icon: Building2 },
    { id: 'ocp', label: 'OCP / Standards', icon: Server },
    { id: 'industry', label: 'Industry News', icon: Globe },
  ];

  const companies = ['Flex', 'Jabil', 'Celestica', 'Benchmark', 'Sanmina'];

  const filteredPressReleases = data?.press_releases?.filter(
    (pr) => !companyFilter || pr.company === companyFilter
  ) || [];

  const filteredOCPNews = data?.ocp_news?.filter(
    (news) => !companyFilter || news.companies_mentioned.includes(companyFilter)
  ) || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
              <div className="bg-gradient-to-br from-blue-500 to-indigo-600 p-2 rounded-xl">
                <Newspaper className="h-6 w-6 text-white" />
              </div>
              News Monitor
            </h1>
            <p className="text-slate-500 mt-1">
              Press Releases, OCP Updates, and Industry AI News
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

      {/* Tabs and Filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        {/* Tabs */}
        <div className="flex gap-2 bg-white p-1 rounded-xl border border-slate-200 shadow-sm">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white shadow-md'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Company Filter */}
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-slate-400" />
          <div className="flex gap-1">
            <button
              onClick={() => setCompanyFilter(null)}
              className={`px-3 py-1 rounded-lg text-sm transition-all ${
                !companyFilter ? 'bg-slate-900 text-white' : 'bg-white text-slate-600 hover:bg-slate-100'
              }`}
            >
              All
            </button>
            {companies.map((company) => (
              <button
                key={company}
                onClick={() => setCompanyFilter(companyFilter === company ? null : company)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${
                  companyFilter === company ? 'text-white' : 'bg-white text-slate-600 hover:bg-slate-100'
                }`}
                style={companyFilter === company ? { backgroundColor: COMPANY_COLORS[company] } : {}}
              >
                {company}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* News Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Press Releases */}
        {(activeTab === 'all' || activeTab === 'press') && (
          <div className={activeTab === 'all' ? '' : 'lg:col-span-3'}>
            <Card className="border-0 shadow-xl h-full">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Building2 className="h-5 w-5 text-blue-600" />
                  Company Press Releases
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {filteredPressReleases.length > 0 ? (
                  filteredPressReleases.map((pr, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge
                              style={{ backgroundColor: COMPANY_COLORS[pr.company], color: 'white' }}
                            >
                              {pr.company}
                            </Badge>
                            <Badge variant="outline">{pr.category}</Badge>
                          </div>
                          <h4 className="font-semibold text-slate-900 mb-1">{pr.title}</h4>
                          <p className="text-sm text-slate-600 mb-2">{pr.summary}</p>
                          <div className="flex items-center gap-2 text-xs text-slate-400">
                            <Clock className="h-3 w-3" />
                            {pr.date}
                          </div>
                        </div>
                        <a
                          href={pr.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 text-slate-400 hover:text-blue-500 hover:bg-blue-50 rounded-lg transition-colors"
                        >
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500 text-center py-4">No press releases found</p>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* OCP News */}
        {(activeTab === 'all' || activeTab === 'ocp') && (
          <div className={activeTab === 'all' ? '' : 'lg:col-span-3'}>
            <Card className="border-0 shadow-xl h-full">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5 text-green-600" />
                  OCP / Open Standards
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {filteredOCPNews.length > 0 ? (
                  filteredOCPNews.map((news, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-green-50 rounded-xl hover:bg-green-100 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <h4 className="font-semibold text-slate-900 mb-2">{news.title}</h4>
                          <p className="text-sm text-slate-600 mb-2">{news.relevance}</p>
                          {news.companies_mentioned.length > 0 && (
                            <div className="flex items-center gap-2 mb-2">
                              <Tag className="h-3 w-3 text-slate-400" />
                              <div className="flex gap-1">
                                {news.companies_mentioned.map((company) => (
                                  <Badge
                                    key={company}
                                    variant="outline"
                                    className="text-xs"
                                  >
                                    {company}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                          <div className="flex items-center gap-2 text-xs text-slate-400">
                            <Clock className="h-3 w-3" />
                            {news.date}
                          </div>
                        </div>
                        <a
                          href={news.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 text-slate-400 hover:text-green-500 hover:bg-green-100 rounded-lg transition-colors"
                        >
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-slate-500 text-center py-4">No OCP news found</p>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Industry News */}
        {(activeTab === 'all' || activeTab === 'industry') && (
          <div className={activeTab === 'all' ? '' : 'lg:col-span-3'}>
            <Card className="border-0 shadow-xl h-full">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5 text-purple-600" />
                  Industry AI News
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {data?.industry_news?.map((news, idx) => (
                  <div
                    key={idx}
                    className="p-4 bg-purple-50 rounded-xl hover:bg-purple-100 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge className="bg-purple-100 text-purple-700">{news.source}</Badge>
                        </div>
                        <h4 className="font-semibold text-slate-900 mb-2">{news.title}</h4>
                        <p className="text-sm text-slate-600 mb-2">{news.summary}</p>
                        <p className="text-xs text-purple-600 mb-2">
                          Relevance: {news.relevance}
                        </p>
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                          <Clock className="h-3 w-3" />
                          {news.date}
                        </div>
                      </div>
                      <a
                        href={news.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 text-slate-400 hover:text-purple-500 hover:bg-purple-100 rounded-lg transition-colors"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    </div>
                  </div>
                )) || (
                  <p className="text-slate-500 text-center py-4">No industry news found</p>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Data Sources */}
      <Card className="border-0 shadow-lg mt-6">
        <CardContent className="p-4">
          <div className="flex items-center gap-4 flex-wrap">
            <span className="text-sm text-slate-500 flex items-center gap-2">
              <Rss className="h-4 w-4" />
              Data Sources:
            </span>
            <a
              href="https://www.opencompute.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:underline flex items-center gap-1"
            >
              Open Compute Project <ExternalLink className="h-3 w-3" />
            </a>
            <span className="text-slate-300">|</span>
            <span className="text-sm text-slate-500">Company IR Websites</span>
            <span className="text-slate-300">|</span>
            <span className="text-sm text-slate-500">SEC EDGAR</span>
            <span className="text-slate-300">|</span>
            <a
              href="https://futurumgroup.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:underline flex items-center gap-1"
            >
              Futurum Research <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
