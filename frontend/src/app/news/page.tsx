'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  RefreshCw,
  ExternalLink,
  Rss,
  Globe2,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const COMPANY_TICKERS = ['FLEX', 'JBL', 'CLS', 'BHE', 'SANM'] as const;
type CompanyTicker = (typeof COMPANY_TICKERS)[number];

const COMPANY_NAMES: Record<CompanyTicker, string> = {
  FLEX: 'Flex',
  JBL: 'Jabil',
  CLS: 'Celestica',
  BHE: 'Benchmark',
  SANM: 'Sanmina',
};

const COMPANY_COLORS: Record<CompanyTicker, string> = {
  FLEX: '#3B82F6',
  JBL: '#10B981',
  CLS: '#6366F1',
  BHE: '#F59E0B',
  SANM: '#EF4444',
};

const COMPANY_WEBSITES: Record<CompanyTicker, string> = {
  JBL: 'https://www.jabil.com/',
  FLEX: 'https://flex.com/newsroom',
  BHE: 'https://www.bench.com/',
  SANM: 'https://www.sanmina.com/',
  CLS: 'https://www.celestica.com/',
};

const CATEGORY_LABELS: Record<string, string> = {
  earnings: 'Earnings',
  ai: 'AI',
  capex: 'CapEx',
  strategy: 'Strategy',
  operations: 'Operations',
  general: 'General',
};

const CATEGORY_TINTS: Record<string, string> = {
  earnings: 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-400/20',
  ai: 'bg-violet-50 text-violet-700 border-violet-200 dark:bg-violet-500/15 dark:text-violet-300 dark:border-violet-400/20',
  capex: 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-500/15 dark:text-orange-300 dark:border-orange-400/20',
  strategy: 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-500/15 dark:text-blue-300 dark:border-blue-400/20',
  operations: 'bg-cyan-50 text-cyan-700 border-cyan-200 dark:bg-cyan-500/15 dark:text-cyan-300 dark:border-cyan-400/20',
  general: 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-500/15 dark:text-slate-300 dark:border-slate-400/20',
};

const AI_TERMS = ['ai', 'artificial intelligence', 'llm', 'data center', 'nvidia', 'semiconductor', 'liquid cooling', 'immersion cooling', 'thermal management', 'cooling'];
const PRIORITY_TECH_TERMS = ['ai', 'artificial intelligence', 'data center'];

const HIGH_KEYWORDS = ['partnership', 'deal', 'contract', 'wins', 'customer', 'ai platform', 'hyperscaler', 'collaboration', 'acqui'];
const MEDIUM_KEYWORDS = ['expansion', 'factory', 'facility', 'earnings', 'revenue', 'guidance', 'capex', 'investment'];

const TAB_KEYWORDS: Record<string, string[]> = {
  strategic: [...HIGH_KEYWORDS, 'launch', 'introduces', 'selected', 'awarded'],
  capacity: ['factory', 'plant', 'capacity', 'expansion', 'facility', 'manufacturing', 'build', 'investment', 'capex', 'liquid cooling', 'nearshoring', 'mexico', 'india'],
  performance: ['earnings', 'revenue', 'margin', 'guidance', 'forecast', 'results', 'analyst', 'price target', 'backlog'],
};

const FALLBACK_FEED: UnifiedNewsItem[] = [
  {
    title: 'Flex expands AI infrastructure manufacturing programs',
    url: 'https://flex.com/newsroom',
    description: 'Flex reports sustained demand visibility across AI server and data-center related customer programs.',
    source: 'Flex Newsroom',
    categories: ['ai', 'operations'],
    company: 'FLEX',
    categoryLabel: 'AI',
    timestampLabel: 'Source feed',
  },
  {
    title: 'Jabil highlights cloud and AI platform momentum',
    url: 'https://www.jabil.com/about-us/news.html',
    description: 'Jabil comments on advanced compute and AI-capable infrastructure demand across enterprise and cloud customers.',
    source: 'Jabil Newsroom',
    categories: ['ai', 'strategy'],
    company: 'JBL',
    categoryLabel: 'AI',
    timestampLabel: 'Source feed',
  },
  {
    title: 'Industry watch: EMS capacity planning follows AI workload growth',
    url: 'https://www.eetimes.com/',
    description: 'Electronics manufacturing providers continue to rebalance investments around AI and data-center demand.',
    source: 'EE Times',
    categories: ['ai', 'capex'],
    categoryLabel: 'AI',
    timestampLabel: 'Source feed',
  },
];

interface NewsItem {
  title: string;
  url: string;
  backup_url?: string;
  description: string;
  image_url?: string;
  source: string;
  published?: string;
  categories: string[];
  relevance_score?: number;
}

interface UnifiedNewsItem extends NewsItem {
  company?: CompanyTicker;
  categoryLabel: string;
  timestampLabel: string;
}

type ThreatLevel = 'HIGH' | 'MEDIUM' | 'LOW';

function getThreatLevel(item: UnifiedNewsItem): ThreatLevel {
  const text = `${item.title} ${item.description}`.toLowerCase();
  if (HIGH_KEYWORDS.some((kw) => text.includes(kw))) return 'HIGH';
  if (MEDIUM_KEYWORDS.some((kw) => text.includes(kw))) return 'MEDIUM';
  return 'LOW';
}

function ThreatBadge({ level }: { level: ThreatLevel }) {
  if (level === 'HIGH') {
    return (
      <span className="shrink-0 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold bg-red-100 text-red-600 border border-red-300">
        HIGH
      </span>
    );
  }
  if (level === 'MEDIUM') {
    return (
      <span className="shrink-0 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold bg-orange-100 text-orange-600 border border-orange-300">
        MEDIUM
      </span>
    );
  }
  return (
    <span className="shrink-0 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-600 border border-green-300">
      LOW
    </span>
  );
}

function NewsCard({ item }: { item: UnifiedNewsItem }) {
  const level = getThreatLevel(item);
  return (
    <a
      href={getSafeHref(item)}
      target="_blank"
      rel="noopener noreferrer"
      className="flex gap-3 rounded-xl border border-slate-200 bg-white p-3 transition hover:border-slate-300 hover:bg-slate-50 dark:border-slate-800 dark:bg-[#0b1220] dark:hover:border-slate-700 dark:hover:bg-white/[0.03]"
    >
      <div className="pt-0.5">
        <ThreatBadge level={level} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="font-semibold leading-snug text-slate-900 dark:text-slate-100 line-clamp-2">
          {item.title}
        </p>
        <div className="mt-1 flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <Globe2 className="h-3 w-3 shrink-0" />
          <span className="truncate">{item.source}</span>
          <span>·</span>
          <span className="shrink-0">{item.timestampLabel}</span>
          {item.company && (
            <>
              <span>·</span>
              <span
                className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold text-white"
                style={{ backgroundColor: COMPANY_COLORS[item.company] }}
              >
                {item.company}
              </span>
            </>
          )}
        </div>
        {item.description && (
          <p className="mt-1.5 line-clamp-1 text-xs text-slate-600 dark:text-slate-400">
            {item.description}
          </p>
        )}
      </div>
      <ExternalLink className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" />
    </a>
  );
}

function getCategoryKey(categories?: string[]) {
  return categories?.[0] || 'general';
}

function formatPublishedLabel(raw?: string) {
  const value = (raw || '').trim();
  if (!value) return '';
  const parsed = new Date(value);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: '2-digit' });
  }
  return value;
}

function getPublishedTimestamp(raw?: string) {
  const value = (raw || '').trim();
  if (!value) return 0;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return 0;
  return parsed.getTime();
}

function getPublishedDateKey(raw?: string) {
  const ts = getPublishedTimestamp(raw);
  if (!ts) return '';
  const d = new Date(ts);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function hasPublishedTime(raw?: string) {
  const value = (raw || '').trim();
  if (!value) return false;
  return /t\d{1,2}:\d{2}/i.test(value) || /\b\d{1,2}:\d{2}(:\d{2})?\b/.test(value);
}

function getPriorityScore(item: UnifiedNewsItem) {
  const content = `${item.title} ${item.description} ${(item.categories || []).join(' ')}`.toLowerCase();
  let score = 0;
  if (item.company) {
    score += 1000;
    if (item.company === 'FLEX') score += 300;
  }
  if (PRIORITY_TECH_TERMS.some((term) => content.includes(term))) score += 200;
  score += Math.round((item.relevance_score || 0) * 100);
  return score;
}

function toUnifiedItem(item: NewsItem, company?: CompanyTicker): UnifiedNewsItem {
  const categoryKey = getCategoryKey(item.categories);
  const publishedLabel = formatPublishedLabel(item.published);
  return {
    ...item,
    company,
    categoryLabel: CATEGORY_LABELS[categoryKey] || 'General',
    timestampLabel: publishedLabel || item.source || 'News source',
  };
}

function isTrackedOrAIRelated(item: UnifiedNewsItem) {
  if (item.company) return true;
  const content = `${item.title} ${item.description} ${(item.categories || []).join(' ')}`.toLowerCase();
  if (AI_TERMS.some((term) => content.includes(term))) return true;
  return (Object.values(COMPANY_NAMES) as string[]).some((name) => content.includes(name.toLowerCase()));
}

function getDomain(url: string) {
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return '';
  }
}

function getSafeHref(item: NewsItem) {
  const blocked = ['wsj.com', 'ft.com', 'barrons.com', 'bloomberg.com', 'seekingalpha.com', 'fool.com'];
  const domain = getDomain(item.url);
  const shouldUseBackup = blocked.some((d) => domain.endsWith(d));
  if (shouldUseBackup) return item.backup_url || `https://www.google.com/search?q=${encodeURIComponent(item.title)}`;
  return item.url || item.backup_url || '#';
}

type TabId = 'strategic' | 'capacity' | 'performance' | 'fastfeed';

const TABS: { id: TabId; label: string }[] = [
  { id: 'strategic', label: 'Strategic Moves ⭐' },
  { id: 'capacity', label: 'Capacity & Execution' },
  { id: 'performance', label: 'Performance & Signals' },
  { id: 'fastfeed', label: 'Fast Feed' },
];

export default function NewsPage() {
  const [selectedCompany, setSelectedCompany] = useState<string>('ALL');
  const [keyword, setKeyword] = useState('');
  const [activeTab, setActiveTab] = useState<TabId>('strategic');
  const [companyNews, setCompanyNews] = useState<Record<CompanyTicker, NewsItem[]>>({
    FLEX: [],
    JBL: [],
    CLS: [],
    BHE: [],
    SANM: [],
  });
  const [industryNews, setIndustryNews] = useState<NewsItem[]>([]);
  const [comparativeNews, setComparativeNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAllNews = async (showLoader = false, forceRefresh = false) => {
    if (showLoader) setLoading(true);
    try {
      const forceParam = forceRefresh ? '&force_refresh=true' : '';
      const [allNewsRes, industryRes, comparativeRes] = await Promise.allSettled([
        fetch(`${API_URL}/api/news/all?count_per_company=24${forceParam}`),
        fetch(`${API_URL}/api/news/industry?count=20${forceParam}`),
        fetch(`${API_URL}/api/news/comparative${forceRefresh ? '?force_refresh=true' : ''}`),
      ]);

      if (allNewsRes.status === 'fulfilled' && allNewsRes.value.ok) {
        const data = await allNewsRes.value.json();
        const newsMap: Record<CompanyTicker, NewsItem[]> = {
          FLEX: data.companies?.FLEX?.news || [],
          JBL: data.companies?.JBL?.news || [],
          CLS: data.companies?.CLS?.news || [],
          BHE: data.companies?.BHE?.news || [],
          SANM: data.companies?.SANM?.news || [],
        };
        setCompanyNews(newsMap);
      }

      if (industryRes.status === 'fulfilled' && industryRes.value.ok) {
        const data = await industryRes.value.json();
        setIndustryNews(data.news || []);
      }

      if (comparativeRes.status === 'fulfilled' && comparativeRes.value.ok) {
        const data = await comparativeRes.value.json();
        setComparativeNews(data.comparative_news || []);
      }
    } catch (err) {
      console.error('Failed to fetch news:', err);
    } finally {
      if (showLoader) setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllNews(true);
  }, []);

  const refreshNews = async () => {
    setRefreshing(true);
    await fetchAllNews(false, true);
    setRefreshing(false);
  };

  const applyKeywordPreset = (preset: string) => {
    setKeyword((prev) => {
      const current = prev.trim().toLowerCase();
      const next = preset.trim().toLowerCase();
      return current === next ? '' : preset;
    });
  };

  const allCompanyNews = useMemo(() => {
    const merged: UnifiedNewsItem[] = [];
    for (const ticker of COMPANY_TICKERS) {
      const items = companyNews[ticker] || [];
      for (const item of items) merged.push(toUnifiedItem(item, ticker));
    }
    return merged;
  }, [companyNews]);

  const unifiedIndustryNews = useMemo(() => industryNews.map((item) => toUnifiedItem(item)), [industryNews]);
  const unifiedComparativeNews = useMemo(() => comparativeNews.map((item) => toUnifiedItem(item)), [comparativeNews]);

  const baseFilteredFeed = useMemo(() => {
    const hasKeyword = Boolean(keyword.trim());
    let feed = [...allCompanyNews, ...unifiedIndustryNews, ...unifiedComparativeNews];

    if (!hasKeyword) {
      feed = feed.filter(isTrackedOrAIRelated);
    }

    if (selectedCompany !== 'ALL') {
      feed = feed.filter((item) => item.company === selectedCompany);
    }

    if (hasKeyword) {
      const terms = keyword
        .split(',')
        .map((term) => term.trim().toLowerCase())
        .filter(Boolean);
      feed = feed.filter(
        (item) =>
          terms.some((q) =>
            item.title.toLowerCase().includes(q) ||
            item.description.toLowerCase().includes(q) ||
            (item.company ? COMPANY_NAMES[item.company].toLowerCase().includes(q) : false) ||
            item.categories?.some((cat) => cat.toLowerCase().includes(q))
          )
      );
    }

    return feed;
  }, [allCompanyNews, unifiedIndustryNews, unifiedComparativeNews, selectedCompany, keyword]);

  const prioritizedFeed = useMemo(() => {
    return [...baseFilteredFeed].sort((a, b) => {
      const priorityDelta = getPriorityScore(b) - getPriorityScore(a);
      if (priorityDelta !== 0) return priorityDelta;
      const timeDelta = getPublishedTimestamp(b.published) - getPublishedTimestamp(a.published);
      if (timeDelta !== 0) return timeDelta;
      return (b.relevance_score || 0) - (a.relevance_score || 0);
    });
  }, [baseFilteredFeed]);

  const timeSortedFeed = useMemo(() => {
    return [...baseFilteredFeed].sort((a, b) => {
      const dateA = getPublishedDateKey(a.published);
      const dateB = getPublishedDateKey(b.published);
      if (dateA !== dateB) return dateB.localeCompare(dateA);
      const aHasTime = hasPublishedTime(a.published);
      const bHasTime = hasPublishedTime(b.published);
      if (aHasTime && bHasTime) {
        const timeDelta = getPublishedTimestamp(b.published) - getPublishedTimestamp(a.published);
        if (timeDelta !== 0) return timeDelta;
      }
      return (b.relevance_score || 0) - (a.relevance_score || 0);
    });
  }, [baseFilteredFeed]);

  const shouldUseFallback = selectedCompany === 'ALL' && !keyword.trim() && prioritizedFeed.length === 0;
  const baseFeed = shouldUseFallback ? FALLBACK_FEED : prioritizedFeed;
  const fastFeedBase = shouldUseFallback ? FALLBACK_FEED : timeSortedFeed;

  const tabFeed = useMemo(() => {
    if (activeTab === 'fastfeed') return fastFeedBase;
    const keywords = TAB_KEYWORDS[activeTab] || [];
    return baseFeed.filter((item) => {
      const text = `${item.title} ${item.description}`.toLowerCase();
      return keywords.some((kw) => text.includes(kw));
    });
  }, [activeTab, baseFeed, fastFeedBase]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-100 dark:bg-[#070b12] flex items-center justify-center">
        <div className="text-center">
          <div className="mx-auto h-16 w-16 animate-spin rounded-full border-4 border-slate-300 dark:border-slate-700 border-t-cyan-500" />
          <p className="mt-4 text-sm font-medium text-slate-600 dark:text-slate-400">Loading news intelligence...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white text-slate-900 dark:bg-[#060a12] dark:text-slate-100">
      <div className="mx-auto max-w-[1720px] px-5 py-5">

        {/* ── Top bar: keyword presets + refresh ── */}
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-lg bg-cyan-50 px-3 py-1.5 text-sm text-cyan-700 dark:bg-[#0f1726] dark:text-cyan-300">News</span>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600 dark:text-slate-500">
              <span>Only 5 tracked companies +</span>
              {(['data center', 'ai', 'liquid cooling'] as const).map((preset) => (
                <button
                  key={preset}
                  onClick={() => applyKeywordPreset(preset)}
                  className={`rounded-md border px-2 py-1 transition ${
                    keyword.trim().toLowerCase() === preset
                      ? 'border-cyan-300 bg-cyan-50 text-cyan-700 dark:border-cyan-400/60 dark:bg-cyan-500/20 dark:text-cyan-200'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-slate-400 dark:border-slate-700 dark:bg-[#0b111d] dark:text-slate-400 dark:hover:border-slate-600'
                  }`}
                >
                  {preset === 'data center' ? 'Data Center' : preset === 'ai' ? 'AI' : 'Liquid Cooling'}
                </button>
              ))}
              <input
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="Search keyword…"
                className="w-[220px] rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-900 outline-none placeholder:text-slate-500 dark:border-slate-700 dark:bg-[#0b111d] dark:text-slate-100"
              />
            </div>
          </div>

          <button
            onClick={refreshNews}
            disabled={refreshing}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 transition hover:border-slate-400 disabled:opacity-50 dark:border-slate-700 dark:bg-[#0b111d] dark:text-slate-200 dark:hover:border-slate-600"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            {refreshing ? 'Refreshing' : 'Refresh'}
          </button>
        </div>

        {/* ── Company pill filter ── */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <button
            onClick={() => setSelectedCompany('ALL')}
            className={`rounded-lg px-3 py-2 text-sm ${
              selectedCompany === 'ALL'
                ? 'bg-blue-500 text-white'
                : 'bg-white text-slate-700 border border-slate-300 dark:bg-[#0b111d] dark:text-slate-300 dark:border-slate-800'
            }`}
          >
            All
          </button>
          {COMPANY_TICKERS.map((ticker) => (
            <button
              key={ticker}
              onClick={() => setSelectedCompany(ticker)}
              className={`rounded-lg px-3 py-2 text-sm ${
                selectedCompany === ticker
                  ? 'text-white'
                  : 'border border-slate-300 bg-white text-slate-700 dark:border-slate-800 dark:bg-[#0b111d] dark:text-slate-300'
              }`}
              style={selectedCompany === ticker ? { backgroundColor: COMPANY_COLORS[ticker] } : {}}
            >
              {COMPANY_NAMES[ticker]}
            </button>
          ))}
        </div>

        {/* ── Official Company Entrances ── */}
        <div className="mb-4 rounded-xl border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-[#0b1220]">
          <div className="mb-2 text-xs uppercase tracking-wide text-slate-600 dark:text-slate-500">Official Company Entrances</div>
          <div className="flex flex-wrap gap-2">
            {COMPANY_TICKERS.map((ticker) => (
              <a
                key={ticker}
                href={COMPANY_WEBSITES[ticker]}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-sm text-slate-700 transition hover:border-slate-400 dark:border-slate-700 dark:bg-[#0a101c] dark:text-slate-200 dark:hover:border-slate-600 dark:hover:text-white"
              >
                <span
                  className="inline-flex h-5 w-5 items-center justify-center rounded text-[10px] font-bold text-white"
                  style={{ backgroundColor: COMPANY_COLORS[ticker] }}
                >
                  {ticker}
                </span>
                {COMPANY_NAMES[ticker]}
                <ExternalLink className="h-3.5 w-3.5 text-slate-400 dark:text-slate-500" />
              </a>
            ))}
          </div>
        </div>

        {/* ── Four-tab main content ── */}
        <div className="rounded-2xl border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#0a101c]">

          {/* Tab bar */}
          <div className="flex gap-1 overflow-x-auto border-b border-slate-200 px-4 pt-3 dark:border-slate-800 scrollbar-none">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`shrink-0 rounded-t-lg px-4 py-2 text-sm font-medium transition whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-white/[0.04]'
                }`}
              >
                {tab.label}
                {activeTab !== tab.id && (
                  <span className="ml-1.5 rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                    {tab.id === 'fastfeed'
                      ? fastFeedBase.length
                      : (baseFeed.filter((item) => {
                          const text = `${item.title} ${item.description}`.toLowerCase();
                          return (TAB_KEYWORDS[tab.id] || []).some((kw) => text.includes(kw));
                        }).length)}
                  </span>
                )}
              </button>
            ))}
            <div className="ml-auto flex items-center pb-2 pl-2 shrink-0">
              <div className="inline-flex items-center gap-1.5 rounded-full border border-slate-300 px-2.5 py-1 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
                <Rss className="h-3 w-3 text-cyan-500" />
                {tabFeed.length} items
              </div>
            </div>
          </div>

          {/* Tab content */}
          <div className="p-4">
            {tabFeed.length === 0 ? (
              <div className="py-16 text-center text-sm text-slate-400 dark:text-slate-600">
                No news matched for this tab — try adjusting company filter or keyword
              </div>
            ) : (
              <div className="space-y-2 max-h-[700px] overflow-y-auto pr-1 xl:[&::-webkit-scrollbar]:w-1.5 xl:[&::-webkit-scrollbar-thumb]:rounded-full xl:[&::-webkit-scrollbar-thumb]:bg-slate-300 dark:xl:[&::-webkit-scrollbar-thumb]:bg-slate-700 xl:[&::-webkit-scrollbar-track]:bg-transparent">
                {tabFeed.map((item, idx) => (
                  <NewsCard key={`${item.title}-${idx}`} item={item} />
                ))}
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
