'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  FileText,
  Download,
  FileSpreadsheet,
  Presentation,
  Building2,
  CheckCircle,
  XCircle,
  Eye,
  Loader2,
  Clock,
  Zap,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const COMPANIES = ['Flex', 'Jabil', 'Celestica', 'Benchmark', 'Sanmina'];

const COMPANY_COLORS: Record<string, string> = {
  Flex:      '#3B82F6',
  Jabil:     '#10B981',
  Celestica: '#6366F1',
  Benchmark: '#F59E0B',
  Sanmina:   '#EF4444',
};

const SLIDE_OUTLINE = [
  'Executive Summary',
  'AI Infrastructure Demand — Market Context',
  'AI/DC Competitive Ranking',
  'Capacity & Geographic Footprint',
  'Financial Performance & Growth Momentum',
  'Hyperscaler Customer Intelligence',
  'Strategic Moves & Announcements',
  'Risk Landscape',
  'Flex Competitive Position',
  'Recommended Next Steps',
];

interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  available: boolean;
  description: string;
  native_pdf?: boolean;
}

type ReportType = 'intelligence' | 'raw';

function getFormatIcon(format: string) {
  switch (format) {
    case 'excel':      return <FileSpreadsheet className="h-5 w-5 text-green-600" />;
    case 'powerpoint': return <Presentation className="h-5 w-5 text-orange-600" />;
    default:           return <FileText className="h-5 w-5 text-blue-600" />;
  }
}

function formatDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });
}

export default function ReportsPage() {
  const [formats, setFormats]               = useState<ExportFormat[]>([]);
  const [reportType, setReportType]         = useState<ReportType>('intelligence');
  const [selectedCompany, setSelectedCompany] = useState<string>('all');
  const [selectedFormat, setSelectedFormat] = useState<string>('powerpoint');
  const [loading, setLoading]               = useState(true);
  const [downloading, setDownloading]       = useState<string | null>(null);
  const [downloadError, setDownloadError]   = useState<string | null>(null);
  const [recentExports, setRecentExports]   = useState<{ company: string; format: string; timestamp: string }[]>([]);
  const [outlineOpen, setOutlineOpen]       = useState(false);

  useEffect(() => {
    fetchFormats();
    loadRecentExports();
  }, []);

  // When report type changes, reset format to sensible default
  useEffect(() => {
    setSelectedFormat(reportType === 'raw' ? 'excel' : 'powerpoint');
  }, [reportType]);

  const fetchFormats = async () => {
    try {
      const res = await fetch(`${API_URL}/api/exports/formats`);
      if (res.ok) {
        const data = await res.json();
        setFormats(data.formats);
      }
    } catch (err) {
      console.error('Failed to fetch formats:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadRecentExports = () => {
    const stored = localStorage.getItem('recentExports');
    if (stored) {
      try { setRecentExports(JSON.parse(stored).slice(0, 5)); } catch {}
    }
  };

  const saveRecentExport = (company: string, format: string) => {
    const newExport = { company, format, timestamp: new Date().toISOString() };
    const updated = [newExport, ...recentExports].slice(0, 5);
    setRecentExports(updated);
    localStorage.setItem('recentExports', JSON.stringify(updated));
  };

  const downloadReport = async (format: string, company: string) => {
    const key = `${format}-${company}`;
    setDownloading(key);
    setDownloadError(null);
    try {
      const endpoint = company === 'all'
        ? `${API_URL}/api/exports/${format}/comparison/all`
        : `${API_URL}/api/exports/${format}/${company.toLowerCase()}`;

      const res = await fetch(endpoint);
      if (!res.ok) {
        let detail = `Server error ${res.status}`;
        try {
          const body = await res.json();
          detail = body.detail || detail;
        } catch {}
        throw new Error(detail);
      }

      const blob = await res.blob();
      const contentDisposition = res.headers.get('Content-Disposition');
      let filename = `report.${format === 'excel' ? 'xlsx' : 'pptx'}`;
      if (contentDisposition) {
        const match = contentDisposition.match(/filename=(.+)/);
        if (match) filename = match[1];
      }

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      saveRecentExport(company, format);
    } catch (err: any) {
      console.error('Download failed:', err);
      setDownloadError(err.message || 'Export failed. Please try again.');
    } finally {
      setDownloading(null);
    }
  };

  const openPreview = (company: string) => {
    const endpoint = company === 'all'
      ? `${API_URL}/api/exports/preview/comparison/all`
      : `${API_URL}/api/exports/preview/${company.toLowerCase()}`;
    window.open(endpoint, '_blank');
  };

  // Only show formats that are available; hide PDF entirely if unavailable
  const visibleFormats = formats
    .filter(f => f.id !== 'html')
    .filter(f => {
      if (f.id === 'pdf') return f.available; // hide if not installed
      return true;
    })
    .filter(f => {
      // For raw export, only show excel
      if (reportType === 'raw') return f.id === 'excel';
      // For intelligence brief, show excel + powerpoint (not pdf)
      return f.id === 'excel' || f.id === 'powerpoint';
    });

  const scopeLabel = selectedCompany === 'all' ? 'All Companies' : selectedCompany;
  const reportTypeLabel = reportType === 'intelligence' ? 'Competitive Intelligence Brief' : 'Raw Data Export';
  const formatLabel = selectedFormat === 'excel' ? 'Excel' : selectedFormat === 'powerpoint' ? 'PowerPoint' : 'PDF';
  const isAIReport = reportType === 'intelligence' && selectedFormat !== 'excel';
  const estimatedTime = isAIReport ? '~30 seconds' : 'Instant';

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-200 border-t-blue-600 mx-auto" />
          <p className="text-slate-600 dark:text-slate-400 mt-4 font-medium">Loading export options...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 p-6">

      {/* Header */}
      <div className="mb-8 flex items-center gap-4">
        <div className="bg-gradient-to-br from-blue-600 to-indigo-700 p-3 rounded-xl shadow-lg shadow-blue-500/20">
          <FileText className="h-6 w-6 text-white" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white">Intelligence Reports</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-0.5 text-sm">
            Generate AI-powered competitive briefings for Flex leadership
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── LEFT: Report Builder ── */}
        <div className="lg:col-span-2 space-y-5">

          {/* 1. Report Type */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5 text-purple-500" />
                Select Report Type
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* Card A */}
                <button
                  onClick={() => setReportType('intelligence')}
                  className={`p-4 rounded-xl border-2 text-left transition-all ${
                    reportType === 'intelligence'
                      ? 'border-purple-400 bg-purple-50 dark:border-purple-600 dark:bg-purple-900/20'
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl leading-none mt-0.5">📊</span>
                    <div className="flex-1">
                      <p className="font-semibold text-slate-900 dark:text-white text-sm">Competitive Intelligence Brief</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                        AI-generated 10-slide exec briefing with threat levels, rankings, and Flex implications
                      </p>
                      <span className="inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300">
                        AI Generated · ~30 seconds
                      </span>
                    </div>
                  </div>
                </button>

                {/* Card B */}
                <button
                  onClick={() => setReportType('raw')}
                  className={`p-4 rounded-xl border-2 text-left transition-all ${
                    reportType === 'raw'
                      ? 'border-green-400 bg-green-50 dark:border-green-600 dark:bg-green-900/20'
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xl leading-none mt-0.5">📋</span>
                    <div className="flex-1">
                      <p className="font-semibold text-slate-900 dark:text-white text-sm">Raw Data Export</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">
                        Structured data tables for self-analysis
                      </p>
                      <span className="inline-block mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300">
                        Instant
                      </span>
                    </div>
                  </div>
                </button>
              </div>
            </CardContent>
          </Card>

          {/* 2. Scope */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Building2 className="h-5 w-5 text-blue-600" />
                Select Scope
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {/* All Companies */}
                <button
                  onClick={() => setSelectedCompany('all')}
                  className={`p-4 rounded-xl border-2 transition-all col-span-full md:col-span-1 ${
                    selectedCompany === 'all'
                      ? 'border-blue-500 bg-blue-50 dark:border-blue-600 dark:bg-blue-900/20'
                      : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                      All
                    </div>
                    <div className="text-left">
                      <p className="font-semibold text-slate-900 dark:text-white text-sm">All Companies</p>
                      <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5 leading-tight">
                        Recommended — includes cross-company comparison slides
                      </p>
                    </div>
                  </div>
                </button>

                {COMPANIES.map((company) => {
                  const color = COMPANY_COLORS[company] || '#64748b';
                  const isSelected = selectedCompany === company;
                  return (
                    <button
                      key={company}
                      onClick={() => setSelectedCompany(company)}
                      className={`p-4 rounded-xl border-2 transition-all ${
                        isSelected
                          ? 'border-blue-500 bg-blue-50 dark:border-blue-600 dark:bg-blue-900/20'
                          : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold text-sm flex-shrink-0"
                          style={{ backgroundColor: color }}
                        >
                          {company.charAt(0)}
                        </div>
                        <div className="text-left">
                          <p className="font-semibold text-slate-900 dark:text-white text-sm">{company}</p>
                          <p className="text-xs text-slate-400 dark:text-slate-500">Individual report</p>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Individual company note */}
              {selectedCompany !== 'all' && (
                <p className="mt-3 text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 rounded-lg px-3 py-2 border border-slate-100 dark:border-slate-700">
                  ℹ️ Individual report focuses on <span className="font-semibold" style={{ color: COMPANY_COLORS[selectedCompany] }}>{selectedCompany}</span> vs peer benchmarks
                </p>
              )}
            </CardContent>
          </Card>

          {/* 3. Format */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Download className="h-5 w-5 text-green-600" />
                Select Format
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {visibleFormats.map((format) => {
                  const isSelected = selectedFormat === format.id;
                  const desc = format.id === 'excel'
                    ? 'Raw data tables, trend charts, comparison matrices'
                    : format.id === 'powerpoint'
                    ? 'Executive slides with insights and recommendations'
                    : format.description;
                  return (
                    <button
                      key={format.id}
                      onClick={() => setSelectedFormat(format.id)}
                      disabled={!format.available}
                      className={`p-4 rounded-xl border-2 transition-all text-left ${
                        isSelected
                          ? 'border-green-500 bg-green-50 dark:border-green-600 dark:bg-green-900/20'
                          : format.available
                          ? 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                          : 'border-slate-100 bg-slate-50 opacity-50 cursor-not-allowed'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        {getFormatIcon(format.id)}
                        <div className="flex-1">
                          <p className="font-semibold text-slate-900 dark:text-white text-sm">{format.name}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 leading-relaxed">{desc}</p>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {/* 4. Report Preview (slide outline) */}
          {reportType === 'intelligence' && selectedFormat === 'powerpoint' && (
            <Card className="border-0 shadow-xl dark:bg-slate-800/60">
              <button
                type="button"
                className="w-full flex items-center justify-between px-6 py-4 text-left"
                onClick={() => setOutlineOpen((v) => !v)}
              >
                <div>
                  <p className="font-semibold text-slate-900 dark:text-white text-sm">Report Contents</p>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">
                    10-slide competitive intelligence briefing
                  </p>
                </div>
                <span className="flex items-center gap-1 text-xs font-medium text-slate-500 dark:text-slate-400">
                  {outlineOpen ? 'Hide' : 'Show slide outline'}
                  {outlineOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </span>
              </button>
              {outlineOpen && (
                <CardContent className="pt-0">
                  <ol className="space-y-1.5">
                    {SLIDE_OUTLINE.map((title, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm text-slate-600 dark:text-slate-300">
                        <span className="flex-shrink-0 font-bold text-slate-300 dark:text-slate-600 w-5 text-right">
                          {i + 1}.
                        </span>
                        <span>{title}</span>
                      </li>
                    ))}
                  </ol>
                </CardContent>
              )}
            </Card>
          )}

          {/* Error banner */}
          {downloadError && (
            <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-800">
              <XCircle className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
              <span className="flex-1">{downloadError}</span>
              <button onClick={() => setDownloadError(null)} className="text-red-400 hover:text-red-700">
                ✕
              </button>
            </div>
          )}

          {/* 5. Download bar */}
          <Card className="border-0 shadow-xl bg-gradient-to-r from-blue-600 to-indigo-700">
            <CardContent className="p-5">
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div className="text-white">
                  <h3 className="font-semibold text-base leading-tight">
                    {reportTypeLabel} · {scopeLabel} · {formatLabel}
                  </h3>
                  <p className="text-blue-100 text-xs mt-0.5 flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    ⏱ {estimatedTime}
                  </p>
                </div>
                <div className="flex gap-2 flex-shrink-0">
                  <button
                    onClick={() => openPreview(selectedCompany)}
                    className="px-4 py-2 bg-white/20 text-white rounded-xl hover:bg-white/30 transition-all flex items-center gap-2 text-sm"
                  >
                    <Eye className="h-4 w-4" />
                    Preview
                  </button>
                  <button
                    onClick={() => downloadReport(selectedFormat, selectedCompany)}
                    disabled={downloading !== null}
                    className="px-5 py-2 bg-white text-blue-600 rounded-xl hover:bg-blue-50 transition-all flex items-center gap-2 font-semibold text-sm disabled:opacity-50"
                  >
                    {downloading === `${selectedFormat}-${selectedCompany}` ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Download className="h-4 w-4" />
                    )}
                    Download
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ── RIGHT: Sidebar ── */}
        <div className="space-y-5">

          {/* Quick Export */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Zap className="h-5 w-5 text-yellow-500" />
                Quick Export
              </CardTitle>
              <p className="text-xs text-slate-400 dark:text-slate-500">Individual Company Reports</p>
            </CardHeader>
            <CardContent className="space-y-2">
              {COMPANIES.map((company) => {
                const color = COMPANY_COLORS[company] || '#64748b';
                return (
                  <div key={company} className="flex items-center justify-between p-2.5 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
                        style={{ backgroundColor: color }}
                      >
                        {company[0]}
                      </div>
                      <span className="font-medium text-sm text-slate-700 dark:text-slate-200">{company}</span>
                    </div>
                    <div className="flex gap-1.5">
                      <button
                        onClick={() => downloadReport('excel', company)}
                        disabled={downloading !== null}
                        className="p-1.5 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-lg hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
                        title="Download Excel"
                      >
                        <FileSpreadsheet className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => downloadReport('powerpoint', company)}
                        disabled={downloading !== null}
                        className="p-1.5 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 rounded-lg hover:bg-orange-200 dark:hover:bg-orange-900/50 transition-colors"
                        title="Download PowerPoint"
                      >
                        <Presentation className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                );
              })}
              <button
                onClick={() => downloadReport('powerpoint', 'all')}
                disabled={downloading !== null}
                className="w-full mt-1 p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors font-semibold text-sm flex items-center justify-center gap-2 shadow-md shadow-blue-500/20 disabled:opacity-50"
              >
                <Download className="h-4 w-4" />
                Generate Full Competitive Brief (All Companies)
              </button>
            </CardContent>
          </Card>

          {/* Recent Exports */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Clock className="h-5 w-5 text-slate-500" />
                Recent Exports
              </CardTitle>
            </CardHeader>
            <CardContent>
              {recentExports.length > 0 ? (
                <div className="space-y-2">
                  {recentExports.map((exp, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2.5 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                      <div className="flex items-center gap-2">
                        {getFormatIcon(exp.format)}
                        <div>
                          <p className="font-medium text-sm text-slate-800 dark:text-slate-200">
                            {exp.company === 'all' ? 'All Companies' : exp.company}
                          </p>
                          <p className="text-xs text-slate-400">{formatDate(exp.timestamp)}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => downloadReport(exp.format, exp.company)}
                        disabled={downloading !== null}
                        className="p-1.5 text-slate-400 hover:text-blue-600 transition-colors"
                        title="Re-download"
                      >
                        <Download className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-5">
                  <Clock className="h-6 w-6 text-slate-300 dark:text-slate-600 mx-auto mb-2" />
                  <p className="text-sm text-slate-500 dark:text-slate-400">No recent exports</p>
                  <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Your last 5 generated reports will appear here</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Library Status */}
          <Card className="border-0 shadow-xl dark:bg-slate-800/60">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Library Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {formats.filter(f => f.id !== 'html').map((format) => (
                <div key={format.id} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {getFormatIcon(format.id)}
                    <span className="text-sm text-slate-700 dark:text-slate-300">{format.name}</span>
                  </div>
                  {format.available ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-red-500" />
                  )}
                </div>
              ))}
              <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-xs text-blue-700 dark:text-blue-300">Install missing libraries:</p>
                <pre className="mt-1 text-xs bg-white dark:bg-slate-800 p-2 rounded overflow-x-auto text-slate-700 dark:text-slate-300">
                  pip install openpyxl python-pptx weasyprint
                </pre>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
