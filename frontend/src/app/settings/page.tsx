'use client';

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Settings,
  Bell,
  Mail,
  MessageSquare,
  Database,
  RefreshCw,
  Save,
  Check,
  X,
  Zap,
  Shield,
  Clock,
  Server,
  ExternalLink,
  AlertTriangle,
  Newspaper,
  BarChart3,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

interface SystemStatus {
  database: { documents: number; collections: number };
  scheduler: { running: boolean; jobs: number };
  api: { connected: boolean };
}

interface SchedulerJob {
  id: string;
  name: string;
  trigger: string;
  friendly_schedule: string;
  next_run: string | null;
  last_run: string | null;
  status: 'pending' | 'stopped' | 'idle';
}

interface IngestionStatus {
  scheduler: { running: boolean; jobs: SchedulerJob[] };
  downloads: { total_downloaded: number; by_company: Record<string, number>; by_form: Record<string, number> };
}

interface NotificationSettings {
  emailEnabled: boolean;
  slackEnabled: boolean;
  emailAddress: string;
  slackChannel: string;
  alertTypes: {
    capex: boolean;
    sentiment: boolean;
    ai_investment: boolean;
    new_filing: boolean;
    strategic: boolean;
    hyperscaler_deal: boolean;
    facility_capacity: boolean;
  };
  digestFrequency: 'realtime' | 'daily' | 'weekly';
  minSeverity: 'low' | 'medium' | 'high' | 'critical';
}

type TabId = 'preferences' | 'data' | 'alerts-config';
type CheckPhase = 'edgar' | 'filings' | 'downloading' | 'processing' | 'indexing';

const CHECK_PHASES: { id: CheckPhase; label: string }[] = [
  { id: 'edgar',       label: 'Connecting to EDGAR' },
  { id: 'filings',     label: 'Checking filings' },
  { id: 'downloading', label: 'Downloading new filings' },
  { id: 'processing',  label: 'Processing documents' },
  { id: 'indexing',    label: 'Adding to knowledge base' },
];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabId>('preferences');
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);
  const [notificationConfig, setNotificationConfig] = useState<any>(null);
  const [settings, setSettings] = useState<NotificationSettings>({
    emailEnabled: false,
    slackEnabled: false,
    emailAddress: '',
    slackChannel: '#competitive-intel',
    alertTypes: {
      capex: true,
      sentiment: true,
      ai_investment: true,
      new_filing: false,
      strategic: true,
      hyperscaler_deal: true,
      facility_capacity: true,
    },
    digestFrequency: 'daily',
    minSeverity: 'medium',
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [newsFeedStatus, setNewsFeedStatus] = useState<'fresh' | 'stale' | 'unavailable'>('unavailable');
  const [lastAnalysisRun, setLastAnalysisRun] = useState<string | null>(null);
  const [checkingFilings, setCheckingFilings] = useState(false);
  const [checkPhase, setCheckPhase] = useState<CheckPhase | null>(null);
  const [filingResult, setFilingResult] = useState<{
    type: 'success' | 'error';
    newFilings: number;
    newChunks: number;
    message: string;
  } | null>(null);
  const [dataMessage, setDataMessage] = useState<string | null>(null);
  const [dataError, setDataError] = useState<string | null>(null);
  const abortCheck = useRef(false);

  // Knowledge Base state
  const [kbStatus, setKbStatus] = useState<{
    doc_count: number;
    last_indexed: string | null;
    total_local_files: number;
    chunks_at_last_index: number;
  } | null>(null);
  const [reindexing, setReindexing] = useState(false);
  const [reindexProgress, setReindexProgress] = useState<{
    processed: number;
    total: number;
    current_file: string;
    chunks_added: number;
    status: string;
  } | null>(null);
  const [reindexResult, setReindexResult] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const reindexPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetchStatus();
    fetchConfig();
    loadLocalSettings();
    fetchIngestionStatus();
    checkNewsFeedFreshness();
    fetchKbStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const [statsRes, healthRes] = await Promise.allSettled([
        fetch(`${API_URL}/api/analysis/overview`),
        fetch(`${API_URL}/`),
      ]);

      let systemStatus: SystemStatus = {
        database: { documents: 0, collections: 0 },
        scheduler: { running: true, jobs: 2 },
        api: { connected: false },
      };

      if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        const data = await statsRes.value.json();
        systemStatus.database = { documents: data.total_documents || 0, collections: 5 };
        const ts = data.last_updated || data.last_analysis_run;
        if (ts) setLastAnalysisRun(ts);
      }

      if (healthRes.status === 'fulfilled') {
        systemStatus.api = { connected: healthRes.value.ok };
      }

      setStatus(systemStatus);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchIngestionStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/ingestion/status`);
      if (res.ok) setIngestionStatus(await res.json());
    } catch {}
  };

  const checkNewsFeedFreshness = async () => {
    try {
      const res = await fetch(`${API_URL}/api/news?limit=1`);
      if (!res.ok) { setNewsFeedStatus('unavailable'); return; }
      const data = await res.json();
      const articles = data.articles || data.news || [];
      if (articles.length === 0) { setNewsFeedStatus('unavailable'); return; }
      const latestDate = new Date(articles[0].published_at || articles[0].date || '');
      const diffH = (Date.now() - latestDate.getTime()) / 3600000;
      setNewsFeedStatus(diffH < 24 ? 'fresh' : diffH < 72 ? 'stale' : 'unavailable');
    } catch {
      setNewsFeedStatus('unavailable');
    }
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch(`${API_URL}/api/alerts/config`);
      if (res.ok) setNotificationConfig(await res.json());
    } catch {}
  };

  const loadLocalSettings = () => {
    const stored = localStorage.getItem('notificationSettings');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSettings(prev => ({
          ...prev,
          ...parsed,
          alertTypes: { ...prev.alertTypes, ...(parsed.alertTypes || {}) },
        }));
      } catch {}
    }
  };

  const saveSettings = async () => {
    setSaving(true);
    try {
      localStorage.setItem('notificationSettings', JSON.stringify(settings));
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const updateSettings = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const updateAlertType = (type: string, enabled: boolean) => {
    setSettings(prev => ({ ...prev, alertTypes: { ...prev.alertTypes, [type]: enabled } }));
    setSaved(false);
  };

  const regenerateAllAnalysis = async () => {
    setRegenerating(true);
    try {
      await fetch(`${API_URL}/api/analysis/regenerate`, { method: 'POST' });
      await fetchStatus();
    } catch {}
    finally {
      setRegenerating(false);
    }
  };

  const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

  const checkForFilings = async () => {
    const beforeDownloads = ingestionStatus?.downloads.total_downloaded ?? 0;
    const beforeDocs = status?.database.documents ?? 0;

    abortCheck.current = false;
    setCheckingFilings(true);
    setCheckPhase('edgar');
    setFilingResult(null);
    setDataMessage(null);
    setDataError(null);

    try {
      await sleep(1200);
      if (abortCheck.current) return;
      setCheckPhase('filings');

      const res = await fetch(`${API_URL}/api/ingestion/check-filings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days_back: 30, filing_types: ['10-K', '10-Q', '8-K'] }),
      });
      if (!res.ok) {
        let detail = `Server error ${res.status}`;
        try { detail = (await res.json()).detail || detail; } catch {}
        throw new Error(detail);
      }

      await sleep(2000);
      if (abortCheck.current) return;
      setCheckPhase('downloading');

      // Poll for download count changes (up to 60 s)
      let newFilings = 0;
      const dlStart = Date.now();
      while (Date.now() - dlStart < 60000) {
        await sleep(3000);
        if (abortCheck.current) return;
        try {
          const statusRes = await fetch(`${API_URL}/api/ingestion/status`);
          if (statusRes.ok) {
            const data: IngestionStatus = await statusRes.json();
            setIngestionStatus(data);
            const nowTotal = data.downloads.total_downloaded;
            if (nowTotal > beforeDownloads) {
              newFilings = nowTotal - beforeDownloads;
              break;
            }
          }
        } catch {}
        // After 18 s with no change, assume nothing new
        if (Date.now() - dlStart > 18000) break;
      }

      if (abortCheck.current) return;
      setCheckPhase('processing');
      await sleep(2500);
      if (abortCheck.current) return;
      setCheckPhase('indexing');

      // Poll for vector DB doc count changes (up to 30 s), only if new filings found
      let newChunks = 0;
      if (newFilings > 0) {
        const idxStart = Date.now();
        while (Date.now() - idxStart < 30000) {
          await sleep(4000);
          if (abortCheck.current) return;
          try {
            const statsRes = await fetch(`${API_URL}/api/analysis/overview`);
            if (statsRes.ok) {
              const data = await statsRes.json();
              const nowDocs = data.total_documents || 0;
              if (nowDocs > beforeDocs) {
                newChunks = nowDocs - beforeDocs;
                // Refresh System Status panel
                setStatus(prev => prev ? { ...prev, database: { ...prev.database, documents: nowDocs } } : prev);
                break;
              }
            }
          } catch {}
        }
      }

      await fetchIngestionStatus();

      const result = {
        type: 'success' as const,
        newFilings,
        newChunks,
        message: newFilings > 0
          ? `${newFilings} new filing${newFilings > 1 ? 's' : ''} downloaded${newChunks > 0 ? ` · ${newChunks.toLocaleString()} chunks indexed` : ''}`
          : 'Up to date — no new filings found',
      };
      setFilingResult(result);
      // Auto-dismiss success after 5 s
      setTimeout(() => setFilingResult(r => r?.type === 'success' ? null : r), 5000);

    } catch (err: any) {
      setFilingResult({
        type: 'error',
        newFilings: 0,
        newChunks: 0,
        message: err.message || 'EDGAR check failed — please try again',
      });
    } finally {
      setCheckPhase(null);
      setCheckingFilings(false);
    }
  };

  const fetchKbStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/ingestion/kb-status`);
      if (res.ok) setKbStatus(await res.json());
    } catch {}
  };

  const startReindex = async () => {
    setReindexing(true);
    setReindexResult(null);
    setReindexProgress({ processed: 0, total: 0, current_file: '', chunks_added: 0, status: 'running' });

    try {
      const res = await fetch(`${API_URL}/api/ingestion/reindex-all`, { method: 'POST' });
      if (!res.ok) {
        let detail = `Server error ${res.status}`;
        try { detail = (await res.json()).detail || detail; } catch {}
        throw new Error(detail);
      }
      const init = await res.json();
      setReindexProgress(p => p ? { ...p, total: init.total_files } : p);

      // Poll progress every 2 s
      reindexPollRef.current = setInterval(async () => {
        try {
          const progRes = await fetch(`${API_URL}/api/ingestion/reindex-progress`);
          if (!progRes.ok) return;
          const prog = await progRes.json();
          setReindexProgress({
            processed: prog.processed,
            total: prog.total,
            current_file: prog.current_file,
            chunks_added: prog.chunks_added,
            status: prog.status,
          });

          if (prog.status === 'done' || prog.status === 'error') {
            clearInterval(reindexPollRef.current!);
            reindexPollRef.current = null;
            setReindexing(false);
            if (prog.status === 'done') {
              setReindexResult({
                type: 'success',
                message: `Indexed ${prog.processed} document${prog.processed !== 1 ? 's' : ''} — ${prog.chunks_added.toLocaleString()} chunks added to knowledge base`,
              });
              await fetchKbStatus();
              await fetchStatus();
              setTimeout(() => setReindexResult(r => r?.type === 'success' ? null : r), 8000);
            } else {
              setReindexResult({ type: 'error', message: prog.error || 'Re-index failed' });
            }
            setReindexProgress(null);
          }
        } catch {}
      }, 2000);

    } catch (err: any) {
      clearInterval(reindexPollRef.current!);
      reindexPollRef.current = null;
      setReindexing(false);
      setReindexProgress(null);
      setReindexResult({ type: 'error', message: err.message || 'Re-index failed' });
    }
  };

  const toggleScheduler = async (start: boolean) => {
    try {
      const endpoint = start ? 'start-scheduler' : 'stop-scheduler';
      const res = await fetch(`${API_URL}/api/ingestion/${endpoint}`, { method: 'POST' });
      if (res.ok) {
        await fetchIngestionStatus();
        setDataMessage(`Scheduler ${start ? 'started' : 'stopped'} successfully`);
      }
    } catch {
      setDataError(`Failed to ${start ? 'start' : 'stop'} scheduler`);
    }
  };

  const newsFeedDot = newsFeedStatus === 'fresh' ? 'bg-green-500' : newsFeedStatus === 'stale' ? 'bg-yellow-400' : 'bg-slate-300';
  const newsFeedLabel = newsFeedStatus === 'fresh' ? 'Fresh' : newsFeedStatus === 'stale' ? 'Stale (>24h)' : 'Unavailable';

  const TABS: { id: TabId; label: string; emoji: string }[] = [
    { id: 'preferences', label: 'Preferences', emoji: '⚙️' },
    { id: 'data', label: 'Data & Sources', emoji: '🗄️' },
    { id: 'alerts-config', label: 'Alerts Config', emoji: '🔔' },
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-slate-200 border-t-slate-600 mx-auto" />
          <p className="text-slate-600 mt-4 font-medium">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className="bg-gradient-to-br from-slate-700 to-slate-900 p-3 rounded-xl shadow-lg">
            <Settings className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Settings</h1>
            <p className="text-slate-500 mt-1">Configure your platform preferences and data sources</p>
          </div>
        </div>
        {activeTab === 'preferences' && (
          <button
            onClick={saveSettings}
            disabled={saving}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-medium transition-all ${
              saved ? 'bg-green-500 text-white' : 'bg-slate-900 text-white hover:bg-slate-800'
            }`}
          >
            {saving ? <RefreshCw className="h-4 w-4 animate-spin" /> : saved ? <Check className="h-4 w-4" /> : <Save className="h-4 w-4" />}
            {saved ? 'Saved!' : 'Save Changes'}
          </button>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-8 border-b border-slate-200 pb-0">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-5 py-3 text-sm font-medium rounded-t-xl transition-all border border-b-0 -mb-px ${
              activeTab === tab.id
                ? 'bg-white border-slate-200 text-slate-900 shadow-sm'
                : 'bg-transparent border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            <span>{tab.emoji}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── TAB: PREFERENCES ── */}
      {activeTab === 'preferences' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            {/* Notification Channels */}
            <Card className="border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="h-5 w-5 text-orange-500" />
                  Notification Channels
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Email */}
                <div className="p-4 bg-slate-50 rounded-xl">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <Mail className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Email Notifications</h3>
                        <p className="text-sm text-slate-500">Receive alerts via email</p>
                      </div>
                    </div>
                    <button
                      onClick={() => updateSettings('emailEnabled', !settings.emailEnabled)}
                      className={`relative w-12 h-6 rounded-full transition-colors ${settings.emailEnabled ? 'bg-blue-600' : 'bg-slate-300'}`}
                    >
                      <span className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${settings.emailEnabled ? 'translate-x-6' : ''}`} />
                    </button>
                  </div>
                  {settings.emailEnabled && (
                    <input
                      type="email"
                      value={settings.emailAddress}
                      onChange={(e) => updateSettings('emailAddress', e.target.value)}
                      placeholder="Enter your email address"
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg mb-2"
                    />
                  )}
                  <div className="mt-2 text-xs">
                    {notificationConfig?.email?.enabled ? (
                      <span className="text-green-600">✓ SendGrid API configured</span>
                    ) : (
                      <span className="text-amber-600">
                        Email alerts disabled — configure SendGrid API key to enable.{' '}
                        <a
                          href="https://docs.sendgrid.com/for-developers/sending-email/api-getting-started"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="underline underline-offset-2 inline-flex items-center gap-0.5"
                        >
                          Setup Guide <ExternalLink className="h-3 w-3" />
                        </a>
                      </span>
                    )}
                  </div>
                </div>

                {/* Slack */}
                <div className="p-4 bg-slate-50 rounded-xl">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="bg-purple-100 p-2 rounded-lg">
                        <MessageSquare className="h-5 w-5 text-purple-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold">Slack Notifications</h3>
                        <p className="text-sm text-slate-500">Receive alerts in Slack</p>
                      </div>
                    </div>
                    <button
                      onClick={() => updateSettings('slackEnabled', !settings.slackEnabled)}
                      className={`relative w-12 h-6 rounded-full transition-colors ${settings.slackEnabled ? 'bg-purple-600' : 'bg-slate-300'}`}
                    >
                      <span className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${settings.slackEnabled ? 'translate-x-6' : ''}`} />
                    </button>
                  </div>
                  {settings.slackEnabled && (
                    <input
                      type="text"
                      value={settings.slackChannel}
                      onChange={(e) => updateSettings('slackChannel', e.target.value)}
                      placeholder="#channel-name"
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg mb-2"
                    />
                  )}
                  <div className="mt-2 text-xs text-slate-500">
                    {notificationConfig?.slack?.enabled ? (
                      <span className="text-green-600">✓ Slack webhook configured</span>
                    ) : (
                      <span className="text-amber-600">⚠ Slack webhook not configured</span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Alert Types */}
            <Card className="border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  Alert Types
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    { key: 'capex', label: 'CapEx Anomalies', description: 'Unusual capital expenditure changes', icon: '💰' },
                    { key: 'sentiment', label: 'Sentiment Shifts', description: 'Significant tone changes in filings', icon: '📊' },
                    { key: 'ai_investment', label: 'AI Investment Changes', description: 'Changes in AI/Data Center focus', icon: '🧠' },
                    { key: 'new_filing', label: 'New Filings', description: 'When new SEC filings are detected', icon: '📄' },
                    { key: 'strategic', label: 'Strategic Changes', description: 'Mergers, acquisitions, restructuring', icon: '🎯' },
                    { key: 'hyperscaler_deal', label: 'Hyperscaler Deal Alerts', description: 'New hyperscaler customer wins or losses', icon: '🎯' },
                    { key: 'facility_capacity', label: 'Facility & Capacity Alerts', description: 'Plant expansions, closures, or capacity shifts', icon: '📍' },
                  ].map((type) => (
                    <div key={type.key} className="p-4 bg-slate-50 rounded-xl">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="text-2xl">{type.icon}</span>
                          <div>
                            <h4 className="font-medium">{type.label}</h4>
                            <p className="text-xs text-slate-500">{type.description}</p>
                          </div>
                        </div>
                        <button
                          onClick={() => updateAlertType(type.key, !settings.alertTypes[type.key as keyof typeof settings.alertTypes])}
                          className={`relative w-10 h-5 rounded-full transition-colors ${
                            settings.alertTypes[type.key as keyof typeof settings.alertTypes] ? 'bg-green-500' : 'bg-slate-300'
                          }`}
                        >
                          <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                            settings.alertTypes[type.key as keyof typeof settings.alertTypes] ? 'translate-x-5' : ''
                          }`} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Delivery Settings */}
            <Card className="border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-blue-500" />
                  Delivery Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-3">Digest Frequency</label>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { value: 'realtime', label: 'Real-time', desc: 'Instant alerts' },
                      { value: 'daily', label: 'Daily', desc: 'Once per day' },
                      { value: 'weekly', label: 'Weekly', desc: 'Once per week' },
                    ].map((freq) => (
                      <button
                        key={freq.value}
                        onClick={() => updateSettings('digestFrequency', freq.value)}
                        className={`p-4 rounded-xl border-2 transition-all ${
                          settings.digestFrequency === freq.value ? 'border-blue-500 bg-blue-50' : 'border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        <div className="font-medium">{freq.label}</div>
                        <div className="text-xs text-slate-500">{freq.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-3">Minimum Severity</label>
                  <div className="grid grid-cols-4 gap-3">
                    {[
                      { value: 'low', label: 'Low', color: 'bg-blue-100 border-blue-300 text-blue-700' },
                      { value: 'medium', label: 'Medium', color: 'bg-yellow-100 border-yellow-300 text-yellow-700' },
                      { value: 'high', label: 'High', color: 'bg-orange-100 border-orange-300 text-orange-700' },
                      { value: 'critical', label: 'Critical', color: 'bg-red-100 border-red-300 text-red-700' },
                    ].map((sev) => (
                      <button
                        key={sev.value}
                        onClick={() => updateSettings('minSeverity', sev.value)}
                        className={`p-3 rounded-xl border-2 transition-all ${
                          settings.minSeverity === sev.value ? sev.color : 'border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        <div className="font-medium text-sm">{sev.label}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* System Status */}
            <Card className="border-0 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5 text-green-500" />
                  System Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-green-50 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full animate-pulse ${status?.api.connected ? 'bg-green-500' : 'bg-red-400'}`} />
                      <span className="font-medium">API Server</span>
                    </div>
                    <Badge className={status?.api.connected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}>
                      {status?.api.connected ? 'Connected' : 'Disconnected'}
                    </Badge>
                  </div>
                </div>

                <div className="p-4 bg-slate-50 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <Database className="h-5 w-5 text-blue-600" />
                    <span className="font-medium">Vector Database</span>
                  </div>
                  {status?.database.documents === 0 && (
                    <div className="flex items-center gap-2 mb-3 p-2 bg-yellow-50 rounded-lg text-xs text-yellow-700">
                      <AlertTriangle className="h-4 w-4 shrink-0" />
                      No documents found — run ingestion to populate the database.
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-2xl font-bold text-slate-900">{status?.database.documents.toLocaleString()}</p>
                      <p className="text-xs text-slate-500">Documents</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-slate-900">{status?.database.collections}</p>
                      <p className="text-xs text-slate-500">Companies</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-slate-50 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <Newspaper className="h-5 w-5 text-blue-500" />
                    <span className="font-medium">News Feed</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-2.5 h-2.5 rounded-full ${newsFeedDot}`} />
                      <span className="text-sm">{newsFeedLabel}</span>
                    </div>
                  </div>
                </div>

                {lastAnalysisRun && (
                  <div className="p-4 bg-slate-50 rounded-xl">
                    <div className="flex items-center gap-3 mb-2">
                      <BarChart3 className="h-5 w-5 text-purple-600" />
                      <span className="font-medium">Last Analysis Run</span>
                    </div>
                    <p className="text-sm text-slate-500">
                      {new Date(lastAnalysisRun).toLocaleString('en-US', {
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                      })}
                    </p>
                  </div>
                )}

                <div className="p-4 bg-slate-50 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <Clock className="h-5 w-5 text-purple-600" />
                    <span className="font-medium">Scheduler</span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Status</span>
                      <Badge className="bg-green-100 text-green-700">Running</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Active Jobs</span>
                      <span className="font-medium">{status?.scheduler.jobs || 2}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card className="border-0 shadow-xl">
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <button
                  onClick={() => window.location.href = '/alerts'}
                  className="w-full p-3 bg-orange-50 text-orange-700 rounded-xl hover:bg-orange-100 transition-all text-left flex items-center gap-3"
                >
                  <Bell className="h-5 w-5" />
                  <span>View All Alerts</span>
                </button>
                <button
                  onClick={() => setActiveTab('data')}
                  className="w-full p-3 bg-blue-50 text-blue-700 rounded-xl hover:bg-blue-100 transition-all text-left flex items-center gap-3"
                >
                  <Database className="h-5 w-5" />
                  <span>Manage Data Sources</span>
                </button>
                <button
                  onClick={regenerateAllAnalysis}
                  disabled={regenerating}
                  className="w-full p-3 bg-purple-50 text-purple-700 rounded-xl hover:bg-purple-100 transition-all text-left flex items-center gap-3 disabled:opacity-50"
                >
                  <RefreshCw className={`h-5 w-5 ${regenerating ? 'animate-spin' : ''}`} />
                  <span>Regenerate All Analysis</span>
                </button>
                <button
                  onClick={fetchStatus}
                  className="w-full p-3 bg-slate-50 text-slate-700 rounded-xl hover:bg-slate-100 transition-all text-left flex items-center gap-3"
                >
                  <RefreshCw className="h-5 w-5" />
                  <span>Refresh Status</span>
                </button>
              </CardContent>
            </Card>

            {/* Environment (dev only) */}
            {process.env.NODE_ENV === 'development' && (
              <Card className="border-0 shadow-xl bg-gradient-to-br from-slate-800 to-slate-900 text-white">
                <CardContent className="p-6">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <Shield className="h-5 w-5 text-blue-400" />
                    Environment
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-slate-400">API URL</span>
                      <span className="font-mono text-xs">{API_URL}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Mode</span>
                      <span>Development</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Version</span>
                      <span>1.0.0</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
            {process.env.NODE_ENV !== 'development' && (
              <div className="p-4 bg-slate-50 rounded-xl text-sm text-slate-500 text-center border border-slate-200">
                System: Online · Version 1.0.0
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── TAB: DATA & SOURCES ── */}
      {activeTab === 'data' && (
        <div className="space-y-6 max-w-4xl">
          <div className="flex items-center gap-3 mb-2">
            <Database className="h-5 w-5 text-blue-600" />
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Data & Sources</h2>
              <p className="text-sm text-slate-500">Manage automated SEC filing downloads and data ingestion</p>
            </div>
          </div>

          {dataError && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {dataError}
              <button onClick={() => setDataError(null)} className="ml-auto"><X className="h-4 w-4" /></button>
            </div>
          )}
          {dataMessage && (
            <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-xl flex items-center gap-2">
              <Check className="h-4 w-4 shrink-0" />
              {dataMessage}
              <button onClick={() => setDataMessage(null)} className="ml-auto"><X className="h-4 w-4" /></button>
            </div>
          )}
          {filingResult && (
            <div className={`flex items-start gap-3 px-4 py-3 rounded-xl border text-sm ${
              filingResult.type === 'success'
                ? 'bg-green-50 border-green-200 text-green-800'
                : 'bg-red-50 border-red-200 text-red-800'
            }`}>
              <span className="text-base leading-none mt-0.5">{filingResult.type === 'success' ? '✅' : '❌'}</span>
              <span className="flex-1 font-medium">{filingResult.message}</span>
              {filingResult.type === 'error' && (
                <button onClick={() => setFilingResult(null)} className="text-red-400 hover:text-red-700">
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
          )}

          {/* Scheduler */}
          <Card className="border-0 shadow-xl">
            <CardContent className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Automated Scheduler</h3>
                <div className="flex items-center gap-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    ingestionStatus?.scheduler.running ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-700'
                  }`}>
                    {ingestionStatus?.scheduler.running ? 'Running' : 'Stopped'}
                  </span>
                  <button
                    onClick={() => toggleScheduler(!ingestionStatus?.scheduler.running)}
                    className={`px-4 py-2 rounded-xl text-white text-sm font-medium ${
                      ingestionStatus?.scheduler.running ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                    }`}
                  >
                    {ingestionStatus?.scheduler.running ? 'Stop' : 'Start'}
                  </button>
                </div>
              </div>

              {ingestionStatus?.scheduler.jobs && ingestionStatus.scheduler.jobs.length > 0 && (
                <div className="space-y-2">
                  {ingestionStatus.scheduler.jobs.map((job) => {
                    const statusColors =
                      job.status === 'pending'
                        ? 'bg-blue-100 text-blue-600 border border-blue-300'
                        : job.status === 'stopped'
                        ? 'bg-slate-100 text-slate-500 border border-slate-300'
                        : 'bg-slate-100 text-slate-400 border border-slate-200';
                    const statusLabel =
                      job.status === 'pending' ? 'Pending'
                      : job.status === 'stopped' ? 'Stopped'
                      : 'Idle';

                    const fmtDatetime = (iso: string | null) => {
                      if (!iso) return null;
                      try {
                        const d = new Date(iso);
                        const date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                        const time = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
                        return `${date} · ${time}`;
                      } catch { return iso; }
                    };

                    return (
                      <div key={job.id} className="bg-slate-50 p-4 rounded-xl">
                        <div className="flex justify-between items-start mb-2">
                          <span className="font-semibold text-sm text-slate-900">{job.name}</span>
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${statusColors}`}>
                            {statusLabel}
                          </span>
                        </div>
                        <div className="grid grid-cols-3 gap-x-4 text-xs text-slate-500">
                          <div>
                            <span className="text-slate-400">Schedule</span>
                            <div className="font-medium text-slate-700 mt-0.5">
                              {job.friendly_schedule || job.trigger}
                            </div>
                          </div>
                          <div>
                            <span className="text-slate-400">Next run</span>
                            <div className="font-medium text-slate-700 mt-0.5">
                              {fmtDatetime(job.next_run) || 'N/A'}
                            </div>
                          </div>
                          <div>
                            <span className="text-slate-400">Last run</span>
                            <div className="font-medium text-slate-700 mt-0.5">
                              {fmtDatetime(job.last_run) || 'Never'}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          {/* SEC Filings */}
          <Card className="border-0 shadow-xl">
            <CardContent className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold">SEC Filings</h3>
                  <p className="text-slate-500 text-sm">Check for new 10-K, 10-Q, and 8-K filings from EDGAR</p>
                </div>
                <button
                  onClick={checkForFilings}
                  disabled={checkingFilings}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed transition-all"
                >
                  {checkingFilings ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      Checking EDGAR...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4" />
                      Check for New Filings
                    </>
                  )}
                </button>
              </div>

              {/* Progress strip */}
              {checkPhase && (
                <div className="mb-5 p-4 bg-blue-50 border border-blue-100 rounded-xl">
                  <div className="flex items-center gap-1.5 mb-3 flex-wrap">
                    {CHECK_PHASES.map((phase, idx) => {
                      const currentIdx = CHECK_PHASES.findIndex(p => p.id === checkPhase);
                      const isDone = idx < currentIdx;
                      const isActive = idx === currentIdx;
                      return (
                        <div key={phase.id} className="flex items-center gap-1.5">
                          {idx > 0 && (
                            <div className={`h-px w-5 ${isDone ? 'bg-blue-400' : 'bg-slate-200'}`} />
                          )}
                          <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
                            isDone    ? 'bg-blue-500' :
                            isActive  ? 'bg-blue-500 animate-pulse ring-2 ring-blue-300' :
                            'bg-slate-200'
                          }`} />
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-sm font-medium text-blue-800 flex items-center gap-2">
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    {CHECK_PHASES.find(p => p.id === checkPhase)?.label}...
                  </p>
                  {(checkPhase === 'processing' || checkPhase === 'indexing') && (
                    <div className="mt-2 h-1.5 bg-blue-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-400 rounded-full transition-all duration-1000"
                        style={{ width: checkPhase === 'processing' ? '50%' : '90%' }}
                      />
                    </div>
                  )}
                </div>
              )}

              {ingestionStatus?.downloads && (
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-xl text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {kbStatus?.total_local_files ?? ingestionStatus.downloads.total_downloaded}
                    </div>
                    <div className="text-sm text-slate-500 mt-1">Local Filing Files</div>
                  </div>
                  <div className="bg-green-50 p-4 rounded-xl text-center">
                    <div className="text-2xl font-bold text-green-600">{Object.keys(ingestionStatus.downloads.by_company).length}</div>
                    <div className="text-sm text-slate-500 mt-1">Companies</div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-xl text-center">
                    <div className="text-2xl font-bold text-purple-600">{Object.keys(ingestionStatus.downloads.by_form).length}</div>
                    <div className="text-sm text-slate-500 mt-1">Filing Types</div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Knowledge Base */}
          <Card className="border-0 shadow-xl">
            <CardContent className="p-6">
              {/* Header row */}
              <div className="flex justify-between items-start mb-5">
                <div>
                  <h3 className="text-lg font-semibold">Knowledge Base Status</h3>
                  <p className="text-slate-500 text-sm">Vector database used by AI Chat and Intelligence Reports</p>
                </div>
                <button
                  onClick={startReindex}
                  disabled={reindexing}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-xl text-sm font-medium hover:bg-purple-700 disabled:opacity-60 disabled:cursor-not-allowed transition-all"
                >
                  {reindexing ? (
                    <><RefreshCw className="h-4 w-4 animate-spin" />Re-indexing...</>
                  ) : (
                    <><Database className="h-4 w-4" />Re-index All Documents</>
                  )}
                </button>
              </div>

              {/* Empty warning */}
              {kbStatus !== null && kbStatus.doc_count === 0 && (
                <div className="mb-5 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-red-800 text-sm">Knowledge base is empty</p>
                    <p className="text-red-700 text-xs mt-1">
                      AI Chat and Intelligence Reports cannot access any filing data.
                      Click "Re-index All Documents" or download new filings first.
                    </p>
                  </div>
                </div>
              )}

              {/* Stats grid */}
              <div className="grid grid-cols-3 gap-4 mb-5">
                <div className="bg-slate-50 p-4 rounded-xl text-center">
                  <div className="text-2xl font-bold text-slate-900">
                    {kbStatus ? kbStatus.doc_count.toLocaleString() : '—'}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">Documents Indexed</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-xl text-center">
                  <div className="text-sm font-semibold text-slate-700 leading-tight min-h-[2rem] flex items-center justify-center">
                    {kbStatus?.last_indexed
                      ? (() => {
                          const d = new Date(kbStatus.last_indexed);
                          const date = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                          const time = d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
                          return `${date} · ${time}`;
                        })()
                      : '—'}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">Last Updated</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-xl text-center">
                  {kbStatus === null ? (
                    <div className="text-sm font-semibold text-slate-400">—</div>
                  ) : kbStatus.doc_count === 0 ? (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-600 border border-red-300">
                      Empty
                    </span>
                  ) : kbStatus.last_indexed === null ? (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-500 border border-slate-300">
                      Unknown
                    </span>
                  ) : (() => {
                    const age = Date.now() - new Date(kbStatus.last_indexed).getTime();
                    const sevenDays = 7 * 24 * 60 * 60 * 1000;
                    return age > sevenDays ? (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-orange-100 text-orange-600 border border-orange-300">
                        Outdated
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-600 border border-green-300">
                        Ready
                      </span>
                    );
                  })()}
                  <div className="text-xs text-slate-500 mt-1">Status</div>
                </div>
              </div>

              {/* Local files count */}
              {kbStatus && (
                <p className="text-xs text-slate-400 mb-4">
                  {kbStatus.total_local_files} local filing files found across all company directories
                </p>
              )}

              {/* Re-index progress */}
              {reindexProgress && (
                <div className="mb-4 p-4 bg-purple-50 border border-purple-100 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium text-purple-800 flex items-center gap-2">
                      <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                      {reindexProgress.current_file
                        ? `Indexing ${reindexProgress.current_file}`
                        : 'Starting re-index...'}
                    </p>
                    <span className="text-xs text-purple-600 font-mono tabular-nums">
                      {reindexProgress.processed}/{reindexProgress.total || '?'}
                    </span>
                  </div>
                  <div className="h-1.5 bg-purple-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-500 rounded-full transition-all duration-500"
                      style={{
                        width: reindexProgress.total > 0
                          ? `${Math.round((reindexProgress.processed / reindexProgress.total) * 100)}%`
                          : '5%',
                      }}
                    />
                  </div>
                  {reindexProgress.chunks_added > 0 && (
                    <p className="text-xs text-purple-600 mt-2">
                      {reindexProgress.chunks_added.toLocaleString()} chunks indexed so far
                    </p>
                  )}
                </div>
              )}

              {/* Re-index result */}
              {reindexResult && (
                <div className={`flex items-start gap-3 p-3 rounded-xl border text-sm ${
                  reindexResult.type === 'success'
                    ? 'bg-green-50 border-green-200 text-green-800'
                    : 'bg-red-50 border-red-200 text-red-800'
                }`}>
                  <span className="text-base leading-none mt-0.5">
                    {reindexResult.type === 'success' ? '✅' : '❌'}
                  </span>
                  <span className="flex-1 font-medium">{reindexResult.message}</span>
                  {reindexResult.type === 'error' && (
                    <button onClick={() => setReindexResult(null)} className="text-red-400 hover:text-red-700">
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Earnings Calendar note */}
          <div className="flex items-center gap-3 p-4 bg-blue-50 border border-blue-200 rounded-xl text-sm text-blue-800">
            <Clock className="h-5 w-5 text-blue-500 shrink-0" />
            <span>
              For earnings dates and company reporting schedules, visit the{' '}
              <a href="/calendar" className="underline underline-offset-2 font-medium hover:text-blue-900">
                Earnings Intelligence
              </a>{' '}
              page.
            </span>
          </div>
        </div>
      )}

      {/* ── TAB: ALERTS CONFIG ── */}
      {activeTab === 'alerts-config' && (
        <div className="max-w-2xl">
          <Card className="border-0 shadow-xl">
            <CardContent className="p-12 text-center">
              <div className="bg-slate-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4">
                <Bell className="h-9 w-9 text-slate-400" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-2">Alerts Configuration</h3>
              <p className="text-slate-500 mb-4">
                Advanced alert rule builder and scheduling coming soon.
              </p>
              <p className="text-sm text-slate-400">
                For now, configure alert types and notification channels in the{' '}
                <button
                  onClick={() => setActiveTab('preferences')}
                  className="text-blue-500 underline underline-offset-2"
                >
                  Preferences
                </button>{' '}
                tab.
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
