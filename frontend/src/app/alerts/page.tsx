'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Bell,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Brain,
  RefreshCw,
  Check,
  X,
  Clock,
  Filter,
  Trash2,
  Mail,
  MessageSquare,
  Settings,
  Send,
  CheckCircle,
  XCircle,
  Zap,
  Info,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

interface Alert {
  id: number;
  type: string;
  severity: string;
  company: string;
  title: string;
  message?: string;
  description?: string;
  data: Record<string, any>;
  created_at: string;
  read: boolean;
  dismissed: boolean;
}

interface AlertSummary {
  total_active: number;
  unread: number;
  by_severity: Record<string, number>;
  by_type: Record<string, number>;
  by_company: Record<string, number>;
  has_critical: boolean;
  has_high: boolean;
}

interface NotificationConfig {
  email: { enabled: boolean; has_api_key: boolean; from_email: string };
  slack: { enabled: boolean; has_webhook: boolean; has_bot_token: boolean; default_channel: string };
}

const COMPANY_COLORS: Record<string, string> = {
  Flex: '#3B82F6',
  Jabil: '#10B981',
  Celestica: '#6366F1',
  Benchmark: '#F59E0B',
  Sanmina: '#EF4444',
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-100 text-red-600 border-red-300',
  high:     'bg-red-100 text-red-600 border-red-300',
  medium:   'bg-orange-100 text-orange-600 border-orange-300',
  low:      'bg-green-100 text-green-600 border-green-300',
};

const TYPE_ICONS: Record<string, any> = {
  capex_anomaly: TrendingUp,
  capex_spike: TrendingUp,
  capex_drop: TrendingDown,
  sentiment_shift: TrendingDown,
  sentiment_positive: TrendingUp,
  sentiment_negative: TrendingDown,
  ai_investment_change: Brain,
  ai_investment_surge: Brain,
  new_filing: Bell,
  strategic_change: Zap,
};

const MOCK_ALERTS: Alert[] = [
  {
    id: -1,
    type: 'strategic_change',
    severity: 'high',
    company: 'Celestica',
    title: 'Strategic Pivot: Defense Electronics Expansion',
    message: 'Celestica expands defense electronics capability with new program wins, signaling strategic pivot toward high-margin verticals. Recent filings mention 3 new defense program awards.',
    description: '',
    data: {},
    created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    read: false,
    dismissed: false,
  },
  {
    id: -2,
    type: 'ai_investment_change',
    severity: 'medium',
    company: 'Jabil',
    title: 'AI/ML Infrastructure Mentions Up 34% QoQ',
    message: "Jabil's AI/ML infrastructure mentions increased 34% quarter-over-quarter in recent earnings calls and filings, indicating accelerating investment thesis and potential CapEx reallocation.",
    description: '',
    data: {},
    created_at: new Date(Date.now() - 18 * 60 * 60 * 1000).toISOString(),
    read: false,
    dismissed: false,
  },
  {
    id: -3,
    type: 'capex_anomaly',
    severity: 'medium',
    company: 'Benchmark',
    title: 'CapEx Anomaly: +22% vs Prior Year Despite Revenue Headwinds',
    message: 'Benchmark Electronics Q3 CapEx rose 22% year-over-year despite revenue headwinds — potential capacity expansion signal or strategic positioning ahead of demand recovery.',
    description: '',
    data: {},
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    read: true,
    dismissed: false,
  },
];

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [config, setConfig] = useState<NotificationConfig | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [checking, setChecking] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [testEmail, setTestEmail] = useState('');
  const [testSlackChannel, setTestSlackChannel] = useState('');
  const [sendingTest, setSendingTest] = useState(false);
  const [testResult, setTestResult] = useState<{ type: string; success: boolean; message: string } | null>(null);
  const [sampleBannerDismissed, setSampleBannerDismissed] = useState(false);
  const [mockAlerts, setMockAlerts] = useState<Alert[]>(MOCK_ALERTS);

  useEffect(() => {
    fetchAlerts();
    fetchConfig();
  }, [filter]);

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filter !== 'all') {
        if (['critical', 'high', 'medium', 'low'].includes(filter)) {
          params.set('severity', filter);
        } else if (filter === 'unread') {
          params.set('unread_only', 'true');
        } else {
          params.set('company', filter);
        }
      }

      const [alertsRes, summaryRes] = await Promise.all([
        fetch(`${API_URL}/api/alerts?${params.toString()}`),
        fetch(`${API_URL}/api/alerts/summary`),
      ]);

      if (alertsRes.ok) {
        const data = await alertsRes.json();
        setAlerts(data.alerts || []);
      }
      if (summaryRes.ok) {
        setSummary(await summaryRes.json());
      }
    } catch (err) {
      console.error('Failed to fetch alerts:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch(`${API_URL}/api/alerts/config`);
      if (res.ok) setConfig(await res.json());
    } catch (err) {
      console.error('Failed to fetch config:', err);
    }
  };

  const checkForAlerts = async () => {
    setChecking(true);
    try {
      await fetch(`${API_URL}/api/alerts/detect`, { method: 'POST' });
      await fetchAlerts();
    } catch (err) {
      console.error('Failed to check for alerts:', err);
    } finally {
      setChecking(false);
    }
  };

  const markAsRead = async (alertId: number) => {
    if (alertId < 0) {
      setMockAlerts(prev => prev.map(a => a.id === alertId ? { ...a, read: true } : a));
      return;
    }
    try {
      await fetch(`${API_URL}/api/alerts/${alertId}/read`, { method: 'POST' });
      setAlerts(alerts.map(a => a.id === alertId ? { ...a, read: true } : a));
    } catch (err) {
      console.error('Failed to mark alert as read:', err);
    }
  };

  const dismissAlert = async (alertId: number) => {
    if (alertId < 0) {
      setMockAlerts(prev => prev.filter(a => a.id !== alertId));
      return;
    }
    try {
      await fetch(`${API_URL}/api/alerts/${alertId}/dismiss`, { method: 'POST' });
      setAlerts(alerts.filter(a => a.id !== alertId));
    } catch (err) {
      console.error('Failed to dismiss alert:', err);
    }
  };

  const clearAllAlerts = async () => {
    if (!confirm('Are you sure you want to clear all alerts?')) return;
    try {
      await fetch(`${API_URL}/api/alerts`, { method: 'DELETE' });
      setAlerts([]);
      await fetchAlerts();
    } catch (err) {
      console.error('Failed to clear alerts:', err);
    }
  };

  const sendTestEmail = async () => {
    if (!testEmail) return;
    setSendingTest(true);
    try {
      const res = await fetch(`${API_URL}/api/alerts/test/email?email=${encodeURIComponent(testEmail)}`, { method: 'POST' });
      const data = await res.json();
      setTestResult({
        type: 'email',
        success: data.success || data.logged,
        message: data.logged ? 'Email logged (SendGrid not configured)' : data.success ? 'Email sent successfully' : 'Failed to send email',
      });
    } catch {
      setTestResult({ type: 'email', success: false, message: 'Failed to send test email' });
    } finally {
      setSendingTest(false);
    }
  };

  const sendTestSlack = async () => {
    setSendingTest(true);
    try {
      const params = testSlackChannel ? `?channel=${encodeURIComponent(testSlackChannel)}` : '';
      const res = await fetch(`${API_URL}/api/alerts/test/slack${params}`, { method: 'POST' });
      const data = await res.json();
      setTestResult({
        type: 'slack',
        success: data.success || data.logged,
        message: data.logged ? 'Message logged (Slack not configured)' : data.success ? 'Slack message sent' : 'Failed to send Slack message',
      });
    } catch {
      setTestResult({ type: 'slack', success: false, message: 'Failed to send test Slack message' });
    } finally {
      setSendingTest(false);
    }
  };

  const sendDigest = async (channel: 'email' | 'slack') => {
    setSendingTest(true);
    try {
      const body = channel === 'email'
        ? { email: testEmail || 'test@example.com' }
        : { slack_channel: testSlackChannel || '#competitive-intel' };
      const res = await fetch(`${API_URL}/api/alerts/notify/digest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setTestResult({ type: channel, success: true, message: `Digest sent with ${data.alert_count} alerts` });
    } catch {
      setTestResult({ type: channel, success: false, message: 'Failed to send digest' });
    } finally {
      setSendingTest(false);
    }
  };

  const getAlertIcon = (type: string) => {
    const Icon = TYPE_ICONS[type] || AlertTriangle;
    return <Icon className="h-5 w-5" />;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = Date.now();
    const diffMs = now - date.getTime();
    const diffH = Math.floor(diffMs / 3600000);
    if (diffH < 1) return 'Just now';
    if (diffH < 24) return `${diffH}h ago`;
    const diffD = Math.floor(diffH / 24);
    if (diffD < 7) return `${diffD}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  // Merge real alerts + visible mock alerts, apply filter
  const visibleMockAlerts = alerts.length === 0 ? mockAlerts.filter(a => {
    if (filter === 'all') return true;
    if (filter === 'unread') return !a.read;
    if (['critical', 'high', 'medium', 'low'].includes(filter)) return a.severity === filter;
    return a.company === filter;
  }) : [];

  const allVisible = [...alerts, ...visibleMockAlerts];
  const showingMocks = alerts.length === 0 && mockAlerts.length > 0;

  const statVal = (n: number | undefined) => (n === undefined || n === 0) ? '—' : n;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-orange-200 border-t-orange-600 mx-auto" />
          <p className="text-slate-600 mt-4 font-medium">Loading alerts...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-br from-orange-500 to-red-600 p-3 rounded-xl shadow-lg shadow-orange-500/20">
              <Bell className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-900">Intelligence Alerts</h1>
              <p className="text-slate-500 mt-1">Anomalies and significant changes across EMS competitors</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2.5 rounded-xl border transition-all ${
                showSettings
                  ? 'bg-slate-900 text-white border-slate-900'
                  : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
              }`}
              title="Notification Settings"
            >
              <Settings className="h-5 w-5" />
            </button>
            <button
              onClick={checkForAlerts}
              disabled={checking}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-xl hover:shadow-lg transition-all disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${checking ? 'animate-spin' : ''}`} />
              Check for Alerts
            </button>
            {alerts.length > 0 && (
              <button
                onClick={clearAllAlerts}
                className="flex items-center gap-2 px-4 py-2 bg-white rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition-all"
              >
                <Trash2 className="h-4 w-4" />
                Clear All
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <Card className="border-0 shadow-xl mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Notification Settings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Email Settings */}
              <div className="p-4 bg-slate-50 rounded-xl">
                <div className="flex items-center gap-2 mb-4">
                  <Mail className="h-5 w-5 text-blue-600" />
                  <h3 className="font-semibold">Email Notifications</h3>
                  {config?.email.enabled ? (
                    <Badge className="bg-green-100 text-green-700">Configured</Badge>
                  ) : (
                    <Badge className="bg-yellow-100 text-yellow-700">Not Configured</Badge>
                  )}
                </div>
                <div className="space-y-3">
                  <div className="text-sm text-slate-600">
                    <p><strong>From:</strong> {config?.email.from_email || 'Not set'}</p>
                    <p><strong>API Key:</strong> {config?.email.has_api_key ? '••••••••' : 'Not set'}</p>
                  </div>
                  <div className="flex gap-2">
                    <input
                      type="email"
                      value={testEmail}
                      onChange={(e) => setTestEmail(e.target.value)}
                      placeholder="Enter email for test"
                      className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm"
                    />
                    <button
                      onClick={sendTestEmail}
                      disabled={sendingTest || !testEmail}
                      className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
                    >
                      <Send className="h-4 w-4" />
                    </button>
                  </div>
                  <button
                    onClick={() => sendDigest('email')}
                    disabled={sendingTest}
                    className="w-full px-4 py-2 bg-blue-100 text-blue-700 rounded-lg text-sm hover:bg-blue-200"
                  >
                    Send Email Digest
                  </button>
                </div>
              </div>

              {/* Slack Settings */}
              <div className="p-4 bg-slate-50 rounded-xl">
                <div className="flex items-center gap-2 mb-4">
                  <MessageSquare className="h-5 w-5 text-purple-600" />
                  <h3 className="font-semibold">Slack Notifications</h3>
                  {config?.slack.enabled ? (
                    <Badge className="bg-green-100 text-green-700">Configured</Badge>
                  ) : (
                    <Badge className="bg-yellow-100 text-yellow-700">Not Configured</Badge>
                  )}
                </div>
                <div className="space-y-3">
                  <div className="text-sm text-slate-600">
                    <p><strong>Channel:</strong> {config?.slack.default_channel || 'Not set'}</p>
                    <p><strong>Webhook:</strong> {config?.slack.has_webhook ? 'Configured' : 'Not set'}</p>
                  </div>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={testSlackChannel}
                      onChange={(e) => setTestSlackChannel(e.target.value)}
                      placeholder="#channel (optional)"
                      className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm"
                    />
                    <button
                      onClick={sendTestSlack}
                      disabled={sendingTest}
                      className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 disabled:opacity-50"
                    >
                      <Send className="h-4 w-4" />
                    </button>
                  </div>
                  <button
                    onClick={() => sendDigest('slack')}
                    disabled={sendingTest}
                    className="w-full px-4 py-2 bg-purple-100 text-purple-700 rounded-lg text-sm hover:bg-purple-200"
                  >
                    Send Slack Digest
                  </button>
                </div>
              </div>
            </div>

            {testResult && (
              <div className={`mt-4 p-4 rounded-lg flex items-center gap-3 ${testResult.success ? 'bg-green-50' : 'bg-red-50'}`}>
                {testResult.success ? (
                  <CheckCircle className="h-5 w-5 text-green-600" />
                ) : (
                  <XCircle className="h-5 w-5 text-red-600" />
                )}
                <span className={testResult.success ? 'text-green-700' : 'text-red-700'}>{testResult.message}</span>
                <button onClick={() => setTestResult(null)} className="ml-auto text-slate-400 hover:text-slate-600">
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-2">Configuration</h4>
              <p className="text-sm text-blue-700">
                Set the following environment variables in your <code className="bg-blue-100 px-1 rounded">backend/.env</code> file:
              </p>
              <pre className="mt-2 p-2 bg-white rounded text-xs text-slate-600 overflow-x-auto">
{`# Email (SendGrid)
SENDGRID_API_KEY=your_sendgrid_api_key
ALERT_FROM_EMAIL=alerts@yourcompany.com

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
# or
SLACK_BOT_TOKEN=xoxb-your-token`}
              </pre>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <Card className="border-0 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="bg-orange-100 p-2 rounded-lg">
                <Bell className="h-5 w-5 text-orange-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">{statVal(summary?.total_active)}</p>
                <p className="text-xs text-slate-500">Active Alerts</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="bg-blue-100 p-2 rounded-lg">
                <Clock className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">{statVal(summary?.unread)}</p>
                <p className="text-xs text-slate-500">Unread</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="bg-red-100 p-2 rounded-lg">
                <AlertTriangle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">
                  {statVal(
                    summary
                      ? (summary.by_severity?.high || 0) + (summary.by_severity?.critical || 0)
                      : undefined
                  )}
                </p>
                <p className="text-xs text-slate-500">High Priority</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="bg-purple-100 p-2 rounded-lg">
                <Brain className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">
                  {statVal(
                    summary
                      ? (summary.by_type?.ai_investment_change || 0) + (summary.by_type?.ai_investment_surge || 0)
                      : undefined
                  )}
                </p>
                <p className="text-xs text-slate-500">AI Changes</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Sample alerts banner */}
      {showingMocks && !sampleBannerDismissed && (
        <div className="flex items-start gap-3 p-4 mb-6 bg-blue-50 border border-blue-200 rounded-xl">
          <Info className="h-5 w-5 text-blue-600 mt-0.5 shrink-0" />
          <p className="text-sm text-blue-800 flex-1">
            <span className="font-semibold">Sample alerts shown below</span> — connect live data sources in{' '}
            <a href="/settings" className="underline underline-offset-2 hover:text-blue-900">Settings</a>{' '}
            to receive real-time alerts.
          </p>
          <button onClick={() => setSampleBannerDismissed(true)} className="text-blue-400 hover:text-blue-700">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Filters */}
      <Card className="border-0 shadow-lg mb-6">
        <CardContent className="p-4">
          <div className="flex items-center gap-3 flex-wrap">
            <Filter className="h-4 w-4 text-slate-500" />
            <span className="text-sm text-slate-600 font-medium">Filter:</span>

            {['all', 'unread', 'critical', 'high', 'medium', 'low'].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  filter === f ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}

            <div className="w-px h-6 bg-slate-200 mx-2" />

            {Object.keys(COMPANY_COLORS).map((company) => (
              <button
                key={company}
                onClick={() => setFilter(company)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  filter === company ? 'text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
                style={{ backgroundColor: filter === company ? COMPANY_COLORS[company] : undefined }}
              >
                {company}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Alerts List */}
      {allVisible.length === 0 ? (
        <Card className="border-0 shadow-xl">
          <CardContent className="p-12 text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="bg-slate-100 rounded-full w-20 h-20 flex items-center justify-center relative mx-auto">
                <Bell className="h-9 w-9 text-slate-400" />
                <Clock className="h-5 w-5 text-slate-300 absolute bottom-2 right-2" />
              </div>
            </div>
            <h3 className="text-xl font-semibold text-slate-900 mb-2">No new alerts</h3>
            <p className="text-slate-500 mb-1">
              {filter === 'all'
                ? 'The system is monitoring all 5 EMS companies for significant changes.'
                : 'No alerts match the current filter.'}
            </p>
            {filter === 'all' && (
              <p className="text-xs text-slate-400 mb-6">
                Alert triggers are configured in{' '}
                <a href="/settings" className="text-blue-500 underline underline-offset-2">Settings</a>
              </p>
            )}
            <button
              onClick={checkForAlerts}
              className="mt-2 px-6 py-2 bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-xl hover:shadow-lg transition-all"
            >
              Check for New Alerts
            </button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {allVisible.map((alert) => (
            <Card
              key={alert.id}
              className={`border-0 shadow-lg transition-all hover:shadow-xl ${
                !alert.read ? 'ring-2 ring-orange-200' : ''
              } ${alert.id < 0 ? 'opacity-90' : ''}`}
            >
              <CardContent className="p-0">
                <div className="flex">
                  <div
                    className="w-1.5 rounded-l-lg"
                    style={{
                      backgroundColor:
                        alert.severity === 'critical' ? '#DC2626' :
                        alert.severity === 'high'     ? '#DC2626' :
                        alert.severity === 'medium'   ? '#EA580C' :
                        '#16A34A',
                    }}
                  />
                  <div className="flex-1 p-5">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-4">
                        <div
                          className={`p-2 rounded-lg ${
                            alert.type.includes('capex') ? 'bg-orange-100 text-orange-600' :
                            alert.type.includes('sentiment') ? 'bg-purple-100 text-purple-600' :
                            alert.type.includes('ai') ? 'bg-blue-100 text-blue-600' :
                            'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {getAlertIcon(alert.type)}
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold text-slate-900">{alert.title}</h3>
                            {!alert.read && <span className="w-2 h-2 rounded-full bg-orange-500" />}
                          </div>
                          <p className="text-sm text-slate-600 mb-3">{alert.message || alert.description}</p>
                          <div className="flex items-center gap-3">
                            <Badge
                              className="text-white"
                              style={{ backgroundColor: COMPANY_COLORS[alert.company] || '#64748B' }}
                            >
                              {alert.company}
                            </Badge>
                            <Badge className={`${SEVERITY_COLORS[alert.severity]} border`}>
                              {alert.severity.toUpperCase()}
                            </Badge>
                            <span className="text-xs text-slate-400">{formatDate(alert.created_at)}</span>
                            {alert.id < 0 && (
                              <span className="text-xs text-slate-300 italic">sample</span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {!alert.read && (
                          <button
                            onClick={() => markAsRead(alert.id)}
                            className="p-2 text-slate-400 hover:text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                            title="Mark as read"
                          >
                            <Check className="h-4 w-4" />
                          </button>
                        )}
                        <button
                          onClick={() => dismissAlert(alert.id)}
                          className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Dismiss"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
