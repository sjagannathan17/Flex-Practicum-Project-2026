'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  MessageSquare, 
  LayoutDashboard, 
  Building2, 
  Settings,
  FileText,
  BarChart3,
  TrendingUp,
  Activity,
  Database,
  Sparkles,
  ChevronRight,
  Brain,
  LineChart,
  Globe,
  GitCompare,
  Bell,
  Newspaper,
  CalendarDays
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/chat', label: 'AI Chat', icon: MessageSquare, badge: 'AI' },
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/ai-investments', label: 'Big 5 AI CapEx', icon: TrendingUp, badge: 'NEW' },
  { href: '/competitor-investments', label: 'EMS Competitors', icon: GitCompare, badge: 'NEW' },
  { href: '/news-monitor', label: 'News Monitor', icon: Newspaper, badge: 'NEW' },
  { href: '/companies', label: 'Companies', icon: Building2 },
  { href: '/analysis', label: 'Analysis', icon: BarChart3 },
  { href: '/analytics', label: 'Analytics', icon: Brain },
  { href: '/news', label: 'News Feed', icon: Newspaper },
  { href: '/calendar', label: 'Calendar', icon: CalendarDays },
  { href: '/heatmap', label: 'Heatmap', icon: Globe },
  { href: '/compare', label: 'Compare', icon: GitCompare },
  { href: '/sentiment', label: 'Sentiment', icon: Activity },
  { href: '/alerts', label: 'Alerts', icon: Bell },
  { href: '/data', label: 'Data Mgmt', icon: Database },
  { href: '/reports', label: 'Reports', icon: FileText },
  { href: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-72 bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800 text-white h-screen flex flex-col shadow-2xl">
      {/* Logo */}
      <div className="p-6 border-b border-slate-700/50">
        <Link href="/" className="flex items-center gap-4 group">
          <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl p-2.5 shadow-lg shadow-blue-500/25 group-hover:shadow-blue-500/40 transition-shadow">
            <TrendingUp className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight tracking-tight">Flex Competitive</h1>
            <h1 className="font-bold text-lg leading-tight tracking-tight">Intelligence Platform</h1>
            <div className="flex items-center gap-1.5 mt-1">
              <Sparkles className="h-3 w-3 text-purple-400" />
              <p className="text-xs font-medium bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">AI Powered</p>
            </div>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-4">Navigation</p>
        <ul className="space-y-1.5">
          {navItems.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
            const Icon = item.icon;
            
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group relative',
                    isActive 
                      ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/25' 
                      : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
                  )}
                >
                  <Icon className={cn(
                    'h-5 w-5 transition-transform group-hover:scale-110',
                    isActive && 'drop-shadow-lg'
                  )} />
                  <span className="font-medium">{item.label}</span>
                  {item.badge && (
                    <span className={cn(
                      'ml-auto text-xs px-2 py-0.5 rounded-full font-semibold',
                      isActive 
                        ? 'bg-white/20 text-white' 
                        : 'bg-purple-500/20 text-purple-400'
                    )}>
                      {item.badge}
                    </span>
                  )}
                  {isActive && (
                    <ChevronRight className="h-4 w-4 ml-auto opacity-70" />
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Stats Card */}
      <div className="p-4">
        <div className="bg-gradient-to-br from-slate-800 to-slate-800/50 rounded-xl p-4 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-xs font-medium text-slate-400">System Active</span>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-2xl font-bold text-white">5</p>
              <p className="text-xs text-slate-500">Companies</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-white">18k+</p>
              <p className="text-xs text-slate-500">Documents</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700/50">
        <div className="flex flex-wrap gap-1.5">
          {['Flex', 'Jabil', 'Celestica', 'Benchmark', 'Sanmina'].map((company) => (
            <span 
              key={company}
              className="text-xs px-2 py-1 bg-slate-800 rounded-md text-slate-400 hover:text-white hover:bg-slate-700 transition-colors cursor-default"
            >
              {company}
            </span>
          ))}
        </div>
      </div>
    </aside>
  );
}
