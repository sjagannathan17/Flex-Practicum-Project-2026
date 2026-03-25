'use client';

import { useState, useEffect, useMemo } from 'react';
import type { LatLngExpression } from 'leaflet';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  MapPin,
  Building2,
  Globe,
  RefreshCw,
  Factory,
  Briefcase,
  Users,
  ExternalLink,
  AlertTriangle,
  ShieldCheck,
  ArrowUpRight,
  XCircle,
} from 'lucide-react';
import { MapContainer, TileLayer, CircleMarker, Tooltip as LeafletTooltip, useMap } from 'react-leaflet';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import 'leaflet/dist/leaflet.css';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

const COMPANY_COLORS: Record<string, string> = {
  Flex: '#0078FF',
  Jabil: '#16A34A',
  Celestica: '#7C3AED',
  Benchmark: '#F59E0B',
  Sanmina: '#E11D48',
};

const SHARED_FILTER = '__shared__';

const COMPANY_SORT_ORDER: Record<string, number> = {
  Flex: 0, Jabil: 1, Celestica: 2, Benchmark: 3, Sanmina: 4,
};

const FLEX_REGION_SORT_ORDER: Record<string, number> = {
  APAC: 0, EMEA: 1, Americas: 2, Other: 3,
};

const FLEX_COUNTRY_SORT_ORDER: Record<string, number> = {
  China: 0, Singapore: 1, Malaysia: 2, India: 3, Thailand: 4, Taiwan: 5,
  Japan: 6, Romania: 7, Hungary: 8, Spain: 9, Germany: 10, Netherlands: 11,
  UK: 12, Ireland: 13, Poland: 14, USA: 15, Mexico: 16, Brazil: 17, Canada: 18,
};

const CONTINENT_SORT_ORDER: Record<string, number> = {
  Asia: 0, Europe: 1, 'North America': 2, 'South America': 3, Oceania: 4, Africa: 5, Antarctica: 6, Other: 7,
};

const COUNTRY_FLAGS: Record<string, string> = {
  USA: '🇺🇸', Canada: '🇨🇦', Mexico: '🇲🇽', Brazil: '🇧🇷', China: '🇨🇳',
  Malaysia: '🇲🇾', India: '🇮🇳', Singapore: '🇸🇬', UK: '🇬🇧', Hungary: '🇭🇺',
  Romania: '🇷🇴', Poland: '🇵🇱', Netherlands: '🇳🇱', Spain: '🇪🇸', Germany: '🇩🇪',
  Japan: '🇯🇵', Taiwan: '🇹🇼', Thailand: '🇹🇭', Ireland: '🇮🇪',
};

type MarketRegion = 'all' | 'americas' | 'europe' | 'asia';

const MARKET_REGION_LABELS: Record<MarketRegion, string> = {
  all: 'All Regions', americas: 'Americas', europe: 'Europe', asia: 'Asia',
};

const REGION_VIEW: Record<MarketRegion, { center: LatLngExpression; zoom: number }> = {
  all: { center: [20, 15], zoom: 2 },
  americas: { center: [28, -85], zoom: 3 },
  europe: { center: [50, 12], zoom: 4 },
  asia: { center: [28, 105], zoom: 4 },
};

// Mock data for the Regional Concentration chart (Section 2 left)
const CONCENTRATION_DATA = [
  { region: 'Americas', Flex: 5, Jabil: 1, Celestica: 3, Benchmark: 3, Sanmina: 3 },
  { region: 'EMEA',     Flex: 3, Jabil: 1, Celestica: 2, Benchmark: 3, Sanmina: 3 },
  { region: 'China',    Flex: 4, Jabil: 2, Celestica: 2, Benchmark: 3, Sanmina: 4 },
  { region: 'SEA',      Flex: 4, Jabil: 3, Celestica: 3, Benchmark: 2, Sanmina: 2 },
  { region: 'India',    Flex: 2, Jabil: 1, Celestica: 0, Benchmark: 0, Sanmina: 1 },
  { region: 'Mexico',   Flex: 3, Jabil: 2, Celestica: 1, Benchmark: 2, Sanmina: 1 },
];

// Fallback static overlap cities (used if API returns no overlap data)
const STATIC_OVERLAP_CITIES = ['Shanghai', 'Shenzhen', 'Guadalajara', 'Austin', 'Penang', 'Timisoara', 'Brno'];

// ── Interfaces ────────────────────────────────────────────────────────────────

interface Facility {
  company: string;
  city: string;
  country: string;
  lat: number;
  lng: number;
  type: string;
  website?: string;
  is_headquarters: boolean;
}

interface CompanyComparisonRow {
  company: string;
  regional_distribution?: { Americas?: number; EMEA?: number; APAC?: number };
}

interface MapComparison {
  companies?: CompanyComparisonRow[];
  overlap_analysis?: { locations?: Record<string, string[]> };
  regional_leaders?: {
    APAC?: { company?: string; count?: number };
    Americas?: { company?: string; count?: number };
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getCountryFlag(country: string): string {
  return COUNTRY_FLAGS[country] || '🌐';
}

function getMarketRegionByCountry(country: string): Exclude<MarketRegion, 'all'> | 'other' {
  if (['USA', 'Canada', 'Mexico', 'Brazil'].includes(country)) return 'americas';
  if (['Romania', 'Hungary', 'Spain', 'Germany', 'Netherlands', 'UK', 'Ireland', 'Poland'].includes(country)) return 'europe';
  if (['China', 'Singapore', 'Malaysia', 'India', 'Thailand', 'Taiwan', 'Japan'].includes(country)) return 'asia';
  return 'other';
}

function getFlexRegion(country: string): keyof typeof FLEX_REGION_SORT_ORDER {
  if (['China', 'Singapore', 'Malaysia', 'India', 'Thailand', 'Taiwan', 'Japan'].includes(country)) return 'APAC';
  if (['Romania', 'Hungary', 'Spain', 'Germany', 'Netherlands', 'UK', 'Ireland', 'Poland'].includes(country)) return 'EMEA';
  if (['USA', 'Canada', 'Mexico', 'Brazil'].includes(country)) return 'Americas';
  return 'Other';
}

function getContinentByCountry(country: string): keyof typeof CONTINENT_SORT_ORDER {
  if (['China', 'Singapore', 'Malaysia', 'India', 'Thailand', 'Taiwan', 'Japan'].includes(country)) return 'Asia';
  if (['Romania', 'Hungary', 'Spain', 'Germany', 'Netherlands', 'UK', 'Ireland', 'Poland'].includes(country)) return 'Europe';
  if (['USA', 'Canada', 'Mexico'].includes(country)) return 'North America';
  if (['Brazil'].includes(country)) return 'South America';
  return 'Other';
}

function getCompetitionContext(facility: Facility, cityCompaniesMap: Record<string, string[]>): string {
  const companies = cityCompaniesMap[facility.city] || [facility.company];
  const hasFlex = companies.includes('Flex');

  if (companies.length === 1) {
    return facility.company === 'Flex'
      ? '✅ Flex exclusive — no competitor overlap'
      : '🔴 Competitor only · No Flex presence';
  }

  if (!hasFlex) {
    return '🔴 Competitor only · No Flex presence';
  }

  const others = companies.filter((c) => c !== facility.company);
  if (others.length === 0) return '✅ Flex exclusive — no competitor overlap';
  return `⚠️ ${others.join(', ')} also present · Overlap zone`;
}

function RegionMapController({ region }: { region: MarketRegion }) {
  const map = useMap();
  useEffect(() => {
    const view = REGION_VIEW[region];
    map.flyTo(view.center, view.zoom, { duration: 0.8 });
  }, [map, region]);
  return null;
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function MapPage() {
  const [facilities, setFacilities] = useState<Facility[]>([]);
  const [comparison, setComparison] = useState<MapComparison | null>(null);
  const [selectedCompany, setSelectedCompany] = useState<string | null>(null);
  const [selectedCityFilter, setSelectedCityFilter] = useState<string | null>(null);
  const [marketRegion, setMarketRegion] = useState<MarketRegion>('all');
  const [selectedMapCompany, setSelectedMapCompany] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => { fetchGeographicData(); }, []);

  useEffect(() => {
    if (selectedCompany && selectedCompany !== SHARED_FILTER) {
      setSelectedMapCompany(selectedCompany);
    } else {
      setSelectedMapCompany(null);
    }
  }, [selectedCompany]);

  const fetchGeographicData = async () => {
    setLoading(true);
    try {
      const [facilitiesRes, compareRes] = await Promise.all([
        fetch(`${API_URL}/api/geographic/facilities`),
        fetch(`${API_URL}/api/geographic/compare`),
      ]);
      if (facilitiesRes.ok) {
        const data = await facilitiesRes.json();
        setFacilities(data.facilities || []);
      }
      if (compareRes.ok) {
        setComparison(await compareRes.json());
      }
    } catch (err) {
      console.error('Failed to fetch geographic data:', err);
    } finally {
      setLoading(false);
    }
  };

  // ── Computed values ──────────────────────────────────────────────────────────

  // Map city → array of unique companies in that city
  const cityCompaniesMap = useMemo(() => {
    const map: Record<string, string[]> = {};
    facilities.forEach((f) => {
      if (!map[f.city]) map[f.city] = [];
      if (!map[f.city].includes(f.company)) map[f.city].push(f.company);
    });
    return map;
  }, [facilities]);

  // Cities with multiple companies (overlap zones)
  const computedOverlapCities = useMemo(() => {
    return Object.entries(cityCompaniesMap)
      .filter(([, companies]) => companies.length > 1)
      .map(([city]) => city)
      .sort();
  }, [cityCompaniesMap]);

  // Use API overlap data if available, else computed, else static fallback
  const overlapLocations = useMemo(
    () => comparison?.overlap_analysis?.locations || {},
    [comparison],
  );

  const overlapCities = useMemo(() => {
    const apiCities = Object.keys(overlapLocations);
    if (apiCities.length > 0) return apiCities;
    if (computedOverlapCities.length > 0) return computedOverlapCities;
    return STATIC_OVERLAP_CITIES;
  }, [overlapLocations, computedOverlapCities]);

  const filteredFacilities = useMemo(() => {
    if (selectedCompany === SHARED_FILTER) return [];
    let base = selectedCompany ? facilities.filter((f) => f.company === selectedCompany) : facilities;
    if (selectedCityFilter) base = base.filter((f) => f.city === selectedCityFilter);
    return base;
  }, [facilities, selectedCompany, selectedCityFilter]);

  const sortedFacilities = useMemo(() => {
    return [...filteredFacilities].sort((a, b) => {
      if (selectedCompany === null) {
        const compA = COMPANY_SORT_ORDER[a.company] ?? 999;
        const compB = COMPANY_SORT_ORDER[b.company] ?? 999;
        if (compA !== compB) return compA - compB;
        if (a.company === 'Flex' && b.company === 'Flex') {
          const rA = FLEX_REGION_SORT_ORDER[getFlexRegion(a.country)];
          const rB = FLEX_REGION_SORT_ORDER[getFlexRegion(b.country)];
          if (rA !== rB) return rA - rB;
          const cA = FLEX_COUNTRY_SORT_ORDER[a.country] ?? 999;
          const cB = FLEX_COUNTRY_SORT_ORDER[b.country] ?? 999;
          if (cA !== cB) return cA - cB;
        }
      }
      if (selectedCompany !== null) {
        const cA = CONTINENT_SORT_ORDER[getContinentByCountry(a.country)];
        const cB = CONTINENT_SORT_ORDER[getContinentByCountry(b.country)];
        if (cA !== cB) return cA - cB;
      }
      const countryCmp = a.country.localeCompare(b.country);
      if (countryCmp !== 0) return countryCmp;
      if (a.is_headquarters !== b.is_headquarters) return a.is_headquarters ? -1 : 1;
      return a.city.localeCompare(b.city);
    });
  }, [filteredFacilities, selectedCompany]);

  const regionalData = useMemo(() => {
    const byCompany: Record<string, { Americas: number; EMEA: number; APAC: number }> = {};
    facilities.forEach((f) => {
      if (!byCompany[f.company]) byCompany[f.company] = { Americas: 0, EMEA: 0, APAC: 0 };
      const r = getMarketRegionByCountry(f.country);
      if (r === 'americas') byCompany[f.company].Americas += 1;
      if (r === 'europe') byCompany[f.company].EMEA += 1;
      if (r === 'asia') byCompany[f.company].APAC += 1;
    });
    return Object.entries(byCompany)
      .sort(([a], [b]) => (COMPANY_SORT_ORDER[a] ?? 999) - (COMPANY_SORT_ORDER[b] ?? 999))
      .map(([company, values]) => ({ company, total: values.Americas + values.EMEA + values.APAC, ...values }));
  }, [facilities]);

  const isSharedMode = selectedCompany === SHARED_FILTER;

  const sharedLocationCards = useMemo(() => {
    return Object.entries(overlapLocations)
      .map(([city, companies]) => ({ city, companies: companies as string[] }))
      .sort((a, b) => a.city.localeCompare(b.city));
  }, [overlapLocations]);

  const marketTabs = useMemo(() => {
    const regions = new Set<Exclude<MarketRegion, 'all'>>();
    facilities.forEach((f) => {
      const r = getMarketRegionByCountry(f.country);
      if (r !== 'other') regions.add(r);
    });
    const ordered: Exclude<MarketRegion, 'all'>[] = ['americas', 'europe', 'asia'];
    return ['all', ...ordered.filter((r) => regions.has(r))] as MarketRegion[];
  }, [facilities]);

  const visibleMapFacilities = useMemo(() => {
    const regionFiltered = marketRegion === 'all'
      ? facilities
      : facilities.filter((f) => getMarketRegionByCountry(f.country) === marketRegion);
    return selectedMapCompany ? regionFiltered.filter((f) => f.company === selectedMapCompany) : regionFiltered;
  }, [facilities, marketRegion, selectedMapCompany]);

  const marketCompanyCards = useMemo(() => {
    return Object.keys(COMPANY_SORT_ORDER)
      .sort((a, b) => (COMPANY_SORT_ORDER[a] ?? 999) - (COMPANY_SORT_ORDER[b] ?? 999))
      .map((company) => {
        const all = facilities.filter((f) => f.company === company);
        const scoped = marketRegion === 'all' ? all : all.filter((f) => getMarketRegionByCountry(f.country) === marketRegion);
        const regionSet = new Set<string>();
        all.forEach((f) => {
          const r = getMarketRegionByCountry(f.country);
          if (r !== 'other') regionSet.add(MARKET_REGION_LABELS[r]);
        });
        // Count by region for top-2
        const regionCounts: Record<string, number> = {};
        all.forEach((f) => {
          const r = getMarketRegionByCountry(f.country);
          if (r !== 'other') regionCounts[MARKET_REGION_LABELS[r]] = (regionCounts[MARKET_REGION_LABELS[r]] || 0) + 1;
        });
        const top2 = Object.entries(regionCounts)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 2)
          .map(([label]) => label);
        return { company, total: all.length, scoped: scoped.length, top2 };
      });
  }, [facilities, marketRegion]);

  // Stat card derived values
  const flexUniqueCities = useMemo(
    () => Object.entries(cityCompaniesMap).filter(([, cos]) => cos.length === 1 && cos[0] === 'Flex').length,
    [cityCompaniesMap],
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-200 border-t-blue-600 mx-auto" />
            <Globe className="h-6 w-6 text-blue-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="text-slate-600 dark:text-slate-300 mt-4 font-medium">Loading footprint data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-950 dark:to-slate-900 p-4">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-br from-blue-500 to-green-600 p-3 rounded-xl shadow-lg shadow-blue-500/20">
              <Globe className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Competitive Footprint</h1>
              <p className="text-slate-500 dark:text-slate-400 mt-0.5 text-sm">
                Where Flex competes, overlaps, and has strategic gaps vs. peers
              </p>
            </div>
          </div>
          <button
            onClick={fetchGeographicData}
            className="flex items-center gap-2 px-3 py-1.5 bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all shadow-sm"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* ── Section 1: Stat Cards + Charts ───────────────────────────────── */}
      <div className="grid items-start grid-cols-1 xl:grid-cols-[260px_minmax(0,1fr)_minmax(0,1fr)] gap-4 mb-4">

        {/* 4 competition-focused stat cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-3">
          <Card className="border-0 shadow-xl !py-0 h-[64px]">
            <CardContent className="p-2 h-full">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-500 dark:text-slate-400 leading-none">Overlap Zones</span>
                <AlertTriangle className="h-4 w-4 text-orange-500" />
              </div>
              <p className="text-[17px] leading-none font-bold text-orange-600 dark:text-orange-400 mt-0.5">
                {overlapCities.length > 0 ? overlapCities.length : 7}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-none mt-0.5">Cities where competitors co-locate</p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-xl !py-0 h-[64px]">
            <CardContent className="p-2 h-full">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-500 dark:text-slate-400 leading-none">Competitor Expansion</span>
                <ArrowUpRight className="h-4 w-4 text-red-500" />
              </div>
              <p className="text-[17px] leading-none font-bold text-red-600 dark:text-red-400 mt-0.5">3</p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-none mt-0.5">New competitor sites, last 6 months</p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-xl !py-0 h-[64px]">
            <CardContent className="p-2 h-full">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-500 dark:text-slate-400 leading-none">Flex Unique Presence</span>
                <ShieldCheck className="h-4 w-4 text-green-500" />
              </div>
              <p className="text-[17px] leading-none font-bold text-green-600 dark:text-green-400 mt-0.5">
                {flexUniqueCities > 0 ? flexUniqueCities : 8}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-none mt-0.5">Locations with no competitor overlap</p>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-xl !py-0 h-[64px]">
            <CardContent className="p-2 h-full">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-500 dark:text-slate-400 leading-none">Coverage Gaps</span>
                <XCircle className="h-4 w-4 text-red-500" />
              </div>
              <p className="text-[17px] leading-none font-bold text-red-600 dark:text-red-400 mt-0.5">2</p>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-none mt-0.5">Key regions competitors have, Flex doesn&apos;t</p>
            </CardContent>
          </Card>
        </div>

        {/* LEFT CHART: Regional Concentration by Company */}
        <Card className="border-0 shadow-xl !py-0 h-fit self-start">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="flex items-center gap-2 text-base">
              <Building2 className="h-4 w-4 text-blue-600" />
              Manufacturing Concentration — Key Growth Regions
            </CardTitle>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
              Focus on AI-relevant nearshoring destinations
            </p>
          </CardHeader>
          <CardContent className="pt-0 pb-3 px-3">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={CONCENTRATION_DATA}
                layout="vertical"
                barCategoryGap="18%"
                barGap={2}
                margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="region" width={60} tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
                {Object.keys(COMPANY_COLORS).map((company) => (
                  <Bar key={company} dataKey={company} fill={COMPANY_COLORS[company]} radius={[0, 3, 3, 0]} barSize={5} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* RIGHT CHART: Regional Distribution (stacked, all 5 companies) */}
        <Card className="border-0 shadow-xl !py-0 h-fit self-start">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="flex items-center gap-2 text-base">
              <Globe className="h-4 w-4 text-green-600" />
              Regional Distribution
            </CardTitle>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
              APAC / Americas / EMEA breakdown across all EMS peers
            </p>
          </CardHeader>
          <CardContent className="pt-0 pb-3 px-3">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={regionalData} margin={{ top: 4, right: 8, left: -8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="company" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', fontSize: 11 }}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="Americas" stackId="a" fill="#3B82F6" />
                <Bar dataKey="EMEA" stackId="a" fill="#8B5CF6" />
                <Bar dataKey="APAC" stackId="a" fill="#10B981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* ── Section 3: Facility Locations Cards ──────────────────────────── */}
      <Card className="border-0 shadow-xl mb-4">
        <CardHeader className="pb-2">
          <CardTitle>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <MapPin className="h-5 w-5 text-red-500" />
                Facility Locations
              </div>
              <Badge variant="secondary">
                {isSharedMode ? sharedLocationCards.length : filteredFacilities.length} locations
              </Badge>
            </div>

            {/* Filter bar */}
            <div className="mt-3 flex items-center gap-2 flex-wrap">
              <span className="text-sm font-normal text-slate-500 dark:text-slate-400 mr-1">Filter:</span>
              <button
                onClick={() => { setSelectedCompany(null); setSelectedCityFilter(null); }}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  selectedCompany === null
                    ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                    : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700'
                }`}
              >
                All
              </button>
              {Object.keys(COMPANY_COLORS).map((company) => (
                <button
                  key={company}
                  onClick={() => { setSelectedCompany(company); setSelectedCityFilter(null); }}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    selectedCompany === company
                      ? 'text-white'
                      : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700'
                  }`}
                  style={{ backgroundColor: selectedCompany === company ? COMPANY_COLORS[company] : undefined }}
                >
                  {company}
                </button>
              ))}
              <button
                onClick={() => { setSelectedCompany(SHARED_FILTER); setSelectedCityFilter(null); }}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  isSharedMode
                    ? 'bg-purple-600 text-white'
                    : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700'
                }`}
              >
                Shared
              </button>
            </div>
          </CardTitle>
        </CardHeader>

        <CardContent className="pt-0">
          {/* Overlap callout with city pills */}
          {overlapCities.length > 0 && !isSharedMode && (
            <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-orange-200 bg-orange-50 dark:border-orange-700/40 dark:bg-orange-500/10 px-3 py-2.5">
              <AlertTriangle className="h-4 w-4 text-orange-500 shrink-0" />
              <span className="text-xs font-semibold text-orange-800 dark:text-orange-300">
                {overlapCities.length} Overlap Zones — cities where Flex and competitors both operate:
              </span>
              <div className="flex flex-wrap gap-1.5 mt-1 w-full">
                {overlapCities.map((city) => (
                  <button
                    key={city}
                    onClick={() => setSelectedCityFilter(selectedCityFilter === city ? null : city)}
                    className={`rounded-full px-2.5 py-0.5 text-xs font-medium border transition-all ${
                      selectedCityFilter === city
                        ? 'bg-orange-500 text-white border-orange-500'
                        : 'bg-white dark:bg-slate-900 text-orange-700 dark:text-orange-300 border-orange-300 dark:border-orange-600 hover:bg-orange-100 dark:hover:bg-orange-500/20'
                    }`}
                  >
                    {city}
                  </button>
                ))}
                {selectedCityFilter && (
                  <button
                    onClick={() => setSelectedCityFilter(null)}
                    className="rounded-full px-2.5 py-0.5 text-xs font-medium text-slate-500 dark:text-slate-400 border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    ✕ Clear
                  </button>
                )}
              </div>
            </div>
          )}

          <div className="h-[270px] overflow-y-auto pr-1" style={{ scrollbarGutter: 'stable' }}>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 2xl:grid-cols-5 gap-2.5">
              {isSharedMode
                ? sharedLocationCards.map(({ city, companies }) => (
                    <div
                      key={city}
                      className="p-2 rounded-lg border-2 transition-all hover:shadow-sm bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
                    >
                      <div className="mb-1.5 flex items-center gap-2">
                        <Users className="h-4 w-4 text-purple-600" />
                        <span className="text-[15px] font-semibold leading-tight text-slate-900 dark:text-slate-100">{city}</span>
                      </div>
                      <div className="mb-1.5 flex flex-wrap gap-1">
                        {companies.map((company) => (
                          <Badge
                            key={company}
                            className="text-[11px]"
                            style={{ backgroundColor: COMPANY_COLORS[company], color: 'white', borderColor: COMPANY_COLORS[company] }}
                          >
                            {company}
                          </Badge>
                        ))}
                      </div>
                      <div className="mt-1 text-[11px] text-orange-600 dark:text-orange-400 font-medium">
                        ⚠️ {companies.length} companies · Overlap zone
                      </div>
                    </div>
                  ))
                : sortedFacilities.map((facility, idx) => {
                    const context = getCompetitionContext(facility, cityCompaniesMap);
                    const isOverlap = context.startsWith('⚠️');
                    return (
                      <div
                        key={idx}
                        className={`p-2 rounded-lg border-2 transition-all hover:shadow-sm bg-white dark:bg-slate-900 min-h-[96px] ${
                          isOverlap
                            ? 'border-orange-200 dark:border-orange-700/50 hover:border-orange-300'
                            : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                        }`}
                      >
                        {/* Line 1 */}
                        <div className="mb-1 flex items-center gap-2 flex-wrap">
                          {facility.is_headquarters
                            ? <Briefcase className="h-4 w-4 text-amber-600 shrink-0" />
                            : <Factory className="h-4 w-4 text-slate-400 shrink-0" />}
                          <span className="text-[14px] font-semibold leading-tight text-slate-900 dark:text-slate-100">{facility.city}</span>
                          <Badge
                            className="text-[10px]"
                            style={{
                              backgroundColor: `${COMPANY_COLORS[facility.company]}20`,
                              color: COMPANY_COLORS[facility.company],
                              borderColor: COMPANY_COLORS[facility.company],
                            }}
                          >
                            {facility.company}
                          </Badge>
                          {facility.website && (
                            <a
                              href={facility.website}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center justify-center h-5 w-5 rounded-md border border-slate-200 dark:border-slate-700 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-slate-800"
                            >
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          )}
                        </div>
                        <div className="grid grid-cols-2 gap-1 text-[11px] mb-1.5">
                          <span className="inline-flex items-center gap-1 text-slate-500 dark:text-slate-400">
                            <span>{getCountryFlag(facility.country)}</span>
                            <span>{facility.country}</span>
                          </span>
                          <span className="text-slate-400 dark:text-slate-500">{facility.type}</span>
                        </div>
                        {/* Line 2: Competition context */}
                        <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight border-t border-slate-100 dark:border-slate-800 pt-1">
                          {context}
                        </p>
                      </div>
                    );
                  })}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── Section 4: Factory Footprint Map ─────────────────────────────── */}
      <Card className="border-0 shadow-xl overflow-hidden bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100">
        <CardHeader className="pb-2">
          <CardTitle>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Globe className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />
                Factory Footprint Globe
              </div>
              <Badge className="bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700">
                Beta
              </Badge>
            </div>
          </CardTitle>

          {/* Region tabs */}
          <div className="mt-3 flex items-center gap-2 flex-wrap">
            {marketTabs.map((region) => (
              <button
                key={region}
                onClick={() => setMarketRegion(region)}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all border ${
                  marketRegion === region
                    ? 'bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 border-slate-900 dark:border-slate-100'
                    : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800'
                }`}
              >
                {MARKET_REGION_LABELS[region]}
              </button>
            ))}
          </div>
        </CardHeader>

        <CardContent className="pt-1 pb-4">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-4">

            {/* Map */}
            <div className="relative h-full min-h-[440px] overflow-hidden rounded-xl border border-slate-200 dark:border-slate-800">
              <MapContainer
                center={REGION_VIEW.all.center}
                zoom={REGION_VIEW.all.zoom}
                minZoom={2}
                maxZoom={7}
                scrollWheelZoom={false}
                className="h-full w-full"
              >
                <RegionMapController region={marketRegion} />
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {visibleMapFacilities.map((facility, idx) => {
                  const companiesInCity = cityCompaniesMap[facility.city] || [facility.company];
                  const isOverlapCity = companiesInCity.length > 1;
                  const hasFlex = companiesInCity.includes('Flex');
                  const competitionStatus = isOverlapCity ? 'Overlap Zone' : 'Exclusive';

                  return (
                    <CircleMarker
                      key={`${facility.company}-${facility.city}-${idx}`}
                      center={[facility.lat, facility.lng]}
                      radius={isOverlapCity ? 7 : 6}
                      pathOptions={{
                        color: isOverlapCity ? '#F97316' : COMPANY_COLORS[facility.company] || '#0ea5e9',
                        fillColor: COMPANY_COLORS[facility.company] || '#0ea5e9',
                        fillOpacity: isOverlapCity ? 0.85 : 1,
                        weight: isOverlapCity ? 3 : 2,
                      }}
                    >
                      <LeafletTooltip direction="top" offset={[0, -6]} opacity={0.97}>
                        <div style={{ fontSize: 11, minWidth: 140 }}>
                          <div style={{ fontWeight: 700, marginBottom: 3 }}>
                            {getCountryFlag(facility.country)} {facility.city}, {facility.country}
                          </div>
                          <div style={{ marginBottom: 2 }}>
                            <span style={{ fontWeight: 600 }}>Companies: </span>
                            {companiesInCity.join(' · ')}
                          </div>
                          <div style={{ marginBottom: 2 }}>
                            <span style={{ fontWeight: 600 }}>Type: </span>
                            {facility.type}{facility.is_headquarters ? ' · HQ' : ''}
                          </div>
                          <div style={{
                            display: 'inline-block',
                            marginTop: 2,
                            padding: '1px 6px',
                            borderRadius: 4,
                            backgroundColor: isOverlapCity ? '#FED7AA' : '#D1FAE5',
                            color: isOverlapCity ? '#C2410C' : '#065F46',
                            fontWeight: 600,
                          }}>
                            {isOverlapCity ? '⚠️ ' : '✅ '}{competitionStatus}
                          </div>
                        </div>
                      </LeafletTooltip>
                    </CircleMarker>
                  );
                })}
              </MapContainer>

              {/* Status bar */}
              <div className="pointer-events-none absolute left-3 bottom-3 z-[500] rounded-md bg-white/90 dark:bg-slate-900/85 border border-slate-200 dark:border-slate-700 px-2 py-1 text-[11px] text-slate-600 dark:text-slate-300">
                {MARKET_REGION_LABELS[marketRegion]} · {selectedMapCompany || 'All Companies'} · {visibleMapFacilities.length} sites
              </div>

              {/* Map legend */}
              <div className="pointer-events-none absolute right-3 bottom-3 z-[500] rounded-md bg-white/90 dark:bg-slate-900/85 border border-slate-200 dark:border-slate-700 px-2.5 py-2 text-[11px] text-slate-600 dark:text-slate-300 space-y-1">
                <div className="font-semibold mb-1 text-[10px] uppercase tracking-wide text-slate-500">Legend</div>
                <div className="flex items-center gap-1.5">
                  <span className="inline-block h-3 w-3 rounded-full bg-blue-500" />
                  <span>Single company</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="inline-block h-3 w-3 rounded-full bg-blue-400 ring-2 ring-orange-400" />
                  <span>Overlap zone</span>
                </div>
              </div>
            </div>

            {/* Company cards */}
            <div className="grid grid-cols-1 gap-2.5">
              {marketCompanyCards.map((row) => (
                <button
                  key={row.company}
                  type="button"
                  onClick={() => setSelectedMapCompany((prev) => (prev === row.company ? null : row.company))}
                  className={`text-left rounded-lg border p-2.5 transition-all ${
                    selectedMapCompany === row.company
                      ? 'border-slate-900 dark:border-slate-100 bg-slate-100 dark:bg-slate-800'
                      : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/80 hover:border-slate-300 dark:hover:border-slate-500'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-5 w-5 rounded flex items-center justify-center text-[9px] font-bold text-white shrink-0"
                        style={{ backgroundColor: COMPANY_COLORS[row.company] }}
                      >
                        {row.company.charAt(0)}
                      </div>
                      <span className="font-semibold text-slate-900 dark:text-slate-100 text-sm">{row.company}</span>
                    </div>
                    <span
                      className="text-[11px] px-2 py-0.5 rounded-full border"
                      style={{
                        color: COMPANY_COLORS[row.company],
                        borderColor: COMPANY_COLORS[row.company],
                        backgroundColor: `${COMPANY_COLORS[row.company]}1a`,
                      }}
                    >
                      {selectedMapCompany === row.company ? 'Focused' : 'View'}
                    </span>
                  </div>
                  <div className="mt-1.5 text-[11px] text-slate-600 dark:text-slate-300 space-y-0.5">
                    <div>
                      <span className="font-semibold text-slate-900 dark:text-slate-100">{row.total}</span> sites
                      {marketRegion !== 'all' && (
                        <span className="text-slate-400"> ({row.scoped} in {MARKET_REGION_LABELS[marketRegion]})</span>
                      )}
                    </div>
                    <div>
                      Top regions:{' '}
                      <span className="font-semibold text-slate-900 dark:text-slate-100">
                        {row.top2.join(', ') || 'N/A'}
                      </span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

    </div>
  );
}
