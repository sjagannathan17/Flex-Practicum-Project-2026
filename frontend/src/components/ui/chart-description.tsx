import { Info } from 'lucide-react';

interface ChartDescriptionProps {
  description: string;
  source?: string;
  lastUpdated?: string;
}

export function ChartDescription({ description, source, lastUpdated }: ChartDescriptionProps) {
  return (
    <div className="mt-3 pt-3 border-t border-slate-100">
      <div className="flex items-start gap-2">
        <Info className="h-4 w-4 text-slate-400 mt-0.5 flex-shrink-0" />
        <div>
          <p className="text-sm text-slate-600">{description}</p>
          {(source || lastUpdated) && (
            <p className="text-xs text-slate-400 mt-1">
              {source && <span>Source: {source}</span>}
              {source && lastUpdated && <span> &middot; </span>}
              {lastUpdated && <span>Updated: {lastUpdated}</span>}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
