"""
Analytics module for advanced analysis capabilities.
"""
from .sentiment import (
    analyze_company_sentiment,
    compare_company_sentiments,
    detect_sentiment_changes,
    analyze_lexicon_sentiment,
)
from .anomaly import (
    detect_capex_anomalies,
    detect_sentiment_shifts,
    detect_ai_investment_changes,
    get_all_anomalies,
)
from .trends import (
    analyze_company_trends,
    compare_company_trends,
)
from .classifier import (
    classify_company_investments,
    compare_investment_focus,
    get_ai_investment_leaders,
)
from .geographic import (
    get_company_facilities,
    get_regional_distribution,
    get_all_facilities_map,
    analyze_regional_investments,
    compare_geographic_footprints,
)

__all__ = [
    # Sentiment
    "analyze_company_sentiment",
    "compare_company_sentiments",
    "detect_sentiment_changes",
    "analyze_lexicon_sentiment",
    # Anomaly
    "detect_capex_anomalies",
    "detect_sentiment_shifts",
    "detect_ai_investment_changes",
    "get_all_anomalies",
    # Trends
    "analyze_company_trends",
    "compare_company_trends",
    # Classifier
    "classify_company_investments",
    "compare_investment_focus",
    "get_ai_investment_leaders",
    # Geographic
    "get_company_facilities",
    "get_regional_distribution",
    "get_all_facilities_map",
    "analyze_regional_investments",
    "compare_geographic_footprints",
    # Table Extractor
    "extract_company_financials",
    "extract_capex_breakdown",
    "compare_company_financials",
]

from .table_extractor import (
    extract_company_financials,
    extract_capex_breakdown,
    compare_company_financials,
)
