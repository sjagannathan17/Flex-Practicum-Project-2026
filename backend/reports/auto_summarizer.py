"""
Auto-summarizer for new SEC filings.
Automatically generates summaries when new documents are ingested.
"""
import re
from datetime import datetime
from typing import Optional

from backend.core.config import COMPANIES
from backend.rag.retriever import get_company_documents
# Note: generate_response is synchronous but we keep async interface for consistency
from backend.analytics.sentiment import analyze_company_sentiment
from backend.analytics.classifier import classify_company_investments


class AutoSummarizer:
    """Automatically summarizes new SEC filings."""
    
    def __init__(self):
        self._summaries_cache = {}
    
    async def summarize_document(self, document_text: str, filing_type: str, company: str) -> dict:
        """
        Generate a summary of a single document using rule-based extraction.
        """
        try:
            summary = self._generate_quick_summary(document_text, filing_type)
            
            return {
                "company": company,
                "filing_type": filing_type,
                "summary": summary,
                "generated_at": datetime.now().isoformat(),
                "document_length": len(document_text),
            }
        except Exception as e:
            return {
                "company": company,
                "filing_type": filing_type,
                "summary": f"Auto-summary generation failed: {str(e)}",
                "generated_at": datetime.now().isoformat(),
                "error": True,
            }
    
    async def summarize_latest_filings(self, company: str, limit: int = 3) -> dict:
        """
        Summarize the latest filings for a company.
        """
        cache_key = f"{company}_{limit}"
        if cache_key in self._summaries_cache:
            cached = self._summaries_cache[cache_key]
            # Cache for 1 hour
            if (datetime.now() - datetime.fromisoformat(cached['generated_at'])).seconds < 3600:
                return cached
        
        documents = get_company_documents(company, limit=limit)
        
        summaries = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            text = doc.get('content', '')[:5000]
            
            filing_type = metadata.get('filing_type', 'Unknown')
            
            # Generate quick summary without LLM for speed
            quick_summary = self._generate_quick_summary(text, filing_type)
            summaries.append({
                "filing_type": filing_type,
                "fiscal_year": metadata.get('fiscal_year', 'N/A'),
                "summary": quick_summary,
            })
        
        result = {
            "company": company,
            "summaries": summaries,
            "generated_at": datetime.now().isoformat(),
        }
        
        self._summaries_cache[cache_key] = result
        return result
    
    def _generate_quick_summary(self, text: str, filing_type: str) -> str:
        """Generate a quick rule-based summary without LLM."""
        summary_points = []
        
        # Extract revenue mentions
        revenue_match = re.search(r'revenue[s]?\s+(?:of|was|were|totaled)?\s*\$?([\d,.]+)\s*(billion|million)?', 
                                  text.lower())
        if revenue_match:
            amount = revenue_match.group(1)
            unit = revenue_match.group(2) or ''
            summary_points.append(f"Revenue: ${amount} {unit}".strip())
        
        # Look for AI/data center mentions
        ai_keywords = ['artificial intelligence', 'AI', 'data center', 'machine learning', 'GPU']
        ai_mentions = sum(1 for kw in ai_keywords if kw.lower() in text.lower())
        if ai_mentions > 0:
            summary_points.append(f"Contains {ai_mentions} AI/data center references")
        
        # Look for growth indicators
        if re.search(r'(strong|robust|significant)\s+(growth|demand|performance)', text.lower()):
            summary_points.append("Indicates strong performance")
        
        if re.search(r'(challenges?|headwinds?|decline|decrease)', text.lower()):
            summary_points.append("Notes some challenges or headwinds")
        
        # Look for capital expenditure
        capex_match = re.search(r'capital\s+expenditure[s]?\s+(?:of|was|were)?\s*\$?([\d,.]+)', text.lower())
        if capex_match:
            summary_points.append(f"CapEx mentioned: ${capex_match.group(1)}")
        
        if not summary_points:
            summary_points.append(f"{filing_type} filing - standard disclosure document")
        
        return "; ".join(summary_points)
    
    async def generate_company_brief(self, company: str) -> dict:
        """
        Generate a comprehensive intelligence brief for a company.
        """
        # Get sentiment and classification
        sentiment = analyze_company_sentiment(company)
        classification = classify_company_investments(company)
        latest_summaries = await self.summarize_latest_filings(company, limit=5)
        
        # Compile brief
        brief = {
            "company": company,
            "generated_at": datetime.now().isoformat(),
            "executive_summary": self._compile_executive_summary(sentiment, classification),
            "sentiment_analysis": {
                "score": sentiment.get('sentiment_score', 0),
                "outlook": sentiment.get('overall_sentiment', 'neutral'),
            },
            "investment_focus": {
                "ai_percentage": classification.get('overall_ai_focus_percentage', 0),
                "classification": classification.get('investment_focus', 'balanced'),
            },
            "recent_filings": latest_summaries.get('summaries', []),
        }
        
        return brief
    
    def _compile_executive_summary(self, sentiment: dict, classification: dict) -> str:
        """Compile an executive summary from analytics data."""
        sentiment_score = sentiment.get('sentiment_score', 0)
        ai_focus = classification.get('overall_ai_focus_percentage', 0)
        
        sentiment_desc = "positive" if sentiment_score > 0.6 else "neutral" if sentiment_score > 0.4 else "cautious"
        ai_desc = "strong" if ai_focus > 40 else "moderate" if ai_focus > 20 else "limited"
        
        return f"Company demonstrates {sentiment_desc} sentiment in recent filings with {ai_desc} AI/data center investment focus ({ai_focus:.0f}%). Investment classification: {classification.get('investment_focus', 'balanced')}."


# Singleton instance
_auto_summarizer = AutoSummarizer()


async def summarize_new_filing(document_text: str, filing_type: str, company: str) -> dict:
    """Summarize a new filing."""
    return await _auto_summarizer.summarize_document(document_text, filing_type, company)


async def get_company_summaries(company: str, limit: int = 3) -> dict:
    """Get summaries of latest filings."""
    return await _auto_summarizer.summarize_latest_filings(company, limit)


async def generate_intelligence_brief(company: str) -> dict:
    """Generate comprehensive intelligence brief."""
    return await _auto_summarizer.generate_company_brief(company)
