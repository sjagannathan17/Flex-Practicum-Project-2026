"""
PDF export functionality.
Generates professional PDF reports using HTML templates.
"""
import io
from datetime import datetime
from typing import Optional

try:
    from weasyprint import HTML, CSS
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

from backend.core.config import COMPANIES
from backend.analytics.sentiment import analyze_company_sentiment
from backend.analytics.classifier import classify_company_investments, compare_investment_focus
from backend.analytics.trends import analyze_company_trends
from backend.analytics.geographic import get_company_facilities
from backend.analytics.anomaly import get_all_anomalies


def _generate_html_report(company: Optional[str] = None) -> str:
    """Generate HTML content for the report."""
    
    if company:
        company_title = company.title()
        
        # Get data
        sentiment = analyze_company_sentiment(company_title)
        classification = classify_company_investments(company_title)
        trends = analyze_company_trends(company_title)
        facilities = get_company_facilities(company_title)
        
        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: letter;
                    margin: 0.75in;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    color: #1f2937;
                    line-height: 1.6;
                }}
                .header {{
                    background: linear-gradient(135deg, #1e40af, #3b82f6);
                    color: white;
                    padding: 40px;
                    margin: -0.75in -0.75in 30px -0.75in;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 32px;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section h2 {{
                    color: #1e40af;
                    border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 8px;
                    margin-bottom: 15px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: #f3f4f6;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 28px;
                    font-weight: bold;
                    color: #1e40af;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #6b7280;
                    text-transform: uppercase;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th {{
                    background: #1e40af;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #e5e7eb;
                }}
                tr:nth-child(even) {{
                    background: #f9fafb;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                    text-align: center;
                    color: #9ca3af;
                    font-size: 12px;
                }}
                .badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 9999px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                .badge-positive {{ background: #dcfce7; color: #166534; }}
                .badge-neutral {{ background: #f3f4f6; color: #374151; }}
                .badge-negative {{ background: #fee2e2; color: #991b1b; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{company_title} Analysis Report</h1>
                <p>Competitive Intelligence Report • {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{sentiment.get('sentiment_score', 0):.0%}</div>
                    <div class="metric-label">Sentiment Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{classification.get('overall_ai_focus_percentage', 0):.0f}%</div>
                    <div class="metric-label">AI Focus</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{trends.get('overall_outlook', 'N/A').title()}</div>
                    <div class="metric-label">Trend Outlook</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{facilities.get('total_facilities', 0)}</div>
                    <div class="metric-label">Facilities</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Investment Analysis</h2>
                <p><strong>Investment Focus:</strong> {classification.get('investment_focus', 'Balanced')}</p>
                <p><strong>AI/Data Center Focus:</strong> {classification.get('overall_ai_focus_percentage', 0):.1f}%</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Documents</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for category, data in classification.get('investment_breakdown', {}).items():
            html += f"""
                        <tr>
                            <td>{category.replace('_', ' ').title()}</td>
                            <td>{data.get('count', 0)}</td>
                            <td>{data.get('percentage', 0):.1f}%</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Trend Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Direction</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for name, key in [('CapEx', 'capex_trend'), ('AI Focus', 'ai_focus_trend'), ('Sentiment', 'sentiment_trend')]:
            trend = trends.get(key, {})
            direction = trend.get('direction', 'stable').title()
            confidence = trend.get('confidence', 0)
            html += f"""
                        <tr>
                            <td>{name}</td>
                            <td>{direction}</td>
                            <td>{confidence:.0f}%</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Geographic Presence</h2>
        """
        
        hq = facilities.get('headquarters', {})
        if hq:
            html += f"""
                <p><strong>Headquarters:</strong> {hq.get('city', 'N/A')}, {hq.get('country', 'N/A')}</p>
            """
        
        html += """
                <table>
                    <thead>
                        <tr>
                            <th>City</th>
                            <th>Country</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for facility in facilities.get('facilities', [])[:10]:
            html += f"""
                        <tr>
                            <td>{facility.get('city', '')}</td>
                            <td>{facility.get('country', '')}</td>
                            <td>{facility.get('type', 'Manufacturing')}</td>
                        </tr>
            """
        
        html += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Flex Competitive Intelligence Platform • AI Powered</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
    else:
        # Comparison report
        sentiment_data = {"companies": {c['name'].split()[0]: analyze_company_sentiment(c['name'].split()[0]) for c in COMPANIES.values()}}
        investment_data = compare_investment_focus()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: letter;
                    margin: 0.75in;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    color: #1f2937;
                    line-height: 1.6;
                }}
                .header {{
                    background: linear-gradient(135deg, #1e40af, #3b82f6);
                    color: white;
                    padding: 40px;
                    margin: -0.75in -0.75in 30px -0.75in;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 32px;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section h2 {{
                    color: #1e40af;
                    border-bottom: 2px solid #e5e7eb;
                    padding-bottom: 8px;
                    margin-bottom: 15px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th {{
                    background: #1e40af;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #e5e7eb;
                }}
                tr:nth-child(even) {{
                    background: #f9fafb;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #e5e7eb;
                    text-align: center;
                    color: #9ca3af;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>EMS Industry Comparison</h1>
                <p>Competitive Intelligence Report • {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="section">
                <h2>Company Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Company</th>
                            <th>Sentiment</th>
                            <th>AI Focus</th>
                            <th>Classification</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for ticker, config in COMPANIES.items():
            company_name = config['name'].split()[0]
            sent = sentiment_data.get('companies', {}).get(company_name, {})
            inv = investment_data.get('companies', {}).get(company_name, {})
            
            html += f"""
                        <tr>
                            <td><strong>{company_name}</strong></td>
                            <td>{sent.get('sentiment_score', 0):.0%}</td>
                            <td>{inv.get('ai_focus_percentage', 0):.0f}%</td>
                            <td>{inv.get('focus', 'N/A')}</td>
                        </tr>
            """
        
        html += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Flex Competitive Intelligence Platform • AI Powered</p>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
    
    return html


def generate_pdf_report(company: Optional[str] = None) -> bytes:
    """
    Generate a PDF report.
    
    Args:
        company: Optional company name. If None, generates comparison report.
    
    Returns:
        PDF file as bytes
    """
    html_content = _generate_html_report(company)
    
    if HAS_WEASYPRINT:
        # Use WeasyPrint for proper PDF generation
        html = HTML(string=html_content)
        output = io.BytesIO()
        html.write_pdf(output)
        output.seek(0)
        return output.getvalue()
    else:
        # Fallback: return HTML with PDF-like styling info
        # The frontend can use browser print functionality
        return html_content.encode('utf-8')


def generate_html_preview(company: Optional[str] = None) -> str:
    """
    Generate HTML preview of the report.
    Can be used for browser rendering or print-to-PDF.
    """
    return _generate_html_report(company)
