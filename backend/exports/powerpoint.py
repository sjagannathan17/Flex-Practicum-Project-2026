"""
PowerPoint export functionality using python-pptx.
Generates professional presentation decks.
"""
import io
from datetime import datetime
from typing import Optional

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.enum.shapes import MSO_SHAPE
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

# Alias for compatibility
RGBColor = RGBColor if HAS_PPTX else None

from backend.core.config import COMPANIES
from backend.analytics.sentiment import analyze_company_sentiment
from backend.analytics.classifier import classify_company_investments, compare_investment_focus
from backend.analytics.trends import analyze_company_trends
from backend.analytics.geographic import get_company_facilities
from backend.analytics.anomaly import get_all_anomalies


def _add_title_slide(prs, title: str, subtitle: str = ""):
    """Add a title slide to the presentation."""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Background
    background = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0),
        Inches(10), Inches(7.5)
    )
    background.fill.solid()
    background.fill.fore_color.rgb = RGBColor(30, 64, 175)  # Blue
    background.line.fill.background()
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    if subtitle:
        subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(9), Inches(1))
        tf = subtitle_box.text_frame
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(24)
        p.font.color.rgb = RGBColor(191, 219, 254)  # Light blue
        p.alignment = PP_ALIGN.CENTER
    
    # Date
    date_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.5), Inches(9), Inches(0.5))
    tf = date_box.text_frame
    p = tf.paragraphs[0]
    p.text = datetime.now().strftime('%B %d, %Y')
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(191, 219, 254)
    p.alignment = PP_ALIGN.CENTER
    
    return slide


def _add_section_slide(prs, title: str):
    """Add a section divider slide."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Background
    background = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0),
        Inches(10), Inches(7.5)
    )
    background.fill.solid()
    background.fill.fore_color.rgb = RGBColor(59, 130, 246)  # Blue-500
    background.line.fill.background()
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(9), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER
    
    return slide


def _add_content_slide(prs, title: str, content: list):
    """Add a content slide with bullet points."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title bar
    title_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0),
        Inches(10), Inches(1.2)
    )
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = RGBColor(30, 64, 175)
    title_bar.line.fill.background()
    
    # Title text
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(9), Inches(0.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # Content
    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    
    for i, item in enumerate(content):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        
        p.text = f"• {item}"
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(31, 41, 55)
        p.space_after = Pt(12)
    
    return slide


def _add_metrics_slide(prs, title: str, metrics: list):
    """Add a slide with key metrics in boxes."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title bar
    title_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0),
        Inches(10), Inches(1.2)
    )
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = RGBColor(30, 64, 175)
    title_bar.line.fill.background()
    
    # Title text
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(9), Inches(0.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # Metrics grid
    num_metrics = min(len(metrics), 4)
    box_width = 2.0
    box_height = 1.8
    start_x = (10 - (num_metrics * box_width + (num_metrics - 1) * 0.3)) / 2
    
    for i, (label, value, color) in enumerate(metrics[:4]):
        x = start_x + i * (box_width + 0.3)
        
        # Metric box
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(2.5),
            Inches(box_width), Inches(box_height)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(*color)
        box.line.fill.background()
        
        # Value
        value_box = slide.shapes.add_textbox(Inches(x), Inches(2.7), Inches(box_width), Inches(0.8))
        tf = value_box.text_frame
        p = tf.paragraphs[0]
        p.text = str(value)
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER
        
        # Label
        label_box = slide.shapes.add_textbox(Inches(x), Inches(3.6), Inches(box_width), Inches(0.5))
        tf = label_box.text_frame
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(14)
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER
    
    return slide


def _add_comparison_table(prs, title: str, headers: list, rows: list):
    """Add a comparison table slide."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title bar
    title_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0),
        Inches(10), Inches(1.2)
    )
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = RGBColor(30, 64, 175)
    title_bar.line.fill.background()
    
    # Title text
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(9), Inches(0.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # Table
    num_cols = len(headers)
    num_rows = len(rows) + 1  # +1 for header
    
    table = slide.shapes.add_table(
        num_rows, num_cols,
        Inches(0.5), Inches(1.5),
        Inches(9), Inches(num_rows * 0.6)
    ).table
    
    # Header row
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 64, 175)
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER
    
    # Data rows
    for row_idx, row_data in enumerate(rows, 1):
        for col_idx, value in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = str(value)
            if row_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(243, 244, 246)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(12)
            p.alignment = PP_ALIGN.CENTER
    
    return slide


def generate_powerpoint_report(company: Optional[str] = None) -> bytes:
    """
    Generate a PowerPoint presentation.
    
    Args:
        company: Optional company name. If None, generates comparison report.
    
    Returns:
        PPTX file as bytes
    """
    if not HAS_PPTX:
        raise ImportError("python-pptx is required for PowerPoint export. Install with: pip install python-pptx")
    
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    if company:
        # Single company report
        company_title = company.title()
        
        # Get data
        sentiment = analyze_company_sentiment(company_title)
        classification = classify_company_investments(company_title)
        trends = analyze_company_trends(company_title)
        facilities = get_company_facilities(company_title)
        
        # Title slide
        _add_title_slide(
            prs,
            f"{company_title} Analysis",
            "Competitive Intelligence Report"
        )
        
        # Key Metrics slide
        _add_metrics_slide(prs, "Key Metrics", [
            ("Sentiment", f"{sentiment.get('sentiment_score', 0):.0%}", (34, 197, 94)),  # Green
            ("AI Focus", f"{classification.get('overall_ai_focus_percentage', 0):.0f}%", (139, 92, 246)),  # Purple
            ("Outlook", trends.get('overall_outlook', 'N/A').title(), (59, 130, 246)),  # Blue
            ("Facilities", str(facilities.get('total_facilities', 0)), (249, 115, 22)),  # Orange
        ])
        
        # Investment Focus
        _add_section_slide(prs, "Investment Focus")
        
        investment_content = [
            f"Overall AI/Data Center Focus: {classification.get('overall_ai_focus_percentage', 0):.1f}%",
            f"Investment Classification: {classification.get('investment_focus', 'Balanced')}",
            f"Documents Analyzed: {classification.get('total_documents', 0)}",
        ]
        breakdown = classification.get('investment_breakdown', {})
        for category, data in breakdown.items():
            investment_content.append(f"{category.replace('_', ' ').title()}: {data.get('count', 0)} documents ({data.get('percentage', 0):.1f}%)")
        
        _add_content_slide(prs, "Investment Analysis", investment_content)
        
        # Trends
        _add_section_slide(prs, "Trend Analysis")
        
        trend_content = [
            f"Overall Outlook: {trends.get('overall_outlook', 'N/A').title()}",
            f"CapEx Trend: {trends.get('capex_trend', {}).get('direction', 'N/A').title()} (Confidence: {trends.get('capex_trend', {}).get('confidence', 0):.0f}%)",
            f"AI Focus Trend: {trends.get('ai_focus_trend', {}).get('direction', 'N/A').title()}",
            f"Sentiment Trend: {trends.get('sentiment_trend', {}).get('direction', 'N/A').title()}",
        ]
        _add_content_slide(prs, "Trend Summary", trend_content)
        
        # Geographic Presence
        _add_section_slide(prs, "Geographic Presence")
        
        hq = facilities.get('headquarters', {})
        geo_content = [
            f"Headquarters: {hq.get('city', 'N/A')}, {hq.get('country', 'N/A')}",
            f"Total Facilities: {facilities.get('total_facilities', 0)}",
        ]
        for facility in facilities.get('facilities', [])[:5]:
            geo_content.append(f"{facility.get('city', '')}, {facility.get('country', '')} - {facility.get('type', 'Manufacturing')}")
        
        _add_content_slide(prs, "Global Footprint", geo_content)
        
    else:
        # Comparison report
        sentiment_data = {"companies": {c['name'].split()[0]: analyze_company_sentiment(c['name'].split()[0]) for c in COMPANIES.values()}}
        investment_data = compare_investment_focus()
        
        _add_title_slide(
            prs,
            "EMS Industry Analysis",
            "Competitive Intelligence Comparison"
        )
        
        # Comparison Table
        headers = ["Company", "Sentiment", "AI Focus", "Classification"]
        rows = []
        for ticker, config in COMPANIES.items():
            company_name = config['name'].split()[0]
            sent = sentiment_data.get('companies', {}).get(company_name, {})
            inv = investment_data.get('companies', {}).get(company_name, {})
            rows.append([
                company_name,
                f"{sent.get('sentiment_score', 0):.0%}",
                f"{inv.get('ai_focus_percentage', 0):.0f}%",
                inv.get('focus', 'N/A')
            ])
        
        _add_comparison_table(prs, "Company Comparison", headers, rows)
        
        # Leaders
        _add_section_slide(prs, "Industry Leaders")
        
        leaders_content = []
        if investment_data.get('ai_leaders'):
            leaders_content.append(f"AI Investment Leader: {investment_data['ai_leaders'][0] if investment_data['ai_leaders'] else 'N/A'}")
        if sentiment_data.get('highest_sentiment'):
            leaders_content.append(f"Highest Sentiment: {sentiment_data['highest_sentiment']}")
        
        _add_content_slide(prs, "Market Leaders", leaders_content)
    
    # Save to bytes
    output = io.BytesIO()
    prs.save(output)
    output.seek(0)
    return output.getvalue()
