"""
API routes for report exports.
"""
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.exports.excel import generate_excel_report, generate_comparison_excel
from backend.exports.powerpoint import generate_powerpoint_report

try:
    from backend.exports.pdf import generate_pdf_report, generate_html_preview
    HAS_PDF = True
except (ImportError, OSError):
    generate_pdf_report = None
    generate_html_preview = None
    HAS_PDF = False

router = APIRouter()


@router.get("/exports/excel/{company}")
async def export_company_excel(company: str):
    try:
        excel_bytes = generate_excel_report(company)
        return Response(content=excel_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f"attachment; filename={company.lower()}_analysis.xlsx"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/excel/comparison/all")
async def export_comparison_excel():
    try:
        excel_bytes = generate_comparison_excel()
        return Response(content=excel_bytes, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": "attachment; filename=ems_comparison.xlsx"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/powerpoint/comparison/all")
async def export_comparison_powerpoint():
    """Generate AI-powered full comparison PowerPoint (all 5 companies)."""
    try:
        pptx_bytes = generate_powerpoint_report(company=None)
        return Response(
            content=pptx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": "attachment; filename=flex_competitive_intelligence_brief.pptx"},
        )
    except ValueError as e:
        # User-safe parse error from LLM pipeline
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/powerpoint/{company}")
async def export_company_powerpoint(company: str):
    """Generate AI-powered single-company PowerPoint report."""
    try:
        pptx_bytes = generate_powerpoint_report(company)
        safe_name = re.sub(r"[^\w\-]", "_", company.lower())
        return Response(
            content=pptx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f"attachment; filename={safe_name}_competitive_brief.pptx"},
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/pdf/{company}")
async def export_company_pdf(company: str):
    if not generate_pdf_report:
        raise HTTPException(status_code=501, detail="PDF export unavailable: WeasyPrint not installed")
    try:
        pdf_bytes = generate_pdf_report(company)
        mt = "application/pdf" if pdf_bytes[:4] == b'%PDF' else "text/html"
        return Response(content=pdf_bytes, media_type=mt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/preview/{company}")
async def preview_company_report(company: str):
    if not generate_html_preview:
        raise HTTPException(status_code=501, detail="Preview unavailable")
    try:
        return Response(content=generate_html_preview(company), media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/formats")
async def get_available_formats():
    return {"formats": [
        {"id": "excel", "name": "Excel", "extension": ".xlsx", "available": True},
        {"id": "powerpoint", "name": "PowerPoint", "extension": ".pptx", "available": True},
        {"id": "pdf", "name": "PDF", "extension": ".pdf", "available": HAS_PDF},
        {"id": "html", "name": "HTML Preview", "extension": ".html", "available": True},
    ]}
