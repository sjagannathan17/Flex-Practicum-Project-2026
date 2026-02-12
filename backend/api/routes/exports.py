"""
API routes for report exports.
Supports Excel, PowerPoint, and PDF exports.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from typing import Optional

from backend.exports.excel import generate_excel_report, generate_comparison_excel
from backend.exports.powerpoint import generate_powerpoint_report
from backend.exports.pdf import generate_pdf_report, generate_html_preview

router = APIRouter()


@router.get("/exports/excel/{company}")
async def export_company_excel(company: str):
    """
    Export company analysis to Excel.
    """
    try:
        excel_bytes = generate_excel_report(company)
        
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={company.lower()}_analysis.xlsx"
            }
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/excel/comparison/all")
async def export_comparison_excel():
    """
    Export company comparison to Excel.
    """
    try:
        excel_bytes = generate_comparison_excel()
        
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=ems_comparison.xlsx"
            }
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/powerpoint/{company}")
async def export_company_powerpoint(company: str):
    """
    Export company analysis to PowerPoint.
    """
    try:
        pptx_bytes = generate_powerpoint_report(company)
        
        return Response(
            content=pptx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": f"attachment; filename={company.lower()}_presentation.pptx"
            }
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/powerpoint/comparison/all")
async def export_comparison_powerpoint():
    """
    Export company comparison to PowerPoint.
    """
    try:
        pptx_bytes = generate_powerpoint_report(None)
        
        return Response(
            content=pptx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={
                "Content-Disposition": "attachment; filename=ems_comparison.pptx"
            }
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/pdf/{company}")
async def export_company_pdf(company: str):
    """
    Export company analysis to PDF.
    """
    try:
        pdf_bytes = generate_pdf_report(company)
        
        # Check if it's actual PDF or HTML fallback
        if pdf_bytes[:4] == b'%PDF':
            media_type = "application/pdf"
            filename = f"{company.lower()}_report.pdf"
        else:
            media_type = "text/html"
            filename = f"{company.lower()}_report.html"
        
        return Response(
            content=pdf_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/pdf/comparison/all")
async def export_comparison_pdf():
    """
    Export company comparison to PDF.
    """
    try:
        pdf_bytes = generate_pdf_report(None)
        
        if pdf_bytes[:4] == b'%PDF':
            media_type = "application/pdf"
            filename = "ems_comparison.pdf"
        else:
            media_type = "text/html"
            filename = "ems_comparison.html"
        
        return Response(
            content=pdf_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/preview/{company}")
async def preview_company_report(company: str):
    """
    Get HTML preview of company report.
    Can be printed to PDF from browser.
    """
    try:
        html = generate_html_preview(company)
        return Response(content=html, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/preview/comparison/all")
async def preview_comparison_report():
    """
    Get HTML preview of comparison report.
    """
    try:
        html = generate_html_preview(None)
        return Response(content=html, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exports/formats")
async def get_available_formats():
    """
    Get list of available export formats.
    """
    try:
        from backend.exports.excel import HAS_OPENPYXL
        from backend.exports.powerpoint import HAS_PPTX
        from backend.exports.pdf import HAS_WEASYPRINT
    except ImportError:
        HAS_OPENPYXL = False
        HAS_PPTX = False
        HAS_WEASYPRINT = False
    
    return {
        "formats": [
            {
                "id": "excel",
                "name": "Excel",
                "extension": ".xlsx",
                "available": HAS_OPENPYXL,
                "description": "Microsoft Excel spreadsheet with multiple sheets",
            },
            {
                "id": "powerpoint",
                "name": "PowerPoint",
                "extension": ".pptx",
                "available": HAS_PPTX,
                "description": "Microsoft PowerPoint presentation",
            },
            {
                "id": "pdf",
                "name": "PDF",
                "extension": ".pdf",
                "available": True,  # HTML fallback always available
                "native_pdf": HAS_WEASYPRINT,
                "description": "PDF report (HTML fallback if WeasyPrint not installed)",
            },
            {
                "id": "html",
                "name": "HTML Preview",
                "extension": ".html",
                "available": True,
                "description": "Web-based preview, can print to PDF from browser",
            },
        ]
    }
