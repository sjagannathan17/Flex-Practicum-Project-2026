"""
API routes for geographic analysis and facility mapping.
"""
from fastapi import APIRouter, HTTPException

from backend.analytics.geographic import (
    get_company_facilities,
    get_regional_distribution,
    get_all_facilities_map,
    analyze_regional_investments,
    compare_geographic_footprints,
)

router = APIRouter()


@router.get("/geographic/facilities")
async def get_all_facilities():
    """
    Get all facilities for map visualization.
    Returns facility locations with coordinates for all companies.
    """
    try:
        return get_all_facilities_map()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geographic/facilities/{company}")
async def get_facilities(company: str):
    """
    Get facilities for a specific company.
    """
    try:
        result = get_company_facilities(company)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geographic/distribution/{company}")
async def get_distribution(company: str):
    """
    Get regional distribution for a company.
    """
    try:
        result = get_regional_distribution(company)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geographic/investments/{company}")
async def get_regional_investments(company: str):
    """
    Analyze regional investment mentions for a company.
    """
    try:
        return analyze_regional_investments(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geographic/compare")
async def compare_footprints():
    """
    Compare geographic footprints across all companies.
    """
    try:
        return compare_geographic_footprints()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geographic/heatmap")
async def get_geographic_heatmap():
    """
    Get data formatted for heatmap visualization.
    """
    try:
        all_facilities = get_all_facilities_map()
        comparison = compare_geographic_footprints()
        
        # Format for heatmap
        heatmap_data = []
        for facility in all_facilities["facilities"]:
            heatmap_data.append({
                "lat": facility["lat"],
                "lng": facility["lng"],
                "intensity": 1.0 if facility["is_headquarters"] else 0.5,
                "company": facility["company"],
                "city": facility["city"],
                "type": facility["type"],
            })
        
        return {
            "heatmap_points": heatmap_data,
            "total_facilities": all_facilities["total_count"],
            "regional_leaders": comparison["regional_leaders"],
            "shared_locations": comparison["overlap_analysis"]["shared_locations"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
