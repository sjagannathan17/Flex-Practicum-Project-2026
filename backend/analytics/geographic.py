"""
Geographic analysis for EMS company facilities.
Provides facility mapping and regional investment analysis.
"""
from typing import Optional
from collections import defaultdict

from backend.rag.retriever import search_documents


# Known EMS facility locations (from public filings and company information)
KNOWN_FACILITIES = {
    "Flex": {
        "headquarters": {"city": "Singapore", "country": "Singapore", "lat": 1.3521, "lng": 103.8198},
        "facilities": [
            {"city": "Guadalajara", "country": "Mexico", "lat": 20.6597, "lng": -103.3496, "type": "Manufacturing"},
            {"city": "Austin", "country": "USA", "lat": 30.2672, "lng": -97.7431, "type": "Design Center"},
            {"city": "Shanghai", "country": "China", "lat": 31.2304, "lng": 121.4737, "type": "Manufacturing"},
            {"city": "Penang", "country": "Malaysia", "lat": 5.4141, "lng": 100.3288, "type": "Manufacturing"},
            {"city": "Chennai", "country": "India", "lat": 13.0827, "lng": 80.2707, "type": "Design Center"},
            {"city": "Zhuhai", "country": "China", "lat": 22.2769, "lng": 113.5678, "type": "Manufacturing"},
            {"city": "Sorocaba", "country": "Brazil", "lat": -23.5015, "lng": -47.4526, "type": "Manufacturing"},
            {"city": "Timisoara", "country": "Romania", "lat": 45.7489, "lng": 21.2087, "type": "Manufacturing"},
            {"city": "Suzhou", "country": "China", "lat": 31.2989, "lng": 120.5853, "type": "Manufacturing"},
        ]
    },
    "Jabil": {
        "headquarters": {"city": "St. Petersburg", "country": "USA", "lat": 27.7676, "lng": -82.6403},
        "facilities": [
            {"city": "Penang", "country": "Malaysia", "lat": 5.4141, "lng": 100.3288, "type": "Manufacturing"},
            {"city": "Wuxi", "country": "China", "lat": 31.4912, "lng": 120.3119, "type": "Manufacturing"},
            {"city": "Chihuahua", "country": "Mexico", "lat": 28.6353, "lng": -106.0889, "type": "Manufacturing"},
            {"city": "Livingston", "country": "UK", "lat": 55.9024, "lng": -3.5159, "type": "Manufacturing"},
            {"city": "Budapest", "country": "Hungary", "lat": 47.4979, "lng": 19.0402, "type": "Manufacturing"},
            {"city": "Guadalajara", "country": "Mexico", "lat": 20.6597, "lng": -103.3496, "type": "Manufacturing"},
            {"city": "Shenzhen", "country": "China", "lat": 22.5431, "lng": 114.0579, "type": "Manufacturing"},
            {"city": "San Jose", "country": "USA", "lat": 37.3382, "lng": -121.8863, "type": "Design Center"},
        ]
    },
    "Celestica": {
        "headquarters": {"city": "Toronto", "country": "Canada", "lat": 43.6532, "lng": -79.3832},
        "facilities": [
            {"city": "Monterrey", "country": "Mexico", "lat": 25.6866, "lng": -100.3161, "type": "Manufacturing"},
            {"city": "Suzhou", "country": "China", "lat": 31.2989, "lng": 120.5853, "type": "Manufacturing"},
            {"city": "Kulim", "country": "Malaysia", "lat": 5.3717, "lng": 100.5627, "type": "Manufacturing"},
            {"city": "Valencia", "country": "Spain", "lat": 39.4699, "lng": -0.3763, "type": "Manufacturing"},
            {"city": "Oradea", "country": "Romania", "lat": 47.0458, "lng": 21.9189, "type": "Manufacturing"},
            {"city": "Portland", "country": "USA", "lat": 45.5051, "lng": -122.6750, "type": "Design Center"},
            {"city": "Fremont", "country": "USA", "lat": 37.5485, "lng": -121.9886, "type": "Manufacturing"},
        ]
    },
    "Benchmark": {
        "headquarters": {"city": "Tempe", "country": "USA", "lat": 33.4255, "lng": -111.9400},
        "facilities": [
            {"city": "Rochester", "country": "USA", "lat": 43.1566, "lng": -77.6088, "type": "Manufacturing"},
            {"city": "Angleton", "country": "USA", "lat": 29.1694, "lng": -95.4316, "type": "Manufacturing"},
            {"city": "Suzhou", "country": "China", "lat": 31.2989, "lng": 120.5853, "type": "Manufacturing"},
            {"city": "Penang", "country": "Malaysia", "lat": 5.4141, "lng": 100.3288, "type": "Manufacturing"},
            {"city": "Guadalajara", "country": "Mexico", "lat": 20.6597, "lng": -103.3496, "type": "Manufacturing"},
            {"city": "Amsterdam", "country": "Netherlands", "lat": 52.3676, "lng": 4.9041, "type": "Design Center"},
        ]
    },
    "Sanmina": {
        "headquarters": {"city": "San Jose", "country": "USA", "lat": 37.3382, "lng": -121.8863},
        "facilities": [
            {"city": "Guadalajara", "country": "Mexico", "lat": 20.6597, "lng": -103.3496, "type": "Manufacturing"},
            {"city": "Shenzhen", "country": "China", "lat": 22.5431, "lng": 114.0579, "type": "Manufacturing"},
            {"city": "Chennai", "country": "India", "lat": 13.0827, "lng": 80.2707, "type": "Manufacturing"},
            {"city": "Kecskemet", "country": "Hungary", "lat": 46.8963, "lng": 19.6897, "type": "Manufacturing"},
            {"city": "Wuxi", "country": "China", "lat": 31.4912, "lng": 120.3119, "type": "Manufacturing"},
            {"city": "Kunshan", "country": "China", "lat": 31.3847, "lng": 120.9837, "type": "Manufacturing"},
            {"city": "Fremont", "country": "USA", "lat": 37.5485, "lng": -121.9886, "type": "Manufacturing"},
        ]
    },
}


def get_company_facilities(company: str) -> dict:
    """Get facility information for a company."""
    company_data = KNOWN_FACILITIES.get(company)
    
    if not company_data:
        return {"company": company, "error": "Company not found"}
    
    return {
        "company": company,
        "headquarters": company_data["headquarters"],
        "facilities": company_data["facilities"],
        "total_facilities": len(company_data["facilities"]) + 1,  # +1 for HQ
    }


def get_regional_distribution(company: str) -> dict:
    """Get regional distribution of facilities."""
    company_data = KNOWN_FACILITIES.get(company)
    
    if not company_data:
        return {"company": company, "error": "Company not found"}
    
    regions = {
        "Americas": ["USA", "Canada", "Mexico", "Brazil"],
        "EMEA": ["UK", "Hungary", "Romania", "Netherlands", "Spain", "Germany"],
        "APAC": ["China", "Malaysia", "Singapore", "India", "Japan", "Taiwan"],
    }
    
    distribution = {"Americas": 0, "EMEA": 0, "APAC": 0}
    
    # Count HQ
    hq_country = company_data["headquarters"]["country"]
    for region, countries in regions.items():
        if hq_country in countries:
            distribution[region] += 1
            break
    
    # Count facilities
    for facility in company_data["facilities"]:
        country = facility["country"]
        for region, countries in regions.items():
            if country in countries:
                distribution[region] += 1
                break
    
    total = sum(distribution.values())
    percentages = {k: round(v / total * 100, 1) if total > 0 else 0 for k, v in distribution.items()}
    
    return {
        "company": company,
        "distribution": distribution,
        "percentages": percentages,
        "primary_region": max(distribution, key=distribution.get),
    }


def get_all_facilities_map() -> dict:
    """Get all facilities for map visualization."""
    all_facilities = []
    
    for company, data in KNOWN_FACILITIES.items():
        # Add headquarters
        hq = data["headquarters"]
        all_facilities.append({
            "company": company,
            "city": hq["city"],
            "country": hq["country"],
            "lat": hq["lat"],
            "lng": hq["lng"],
            "type": "Headquarters",
            "is_headquarters": True,
        })
        
        # Add facilities
        for facility in data["facilities"]:
            all_facilities.append({
                "company": company,
                "city": facility["city"],
                "country": facility["country"],
                "lat": facility["lat"],
                "lng": facility["lng"],
                "type": facility["type"],
                "is_headquarters": False,
            })
    
    return {
        "facilities": all_facilities,
        "total_count": len(all_facilities),
        "by_company": {company: len(data["facilities"]) + 1 for company, data in KNOWN_FACILITIES.items()},
    }


def analyze_regional_investments(company: str) -> dict:
    """
    Analyze regional investment mentions in company filings.
    """
    # Search for regional investment content
    regions_to_search = ["Americas", "Asia", "Europe", "Mexico", "China", "Malaysia", "India"]
    
    regional_mentions = {}
    
    for region in regions_to_search:
        docs = search_documents(
            query=f"{company} {region} investment expansion facility manufacturing",
            company_filter=company,
            n_results=20,
        )
        
        regional_mentions[region] = {
            "mentions": len(docs),
            "sample_context": docs[0]["content"][:200] if docs else None,
        }
    
    # Determine investment focus
    sorted_regions = sorted(regional_mentions.items(), key=lambda x: x[1]["mentions"], reverse=True)
    top_regions = sorted_regions[:3]
    
    return {
        "company": company,
        "regional_mentions": regional_mentions,
        "investment_focus": [{"region": r, "mentions": m["mentions"]} for r, m in top_regions],
        "primary_focus": top_regions[0][0] if top_regions else None,
    }


def compare_geographic_footprints() -> dict:
    """
    Compare geographic footprints across all companies.
    """
    companies = ["Flex", "Jabil", "Celestica", "Benchmark", "Sanmina"]
    
    results = {
        "companies": [],
        "regional_leaders": {
            "Americas": None,
            "EMEA": None,
            "APAC": None,
        },
        "overlap_analysis": {},
    }
    
    regional_counts = {"Americas": {}, "EMEA": {}, "APAC": {}}
    
    for company in companies:
        distribution = get_regional_distribution(company)
        facilities = get_company_facilities(company)
        
        if "error" not in distribution:
            results["companies"].append({
                "company": company,
                "total_facilities": facilities.get("total_facilities", 0),
                "regional_distribution": distribution["distribution"],
                "primary_region": distribution["primary_region"],
            })
            
            # Track for leaders
            for region, count in distribution["distribution"].items():
                regional_counts[region][company] = count
    
    # Determine regional leaders
    for region, counts in regional_counts.items():
        if counts:
            leader = max(counts, key=counts.get)
            results["regional_leaders"][region] = {
                "company": leader,
                "count": counts[leader],
            }
    
    # Analyze overlap (cities with multiple companies)
    city_companies = defaultdict(list)
    for company, data in KNOWN_FACILITIES.items():
        for facility in data["facilities"]:
            city_companies[facility["city"]].append(company)
    
    overlap_cities = {city: companies for city, companies in city_companies.items() if len(companies) > 1}
    results["overlap_analysis"] = {
        "shared_locations": len(overlap_cities),
        "locations": overlap_cities,
    }
    
    return results
