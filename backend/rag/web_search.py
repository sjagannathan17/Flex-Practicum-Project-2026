"""
Web search integration using Brave Search API.
"""
import httpx
from typing import Optional
from backend.core.config import BRAVE_API_KEY, BRAVE_SEARCH_URL, WEB_SEARCH_RESULTS


async def search_web(
    query: str,
    count: int = WEB_SEARCH_RESULTS,
) -> list[dict]:
    """
    Search the web using Brave Search API.
    
    Args:
        query: Search query
        count: Number of results to return
        
    Returns:
        List of web search results
    """
    if not BRAVE_API_KEY:
        print("Warning: BRAVE_API_KEY not set, skipping web search")
        return []
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    
    params = {
        "q": query,
        "count": count,
        "safesearch": "moderate",
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                BRAVE_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        web_results = data.get("web", {}).get("results", [])
        
        for result in web_results[:count]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "published": result.get("age", ""),
            })
        
        return results
        
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def search_web_sync(query: str, count: int = WEB_SEARCH_RESULTS) -> list[dict]:
    """Synchronous version of web search for non-async contexts."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(search_web(query, count))


def format_web_results_for_context(results: list[dict]) -> str:
    """Format web results as context for LLM."""
    if not results:
        return ""
    
    parts = []
    for i, result in enumerate(results, 1):
        parts.append(f"[Web {i}: {result['title']}]\n{result['description']}\nURL: {result['url']}")
    
    return "\n\n".join(parts)
