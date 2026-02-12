"""
Simple in-memory cache for expensive computations.
Reduces repeated API calls and analytics calculations.
"""
from datetime import datetime
from typing import Any, Optional
from functools import wraps
import hashlib
import json


class SimpleCache:
    """Thread-safe in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: dict = {}
        self._timestamps: dict = {}
        self._default_ttl = default_ttl  # 5 minutes default
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key not in self._cache:
            return None
        
        timestamp = self._timestamps.get(key, 0)
        if datetime.now().timestamp() - timestamp > self._default_ttl:
            # Expired
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value with optional custom TTL."""
        self._cache[key] = value
        self._timestamps[key] = datetime.now().timestamp()
    
    def delete(self, key: str):
        """Delete cached value."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
        self._timestamps.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "keys": list(self._cache.keys())[:10],
        }


# Global cache instances with different TTLs
# Analytics data changes infrequently, so longer TTL
analytics_cache = SimpleCache(default_ttl=1800)  # 30 min for analytics (sentiment, trends, etc.)
api_cache = SimpleCache(default_ttl=600)  # 10 min for API responses (dashboard)
search_cache = SimpleCache(default_ttl=1800)  # 30 min for document searches
chat_cache = SimpleCache(default_ttl=600)  # 10 min for chat responses


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(cache: SimpleCache, prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{prefix}:{cache_key(*args, **kwargs)}"
            result = cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator


def async_cached(cache: SimpleCache, prefix: str = ""):
    """Decorator to cache async function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{prefix}:{cache_key(*args, **kwargs)}"
            result = cache.get(key)
            if result is not None:
                return result
            result = await func(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator
