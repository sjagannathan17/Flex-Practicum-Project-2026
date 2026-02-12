"""
Chat API endpoints for RAG-based Q&A.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import uuid
import hashlib

from backend.rag.pipeline import process_query
from backend.rag.memory import (
    get_session_info, 
    clear_session, 
    get_all_sessions,
    get_conversation_history,
    cleanup_expired_sessions,
)
from backend.core.cache import SimpleCache

router = APIRouter()

# Cache for chat responses (5 minute TTL)
_chat_cache = SimpleCache(default_ttl=300)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str = Field(..., description="User's question", min_length=1)
    mode: Literal["rag", "web", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: rag (documents), web (internet), hybrid (both)"
    )
    company_filter: Optional[str] = Field(
        default=None,
        description="Filter to specific company (Flex, Jabil, Celestica, Benchmark, Sanmina)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory"
    )
    include_web: bool = Field(
        default=True,
        description="Include web search results in hybrid mode"
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    query: str
    response: str
    sources: list[dict]
    mode: str
    company_filter: Optional[str]
    session_id: Optional[str]
    chunks_retrieved: int
    web_results_count: int
    usage: dict


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat query using RAG pipeline.
    
    Supports three modes:
    - **rag**: Search only internal documents (SEC filings, earnings calls, etc.)
    - **web**: Search only the web for real-time information
    - **hybrid**: Combine both document and web search (recommended)
    
    Use `session_id` to maintain conversation context across multiple queries.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create cache key for identical queries (only for new sessions)
        cache_key = hashlib.md5(
            f"{request.query}:{request.mode}:{request.company_filter}:{request.include_web}".encode()
        ).hexdigest()
        
        # Check cache for identical queries without session context
        if not request.session_id:
            cached = _chat_cache.get(cache_key)
            if cached:
                cached["session_id"] = session_id  # Assign new session
                return ChatResponse(**cached)
        
        result = await process_query(
            query=request.query,
            mode=request.mode,
            company_filter=request.company_filter,
            session_id=session_id,
            include_web=request.include_web,
        )
        
        # Cache result for future identical queries
        if not request.session_id:
            _chat_cache.set(cache_key, result)
        
        return ChatResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/chat/session/{session_id}")
async def get_session(session_id: str):
    """Get information about a chat session."""
    try:
        info = get_session_info(session_id)
        return info
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a chat session's history."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.post("/chat/quick")
async def quick_chat(query: str, company: Optional[str] = None):
    """
    Quick chat without session tracking.
    Uses RAG mode only for faster responses.
    """
    try:
        result = await process_query(
            query=query,
            mode="rag",
            company_filter=company,
            session_id=None,
            include_web=False,
        )
        return {
            "query": query,
            "response": result["response"],
            "sources": result["sources"][:5],  # Limit sources for quick response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions")
async def list_sessions():
    """Get all active chat sessions."""
    sessions = get_all_sessions()
    return {
        "sessions": sessions,
        "count": len(sessions),
    }


@router.get("/chat/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get conversation history for a session."""
    try:
        info = get_session_info(session_id)
        messages = get_conversation_history(session_id)
        return {
            **info,
            "messages": messages,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/chat/cleanup")
async def cleanup_sessions():
    """Clean up expired sessions."""
    count = cleanup_expired_sessions()
    return {
        "cleaned_up": count,
        "message": f"Removed {count} expired sessions",
    }
