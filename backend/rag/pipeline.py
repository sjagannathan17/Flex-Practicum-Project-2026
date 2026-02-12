"""
Unified RAG pipeline combining retrieval, web search, and generation.
"""
from typing import Optional, Literal
from .retriever import search_documents
from .generator import generate_response
from .web_search import search_web
from .memory import get_conversation_history, add_message

SearchMode = Literal["rag", "web", "hybrid"]


async def process_query(
    query: str,
    mode: SearchMode = "hybrid",
    company_filter: Optional[str] = None,
    session_id: Optional[str] = None,
    include_web: bool = True,
) -> dict:
    """
    Process a user query through the RAG pipeline.
    
    Args:
        query: User's question
        mode: Search mode - "rag" (documents only), "web" (web only), "hybrid" (both)
        company_filter: Optional company to focus on
        session_id: Optional session ID for conversation memory
        include_web: Whether to include web search results
        
    Returns:
        Dict with response, sources, and metadata
    """
    context_chunks = []
    web_results = []
    
    # Get conversation history if session provided
    conversation_history = None
    if session_id:
        conversation_history = get_conversation_history(session_id)
    
    # Retrieve from ChromaDB
    if mode in ("rag", "hybrid"):
        context_chunks = search_documents(
            query=query,
            company_filter=company_filter,
            n_results=10,
        )
    
    # Search web
    if mode in ("web", "hybrid") and include_web:
        # Add company context to web search
        web_query = query
        if company_filter:
            web_query = f"{company_filter} {query}"
        
        web_results = await search_web(web_query, count=5)
    
    # Generate response
    result = generate_response(
        query=query,
        context_chunks=context_chunks,
        web_results=web_results if mode != "rag" else None,
        conversation_history=conversation_history,
    )
    
    # Save to conversation memory if session provided
    if session_id:
        add_message(session_id, "user", query)
        add_message(session_id, "assistant", result["response"])
    
    return {
        "query": query,
        "response": result["response"],
        "sources": result["sources"],
        "mode": mode,
        "company_filter": company_filter,
        "chunks_retrieved": len(context_chunks),
        "web_results_count": len(web_results),
        "session_id": session_id,
        "usage": result.get("usage", {}),
    }


def process_query_sync(
    query: str,
    mode: SearchMode = "hybrid",
    company_filter: Optional[str] = None,
    session_id: Optional[str] = None,
    include_web: bool = True,
) -> dict:
    """Synchronous version of process_query."""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            process_query(query, mode, company_filter, session_id, include_web)
        )
    finally:
        loop.close()


def get_quick_answer(query: str, company: Optional[str] = None) -> str:
    """Get a quick answer without session tracking."""
    result = process_query_sync(query, mode="rag", company_filter=company)
    return result["response"]
