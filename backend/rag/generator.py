"""
LLM response generation using Claude API.
"""
import anthropic
from typing import Optional
from backend.core.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS


def get_claude_client():
    """Get Anthropic client."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def generate_response(
    query: str,
    context_chunks: list[dict],
    web_results: Optional[list[dict]] = None,
    conversation_history: Optional[list[dict]] = None,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Generate a response using Claude with RAG context.
    
    Args:
        query: User's question
        context_chunks: Retrieved document chunks from ChromaDB
        web_results: Optional web search results
        conversation_history: Optional previous messages for multi-turn
        system_prompt: Optional custom system prompt
        
    Returns:
        Dict with response text and citations
    """
    client = get_claude_client()
    
    # Build context from retrieved chunks
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(context_chunks, 1):
        source_info = f"[{chunk['company']}] {chunk['source']} ({chunk['filing_type']}, {chunk['fiscal_year']} {chunk['quarter']})"
        sources.append({
            "id": i,
            "source": chunk["source"],
            "company": chunk["company"],
            "filing_type": chunk["filing_type"],
            "fiscal_year": chunk["fiscal_year"],
            "quarter": chunk["quarter"],
            "similarity": chunk["similarity"],
        })
        context_parts.append(f"[Source {i}: {source_info}]\n{chunk['content']}")
    
    # Add web results if available
    web_sources = []
    if web_results:
        for i, result in enumerate(web_results, len(sources) + 1):
            web_sources.append({
                "id": i,
                "title": result.get("title", "Web Result"),
                "url": result.get("url", ""),
                "type": "web",
            })
            context_parts.append(f"[Web Source {i}: {result.get('title', 'Web')}]\n{result.get('description', '')}")
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Default system prompt
    if not system_prompt:
        system_prompt = """You are an expert financial analyst specializing in the Electronics Manufacturing Services (EMS) industry. 
You analyze capital expenditure strategies, competitive dynamics, and industry trends for companies including Flex, Jabil, Celestica, Benchmark, and Sanmina.

Your task is to answer questions based on the provided document context. Follow these guidelines:

1. **ALWAYS extract and present data**: Even if exact figures aren't available, extract ALL relevant information from the context. Never say "insufficient data" if there's ANY related information.

2. **Use sources**: Always cite specific sources when making claims. Use [Source N] format.

3. **Be specific**: Provide concrete numbers, dates, percentages, and details when available. Include:
   - Total CapEx figures ($ amounts and % of revenue)
   - Revenue by segment
   - Growth rates and YoY comparisons
   - Facility expansions and locations
   - Executive quotes about strategy

4. **Synthesize insights**: If exact data isn't available, derive insights from related data. For example:
   - If AI CapEx % isn't stated, discuss the AI-related segment revenue growth
   - If investment amounts aren't given, describe announced initiatives and partnerships
   - Connect qualitative statements to quantitative context

5. **Compare across companies**: When data exists for multiple companies, create comparisons even if not explicitly asked.

6. **Structure your response**: Use headers, bullet points, and tables to organize information clearly.

7. **Distinguish AI vs Traditional**: When discussing CapEx, differentiate between AI/data center investments and traditional manufacturing.

8. **Only acknowledge gaps briefly**: If specific data is missing, mention it briefly at the end, then suggest what related data IS available.

Focus on capital expenditure, investments, facility expansions, AI initiatives, data centers, and competitive positioning."""

    # Build messages
    messages = []
    
    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current query with context
    user_message = f"""Based on the following document context, please answer this question:

**Question:** {query}

---

**Document Context:**

{context_text}

---

Please provide a comprehensive answer with source citations."""

    messages.append({"role": "user", "content": user_message})
    
    # Call Claude
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    )
    
    response_text = response.content[0].text
    
    return {
        "response": response_text,
        "sources": sources + web_sources,
        "model": LLM_MODEL,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    }


def generate_summary(documents: list[dict], focus: str = "capital expenditure") -> str:
    """Generate a summary of multiple documents."""
    client = get_claude_client()
    
    # Combine document content
    content_parts = []
    for doc in documents[:10]:  # Limit to avoid token overflow
        content_parts.append(f"[{doc.get('company', 'Unknown')}] {doc.get('content', '')[:500]}")
    
    combined = "\n\n---\n\n".join(content_parts)
    
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        system="You are a financial analyst. Provide concise, factual summaries.",
        messages=[{
            "role": "user",
            "content": f"Summarize the following documents focusing on {focus}:\n\n{combined}"
        }]
    )
    
    return response.content[0].text
