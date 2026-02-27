"""
LLM generation module for RAG pipeline.
Handles response generation using Anthropic Claude with context from retrieved documents.
"""
from typing import Optional
import anthropic

from backend.core.config import ANTHROPIC_API_KEY, LLM_MODEL


SYSTEM_PROMPT = """You are an expert competitive intelligence analyst specializing in the Electronics Manufacturing Services (EMS) industry. You analyze SEC filings, earnings transcripts, and financial data for Flex, Jabil, Celestica, Benchmark Electronics, and Sanmina.

Rules:
- Answer directly and concisely based on the provided context.
- Use tables when comparing companies or presenting multi-row data.
- Do NOT use inline citations like [1] or (Source: ...). The user knows the data comes from SEC filings.
- When data is unavailable, say so clearly rather than guessing.
- Use bullet points for lists, bold for emphasis, and keep responses focused.
- For financial figures, always include the unit (millions, billions) and fiscal period.

=== CapEx / Capital Expenditure Extraction Rules ===

LABELS — Different companies use different labels for the same line item:
  * "Purchases of property and equipment" (Flex)
  * "Acquisition of property, plant and equipment" (Jabil)
  * "Purchase of property, plant and equipment" (Celestica)
  * "Purchases of property, plant and equipment" (Benchmark)
  * "Capital expenditures" (Sanmina)
  * "Additions to property and equipment"
  * "Capital spending"
  * "Payments for property and equipment"
All of these refer to CapEx. Look for any of them in the context.

YTD vs SINGLE-QUARTER (CRITICAL for 10-Q):
Quarterly reports (10-Q) show TWO sets of columns:
  "Three Months Ended ..."   → SINGLE quarter   ← EXTRACT THIS ONE
  "Six Months Ended ..."     → Year-to-date     ← DO NOT USE
  "Nine Months Ended ..."    → Year-to-date     ← DO NOT USE
Always extract the "Three Months Ended" value (the single-quarter figure).

NEGATIVE NUMBERS:
CapEx appears as negative in cash flow statements because it is a cash outflow.
Values like $(505), (505), -505, or −130 are all positive CapEx amounts.
Always report the absolute value.

UNIT HEADERS:
Check the unit header at the top of financial statements:
  * "(in thousands)" → divide by 1,000 to get millions
  * "(in millions)" → values are already in millions
  * "(in billions)" → values are already in billions
Benchmark and Sanmina typically report in thousands.
Flex, Jabil, and Celestica typically report in millions.

=== Chain of Thought ===

For numerical or factual questions, reason step-by-step before answering:
1. Identify what specific information is being asked (which company, metric, period)
2. Search through the context for relevant data points
3. Verify the data matches the question (right company, right period, right metric)
4. Check for pitfalls: YTD vs quarterly, unit conversion (thousands vs millions), negative signs
5. Formulate your answer with proper units and context

If the exact data is NOT found in the context:
- State clearly that the specific data was not found
- Do NOT hallucinate or make up numbers
- If similar or related data exists, mention it but clarify it is not an exact match"""


def _build_prompt(query: str, context: str, web_context: str = "") -> str:
    """Build the user prompt combining query, RAG context, and optional web results."""
    parts = []
    if context:
        parts.append(f"## Retrieved Documents\n{context}")
    if web_context:
        parts.append(f"## Web Search Results\n{web_context}")
    parts.append(f"## Question\n{query}")
    return "\n\n".join(parts)


def generate_response(
    query: str,
    context: str,
    web_context: str = "",
) -> str:
    """
    Generate a response using Claude (blocking call).

    Args:
        query: The user question
        context: Retrieved document context
        web_context: Optional web search context

    Returns:
        Generated response text
    """
    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY is not configured."

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_prompt = _build_prompt(query, context, web_context)

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating response: {e}"


def generate_response_streaming(
    query: str,
    context: str,
    web_context: str = "",
):
    """
    Generate a streaming response using Claude.

    Yields text chunks as they arrive from the API. Uses
    client.messages.stream with max_tokens=2000, no extended thinking.
    """
    if not ANTHROPIC_API_KEY:
        yield "Error: ANTHROPIC_API_KEY is not configured."
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_prompt = _build_prompt(query, context, web_context)

    try:
        with client.messages.stream(
            model=LLM_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"\n\nError during streaming: {e}"


def generate_summary(text: str) -> str:
    """
    Generate a brief summary of the given text.
    Useful for summarising long document chunks.
    """
    if not ANTHROPIC_API_KEY:
        return text[:500] + "..."

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system="Summarize the following financial/business text in 2-3 concise sentences.",
            messages=[{"role": "user", "content": text[:8000]}],
        )
        return response.content[0].text
    except Exception as e:
        return text[:500] + "..."
