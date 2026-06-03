import asyncio
import logging

from backend.core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

logger = logging.getLogger(__name__)


async def ask_claude_with_search(question: str) -> str:
    """
    Send question to Claude with the web search tool enabled (live grounding),
    mirroring the Gemini + Google Search contract. Returns the model's answer
    text. Raises on failure.

    The Anthropic web_search tool runs server-side, so a single messages.create
    call returns the final answer — no manual tool-use loop is required.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured in backend/.env")
    # The Anthropic SDK call is blocking; run it off the event loop.
    return await asyncio.to_thread(_ask_claude_with_search_sync, question)


def _ask_claude_with_search_sync(question: str) -> str:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        model = ANTHROPIC_MODEL if "claude" in ANTHROPIC_MODEL else "claude-sonnet-4-6"

        response = client.messages.create(
            model=model,
            max_tokens=8192,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
            messages=[{"role": "user", "content": question}],
        )

        # The answer arrives as one or more text blocks, interleaved with
        # server-side web_search tool-use / tool-result blocks. Keep only text.
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(text_parts).strip()
    except Exception as exc:
        logger.error("Claude web-search call failed: %s", exc)
        raise
