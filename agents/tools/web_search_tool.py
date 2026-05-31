"""
Web search tool using DuckDuckGo.

Provides supplementary information from the web when the textbook
content is insufficient. Returns top results with titles, snippets,
and URLs.
"""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information related to a Computer Architecture topic.

    Use this tool when the textbook content does not cover the user's
    question, or when the user asks about recent developments, specific
    hardware products, benchmarks, or topics outside the textbook's scope.

    Results come from DuckDuckGo and include title, snippet, and URL.

    Args:
        query: A search query string.
               Example: 'Apple M2 chip architecture details'

    Returns:
        Top 3 search results formatted with title, snippet, and URL,
        or an error message if the search fails.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return (
            "Web search unavailable: the 'duckduckgo-search' package is not installed. "
            "Install it with: pip install duckduckgo-search"
        )

    try:
        search_query = query.strip()
        if not search_query:
            return "Error: Empty search query. Please provide a search term."

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=3):
                results.append(r)

        if not results:
            return f"No web results found for: '{search_query}'"

        # Format results
        formatted_parts: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            snippet = r.get("body", r.get("snippet", "No snippet available"))
            url = r.get("href", r.get("link", "No URL"))
            formatted_parts.append(
                f"**Result {i}: {title}**\n"
                f"{snippet}\n"
                f"🔗 {url}"
            )

        return (
            f"Web Search Results for: '{search_query}'\n"
            f"{'=' * 50}\n\n"
            + "\n\n---\n\n".join(formatted_parts)
        )

    except Exception as e:
        error_name = type(e).__name__
        if "Ratelimit" in error_name or "ratelimit" in str(e).lower():
            return "Web search rate-limited. Please wait a moment and try again."
        if "Timeout" in error_name or "timeout" in str(e).lower():
            return "Web search timed out. Please try again with a simpler query."
        return f"Web search error: {error_name}: {e}"
