"""Tool belt for the bounded research loop: approved-knowledge RAG + Tavily enrichment."""

from langchain_core.tools import tool

from .config import TAVILY_API_KEY
from .rag import search_knowledge


@tool
def retrieve_socure_knowledge(query: str) -> str:
    """Search the approved Socure knowledge base (products, case studies, playbooks,
    messaging policy). Returns sections with their KB source ids. Only content
    returned here may be cited as a KB source or used for quantified claims."""
    results = search_knowledge(query)
    if not results:
        return "No approved knowledge found for this query."
    return "\n\n".join(
        f"[{r['source_id']}] {r['doc_title']} (score {r['score']})\n{r['text']}"
        for r in results
    )


@tool
def research_account(query: str) -> str:
    """Search the public web (Tavily) for recent news, initiatives, funding, or
    leadership changes about the prospect account. Cite results as WEB:<n>."""
    if not TAVILY_API_KEY:
        return (
            "Web enrichment unavailable (no TAVILY_API_KEY configured). "
            "Proceed using CRM and knowledge-base context only, and list external "
            "account research under missing_information."
        )
    from langchain_tavily import TavilySearch

    results = TavilySearch(max_results=3).invoke({"query": query})
    items = results.get("results", []) if isinstance(results, dict) else []
    if not items:
        return "No web results found."
    return "\n\n".join(
        f"[WEB:tavily-{i + 1}] {item.get('title', '')} ({item.get('url', '')})\n{item.get('content', '')[:500]}"
        for i, item in enumerate(items)
    )


TOOL_BELT = [retrieve_socure_knowledge, research_account]
