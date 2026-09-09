from __future__ import annotations

from fred import search_economic_documents
from services.foundry_service import FoundryService

INSTRUCTIONS = (
    "You are the ResearchAgent for an economic intelligence platform. "
    "Given a topic, use search_economic_documents to find relevant economic "
    "research and news, and summarize the key findings."
)

research_agent = FoundryService(
    name="research-agent",
    instructions=INSTRUCTIONS,
    tools=[search_economic_documents.coroutine],
)


async def research(topic: str) -> str:
    """Find and summarize relevant economic research/news for `topic`."""
    return await research_agent.ask(
        f"Find and summarize the most relevant recent economic research and news about '{topic}'."
    )
