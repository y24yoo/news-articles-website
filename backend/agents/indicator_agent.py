from __future__ import annotations

from fred import get_fred_observations, search_fred_indicators
from services.foundry_service import FoundryService

INSTRUCTIONS = (
    "You are the IndicatorAgent for an economic intelligence platform. "
    "Given a FRED category, use search_fred_indicators to find the best "
    "indicators representing it, and get_fred_observations to check their "
    "recent data. Select the best 2-3 indicators and explain why."
)

indicator_agent = FoundryService(
    name="indicator-agent",
    instructions=INSTRUCTIONS,
    tools=[search_fred_indicators.coroutine, get_fred_observations.coroutine],
)


async def select_indicators(category: str) -> str:
    """Select and justify the best FRED indicators for `category`."""
    return await indicator_agent.ask(
        f"Select the best 2-3 indicators to represent the '{category}' category. "
        f"Use search_fred_indicators with category='{category}'. Explain your choices."
    )
