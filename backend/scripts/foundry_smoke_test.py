"""Manual smoke test for Phase 2 of the Azure Foundry migration (see AGENTS.md).

Confirms a basic Microsoft Agent Framework agent, backed by Azure AI Foundry,
can respond. This makes a real network call and needs real Azure credentials,
so it is NOT part of the automated pytest suite.

Setup:
    1. `az login` (or otherwise satisfy DefaultAzureCredential)
    2. Copy backend/.env.example to backend/.env and fill in
       FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MODEL.

Run from the backend/ directory:
    python scripts/foundry_smoke_test.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from services.foundry_service import FoundryService


async def main() -> None:
    load_dotenv()
    service = FoundryService()
    response = await service.ask("Reply with the single word: pong")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
