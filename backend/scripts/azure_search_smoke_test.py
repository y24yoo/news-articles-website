"""Manual smoke test for Phase 5 of the Azure Foundry migration (see AGENTS.md).

Confirms the fred-series/fred-observations indexes can be created against a
real Azure AI Search resource, ingests CPI indicator metadata directly into
fred-series (no MongoDB involved -- it's been removed), and runs one
semantic search against it. This makes real network calls and needs real
Azure + FRED credentials, so it is NOT part of the automated pytest suite.

Setup:
    1. `az login` (or otherwise satisfy DefaultAzureCredential)
    2. Fill in AZURE_SEARCH_ENDPOINT and FRED_API_KEY in backend/.env
       (copy from .env.example) and create backend/api_key.py per the README

Run from the backend/ directory:
    python scripts/azure_search_smoke_test.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from fred import azure_search_service, cpi


async def main() -> None:
    load_dotenv()
    await azure_search_service.ensure_indexes()
    await cpi()
    results = await azure_search_service.search_fred_series("important inflation indicators", category="cpi")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
