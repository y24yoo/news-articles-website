"""Manual smoke test for Phase 4 of the Azure Foundry migration (see AGENTS.md).

Confirms the fred-series/fred-observations/economic-documents indexes can be
created against a real Azure AI Search resource, then migrates CPI data from
MongoDB into Azure AI Search and runs one semantic search against it. This
makes real network calls and needs real Azure credentials + a running
MongoDB with CPI data already ingested (see fred.py's cpi()/selector_cpi()),
so it is NOT part of the automated pytest suite.

Setup:
    1. `az login` (or otherwise satisfy DefaultAzureCredential)
    2. Fill in AZURE_SEARCH_ENDPOINT in backend/.env (copy from .env.example)
    3. Have MongoDB running with CPI data already populated

Run from the backend/ directory:
    python scripts/azure_search_smoke_test.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from fred import azure_search_service, sync_cpi_to_azure_search


async def main() -> None:
    load_dotenv()
    await azure_search_service.ensure_indexes()
    await sync_cpi_to_azure_search()
    results = await azure_search_service.search_fred_series("important inflation indicators", category="cpi")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
