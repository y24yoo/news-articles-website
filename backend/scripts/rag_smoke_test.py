"""Manual smoke test for Phase 6 of the Azure Foundry migration (see AGENTS.md).

Confirms economic-documents RAG works end to end: search + scrape recent
economic news via Firecrawl, embed and index it into Azure AI Search's
economic-documents index, then run one hybrid (keyword + vector + semantic)
search against it. This makes real network calls and needs real Azure,
Foundry embedding, and Firecrawl credentials, so it is NOT part of the
automated pytest suite.

Setup:
    1. `az login` (or otherwise satisfy DefaultAzureCredential)
    2. Fill in AZURE_SEARCH_ENDPOINT, FOUNDRY_MODELS_ENDPOINT,
       FOUNDRY_EMBEDDING_MODEL, and FIRECRAWL_API_KEY in backend/.env
       (copy from .env.example) and create backend/api_key.py per the README

Run from the backend/ directory:
    python scripts/rag_smoke_test.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from fred import azure_search_service, index_economic_news


async def main() -> None:
    load_dotenv()
    await azure_search_service.ensure_indexes()
    await index_economic_news("latest inflation and interest rate news")
    results = await azure_search_service.search_economic_documents("federal reserve interest rate decision")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
