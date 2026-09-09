from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any


def document_id_for_url(url: str) -> str:
    """A stable, Azure-Search-key-safe id for a URL (re-scraping the same URL
    later updates the same document instead of duplicating it)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _default_client_factory(api_key: str) -> Any:
    from firecrawl import AsyncV1FirecrawlApp

    return AsyncV1FirecrawlApp(api_key=api_key)


class FirecrawlService:
    """Thin wrapper around Firecrawl's search+scrape, for economic-news RAG
    ingestion (Phase 6's "economic news" / "Firecrawl content" sources for
    the economic-documents index).

    The SDK is lazily imported and the client is injectable, so this module
    stays importable and testable without firecrawl-py installed or a live
    API key.
    """

    def __init__(self, api_key: str, client_factory: Callable[[str], Any] = _default_client_factory):
        self._api_key = api_key
        self._client_factory = client_factory
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self._api_key)
        return self._client

    async def search_economic_news(self, query: str, limit: int = 5) -> list[dict]:
        """Search + scrape recent web content for `query`.

        Returns [{"id", "title", "content", "source"}, ...] shaped for
        AzureSearchService.index_economic_documents. Results with no
        scraped content (e.g. a blocked page) are skipped.
        """
        from firecrawl.v1.client import V1ScrapeOptions

        client = self._get_client()
        response = await client.search(query, limit=limit, scrape_options=V1ScrapeOptions(formats=["markdown"]))
        documents = []
        for result in response.data:
            content = result.get("markdown") or result.get("description") or ""
            if not content:
                continue
            documents.append({
                "id": document_id_for_url(result["url"]),
                "title": result.get("title", ""),
                "content": content,
                "source": result["url"],
            })
        return documents
