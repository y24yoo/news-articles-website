from types import SimpleNamespace

import pytest
from services.firecrawl_service import FirecrawlService, document_id_for_url


class FakeFirecrawlClient:
    def __init__(self, results):
        self._results = results
        self.search_calls = []

    async def search(self, query, limit=None, scrape_options=None):
        self.search_calls.append((query, limit, scrape_options))
        return SimpleNamespace(data=self._results)


def make_service(results):
    client = FakeFirecrawlClient(results)
    service = FirecrawlService(api_key="test-key", client_factory=lambda api_key: client)
    return service, client


def test_document_id_for_url_is_stable_and_hex():
    assert document_id_for_url("https://example.com/a") == document_id_for_url("https://example.com/a")
    assert document_id_for_url("https://example.com/a") != document_id_for_url("https://example.com/b")
    assert all(c in "0123456789abcdef" for c in document_id_for_url("https://example.com/a"))


@pytest.mark.asyncio
async def test_search_economic_news_maps_results_to_documents():
    service, client = make_service([
        {"url": "https://example.com/a", "title": "Fed holds rates", "markdown": "content about the Fed"},
    ])

    docs = await service.search_economic_news("federal reserve", limit=3)

    assert docs == [{
        "id": document_id_for_url("https://example.com/a"),
        "title": "Fed holds rates",
        "content": "content about the Fed",
        "source": "https://example.com/a",
    }]
    assert client.search_calls[0][0] == "federal reserve"
    assert client.search_calls[0][1] == 3


@pytest.mark.asyncio
async def test_search_economic_news_skips_results_with_no_content():
    service, _client = make_service([
        {"url": "https://example.com/a", "title": "Blocked page", "markdown": ""},
        {"url": "https://example.com/b", "title": "Has content", "markdown": "real content"},
    ])

    docs = await service.search_economic_news("federal reserve")

    assert [d["source"] for d in docs] == ["https://example.com/b"]


@pytest.mark.asyncio
async def test_search_economic_news_falls_back_to_description_when_no_markdown():
    service, _client = make_service([
        {"url": "https://example.com/a", "title": "A", "description": "a short description"},
    ])

    docs = await service.search_economic_news("federal reserve")

    assert docs[0]["content"] == "a short description"


@pytest.mark.asyncio
async def test_firecrawl_client_created_lazily_once():
    calls = []

    def factory(api_key):
        calls.append(api_key)
        return FakeFirecrawlClient([])

    service = FirecrawlService(api_key="test-key", client_factory=factory)

    await service.search_economic_news("first")
    await service.search_economic_news("second")

    assert calls == ["test-key"]
