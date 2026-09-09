import pytest

from services.azure_search_service import (
    AzureSearchService,
    build_economic_documents_index,
    build_fred_observations_index,
    build_fred_series_index,
)


class FakeAsyncResults:
    def __init__(self, docs):
        self._docs = docs

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for doc in self._docs:
            yield doc


class FakeSearchClient:
    def __init__(self, endpoint, index_name, credential, docs=None):
        self.endpoint = endpoint
        self.index_name = index_name
        self.credential = credential
        self.uploaded = []
        self.search_calls = []
        self._docs = docs or []

    async def merge_or_upload_documents(self, documents):
        self.uploaded.extend(documents)
        return documents

    async def search(self, search_text=None, **kwargs):
        self.search_calls.append({"search_text": search_text, **kwargs})
        return FakeAsyncResults(self._docs)


class FakeIndexClient:
    def __init__(self, endpoint, credential):
        self.endpoint = endpoint
        self.credential = credential
        self.created_indexes = []

    async def create_or_update_index(self, index):
        self.created_indexes.append(index)
        return index


class FakeEmbedding:
    def __init__(self, vector):
        self.vector = vector


class FakeEmbeddingClient:
    def __init__(self):
        self.embed_calls = []

    async def get_embeddings(self, values):
        self.embed_calls.append(list(values))
        return [FakeEmbedding([0.1, 0.2, 0.3]) for _ in values]


def make_service(**kwargs):
    search_clients = {}

    def search_client_factory(endpoint, index_name, credential):
        client = FakeSearchClient(endpoint, index_name, credential, docs=kwargs.get("docs"))
        search_clients[index_name] = client
        return client

    index_clients = []

    def index_client_factory(endpoint, credential):
        client = FakeIndexClient(endpoint, credential)
        index_clients.append(client)
        return client

    embedding_client = kwargs.get("embedding_client") or FakeEmbeddingClient()

    service = AzureSearchService(
        endpoint="https://example.search.windows.net",
        credential_factory=lambda: "fake-credential",
        index_client_factory=index_client_factory,
        search_client_factory=search_client_factory,
        embedding_client_factory=lambda: embedding_client,
    )
    return service, search_clients, index_clients, embedding_client


def test_index_builders_use_configured_names():
    assert build_fred_series_index("custom-series").name == "custom-series"
    assert build_fred_observations_index("custom-observations").name == "custom-observations"
    assert build_economic_documents_index("custom-documents").name == "custom-documents"


def test_service_reads_index_names_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_INDEX_FRED_SERIES", "env-fred-series")
    service = AzureSearchService(endpoint="https://example.search.windows.net")
    assert service.fred_series_index == "env-fred-series"


@pytest.mark.asyncio
async def test_ensure_indexes_creates_all_three():
    service, _search_clients, index_clients, _embedding_client = make_service()

    await service.ensure_indexes()

    assert len(index_clients) == 1
    created_names = [index.name for index in index_clients[0].created_indexes]
    assert created_names == ["fred-series", "fred-observations", "economic-documents"]


@pytest.mark.asyncio
async def test_index_fred_series_uploads_to_the_right_client():
    service, search_clients, _index_clients, _embedding_client = make_service()
    docs = [{"series_id": "CPIAUCSL", "title": "CPI"}]

    await service.index_fred_series(docs)

    assert search_clients["fred-series"].uploaded == docs


@pytest.mark.asyncio
async def test_index_fred_observations_uploads_to_the_right_client():
    service, search_clients, _index_clients, _embedding_client = make_service()
    docs = [{"id": "CPIAUCSL_2024-01-01", "series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0"}]

    await service.index_fred_observations(docs)

    assert search_clients["fred-observations"].uploaded == docs


@pytest.mark.asyncio
async def test_search_fred_series_uses_semantic_query_type():
    service, search_clients, _index_clients, _embedding_client = make_service(docs=[{"series_id": "CPIAUCSL"}])

    result = await service.search_fred_series("important inflation indicators", category="cpi")

    call = search_clients["fred-series"].search_calls[0]
    assert call["query_type"] == "semantic"
    assert call["filter"] == "category eq 'cpi'"
    assert result == [{"series_id": "CPIAUCSL"}]


@pytest.mark.asyncio
async def test_search_fred_series_without_category_omits_filter():
    service, search_clients, _index_clients, _embedding_client = make_service()

    await service.search_fred_series("inflation")

    assert search_clients["fred-series"].search_calls[0]["filter"] is None


@pytest.mark.asyncio
async def test_get_fred_observations_builds_filter_from_dates():
    service, search_clients, _index_clients, _embedding_client = make_service()

    await service.get_fred_observations("CPIAUCSL", start_date="2020-01-01", end_date="2024-01-01")

    call = search_clients["fred-observations"].search_calls[0]
    assert call["filter"] == "series_id eq 'CPIAUCSL' and date ge '2020-01-01' and date le '2024-01-01'"


@pytest.mark.asyncio
async def test_get_fred_observations_escapes_quotes_in_series_id():
    service, search_clients, _index_clients, _embedding_client = make_service()

    await service.get_fred_observations("O'BRIEN")

    call = search_clients["fred-observations"].search_calls[0]
    assert call["filter"] == "series_id eq 'O''BRIEN'"


@pytest.mark.asyncio
async def test_get_fred_series_filters_by_category_only():
    service, search_clients, _index_clients, _embedding_client = make_service(docs=[{"series_id": "CPIAUCSL"}])

    result = await service.get_fred_series("cpi")

    call = search_clients["fred-series"].search_calls[0]
    assert call["filter"] == "category eq 'cpi'"
    assert "query_type" not in call
    assert result == [{"series_id": "CPIAUCSL"}]


@pytest.mark.asyncio
async def test_get_observations_by_category_filters_and_orders():
    service, search_clients, _index_clients, _embedding_client = make_service(docs=[{"series_id": "CPIAUCSL", "date": "2024-01-01"}])

    result = await service.get_observations_by_category("cpi")

    call = search_clients["fred-observations"].search_calls[0]
    assert call["filter"] == "category eq 'cpi'"
    assert call["order_by"] == ["series_id asc", "date asc"]
    assert result == [{"series_id": "CPIAUCSL", "date": "2024-01-01"}]


@pytest.mark.asyncio
async def test_index_economic_documents_embeds_content_and_uploads():
    service, search_clients, _index_clients, embedding_client = make_service()
    docs = [{"id": "abc", "title": "Fed holds rates steady", "content": "The Federal Reserve...", "source": "https://example.com"}]

    await service.index_economic_documents(docs)

    assert embedding_client.embed_calls == [["The Federal Reserve..."]]
    uploaded = search_clients["economic-documents"].uploaded
    assert uploaded[0]["id"] == "abc"
    assert uploaded[0]["content_vector"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_search_economic_documents_uses_hybrid_vector_and_semantic_query():
    service, search_clients, _index_clients, embedding_client = make_service(
        docs=[{"id": "abc", "title": "Fed holds rates steady"}]
    )

    result = await service.search_economic_documents("federal reserve interest rate decision", top=3)

    assert embedding_client.embed_calls == [["federal reserve interest rate decision"]]
    call = search_clients["economic-documents"].search_calls[0]
    assert call["search_text"] == "federal reserve interest rate decision"
    assert call["query_type"] == "semantic"
    assert len(call["vector_queries"]) == 1
    vector_query = call["vector_queries"][0]
    assert vector_query.vector == [0.1, 0.2, 0.3]
    assert vector_query.fields == "content_vector"
    assert vector_query.k_nearest_neighbors == 3
    assert result == [{"id": "abc", "title": "Fed holds rates steady"}]


@pytest.mark.asyncio
async def test_embedding_client_is_created_lazily_once():
    calls = []

    def embedding_factory():
        calls.append(1)
        return FakeEmbeddingClient()

    service = AzureSearchService(
        endpoint="https://example.search.windows.net",
        credential_factory=lambda: "fake-credential",
        search_client_factory=lambda endpoint, index_name, credential: FakeSearchClient(endpoint, index_name, credential),
        embedding_client_factory=embedding_factory,
    )

    await service.search_economic_documents("first")
    await service.search_economic_documents("second")

    assert len(calls) == 1
