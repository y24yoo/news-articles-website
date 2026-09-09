from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

FRED_SERIES_INDEX_ENV_VAR = "AZURE_SEARCH_INDEX_FRED_SERIES"
FRED_OBSERVATIONS_INDEX_ENV_VAR = "AZURE_SEARCH_INDEX_FRED_OBSERVATIONS"
ECONOMIC_DOCUMENTS_INDEX_ENV_VAR = "AZURE_SEARCH_INDEX_DOCUMENTS"

DEFAULT_FRED_SERIES_INDEX = "fred-series"
DEFAULT_FRED_OBSERVATIONS_INDEX = "fred-observations"
DEFAULT_ECONOMIC_DOCUMENTS_INDEX = "economic-documents"

_FRED_SERIES_SEMANTIC_CONFIG = "fred-series-semantic-config"
_ECONOMIC_DOCUMENTS_SEMANTIC_CONFIG = "economic-documents-semantic-config"
_ECONOMIC_DOCUMENTS_VECTOR_ALGORITHM = "economic-documents-hnsw"
_ECONOMIC_DOCUMENTS_VECTOR_PROFILE = "economic-documents-vector-profile"
# text-embedding-3-small's output dimensionality. economic-documents isn't
# populated until Phase 6 (RAG), but the schema needs a dimension up front.
_ECONOMIC_DOCUMENTS_EMBEDDING_DIMENSIONS = 1536


def _escape_odata_literal(value: str) -> str:
    """Escape a value for safe interpolation into an OData filter string literal."""
    return value.replace("'", "''")


def build_fred_series_index(name: str = DEFAULT_FRED_SERIES_INDEX) -> Any:
    """fred-series: searchable FRED indicator metadata, semantic search for indicator selection."""
    from azure.search.documents.indexes.models import (
        SearchableField,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
    )
    from azure.search.documents.indexes.models import SearchFieldDataType as T

    fields = [
        SimpleField(name="series_id", type=T.String, key=True, filterable=True),
        SearchableField(name="title"),
        SimpleField(name="category", type=T.String, filterable=True, facetable=True),
        SimpleField(name="frequency", type=T.String, filterable=True, facetable=True),
        SimpleField(name="units", type=T.String, filterable=True),
        SimpleField(name="seasonal_adjustment", type=T.String, filterable=True, facetable=True),
        SearchableField(name="notes"),
    ]
    semantic_search = SemanticSearch(
        default_configuration_name=_FRED_SERIES_SEMANTIC_CONFIG,
        configurations=[
            SemanticConfiguration(
                name=_FRED_SERIES_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="notes")],
                ),
            )
        ],
    )
    return SearchIndex(name=name, fields=fields, semantic_search=semantic_search)


def build_fred_observations_index(name: str = DEFAULT_FRED_OBSERVATIONS_INDEX) -> Any:
    """fred-observations: numeric observations, filter-based retrieval only (no vector search)."""
    from azure.search.documents.indexes.models import SearchFieldDataType as T
    from azure.search.documents.indexes.models import SearchIndex, SimpleField

    fields = [
        SimpleField(name="id", type=T.String, key=True),
        SimpleField(name="series_id", type=T.String, filterable=True, facetable=True),
        SimpleField(name="date", type=T.String, filterable=True, sortable=True),
        SimpleField(name="value", type=T.String),
        SimpleField(name="category", type=T.String, filterable=True, facetable=True),
    ]
    return SearchIndex(name=name, fields=fields)


def build_economic_documents_index(name: str = DEFAULT_ECONOMIC_DOCUMENTS_INDEX) -> Any:
    """economic-documents: keyword + vector + hybrid retrieval for RAG (populated in Phase 6)."""
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )
    from azure.search.documents.indexes.models import SearchFieldDataType as T

    fields = [
        SimpleField(name="id", type=T.String, key=True),
        SearchableField(name="title"),
        SearchableField(name="content"),
        SimpleField(name="source", type=T.String, filterable=True, facetable=True),
        SimpleField(name="published_at", type=T.String, filterable=True, sortable=True),
        SearchField(
            name="content_vector",
            type=T.Collection(T.Single),
            searchable=True,
            vector_search_dimensions=_ECONOMIC_DOCUMENTS_EMBEDDING_DIMENSIONS,
            vector_search_profile_name=_ECONOMIC_DOCUMENTS_VECTOR_PROFILE,
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=_ECONOMIC_DOCUMENTS_VECTOR_ALGORITHM)],
        profiles=[
            VectorSearchProfile(
                name=_ECONOMIC_DOCUMENTS_VECTOR_PROFILE,
                algorithm_configuration_name=_ECONOMIC_DOCUMENTS_VECTOR_ALGORITHM,
            )
        ],
    )
    semantic_search = SemanticSearch(
        default_configuration_name=_ECONOMIC_DOCUMENTS_SEMANTIC_CONFIG,
        configurations=[
            SemanticConfiguration(
                name=_ECONOMIC_DOCUMENTS_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ],
    )
    return SearchIndex(name=name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)


def _default_credential_factory() -> Any:
    from azure.identity.aio import DefaultAzureCredential

    return DefaultAzureCredential()


def _default_embedding_client_factory() -> Any:
    from agent_framework_foundry import FoundryEmbeddingClient
    from azure.identity.aio import DefaultAzureCredential

    # Reads FOUNDRY_MODELS_ENDPOINT / FOUNDRY_EMBEDDING_MODEL from the
    # environment itself (see backend/.env.example).
    return FoundryEmbeddingClient(credential=DefaultAzureCredential())


def _default_index_client_factory(endpoint: str, credential: Any) -> Any:
    from azure.search.documents.indexes.aio import SearchIndexClient

    return SearchIndexClient(endpoint, credential)


def _default_search_client_factory(endpoint: str, index_name: str, credential: Any) -> Any:
    from azure.search.documents.aio import SearchClient

    return SearchClient(endpoint, index_name, credential)


class AzureSearchService:
    """Manages the fred-series/fred-observations/economic-documents indexes and
    the query patterns needed so far. As of Phase 5, this is the sole backing
    store for FRED indicator/observation data -- MongoDB has been removed.

    All Azure SDK clients are lazily created via injectable factories so this
    module stays importable and testable without the azure-search-documents /
    azure-identity SDKs installed, and without live Azure credentials.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        fred_series_index: str | None = None,
        fred_observations_index: str | None = None,
        economic_documents_index: str | None = None,
        credential_factory: Callable[[], Any] = _default_credential_factory,
        index_client_factory: Callable[[str, Any], Any] = _default_index_client_factory,
        search_client_factory: Callable[[str, str, Any], Any] = _default_search_client_factory,
        embedding_client_factory: Callable[[], Any] = _default_embedding_client_factory,
    ):
        self._endpoint = endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT")
        self.fred_series_index = fred_series_index or os.environ.get(
            FRED_SERIES_INDEX_ENV_VAR, DEFAULT_FRED_SERIES_INDEX
        )
        self.fred_observations_index = fred_observations_index or os.environ.get(
            FRED_OBSERVATIONS_INDEX_ENV_VAR, DEFAULT_FRED_OBSERVATIONS_INDEX
        )
        self.economic_documents_index = economic_documents_index or os.environ.get(
            ECONOMIC_DOCUMENTS_INDEX_ENV_VAR, DEFAULT_ECONOMIC_DOCUMENTS_INDEX
        )
        self._credential_factory = credential_factory
        self._index_client_factory = index_client_factory
        self._search_client_factory = search_client_factory
        self._embedding_client_factory = embedding_client_factory
        self._search_clients: dict[str, Any] = {}
        self._embedding_client: Any = None

    def _get_search_client(self, index_name: str) -> Any:
        if index_name not in self._search_clients:
            self._search_clients[index_name] = self._search_client_factory(
                self._endpoint, index_name, self._credential_factory()
            )
        return self._search_clients[index_name]

    async def _embed_text(self, text: str) -> list[float]:
        if self._embedding_client is None:
            self._embedding_client = self._embedding_client_factory()
        embeddings = await self._embedding_client.get_embeddings([text])
        return embeddings[0].vector

    async def ensure_indexes(self) -> None:
        """Create (or update) all three indexes."""
        index_client = self._index_client_factory(self._endpoint, self._credential_factory())
        indexes = (
            build_fred_series_index(self.fred_series_index),
            build_fred_observations_index(self.fred_observations_index),
            build_economic_documents_index(self.economic_documents_index),
        )
        for index in indexes:
            await index_client.create_or_update_index(index)

    async def index_fred_series(self, documents: Sequence[dict]) -> Any:
        """Upload/merge FRED indicator metadata documents into fred-series."""
        client = self._get_search_client(self.fred_series_index)
        return await client.merge_or_upload_documents(list(documents))

    async def index_fred_observations(self, documents: Sequence[dict]) -> Any:
        """Upload/merge FRED observation documents into fred-observations."""
        client = self._get_search_client(self.fred_observations_index)
        return await client.merge_or_upload_documents(list(documents))

    async def search_fred_series(self, query: str, category: str | None = None, top: int = 5) -> list[dict]:
        """Semantic search over fred-series (e.g. "important inflation indicators")."""
        client = self._get_search_client(self.fred_series_index)
        results = await client.search(
            search_text=query,
            query_type="semantic",
            semantic_configuration_name=_FRED_SERIES_SEMANTIC_CONFIG,
            filter=f"category eq '{_escape_odata_literal(category)}'" if category else None,
            top=top,
        )
        return [doc async for doc in results]

    async def get_fred_series(self, category: str) -> list[dict]:
        """Filter-based retrieval of every indexed indicator's metadata for a
        category (no ranking/relevance -- used to enumerate what's indexed,
        not to select among it)."""
        client = self._get_search_client(self.fred_series_index)
        results = await client.search(
            search_text="*",
            filter=f"category eq '{_escape_odata_literal(category)}'",
        )
        return [doc async for doc in results]

    async def get_fred_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Filter-based retrieval over fred-observations. No vector search."""
        filters = [f"series_id eq '{_escape_odata_literal(series_id)}'"]
        if start_date:
            filters.append(f"date ge '{_escape_odata_literal(start_date)}'")
        if end_date:
            filters.append(f"date le '{_escape_odata_literal(end_date)}'")
        client = self._get_search_client(self.fred_observations_index)
        results = await client.search(search_text="*", filter=" and ".join(filters), order_by=["date asc"])
        return [doc async for doc in results]

    async def get_observations_by_category(self, category: str) -> list[dict]:
        """Filter-based retrieval of every indexed observation for a category,
        across all its series. No vector search."""
        client = self._get_search_client(self.fred_observations_index)
        results = await client.search(
            search_text="*",
            filter=f"category eq '{_escape_odata_literal(category)}'",
            order_by=["series_id asc", "date asc"],
        )
        return [doc async for doc in results]

    async def index_economic_documents(self, documents: Sequence[dict]) -> Any:
        """Embed each document's `content` and upload/merge into economic-documents.

        Each document needs at least `id` and `content`; `title`, `source`,
        and `published_at` are optional per the index schema.
        """
        enriched = []
        for doc in documents:
            vector = await self._embed_text(doc["content"])
            enriched.append({**doc, "content_vector": vector})
        client = self._get_search_client(self.economic_documents_index)
        return await client.merge_or_upload_documents(enriched)

    async def search_economic_documents(self, query: str, top: int = 5) -> list[dict]:
        """Hybrid retrieval over economic-documents: keyword + vector + semantic reranking."""
        from azure.search.documents.models import VectorizedQuery

        vector = await self._embed_text(query)
        client = self._get_search_client(self.economic_documents_index)
        results = await client.search(
            search_text=query,
            vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=top, fields="content_vector")],
            query_type="semantic",
            semantic_configuration_name=_ECONOMIC_DOCUMENTS_SEMANTIC_CONFIG,
            top=top,
        )
        return [doc async for doc in results]
