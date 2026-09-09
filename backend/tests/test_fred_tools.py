import re

import fred
import pytest


class FakeCollection:
    def __init__(self, docs):
        self._docs = docs
        self.find_calls = []

    def find(self, mongo_filter=None, projection=None):
        self.find_calls.append((mongo_filter, projection))
        if not mongo_filter:
            return list(self._docs)
        regex = mongo_filter["title"]["$regex"]
        pattern = re.compile(regex, re.IGNORECASE)
        return [doc for doc in self._docs if pattern.search(doc.get("title", ""))]


class FakeFredService:
    def __init__(self, observations):
        self._observations = observations
        self.calls = []

    async def get_series_observations(self, series_id, observation_start=None, observation_end=None):
        self.calls.append((series_id, observation_start, observation_end))
        return self._observations


def test_resolve_indicator_collection_known_category(monkeypatch):
    cpi_collection = FakeCollection([])
    monkeypatch.setattr(fred, "client", {"cpi": {"cpi": cpi_collection}})

    assert fred._resolve_indicator_collection("cpi") is cpi_collection


def test_resolve_indicator_collection_falls_back_to_interest_rate_sector(monkeypatch):
    sector_collection = FakeCollection([])
    monkeypatch.setattr(fred, "client", {"interest_rate": {"Treasury_Bills": sector_collection}})

    assert fred._resolve_indicator_collection("Treasury_Bills") is sector_collection


def test_search_fred_indicators_returns_all_docs_without_query(monkeypatch):
    docs = [{"id": "CPIAUCSL", "title": "CPI for All Urban Consumers"}]
    monkeypatch.setattr(fred, "client", {"cpi": {"cpi": FakeCollection(docs)}})

    result = fred.search_fred_indicators.invoke({"category": "cpi"})

    assert result == docs


def test_search_fred_indicators_filters_by_query(monkeypatch):
    docs = [
        {"id": "CPIAUCSL", "title": "Consumer Price Index"},
        {"id": "CPILFESL", "title": "Core CPI Less Food and Energy"},
    ]
    monkeypatch.setattr(fred, "client", {"cpi": {"cpi": FakeCollection(docs)}})

    result = fred.search_fred_indicators.invoke({"category": "cpi", "query": "core"})

    assert result == [docs[1]]


@pytest.mark.asyncio
async def test_get_fred_observations_passes_dates_to_fred_service(monkeypatch):
    fake_service = FakeFredService([{"date": "2024-01-01", "value": "1.0"}])
    monkeypatch.setattr(fred, "fred_service", fake_service)

    result = await fred.get_fred_observations.ainvoke(
        {"series_id": "CPIAUCSL", "start_date": "2020-01-01", "end_date": "2024-01-01"}
    )

    assert result == [{"date": "2024-01-01", "value": "1.0"}]
    assert fake_service.calls == [("CPIAUCSL", "2020-01-01", "2024-01-01")]


@pytest.mark.asyncio
async def test_get_fred_observations_defaults_dates_to_none(monkeypatch):
    fake_service = FakeFredService([])
    monkeypatch.setattr(fred, "fred_service", fake_service)

    await fred.get_fred_observations.ainvoke({"series_id": "CPIAUCSL"})

    assert fake_service.calls == [("CPIAUCSL", None, None)]


class FakeAzureSearchService:
    def __init__(self):
        self.fred_series_docs = []
        self.fred_observations_docs = []

    async def index_fred_series(self, documents):
        self.fred_series_docs.extend(documents)

    async def index_fred_observations(self, documents):
        self.fred_observations_docs.extend(documents)


def test_to_fred_series_document_maps_id_to_series_id_and_adds_category():
    metadata = {
        "id": "CPIAUCSL",
        "title": "Consumer Price Index",
        "frequency": "Monthly",
        "units": "Index",
        "seasonal_adjustment": "SA",
        "notes": "Some notes",
    }

    doc = fred._to_fred_series_document("cpi", metadata)

    assert doc == {
        "series_id": "CPIAUCSL",
        "title": "Consumer Price Index",
        "category": "cpi",
        "frequency": "Monthly",
        "units": "Index",
        "seasonal_adjustment": "SA",
        "notes": "Some notes",
    }


def test_to_fred_observation_documents_flattens_one_row_per_date():
    selected_docs = [
        {
            "metadata": {"id": "CPIAUCSL"},
            "data": [
                {"date": "2024-01-01", "value": "1.0"},
                {"date": "2024-02-01", "value": "1.1"},
            ],
        }
    ]

    docs = fred._to_fred_observation_documents("cpi", selected_docs)

    assert docs == [
        {"id": "CPIAUCSL_2024-01-01", "series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"},
        {"id": "CPIAUCSL_2024-02-01", "series_id": "CPIAUCSL", "date": "2024-02-01", "value": "1.1", "category": "cpi"},
    ]


@pytest.mark.asyncio
async def test_sync_indicator_to_azure_search_pushes_series_and_observations(monkeypatch):
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    metadata_collection = FakeCollection([
        {"id": "CPIAUCSL", "title": "CPI", "frequency": "Monthly", "units": "Index", "seasonal_adjustment": "SA"},
    ])
    selected_collection = FakeCollection([
        {"metadata": {"id": "CPIAUCSL"}, "data": [{"date": "2024-01-01", "value": "1.0"}]},
    ])

    await fred.sync_indicator_to_azure_search("cpi", metadata_collection, selected_collection)

    assert fake_search.fred_series_docs == [
        {
            "series_id": "CPIAUCSL",
            "title": "CPI",
            "category": "cpi",
            "frequency": "Monthly",
            "units": "Index",
            "seasonal_adjustment": "SA",
            "notes": None,
        }
    ]
    assert fake_search.fred_observations_docs == [
        {"id": "CPIAUCSL_2024-01-01", "series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"}
    ]


@pytest.mark.asyncio
async def test_sync_indicator_to_azure_search_skips_empty_collections(monkeypatch):
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred.sync_indicator_to_azure_search("cpi", FakeCollection([]), FakeCollection([]))

    assert fake_search.fred_series_docs == []
    assert fake_search.fred_observations_docs == []
