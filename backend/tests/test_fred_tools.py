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


class FakeDatabase(dict):
    def list_collection_names(self):
        return list(self.keys())


class FakeFredService:
    def __init__(self, observations):
        self._observations = observations
        self.calls = []

    async def get_series_observations(self, series_id, observation_start=None, observation_end=None):
        self.calls.append((series_id, observation_start, observation_end))
        return self._observations


class FakeAzureSearchService:
    def __init__(self, search_results=None):
        self.fred_series_docs = []
        self.fred_observations_docs = []
        self.search_calls = []
        self._search_results = search_results or []

    async def index_fred_series(self, documents):
        self.fred_series_docs.extend(documents)

    async def index_fred_observations(self, documents):
        self.fred_observations_docs.extend(documents)

    async def search_fred_series(self, query, category=None, top=5):
        self.search_calls.append((query, category, top))
        return self._search_results


@pytest.mark.asyncio
async def test_search_fred_indicators_uses_wildcard_query_when_empty(monkeypatch):
    fake_search = FakeAzureSearchService(search_results=[{"series_id": "CPIAUCSL"}])
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    result = await fred.search_fred_indicators.ainvoke({"category": "cpi"})

    assert result == [{"series_id": "CPIAUCSL"}]
    assert fake_search.search_calls == [("*", "cpi", 5)]


@pytest.mark.asyncio
async def test_search_fred_indicators_passes_query_and_category_through(monkeypatch):
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred.search_fred_indicators.ainvoke({"category": "cpi", "query": "important inflation indicators"})

    assert fake_search.search_calls == [("important inflation indicators", "cpi", 5)]


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


@pytest.mark.asyncio
async def test_sync_gdp_to_azure_search_pushes_both_gdp_and_shared_gdp(monkeypatch):
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    gdp_metadata = FakeCollection([
        {"id": "GDP", "title": "Gross Domestic Product", "frequency": "Quarterly", "units": "Bil. $", "seasonal_adjustment": "SA"},
    ])
    gdp_selected = FakeCollection([{"metadata": {"id": "GDP"}, "data": [{"date": "2024-01-01", "value": "100"}]}])
    shared_gdp_metadata = FakeCollection([
        {"id": "A191RL1Q225SBEA", "title": "Real GDP Growth", "frequency": "Quarterly", "units": "Percent", "seasonal_adjustment": "SA"},
    ])
    shared_gdp_selected = FakeCollection(
        [{"metadata": {"id": "A191RL1Q225SBEA"}, "data": [{"date": "2024-01-01", "value": "2.5"}]}]
    )

    monkeypatch.setattr(fred, "client", {
        "gdp": {
            "gdp": gdp_metadata,
            "selected_gdp": gdp_selected,
            "shared_gdp": shared_gdp_metadata,
            "selected_shared_gdp": shared_gdp_selected,
        },
    })

    await fred.sync_gdp_to_azure_search()

    series_categories = {doc["category"] for doc in fake_search.fred_series_docs}
    assert series_categories == {"gdp", "shared_gdp"}
    assert len(fake_search.fred_observations_docs) == 2


@pytest.mark.asyncio
async def test_sync_interest_rate_to_azure_search_pushes_each_sector_with_selected_data(monkeypatch):
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    source_db = FakeDatabase({
        "Treasury_Bills": FakeCollection([
            {"id": "TB3MS", "title": "3-Month Treasury Bill", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "NSA"},
        ]),
        "Monetary_Policy": FakeCollection([
            {"id": "FEDFUNDS", "title": "Federal Funds Rate", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "NSA"},
        ]),
    })
    target_db = FakeDatabase({
        "Treasury_Bills": FakeCollection([{"metadata": {"id": "TB3MS"}, "data": [{"date": "2024-01-01", "value": "5.0"}]}]),
        # Monetary_Policy has no selected-indicator collection yet -- should be skipped, not error.
    })
    monkeypatch.setattr(fred, "client", {"interest_rate": source_db, "selected_interest_rate": target_db})

    await fred.sync_interest_rate_to_azure_search()

    assert [doc["series_id"] for doc in fake_search.fred_series_docs] == ["TB3MS"]
    assert [doc["series_id"] for doc in fake_search.fred_observations_docs] == ["TB3MS"]


@pytest.mark.asyncio
async def test_sync_all_indicators_to_azure_search_calls_every_category(monkeypatch):
    calls = []

    async def record(name):
        calls.append(name)

    monkeypatch.setattr(fred, "sync_cpi_to_azure_search", lambda: record("cpi"))
    monkeypatch.setattr(fred, "sync_ppi_to_azure_search", lambda: record("ppi"))
    monkeypatch.setattr(fred, "sync_gdp_to_azure_search", lambda: record("gdp"))
    monkeypatch.setattr(fred, "sync_interest_rate_to_azure_search", lambda: record("interest_rate"))
    monkeypatch.setattr(fred, "sync_unemployment_rate_to_azure_search", lambda: record("unemployment_rate"))

    await fred.sync_all_indicators_to_azure_search()

    assert calls == ["cpi", "ppi", "gdp", "interest_rate", "unemployment_rate"]
