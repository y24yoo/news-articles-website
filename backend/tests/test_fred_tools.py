import fred
import pytest
from services.evaluation_service import EvaluationService
from services.fred_service import FredApiError


class FakeFredService:
    def __init__(
        self,
        category_series=None,
        category_children=None,
        observations_by_id=None,
        bad_series_ids=None,
    ):
        self._category_series = category_series or {}
        self._category_children = category_children or {}
        self._observations_by_id = observations_by_id or {}
        self._bad_series_ids = bad_series_ids or set()
        self.observation_calls = []

    async def get_category_series(self, category_id):
        return self._category_series.get(category_id, [])

    async def get_category_children(self, category_id):
        return self._category_children.get(category_id, [])

    async def get_series_observations(self, series_id, observation_start=None, observation_end=None):
        self.observation_calls.append((series_id, observation_start, observation_end))
        if series_id in self._bad_series_ids:
            raise FredApiError(500, {"error": "boom"})
        return self._observations_by_id.get(series_id, [])


class FakeAzureSearchService:
    def __init__(
        self,
        fred_series_by_category=None,
        observations_by_category=None,
        search_results=None,
        economic_document_results=None,
    ):
        self.fred_series_docs = []
        self.fred_observations_docs = []
        self.economic_documents = []
        self.search_calls = []
        self.economic_document_search_calls = []
        self._search_results = search_results or []
        self._fred_series_by_category = fred_series_by_category or {}
        self._observations_by_category = observations_by_category or {}
        self._economic_document_results = economic_document_results or []

    async def index_fred_series(self, documents):
        self.fred_series_docs.extend(documents)

    async def index_fred_observations(self, documents):
        self.fred_observations_docs.extend(documents)

    async def search_fred_series(self, query, category=None, top=5):
        self.search_calls.append((query, category, top))
        return self._search_results

    async def get_fred_series(self, category):
        return self._fred_series_by_category.get(category, [])

    async def get_observations_by_category(self, category):
        return self._observations_by_category.get(category, [])

    async def index_economic_documents(self, documents):
        self.economic_documents.extend(documents)

    async def search_economic_documents(self, query, top=5):
        self.economic_document_search_calls.append((query, top))
        return self._economic_document_results


class FakeDaytonaService:
    def __init__(self):
        self.calls = []
        self.report_path = "indicators/cpi/report.html"

    async def generate_economic_report(self, indicator_type, datasets):
        self.calls.append((indicator_type, datasets))
        return self.report_path


class FakeAgent:
    def __init__(self, structured_response):
        self._structured_response = structured_response
        self.prompts = []

    async def ainvoke(self, payload):
        self.prompts.append(payload["messages"][0]["content"])
        return {"structured_response": self._structured_response}


def cpi_metadata(series_id="CPIAUCSL", title="CPI"):
    return fred.FredSeriesMetadata(
        id=series_id, title=title, frequency="Monthly", units="Index", seasonal_adjustment="SA", notes=None
    )


# --- search_fred_indicators / get_fred_observations tools --------------------


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
async def test_search_fred_indicators_records_tool_call_and_relevance(monkeypatch):
    fake_search = FakeAzureSearchService(
        search_results=[{"series_id": "CPIAUCSL", "@search.rerankerScore": 2.5}]
    )
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    await fred.search_fred_indicators.ainvoke({"category": "cpi"})

    assert fake_eval.tool_call_success_rate("search_fred_indicators") == 1.0
    assert fake_eval.average_retrieval_relevance("search_fred_indicators") == 2.5


@pytest.mark.asyncio
async def test_get_fred_observations_passes_dates_to_fred_service(monkeypatch):
    fake_service = FakeFredService(observations_by_id={"CPIAUCSL": [{"date": "2024-01-01", "value": "1.0"}]})
    monkeypatch.setattr(fred, "fred_service", fake_service)

    result = await fred.get_fred_observations.ainvoke(
        {"series_id": "CPIAUCSL", "start_date": "2020-01-01", "end_date": "2024-01-01"}
    )

    assert result == [{"date": "2024-01-01", "value": "1.0"}]
    assert fake_service.observation_calls == [("CPIAUCSL", "2020-01-01", "2024-01-01")]


@pytest.mark.asyncio
async def test_get_fred_observations_defaults_dates_to_none(monkeypatch):
    fake_service = FakeFredService()
    monkeypatch.setattr(fred, "fred_service", fake_service)

    await fred.get_fred_observations.ainvoke({"series_id": "CPIAUCSL"})

    assert fake_service.observation_calls == [("CPIAUCSL", None, None)]


@pytest.mark.asyncio
async def test_get_fred_observations_records_tool_call_outcome(monkeypatch):
    fake_service = FakeFredService(observations_by_id={"CPIAUCSL": []})
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    await fred.get_fred_observations.ainvoke({"series_id": "CPIAUCSL"})

    assert fake_eval.tool_call_success_rate("get_fred_observations") == 1.0


@pytest.mark.asyncio
async def test_get_fred_observations_records_failure_on_error(monkeypatch):
    fake_service = FakeFredService(bad_series_ids={"BADSERIES"})
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    with pytest.raises(FredApiError):
        await fred.get_fred_observations.ainvoke({"series_id": "BADSERIES"})

    assert fake_eval.tool_call_success_rate("get_fred_observations") == 0.0


# --- pure mapping helpers -----------------------------------------------------


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
async def test_build_indicator_dataset_merges_metadata_with_observations(monkeypatch):
    fake_search = FakeAzureSearchService(
        fred_series_by_category={
            "cpi": [
                {"series_id": "CPIAUCSL", "title": "CPI", "category": "cpi"},
                {"series_id": "CPILFESL", "title": "Core CPI", "category": "cpi"},
            ]
        },
        observations_by_category={
            "cpi": [
                {"series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"},
                {"series_id": "CPIAUCSL", "date": "2024-02-01", "value": "1.1", "category": "cpi"},
            ]
        },
    )
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    dataset = await fred._build_indicator_dataset("cpi")

    assert dataset == [
        {
            "series_id": "CPIAUCSL",
            "title": "CPI",
            "category": "cpi",
            "data": [{"date": "2024-01-01", "value": "1.0"}, {"date": "2024-02-01", "value": "1.1"}],
        },
        {"series_id": "CPILFESL", "title": "Core CPI", "category": "cpi", "data": []},
    ]


@pytest.mark.asyncio
async def test_run_economic_analysis_builds_dataset_and_calls_daytona(monkeypatch):
    fake_search = FakeAzureSearchService(
        fred_series_by_category={"cpi": [{"series_id": "CPIAUCSL", "title": "CPI", "category": "cpi"}]},
        observations_by_category={"cpi": [{"series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"}]},
    )
    fake_daytona = FakeDaytonaService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "daytona_service", fake_daytona)

    result = await fred.run_economic_analysis("cpi")

    indicator_type, datasets = fake_daytona.calls[0]
    assert indicator_type == "cpi"
    assert [d.filename for d in datasets] == ["data.json"]
    assert result == fake_daytona.report_path


@pytest.mark.asyncio
async def test_run_economic_analysis_records_completion_and_validation_when_report_exists(monkeypatch, tmp_path):
    report_file = tmp_path / "report.html"
    report_file.write_text('<div class="prose">A real report</div>', encoding="utf-8")
    fake_search = FakeAzureSearchService(fred_series_by_category={"cpi": []}, observations_by_category={"cpi": []})
    fake_daytona = FakeDaytonaService()
    fake_daytona.report_path = str(report_file)
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "daytona_service", fake_daytona)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    await fred.run_economic_analysis("cpi")

    assert fake_eval.report_completion_rate() == 1.0
    assert fake_eval.html_validation_success_rate() == 1.0
    assert fake_eval.tool_call_success_rate("run_economic_analysis") == 1.0


@pytest.mark.asyncio
async def test_run_economic_analysis_records_incomplete_when_report_missing(monkeypatch):
    fake_search = FakeAzureSearchService(fred_series_by_category={"cpi": []}, observations_by_category={"cpi": []})
    fake_daytona = FakeDaytonaService()
    fake_daytona.report_path = "does/not/exist/report.html"
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "daytona_service", fake_daytona)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    await fred.run_economic_analysis("cpi")

    assert fake_eval.report_completion_rate() == 0.0
    assert fake_eval.html_validation_success_rate() is None


@pytest.mark.asyncio
async def test_index_observations_for_selection_fetches_and_flattens(monkeypatch):
    fake_service = FakeFredService(
        observations_by_id={
            "CPIAUCSL": [{"date": "2024-01-01", "value": "1.0"}],
            "CPILFESL": [{"date": "2024-01-01", "value": "2.0"}],
        }
    )
    fake_search = FakeAzureSearchService(
        fred_series_by_category={"cpi": [
            {"series_id": "CPIAUCSL"},
            {"series_id": "CPILFESL"},
        ]}
    )
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    selected = [
        fred.Reason(metadata=cpi_metadata("CPIAUCSL", "CPI"), reason="headline"),
        fred.Reason(metadata=cpi_metadata("CPILFESL", "Core CPI"), reason="core"),
    ]

    await fred._index_observations_for_selection("cpi", selected)

    assert fake_search.fred_observations_docs == [
        {"id": "CPIAUCSL_2024-01-01", "series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"},
        {"id": "CPILFESL_2024-01-01", "series_id": "CPILFESL", "date": "2024-01-01", "value": "2.0", "category": "cpi"},
    ]


@pytest.mark.asyncio
async def test_index_observations_for_selection_records_selection_accuracy(monkeypatch):
    fake_service = FakeFredService(observations_by_id={"CPIAUCSL": [], "MADEUP": []})
    fake_search = FakeAzureSearchService(fred_series_by_category={"cpi": [{"series_id": "CPIAUCSL"}]})
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    selected = [
        fred.Reason(metadata=cpi_metadata("CPIAUCSL", "CPI"), reason="headline"),
        fred.Reason(metadata=cpi_metadata("MADEUP", "Hallucinated"), reason="not real"),
    ]

    await fred._index_observations_for_selection("cpi", selected)

    assert fake_eval.indicator_selection_accuracy() == 0.5


# --- ingestion -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_category_series_indexes_all_by_default(monkeypatch):
    fake_service = FakeFredService(category_series={9: [
        {"id": "CPIAUCSL", "title": "CPI", "frequency": "Monthly", "units": "Index", "seasonal_adjustment": "SA"},
        {"id": "CPILFESL", "title": "Core CPI", "frequency": "Monthly", "units": "Index", "seasonal_adjustment": "SA"},
    ]})
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred._ingest_category_series("cpi", 9)

    assert [doc["series_id"] for doc in fake_search.fred_series_docs] == ["CPIAUCSL", "CPILFESL"]
    assert all(doc["category"] == "cpi" for doc in fake_search.fred_series_docs)


@pytest.mark.asyncio
async def test_ingest_category_series_applies_keep_filter(monkeypatch):
    fake_service = FakeFredService(category_series={1: [
        {"id": "UNRATE", "title": "Unemployment Rate", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "SA"},
        {"id": "OTHER", "title": "Something Else", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "SA"},
    ]})
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred._ingest_category_series("unemployment_rate", 1, keep=lambda raw: raw["id"].startswith("UNRATE"))

    assert [doc["series_id"] for doc in fake_search.fred_series_docs] == ["UNRATE"]


@pytest.mark.asyncio
async def test_ingest_category_series_skips_indexing_when_nothing_kept(monkeypatch):
    fake_service = FakeFredService(category_series={1: [
        {"id": "OTHER", "title": "x", "frequency": "Monthly", "units": "u", "seasonal_adjustment": "s"},
    ]})
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred._ingest_category_series("x", 1, keep=lambda raw: False)

    assert fake_search.fred_series_docs == []


@pytest.mark.asyncio
async def test_cpi_ingests_with_cpi_category(monkeypatch):
    fake_service = FakeFredService(category_series={fred.cpi_id: [
        {"id": "CPIAUCSL", "title": "CPI", "frequency": "Monthly", "units": "Index", "seasonal_adjustment": "SA"},
    ]})
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred.cpi()

    assert fake_search.fred_series_docs[0]["category"] == "cpi"


@pytest.mark.asyncio
async def test_unemployment_rate_filters_to_unrate_prefix(monkeypatch):
    fake_service = FakeFredService(category_series={fred.unemployment_rate_id: [
        {"id": "UNRATE", "title": "Unemployment Rate", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "SA"},
        {"id": "SOMETHINGELSE", "title": "Other", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "SA"},
    ]})
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred.unemployment_rate()

    assert [doc["series_id"] for doc in fake_search.fred_series_docs] == ["UNRATE"]


@pytest.mark.asyncio
async def test_interest_rate_ingests_each_sector_and_returns_sector_names(monkeypatch):
    fake_service = FakeFredService(
        category_children={fred.interest_rate_id: [
            {"id": 100, "name": "Treasury Bills"},
            {"id": 101, "name": "Monetary Policy"},
        ]},
        category_series={
            100: [{"id": "TB3MS", "title": "3-Month T-Bill", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "NSA"}],
            101: [{"id": "FEDFUNDS", "title": "Fed Funds Rate", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "NSA"}],
        },
    )
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    sector_names = await fred.interest_rate()

    assert sector_names == ["Treasury_Bills", "Monetary_Policy"]
    assert {doc["category"] for doc in fake_search.fred_series_docs} == {"Treasury_Bills", "Monetary_Policy"}


# --- observation refresh (no LLM selection) -------------------------------------


@pytest.mark.asyncio
async def test_observation_ppi_indexes_observations_for_every_indicator(monkeypatch):
    fake_search = FakeAzureSearchService(fred_series_by_category={
        "ppi": [{"series_id": "PPIACO", "title": "PPI", "category": "ppi"}],
    })
    fake_service = FakeFredService(observations_by_id={"PPIACO": [{"date": "2024-01-01", "value": "1.0"}]})
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "fred_service", fake_service)

    await fred.observation_ppi()

    assert fake_search.fred_observations_docs == [
        {"id": "PPIACO_2024-01-01", "series_id": "PPIACO", "date": "2024-01-01", "value": "1.0", "category": "ppi"}
    ]


@pytest.mark.asyncio
async def test_refresh_observations_skips_indicators_with_bad_response(monkeypatch):
    fake_search = FakeAzureSearchService(fred_series_by_category={
        "gdp": [
            {"series_id": "GDP", "title": "GDP", "category": "gdp"},
            {"series_id": "BADSERIES", "title": "Bad", "category": "gdp"},
        ],
    })
    fake_service = FakeFredService(
        observations_by_id={"GDP": [{"date": "2024-01-01", "value": "100"}]},
        bad_series_ids={"BADSERIES"},
    )
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "fred_service", fake_service)

    await fred._refresh_observations("gdp")

    assert fake_search.fred_observations_docs == [
        {"id": "GDP_2024-01-01", "series_id": "GDP", "date": "2024-01-01", "value": "100", "category": "gdp"}
    ]


# --- sandbox report generation --------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_cpi_builds_dataset_and_calls_daytona(monkeypatch):
    fake_search = FakeAzureSearchService(
        fred_series_by_category={"cpi": [{"series_id": "CPIAUCSL", "title": "CPI", "category": "cpi"}]},
        observations_by_category={"cpi": [{"series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"}]},
    )
    fake_daytona = FakeDaytonaService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "daytona_service", fake_daytona)

    await fred.sandbox_cpi()

    assert len(fake_daytona.calls) == 1
    indicator_type, datasets = fake_daytona.calls[0]
    assert indicator_type == "cpi"
    assert [d.filename for d in datasets] == ["data.json", "research.json"]
    assert datasets[0].data[0]["series_id"] == "CPIAUCSL"
    assert len(fake_search.economic_document_search_calls) == 1


@pytest.mark.asyncio
async def test_sandbox_gdp_builds_two_datasets(monkeypatch):
    fake_search = FakeAzureSearchService(
        fred_series_by_category={
            "gdp": [{"series_id": "GDP", "title": "GDP", "category": "gdp"}],
            "shared_gdp": [{"series_id": "A191RL1Q225SBEA", "title": "Real GDP Growth", "category": "shared_gdp"}],
        },
        observations_by_category={},
    )
    fake_daytona = FakeDaytonaService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "daytona_service", fake_daytona)

    await fred.sandbox_gdp()

    indicator_type, datasets = fake_daytona.calls[0]
    assert indicator_type == "gdp"
    assert [d.filename for d in datasets] == ["data.json1", "data.json2", "research.json"]


@pytest.mark.asyncio
async def test_sandbox_interest_rate_builds_one_dataset_per_discovered_sector(monkeypatch):
    fake_service = FakeFredService(
        category_children={fred.interest_rate_id: [{"id": 1, "name": "Treasury Bills"}]},
        category_series={1: [
            {"id": "TB3MS", "title": "T-Bill", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "NSA"},
        ]},
    )
    fake_search = FakeAzureSearchService(
        fred_series_by_category={"Treasury_Bills": [{"series_id": "TB3MS", "title": "T-Bill", "category": "Treasury_Bills"}]},
        observations_by_category={},
    )
    fake_daytona = FakeDaytonaService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "daytona_service", fake_daytona)

    await fred.sandbox_interest_rate()

    indicator_type, datasets = fake_daytona.calls[0]
    assert indicator_type == "interest_rate"
    assert [d.filename for d in datasets] == ["data_Treasury_Bills.json", "research.json"]


# --- LLM-driven selection --------------------------------------------------------


@pytest.mark.asyncio
async def test_selector_cpi_ingests_then_indexes_selected_observations(monkeypatch):
    fake_service = FakeFredService(
        category_series={fred.cpi_id: [
            {"id": "CPIAUCSL", "title": "CPI", "frequency": "Monthly", "units": "Index", "seasonal_adjustment": "SA"},
        ]},
        observations_by_id={"CPIAUCSL": [{"date": "2024-01-01", "value": "1.0"}]},
    )
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    structured_response = fred.ListFredSeriesMetadata(
        list_metadata=[fred.Reason(metadata=cpi_metadata(), reason="headline inflation")]
    )
    fake_agent = FakeAgent(structured_response)
    monkeypatch.setattr(fred, "create_agent", lambda *args, **kwargs: fake_agent)
    monkeypatch.setattr(fred, "ChatOpenAI", lambda *args, **kwargs: object())

    await fred.selector_cpi()

    assert fake_search.fred_series_docs[0]["series_id"] == "CPIAUCSL"
    assert fake_search.fred_observations_docs == [
        {"id": "CPIAUCSL_2024-01-01", "series_id": "CPIAUCSL", "date": "2024-01-01", "value": "1.0", "category": "cpi"}
    ]
    assert "category='cpi'" in fake_agent.prompts[0]


@pytest.mark.asyncio
async def test_selector_interest_rate_selects_and_indexes_per_sector(monkeypatch):
    fake_service = FakeFredService(
        category_children={fred.interest_rate_id: [{"id": 1, "name": "Treasury Bills"}]},
        category_series={1: [
            {"id": "TB3MS", "title": "T-Bill", "frequency": "Monthly", "units": "Percent", "seasonal_adjustment": "NSA"},
        ]},
        observations_by_id={"TB3MS": [{"date": "2024-01-01", "value": "5.0"}]},
    )
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "fred_service", fake_service)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    structured_response = fred.ListFredSeriesMetadata(
        list_metadata=[fred.Reason(
            metadata=cpi_metadata("TB3MS", "T-Bill"),
            reason="benchmark rate",
        )]
    )
    fake_agent = FakeAgent(structured_response)
    monkeypatch.setattr(fred, "create_agent", lambda *args, **kwargs: fake_agent)
    monkeypatch.setattr(fred, "ChatOpenAI", lambda *args, **kwargs: object())

    await fred.selector_interest_rate()

    assert fake_search.fred_observations_docs == [
        {"id": "TB3MS_2024-01-01", "series_id": "TB3MS", "date": "2024-01-01", "value": "5.0", "category": "Treasury_Bills"}
    ]
    assert "category='Treasury_Bills'" in fake_agent.prompts[0]


# --- Phase 6: RAG over economic documents ---------------------------------------


class FakeFirecrawlService:
    def __init__(self, documents=None):
        self._documents = documents or []
        self.search_calls = []

    async def search_economic_news(self, query, limit=5):
        self.search_calls.append((query, limit))
        return self._documents


@pytest.mark.asyncio
async def test_search_economic_documents_delegates_to_azure_search(monkeypatch):
    fake_search = FakeAzureSearchService(
        economic_document_results=[{"id": "abc", "title": "Fed holds rates steady"}]
    )
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    result = await fred.search_economic_documents.ainvoke({"query": "federal reserve rate decision"})

    assert result == [{"id": "abc", "title": "Fed holds rates steady"}]
    assert fake_search.economic_document_search_calls == [("federal reserve rate decision", 5)]


@pytest.mark.asyncio
async def test_search_economic_documents_records_tool_call_and_relevance(monkeypatch):
    fake_search = FakeAzureSearchService(
        economic_document_results=[{"id": "abc", "@search.rerankerScore": 1.5}]
    )
    fake_eval = EvaluationService()
    monkeypatch.setattr(fred, "azure_search_service", fake_search)
    monkeypatch.setattr(fred, "evaluation_service", fake_eval)

    await fred.search_economic_documents.ainvoke({"query": "federal reserve"})

    assert fake_eval.tool_call_success_rate("search_economic_documents") == 1.0
    assert fake_eval.average_retrieval_relevance("search_economic_documents") == 1.5


@pytest.mark.asyncio
async def test_index_economic_news_indexes_fetched_documents(monkeypatch):
    fake_firecrawl = FakeFirecrawlService(documents=[
        {"id": "abc", "title": "Fed holds rates steady", "content": "...", "source": "https://example.com"},
    ])
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "firecrawl_service", fake_firecrawl)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred.index_economic_news("inflation news", limit=3)

    assert fake_firecrawl.search_calls == [("inflation news", 3)]
    assert fake_search.economic_documents == [
        {"id": "abc", "title": "Fed holds rates steady", "content": "...", "source": "https://example.com"}
    ]


@pytest.mark.asyncio
async def test_index_economic_news_skips_indexing_when_nothing_found(monkeypatch):
    fake_firecrawl = FakeFirecrawlService(documents=[])
    fake_search = FakeAzureSearchService()
    monkeypatch.setattr(fred, "firecrawl_service", fake_firecrawl)
    monkeypatch.setattr(fred, "azure_search_service", fake_search)

    await fred.index_economic_news()

    assert fake_search.economic_documents == []
