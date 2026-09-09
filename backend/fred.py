from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from api_key import firecrawl_api, fred_api
from services.azure_search_service import AzureSearchService
from services.daytona_service import DatasetFile, DaytonaService
from services.evaluation_service import EvaluationService
from services.firecrawl_service import FirecrawlService
from services.fred_service import FredApiError, FredService


class FredSeriesMetadata(BaseModel):
    id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    notes: str | None

class Reason(BaseModel):
    metadata: FredSeriesMetadata
    reason: str

class ListFredSeriesMetadata(BaseModel):
    list_metadata: list[Reason]


# Global
load_dotenv()
fred_service = FredService(api_key=fred_api)
daytona_service = DaytonaService()
azure_search_service = AzureSearchService()
firecrawl_service = FirecrawlService(api_key=firecrawl_api)
evaluation_service = EvaluationService()
cpi_id = 9
ppi_id = 31
interest_rate_id = 22
unemployment_rate_id = 32447
gdp_id = 106
shared_gdp_id = 33020

def _to_fred_series_document(category: str, metadata: dict) -> dict:
    """Map a FredService.get_series_metadata() dict onto the fred-series index schema."""
    return {
        "series_id": metadata["id"],
        "title": metadata["title"],
        "category": category,
        "frequency": metadata["frequency"],
        "units": metadata["units"],
        "seasonal_adjustment": metadata["seasonal_adjustment"],
        "notes": metadata.get("notes"),
    }

def _to_fred_observation_documents(category: str, selected_docs) -> list[dict]:
    """Flatten [{metadata: {id: ...}, data: [{date, value}, ...]}, ...] onto the
    fred-observations index schema (one row per series/date)."""
    documents = []
    for doc in selected_docs:
        series_id = doc["metadata"]["id"]
        for point in doc.get("data", []):
            documents.append({
                "id": f"{series_id}_{point['date']}",
                "series_id": series_id,
                "date": point["date"],
                "value": point["value"],
                "category": category,
            })
    return documents

async def _build_indicator_dataset(category: str) -> list[dict]:
    """Combine a category's indexed indicator metadata with its indexed
    observations, for feeding to Daytona's economic-report analysis."""
    series_metadata = await azure_search_service.get_fred_series(category)
    observations_by_series: dict[str, list[dict]] = {}
    for obs in await azure_search_service.get_observations_by_category(category):
        observations_by_series.setdefault(obs["series_id"], []).append(
            {"date": obs["date"], "value": obs["value"]}
        )
    return [
        {**metadata, "data": observations_by_series.get(metadata["series_id"], [])}
        for metadata in series_metadata
    ]

async def run_economic_analysis(category: str) -> str:
    """Run Daytona-based statistics/chart/report generation over a category's
    indexed indicator data. This is the "Analysis Tool" from AGENTS.md's
    Agent Design section -- pure numeric analysis, no research bundled in
    (that's ResearchAgent's job; see backend/agents/). Returns the path to
    the generated report.html.

    Records Phase 8 metrics: tool-call success rate/latency, report
    completion rate (did a report.html actually get produced), and HTML
    validation success rate (does it meet the strict format rules).
    """
    data = await _build_indicator_dataset(category)
    async with evaluation_service.measure_tool_call("run_economic_analysis"):
        report_path = await daytona_service.generate_economic_report(category, [DatasetFile("data.json", data)])
    report_file = Path(report_path)
    report_produced = report_file.exists()
    evaluation_service.record_report_generation(report_produced)
    if report_produced:
        evaluation_service.validate_and_record_report(report_file.read_text(encoding="utf-8"))
    return str(report_path)

async def _index_observations_for_selection(category: str, selected: list[Reason]) -> None:
    """Fetch live FRED observations for each LLM-selected indicator and index
    them into fred-observations. Records Phase 8's indicator selection
    accuracy: did the LLM pick from real, known indicators for the
    category, or hallucinate a series_id?"""
    known_series_ids = {doc["series_id"] for doc in await azure_search_service.get_fred_series(category)}
    evaluation_service.record_indicator_selection(
        selected_series_ids=[item.metadata.id for item in selected],
        valid_series_ids=known_series_ids,
    )
    selected_docs = []
    for item in selected:
        data = await fred_service.get_series_observations(item.metadata.id)
        selected_docs.append({"metadata": {"id": item.metadata.id}, "data": data})
    documents = _to_fred_observation_documents(category, selected_docs)
    if documents:
        await azure_search_service.index_fred_observations(documents)

async def _ingest_category_series(category: str, category_id: int, *, keep=lambda raw: True) -> None:
    """Fetch a FRED category's series and index each one's metadata into
    Azure AI Search's fred-series index."""
    series = await fred_service.get_category_series(category_id)
    documents = [
        _to_fred_series_document(category, FredService.get_series_metadata(raw))
        for raw in series
        if keep(raw)
    ]
    if documents:
        await azure_search_service.index_fred_series(documents)

async def cpi() -> None:
    await _ingest_category_series("cpi", cpi_id)

@tool
async def search_fred_indicators(category: str, query: str = ""):
    """Semantically search FRED indicator metadata for a category via Azure AI Search.

    `category` is one of cpi/ppi/gdp/shared_gdp/unemployment_rate, or an
    interest-rate sector name (see interest_rate() for the available names).
    `query`, if given (e.g. "important inflation indicators"), ranks results
    semantically; if empty, returns all indicators in the category.
    """
    async with evaluation_service.measure_tool_call("search_fred_indicators"):
        results = await azure_search_service.search_fred_series(query or "*", category=category)
    evaluation_service.record_retrieval_relevance("search_fred_indicators", results)
    return results

@tool
async def get_fred_observations(series_id: str, start_date: str | None = None, end_date: str | None = None):
    """Fetch a FRED series' observations directly from the FRED API.

    `start_date`/`end_date` are "YYYY-MM-DD" strings; both are optional
    (start_date defaults to a multi-year lookback, end_date to "today").
    """
    async with evaluation_service.measure_tool_call("get_fred_observations"):
        return await fred_service.get_series_observations(
            series_id, observation_start=start_date, observation_end=end_date
        )

@tool
async def search_economic_documents(query: str):
    """Hybrid search (keyword + vector + semantic reranking) over indexed
    economic research/news documents -- Federal Reserve releases, FOMC
    statements, BLS/BEA content, economic news, and Firecrawl content.
    """
    async with evaluation_service.measure_tool_call("search_economic_documents"):
        results = await azure_search_service.search_economic_documents(query)
    evaluation_service.record_retrieval_relevance("search_economic_documents", results)
    return results

async def index_economic_news(
    query: str = "latest economic news inflation jobs markets central banks",
    limit: int = 5,
) -> None:
    """Phase 6: fetch recent economic news via Firecrawl and index it into
    economic-documents for RAG. Covers the "economic news" / "Firecrawl
    content" sources from AGENTS.md's economic-documents spec; Federal
    Reserve releases, FOMC statements, and BLS/BEA content are not sourced
    by this function (no specific feed/URL for them is wired up yet).
    """
    documents = await firecrawl_service.search_economic_news(query, limit=limit)
    if documents:
        await azure_search_service.index_economic_documents(documents)

async def selector_cpi() -> None:
    await cpi()
    model = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    agent = create_agent(model, response_format=ListFredSeriesMetadata, tools=[search_fred_indicators])
    result = await agent.ainvoke({"messages": [{"role": "user", "content": (
                        "Select the best 3 indicators to represent the CPI. "
                        "You MUST call search_fred_indicators with category='cpi' "
                        "and only use those indicators. "
                        "Explain why each indicator is chosen in the reason object.")}]})
    await _index_observations_for_selection("cpi", result["structured_response"].list_metadata)

async def ppi() -> None:
    await _ingest_category_series("ppi", ppi_id)

async def observation_ppi() -> None:
    """No LLM selection for PPI -- index observations for every ingested indicator."""
    for indicator in await azure_search_service.get_fred_series("ppi"):
        await asyncio.sleep(0.25)
        data = await fred_service.get_series_observations(indicator["series_id"])
        documents = _to_fred_observation_documents("ppi", [{"metadata": {"id": indicator["series_id"]}, "data": data}])
        if documents:
            await azure_search_service.index_fred_observations(documents)

async def interest_rate() -> list[str]:
    """Ingests every interest-rate sector's series, keyed by sector name as
    the fred-series `category`. Returns the sector names discovered, since
    there's no Mongo collection listing to derive them from anymore."""
    categories = await fred_service.get_category_children(interest_rate_id)
    sector_names = [category["name"].replace(" ", "_") for category in categories]
    for category, sector_name in zip(categories, sector_names):
        await _ingest_category_series(sector_name, category["id"])
    return sector_names

async def selector_interest_rate() -> None:
    sector_names = await interest_rate()
    model = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    agent = create_agent(model, response_format=ListFredSeriesMetadata, tools=[search_fred_indicators])

    async def select_sector(rate: str) -> None:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": (
                    f"Select the best 2 indicators to represent the '{rate}' rate. "
                    f"You MUST call search_fred_indicators with category='{rate}' "
                    f"and only use those indicators. "
                    f"If '{rate}' is not an important rate to analysis economy pick 0 or 1 indicator"
                    f"Explain why each indicator is chosen in the reason object.")}]})
        await _index_observations_for_selection(rate, result["structured_response"].list_metadata)

    async with asyncio.TaskGroup() as tg:
        for rate in sector_names:
            tg.create_task(select_sector(rate))

async def unemployment_rate() -> None:
    await _ingest_category_series(
        "unemployment_rate",
        unemployment_rate_id,
        keep=lambda raw: raw["id"].startswith("UNRATE"),
    )

async def observation_unemployment_rate() -> None:
    """No LLM selection for unemployment rate -- index observations for every ingested indicator."""
    await unemployment_rate()
    for indicator in await azure_search_service.get_fred_series("unemployment_rate"):
        data = await fred_service.get_series_observations(indicator["series_id"])
        documents = _to_fred_observation_documents(
            "unemployment_rate", [{"metadata": {"id": indicator["series_id"]}, "data": data}]
        )
        if documents:
            await azure_search_service.index_fred_observations(documents)

async def gdp() -> None:
    await _ingest_category_series("gdp", gdp_id)

async def shared_gdp() -> None:
    await _ingest_category_series("shared_gdp", shared_gdp_id)

async def _refresh_observations(category: str) -> None:
    """No LLM selection for gdp/shared_gdp -- refresh observations for every
    ingested indicator in the category, tolerating individual bad responses."""
    for indicator in await azure_search_service.get_fred_series(category):
        await asyncio.sleep(0.25)
        try:
            data = await fred_service.get_series_observations(indicator["series_id"])
        except FredApiError as error:
            print("Bad response for:", indicator["series_id"])
            print(error)
            continue
        documents = _to_fred_observation_documents(category, [{"metadata": {"id": indicator["series_id"]}, "data": data}])
        if documents:
            await azure_search_service.index_fred_observations(documents)

async def observation_gdp() -> None:
    # await gdp()
    await _refresh_observations("gdp")

async def observation_shared_gdp() -> None:
    # await shared_gdp()
    await _refresh_observations("shared_gdp")

async def sandbox_cpi() -> None:
    data = await _build_indicator_dataset("cpi")
    research = await azure_search_service.search_economic_documents("CPI inflation trends and analysis")
    await daytona_service.generate_economic_report("cpi", [DatasetFile("data.json", data), DatasetFile("research.json", research)])

async def sandbox_ppi() -> None:
    data = await _build_indicator_dataset("ppi")
    research = await azure_search_service.search_economic_documents("PPI producer price inflation trends and analysis")
    await daytona_service.generate_economic_report("ppi", [DatasetFile("data.json", data), DatasetFile("research.json", research)])

async def sandbox_interest_rate() -> None:
    # Sector names are discovered dynamically (Azure AI Search facets have no
    # concept of "interest rate sector" grouping the way separate Mongo
    # collections did), so this re-ingests (idempotent) rather than assuming
    # a fixed, hardcoded sector list.
    sector_names = await interest_rate()
    datasets = [DatasetFile(f"data_{sector}.json", await _build_indicator_dataset(sector)) for sector in sector_names]
    research = await azure_search_service.search_economic_documents("interest rate monetary policy trends and analysis")
    datasets.append(DatasetFile("research.json", research))
    await daytona_service.generate_economic_report("interest_rate", datasets)

async def sandbox_unemployment_rate() -> None:
    data = await _build_indicator_dataset("unemployment_rate")
    research = await azure_search_service.search_economic_documents("unemployment labor market trends and analysis")
    await daytona_service.generate_economic_report(
        "unemployment_rate", [DatasetFile("data.json", data), DatasetFile("research.json", research)]
    )

async def sandbox_gdp() -> None:
    research = await azure_search_service.search_economic_documents("GDP economic growth trends and analysis")
    datasets = [
        DatasetFile("data.json1", await _build_indicator_dataset("gdp")),
        DatasetFile("data.json2", await _build_indicator_dataset("shared_gdp")),
        DatasetFile("research.json", research),
    ]
    await daytona_service.generate_economic_report("gdp", datasets)

async def summarizer():
    model = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    agent = create_agent(model)

    input_files = [
        "./indicators/cpi/report.html",
        "./indicators/ppi/report.html",
        "./indicators/gdp/report.html",
        "./indicators/rate/report.html",
        "./indicators/unemployment_rate/report.html",
    ]

    reports = []
    for file_path in input_files:
        path = Path(file_path)
        if path.exists():
            reports.append(f"<h2>{path.parent.name}</h2>\n{path.read_text(encoding='utf-8')}")
        else:
            reports.append(f"<h2>{path.parent.name}</h2>\n<p>Missing file: {file_path}</p>")

    prompt = f"""
                You are given several HTML economic indicator reports.

                TASK:
                - Summarize all reports into ONE clean HTML document.
                - Output MUST be valid HTML only (no markdown, no explanations).

                STRICT FORMAT REQUIREMENT:
                - The ENTIRE output must be wrapped inside:
                <div class="prose">
                ...
                </div>

                - Do NOT output anything before or after this div.
                - Keep headings and structure clean and readable.

                Include sections for:
                - CPI
                - PPI
                - GDP
                - Interest Rate
                - Unemployment Rate Reports: {chr(10).join(reports)}"""

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": prompt}]
    })

    html = result.get("structured_response") or result["messages"][-1].content

    # ---- HARD ENFORCEMENT (guardrail) ----
    html = html.strip()
    if not html.startswith('<div class="prose">'):
        html = f'<div class="prose">\n{html}\n</div>'

    output_dir = Path("./indicators/index")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "report.html"
    output_file.write_text(html, encoding="utf-8")

    return str(output_file)

if __name__ == "__main__":
    asyncio.run(summarizer())
