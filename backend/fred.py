from __future__ import annotations

import asyncio
from pathlib import Path

from api_key import fred_api
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pymongo import MongoClient
from services.daytona_service import DatasetFile, DaytonaService
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
CONNECTION_STRING = "mongodb://127.0.0.1:27017/?directConnection=true"
client = MongoClient(CONNECTION_STRING)
fred_service = FredService(api_key=fred_api)
daytona_service = DaytonaService()
cpi_id = 9
ppi_id = 31
interest_rate_id = 22
unemployment_rate_id = 32447
gdp_id = 106
shared_gdp_id = 33020

# Maps a category name to the Mongo (database, collection) holding its indicator
# metadata. Anything not listed here is treated as an interest-rate sector name
# (see interest_rate(), which creates one collection per sector).
_CATEGORY_COLLECTIONS = {
    "cpi": ("cpi", "cpi"),
    "ppi": ("ppi", "ppi"),
    "gdp": ("gdp", "gdp"),
    "shared_gdp": ("gdp", "shared_gdp"),
    "unemployment_rate": ("unemployment_rate", "unemployment_rate"),
}

def _resolve_indicator_collection(category: str):
    db_name, collection_name = _CATEGORY_COLLECTIONS.get(category, ("interest_rate", category))
    return client[db_name][collection_name]

@tool
def search_fred_indicators(category: str, query: str = ""):
    """Search stored FRED indicator metadata for a category.

    `category` is one of cpi/ppi/gdp/shared_gdp/unemployment_rate, or an
    interest-rate sector name (see interest_rate() for the available names).
    `query`, if given, filters by a case-insensitive substring match against
    the indicator title.
    """
    collection = _resolve_indicator_collection(category)
    mongo_filter = {"title": {"$regex": query, "$options": "i"}} if query else {}
    return list(collection.find(mongo_filter, {"_id": 0}))

@tool
async def get_fred_observations(series_id: str, start_date: str | None = None, end_date: str | None = None):
    """Fetch a FRED series' observations directly from the FRED API.

    `start_date`/`end_date` are "YYYY-MM-DD" strings; both are optional
    (start_date defaults to a multi-year lookback, end_date to "today").
    """
    return await fred_service.get_series_observations(series_id, observation_start=start_date, observation_end=end_date)

async def _ingest_category_series(collection, category_id, *, keep=lambda raw: True):
    """Fetch a FRED category's series and insert each one's metadata into `collection`."""
    series = await fred_service.get_category_series(category_id)
    for raw in series:
        if keep(raw):
            collection.insert_one(FredService.get_series_metadata(raw))

async def cpi():
    await _ingest_category_series(client["cpi"]["cpi"], cpi_id)

async def selector_cpi():
    await cpi()
    model = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    agent = create_agent(model, response_format=ListFredSeriesMetadata, tools=[search_fred_indicators])
    db = client["cpi"]
    result = agent.invoke({"messages": [{"role": "user", "content": (
                        "Select the best 3 indicators to represent the CPI. "
                        "You MUST call search_fred_indicators with category='cpi' "
                        "and only use those indicators. "
                        "Explain why each indicator is chosen in the reason object.")}]})
    docs = [item.model_dump() for item in result["structured_response"].list_metadata]
    db["selected_cpi"].insert_many(docs)

async def observation_cpi():
    await selector_cpi()
    db = client["cpi"]
    for obj in db["selected_cpi"].find():
        data = await fred_service.get_series_observations(obj["metadata"]["id"])
        db["selected_cpi"].update_one({"_id": obj["_id"]}, {"$set": {"data": data}})

async def ppi():
    await _ingest_category_series(client["ppi"]["ppi"], ppi_id)

async def observation_ppi():
    db = client["ppi"]
    for obj in db["ppi"].find({}, {"_id": 0}):
        await asyncio.sleep(0.25)
        data = await fred_service.get_series_observations(obj["id"])
        db["selected_ppi"].insert_one({"metadata": obj, "data": data})

async def interest_rate():
    db = client["interest_rate"]
    categories = await fred_service.get_category_children(interest_rate_id)
    for category in categories:
        name = category["name"].replace(" ", "_")
        series = await fred_service.get_category_series(category["id"])
        for raw in series:
            db[name].insert_one(FredService.get_series_metadata(raw))

async def selector_interest_rate():
    model = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    agent = create_agent(model, response_format=ListFredSeriesMetadata, tools=[search_fred_indicators])
    source_db = client["interest_rate"]
    target_db = client["selected_interest_rate"]
    tasks = []
    async with asyncio.TaskGroup() as tg:
        for rate in source_db.list_collection_names():
            task = tg.create_task(agent.ainvoke({"messages": [{"role": "user", "content": (
                        f"Select the best 2 indicators to represent the '{rate}' rate. "
                        f"You MUST call search_fred_indicators with category='{rate}' "
                        f"and only use those indicators. "
                        f"If '{rate}' is not an important rate to analysis economy pick 0 or 1 indicator"
                        f"Explain why each indicator is chosen in the reason object.")}]}))
            tasks.append((rate, task))

    for rate, task in tasks:
        result = task.result()
        structured = result["structured_response"]
        docs = [item.model_dump() for item in structured.list_metadata]
        if docs:
            target_db[rate].insert_many(docs)

async def observation_interest_rate():
    db = client["selected_interest_rate"]
    for collection in db.list_collection_names():
        for obj in db[collection].find({}):
            try:
                data = await fred_service.get_series_observations(obj["metadata"]["id"])
            except FredApiError as error:
                print("Bad response for:", obj["metadata"]["id"])
                print(error)
                continue
            db[collection].update_one({"_id": obj["_id"]}, {"$set": {"data": data}})

async def unemployment_rate():
    await _ingest_category_series(
        client["unemployment_rate"]["unemployment_rate"],
        unemployment_rate_id,
        keep=lambda raw: raw["id"].startswith("UNRATE"),
    )

async def observation_unemployment_rate():
    await unemployment_rate()
    db = client["unemployment_rate"]
    for obj in db["unemployment_rate"].find({}, {"_id": 0}):
        data = await fred_service.get_series_observations(obj["id"])
        db["selected_unemployment_rate"].insert_one({"metadata": obj, "data": data})

async def gdp():
    await _ingest_category_series(client["gdp"]["gdp"], gdp_id)

async def shared_gdp():
    await _ingest_category_series(client["gdp"]["shared_gdp"], shared_gdp_id)

async def _refresh_observations_upsert(source_collection, target_collection):
    """Fetch fresh observations for each doc in `source_collection` and upsert into `target_collection`."""
    for obj in source_collection.find({}, {"_id": 0}):
        await asyncio.sleep(0.25)
        try:
            data = await fred_service.get_series_observations(obj["id"])
        except FredApiError as error:
            print("Bad response for:", obj["id"])
            print(error)
            continue
        target_collection.update_one(
            {"metadata.id": obj["id"]},
            {"$set": {"metadata": obj, "data": data}},
            upsert=True,
        )

async def observation_gdp():
    # await gdp()
    db = client["gdp"]
    await _refresh_observations_upsert(db["gdp"], db["selected_gdp"])

async def observation_shared_gdp():
    # await shared_gdp()
    db = client["gdp"]
    await _refresh_observations_upsert(db["shared_gdp"], db["selected_shared_gdp"])

async def sandbox_cpi():
    db = client["cpi"]
    data = list(db["selected_cpi"].find({}, {"_id": 0}))
    await daytona_service.generate_economic_report("cpi", [DatasetFile("data.json", data)])

async def sandbox_ppi():
    db = client["ppi"]
    data = list(db["selected_ppi"].find({}, {"_id": 0}))
    await daytona_service.generate_economic_report("ppi", [DatasetFile("data.json", data)])

async def sandbox_interest_rate():
    db = client["selected_interest_rate"]
    datasets = [
        DatasetFile("data.json1", list(db["FRB_Rates_-_discount,_fed_funds,_primary_credit"].find({}, {"_id": 0}))),
        DatasetFile("data.json2", list(db["Monetary_Policy"].find({}, {"_id": 0}))),
        DatasetFile("data.json3", list(db["Treasury_Bills"].find({}, {"_id": 0}))),
        DatasetFile("data.json4", list(db["Treasury_Inflation-Indexed_Securities"].find({}, {"_id": 0}))),
    ]
    await daytona_service.generate_economic_report("interest_rate", datasets)

async def sandbox_unemployment_rate():
    db = client["unemployment_rate"]
    data = list(db["selected_unemployment_rate"].find({}, {"_id": 0}))
    await daytona_service.generate_economic_report("unemployment_rate", [DatasetFile("data.json", data)])

async def sandbox_gdp():
    db = client["gdp"]
    datasets = [
        DatasetFile("data.json1", list(db["selected_gdp"].find({}, {"_id": 0}))),
        DatasetFile("data.json2", list(db["selected_shared_gdp"].find({}, {"_id": 0}))),
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
