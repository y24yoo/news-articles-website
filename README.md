# Economic Intelligence Platform

An economic-analysis backend (`backend/fred.py` + `backend/agents/`) that pulls FRED
indicators, researches economic news, runs statistical analysis in a sandbox, and
writes a combined report — plus a separate, unrelated news-scraping backend
(`backend/articles.py`).

**Migration status:** all 8 phases of the Azure Foundry migration described in
[`AGENTS.md`](./AGENTS.md) are merged to `main` — MongoDB → Azure AI Search,
LangChain tools generalized, a Microsoft Agent Framework / Foundry multi-agent
workflow, RAG over economic documents, and an evaluation harness. See
[`architecture.svg`](./architecture.svg) for the current architecture.
**None of it has been run against live Azure, FRED, or Firecrawl credentials yet** —
everything is built and unit-tested against fakes. Run the scripts in
`backend/scripts/` once you have real credentials to actually verify it.

## Requirements

- **Python 3.10+** (the code uses `agent-framework`, which requires it, and modern
  type-hint syntax throughout). If you only have an older Python, see AGENTS.md's
  memory notes on installing one via Homebrew (`brew install python@3.12`).
- An Azure subscription with an AI Foundry project and an Azure AI Search resource,
  if you want to run anything for real (not required to run the test suite).

## Setup

```bash
cd backend
python3 -m venv ../.venv        # or use your own venv location
source ../.venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + pytest/ruff; use requirements.txt for runtime-only
```

Create `backend/api_key.py` (gitignored — never commit this):

```python
daytona_api = "YOUR_DAYTONA_API_KEY"
fred_api = "YOUR_FRED_API_KEY"
firecrawl_api = "YOUR_FIRECRAWL_API_KEY"
```

Copy `backend/.env.example` to `backend/.env` and fill in the real values:

```env
FOUNDRY_PROJECT_ENDPOINT=
FOUNDRY_MODEL=

FOUNDRY_MODELS_ENDPOINT=
FOUNDRY_EMBEDDING_MODEL=text-embedding-3-small

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_INDEX_FRED_SERIES=fred-series
AZURE_SEARCH_INDEX_FRED_OBSERVATIONS=fred-observations
AZURE_SEARCH_INDEX_DOCUMENTS=economic-documents

FRED_API_KEY=
DAYTONA_API_KEY=
FIRECRAWL_API_KEY=
```

Azure services authenticate via `DefaultAzureCredential` — no Azure keys are
hardcoded or read from `.env`. Satisfy it with, e.g.:

```bash
az login
```

There is no database to run — MongoDB was removed in Phase 5. Azure AI Search is
the only backing store for FRED data.

## Running

Everything below is invoked **from the `backend/` directory**, and most things
that import sibling packages (`fred`, `services`, `agents`) must be run **as a
module** (`python -m ...`), not as a bare script — running
`python scripts/foo.py` directly fails with `ModuleNotFoundError` because Python
puts the script's own directory on `sys.path`, not `backend/`.

```bash
cd backend

# Run the test suite (works with zero credentials — everything is mocked/faked)
pytest              # or: cd .. && pytest  (pytest.ini points pythonpath at backend/)

# Lint
ruff check .

# Run the full multi-agent pipeline for one category (needs real credentials)
python -m agents.orchestrator

# Manual smoke tests (each needs real credentials; not part of the test suite)
python -m scripts.foundry_smoke_test        # Phase 2: confirm the Foundry agent responds
python -m scripts.azure_search_smoke_test   # Phase 4/5: create indexes, ingest CPI, search
python -m scripts.rag_smoke_test            # Phase 6: ingest economic news, hybrid search
```

To ingest FRED data or run individual pipeline stages directly, use `fred.py`'s
functions from a Python shell (`cd backend && python`), e.g.:

```python
import asyncio
from fred import cpi, selector_cpi, sandbox_cpi

asyncio.run(selector_cpi())   # ingest CPI series, LLM-select the best 3, index observations
asyncio.run(sandbox_cpi())    # generate the CPI report (Daytona) + attach research
```

## Project layout

```text
backend/
├── fred.py                  # orchestration: ingestion, LLM selection, tools, report generation
├── articles.py               # separate news-scraping backend (untouched by the migration)
├── agents/                   # Phase 7 multi-agent workflow
│   ├── indicator_agent.py
│   ├── research_agent.py
│   ├── writer_agent.py
│   └── orchestrator.py
├── services/
│   ├── fred_service.py       # pure async FRED API client
│   ├── daytona_service.py    # sandboxed analysis (statistics + charts + HTML)
│   ├── foundry_service.py    # Microsoft Agent Framework / Foundry agent wrapper
│   ├── azure_search_service.py  # fred-series / fred-observations / economic-documents indexes
│   ├── firecrawl_service.py  # economic news search + scrape, for RAG ingestion
│   ├── evaluation_service.py # Phase 8: real metrics (tool success rate, HTML validation, ...)
│   └── report_prompts.py     # the shared Daytona analysis prompt
├── scripts/                   # manual, credential-requiring smoke tests (not in pytest)
├── tests/                     # pytest suite — fakes/mocks throughout, no live services
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # + pytest, ruff
└── .env.example
```

## FRED category reference

```text
parent id: https://api.stlouisfed.org/fred/category?category_id=22&api_key={fred_api}&file_type=json

cpi_id = 9
ppi_id = 31
interest_rate_id = 22
unemployment_rate_id = 32447
gdp_id = 106
shared_gdp_id = 33020
```

## Known gaps

- **Nothing has been run against live Azure/FRED/Firecrawl.** Every service was
  built against the real installed SDKs (verified method signatures, response
  shapes, and field names directly), and every code path is covered by unit
  tests using fakes — but the actual network calls have never happened.
- **`articles.py`'s Firecrawl calls are likely broken.** `firecrawl-py` 4.42+'s
  top-level `Firecrawl`/`AsyncFirecrawl` classes dropped general web
  search/scrape entirely (discovered while building Phase 6's RAG ingestion,
  which uses `AsyncV1FirecrawlApp` instead). `articles.py` was not touched —
  out of scope for the FRED/Foundry migration.
- **FastAPI** is mentioned in `AGENTS.md`'s target architecture but was never
  gated behind a specific migration phase, so it hasn't been built.
- **`economic-documents` is empty** until someone runs `index_economic_news()`
  (or the RAG smoke test) with real Firecrawl + Foundry-embedding credentials.
