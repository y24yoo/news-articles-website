# AGENTS.md

# Economic Intelligence Platform — Azure Foundry Migration

## Goal

Migrate the current economic AI project from:

- LangChain
- OpenAI Agents
- MongoDB

toward:

- Microsoft Foundry
- Microsoft Agent Framework
- Azure OpenAI
- Azure AI Search
- FastAPI
- Daytona

Keep the existing FRED economic-analysis functionality working throughout the migration.

Do not rewrite everything at once.

---

## Target Architecture

```text
Microsoft Foundry / Agent Framework
 ↓
Orchestrator Agent
 ├── FRED Tool
 │     ↓
 │ Azure AI Search
 │
 ├── Research Tool
 │     ↓
 │ Azure AI Search
 │
 └── Analysis Tool
       ↓
     Daytona
       ↓
 statistics + charts + HTML

 ↓
Final Economic Report
```

---

## Responsibilities

### Azure AI Search

Use Azure AI Search instead of MongoDB.

Create separate indexes.

### `fred-series`

Store searchable FRED indicator metadata:

```text
series_id
title
category
frequency
units
seasonal_adjustment
notes
```

Use semantic/hybrid search when selecting indicators.

Example:

```text
"important inflation indicators"
```

### `fred-observations`

Store numeric observations:

```text
series_id
date
value
category
```

Retrieve using filters.

Do NOT use vector search for raw numeric observations.

### `economic-documents`

Store:

- Federal Reserve releases
- FOMC statements
- BLS / BEA content
- economic news
- Firecrawl content

Use:

```text
keyword search
+
vector search
+
hybrid retrieval
```

for RAG.

---

## Agent Design

Start with ONE agent.

Do not build the multi-agent architecture immediately.

Initial agent should have tools:

```text
search_fred_indicators()
get_fred_observations()
search_economic_documents()
run_economic_analysis()
```

Later split into:

```text
Orchestrator
 ├── IndicatorAgent
 ├── ResearchAgent
 ├── AnalysisAgent
 └── WriterAgent
```

---

## Project Structure

Refactor toward:

```text
src/
├── agents/
│   ├── orchestrator.py
│   ├── indicator_agent.py
│   ├── research_agent.py
│   └── writer_agent.py
│
├── tools/
│   ├── fred_tools.py
│   ├── search_tools.py
│   └── analysis_tools.py
│
├── services/
│   ├── fred_service.py
│   ├── foundry_service.py
│   ├── search_service.py
│   └── daytona_service.py
│
├── api/
│   └── app.py
│
├── models/
│   └── economic.py
│
└── main.py

tests/
scripts/
```

Do not create files until they are needed.

---

## FRED Service

Move FRED API code into:

```text
src/services/fred_service.py
```

Functions should resemble:

```python
get_category_series()
get_category_children()
get_series_observations()
get_series_metadata()
```

Do not mix API requests with agent logic.

---

## Async Code

The existing code mixes:

```python
asyncio
requests
time.sleep
```

Fix this.

Do not use:

```python
requests.get()
time.sleep()
```

inside async functions.

Prefer:

```python
httpx.AsyncClient
await asyncio.sleep()
```

---

## Daytona

Keep Daytona.

Its role is sandboxed execution for:

- Python analysis
- statistics
- correlations
- charts
- HTML generation

Refactor duplicated functions like:

```text
sandbox_cpi()
sandbox_ppi()
sandbox_gdp()
sandbox_interest_rate()
sandbox_unemployment_rate()
```

into one reusable function:

```python
generate_economic_report(
    indicator_type,
    dataset,
)
```

Do not duplicate the same long prompt five times.

---

## FastAPI

Expose the application through FastAPI.

Minimum endpoints:

```text
GET /health
POST /analysis
GET /reports/{report_id}
```

Example:

```json
POST /analysis

{
  "topic": "US inflation"
}
```

---

## Environment Variables

Use `.env`.

Create `.env.example`.

```env
FOUNDRY_PROJECT_ENDPOINT=
FOUNDRY_MODEL=

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_INDEX_FRED_SERIES=fred-series
AZURE_SEARCH_INDEX_FRED_OBSERVATIONS=fred-observations
AZURE_SEARCH_INDEX_DOCUMENTS=economic-documents

FRED_API_KEY=
DAYTONA_API_KEY=
FIRECRAWL_API_KEY=
```

Never commit credentials.

---

## Authentication

Prefer:

```python
DefaultAzureCredential()
```

or:

```python
AzureCliCredential()
```

for Azure authentication.

Never hardcode Azure keys.

---

## Migration Plan

### Phase 1 — Refactor Current Code

Before adding Azure:

- separate FRED logic
- separate Daytona logic
- remove duplicated prompts
- fix blocking async calls
- add tests

Keep current functionality working.

---

### Phase 2 — Add Microsoft Foundry

Add:

```text
Microsoft Agent Framework
Microsoft Foundry
Azure OpenAI
```

Create one basic agent.

Confirm it can respond successfully.

---

### Phase 3 — Convert Tools

Replace LangChain `@tool` functions.

Instead of separate tools like:

```text
search_indicators_cpi
search_indicators_interest_rate
```

prefer generic tools:

```python
search_fred_indicators(category, query)

get_fred_observations(
    series_id,
    start_date,
    end_date,
)
```

---

### Phase 4 — Azure AI Search

Create:

```text
fred-series
fred-observations
economic-documents
```

Migrate CPI first.

Once CPI works, migrate:

```text
PPI
GDP
interest rates
unemployment
```

Do not remove MongoDB until Azure AI Search works.

---

### Phase 5 — Remove MongoDB

After all economic data works through Azure AI Search:

remove:

```text
pymongo
MongoClient
MongoDB configuration
```

---

### Phase 6 — Add RAG

Index economic documents.

Add:

```python
search_economic_documents(query)
```

Use Azure AI Search hybrid retrieval.

The agent should combine:

```text
FRED numerical data
+
economic research/news
```

before generating a report.

---

### Phase 7 — Multi-Agent Workflow

After the single agent works reliably:

```text
IndicatorAgent
    ↓
ResearchAgent
    ↓
AnalysisAgent
    ↓
WriterAgent
```

Do not add agents unless each one has a clear responsibility.

---

### Phase 8 — Evaluation

Measure real metrics.

Examples:

```text
tool-call success rate
indicator selection accuracy
report completion rate
report generation latency
retrieval relevance
HTML validation success rate
```

Never invent metrics.

---

## Testing

Use:

```bash
pytest
ruff check .
```

Unit tests should cover:

- FRED parsing
- Azure Search query construction
- input validation
- report validation
- agent tools

Cloud integration tests should be separate.

---

## Git Workflow

Use feature branches such as:

```text
feat/project-refactor
feat/foundry-agent
feat/azure-ai-search
feat/fred-tools
feat/research-rag
feat/daytona-refactor
feat/multi-agent
feat/fastapi
feat/evaluation
```

Use meaningful commits:

```text
refactor: extract FRED service
feat: add Foundry agent
feat: add Azure AI Search integration
feat: expose FRED data as agent tools
feat: add economic RAG search
refactor: generalize Daytona analysis
test: add agent evaluation suite
```

Avoid:

```text
update
fix
stuff
final
final2
```

---

## Security

Never commit:

```text
.env
API keys
Azure credentials
access tokens
connection strings
```

Always inspect:

```bash
git status
git diff
```

before committing.

---

## Claude code Rules

Before changing code:

1. inspect the repository
2. understand the current execution flow
3. identify the entry point
4. run existing tests if possible

When modifying code:

1. make small changes
2. preserve working behavior
3. avoid unrelated refactors
4. add tests
5. use typed Python
6. do not invent Azure SDK methods

After modifying code:

1. run tests
2. run linting
3. inspect the diff
4. summarize changed files
5. mention anything that could not be tested

Never claim an Azure integration works if credentials were unavailable.

---

# First Claude code Task

Start ONLY with the refactor.

Do not migrate to Azure yet.

Perform:

1. inspect the repository
2. extract FRED API logic into `FredService`
3. extract Daytona logic into `DaytonaService`
4. consolidate duplicated `sandbox_*` functions
5. extract the repeated report prompt
6. replace blocking calls inside async code
7. preserve current behavior
8. add tests
9. run tests
10. stop and show the changes

Wait for approval before starting the Foundry migration.
