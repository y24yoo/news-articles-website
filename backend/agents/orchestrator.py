"""Phase 7 of the Azure Foundry migration (see AGENTS.md): the multi-agent
workflow, run only after the single agent (Phase 2) works reliably.

IndicatorAgent -> ResearchAgent -> AnalysisAgent -> WriterAgent

AnalysisAgent is deliberately not implemented as a separate LLM-wrapped
agent here: its entire job would be "call run_economic_analysis," a
deterministic tool call with no real decision for an LLM to make (Daytona's
own internal deep-agent already supplies the analytical intelligence for
that step -- see services/daytona_service.py). Per AGENTS.md's own rule --
"Do not add agents unless each one has a clear responsibility" -- wrapping
it in another agent would add a stage with no responsibility beyond what
the tool already does. The orchestrator calls it directly instead.
"""

from __future__ import annotations

import asyncio

from fred import run_economic_analysis

from agents.indicator_agent import select_indicators
from agents.research_agent import research
from agents.writer_agent import write_report


async def run_report_pipeline(category: str) -> str:
    """Run the full IndicatorAgent -> ResearchAgent -> AnalysisAgent ->
    WriterAgent pipeline for `category` and return the final report text."""
    indicator_summary = await select_indicators(category)
    research_summary = await research(category)
    analysis_report_path = await run_economic_analysis(category)
    return await write_report(indicator_summary, research_summary, analysis_report_path)


if __name__ == "__main__":
    print(asyncio.run(run_report_pipeline("cpi")))
