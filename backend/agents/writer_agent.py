from __future__ import annotations

from services.foundry_service import FoundryService

INSTRUCTIONS = (
    "You are the WriterAgent for an economic intelligence platform. Combine "
    "an indicator selection, research findings, and a statistical analysis "
    "report into one polished, well-organized economic report. Do not "
    "invent facts beyond what's given to you."
)

writer_agent = FoundryService(name="writer-agent", instructions=INSTRUCTIONS)


async def write_report(indicator_summary: str, research_summary: str, analysis_report_path: str) -> str:
    """Synthesize the pipeline's three prior stages into one final report."""
    return await writer_agent.ask(
        "Combine the following into one polished economic report:\n\n"
        f"## Indicator Selection\n{indicator_summary}\n\n"
        f"## Research Findings\n{research_summary}\n\n"
        f"## Statistical Analysis\nSee the generated report at: {analysis_report_path}"
    )
