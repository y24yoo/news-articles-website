import pytest
from agents import indicator_agent as indicator_agent_module
from agents import orchestrator
from agents import research_agent as research_agent_module
from agents import writer_agent as writer_agent_module


class FakeFoundryService:
    def __init__(self, response="fake response"):
        self._response = response
        self.prompts = []

    async def ask(self, prompt):
        self.prompts.append(prompt)
        return self._response


@pytest.mark.asyncio
async def test_select_indicators_asks_agent_with_category(monkeypatch):
    fake_agent = FakeFoundryService("CPIAUCSL and CPILFESL are the best indicators.")
    monkeypatch.setattr(indicator_agent_module, "indicator_agent", fake_agent)

    result = await indicator_agent_module.select_indicators("cpi")

    assert result == "CPIAUCSL and CPILFESL are the best indicators."
    assert "category='cpi'" in fake_agent.prompts[0]


@pytest.mark.asyncio
async def test_research_asks_agent_with_topic(monkeypatch):
    fake_agent = FakeFoundryService("Recent CPI data shows inflation cooling.")
    monkeypatch.setattr(research_agent_module, "research_agent", fake_agent)

    result = await research_agent_module.research("cpi")

    assert result == "Recent CPI data shows inflation cooling."
    assert "cpi" in fake_agent.prompts[0]


@pytest.mark.asyncio
async def test_write_report_combines_all_three_inputs(monkeypatch):
    fake_agent = FakeFoundryService("Final combined report.")
    monkeypatch.setattr(writer_agent_module, "writer_agent", fake_agent)

    result = await writer_agent_module.write_report(
        "indicator summary", "research summary", "indicators/cpi/report.html"
    )

    assert result == "Final combined report."
    prompt = fake_agent.prompts[0]
    assert "indicator summary" in prompt
    assert "research summary" in prompt
    assert "indicators/cpi/report.html" in prompt


@pytest.mark.asyncio
async def test_run_report_pipeline_calls_all_four_stages_in_order(monkeypatch):
    calls = []

    async def fake_select_indicators(category):
        calls.append(("select_indicators", category))
        return "indicator summary"

    async def fake_research(topic):
        calls.append(("research", topic))
        return "research summary"

    async def fake_run_economic_analysis(category):
        calls.append(("run_economic_analysis", category))
        return "indicators/cpi/report.html"

    async def fake_write_report(indicator_summary, research_summary, analysis_report_path):
        calls.append(("write_report", indicator_summary, research_summary, analysis_report_path))
        return "final report"

    monkeypatch.setattr(orchestrator, "select_indicators", fake_select_indicators)
    monkeypatch.setattr(orchestrator, "research", fake_research)
    monkeypatch.setattr(orchestrator, "run_economic_analysis", fake_run_economic_analysis)
    monkeypatch.setattr(orchestrator, "write_report", fake_write_report)

    result = await orchestrator.run_report_pipeline("cpi")

    assert result == "final report"
    assert calls == [
        ("select_indicators", "cpi"),
        ("research", "cpi"),
        ("run_economic_analysis", "cpi"),
        ("write_report", "indicator summary", "research summary", "indicators/cpi/report.html"),
    ]
