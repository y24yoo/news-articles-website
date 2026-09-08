from pathlib import Path
from types import SimpleNamespace

import pytest
from services.daytona_service import DatasetFile, DaytonaService


class FakeSandboxFs:
    def __init__(self, files):
        self._files = files
        self.download_calls = []

    def list_files(self, path):
        return self._files

    def download_files(self, requests):
        self.download_calls.extend(requests)
        return [
            SimpleNamespace(source=r.source, destination=r.destination, error=None, result=r.destination)
            for r in requests
        ]


class FakeSandbox:
    def __init__(self, files):
        self.fs = FakeSandboxFs(files)
        self.deleted = False

    def delete(self):
        self.deleted = True


class FakeDaytonaClient:
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def create(self):
        return self._sandbox


class FakeBackend:
    def __init__(self):
        self.uploaded = []

    def upload_files(self, files):
        self.uploaded.extend(files)


class FakeAgent:
    def __init__(self):
        self.prompts = []

    def stream(self, payload):
        self.prompts.append(payload["messages"][0]["content"])
        return iter([])


def make_service(files):
    sandbox = FakeSandbox(files)
    backend = FakeBackend()
    agent = FakeAgent()
    service = DaytonaService(
        daytona_client=FakeDaytonaClient(sandbox),
        backend_factory=lambda _sandbox: backend,
        agent_factory=lambda _backend, _model: agent,
        download_request_factory=lambda source, destination: SimpleNamespace(
            source=source, destination=destination
        ),
    )
    return service, sandbox, backend, agent


@pytest.mark.asyncio
async def test_generate_economic_report_uploads_all_datasets():
    service, _sandbox, backend, _agent = make_service(files=[])
    datasets = [DatasetFile("data.json1", {"a": 1}), DatasetFile("data.json2", {"b": 2})]

    await service.generate_economic_report("interest_rate", datasets)

    uploaded_paths = [path for path, _content in backend.uploaded]
    assert uploaded_paths == [
        "/home/daytona/data/data.json1",
        "/home/daytona/data/data.json2",
    ]


@pytest.mark.asyncio
async def test_generate_economic_report_builds_prompt_with_correct_chart_prefix():
    service, _sandbox, _backend, agent = make_service(files=[])

    await service.generate_economic_report("cpi", [DatasetFile("data.json", {})])

    assert len(agent.prompts) == 1
    assert '<img src="../../../indicators/cpi/chart_N.png"' in agent.prompts[0]


@pytest.mark.asyncio
async def test_generate_economic_report_downloads_only_html_and_png_files():
    files = [
        SimpleNamespace(name="report.html"),
        SimpleNamespace(name="chart_1.png"),
        SimpleNamespace(name="notes.txt"),
    ]
    service, sandbox, _backend, _agent = make_service(files=files)

    result = await service.generate_economic_report("ppi", [DatasetFile("data.json", {})])

    downloaded_sources = {call.source for call in sandbox.fs.download_calls}
    assert downloaded_sources == {
        "/home/daytona/output/report.html",
        "/home/daytona/output/chart_1.png",
    }
    assert result == Path("indicators/ppi/report.html")


@pytest.mark.asyncio
async def test_generate_economic_report_deletes_sandbox_when_done():
    service, sandbox, _backend, _agent = make_service(files=[])

    await service.generate_economic_report("gdp", [DatasetFile("data.json", {})])

    assert sandbox.deleted is True


@pytest.mark.asyncio
async def test_generate_economic_report_rejects_unknown_indicator_type():
    service, _sandbox, _backend, _agent = make_service(files=[])

    with pytest.raises(ValueError):
        await service.generate_economic_report("not_a_real_indicator", [DatasetFile("data.json", {})])


@pytest.mark.asyncio
async def test_generate_economic_report_accepts_explicit_output_dir():
    service, sandbox, _backend, agent = make_service(
        files=[SimpleNamespace(name="report.html")]
    )

    result = await service.generate_economic_report(
        "cpi", [DatasetFile("data.json", {})], output_dir="custom/output"
    )

    assert str(result) == "custom/output/report.html"
    assert sandbox.fs.download_calls[0].destination == "custom/output/report.html"
    assert "../../../custom/output/chart_N.png" in agent.prompts[0]
