from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.report_prompts import build_economic_report_prompt

# Maps an indicator type to the report output directory used by the rest of the site.
INDICATOR_OUTPUT_DIRS = {
    "cpi": "indicators/cpi",
    "ppi": "indicators/ppi",
    "gdp": "indicators/gdp",
    "interest_rate": "indicators/rate",
    "unemployment_rate": "indicators/unemployment_rate",
}


@dataclass
class DatasetFile:
    """A dataset to upload into the sandbox before analysis, named as it should appear on disk."""

    filename: str
    data: Any


def _default_backend_factory(sandbox: Any) -> Any:
    from langchain_daytona import DaytonaSandbox

    return DaytonaSandbox(sandbox=sandbox)


def _default_agent_factory(backend: Any, model: str) -> Any:
    from deepagents import create_deep_agent

    return create_deep_agent(model=model, backend=backend)


def _default_download_request_factory(source: str, destination: str) -> Any:
    from daytona import FileDownloadRequest

    return FileDownloadRequest(source=source, destination=destination)


def _default_daytona_client() -> Any:
    from api_key import daytona_api
    from daytona import Daytona, DaytonaConfig

    return Daytona(DaytonaConfig(api_key=daytona_api))


class DaytonaService:
    """Runs sandboxed economic-report generation in Daytona.

    Heavy SDK dependencies (daytona, langchain_daytona, deepagents) are only imported
    lazily, the first time they're actually needed, so this module stays importable
    and the orchestration logic stays testable without those packages installed.
    """

    def __init__(
        self,
        daytona_client: Any | None = None,
        backend_factory: Callable[[Any], Any] = _default_backend_factory,
        agent_factory: Callable[[Any, str], Any] = _default_agent_factory,
        download_request_factory: Callable[[str, str], Any] = _default_download_request_factory,
        model: str = "gpt-5.4",
    ):
        self._daytona_client = daytona_client
        self._backend_factory = backend_factory
        self._agent_factory = agent_factory
        self._download_request_factory = download_request_factory
        self._model = model

    def _get_daytona_client(self) -> Any:
        if self._daytona_client is None:
            self._daytona_client = _default_daytona_client()
        return self._daytona_client

    async def generate_economic_report(
        self,
        indicator_type: str,
        datasets: Sequence[DatasetFile],
        output_dir: str | Path | None = None,
    ) -> Path:
        """Analyze `datasets` in a Daytona sandbox and download the resulting report/charts.

        Replaces the previous sandbox_cpi/sandbox_ppi/sandbox_gdp/sandbox_interest_rate/
        sandbox_unemployment_rate functions, which duplicated this same flow and prompt.
        """
        if output_dir is None:
            if indicator_type not in INDICATOR_OUTPUT_DIRS:
                raise ValueError(f"Unknown indicator_type: {indicator_type!r}")
            output_dir = INDICATOR_OUTPUT_DIRS[indicator_type]
        output_dir = Path(output_dir)

        daytona_client = self._get_daytona_client()
        sandbox = daytona_client.create()
        backend = self._backend_factory(sandbox)
        agent = self._agent_factory(backend, self._model)

        data_paths = [f"/home/daytona/data/{dataset.filename}" for dataset in datasets]
        upload_pairs = [
            (path, json.dumps(dataset.data).encode("utf-8"))
            for path, dataset in zip(data_paths, datasets)
        ]
        backend.upload_files(upload_pairs)

        prompt = build_economic_report_prompt(
            data_paths=data_paths,
            chart_path_prefix=f"../../../{output_dir.as_posix()}",
        )

        for step in agent.stream({"messages": [{"role": "user", "content": prompt}]}):
            for update in step.values():
                if update and (messages := update.get("messages")) and isinstance(messages, list):
                    for message in messages:
                        message.pretty_print()

        files = sandbox.fs.list_files("/home/daytona/output")
        files_to_download = [
            self._download_request_factory(
                f"/home/daytona/output/{f.name}",
                str(output_dir / f.name),
            )
            for f in files
            if f.name.endswith(".html") or f.name.endswith(".png")
        ]
        results = sandbox.fs.download_files(files_to_download)
        for result in results:
            if result.error:
                print(f"Error downloading {result.source}: {result.error}")
            else:
                print(f"Downloaded {result.source} to {result.result}")

        sandbox.delete()
        return output_dir / "report.html"
