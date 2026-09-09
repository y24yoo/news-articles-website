from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

_PROSE_WRAPPER_OPEN = '<div class="prose">'
_PROSE_WRAPPER_CLOSE = "</div>"


@dataclass
class ToolCallRecord:
    tool_name: str
    success: bool
    latency_seconds: float
    error: str | None = None


@dataclass
class ReportValidationResult:
    valid: bool
    img_tag_count: int
    errors: list[str] = field(default_factory=list)


def validate_report_html(html: str, expected_chart_count: int | None = None) -> ReportValidationResult:
    """Check a generated report against the strict HTML format rules from
    services/report_prompts.py's own prompt: wrapped in exactly one
    <div class="prose">...</div>, and (if `expected_chart_count` is given)
    the <img> tag count matching the number of chart files produced.
    """
    errors: list[str] = []
    stripped = html.strip()
    if not stripped.startswith(_PROSE_WRAPPER_OPEN):
        errors.append(f'report does not start with {_PROSE_WRAPPER_OPEN}')
    if not stripped.endswith(_PROSE_WRAPPER_CLOSE):
        errors.append(f"report does not end with {_PROSE_WRAPPER_CLOSE}")
    img_tag_count = len(re.findall(r"<img\b", html, flags=re.IGNORECASE))
    if expected_chart_count is not None and img_tag_count != expected_chart_count:
        errors.append(f"expected {expected_chart_count} <img> tags, found {img_tag_count}")
    return ReportValidationResult(valid=not errors, img_tag_count=img_tag_count, errors=errors)


class EvaluationService:
    """Phase 8: measures the real metrics AGENTS.md names as examples --
    tool-call success rate, indicator selection accuracy, report completion
    rate, report generation latency, retrieval relevance, HTML validation
    success rate. Every number here comes from data this codebase actually
    produces (tool call outcomes, generated report.html files, Azure AI
    Search's own @search.rerankerScore) -- none of it is fabricated.
    """

    def __init__(self) -> None:
        self._tool_calls: list[ToolCallRecord] = []
        self._report_generation_results: list[bool] = []
        self._report_validations: list[ReportValidationResult] = []
        self._indicator_selection_hits: list[bool] = []
        self._retrieval_scores: dict[str, list[float]] = {}

    # -- tool-call success rate / latency ------------------------------------

    @asynccontextmanager
    async def measure_tool_call(self, tool_name: str):
        """Wrap a tool call, recording whether it raised and how long it took."""
        start = time.monotonic()
        try:
            yield
        except Exception as error:
            self._tool_calls.append(ToolCallRecord(tool_name, False, time.monotonic() - start, str(error)))
            raise
        else:
            self._tool_calls.append(ToolCallRecord(tool_name, True, time.monotonic() - start))

    def _filter_tool_calls(self, tool_name: str | None) -> list[ToolCallRecord]:
        if tool_name is None:
            return list(self._tool_calls)
        return [call for call in self._tool_calls if call.tool_name == tool_name]

    def tool_call_success_rate(self, tool_name: str | None = None) -> float | None:
        calls = self._filter_tool_calls(tool_name)
        if not calls:
            return None
        return sum(1 for call in calls if call.success) / len(calls)

    def average_tool_call_latency(self, tool_name: str | None = None) -> float | None:
        calls = self._filter_tool_calls(tool_name)
        if not calls:
            return None
        return sum(call.latency_seconds for call in calls) / len(calls)

    # -- report completion rate / HTML validation success rate --------------

    def record_report_generation(self, report_produced: bool) -> None:
        """Did this run actually produce a report.html file at all?"""
        self._report_generation_results.append(report_produced)

    def report_completion_rate(self) -> float | None:
        if not self._report_generation_results:
            return None
        return sum(self._report_generation_results) / len(self._report_generation_results)

    def validate_and_record_report(self, html: str, expected_chart_count: int | None = None) -> ReportValidationResult:
        """Of the reports that WERE produced, does this one meet the strict HTML format rules?"""
        result = validate_report_html(html, expected_chart_count=expected_chart_count)
        self._report_validations.append(result)
        return result

    def html_validation_success_rate(self) -> float | None:
        if not self._report_validations:
            return None
        return sum(1 for result in self._report_validations if result.valid) / len(self._report_validations)

    # -- indicator selection accuracy ----------------------------------------

    def record_indicator_selection(self, selected_series_ids: Sequence[str], valid_series_ids: Iterable[str]) -> None:
        """Did an LLM's indicator selection actually pick from real, known
        indicators for the category, or hallucinate a series_id? This is a
        validity proxy, not a judgment of *quality* of the picks -- there's
        no ground-truth "best indicator" label to compare against."""
        valid = set(valid_series_ids)
        self._indicator_selection_hits.extend(series_id in valid for series_id in selected_series_ids)

    def indicator_selection_accuracy(self) -> float | None:
        if not self._indicator_selection_hits:
            return None
        return sum(self._indicator_selection_hits) / len(self._indicator_selection_hits)

    # -- retrieval relevance --------------------------------------------------

    def record_retrieval_relevance(self, source: str, results: Sequence[dict]) -> None:
        """Record Azure AI Search's own semantic reranker score
        (@search.rerankerScore) for a batch of search results. Results
        without that key (e.g. filter-only queries) are skipped -- there's
        no relevance score to record for those."""
        for doc in results:
            score = doc.get("@search.rerankerScore")
            if score is not None:
                self._retrieval_scores.setdefault(source, []).append(score)

    def average_retrieval_relevance(self, source: str | None = None) -> float | None:
        if source is not None:
            scores = self._retrieval_scores.get(source, [])
        else:
            scores = [score for source_scores in self._retrieval_scores.values() for score in source_scores]
        if not scores:
            return None
        return sum(scores) / len(scores)

    # -- summary ----------------------------------------------------------------

    def summary(self) -> dict:
        """All Phase 8 metrics in one place. Any metric with no recorded
        data yet is None, not a fabricated 0 or 1."""
        return {
            "tool_call_success_rate": self.tool_call_success_rate(),
            "average_tool_call_latency_seconds": self.average_tool_call_latency(),
            "report_completion_rate": self.report_completion_rate(),
            "html_validation_success_rate": self.html_validation_success_rate(),
            "indicator_selection_accuracy": self.indicator_selection_accuracy(),
            "average_retrieval_relevance": self.average_retrieval_relevance(),
        }
