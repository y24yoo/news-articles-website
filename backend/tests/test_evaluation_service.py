import pytest

from services.evaluation_service import EvaluationService, validate_report_html

# --- validate_report_html -----------------------------------------------------


def test_validate_report_html_accepts_well_formed_report():
    html = '<div class="prose">\n<h1>Title</h1><img src="chart_1.png"></div>'

    result = validate_report_html(html, expected_chart_count=1)

    assert result.valid is True
    assert result.img_tag_count == 1
    assert result.errors == []


def test_validate_report_html_rejects_missing_opening_wrapper():
    html = '<h1>Title</h1></div>'

    result = validate_report_html(html)

    assert result.valid is False
    assert any("does not start with" in error for error in result.errors)


def test_validate_report_html_rejects_missing_closing_wrapper():
    html = '<div class="prose"><h1>Title</h1>'

    result = validate_report_html(html)

    assert result.valid is False
    assert any("does not end with" in error for error in result.errors)


def test_validate_report_html_rejects_wrong_img_count():
    html = '<div class="prose"><img src="chart_1.png"></div>'

    result = validate_report_html(html, expected_chart_count=2)

    assert result.valid is False
    assert result.img_tag_count == 1
    assert any("expected 2" in error for error in result.errors)


def test_validate_report_html_ignores_chart_count_when_not_given():
    html = '<div class="prose"><img src="chart_1.png"></div>'

    result = validate_report_html(html)

    assert result.valid is True


def test_validate_report_html_tolerates_surrounding_whitespace():
    html = '  \n<div class="prose">content</div>\n  '

    result = validate_report_html(html)

    assert result.valid is True


# --- tool-call success rate / latency -----------------------------------------


@pytest.mark.asyncio
async def test_measure_tool_call_records_success():
    service = EvaluationService()

    async with service.measure_tool_call("my_tool"):
        pass

    assert service.tool_call_success_rate("my_tool") == 1.0
    assert service.average_tool_call_latency("my_tool") is not None


@pytest.mark.asyncio
async def test_measure_tool_call_records_failure_and_reraises():
    service = EvaluationService()

    with pytest.raises(ValueError):
        async with service.measure_tool_call("my_tool"):
            raise ValueError("boom")

    assert service.tool_call_success_rate("my_tool") == 0.0


@pytest.mark.asyncio
async def test_tool_call_success_rate_mixed_outcomes():
    service = EvaluationService()

    async with service.measure_tool_call("my_tool"):
        pass
    with pytest.raises(RuntimeError):
        async with service.measure_tool_call("my_tool"):
            raise RuntimeError("boom")

    assert service.tool_call_success_rate("my_tool") == 0.5


def test_tool_call_success_rate_none_when_no_data():
    service = EvaluationService()

    assert service.tool_call_success_rate() is None
    assert service.average_tool_call_latency() is None


@pytest.mark.asyncio
async def test_tool_call_success_rate_filters_by_tool_name():
    service = EvaluationService()

    async with service.measure_tool_call("tool_a"):
        pass
    with pytest.raises(RuntimeError):
        async with service.measure_tool_call("tool_b"):
            raise RuntimeError("boom")

    assert service.tool_call_success_rate("tool_a") == 1.0
    assert service.tool_call_success_rate("tool_b") == 0.0
    assert service.tool_call_success_rate() == 0.5


# --- report completion rate / HTML validation success rate ---------------------


def test_report_completion_rate_tracks_produced_vs_missing():
    service = EvaluationService()

    service.record_report_generation(True)
    service.record_report_generation(False)

    assert service.report_completion_rate() == 0.5


def test_report_completion_rate_none_when_no_data():
    service = EvaluationService()

    assert service.report_completion_rate() is None


def test_html_validation_success_rate_tracks_validate_and_record_report():
    service = EvaluationService()

    service.validate_and_record_report('<div class="prose">ok</div>')
    service.validate_and_record_report('missing wrapper')

    assert service.html_validation_success_rate() == 0.5


# --- indicator selection accuracy ----------------------------------------------


def test_indicator_selection_accuracy_counts_valid_and_invalid():
    service = EvaluationService()

    service.record_indicator_selection(["A", "B"], valid_series_ids={"A"})

    assert service.indicator_selection_accuracy() == 0.5


def test_indicator_selection_accuracy_none_when_no_data():
    service = EvaluationService()

    assert service.indicator_selection_accuracy() is None


def test_indicator_selection_accuracy_accumulates_across_calls():
    service = EvaluationService()

    service.record_indicator_selection(["A"], valid_series_ids={"A"})
    service.record_indicator_selection(["B"], valid_series_ids={"A"})

    assert service.indicator_selection_accuracy() == 0.5


# --- retrieval relevance ---------------------------------------------------------


def test_average_retrieval_relevance_reads_reranker_score():
    service = EvaluationService()

    service.record_retrieval_relevance("search_fred_indicators", [
        {"series_id": "A", "@search.rerankerScore": 2.0},
        {"series_id": "B", "@search.rerankerScore": 4.0},
    ])

    assert service.average_retrieval_relevance("search_fred_indicators") == 3.0


def test_average_retrieval_relevance_skips_results_without_score():
    service = EvaluationService()

    service.record_retrieval_relevance("search_fred_indicators", [
        {"series_id": "A"},
        {"series_id": "B", "@search.rerankerScore": 4.0},
    ])

    assert service.average_retrieval_relevance("search_fred_indicators") == 4.0


def test_average_retrieval_relevance_none_when_no_scores():
    service = EvaluationService()

    service.record_retrieval_relevance("search_fred_indicators", [{"series_id": "A"}])

    assert service.average_retrieval_relevance("search_fred_indicators") is None
    assert service.average_retrieval_relevance() is None


def test_average_retrieval_relevance_across_all_sources():
    service = EvaluationService()

    service.record_retrieval_relevance("fred", [{"@search.rerankerScore": 2.0}])
    service.record_retrieval_relevance("documents", [{"@search.rerankerScore": 4.0}])

    assert service.average_retrieval_relevance() == 3.0


# --- summary ------------------------------------------------------------------------


def test_summary_reports_none_for_unrecorded_metrics():
    service = EvaluationService()

    summary = service.summary()

    assert summary == {
        "tool_call_success_rate": None,
        "average_tool_call_latency_seconds": None,
        "report_completion_rate": None,
        "html_validation_success_rate": None,
        "indicator_selection_accuracy": None,
        "average_retrieval_relevance": None,
    }


@pytest.mark.asyncio
async def test_summary_reflects_recorded_metrics():
    service = EvaluationService()
    async with service.measure_tool_call("tool"):
        pass
    service.record_report_generation(True)
    service.validate_and_record_report('<div class="prose">ok</div>')
    service.record_indicator_selection(["A"], valid_series_ids={"A"})
    service.record_retrieval_relevance("tool", [{"@search.rerankerScore": 1.0}])

    summary = service.summary()

    assert summary["tool_call_success_rate"] == 1.0
    assert summary["report_completion_rate"] == 1.0
    assert summary["html_validation_success_rate"] == 1.0
    assert summary["indicator_selection_accuracy"] == 1.0
    assert summary["average_retrieval_relevance"] == 1.0
