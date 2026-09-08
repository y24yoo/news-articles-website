from services.report_prompts import build_economic_report_prompt


def test_build_economic_report_prompt_includes_all_data_paths():
    prompt = build_economic_report_prompt(
        data_paths=["/home/daytona/data/data.json1", "/home/daytona/data/data.json2"],
        chart_path_prefix="../../../indicators/gdp",
    )
    assert "'/home/daytona/data/data.json1'" in prompt
    assert "'/home/daytona/data/data.json2'" in prompt


def test_build_economic_report_prompt_uses_chart_path_prefix():
    prompt = build_economic_report_prompt(
        data_paths=["/home/daytona/data/data.json"],
        chart_path_prefix="../../../indicators/cpi",
    )
    assert '<img src="../../../indicators/cpi/chart_N.png"' in prompt


def test_build_economic_report_prompt_single_dataset():
    prompt = build_economic_report_prompt(
        data_paths=["/home/daytona/data/data.json"],
        chart_path_prefix="../../../indicators/ppi",
    )
    assert "Analyze the dataset located at: '/home/daytona/data/data.json'" in prompt
