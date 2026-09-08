from __future__ import annotations

from collections.abc import Sequence

_ECONOMIC_REPORT_PROMPT_TEMPLATE = """Analyze the dataset located at: {data_paths} and produce
            a professional economic research report in the style of the Federal Reserve Bank of New York's
            Liberty Street Economics blog (specifically modeled after "Monitoring Real Activity in Real Time:
            The Weekly Economic Index").

            ## Analysis Requirements

            Conduct a thorough economic analysis of the dataset including:
            1. Identify the key economic indicators present in the data and their time coverage
            2. Compute summary statistics (mean, median, std deviation, min/max, recent vs. historical values)
            3. Calculate period-over-period changes (week-over-week, month-over-month, year-over-year as applicable)
            4. Identify significant turning points, recessions, recoveries, or anomalies
            5. Compute correlations between indicators if multiple series exist
            6. If the data contains a composite index, decompose it into contributing components
            7. Compare current readings to historical benchmarks (e.g., pre-pandemic, prior recessions)

            ## File Output Requirements

            - Create directory: /home/daytona/output/
            - Create chart files: /home/daytona/output/chart_1.png, chart_2.png, chart_3.png, etc.
            Charts should include (as supported by the data):
                * A headline time-series chart of the main index/indicator
                * A chart highlighting recession periods (shaded regions) vs. the indicator
                * Component contribution chart (stacked bar or area chart)
                * Recent-period zoomed-in view (last 12-24 months)
                * Cross-indicator comparison or correlation visualization
                * Distribution or histogram of changes/returns
            - Save the final report as a SINGLE HTML file: /home/daytona/output/report.html

            ## Report Content & Structure

            The report should read like a Liberty Street Economics post — analytical, data-driven, and
            written for an informed-but-general audience. Include:

            1. **Title** — informative and specific (e.g., "Tracking [Indicator]: What the Latest Data Reveal")
            2. **Lead paragraph** — state the motivation: why this measurement matters, what question it answers
            3. **Background section** — explain the indicator(s), methodology, and economic context
            4. **Findings section** — walk through each chart with interpretation; reference what the data show
            5. **Historical comparison** — situate current readings against past episodes
            6. **Caveats / limitations** — note data constraints, revision risk, or interpretive cautions
            7. **Conclusion** — synthesize the takeaway and policy or forecasting implications

            Write in clear, measured prose. Use specific numbers from the data. Reference each chart inline
            in the narrative (e.g., "As shown in the chart below..."). Avoid hype; mirror the sober analytical
            tone of Federal Reserve research blogs.

            ## HTML Format Requirements (strict)

            - NO CSS, NO JavaScript — pure HTML only
            - The count of <img> tags in report.html MUST equal the count of chart_*.png files produced
            - Insert each PNG using a relative path, formatted as: <img src="{chart_path_prefix}/chart_N.png" alt="...">
            - The entire report body MUST be wrapped exactly as follows:

            <div class="prose">
                (all report content goes here — headings, paragraphs, images, etc.)
            </div>

            Use semantic HTML inside the wrapper: <h1>, <h2>, <h3> for hierarchy, <p> for paragraphs,
            <ul>/<ol> for lists, <table> for tabular data, and <img> for charts. Place each <img> tag
            immediately after the paragraph that introduces or discusses it.

            ## Final Verification

            Before finishing, verify:
            - All chart_*.png files exist in /home/daytona/output/
            - report.html exists, opens with <div class="prose"> and closes with </div>
            - The number of <img> tags equals the number of chart_*.png files
            - Every chart referenced in the HTML actually exists on disk
            - The narrative references specific values from the data, not generic statements
            """


def build_economic_report_prompt(data_paths: Sequence[str], chart_path_prefix: str) -> str:
    """Build the shared Daytona analysis prompt for a set of uploaded dataset paths."""
    formatted_paths = ", ".join(f"'{path}'" for path in data_paths)
    return _ECONOMIC_REPORT_PROMPT_TEMPLATE.format(
        data_paths=formatted_paths,
        chart_path_prefix=chart_path_prefix,
    )
