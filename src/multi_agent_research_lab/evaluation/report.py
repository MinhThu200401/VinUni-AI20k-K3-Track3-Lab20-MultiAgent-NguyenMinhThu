"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    analysis: str = "",
    trace_links: list[str] | None = None,
) -> str:
    """Render benchmark metrics to a markdown report with a comparison table,
    optional trace links, and a free-text failure-mode analysis section.
    """

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    if trace_links:
        lines += ["", "## Trace evidence", ""]
        lines += [f"- {link}" for link in trace_links]

    if analysis:
        lines += ["", "## Analysis", "", analysis]

    return "\n".join(lines) + "\n"
