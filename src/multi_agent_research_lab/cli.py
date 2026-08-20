"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import save_trace
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    request = _parse_query(query)
    response = LLMClient().complete(
        "You are a research assistant. Give a concise, accurate answer and state uncertainty.",
        request.query,
    )
    state = ResearchState(request=request, final_answer=response.content)
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
    )
    state.add_trace_event(
        "baseline",
        {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
    )
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", help="Path to benchmark config yaml")
    ] = "configs/lab_default.yaml",
    out_path: Annotated[
        str, typer.Option("--out", help="Where to write the markdown report")
    ] = "reports/benchmark_report.md",
) -> None:
    """Run baseline vs multi-agent over the configured queries and write a report."""

    _init()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    queries: list[str] = (config or {}).get("benchmark", {}).get("queries", [])
    if not queries:
        console.print(Panel.fit(f"No benchmark queries found in {config_path}", style="red"))
        raise typer.Exit(code=1)

    llm = LLMClient()

    def run_baseline(query: str) -> ResearchState:
        request = ResearchQuery(query=query)
        response = llm.complete(
            "You are a research assistant. Give a concise, accurate answer and state uncertainty.",
            request.query,
        )
        state = ResearchState(request=request, final_answer=response.content)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        return state

    def run_multi(query: str) -> ResearchState:
        request = ResearchQuery(query=query)
        return MultiAgentWorkflow(llm=llm).run(ResearchState(request=request))

    metrics = []
    trace_links: list[str] = []
    failures: list[str] = []
    for index, query in enumerate(queries, 1):
        label = f"Q{index}"
        console.print(f"[bold]Running {label}[/bold]: {query}")

        _, baseline_metrics = run_benchmark(f"baseline [{label}]", query, run_baseline)
        metrics.append(baseline_metrics)
        if baseline_metrics.failure_rate:
            failures.append(f"{baseline_metrics.run_name}: {baseline_metrics.notes}")

        multi_state, multi_metrics = run_benchmark(f"multi-agent [{label}]", query, run_multi)
        metrics.append(multi_metrics)
        if multi_state is not None:
            trace_path = save_trace(multi_state, f"multi-agent-{label}")
            trace_links.append(f"`{trace_path.as_posix()}` — trace for {label}: {query}")
        if multi_metrics.failure_rate:
            failures.append(f"{multi_metrics.run_name}: {multi_metrics.notes}")

    analysis_lines = [
        f"Ran {len(queries)} queries from `{config_path}` through both the single-agent "
        "baseline and the multi-agent workflow (see table above).",
    ]
    analysis_lines.append(
        "Failure modes observed:" if failures else "No run failures observed in this pass."
    )
    analysis_lines += [f"- {failure}" for failure in failures]

    analysis_lines += [
        "",
        "**Limitations of these metrics.** `failure_rate` only catches raised exceptions — "
        "it does not verify the answer is correct, so a confidently wrong or hallucinated "
        "answer still reads as 0% failure. `quality_score` is a length + citation-presence "
        "heuristic, and `citation_coverage` only checks that a `[n]` marker exists, not that "
        "it points to the right claim. These are infrastructure-style signals (did the run "
        "complete) rather than outcome-style signals (did the user get a correct answer) — "
        "the same gap the monitoring literature describes: a healthy-looking system can still "
        "fail the people using it. A trustworthy benchmark needs a human or Critic-agent pass "
        "that checks factual correctness against `sources`, not just presence of citations.",
    ]

    report = render_markdown_report(
        metrics, analysis="\n".join(analysis_lines), trace_links=trace_links
    )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report, encoding="utf-8")
    console.print(Panel.fit(f"Wrote {out_path}", title="Benchmark"))


if __name__ == "__main__":
    app()
