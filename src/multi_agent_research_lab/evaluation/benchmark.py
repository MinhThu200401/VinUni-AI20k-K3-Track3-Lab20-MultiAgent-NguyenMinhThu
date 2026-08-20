"""Benchmark harness for single-agent vs multi-agent runs."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

# Approximate gpt-4o-mini pricing (USD per token). Adjust if the model changes.
_INPUT_PRICE_PER_TOKEN = 0.150 / 1_000_000
_OUTPUT_PRICE_PER_TOKEN = 0.600 / 1_000_000

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _estimate_cost_usd(state: ResearchState) -> float | None:
    """Sum token usage across every agent call and price it."""

    input_tokens = 0
    output_tokens = 0
    has_usage = False
    for result in state.agent_results:
        in_tok = result.metadata.get("input_tokens")
        out_tok = result.metadata.get("output_tokens")
        if in_tok is not None:
            input_tokens += in_tok
            has_usage = True
        if out_tok is not None:
            output_tokens += out_tok
            has_usage = True
    if not has_usage:
        return None
    return input_tokens * _INPUT_PRICE_PER_TOKEN + output_tokens * _OUTPUT_PRICE_PER_TOKEN


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of available sources actually cited (e.g. `[1]`) in the final answer."""

    if not state.sources or not state.final_answer:
        return None
    cited = {int(match) for match in _CITATION_PATTERN.findall(state.final_answer)}
    valid_cited = {index for index in cited if 1 <= index <= len(state.sources)}
    return min(len(valid_cited) / len(state.sources), 1.0)


def _quality_score(state: ResearchState) -> float | None:
    """Heuristic 0-10 quality proxy: presence, length, citations, and errors.

    Not a substitute for the peer-review rubric — use this to catch obvious
    regressions (empty answers, no citations) between benchmark runs.
    """

    if not state.final_answer:
        return 0.0
    score = 4.0
    length = len(state.final_answer)
    if length >= 200:
        score += 2.0
    elif length >= 80:
        score += 1.0
    if state.sources and _CITATION_PATTERN.search(state.final_answer):
        score += 3.0
    score -= min(len(state.errors), 3) * 1.0
    return max(0.0, min(score, 10.0))


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Run `runner(query)`, measure latency, and score the resulting state."""

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going on agent failure
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"Run raised {type(exc).__name__}: {exc}",
        )
        return None, metrics

    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost_usd(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if state.errors else 0.0,
        notes="; ".join(state.errors) if state.errors else "",
    )
    return state, metrics
