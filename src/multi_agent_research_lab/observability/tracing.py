"""Tracing hooks.

This file intentionally avoids binding to one provider. When LANGSMITH_API_KEY or
LANGFUSE_* keys are configured, LangGraph/LangChain will already emit spans to that
provider automatically. As a provider-agnostic fallback (and so a trace always exists
even without cloud credentials), `save_trace` dumps the run's local JSON trace to disk.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.state import ResearchState


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def save_trace(state: ResearchState, run_name: str, out_dir: str = "reports/traces") -> Path:
    """Persist `state.trace` and `state.agent_results` as local trace evidence.

    Used when no LangSmith/Langfuse project is configured; the resulting file is
    referenced from the benchmark report as trace evidence for a run.
    """

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{run_name}-{timestamp}.json"
    payload = {
        "run_name": run_name,
        "route_history": state.route_history,
        "trace": state.trace,
        "agent_results": [result.model_dump() for result in state.agent_results],
        "errors": state.errors,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
