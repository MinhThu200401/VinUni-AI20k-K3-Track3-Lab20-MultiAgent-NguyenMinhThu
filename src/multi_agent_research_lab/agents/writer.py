"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Synthesize a clear response with source references.
        """

        sources = "\n".join(
            f"[{index}] {source.title} — {source.url or 'local fallback'}"
            for index, source in enumerate(state.sources, 1)
        )
        response = self.llm.complete(
            "You are a careful writer. Answer for the requested audience and cite sources as [n].",
            (
                f"Question: {state.request.query}\nAudience: {state.request.audience}\n\n"
                f"Analysis:\n{state.analysis_notes}\n\nAllowed citations:\n{sources}"
            ),
        )
        state.final_answer = response.content
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
        state.add_trace_event("writer", {})
        return state
