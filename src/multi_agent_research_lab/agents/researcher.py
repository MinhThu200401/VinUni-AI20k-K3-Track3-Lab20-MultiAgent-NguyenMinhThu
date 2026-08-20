"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient | None = None, search: SearchClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Search, capture citations, and create concise notes.
        """

        state.sources = self.search.search(state.request.query, state.request.max_sources)
        source_text = "\n".join(
            f"[{index}] {source.title}: {source.snippet} ({source.url or 'no URL'})"
            for index, source in enumerate(state.sources, 1)
        )
        response = self.llm.complete(
            "You are a researcher. Summarize only the supplied sources; flag weak evidence.",
            f"Question: {state.request.query}\n\nSources:\n{source_text}",
        )
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event("researcher", {"source_count": len(state.sources)})
        return state
