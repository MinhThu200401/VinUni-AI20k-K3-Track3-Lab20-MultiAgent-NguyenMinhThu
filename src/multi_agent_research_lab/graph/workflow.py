"""LangGraph workflow skeleton."""

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        shared_llm = llm or LLMClient()
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent(shared_llm, search or SearchClient())
        self.analyst = AnalystAgent(shared_llm)
        self.writer = WriterAgent(shared_llm)

    @staticmethod
    def _node(agent: BaseAgent) -> Callable[[ResearchState], dict[str, object]]:
        def invoke(data: ResearchState) -> dict[str, object]:
            result = agent.run(data)
            return dict(result.model_dump())

        return invoke

    def build(self) -> object:
        """Create a LangGraph graph.

        Suggested nodes: supervisor, researcher, analyst, writer, optional critic.
        """

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._node(self.supervisor))
        graph.add_node("researcher", self._node(self.researcher))
        graph.add_node("analyst", self._node(self.analyst))
        graph.add_node("writer", self._node(self.writer))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", "done": END},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Compile the graph, invoke it, and convert the result back to ResearchState.
        """

        graph = self.build()
        result = graph.invoke(  # type: ignore[attr-defined]
            state,
            config={"recursion_limit": self.supervisor.settings.max_iterations * 3},
        )
        return ResearchState.model_validate(result)
