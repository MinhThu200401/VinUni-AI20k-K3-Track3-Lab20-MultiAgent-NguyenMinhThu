from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_in_dependency_order() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()
    assert supervisor.run(state).route_history[-1] == "researcher"
    state.sources.append({"title": "A", "snippet": "Evidence"})  # type: ignore[arg-type]
    state.research_notes = "notes"
    assert supervisor.run(state).route_history[-1] == "analyst"
    state.analysis_notes = "analysis"
    assert supervisor.run(state).route_history[-1] == "writer"
    state.final_answer = "answer"
    assert supervisor.run(state).route_history[-1] == "done"
