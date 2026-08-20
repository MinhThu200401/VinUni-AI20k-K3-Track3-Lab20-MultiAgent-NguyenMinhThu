from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        del system_prompt, user_prompt
        return LLMResponse(content="Evidence-backed output [1].", input_tokens=10, output_tokens=5)


class FakeSearch:
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        del query, max_results
        return [SourceDocument(title="Test source", url="https://example.com", snippet="Fact")]


def test_workflow_runs_all_agents() -> None:
    workflow = MultiAgentWorkflow(llm=FakeLLM(), search=FakeSearch())  # type: ignore[arg-type]
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    result = workflow.run(state)

    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert result.final_answer == "Evidence-backed output [1]."
    assert len(result.agent_results) == 3
