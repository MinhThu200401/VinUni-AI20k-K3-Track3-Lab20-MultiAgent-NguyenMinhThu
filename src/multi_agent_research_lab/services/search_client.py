"""Search client abstraction for ResearcherAgent."""

import json
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Use Tavily when configured and an explicit local fallback otherwise.
        """

        if not self.settings.tavily_api_key:
            return [
                SourceDocument(
                    title="Local fallback source",
                    snippet=(
                        "No TAVILY_API_KEY is configured. Treat this as a search stub for: "
                        f"{query}. Configure Tavily for evidence-backed web research."
                    ),
                    metadata={"provider": "local", "is_fallback": True},
                )
            ]
        payload = json.dumps(
            {"api_key": self.settings.tavily_api_key, "query": query, "max_results": max_results}
        ).encode()
        request = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return [
            SourceDocument(
                title=item.get("title") or "Untitled",
                url=item.get("url"),
                snippet=item.get("content") or "",
                metadata={"score": item.get("score"), "provider": "tavily"},
            )
            for item in data.get("results", [])[:max_results]
        ]
