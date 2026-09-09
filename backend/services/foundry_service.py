from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

DEFAULT_AGENT_NAME = "economic-intelligence-assistant"
DEFAULT_AGENT_INSTRUCTIONS = (
    "You are a helpful assistant for an economic intelligence platform. "
    "Answer concisely and factually."
)


def _default_chat_client_factory() -> Any:
    from agent_framework_foundry import FoundryChatClient
    from azure.identity import DefaultAzureCredential

    # FoundryChatClient reads FOUNDRY_PROJECT_ENDPOINT / FOUNDRY_MODEL from the
    # environment itself (see backend/.env.example). DefaultAzureCredential
    # covers az login, managed identity, etc. without hardcoding any key.
    return FoundryChatClient(credential=DefaultAzureCredential())


class FoundryService:
    """Wraps a single Microsoft Agent Framework agent backed by Azure AI Foundry.

    Originally added in Phase 2 as one toolless agent; Phase 7 added optional
    `tools` so this same wrapper can back each of the multi-agent workflow's
    role-specific agents (see backend/agents/). The chat client is lazily
    created via an injectable factory so this module stays importable and
    testable without the agent-framework / azure SDKs installed, and without
    live Azure credentials.
    """

    def __init__(
        self,
        chat_client_factory: Callable[[], Any] = _default_chat_client_factory,
        name: str = DEFAULT_AGENT_NAME,
        instructions: str = DEFAULT_AGENT_INSTRUCTIONS,
        tools: Sequence[Callable[..., Any]] | None = None,
    ):
        self._chat_client_factory = chat_client_factory
        self._name = name
        self._instructions = instructions
        self._tools = tools
        self._agent: Any = None

    def _get_agent(self) -> Any:
        if self._agent is None:
            chat_client = self._chat_client_factory()
            self._agent = chat_client.as_agent(name=self._name, instructions=self._instructions, tools=self._tools)
        return self._agent

    async def ask(self, prompt: str) -> str:
        """Send a single prompt to the agent and return its text response."""
        agent = self._get_agent()
        response = await agent.run(prompt)
        return response.text
