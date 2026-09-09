import pytest
from services.foundry_service import FoundryService


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeAgent:
    def __init__(self):
        self.prompts = []

    async def run(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse(f"echo: {prompt}")


class FakeChatClient:
    def __init__(self):
        self.as_agent_calls = []
        self.agent = FakeAgent()

    def as_agent(self, **kwargs):
        self.as_agent_calls.append(kwargs)
        return self.agent


@pytest.mark.asyncio
async def test_ask_returns_agent_response_text():
    chat_client = FakeChatClient()
    service = FoundryService(chat_client_factory=lambda: chat_client)

    result = await service.ask("Hello!")

    assert result == "echo: Hello!"


@pytest.mark.asyncio
async def test_ask_passes_name_and_instructions_to_as_agent():
    chat_client = FakeChatClient()
    service = FoundryService(
        chat_client_factory=lambda: chat_client,
        name="my-agent",
        instructions="Be terse.",
    )

    await service.ask("ping")

    assert chat_client.as_agent_calls == [{"name": "my-agent", "instructions": "Be terse.", "tools": None}]


@pytest.mark.asyncio
async def test_ask_passes_tools_to_as_agent():
    chat_client = FakeChatClient()

    def one_plus_one():
        return 2

    service = FoundryService(chat_client_factory=lambda: chat_client, tools=[one_plus_one])

    await service.ask("ping")

    assert chat_client.as_agent_calls[0]["tools"] == [one_plus_one]


@pytest.mark.asyncio
async def test_ask_reuses_the_same_agent_across_calls():
    chat_client = FakeChatClient()
    service = FoundryService(chat_client_factory=lambda: chat_client)

    await service.ask("first")
    await service.ask("second")

    assert len(chat_client.as_agent_calls) == 1
    assert chat_client.agent.prompts == ["first", "second"]


@pytest.mark.asyncio
async def test_ask_only_creates_chat_client_once():
    calls = []

    def factory():
        calls.append(1)
        return FakeChatClient()

    service = FoundryService(chat_client_factory=factory)

    await service.ask("first")
    await service.ask("second")

    assert len(calls) == 1
