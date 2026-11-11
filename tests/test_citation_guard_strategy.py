"""MafLiteStrategy applies the citation guard to the streamed answer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from connectors.obo import RetrievalAuthorizationMode
from strategies import maf_lite_strategy as lite
from strategies.foundry_iq_context_provider import FoundryIQContextProvider

RETRIEVED = "https://dev.azure.com/org/proj/_git/repo?path=%2Fdocs%2F07-dns.md&version=GC9b8d91ea&_a=contents"
INVENTED = "https://docs.microsoft.com/azure/ai-services/standards/model-deployment-types"

# Split mid-link on purpose.
ANSWER_PARTS = [
    "Forward to the inbound endpoint. [DNS](",
    RETRIEVED[:40],
    RETRIEVED[40:] + ")\nSee ",
    f"[Standards]({INVENTED}) too. [DNS]({RETRIEVED})",
]


class FakeChatAgent:
    """Stands in for ChatAgent; retrieval happens inside the stream, as in production."""

    parts: list[str] = []
    provider = None
    retrieved: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get_new_thread(self):
        return object()

    async def run_stream(self, messages, **kwargs):
        if self.provider is not None:
            self.provider._retrieved_links = set(self.retrieved)
        for part in self.parts:
            yield SimpleNamespace(text=part, contents=[])


@pytest.fixture
def make_strategy(patch_dependencies, mock_config, monkeypatch):
    def make(*, mode=None, intent="question", history=None, retrieved=(RETRIEVED,)):
        if mode is not None:
            mock_config.get.side_effect = _with_override(mock_config.get.side_effect, "CITATION_GUARD_MODE", mode)
        monkeypatch.setattr(lite, "get_config", lambda: mock_config)
        monkeypatch.setattr(lite, "ChatAgent", FakeChatAgent)
        strategy = lite.MafLiteStrategy()
        strategy.conversation = {"id": "c", "messages": list(history or [])}
        strategy._read_prompt = AsyncMock(return_value="Instructions")
        strategy._classify_intent = AsyncMock(return_value=intent)
        strategy._cached_instructions = "Instructions"
        strategy._get_or_create_chat_client = lambda: object()
        strategy._ensure_user_memory = AsyncMock()
        provider = FoundryIQContextProvider()
        strategy._create_search_provider = AsyncMock(return_value=provider)
        FakeChatAgent.parts = ANSWER_PARTS
        FakeChatAgent.provider = provider
        FakeChatAgent.retrieved = tuple(retrieved)
        return strategy

    return make


def _with_override(original, key, value):
    def get(name, *args, **kwargs):
        if name == key:
            return value
        if callable(original):
            return original(name, *args, **kwargs)
        return args[0] if args else None

    return get


async def run(strategy):
    return "".join([c async for c in strategy.initiate_agent_flow("how do I fix DNS?") if isinstance(c, str)])


async def test_invented_and_repeated_links_are_removed_from_stream_and_history(make_strategy):
    strategy = make_strategy()
    streamed = await run(strategy)
    assert INVENTED not in streamed
    assert streamed.count(RETRIEVED) == 1
    assert "Standards too." in streamed
    assert strategy.conversation["messages"][-1] == {"role": "assistant", "text": streamed}


async def test_off_mode_leaves_the_answer_untouched(make_strategy):
    strategy = make_strategy(mode="off")
    assert await run(strategy) == "".join(ANSWER_PARTS)


async def test_log_only_mode_leaves_the_answer_untouched(make_strategy):
    strategy = make_strategy(mode="log_only")
    assert await run(strategy) == "".join(ANSWER_PARTS)


async def test_invalid_mode_enforces(make_strategy):
    strategy = make_strategy(mode="nonsense")
    assert INVENTED not in await run(strategy)


async def test_links_from_earlier_assistant_turns_remain_citable(make_strategy):
    history = [{"role": "assistant", "text": f"Earlier [DNS]({RETRIEVED})"}]
    strategy = make_strategy(intent="no_retrieval", history=history, retrieved=())
    streamed = await run(strategy)
    assert RETRIEVED in streamed and INVENTED not in streamed


async def test_a_link_that_was_never_retrieved_or_shown_is_removed(make_strategy):
    strategy = make_strategy(intent="no_retrieval", retrieved=())
    assert RETRIEVED not in await run(strategy)


async def test_foundry_iq_provider_records_raw_and_signed_links():
    record = {"title": "DNS", "link": RETRIEVED, "content": "text"}
    client = SimpleNamespace(retrieve=AsyncMock(return_value=[record]), mcp_config=None)
    provider = FoundryIQContextProvider(top_k=5, authorization_mode=RetrievalAuthorizationMode.SERVICE_ONLY)
    assert provider.retrieved_links == frozenset()
    with (
        patch("strategies.foundry_iq_context_provider.get_foundry_iq_client", return_value=client),
        patch("strategies.foundry_iq_context_provider.sign_blob_url", AsyncMock(return_value=RETRIEVED + "&sig=x")),
    ):
        from agent_framework import ChatMessage

        await provider.invoking(ChatMessage(role="user", text="dns"))
        assert provider.retrieved_links == {RETRIEVED, RETRIEVED + "&sig=x"}
        client.retrieve.return_value = []
        await provider.invoking(ChatMessage(role="user", text="dns again"))
        assert provider.retrieved_links == frozenset()
