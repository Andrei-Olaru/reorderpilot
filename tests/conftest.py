"""Suite-wide safety: tests never send traces to LangSmith, whatever the developer's .env says."""

import pytest


@pytest.fixture(autouse=True)
def _no_langsmith_tracing(monkeypatch):
    """Set the tracing flags to false; config.get_settings never lets .env override an existing variable."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def backend(tmp_path):
    """Fake live-call copilot backend: scripted LLM, tiny dataset + fake model, temp memory.db, offline embedding.

    Shared by the Streamlit tab tests and the /copilot API tests (no network, no real model, no real audio).
    """
    from reorderpilot.agent.tools import ToolContext
    from reorderpilot.copilot.call_session import CopilotBackend
    from reorderpilot.memory.vector_store import VectorMemory
    from tests.test_agent_loop import AS_OF as as_of
    from tests.test_agent_loop import FakeBundle, ScriptedLLM, make_tx
    from tests.test_graph import WordHashEmbedding

    tx, bundle = make_tx(), FakeBundle()
    ctx = ToolContext(as_of=as_of, db_path=tmp_path / "memory.db", tx_loader=lambda: tx,
                      bundle_loader=lambda: bundle)
    return CopilotBackend(llm=ScriptedLLM([]), ctx=ctx, make_rotator=lambda: None,
                          memory=VectorMemory(tmp_path / "chroma", embedding_function=WordHashEmbedding()))
