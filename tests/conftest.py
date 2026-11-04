"""Suite-wide safety: tests never send traces to LangSmith, whatever the developer's .env says."""

import pytest


@pytest.fixture(autouse=True)
def _no_langsmith_tracing(monkeypatch):
    """Set the tracing flags to false; config.get_settings never lets .env override an existing variable."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
