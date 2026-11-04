"""Smoke test: the package imports, config defaults are right, and the shared schemas validate."""

import importlib
from datetime import date

import pytest
from pydantic import ValidationError

MODULES = [
    "reorderpilot.config", "reorderpilot.data.loader", "reorderpilot.data.clean", "reorderpilot.features",
    "reorderpilot.model.baselines", "reorderpilot.model.train", "reorderpilot.model.predict",
    "reorderpilot.model.evaluate", "reorderpilot.memory.store", "reorderpilot.agent.tools",
    "reorderpilot.agent.loop", "reorderpilot.agent.prompts", "reorderpilot.agent.llm",
    "reorderpilot.api.main", "reorderpilot.api.schemas",
]


def test_scaffold_imports_config_and_schemas(monkeypatch, tmp_path):
    for name in MODULES:
        importlib.import_module(name)

    from reorderpilot.api.schemas import FeedbackEvent, Recommendation
    from reorderpilot.config import get_settings

    monkeypatch.delenv("AS_OF_DATE", raising=False)
    monkeypatch.setenv("REORDERPILOT_DATA_DIR", str(tmp_path))
    settings = get_settings(env_file=None)
    assert settings.as_of_date == date(2011, 11, 1)
    assert settings.memory_db_path == tmp_path / "memory.db"

    Recommendation(stock_code="85099B", description="JUMBO BAG RED RETROSPOT", probability=0.4, reason_text="x")
    with pytest.raises(ValidationError):
        FeedbackEvent(customer_id="12345", outcome="maybe")


@pytest.mark.parametrize(("env", "expected"), [
    ({}, False),
    ({"LANGSMITH_TRACING": "true"}, True),
    ({"LANGCHAIN_TRACING_V2": "true"}, True),          # legacy name used by many LangSmith guides
    ({"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}, False),
])
def test_tracing_flag_accepts_both_env_names(monkeypatch, env, expected):
    from reorderpilot.config import get_settings

    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert get_settings(env_file=None).langsmith_tracing is expected
