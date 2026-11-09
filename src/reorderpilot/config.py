"""Central configuration: paths, simulated date, split cutoffs, LLM settings.

Rule (SPEC section 5): OS environment variables win; `.env` is only a fallback
(`load_dotenv(override=False)` never overwrites a variable that is already set).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Fixed, not environment-driven (SPEC section 6).
RANDOM_SEED = 42
LABEL_WINDOW_DAYS = 30
MIN_PRIOR_ORDERS = 3  # cold-start: fewer prior orders -> abstain
TRAIN_CUTOFFS = [date(2010, m, 1) for m in range(3, 13)] + [date(2011, m, 1) for m in range(1, 7)]
VALID_CUTOFFS = [date(2011, 7, 1), date(2011, 8, 1)]
TEST_CUTOFFS = [date(2011, 9, 1), date(2011, 10, 1), date(2011, 11, 1)]

# Memory policy (SPEC section 7): days a declined product stays suppressed, by reason.
SUPPRESSION_DAYS = {"overstocked": 30, "seasonal": 30, "switched_supplier": 90}

DEFAULT_AS_OF_DATE = "2011-11-01"
DEFAULT_LLM_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
AGENT_MAX_STEPS = 6


@dataclass(frozen=True)
class Settings:
    """Values that may differ per machine or run (read from the environment)."""

    data_dir: Path
    as_of_date: date
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    langsmith_tracing: bool

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def transactions_path(self) -> Path:
        return self.processed_dir / "transactions.parquet"

    @property
    def memory_db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def chroma_dir(self) -> Path:
        """ChromaDB vector index (derived from memory.db, safe to delete and rebuild)."""
        return self.data_dir / "chroma"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "llm_cache"

    @property
    def llm_cache_db(self) -> Path:
        return self.data_dir / "llm_cache.db"


def get_settings(env_file: Path | None = PROJECT_ROOT / ".env") -> Settings:
    """Build Settings from OS env, falling back to `env_file` (pass None to skip .env, e.g. in tests)."""
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    data_dir = Path(os.environ.get("REORDERPILOT_DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return Settings(
        data_dir=data_dir,
        as_of_date=date.fromisoformat(os.environ.get("AS_OF_DATE", DEFAULT_AS_OF_DATE)),
        llm_base_url=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
        langsmith_tracing=_tracing_enabled(),
    )


def _tracing_enabled() -> bool:
    """LangSmith tracing flag: LANGSMITH_TRACING (current SDK name) or LANGCHAIN_TRACING_V2 (legacy name)."""
    return any(os.environ.get(name, "").strip().strip('"').lower() == "true"
               for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"))
