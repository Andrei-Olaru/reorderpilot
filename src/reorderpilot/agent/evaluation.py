"""Helpers shared by the agent eval scripts (eval B, C, D): LLM setup, temp memory, answer checks, result files.

Evals never touch the real memory.db: every case runs against a fresh temporary SQLite file.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from reorderpilot.agent.grounding import CODE_RE
from reorderpilot.agent.llm import LLMConfig, OpenAICompatibleLLM
from reorderpilot.agent.tools import ToolContext
from reorderpilot.config import EVAL_RESULTS_DIR, get_settings
from reorderpilot.data.loader import load_transactions

SUPPRESSION_WORDS = ("suppress", "overstock", "declin", "hold off", "not recommend", "skip", "exclud", "avoid",
                     "paused", "removed", "not offer", "won't offer", "do not offer")


def common_args(description: str) -> argparse.ArgumentParser:
    """--limit, --model, --no-cache: shared by all agent eval scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--limit", type=int, default=None, help="run only the first N cases (saves tokens)")
    p.add_argument("--model", default=None, help="override LLM_MODEL")
    p.add_argument("--no-cache", action="store_true", help="do not reuse cached LLM responses")
    return p


def make_llm(model: str | None, use_cache: bool) -> OpenAICompatibleLLM:
    """Real LLM client; responses cached in DATA_DIR/llm_cache so reruns cost no tokens."""
    cache = get_settings().llm_cache_dir if use_cache else None
    return OpenAICompatibleLLM(LLMConfig.from_env(model=model, cache_dir=cache))


@contextmanager
def temp_ctx(as_of: date, memory_enabled: bool = True) -> Iterator[ToolContext]:
    """ToolContext backed by a throwaway memory.db."""
    with tempfile.TemporaryDirectory() as d:
        yield ToolContext(as_of=as_of, db_path=Path(d) / "memory.db", memory_enabled=memory_enabled)


_UNICODE_FIXES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
                                " ": " ", " ": " ", " ": " "})


def _norm(text: str) -> str:
    """Upper-case, unify unicode dashes/spaces (LLMs emit U+2011 etc.), drop dots, collapse whitespace."""
    return " ".join(text.translate(_UNICODE_FIXES).upper().replace(".", " ").split())


def catalog() -> dict[str, str]:
    """stock_code -> normalized description for every product in the data."""
    tx = load_transactions()
    return {c: _norm(d) for c, d in tx.drop_duplicates("stock_code")[["stock_code", "description"]].itertuples(
        index=False)}


def _collect(obj: object, key: str, found: set[str]) -> set[str]:
    """All values of `key` anywhere inside nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                found.add(v)
            _collect(v, key, found)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, key, found)
    return found


def tool_codes(tool_outputs: list[dict]) -> set[str]:
    """Stock codes that appeared in any tool result of the turn."""
    return _collect([t["result"] for t in tool_outputs], "stock_code", set())


def mentioned_products(answer: str, cat: dict[str, str], ignore: set[str] = frozenset()) -> set[str]:
    """Products named in the answer, by stock code or by full description (customer ids in `ignore`)."""
    text = _norm(answer)
    codes = {m for m in CODE_RE.findall(text) if m in cat and m not in ignore}
    by_desc = {c: d for c, d in cat.items() if len(d) >= 12 and d in text}
    # "HEART T-LIGHT HOLDER" inside "WHITE HANGING HEART T-LIGHT HOLDER" is one product, not two.
    longest = {c for c, d in by_desc.items() if not any(d != o and d in o for o in by_desc.values())}
    return codes | longest


def hallucinated_products(answer: str, tool_outputs: list[dict], cat: dict[str, str], user_message: str) -> set[str]:
    """Products in the answer that no tool returned (SPEC section 11 D: no hallucinated products)."""
    ids = set(CODE_RE.findall(user_message)) | _collect([t["result"] for t in tool_outputs], "customer_id", set())
    return mentioned_products(answer, cat, ids) - tool_codes(tool_outputs)


def classify_mention(answer: str, code: str, description: str) -> str:
    """'absent' | 'mentioned_suppressed' | 'recommended' for one product in an answer (line-level heuristic)."""
    desc = _norm(description)
    lines = [ln for ln in answer.splitlines() if code in _norm(ln) or (desc and desc in _norm(ln))]
    if not lines:
        return "absent"
    if all(any(w in ln.lower() for w in SUPPRESSION_WORDS) for ln in lines):
        return "mentioned_suppressed"
    return "recommended"


def write_results(name: str, data: dict, markdown: str) -> Path:
    """Write eval/results/<name>.json and <name>.md; returns the markdown path."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_RESULTS_DIR / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    md = EVAL_RESULTS_DIR / f"{name}.md"
    md.write_text(markdown, encoding="utf-8")
    return md
