"""OpenAI-compatible LLM client with rate limiting, key failover + 429 backoff, and optional on-disk caches.

Any object with `complete(messages, tools) -> dict` can drive the agent loop (tests use a scripted fake).
The returned dict is an OpenAI-style assistant message: {"role", "content", "tool_calls": [...]}.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import openai
from langsmith import traceable

from reorderpilot.agent.key_rotator import (
    RETRYABLE_ERRORS,
    GroqKeyRotator,
    SQLiteResponseCache,
    keys_from_env,
)
from reorderpilot.config import get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model could not be reached or kept failing after retries (shown to the user as a clear error)."""


class MalformedToolCallError(LLMError):
    """The provider rejected the model's own tool call as invalid JSON (Groq: HTTP 400 'tool_use_failed')."""


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


@dataclass
class LLMConfig:
    """Connection + behaviour settings. Built from env via `from_env()`; model name is never hard-coded here."""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout_s: float = 60.0
    max_retries: int = 5
    requests_per_minute: int = 25          # stay under Groq free tier (~30 RPM)
    reasoning_effort: str | None = "low"   # gpt-oss on Groq; "none" in env disables (other providers)
    cache_dir: Path | None = None          # set by eval scripts so reruns cost no tokens (one JSON file per request)
    cache_db: Path | None = None           # SQLite response cache for the app (LLM_RESPONSE_CACHE=true)
    api_keys: list[str] = field(default_factory=list)  # failover pool; empty -> [api_key]

    @classmethod
    def from_env(cls, model: str | None = None, cache_dir: Path | None = None) -> LLMConfig:
        s = get_settings()
        effort = os.environ.get("LLM_REASONING_EFFORT", "low")
        keys = keys_from_env(fallback=s.llm_api_key)
        use_db = os.environ.get("LLM_RESPONSE_CACHE", "").strip().strip('"').lower() == "true"
        return cls(base_url=s.llm_base_url, api_key=keys[0] if keys else "", model=model or s.llm_model,
                   reasoning_effort=None if effort == "none" else effort, cache_dir=cache_dir,
                   cache_db=s.llm_cache_db if use_db else None, api_keys=keys)


@dataclass
class OpenAICompatibleLLM:
    """Calls /chat/completions through a key-failover pool. `client` and `sleep` are injectable for tests."""

    config: LLMConfig
    client: object | None = None
    sleep: Callable[[float], None] = time.sleep
    usage: dict = field(default_factory=lambda: {"calls": 0, "cache_hits": 0, "prompt_tokens": 0,
                                                 "completion_tokens": 0})
    _last_call: float = 0.0
    rotator: GroqKeyRotator | None = None
    db_cache: SQLiteResponseCache | None = None

    def __post_init__(self) -> None:
        if self.client is not None:            # tests: one injected client stands in for one key
            keys, factory = ["injected"], lambda _key: self.client
        else:
            keys = self.config.api_keys or ([self.config.api_key] if self.config.api_key else [])
            if not keys:
                raise LLMError("no API key: set GROQ_API_KEYS or LLM_API_KEY (see .env.example)")
            factory = self._make_client
        self.rotator = GroqKeyRotator(keys, factory, max_retries=self.config.max_retries, sleep=self.sleep)
        if self.config.cache_db is not None:
            self.db_cache = SQLiteResponseCache(self.config.cache_db)

    def _make_client(self, key: str) -> openai.OpenAI:
        # max_retries=0: the rotator does its own failover/backoff so it can honour retry-after and log it.
        return openai.OpenAI(base_url=self.config.base_url, api_key=key, timeout=self.config.timeout_s,
                             max_retries=0)

    def _request(self, messages: list[dict], tools: list[dict]) -> dict:
        req = {"model": self.config.model, "messages": messages, "temperature": self.config.temperature}
        if tools:
            req["tools"] = tools
        if self.config.reasoning_effort:
            req["reasoning_effort"] = self.config.reasoning_effort
        return req

    @staticmethod
    def _cache_key(req: dict) -> str:
        return hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()

    def _cache_path(self, req: dict) -> Path | None:
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / f"{self._cache_key(req)}.json"

    def _throttle(self) -> None:
        """Client-side rate limiter: keep at least 60/RPM seconds between requests."""
        gap = 60.0 / self.config.requests_per_minute
        wait = self._last_call + gap - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last_call = time.monotonic()

    @traceable(run_type="llm", name="llm.complete",
               process_inputs=lambda d: {"messages": d.get("messages"), "n_tools": len(d.get("tools") or [])})
    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """One chat completion -> assistant message dict. Retries 429/5xx/timeouts with exponential backoff."""
        req = self._request(messages, tools)
        cache = self._cache_path(req)
        if cache is not None and cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                self.usage["cache_hits"] += 1
                return cached
            except (json.JSONDecodeError, OSError):
                log.warning("ignoring corrupt LLM cache file %s", cache.name)  # fall through to a real call
        db_key = self._cache_key(req) if self.db_cache is not None else None
        if db_key is not None and (hit := self.db_cache.get(db_key)) is not None:
            self.usage["cache_hits"] += 1
            return hit

        def attempt(client: object) -> object:
            self._throttle()
            return client.chat.completions.create(**req)

        try:
            resp = self.rotator.call(attempt)
        except openai.BadRequestError as e:
            if "tool_use_failed" in str(e):
                raise MalformedToolCallError(str(e)[:300]) from e
            raise LLMError(f"request rejected: {str(e)[:300]}") from e
        except RETRYABLE_ERRORS as e:
            raise LLMError(f"gave up after {self.config.max_retries + 1} attempts: {type(e).__name__}") from e
        except openai.APIError as e:
            raise LLMError(f"LLM API error: {str(e)[:300]}") from e

        msg = resp.choices[0].message
        out = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                 for tc in msg.tool_calls]
        self.usage["calls"] += 1
        if resp.usage:
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out), encoding="utf-8")
        if db_key is not None:
            self.db_cache.set(db_key, out)
        return out

