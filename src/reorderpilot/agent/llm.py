"""OpenAI-compatible LLM client with rate limiting, 429 backoff and an optional on-disk cache.

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
    cache_dir: Path | None = None          # set by eval scripts so reruns cost no tokens

    @classmethod
    def from_env(cls, model: str | None = None, cache_dir: Path | None = None) -> LLMConfig:
        s = get_settings()
        effort = os.environ.get("LLM_REASONING_EFFORT", "low")
        return cls(base_url=s.llm_base_url, api_key=s.llm_api_key, model=model or s.llm_model,
                   reasoning_effort=None if effort == "none" else effort, cache_dir=cache_dir)


@dataclass
class OpenAICompatibleLLM:
    """Calls /chat/completions. `client` and `sleep` are injectable for tests."""

    config: LLMConfig
    client: object | None = None
    sleep: Callable[[float], None] = time.sleep
    usage: dict = field(default_factory=lambda: {"calls": 0, "cache_hits": 0, "prompt_tokens": 0,
                                                 "completion_tokens": 0})
    _last_call: float = 0.0

    def __post_init__(self) -> None:
        if self.client is None:
            if not self.config.api_key:
                raise LLMError("LLM_API_KEY is not set (see .env.example)")
            # max_retries=0: we do our own backoff so we can honour retry-after and log it.
            self.client = openai.OpenAI(base_url=self.config.base_url, api_key=self.config.api_key,
                                        timeout=self.config.timeout_s, max_retries=0)

    def _request(self, messages: list[dict], tools: list[dict]) -> dict:
        req = {"model": self.config.model, "messages": messages, "temperature": self.config.temperature}
        if tools:
            req["tools"] = tools
        if self.config.reasoning_effort:
            req["reasoning_effort"] = self.config.reasoning_effort
        return req

    def _cache_path(self, req: dict) -> Path | None:
        if self.config.cache_dir is None:
            return None
        key = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
        return self.config.cache_dir / f"{key}.json"

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
            self.usage["cache_hits"] += 1
            return json.loads(cache.read_text(encoding="utf-8"))

        for attempt in range(self.config.max_retries + 1):
            self._throttle()
            try:
                resp = self.client.chat.completions.create(**req)
                break
            except openai.BadRequestError as e:
                if "tool_use_failed" in str(e):
                    raise MalformedToolCallError(str(e)[:300]) from e
                raise LLMError(f"request rejected: {str(e)[:300]}") from e
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as e:
                if attempt == self.config.max_retries:
                    raise LLMError(f"gave up after {attempt + 1} attempts: {type(e).__name__}") from e
                wait = _retry_after(e) or min(2 ** attempt, 30)
                log.warning("LLM %s, retrying in %.1fs", type(e).__name__, wait)
                self.sleep(wait)
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
        return out


def _retry_after(err: Exception) -> float | None:
    """Seconds from a `retry-after` header, if the provider sent one (capped at 60s)."""
    response = getattr(err, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return min(float(value), 60.0) if value is not None else None
    except ValueError:
        return None
