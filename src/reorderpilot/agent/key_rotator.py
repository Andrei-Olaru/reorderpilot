"""Groq API key failover pool + an on-disk SQLite cache for LLM responses.

`GroqKeyRotator.call(fn)` runs `fn(client)` and, on a retryable error (HTTP 429, 5xx, timeout, connection error),
puts the current key on cooldown and switches to the next key that is not cooling down. Only when every key is
cooling does it sleep (exponential backoff, or the provider's `retry-after`).

Note: Groq rate limits are per organization, so several keys from the SAME org do not add quota. The pool is for
resilience (a revoked/expired key, a key from another project you own), not for dodging limits.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import openai

T = TypeVar("T")

# APITimeoutError is a subclass of APIConnectionError; listed for readability.
RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError)


def parse_keys(raw: str | None, fallback: str = "") -> list[str]:
    """'k1, k2,,k3' -> ['k1', 'k2', 'k3'] (deduplicated, order kept). Falls back to [fallback] when empty."""
    keys = [k.strip().strip('"') for k in (raw or "").split(",")]
    keys = list(dict.fromkeys(k for k in keys if k))
    if not keys and fallback:
        keys = [fallback]
    return keys


def keys_from_env(fallback: str = "") -> list[str]:
    """Keys from `GROQ_API_KEYS` (comma-separated), else the single `fallback` key (usually LLM_API_KEY)."""
    return parse_keys(os.environ.get("GROQ_API_KEYS"), fallback=fallback)


def retry_after_seconds(err: Exception) -> float | None:
    """Seconds from a `retry-after` header, if the provider sent one (capped at 60s)."""
    response = getattr(err, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return min(float(value), 60.0) if value is not None else None
    except ValueError:
        return None


@dataclass
class GroqKeyRotator:
    """Round-robin failover over API keys. `client_factory(key)` builds one SDK client per key (lazily, cached)."""

    keys: list[str]
    client_factory: Callable[[str], object]
    max_retries: int = 5
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    index: int = 0
    _clients: dict[int, object] = field(default_factory=dict)
    _cooldown_until: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("no API keys: set GROQ_API_KEYS or LLM_API_KEY (see .env.example)")

    def client(self) -> object:
        """SDK client for the current key."""
        if self.index not in self._clients:
            self._clients[self.index] = self.client_factory(self.keys[self.index])
        return self._clients[self.index]

    def rotate(self, backoff_s: float) -> float:
        """Cool the current key for `backoff_s`, move to the best next key, return how long to wait before using it."""
        now = self.clock()
        self._cooldown_until[self.index] = now + backoff_s
        n = len(self.keys)
        for step in range(1, n):                      # first choice: a key that is not cooling at all
            candidate = (self.index + step) % n
            if self._cooldown_until.get(candidate, 0.0) <= now:
                self.index = candidate
                return 0.0
        # every key is cooling: wait for the one that frees up first (the current key counts as exactly backoff_s)
        best, wait = self.index, backoff_s
        for i in range(n):
            if i != self.index and self._cooldown_until[i] - now < wait:
                best, wait = i, self._cooldown_until[i] - now
        self.index = best
        return max(wait, 0.0)

    def call(self, fn: Callable[[object], T]) -> T:
        """Run `fn(client)`; on a retryable error fail over / back off. Re-raises the last error when out of retries."""
        for attempt in range(self.max_retries + 1):
            try:
                result = fn(self.client())
            except RETRYABLE_ERRORS as e:
                if attempt == self.max_retries:
                    raise
                wait = self.rotate(retry_after_seconds(e) or min(2 ** attempt, 30))
                if wait > 0:
                    self.sleep(wait)
            else:
                self._cooldown_until.pop(self.index, None)
                return result
        raise AssertionError("unreachable")  # the loop always returns or raises


class SQLiteResponseCache:
    """Key -> JSON value store in one SQLite file. Survives restarts; safe to share between processes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                         "created_at REAL NOT NULL)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open, commit on success, always close (an open handle locks the file on Windows)."""
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, key: str) -> dict | None:
        """Cached value, or None on a miss or an unreadable row."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
        try:
            value = json.loads(row[0]) if row else None
        except json.JSONDecodeError:
            value = None
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: str, value: dict) -> None:
        """Insert or overwrite one entry."""
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO llm_cache (key, value, created_at) VALUES (?, ?, ?)",
                         (key, json.dumps(value), time.time()))
