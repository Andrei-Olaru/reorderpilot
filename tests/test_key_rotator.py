"""Key failover pool + SQLite response cache: failover without sleeping, backoff when all keys cool, persistence."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from langsmith import tracing_context

from reorderpilot.agent.key_rotator import GroqKeyRotator, SQLiteResponseCache, parse_keys
from reorderpilot.agent.llm import LLMConfig, LLMError, OpenAICompatibleLLM


@pytest.fixture(autouse=True)
def no_tracing():
    """Tests must never send traces to LangSmith, whatever .env says."""
    with tracing_context(enabled=False):
        yield


def _err(cls, status, headers=None):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://x"))
    return cls("err", response=response, body=None)


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s
        self.sleeps.append(s)


def _rotator(keys, outcomes_by_key, max_retries=5):
    """Rotator whose 'client' for key k is just k; fn pops the next outcome scripted for that key."""
    clock = FakeClock()
    calls = []

    def fn(key):
        calls.append(key)
        out = outcomes_by_key[key].pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    rot = GroqKeyRotator(keys, client_factory=lambda k: k, max_retries=max_retries, sleep=clock.sleep, clock=clock)
    return rot, fn, calls, clock


def test_parse_keys_strips_dedupes_and_falls_back():
    assert parse_keys(' k1, "k2",,k1 ,k3') == ["k1", "k2", "k3"]
    assert parse_keys("", fallback="solo") == ["solo"]
    assert parse_keys(None) == []


def test_rate_limited_key_fails_over_to_next_key_without_sleeping():
    rot, fn, calls, clock = _rotator(["a", "b"], {"a": [_err(openai.RateLimitError, 429)], "b": ["ok"]})
    assert rot.call(fn) == "ok"
    assert calls == ["a", "b"] and clock.sleeps == []


def test_server_error_also_triggers_failover():
    rot, fn, calls, _ = _rotator(["a", "b"], {"a": [_err(openai.InternalServerError, 503)], "b": ["ok"]})
    assert rot.call(fn) == "ok" and calls == ["a", "b"]


def test_single_key_backs_off_honouring_retry_after():
    rot, fn, calls, clock = _rotator(
        ["a"], {"a": [_err(openai.RateLimitError, 429, {"retry-after": "7"}), _err(openai.RateLimitError, 429), "ok"]})
    assert rot.call(fn) == "ok"
    assert clock.sleeps == [7.0, 2.0]  # retry-after first, then exponential 2**1


def test_all_keys_cooling_waits_for_the_one_that_frees_first():
    rot, fn, calls, clock = _rotator(
        ["a", "b"], {"a": [_err(openai.RateLimitError, 429, {"retry-after": "3"}), "ok"],
                     "b": [_err(openai.RateLimitError, 429, {"retry-after": "20"})]})
    assert rot.call(fn) == "ok"
    assert calls == ["a", "b", "a"]
    assert clock.sleeps == [3.0]  # a cools 3s, b cools 20s -> wait 3s for a, not 20s


def test_gives_up_and_reraises_after_max_retries():
    rot, fn, calls, _ = _rotator(["a", "b"], {"a": [_err(openai.RateLimitError, 429)] * 3,
                                              "b": [_err(openai.RateLimitError, 429)] * 3}, max_retries=2)
    with pytest.raises(openai.RateLimitError):
        rot.call(fn)
    assert len(calls) == 3


def test_non_retryable_error_is_not_retried():
    rot, fn, calls, _ = _rotator(["a", "b"], {"a": [_err(openai.AuthenticationError, 401)], "b": ["ok"]})
    with pytest.raises(openai.AuthenticationError):
        rot.call(fn)
    assert calls == ["a"]


def test_rotator_requires_at_least_one_key():
    with pytest.raises(ValueError):
        GroqKeyRotator([], client_factory=lambda k: k)


def test_sqlite_cache_round_trip_persists_across_instances(tmp_path):
    db = tmp_path / "c.db"
    first = SQLiteResponseCache(db)
    assert first.get("k") is None and first.misses == 1
    first.set("k", {"role": "assistant", "content": "hi"})
    second = SQLiteResponseCache(db)  # a "restart"
    assert second.get("k") == {"role": "assistant", "content": "hi"} and second.hits == 1


def _resp(content):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class CountingClient:
    def __init__(self, outcomes):
        self.outcomes, self.n = list(outcomes), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **req):
        self.n += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_llm_uses_sqlite_cache_for_repeated_requests(tmp_path):
    cfg = LLMConfig(base_url="x", api_key="k", model="m", requests_per_minute=10_000, cache_db=tmp_path / "c.db")
    client = CountingClient([_resp("hello")])
    llm = OpenAICompatibleLLM(cfg, client=client, sleep=lambda s: None)
    msgs = [{"role": "user", "content": "hi"}]
    assert llm.complete(msgs, [])["content"] == "hello"
    assert llm.complete(msgs, [])["content"] == "hello"
    assert client.n == 1 and llm.usage["cache_hits"] == 1


def test_llm_fails_over_between_configured_keys():
    cfg = LLMConfig(base_url="x", api_key="", model="m", requests_per_minute=10_000, api_keys=["k1", "k2"])
    clients = {"k1": CountingClient([_err(openai.RateLimitError, 429)]), "k2": CountingClient([_resp("from k2")])}
    sleeps = []
    llm = OpenAICompatibleLLM(cfg, sleep=sleeps.append)
    llm.rotator.client_factory = clients.__getitem__  # no network: one fake client per key
    assert llm.complete([{"role": "user", "content": "hi"}], [])["content"] == "from k2"
    assert all(s < 1 for s in sleeps)  # only the tiny client-side throttle gap, no backoff


def test_llm_without_any_key_raises_clear_error():
    with pytest.raises(LLMError, match="GROQ_API_KEYS"):
        OpenAICompatibleLLM(LLMConfig(base_url="x", api_key="", model="m"))
