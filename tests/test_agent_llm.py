"""LLM client with a stub transport: 429 backoff honours retry-after, cache, malformed tool call mapping."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from langsmith import tracing_context

from reorderpilot.agent.llm import LLMConfig, LLMError, MalformedToolCallError, OpenAICompatibleLLM


@pytest.fixture(autouse=True)
def no_tracing():
    """Tests must never send traces to LangSmith, whatever .env says."""
    with tracing_context(enabled=False):
        yield


def _resp(content="hi"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def _http_error(cls, status, text="err", headers=None):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://x"))
    return cls(text, response=response, body=None)


class StubClient:
    def __init__(self, outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **req):
        self.requests.append(req)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make_llm(outcomes, tmp_path=None, max_retries=5):
    cfg = LLMConfig(base_url="x", api_key="k", model="m", requests_per_minute=10_000, max_retries=max_retries,
                    cache_dir=tmp_path)
    sleeps = []
    return OpenAICompatibleLLM(cfg, client=StubClient(outcomes), sleep=sleeps.append), sleeps


def test_rate_limit_backoff_honours_retry_after():
    llm, sleeps = make_llm([_http_error(openai.RateLimitError, 429, headers={"retry-after": "7"}), _resp("ok")])
    assert llm.complete([{"role": "user", "content": "q"}], [])["content"] == "ok"
    assert 7.0 in sleeps and llm.client.requests[0]["temperature"] == 0.0


def test_gives_up_after_max_retries():
    llm, _ = make_llm([_http_error(openai.RateLimitError, 429)] * 3, max_retries=2)
    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        llm.complete([{"role": "user", "content": "q"}], [])


def test_cache_returns_saved_reply_without_calling_the_api(tmp_path):
    llm, _ = make_llm([_resp("first")], tmp_path)
    msgs = [{"role": "user", "content": "q"}]
    assert llm.complete(msgs, [])["content"] == "first"
    assert llm.complete(msgs, [])["content"] == "first"       # second call would fail: no outcomes left
    assert llm.usage["cache_hits"] == 1 and len(llm.client.requests) == 1


def test_provider_tool_use_failed_maps_to_malformed_error():
    llm, _ = make_llm([_http_error(openai.BadRequestError, 400, "Error code: 400 - tool_use_failed")])
    with pytest.raises(MalformedToolCallError):
        llm.complete([{"role": "user", "content": "q"}], [])


def test_corrupt_cache_file_is_ignored_and_refetched(tmp_path):
    llm, _ = make_llm([_resp("first"), _resp("fresh")], tmp_path)
    msgs = [{"role": "user", "content": "q"}]
    llm.complete(msgs, [])
    for f in tmp_path.rglob("*.json"):
        f.write_text("{not json", encoding="utf-8")
    assert llm.complete(msgs, [])["content"] == "fresh"
    assert llm.usage["cache_hits"] == 0
