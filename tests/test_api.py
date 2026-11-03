"""API endpoints with FastAPI TestClient. The model, agent and DB are replaced via dependency injection."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from reorderpilot.api import main
from reorderpilot.api.schemas import (
    ChatResponse,
    CustomerDue,
    CustomersDueResult,
    PredictionResult,
    Recommendation,
    ToolCallRecord,
)

AS_OF = date(2011, 11, 1)


def fake_predictor(customer_id: str, top_k: int, as_of: date) -> PredictionResult:
    """Stub of model.predict.predict_reorder: customer 'new' is cold-start, others get 3 fixed products."""
    if customer_id == "new":
        return PredictionResult(customer_id=customer_id, as_of=as_of, abstain=True, why="not enough history")
    recs = [
        Recommendation(stock_code=code, description=f"ITEM {code}", probability=p, reason_text="bought 6 times")
        for code, p in [("85099B", 0.8), ("22423", 0.6), ("47566", 0.4)]
    ]
    return PredictionResult(customer_id=customer_id, as_of=as_of, recommendations=recs[:top_k])


@pytest.fixture
def client(tmp_path):
    main.app.dependency_overrides = {
        main.get_as_of: lambda: AS_OF,
        main.get_db_path: lambda: tmp_path / "memory.db",
        main.get_predictor: lambda: fake_predictor,
        main.get_due_lister: lambda: lambda limit, as_of, db: CustomersDueResult(
            as_of=as_of, customers=[CustomerDue(customer_id="12345", expected_value=99.5, n_candidates=3)][:limit]
        ),
        main.get_chat_handler: lambda: lambda req: ChatResponse(
            answer=f"echo {req.message}", tool_calls=[ToolCallRecord(name="read_memory", args={}, result_summary="")]
        ),
    }
    yield TestClient(main.app)
    main.app.dependency_overrides = {}


def test_health_reports_simulated_date(client):
    assert client.get("/health").json() == {"status": "ok", "as_of": "2011-11-01"}


def test_feedback_then_predict_shows_suppression(client):
    body = {"customer_id": "12345", "outcome": "declined", "stock_code": "85099B", "reason": "overstocked"}
    resp = client.post("/feedback", json=body)
    assert resp.status_code == 201
    assert resp.json()["effect"] == "85099B suppressed until 2011-12-01"

    recs = client.get("/predict/12345", params={"top_k": 2}).json()["recommendations"]
    assert [(r["stock_code"], r["suppressed"]) for r in recs] == [("85099B", True), ("22423", False)]
    assert "overstocked" in recs[0]["suppressed_reason"]


def test_price_feedback_sets_price_sensitive_in_predict_and_memory(client):
    client.post("/feedback", json={"customer_id": "12345", "outcome": "declined", "stock_code": "22423",
                                   "reason": "price", "note": "too expensive"})
    pred = client.get("/predict/12345").json()
    assert pred["price_sensitive"] is True
    assert not any(r["suppressed"] for r in pred["recommendations"])
    mem = client.get("/memory/12345").json()
    assert mem["price_sensitive"] is True
    assert mem["feedback_events"][0]["note"] == "too expensive"


def test_predict_passes_through_abstention(client):
    assert client.get("/predict/new").json()["abstain"] is True


def test_duplicate_feedback_post_stores_one_event(client):
    body = {"customer_id": "12345", "outcome": "declined", "stock_code": "85099B", "reason": "seasonal"}
    first, second = client.post("/feedback", json=body).json(), client.post("/feedback", json=body).json()
    assert first["event"]["id"] == second["event"]["id"]
    assert len(client.get("/memory/12345").json()["feedback_events"]) == 1


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("post", "/feedback", {"customer_id": "1", "outcome": "maybe"}),
        ("post", "/feedback", {"customer_id": "1", "outcome": "declined", "reason": "too_busy"}),
        ("post", "/feedback", {"customer_id": "1", "outcome": "ordered", "note": "x" * 501}),
        ("post", "/feedback", {"outcome": "ordered"}),
        ("get", "/predict/12345?top_k=0", None),
        ("get", "/predict/12345?top_k=abc", None),
        ("get", "/customers/due?limit=51", None),
        ("get", "/memory/12;DROP", None),
        ("post", "/agent/chat", {"session_id": "s1", "message": ""}),
        ("post", "/agent/chat", {"message": "hi"}),
    ],
)
def test_validation_errors_return_422(client, method, url, body):
    resp = client.request(method.upper(), url, json=body)
    assert resp.status_code == 422, resp.text


def test_customers_due_and_chat_use_injected_components(client):
    assert client.get("/customers/due?limit=1").json()["customers"][0]["customer_id"] == "12345"
    chat = client.post("/agent/chat", json={"session_id": "s1", "message": "hi"}).json()
    assert chat["answer"] == "echo hi" and chat["tool_calls"][0]["name"] == "read_memory"


def test_get_chat_handler_is_wired_to_agent_loop(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(main, "make_chat_handler", lambda: sentinel)   # no real LLM client is built
    assert main.get_chat_handler() is sentinel


def test_chat_endpoint_runs_real_agent_loop_with_fake_llm(client, tmp_path):
    from reorderpilot.agent.loop import make_chat_handler
    from reorderpilot.agent.tools import ToolContext

    class OneAnswerLLM:
        def complete(self, messages, tools):
            return {"role": "assistant", "content": f"loop saw: {messages[-1]['content']}"}

    ctx = ToolContext(as_of=AS_OF, db_path=tmp_path / "memory.db")
    main.app.dependency_overrides[main.get_chat_handler] = lambda: make_chat_handler(OneAnswerLLM(), ctx)
    resp = client.post("/agent/chat", json={"session_id": "s1", "message": "hi"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "loop saw: hi", "tool_calls": [], "trace_url": None}
