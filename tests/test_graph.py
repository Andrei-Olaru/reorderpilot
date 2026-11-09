"""v2 LangGraph orchestrator + ChromaDB vector memory, with a scripted fake LLM and an offline fake embedding."""

import hashlib
import json
import re

import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from reorderpilot.agent.graph import (
    NO_CUSTOMER_ANSWER,
    fallback_cards,
    parse_cards,
    parse_transcript,
    run_graph,
)
from reorderpilot.agent.llm import LLMError
from reorderpilot.agent.tools import ToolContext
from reorderpilot.api.schemas import FeedbackEvent
from reorderpilot.memory import store
from reorderpilot.memory.vector_store import VectorMemory
from tests.test_agent_loop import AS_OF, FakeBundle, ScriptedLLM, answer, call, make_tx, tool_reply

LIVE = "[SALES_REP] Hi, anything for the next order?\n[CLIENT] Delivery was late again and your prices went up."
DIM = 64


class WordHashEmbedding(EmbeddingFunction[Documents]):
    """Offline stand-in for the ONNX model: bag of words hashed into 64 dims (shared words -> close vectors)."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (name required by Chroma)
        out = []
        for text in input:
            v = np.zeros(DIM, dtype=np.float32)
            for w in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out

    @staticmethod
    def name() -> str:
        return "word-hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordHashEmbedding":
        return WordHashEmbedding()


@pytest.fixture
def ctx(tmp_path):
    """Same tiny dataset + fake model as the v1 loop tests (customer A buys P1 regularly)."""
    tx, bundle = make_tx(), FakeBundle()
    return ToolContext(as_of=AS_OF, db_path=tmp_path / "memory.db", tx_loader=lambda: tx,
                       bundle_loader=lambda: bundle)


@pytest.fixture
def memory(tmp_path):
    return VectorMemory(tmp_path / "chroma", embedding_function=WordHashEmbedding())


def cards_reply(*cards):
    return answer(json.dumps({"cards": [{"title": t, "say": s, "evidence": "data"} for t, s in cards]}))


# ---------- supervisor / routing ----------

def test_parse_transcript_detects_speaker_tags_only():
    assert parse_transcript(LIVE) == ["[SALES_REP] Hi, anything for the next order?",
                                      "[CLIENT] Delivery was late again and your prices went up."]
    assert parse_transcript("Brief me on customer A") == []          # plain chat message


def test_chat_message_routes_through_v1_loop(ctx, memory):
    llm = ScriptedLLM([tool_reply(call("predict_reorder", {"customer_id": "A"})), answer("A should get P1.")])
    state = run_graph("Brief me on customer A", llm, ctx, memory)
    assert state["mode"] == "chat" and state["trail"] == ["supervisor", "chat_agent", "grounding_guard"]
    assert state["answer"] == "A should get P1." and state["stop_reason"] == "answer"


def test_live_call_runs_all_specialists_in_order(ctx, memory):
    llm = ScriptedLLM([cards_reply(("Offer P1", "P1 is due now."))])
    state = run_graph(LIVE, llm, ctx, memory, customer_id="A")
    assert state["trail"] == ["supervisor", "ml_prediction", "vector_memory", "live_copilot", "grounding_guard"]
    assert state["cards"] == [{"title": "Offer P1", "say": "P1 is due now.", "evidence": "data"}]
    sent = llm.seen[0][1]["content"]                                   # the copilot saw prediction + transcript
    assert '"stock_code":"P1"' in sent and "[CLIENT] Delivery was late" in sent
    assert "Delivery was late" not in llm.seen[0][0]["content"]       # untrusted text never in the system prompt


def test_live_call_without_customer_stops_politely(ctx, memory):
    llm = ScriptedLLM([])                                               # the LLM must never be called
    state = run_graph(LIVE, llm, ctx, memory)
    assert state["trail"] == ["supervisor", "grounding_guard"] and state["answer"] == NO_CUSTOMER_ANSWER


# ---------- copilot robustness + grounding ----------

def test_grounding_guard_drops_card_with_invented_code(ctx, memory):
    llm = ScriptedLLM([cards_reply(("Offer P1", "Due now."), ("Upsell", "Try 99999Z, it is new."))])
    state = run_graph(LIVE, llm, ctx, memory, customer_id="A")
    assert [c["title"] for c in state["cards"]] == ["Offer P1"] and state["removed_codes"] == ["99999Z"]


def test_llm_down_falls_back_to_model_cards(ctx, memory):
    state = run_graph(LIVE, ScriptedLLM([LLMError("429")]), ctx, memory, customer_id="A")
    assert state["stop_reason"] == "fallback_llm_error"
    assert state["cards"][0]["title"].startswith("Offer LUNCH BAG RED RETROSPOT (P1)")


def test_bad_json_falls_back_and_parse_cards_tolerates_fences(ctx, memory):
    assert parse_cards('```json\n{"cards": [{"title": "t", "say": "s"}]}\n```')[0]["title"] == "t"
    assert parse_cards("Sure! Here are some ideas...") is None
    state = run_graph(LIVE, ScriptedLLM([answer("not json")]), ctx, memory, customer_id="A")
    assert state["stop_reason"] == "fallback_bad_json" and state["cards"]


def test_fallback_never_offers_suppressed_or_abstained_products():
    pred = {"abstain": False, "recommendations": [
        {"stock_code": "P1", "description": "X", "probability": 0.9, "reason_text": "r", "suppressed": True},
        {"stock_code": "P2", "description": "Y", "probability": 0.5, "reason_text": "r"}]}
    assert [c["title"] for c in fallback_cards(pred, [])] == ["Offer Y (P2)"]
    abstained = fallback_cards({"abstain": True, "why": "not enough history"}, [])
    assert abstained[0]["title"] == "No product suggestion" and "not enough history" in abstained[0]["say"]


# ---------- vector memory ----------

def test_vector_memory_finds_relevant_note_for_this_customer_only(ctx, memory):
    store.add_customer_note(ctx.db_path, "A", "Complained that delivery was late twice in October.", AS_OF)
    store.add_customer_note(ctx.db_path, "A", "Owner likes the red retrospot range.", AS_OF)
    store.add_customer_note(ctx.db_path, "B", "Delivery late, very angry about delivery.", AS_OF)
    memory.sync_customer(ctx.db_path, "A", AS_OF)
    memory.sync_customer(ctx.db_path, "B", AS_OF)
    hits = memory.search("A", "your delivery was late again", AS_OF, k=2)
    assert hits[0].text.startswith("Complained that delivery was late")
    assert all("angry" not in h.text for h in hits)                    # customer B's note never leaks to A


def test_vector_memory_respects_as_of_and_sync_is_idempotent(ctx, memory):
    from datetime import date
    store.add_customer_note(ctx.db_path, "A", "Asked for a discount on delivery.", AS_OF)
    store.record_feedback(ctx.db_path, FeedbackEvent(customer_id="A", outcome="declined", stock_code="P2",
                                                     reason="price", note="too expensive this month"), AS_OF)
    assert memory.sync_customer(ctx.db_path, "A", AS_OF) == 2
    memory.sync_customer(ctx.db_path, "A", AS_OF)
    assert memory.count() == 2                                          # same ids -> upsert, no duplicates
    assert memory.search("A", "discount delivery", date(2011, 2, 28)) == []   # written later than this as_of
    texts = [h.text for h in memory.search("A", "too expensive price", AS_OF)]
    assert any("declined P2 (price): too expensive" in t for t in texts)


def test_live_call_passes_retrieved_memory_to_copilot(ctx, memory):
    store.add_customer_note(ctx.db_path, "A", "Delivery was late in October; offered free shipping.", AS_OF)
    llm = ScriptedLLM([cards_reply(("Delivery", "Mention the free shipping we offered."))])
    state = run_graph(LIVE, llm, ctx, memory, customer_id="A")
    assert state["memories"][0]["text"].startswith("Delivery was late in October")
    assert "free shipping" in llm.seen[0][1]["content"]
