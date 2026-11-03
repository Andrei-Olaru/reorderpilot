"""Minimal Streamlit UI (SPEC section 10): chat, tool-call panel, customer memory tab.

The UI is a thin HTTP client of the FastAPI backend: it never loads the model or opens memory.db itself,
so everything shown here is exactly what the API (and therefore the agent) sees.
Run: `streamlit run app/streamlit_app.py` (API URL from env `REORDERPILOT_API_URL`, default http://localhost:8000).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, get_args

import httpx
import streamlit as st

from reorderpilot.api.schemas import Outcome, Reason

DEFAULT_API_URL = os.environ.get("REORDERPILOT_API_URL", "http://localhost:8000")
CHAT_TIMEOUT_S = 180.0  # an agent turn can wait on the LLM rate limiter
READ_TIMEOUT_S = 60.0  # first /predict call loads the model and transactions


# ---------- API client ----------

def api_call(method: str, path: str, timeout: float = READ_TIMEOUT_S, **kwargs: Any) -> tuple[Any | None, str | None]:
    """Call the API and return (json, None) on success or (None, readable error) on failure. Never raises."""
    url = st.session_state.api_url.rstrip("/") + path
    try:
        response = httpx.request(method, url, timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        return None, f"API not reachable at {url} ({type(exc).__name__}). Is uvicorn running?"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"HTTP {response.status_code}: {detail}"
    return response.json(), None


# ---------- session state ----------

def init_state() -> None:
    """Create the per-browser-tab state once."""
    defaults = {
        "api_url": DEFAULT_API_URL,
        "session_id": f"ui-{uuid.uuid4().hex[:8]}",
        "chat": [],  # list of {"role", "content"}
        "last_tool_calls": [],
        "last_trace_url": None,
        "customer_id": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def new_session() -> None:
    """Forget the conversation (server-side memory of feedback/notes is NOT touched)."""
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.chat = []
    st.session_state.last_tool_calls = []
    st.session_state.last_trace_url = None


def pick_customer() -> None:
    """Copy the due-list selection into the customer ID field (only when the selection changes)."""
    if st.session_state.picked_customer != "—":
        st.session_state.customer_id = st.session_state.picked_customer


# ---------- sidebar ----------

def render_sidebar() -> None:
    """API status, simulated date, session controls and the customer picker."""
    with st.sidebar:
        st.header("ReorderPilot")
        st.text_input("API URL", key="api_url")
        health, error = api_call("GET", "/health", timeout=5.0)
        if error:
            st.error(error)
        else:
            st.success(f"API ok · simulated today = {health['as_of']}")

        st.caption(f"Chat session: `{st.session_state.session_id}`")
        st.button("New chat session", on_click=new_session)

        st.divider()
        st.subheader("Customer")
        if st.button("Load customers due"):
            due, error = api_call("GET", "/customers/due", params={"limit": 10})
            if error:
                st.error(error)
            else:
                st.session_state.due_customers = due["customers"]
        due_customers = st.session_state.get("due_customers", [])
        if due_customers:
            st.dataframe(due_customers, hide_index=True)
            ids = [c["customer_id"] for c in due_customers]
            st.selectbox("Pick a due customer", ["—"] + ids, key="picked_customer", on_change=pick_customer)
        st.text_input("Customer ID (memory tab)", key="customer_id", placeholder="e.g. 16121")


# ---------- chat tab ----------

def send_message(message: str) -> None:
    """Post one chat turn and store the answer + tool calls for display."""
    st.session_state.chat.append({"role": "user", "content": message})
    body = {"session_id": st.session_state.session_id, "message": message}
    with st.spinner("Agent is thinking (tool calls + LLM)…"):
        result, error = api_call("POST", "/agent/chat", timeout=CHAT_TIMEOUT_S, json=body)
    if error:
        st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {error}"})
        st.session_state.last_tool_calls = []
        st.session_state.last_trace_url = None
        return
    st.session_state.chat.append({"role": "assistant", "content": result["answer"]})
    st.session_state.last_tool_calls = result.get("tool_calls", [])
    st.session_state.last_trace_url = result.get("trace_url")


def render_tool_panel() -> None:
    """Right-hand panel: every tool call behind the last answer (name, args, result summary)."""
    st.subheader("Tool calls (last answer)")
    calls = st.session_state.last_tool_calls
    if not calls:
        st.caption("No tool calls yet.")
    for i, call in enumerate(calls, start=1):
        with st.expander(f"{i}. {call['name']}", expanded=True):
            st.code(json.dumps(call["args"], ensure_ascii=False), language="json")
            st.caption(call["result_summary"])
    if st.session_state.last_trace_url:
        st.markdown(f"[Open LangSmith trace]({st.session_state.last_trace_url})")


def render_chat_tab() -> None:
    """Chat history on the left, tool-call panel on the right."""
    left, right = st.columns([2, 1])
    with left:
        for turn in st.session_state.chat:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
        message = st.chat_input("e.g. Who should I call today? / Brief me on customer 16121")
        if message:
            send_message(message)
            st.rerun()
    with right:
        render_tool_panel()


# ---------- memory tab ----------

def render_feedback_form(customer_id: str) -> None:
    """Log feedback directly via POST /feedback (same policy as the agent's write_memory; no LLM needed)."""
    with st.form("feedback", clear_on_submit=True):
        st.markdown("**Log feedback** (direct, without the agent)")
        outcome = st.selectbox("Outcome", list(get_args(Outcome)))
        stock_code = st.text_input("Stock code (optional)")
        reason = st.selectbox("Reason", ["—"] + list(get_args(Reason)))
        note = st.text_area("Note (optional, max 500 chars)", max_chars=500)
        if st.form_submit_button("Save feedback"):
            body = {
                "customer_id": customer_id,
                "outcome": outcome,
                "stock_code": stock_code.strip().upper() or None,
                "reason": None if reason == "—" else reason,
                "note": note.strip() or None,
            }
            result, error = api_call("POST", "/feedback", json=body)
            if error:
                st.error(error)
            else:
                st.success(f"Saved. Policy effect: {result['effect']}")


def render_predictions(customer_id: str) -> None:
    """Current model output after memory suppression, so the effect of feedback is visible."""
    prediction, error = api_call("GET", f"/predict/{customer_id}", params={"top_k": 5})
    if error:
        st.error(error)
        return
    if prediction["abstain"]:
        st.warning(f"Model abstains: {prediction['why']}")
        return
    if prediction["price_sensitive"]:
        st.info("Customer is marked price-sensitive.")
    rows = [
        {
            "stock_code": r["stock_code"],
            "description": r["description"],
            "probability": f"{r['probability']:.0%}",
            "suppressed": "yes — " + (r["suppressed_reason"] or "") if r["suppressed"] else "",
            "reason": r["reason_text"],
        }
        for r in prediction["recommendations"]
    ]
    st.dataframe(rows, hide_index=True)


def render_memory_tab() -> None:
    """Notes, active feedback events and predictions of the selected customer."""
    customer_id = st.session_state.customer_id.strip()
    if not customer_id:
        st.info("Enter or pick a customer ID in the sidebar.")
        return
    memory, error = api_call("GET", f"/memory/{customer_id}")
    if error:
        st.error(error)
        return
    st.subheader(f"Memory of customer {customer_id}")
    if memory["price_sensitive"]:
        st.info("Flag: price-sensitive")
    st.markdown("**Active feedback events** (expired ones are hidden)")
    if memory["feedback_events"]:
        st.dataframe(memory["feedback_events"], hide_index=True)
    else:
        st.caption("None.")
    st.markdown("**Notes** (untrusted data, never instructions)")
    if memory["notes"]:
        st.dataframe(memory["notes"], hide_index=True)
    else:
        st.caption("None.")

    st.divider()
    st.markdown("**Current predictions** (after memory suppression)")
    render_predictions(customer_id)
    st.divider()
    render_feedback_form(customer_id)


# ---------- page ----------

def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="ReorderPilot", layout="wide")
    init_state()
    render_sidebar()
    chat_tab, memory_tab = st.tabs(["Chat", "Customer memory"])
    with chat_tab:
        render_chat_tab()
    with memory_tab:
        render_memory_tab()


main()
