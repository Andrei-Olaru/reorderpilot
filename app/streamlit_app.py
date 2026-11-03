"""Streamlit UI (SPEC section 10): chat with per-answer tool executions, KPI row, customer memory dashboard.

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
NONE_OPTION = "—"
CHAT_HEIGHT_PX = 520
QUICK_PROMPTS = [
    "Who should I call today?",
    "Brief me on my top due customer",
    "Which products are suppressed for customer 16121?",
]
AVATARS = {"user": ":material/person:", "assistant": ":material/support_agent:"}

# Lightweight styling on top of Streamlit's theme. Colours use translucent greys so light and dark mode both work.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Text only: icons are font ligatures, so overriding their font turns them into words like "double_arrow_right". */
html, body, .stApp, p, li, label, input, textarea, h1, h2, h3, h4, h5, h6,
.rp-header, .rp-label { font-family: 'Inter', sans-serif; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; font-weight: 600; }

.rp-header { display: flex; align-items: baseline; gap: .75rem; margin-bottom: .25rem; }
.rp-header .rp-title { font-size: 1.6rem; font-weight: 700; }
.rp-header .rp-subtitle { font-size: .95rem; opacity: .65; }

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22); border-radius: 10px;
    padding: .85rem 1rem; background: rgba(128,128,128,.04);
}
[data-testid="stMetricLabel"] p { font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
[data-testid="stExpander"] details { border-radius: 8px; border-color: rgba(128,128,128,.22); }
[data-testid="stExpander"] summary p { font-size: .85rem; font-weight: 500; }
[data-testid="stChatMessage"] { border-radius: 10px; padding: .75rem 1rem; }

.stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid rgba(128,128,128,.22); }
.stTabs [data-baseweb="tab"] { padding: .5rem 1rem; font-weight: 500; }

[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
.rp-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; margin: .5rem 0 .25rem; }
</style>
"""


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
        "chat": [],  # list of {"role", "content", "tool_calls", "trace_url"}
        "customer_id": "",
        "due_customers": [],
        "flash": None,  # one-shot success message shown after a rerun
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def new_session() -> None:
    """Forget the conversation (server-side memory of feedback/notes is NOT touched)."""
    st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.chat = []


def pick_customer() -> None:
    """Copy the due-list selection into the customer ID field (only when the selection changes)."""
    if st.session_state.picked_customer != NONE_OPTION:
        st.session_state.customer_id = st.session_state.picked_customer


def load_due_customers() -> None:
    """Fetch the ranked call list into session state."""
    due, error = api_call("GET", "/customers/due", params={"limit": 10})
    if error:
        st.session_state.flash = ("error", error)
    else:
        st.session_state.due_customers = due["customers"]


# ---------- data loading (once per rerun) ----------

def load_customer(customer_id: str) -> dict[str, Any]:
    """Fetch memory and predictions of one customer; errors are returned, not raised."""
    if not customer_id:
        return {}
    memory, memory_error = api_call("GET", f"/memory/{customer_id}")
    prediction, prediction_error = api_call("GET", f"/predict/{customer_id}", params={"top_k": 5})
    return {"memory": memory, "memory_error": memory_error,
            "prediction": prediction, "prediction_error": prediction_error}


def suppressed_count(customer: dict[str, Any]) -> int | None:
    """Number of suppressed recommendations for the loaded customer, or None when unknown."""
    prediction = customer.get("prediction")
    if not prediction or prediction["abstain"]:
        return None
    return sum(1 for r in prediction["recommendations"] if r["suppressed"])


# ---------- layout: header, sidebar, KPI row ----------

def render_header() -> None:
    """Product title line."""
    st.markdown(
        '<div class="rp-header"><span class="rp-title">ReorderPilot</span>'
        '<span class="rp-subtitle">B2B re-order intelligence and sales assistant</span></div>',
        unsafe_allow_html=True,
    )


def render_sidebar(health: dict[str, Any] | None, health_error: str | None) -> None:
    """Connection status, session controls and the customer picker."""
    with st.sidebar:
        st.markdown("### :material/hub: ReorderPilot")
        st.markdown('<div class="rp-label">Connection</div>', unsafe_allow_html=True)
        st.text_input("API URL", key="api_url", label_visibility="collapsed")
        if health_error:
            st.error(health_error, icon=":material/cloud_off:")
        else:
            st.success("API connected", icon=":material/cloud_done:")

        st.markdown('<div class="rp-label">Chat session</div>', unsafe_allow_html=True)
        st.code(st.session_state.session_id, language=None)
        st.button("New session", on_click=new_session, icon=":material/restart_alt:", width="stretch")

        st.divider()
        st.markdown('<div class="rp-label">Customer</div>', unsafe_allow_html=True)
        st.button("Load customers due", on_click=load_due_customers, icon=":material/call:",
                  width="stretch")
        due = st.session_state.due_customers
        if due:
            st.dataframe(
                due, hide_index=True, width="stretch",
                column_config={
                    "customer_id": st.column_config.TextColumn("Customer"),
                    "expected_value": st.column_config.NumberColumn("Exp. value", format="£%.0f"),
                    "n_candidates": st.column_config.NumberColumn("Items"),
                },
            )
            ids = [c["customer_id"] for c in due]
            st.selectbox("Pick a due customer", [NONE_OPTION] + ids, key="picked_customer", on_change=pick_customer)
        st.text_input("Customer ID", key="customer_id", placeholder="e.g. 16121")


def render_kpis(health: dict[str, Any] | None, customer: dict[str, Any]) -> None:
    """Four metric cards: session, simulated date, suppressed items of the customer, customers due."""
    suppressed = suppressed_count(customer)
    customer_id = st.session_state.customer_id.strip()
    cols = st.columns(4)
    cols[0].metric("Active session", st.session_state.session_id,
                   help="Chat history is kept per session; customer memory is shared.")
    cols[1].metric("As-of date", health["as_of"] if health else "Unavailable",
                   help="Simulated 'today': features and memory expiry use this date.")
    cols[2].metric("Suppressed items", NONE_OPTION if suppressed is None else suppressed,
                   help=f"Recommendations hidden by feedback for customer {customer_id}" if customer_id
                   else "Select a customer to see suppressed recommendations.")
    cols[3].metric("Customers due", len(st.session_state.due_customers) or NONE_OPTION,
                   help="Loaded from the sidebar (top 10 by expected value).")


# ---------- chat tab ----------

def send_message(message: str) -> None:
    """Post one chat turn and store the answer together with its tool calls."""
    st.session_state.chat.append({"role": "user", "content": message})
    body = {"session_id": st.session_state.session_id, "message": message}
    with st.spinner("Agent is working (tool calls + LLM)…"):
        result, error = api_call("POST", "/agent/chat", timeout=CHAT_TIMEOUT_S, json=body)
    if error:
        st.session_state.chat.append({"role": "assistant", "content": f"**Request failed.** {error}",
                                      "tool_calls": [], "trace_url": None})
        return
    st.session_state.chat.append({"role": "assistant", "content": result["answer"],
                                  "tool_calls": result.get("tool_calls", []), "trace_url": result.get("trace_url")})


def render_tool_executions(turn: dict[str, Any]) -> None:
    """Collapsed accordion with every tool call behind one answer (name, args as JSON, result summary)."""
    calls = turn.get("tool_calls") or []
    if not calls:
        return
    with st.expander(f"Tool executions ({len(calls)})", icon=":material/build:"):
        for i, call in enumerate(calls, start=1):
            st.markdown(f"**{i}. `{call['name']}`**")
            st.caption(call["result_summary"])
            st.code(json.dumps(call["args"], ensure_ascii=False, indent=2), language="json")
        if turn.get("trace_url"):
            st.link_button("Open LangSmith trace", turn["trace_url"], icon=":material/open_in_new:")


def render_last_run_panel() -> None:
    """Right-hand panel (SPEC section 10): the tool calls of the last answer, plus quick prompts."""
    last = next((t for t in reversed(st.session_state.chat) if t["role"] == "assistant"), None)
    with st.container(border=True):
        st.markdown("##### :material/analytics: Last answer")
        calls = (last or {}).get("tool_calls") or []
        if not calls:
            st.caption("No tool calls yet.")
        for i, call in enumerate(calls, start=1):
            st.markdown(f"{i}. `{call['name']}`  \n<small>{call['result_summary']}</small>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("##### :material/bolt: Quick prompts")
        for i, prompt in enumerate(QUICK_PROMPTS):
            if st.button(prompt, key=f"quick_{i}", width="stretch"):
                send_message(prompt)
                st.rerun()


def render_chat_tab() -> None:
    """Chat history (with tool accordions) on the left, last-run panel on the right."""
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        # Fixed-height scroll area keeps the input right under the history, whatever the right panel's height.
        with st.container(height=CHAT_HEIGHT_PX, border=True):
            if not st.session_state.chat:
                st.caption("Ask about due customers, a customer's briefing, or log feedback in plain language.")
            for turn in st.session_state.chat:
                with st.chat_message(turn["role"], avatar=AVATARS[turn["role"]]):
                    st.markdown(turn["content"])
                    if turn["role"] == "assistant":
                        render_tool_executions(turn)
        message = st.chat_input("Ask the sales agent, e.g. Brief me on customer 16121")
    with right:
        render_last_run_panel()
    if message:
        send_message(message)
        st.rerun()


# ---------- memory tab ----------

def render_briefing(customer_id: str, customer: dict[str, Any]) -> None:
    """Customer flags, current predictions after memory suppression, and saved notes."""
    memory, prediction = customer.get("memory"), customer.get("prediction")
    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown(f"#### Customer {customer_id}")
        if (memory and memory["price_sensitive"]) or (prediction and prediction["price_sensitive"]):
            with top[1]:
                st.badge("Price-sensitive", icon=":material/sell:", color="orange")

        st.markdown('<div class="rp-label">Recommendations (after memory suppression)</div>', unsafe_allow_html=True)
        if customer.get("prediction_error"):
            st.error(customer["prediction_error"])
        elif prediction["abstain"]:
            st.warning(f"Model abstains: {prediction['why']}", icon=":material/block:")
        else:
            rows = [
                {
                    "stock_code": r["stock_code"],
                    "description": r["description"],
                    "probability": r["probability"],
                    "status": "Suppressed" if r["suppressed"] else "Active",
                    "suppression_reason": r["suppressed_reason"] or "",
                    "reason": r["reason_text"],
                }
                for r in prediction["recommendations"]
            ]
            st.dataframe(
                rows, hide_index=True, width="stretch",
                column_config={
                    "stock_code": st.column_config.TextColumn("Stock code"),
                    "description": st.column_config.TextColumn("Description", width="large"),
                    "probability": st.column_config.ProgressColumn("Probability", format="percent",
                                                                   min_value=0.0, max_value=1.0),
                    "status": st.column_config.TextColumn("Status"),
                    "suppression_reason": st.column_config.TextColumn("Suppression"),
                    "reason": st.column_config.TextColumn("Why recommended", width="large"),
                },
            )

    with st.container(border=True):
        st.markdown("##### :material/sticky_note_2: Notes")
        st.caption("Free-text notes are stored as untrusted data and never treated as instructions.")
        notes = (memory or {}).get("notes") or []
        if not notes:
            st.caption("No notes.")
        for note in notes:
            st.markdown(f"<small>{note['created_at']}</small>", unsafe_allow_html=True)
            st.text(note["note"])


def render_suppressions(memory: dict[str, Any]) -> None:
    """Active feedback events with an outcome filter (expired events are hidden by the API)."""
    events = memory["feedback_events"]
    counts = st.columns(3)
    counts[0].metric("Active events", len(events))
    counts[1].metric("Declined", sum(1 for e in events if e["outcome"] == "declined"))
    counts[2].metric("With expiry", sum(1 for e in events if e.get("expires_at")))
    if not events:
        st.caption("No active feedback events for this customer.")
        return
    selected = st.segmented_control("Filter by outcome", list(get_args(Outcome)), selection_mode="multi",
                                    key="event_filter")
    shown = [e for e in events if not selected or e["outcome"] in selected]
    columns = ["created_at", "outcome", "stock_code", "reason", "note", "expires_at"]
    st.dataframe(
        [{k: e.get(k) for k in columns} for e in shown], hide_index=True, width="stretch",
        column_config={
            "created_at": st.column_config.DateColumn("Logged"),
            "outcome": st.column_config.TextColumn("Outcome"),
            "stock_code": st.column_config.TextColumn("Stock code"),
            "reason": st.column_config.TextColumn("Reason"),
            "note": st.column_config.TextColumn("Note", width="large"),
            "expires_at": st.column_config.DateColumn("Expires"),
        },
    )


def render_feedback_form(customer_id: str) -> None:
    """Log feedback directly via POST /feedback (same policy as the agent's write_memory; no LLM needed)."""
    st.caption("Logs feedback directly through the API, bypassing the agent. The same memory policy applies.")
    with st.form("feedback", clear_on_submit=True, border=True):
        left, right = st.columns(2)
        outcome = left.selectbox("Outcome", list(get_args(Outcome)))
        reason = right.selectbox("Reason", [NONE_OPTION] + list(get_args(Reason)))
        stock_code = left.text_input("Stock code (optional)")
        right.text_input("Customer", value=customer_id, disabled=True)
        note = st.text_area("Note (optional)", max_chars=500)
        if st.form_submit_button("Save feedback", type="primary", icon=":material/save:"):
            body = {
                "customer_id": customer_id,
                "outcome": outcome,
                "stock_code": stock_code.strip().upper() or None,
                "reason": None if reason == NONE_OPTION else reason,
                "note": note.strip() or None,
            }
            result, error = api_call("POST", "/feedback", json=body)
            if error:
                st.error(error)
            else:
                # Rerun so the briefing and KPI cards show the new suppression immediately.
                st.session_state.flash = ("success", f"Feedback saved. Policy effect: {result['effect']}")
                st.rerun()


def render_memory_tab(customer: dict[str, Any]) -> None:
    """Customer memory dashboard: briefing, logged suppressions, manual override form."""
    customer_id = st.session_state.customer_id.strip()
    if not customer_id:
        st.info("Enter or pick a customer ID in the sidebar.", icon=":material/person_search:")
        return
    if customer.get("memory_error"):
        st.error(customer["memory_error"])
        return
    briefing, suppressions, override = st.tabs([
        ":material/description: Customer Briefings",
        ":material/block: Logged Suppressions",
        ":material/edit_note: Manual Override Form",
    ])
    with briefing:
        render_briefing(customer_id, customer)
    with suppressions:
        render_suppressions(customer["memory"])
    with override:
        render_feedback_form(customer_id)


# ---------- page ----------

def render_flash() -> None:
    """Show (and clear) a one-shot message set before the last rerun."""
    flash = st.session_state.flash
    if flash:
        kind, text = flash
        (st.success if kind == "success" else st.error)(text)
        st.session_state.flash = None


def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="ReorderPilot", page_icon=":material/hub:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    health, health_error = api_call("GET", "/health", timeout=5.0)
    render_sidebar(health, health_error)
    customer = load_customer(st.session_state.customer_id.strip())

    render_header()
    render_flash()
    render_kpis(health, customer)
    chat_tab, memory_tab = st.tabs([":material/forum: Agent Chat", ":material/database: Customer Memory"])
    with chat_tab:
        render_chat_tab()
    with memory_tab:
        render_memory_tab(customer)


main()
