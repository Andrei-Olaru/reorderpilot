"""Agent tools: suppression on/off, product resolution, validation, due list, tool specs."""

from reorderpilot.agent import tools
from reorderpilot.agent.tools import FindProductToolArgs, ToolContext
from reorderpilot.api.schemas import FeedbackEvent, ListCustomersDueArgs, PredictReorderArgs, ToolError
from reorderpilot.memory import store
from tests.test_agent_loop import AS_OF, ctx  # noqa: F401  (shared fixture)


def test_suppression_applied_only_when_memory_enabled(ctx):  # noqa: F811
    store.record_feedback(ctx.db_path, FeedbackEvent(customer_id="A", outcome="declined", stock_code="P1",
                                                     reason="overstocked"), AS_OF)
    on = tools.predict_reorder(PredictReorderArgs(customer_id="A"), ctx)
    p1 = next(r for r in on.recommendations if r.stock_code == "P1")
    assert p1.suppressed and "overstocked" in p1.suppressed_reason
    off_ctx = ToolContext(AS_OF, ctx.db_path, False, ctx.tx_loader, ctx.bundle_loader)
    off = tools.predict_reorder(PredictReorderArgs(customer_id="A"), off_ctx)
    assert not any(r.suppressed for r in off.recommendations)
    assert tools.read_memory(tools.ReadMemoryArgs(customer_id="A"), off_ctx).feedback_events == []


def test_find_product_exact_fuzzy_and_ambiguous(ctx):  # noqa: F811
    exact = tools.find_product(FindProductToolArgs(query="p1", customer_id="A"), ctx)
    assert exact.matches[0].stock_code == "P1" and not exact.ambiguous
    fuzzy = tools.find_product(FindProductToolArgs(query="red retrospot lunch bags", customer_id="A"), ctx)
    assert fuzzy.matches[0].stock_code == "P1" and not fuzzy.ambiguous
    vague = tools.find_product(FindProductToolArgs(query="lunch bag", customer_id="A"), ctx)
    assert vague.ambiguous and {m.stock_code for m in vague.matches} == {"P1", "P2"}


def test_run_tool_coerces_numeric_customer_id_and_rejects_bad_args(ctx):  # noqa: F811
    tx = ctx.tx_loader()
    tx.loc[tx["customer_id"] == "A", "customer_id"] = "12345"
    ok = tools.run_tool("write_memory", {"customer_id": 12345.0, "outcome": "ordered"}, ctx)
    assert ok.event.customer_id == "12345"
    bad = tools.run_tool("write_memory", {"customer_id": "12345", "outcome": "maybe"}, ctx)
    assert isinstance(bad, ToolError) and "outcome" in bad.error


def test_list_customers_due_excludes_cold_start(ctx):  # noqa: F811
    due = tools.list_customers_due(ListCustomersDueArgs(limit=5), ctx)
    assert [c.customer_id for c in due.customers] == ["A"]          # B has only 2 orders -> abstained


def test_tool_specs_cover_all_tools_without_titles():
    specs = tools.get_tool_specs()
    assert {s["function"]["name"] for s in specs} == set(tools.TOOLS)
    assert '"title"' not in str(specs).replace("'", '"')
