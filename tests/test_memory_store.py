"""Memory store + deterministic policy (SPEC section 7). Every test uses a temporary DB, never the real memory.db."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from reorderpilot.api.schemas import FeedbackEvent, PredictionResult, Recommendation
from reorderpilot.memory import store

DAY0 = date(2011, 11, 1)
CUST = "12345"


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "memory.db"
    store.init_db(path)
    return path


def _decline(reason, stock_code="85099B", customer_id=CUST, note=None):
    return FeedbackEvent(customer_id=customer_id, outcome="declined", stock_code=stock_code, reason=reason, note=note)


def _prediction(*codes):
    recs = [Recommendation(stock_code=c, description=c, probability=0.5, reason_text="x") for c in codes]
    return PredictionResult(customer_id=CUST, as_of=DAY0, recommendations=recs)


@pytest.mark.parametrize(("reason", "days"), [("overstocked", 30), ("seasonal", 30), ("switched_supplier", 90)])
def test_decline_suppresses_product_for_policy_days(db, reason, days):
    rec = store.add_feedback_event(db, _decline(reason), DAY0)
    assert rec.expires_at == DAY0 + timedelta(days=days)
    assert store.active_suppressions(db, CUST, DAY0) == {"85099B": f"declined: {reason} (until {rec.expires_at})"}


def test_suppression_expires_on_expires_at(db):
    store.add_feedback_event(db, _decline("overstocked"), DAY0)
    assert "85099B" in store.active_suppressions(db, CUST, DAY0 + timedelta(days=29))
    assert store.active_suppressions(db, CUST, DAY0 + timedelta(days=30)) == {}
    # Expired events disappear from the default listing but stay in the audit trail.
    assert store.list_feedback_events(db, CUST, DAY0 + timedelta(days=30)) == []
    assert len(store.list_feedback_events(db, CUST, DAY0 + timedelta(days=30), include_expired=True)) == 1


def test_events_from_the_future_are_invisible(db):
    store.add_feedback_event(db, _decline("overstocked"), DAY0)
    assert store.active_suppressions(db, CUST, DAY0 - timedelta(days=1)) == {}
    assert store.list_feedback_events(db, CUST, DAY0 - timedelta(days=1), include_expired=True) == []


def test_price_decline_marks_price_sensitive_without_suppressing(db):
    result = store.record_feedback(db, _decline("price"), DAY0)
    assert result.event.expires_at is None
    assert "price-sensitive" in result.effect
    assert store.active_suppressions(db, CUST, DAY0) == {}
    assert store.is_price_sensitive(db, CUST, DAY0 + timedelta(days=365))
    assert not store.is_price_sensitive(db, "99999", DAY0)


def test_ordered_and_other_reasons_only_record(db):
    store.add_feedback_event(db, FeedbackEvent(customer_id=CUST, outcome="ordered", stock_code="85099B"), DAY0)
    store.add_feedback_event(db, _decline("other", stock_code="22423"), DAY0)
    assert store.active_suppressions(db, CUST, DAY0) == {}
    assert len(store.list_feedback_events(db, CUST, DAY0)) == 2


def test_asked_alternative_is_saved_as_note(db):
    event = FeedbackEvent(customer_id=CUST, outcome="asked_alternative", stock_code="85099B", note="wants a blue one")
    store.record_feedback(db, event, DAY0)
    assert [n.note for n in store.list_customer_notes(db, CUST, DAY0)] == ["wants a blue one"]


def test_duplicate_event_is_stored_once(db):
    first = store.record_feedback(db, _decline("overstocked"), DAY0)
    second = store.record_feedback(db, _decline("overstocked"), DAY0)
    assert second.event.id == first.event.id
    assert second.effect.startswith("duplicate ignored")
    assert len(store.list_feedback_events(db, CUST, DAY0, include_expired=True)) == 1
    # Same decline on a later day is a new fact -> new row; the later expiry wins.
    later = store.add_feedback_event(db, _decline("overstocked"), DAY0 + timedelta(days=10))
    assert later.id != first.event.id
    assert "2011-12-11" in store.active_suppressions(db, CUST, DAY0 + timedelta(days=10))["85099B"]


def test_duplicate_asked_alternative_adds_one_note(db):
    event = FeedbackEvent(customer_id=CUST, outcome="asked_alternative", stock_code="85099B")
    store.record_feedback(db, event, DAY0)
    store.record_feedback(db, event, DAY0)
    assert len(store.list_customer_notes(db, CUST, DAY0)) == 1


def test_note_max_length(db):
    store.add_customer_note(db, CUST, "x" * 500, DAY0)
    with pytest.raises(ValueError):
        store.add_customer_note(db, CUST, "x" * 501, DAY0)
    with pytest.raises(ValidationError):
        _decline("other", note="x" * 501)
    assert len(store.list_customer_notes(db, CUST, DAY0)) == 1


def test_apply_suppressions_flags_items_and_keeps_order(db):
    store.add_feedback_event(db, _decline("overstocked", stock_code="22423"), DAY0)
    store.add_feedback_event(db, _decline("price", stock_code="85099B"), DAY0)
    out = store.apply_suppressions(_prediction("85099B", "22423"), db, DAY0)
    assert [r.stock_code for r in out.recommendations] == ["85099B", "22423"]
    assert [r.suppressed for r in out.recommendations] == [False, True]
    assert out.recommendations[1].suppressed_reason.startswith("declined: overstocked")
    assert out.price_sensitive is True


def test_memory_is_per_customer(db):
    store.add_feedback_event(db, _decline("overstocked"), DAY0)
    store.add_customer_note(db, CUST, "likes red", DAY0)
    other = store.read_memory(db, "99999", DAY0)
    assert other.notes == [] and other.feedback_events == [] and other.price_sensitive is False


def test_messages_last_n_oldest_first(db):
    for i in range(5):
        store.add_message(db, "s1", "user", f"m{i}", None, DAY0)
    store.add_message(db, "s2", "user", "other session", None, DAY0)
    assert [m.content for m in store.list_messages(db, "s1", limit=3)] == ["m2", "m3", "m4"]
    with pytest.raises(ValidationError):
        store.add_message(db, "s1", "hacker", "bad role", None, DAY0)
    assert len(store.list_messages(db, "s1", limit=10)) == 5
