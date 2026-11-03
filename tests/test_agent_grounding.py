"""Runtime grounding guard: answer lines citing stock codes no tool returned are filtered out."""

import json

from reorderpilot.agent.grounding import codes_in, enforce_grounding, grounded_codes


def test_codes_in_finds_stock_codes_case_insensitively():
    assert codes_in("Offer 85123a and 22423, not invoice 536365") == {"85123A", "22423"}


def test_grounded_codes_come_from_tool_results_and_rep_messages():
    messages = [{"role": "system", "content": "Today is 2011-12-01."},
                {"role": "user", "content": "Brief me on 13085"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "content": json.dumps({"recommendations": [{"stock_code": "85123A"}]})}]
    assert grounded_codes(messages) == {"13085", "85123A"}


def test_grounded_answer_is_unchanged():
    text = "Customer 13085:\n- 85123A WHITE HANGING HEART (62%)"
    assert enforce_grounding(text, {"13085", "85123A"}) == (text, [])


def test_invented_code_line_is_removed_and_noted():
    text = "Customer 13085:\n- 85123A heart holder (62%)\n- 99999Z magic mug (80%)"
    safe, removed = enforce_grounding(text, {"13085", "85123A"})
    assert removed == ["99999Z"]
    assert "99999Z" not in safe and "85123A" in safe
    assert "1 line(s) were removed" in safe


def test_note_does_not_repeat_removed_codes():
    """The answer is replayed as history next turn; the code must not become 'grounded' through the note."""
    safe, _ = enforce_grounding("Try 99999Z", set())
    assert codes_in(safe) == set()
