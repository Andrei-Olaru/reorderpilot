"""Runtime anti-hallucination guard: a final answer may only cite stock codes that appeared earlier in the turn.

The system prompt asks the LLM to use only tool data (rule 2), but a prompt is a request, not a guarantee. This
module enforces it in code, deterministically, on every answer before the rep sees it (see DECISIONS D14).

"Grounded" = the code appears in some earlier message of the transcript: a tool result, the rep's own message, or a
replayed earlier turn (whose answer was itself filtered). Any answer line that cites a code-shaped token outside
that set is removed and a short note is appended.
"""

from __future__ import annotations

import re

# Online Retail II stock codes: 5 digits + up to 2 letters (85123A, 22423). Customer ids are 5 digits too, which is
# why the allowed set is built from the whole transcript (customer ids appear in the rep message / tool results).
CODE_RE = re.compile(r"\b(\d{5}[A-Z]{0,2})\b")

# The note deliberately does NOT list the removed codes: answers are replayed as history in later turns, and a code
# written in the note would then look "grounded" there.
REMOVED_NOTE = ("(Note: {n} line(s) were removed because they cited product codes that no tool returned. "
                "Ask me to look the product up if you need it.)")


def codes_in(text: str) -> set[str]:
    """Every code-shaped token in a text (upper-cased first, so '85123a' is caught too)."""
    return set(CODE_RE.findall(text.upper()))


def grounded_codes(messages: list[dict]) -> set[str]:
    """Codes that appeared in any message content so far (tool results are JSON strings, so they are scanned too)."""
    found: set[str] = set()
    for m in messages:
        if isinstance(m.get("content"), str):
            found |= codes_in(m["content"])
    return found


def enforce_grounding(answer: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Drop answer lines that cite codes outside `allowed`; return (safe answer, sorted removed codes)."""
    kept, removed = [], set()
    for line in answer.splitlines():
        invented = codes_in(line) - allowed
        if invented:
            removed |= invented
            continue  # filter the whole line: a half-sentence about a fake product is still misleading
        kept.append(line)
    if not removed:
        return answer, []
    n_lines = len(answer.splitlines()) - len(kept)
    safe = "\n".join(kept).rstrip()
    return f"{safe}\n\n{REMOVED_NOTE.format(n=n_lines)}".lstrip(), sorted(removed)
