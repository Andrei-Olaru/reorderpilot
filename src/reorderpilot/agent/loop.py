"""Hand-written agent loop: call LLM, run tool calls, append results, repeat (max 6 steps).

No framework on purpose: every step is visible here and testable with a scripted fake LLM.

One step = one LLM call. If the reply has tool calls, we run them, append their results to the transcript and loop;
if it has none, that text is the final answer, which first passes the grounding guard (agent/grounding.py).
Every failure path ends with a fixed, polite answer and a `stop_reason`, never an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from langsmith import get_current_run_tree, traceable, tracing_context

from reorderpilot.agent import tools
from reorderpilot.agent.grounding import enforce_grounding, grounded_codes
from reorderpilot.agent.llm import ChatModel, LLMError, MalformedToolCallError
from reorderpilot.agent.prompts import PROMPT_VERSION, system_prompt
from reorderpilot.api.schemas import ChatRequest, ChatResponse, ToolCallRecord
from reorderpilot.config import AGENT_MAX_STEPS
from reorderpilot.memory import store

HISTORY_MESSAGES = 10   # short-term memory replayed per session (user + final assistant texts only)
MALFORMED_RETRIES = 1   # SPEC: one retry on malformed tool-call JSON
MAX_STEPS_ANSWER = ("Sorry, I could not finish this request within {n} steps. "
                    "Please ask a narrower question (e.g. one customer at a time).")
MALFORMED_ANSWER = "Sorry, the language model produced invalid tool calls twice. Please rephrase and try again."
LLM_DOWN_ANSWER = "Sorry, the language model is unavailable right now ({err}). Please try again in a minute."
RETRY_NUDGE = "Your previous tool call had invalid JSON arguments. Call the tool again with a valid JSON object."


@dataclass
class TurnResult:
    """Everything one turn produced: the API response plus full detail for evals and debugging."""

    response: ChatResponse
    tool_outputs: list[dict] = field(default_factory=list)   # {"name", "args", "result"} (result as JSON dict)
    messages: list[dict] = field(default_factory=list)       # full transcript sent to the LLM
    steps: int = 0
    stop_reason: str = "answer"   # answer | max_steps | malformed | llm_error
    removed_codes: list[str] = field(default_factory=list)   # invented stock codes the grounding guard filtered out


def _run_tool_call(call: dict, ctx: tools.ToolContext, out: TurnResult) -> tuple[dict, bool]:
    """Execute one tool call -> (tool message, args_were_malformed)."""
    name, raw = call["function"]["name"], call["function"].get("arguments")
    # Bad JSON is reported back to the LLM as the tool's answer (every tool_call_id needs a reply),
    # and the caller counts it against the retry budget.
    try:
        args = tools.parse_arguments(raw)
    except ValueError as e:
        content = json.dumps({"error": f"arguments were not valid JSON ({e}). {RETRY_NUDGE}"})
        return {"role": "tool", "tool_call_id": call["id"], "content": content}, True
    result = _traced_tool(name, args, ctx)
    data = result.model_dump(mode="json", exclude_none=True)
    out.tool_outputs.append({"name": name, "args": args, "result": data})
    out.response.tool_calls.append(ToolCallRecord(name=name, args=args, result_summary=tools.summarize(result)))
    content = json.dumps(data, separators=(",", ":"))
    return {"role": "tool", "tool_call_id": call["id"], "content": content}, False


@traceable(run_type="tool", name="tool", process_inputs=lambda d: {"name": d["name"], "args": d["args"]})
def _traced_tool(name: str, args: dict, ctx: tools.ToolContext):
    """run_tool wrapped in a LangSmith span (a no-op when tracing is off)."""
    return tools.run_tool(name, args, ctx)


def _trace_url() -> str | None:
    """LangSmith URL of the current agent.turn trace, or None when tracing is off or the lookup fails."""
    run = get_current_run_tree()   # None unless we are inside a traced call with tracing enabled
    if run is None:
        return None
    try:
        return run.get_url()
    except Exception:  # noqa: BLE001 - a tracing hiccup must never break the answer
        return None


@traceable(run_type="chain", name="agent.turn", metadata={"prompt_version": PROMPT_VERSION},
           process_inputs=lambda d: {"messages": d["messages"]})
def _loop(messages: list[dict], llm: ChatModel, ctx: tools.ToolContext, max_steps: int) -> TurnResult:
    """The step loop itself (mutates `messages`); see the module docstring for the flow."""
    out = TurnResult(response=ChatResponse(answer=""), messages=messages)
    specs, retries_left = tools.get_tool_specs(), MALFORMED_RETRIES

    def stop(reason: str, answer: str) -> TurnResult:
        out.stop_reason, out.response.answer = reason, answer
        out.response.trace_url = _trace_url()   # shown as "Open LangSmith trace" in the UI
        return out

    while out.steps < max_steps:
        out.steps += 1
        try:
            reply = llm.complete(messages, specs)
        except MalformedToolCallError:
            if retries_left == 0:
                return stop("malformed", MALFORMED_ANSWER)
            retries_left -= 1
            messages.append({"role": "system", "content": RETRY_NUDGE})
            continue
        except LLMError as e:
            return stop("llm_error", LLM_DOWN_ANSWER.format(err=str(e)[:120]))

        calls = reply.get("tool_calls") or []
        if not calls:
            # Final answer. Anti-hallucination: drop lines citing stock codes not seen earlier in the transcript.
            safe, out.removed_codes = enforce_grounding(reply.get("content") or "", grounded_codes(messages))
            return stop("answer", safe)
        # The assistant message with tool_calls must precede the tool results (OpenAI message format).
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
        malformed = False
        for call in calls:
            msg, bad = _run_tool_call(call, ctx, out)
            messages.append(msg)
            malformed = malformed or bad
        if malformed:  # results are already appended, so the LLM sees the error and can retry next step
            if retries_left == 0:
                return stop("malformed", MALFORMED_ANSWER)
            retries_left -= 1
    return stop("max_steps", MAX_STEPS_ANSWER.format(n=max_steps))


def run_turn(user_message: str, llm: ChatModel, ctx: tools.ToolContext, session_id: str | None = None,
             max_steps: int = AGENT_MAX_STEPS, tracing: bool = False) -> TurnResult:
    """Answer one rep message. With session_id, earlier turns are replayed and this turn is saved."""
    history = []
    if session_id:
        history = [{"role": m.role, "content": m.content}
                   for m in store.list_messages(ctx.db_path, session_id, limit=HISTORY_MESSAGES)]
    messages = [{"role": "system", "content": system_prompt(ctx.as_of)}, *history,
                {"role": "user", "content": user_message}]
    with tracing_context(enabled=tracing):
        out = _loop(messages, llm, ctx, max_steps)
    if session_id:
        calls_json = json.dumps([c.model_dump() for c in out.response.tool_calls]) or None
        store.add_message(ctx.db_path, session_id, "user", user_message, None, ctx.as_of)
        store.add_message(ctx.db_path, session_id, "assistant", out.response.answer, calls_json, ctx.as_of)
    return out


def make_chat_handler(llm: ChatModel | None = None, ctx: tools.ToolContext | None = None):
    """Build the POST /agent/chat handler (real LLM + real memory.db unless fakes are passed in)."""
    from reorderpilot.agent.llm import LLMConfig, OpenAICompatibleLLM
    from reorderpilot.config import get_settings

    settings = get_settings()
    llm = llm or OpenAICompatibleLLM(LLMConfig.from_env())
    ctx = ctx or tools.ToolContext(as_of=settings.as_of_date, db_path=settings.memory_db_path)

    def handler(request: ChatRequest) -> ChatResponse:
        return run_turn(request.message, llm, ctx, session_id=request.session_id,
                        tracing=settings.langsmith_tracing).response

    return handler
