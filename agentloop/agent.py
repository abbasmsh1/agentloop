"""The agent loop: model -> tool calls -> results -> model, with guardrails."""

import json
import os
import time

from .tools import BASE_DIR, REGISTRY, anthropic_specs, openai_specs

MAX_STEPS = 10

SYSTEM = (
    "You are a careful assistant. Use the provided tools when they help. "
    "Never invent tool results. When done, answer the user plainly."
)


class ApprovalNeeded(Exception):
    """A tool needs human approval and no approve callback was given.

    Carries everything needed to resume: pass `state` and a `decision`
    back into run() to continue where the run paused.
    """

    def __init__(self, tool, args, state, run_file):
        super().__init__(f"approval needed for {tool}")
        self.tool = tool
        self.tool_args = args  # not .args: BaseException.args coerces to a tuple
        self.state = state
        self.run_file = run_file


def _log(run_file, entry):
    entry["ts"] = time.time()
    with open(run_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _approve(name, args):
    prompt = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"[approval needed] {name}({prompt}) - approve? [y/N] ", end="")
    return input().strip().lower() == "y"


def run(task=None, approve=_approve, model="gpt-4o-mini", runs_dir=None, work_dir=None,
        state=None, decision=None, max_tokens=2000, system=None, tools=None):
    """Run the agent on a task. Returns (answer, run_file_path).

    approve: callback(name, args) -> bool for tools marked requires_approval.
    Pass approve=None to pause instead: the run raises ApprovalNeeded, and the
    caller resumes it later with state=exc.state plus decision=True/False.
    work_dir: sandbox root for file tools (defaults to the process cwd).
    system: override the default system prompt.
    tools: restrict this run to a subset of registered tool names.
    """
    call = _make_caller(model)
    runs_dir = runs_dir or os.environ.get("RUNS_DIR", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_name = os.path.basename((state or {}).get("run_name") or f"run-{int(time.time())}.jsonl")
    run_file = os.path.join(runs_dir, run_name)

    if state:
        messages = state["messages"]
        pending = list(state["pending"])
        start = state["step"]
        _log(run_file, {"event": "resume", "decision": bool(decision)})
    else:
        messages = [{"role": "system", "content": system or SYSTEM},
                    {"role": "user", "content": task}]
        pending = []
        start = 0
        _log(run_file, {"event": "task", "task": task})

    token = BASE_DIR.set(work_dir) if work_dir else None
    try:
        for step in range(start, MAX_STEPS):
            if not pending:
                text, calls, usage = call(messages, tools, max_tokens)
                _log(run_file, {"event": "model", "model": model, **usage})
                if not calls:
                    _log(run_file, {"event": "answer", "content": text})
                    return text, run_file
                messages.append({"role": "assistant", "content": text, "tool_calls": calls})
                pending = list(calls)

            while pending:
                tc = pending[0]
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"] or "{}")
                in_scope = name in REGISTRY and (tools is None or name in tools)
                if in_scope and REGISTRY[name]["requires_approval"]:
                    if decision is not None:
                        approved, decision = bool(decision), None
                    elif approve is not None:
                        approved = approve(name, args)
                    else:
                        raise ApprovalNeeded(
                            name, args,
                            {"messages": messages, "pending": pending, "step": step,
                             "run_name": run_name},
                            run_file,
                        )
                else:
                    approved = True
                pending.pop(0)
                result = _execute(name, args, approved, run_file, tools)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        _log(run_file, {"event": "abort", "reason": "step budget exhausted"})
        return "Stopped: step budget exhausted before the task finished.", run_file
    finally:
        if token:
            BASE_DIR.reset(token)


def _make_caller(model):
    """Provider adapter: returns call(messages, tools, max_tokens) -> (text, tool_calls, usage).

    History stays in one internal format (OpenAI-style dicts) regardless of
    provider, so pause/resume state is provider-independent to store.
    """
    if model.startswith("claude"):
        import anthropic

        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if token and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # Claude subscription OAuth token (from `claude setup-token`)
            client = anthropic.Anthropic(auth_token=token, max_retries=3, timeout=60,
                                         default_headers={"anthropic-beta": "oauth-2025-04-20"})
        else:
            client = anthropic.Anthropic(max_retries=3, timeout=60)
        return lambda m, t, mt: _anthropic_call(client, model, m, t, mt)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3, timeout=60)
    return lambda m, t, mt: _openai_call(client, model, m, t, mt)


def _openai_call(client, model, messages, tools, max_tokens):
    resp = client.chat.completions.create(model=model, messages=messages,
                                          tools=openai_specs(tools), max_tokens=max_tokens)
    msg = resp.choices[0].message
    calls = [tc.model_dump() for tc in msg.tool_calls or []]
    usage = {}
    if resp.usage:
        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens}
    return msg.content, calls, usage


def _anthropic_call(client, model, messages, tools, max_tokens):
    system = ""
    history = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            history.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": tc["id"],
                               "name": tc["function"]["name"],
                               "input": json.loads(tc["function"]["arguments"] or "{}")})
            history.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}]})
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=history, tools=anthropic_specs(tools))
    text = "".join(b.text for b in resp.content if b.type == "text")
    calls = [{"id": b.id, "type": "function",
              "function": {"name": b.name, "arguments": json.dumps(b.input)}}
             for b in resp.content if b.type == "tool_use"]
    return text, calls, {"prompt_tokens": resp.usage.input_tokens,
                         "completion_tokens": resp.usage.output_tokens}


def _execute(name, args, approved, run_file, tools=None):
    if name not in REGISTRY or (tools is not None and name not in tools):
        # allowlist: only registered tools, only the ones scoped to this run
        _log(run_file, {"event": "blocked", "tool": name, "reason": "not in allowlist"})
        return f"error: tool '{name}' does not exist"
    entry = REGISTRY[name]
    if entry["requires_approval"] and not approved:
        _log(run_file, {"event": "denied", "tool": name, "args": args})
        return "error: the user denied this action"
    try:
        result = entry["fn"](**args)
        _log(run_file, {"event": "tool", "tool": name, "args": args, "result": str(result)[:500]})
        print(f"[tool] {name}: {str(result)[:120]}")
        return str(result)
    except Exception as e:
        _log(run_file, {"event": "tool_error", "tool": name, "args": args, "error": str(e)})
        return f"error: {e}"
