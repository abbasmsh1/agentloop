"""The agent loop: model -> tool calls -> results -> model, with guardrails."""
import json
import os
import time

from .tools import REGISTRY, openai_specs

MAX_STEPS = 10

SYSTEM = (
    "You are a careful assistant. Use the provided tools when they help. "
    "Never invent tool results. When done, answer the user plainly."
)


def _log(run_file, entry):
    entry["ts"] = time.time()
    with open(run_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _approve(name, args):
    print(f"[approval needed] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())}) - approve? [y/N] ", end="")
    return input().strip().lower() == "y"


def run(task, approve=_approve, model="gpt-4o-mini", runs_dir=None):
    """Run the agent on a task. Returns (answer, run_file_path)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    runs_dir = runs_dir or os.environ.get("RUNS_DIR", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_file = os.path.join(runs_dir, f"run-{int(time.time())}.jsonl")
    _log(run_file, {"event": "task", "task": task})

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=model, messages=messages, tools=openai_specs())
        msg = resp.choices[0].message
        if not msg.tool_calls:
            _log(run_file, {"event": "answer", "content": msg.content})
            return msg.content, run_file

        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = _execute(name, args, approve, run_file)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    _log(run_file, {"event": "abort", "reason": "step budget exhausted"})
    return "Stopped: step budget exhausted before the task finished.", run_file


def _execute(name, args, approve, run_file):
    if name not in REGISTRY:  # allowlist: model can only call registered tools
        _log(run_file, {"event": "blocked", "tool": name, "reason": "not in allowlist"})
        return f"error: tool '{name}' does not exist"
    entry = REGISTRY[name]
    if entry["requires_approval"] and not approve(name, args):
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
