# agentloop

[![ci](https://github.com/abbasmsh1/agentloop/actions/workflows/ci.yml/badge.svg)](https://github.com/abbasmsh1/agentloop/actions/workflows/ci.yml)

A minimal, auditable AI agent: tool calling with guardrails, human approval for risky actions, and a full run log.

Reference implementation behind my [Fiverr AI agent gig](https://www.fiverr.com/abbasmsh1). Most agent demos break the moment a tool fails or the model tries something it shouldn't. This one is built around the boring parts that make agents safe to run: permissions, approval gates, retries, and logs.

## Quick start

```bash
pip install -e .
cp .env.example .env        # add your OPENAI_API_KEY

agentloop "What is (17 * 43) + 12? Then save the answer to result.txt"
```

The agent reasons, calls the calculator, then asks for confirmation before writing the file:

```
[tool] calculate: (17 * 43) + 12 = 743
[approval needed] write_file(path=result.txt) - approve? [y/N]
```

## Guardrails

| Guardrail | How |
|-----------|-----|
| Tool allowlist | Agent can only call registered tools, nothing else |
| Scoped toolsets | A run can be restricted to a named subset of tools; the rest are blocked |
| Human approval | Tools marked `requires_approval` pause and ask before running - in the CLI and in the web UI |
| Path sandbox | File tools refuse paths outside an explicit per-run sandbox directory |
| Step budget | Hard cap on loop iterations; no runaway agents |
| Token budget | `max_tokens` cap per model call; usage logged per call |
| Retries | Transient API errors retried with backoff, 60s request timeout |
| Run log | Every model call, tool result, and approval decision appended to `runs/*.jsonl` |

## Web demo: a support agent that can't move money alone

```bash
pip install -e .[web]
uvicorn api.index:app --reload
```

Open http://localhost:8000: a support console for Aster & Pine, a fictional coffee-gear store. The agent has scoped tools - `lookup_order`, `find_orders`, `add_note`, `issue_refund` - a refund policy in its system prompt, and a seeded order book (state lives in `orders.json` inside the per-run sandbox). Reads and notes flow freely; `issue_refund` pauses the run and shows a countersign card - Approve resumes it, Deny feeds the refusal back to the model. The pause survives serverless statelessness: the paused conversation is returned to the browser and posted back with your decision. Every run ends with a receipt-style audit log, token counts included.

Tools outside the web toolset (like `write_file`) are blocked for web runs even though they are registered - toolsets are scoped per run via `run(tools=[...])`.

## Deploy to Vercel

```bash
vercel --prod        # set OPENAI_API_KEY in the Vercel project settings
```

Each run gets its own `/tmp` scratch sandbox and run log (passed explicitly, no `os.chdir`), so nothing leaks between concurrent users on a shared instance. `/api/run` is rate limited per IP. Optional: set `DEMO_TOKEN` to require an `X-Demo-Token` header, protecting your OpenAI credits on a public URL.

## Add your own tool

```python
from agentloop.tools import tool

@tool(description="Look up an order by id", requires_approval=False)
def get_order(order_id: str) -> str:
    return my_crm.lookup(order_id)
```

That's it: the JSON schema is generated from the signature, including parameter types from the annotations (`str`, `int`, `float`, `bool`), and the agent can use it.

## Development

```bash
pip install -e .[dev]
pytest          # offline: the OpenAI client is faked in tests
ruff check .
```

CI runs both on every push.

## Layout

```
agentloop/agent.py    # the loop: model -> tool -> result -> model, pause/resume
agentloop/tools.py    # tool registry, decorator, built-in tools, path sandbox
agentloop/shop.py     # demo use case: support tools + refund policy
agentloop/__main__.py # CLI entry
api/index.py          # web API: approval round-trip, rate limit, per-run sandbox
web/index.html        # support console UI
tests/                # loop, guardrail, shop, and API tests (offline)
```

## Author

Abbas Mustafa — MSc Cybersecurity @ IMT Atlantique. Agents with guardrails, not just prompts.
Hire me on [Fiverr](https://www.fiverr.com/abbasmsh1).
