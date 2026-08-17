# agentloop

A minimal, auditable AI agent: tool calling with guardrails, human approval for risky actions, and a full run log.

Reference implementation behind my [Fiverr AI agent gig](https://www.fiverr.com/abbasmsh1). Most agent demos break the moment a tool fails or the model tries something it shouldn't. This one is built around the boring parts that make agents safe to run: permissions, approval gates, retries, and logs.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY

python -m agentloop "What is (17 * 43) + 12? Then save the answer to result.txt"
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
| Human approval | Tools marked `requires_approval` pause and ask before running |
| Path sandbox | `write_file` refuses paths outside the working directory |
| Step budget | Hard cap on loop iterations; no runaway agents |
| Run log | Every model call and tool result appended to `runs/*.jsonl` |

## Web UI

```bash
uvicorn api.index:app --reload
```

Open http://localhost:8000: chat with the agent, toggle auto-approve for write actions, and expand the full run log under each answer.

## Deploy to Vercel

```bash
vercel --prod        # set OPENAI_API_KEY in the Vercel project settings
```

On serverless the run logs go to `/tmp/runs` and file tools operate in a `/tmp` scratch sandbox.

## Add your own tool

```python
from agentloop.tools import tool

@tool(description="Look up an order by id", requires_approval=False)
def get_order(order_id: str) -> str:
    return my_crm.lookup(order_id)
```

That's it: the schema is generated from the signature and the agent can use it.

## Layout

```
agentloop/agent.py    # the loop: model -> tool -> result -> model
agentloop/tools.py    # tool registry, decorator, built-in tools
agentloop/__main__.py # CLI entry
tests/                # guardrail + tool tests (offline)
```

## Author

Abbas Mustafa — MSc Cybersecurity @ IMT Atlantique. Agents with guardrails, not just prompts.
Hire me on [Fiverr](https://www.fiverr.com/abbasmsh1).
