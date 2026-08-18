# agentloop — product spec

## Vision

Open-source auditable agent engine + hosted console. Teams run LLM agents
against their own tools with hard guardrails: allowlists, human approval on
risky actions, budgets, and a complete audit log for every run.

The engine (`pip install agentloop`) is the distribution and credibility
layer. The hosted console is the product layer: register tools, set
policies, approve actions, browse run history.

## Positioning

General agent platform, not a support-bot vertical. The Aster & Pine
support console remains as the flagship template/demo.

## Architecture

```
agentloop (OSS engine)                 hosted console (later milestone)
  - loop: model -> tools -> model        - auth + orgs
  - registry: python + webhook tools     - tool registration UI (webhooks)
  - policy: approval overrides,          - policy editor
    scoped toolsets, budgets             - approval inbox (countersign)
  - pause/resume approval protocol       - run history (DB-backed RunStore)
  - RunStore interface (jsonl default)   - BYOK provider keys
  - providers: OpenAI + Anthropic
```

Key protocol: a run that hits an approval-gated tool raises
`ApprovalNeeded` with serializable state; any frontend (CLI, web, Slack)
resumes it with a decision. This survives stateless serverless.

## Milestones

- **M1 — engine productization** (this repo, no external services):
  webhook tools (bring-your-own-tools over HTTP, SSRF-guarded), per-run
  policy with approval overrides, RunStore interface, BYOK credentials
  per run. Plan: `docs/superpowers/plans/2026-08-18-engine-productization.md`
- **M2 — hosted console**: auth, DB-backed RunStore and tool registry,
  tool/policy UI, approval inbox. Needs provider choices (DB, auth) made
  with the owner's accounts.
- **M3 — teams and billing**: orgs, roles (who may countersign), usage
  metering, plans.

## Non-goals (for now)

Model fine-tuning, agent marketplaces, non-HTTP tool transports,
multi-agent orchestration.

## Constraints

- Engine stays dependency-light: openai + anthropic SDKs only; stdlib for
  everything else (webhook execution uses urllib).
- Every guardrail keeps an offline test. No test calls a real API.
- BYOK: hosted layer never stores provider keys; they ride each request.
- Subscription OAuth tokens are for local/dev use only, never the hosted
  product.
