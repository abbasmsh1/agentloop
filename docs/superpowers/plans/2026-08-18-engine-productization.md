# Engine Productization (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the agentloop engine into a general platform core: bring-your-own-tools over HTTP, per-run approval policy, pluggable run storage, and per-request provider credentials.

**Architecture:** All changes live in the existing `agentloop` package. The registry gains a programmatic `register_tool` plus an SSRF-guarded webhook tool factory. `run()` gains `require_approval` (policy override), `store` (RunStore), and `credentials` (BYOK). The web API forwards a provider key header per request and reads events through the store.

**Tech Stack:** Python stdlib (`urllib`, `ipaddress`, `socket`) for webhooks; existing openai/anthropic SDKs; pytest.

**Spec:** `docs/PRODUCT.md`

## Global Constraints

- No new dependencies (engine: openai + anthropic + stdlib only).
- All tests offline; fake every network boundary except the loopback test server.
- Webhook URLs must refuse non-http(s) schemes and private/loopback addresses unless `allow_private=True`.
- Provider keys are never written to any store or log.
- `ruff check .` and `pytest -q` green after every task; run both before each commit.

---

### Task 1: register_tool + webhook tools

**Files:**
- Modify: `agentloop/tools.py`
- Test: `tests/test_webhook_tools.py`

**Interfaces:**
- Produces: `register_tool(name, description, fn, params, requires_approval=False)` where `params` is `{arg_name: json_type_str}`; `register_webhook_tool(name, description, url, params, requires_approval=True, timeout=15, allow_private=False)`; `WebhookURLError(ValueError)`.
- The existing `tool()` decorator must keep working unchanged (it becomes a thin wrapper over `register_tool`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_webhook_tools.py
import http.server
import json
import threading

import pytest

from agentloop.tools import (REGISTRY, WebhookURLError, register_tool,
                             register_webhook_tool)


def test_register_tool_builds_spec():
    register_tool("t1", "desc", lambda x: x, {"x": "integer"})
    spec = REGISTRY.pop("t1")["spec"]["function"]
    assert spec["parameters"]["properties"] == {"x": {"type": "integer"}}
    assert spec["parameters"]["required"] == ["x"]


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "http://127.0.0.1:9/hook",
    "http://localhost/hook",
])
def test_webhook_rejects_unsafe_urls(url):
    with pytest.raises(WebhookURLError):
        register_webhook_tool("bad", "d", url, {"a": "string"})
    assert "bad" not in REGISTRY


def test_webhook_tool_calls_endpoint():
    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        register_webhook_tool("crm_lookup", "d", f"http://127.0.0.1:{srv.server_port}/hook",
                              {"order_id": "string"}, allow_private=True)
        entry = REGISTRY.pop("crm_lookup")
        assert entry["requires_approval"] is True  # webhooks default to gated
        assert entry["fn"](order_id="A-1") == '{"ok": true}'
        assert received == {"order_id": "A-1"}
    finally:
        srv.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_webhook_tools.py -v`
Expected: FAIL with `ImportError: cannot import name 'register_tool'`

- [ ] **Step 3: Implement in `agentloop/tools.py`**

Refactor `tool()` to delegate, add the new functions:

```python
import ipaddress
import socket
import urllib.request
from urllib.parse import urlparse


class WebhookURLError(ValueError):
    pass


def register_tool(name, description, fn, params, requires_approval=False):
    """Register any callable as an agent tool. params: {arg_name: json_type}."""
    REGISTRY[name] = {
        "fn": fn,
        "requires_approval": requires_approval,
        "spec": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": v} for k, v in params.items()},
                    "required": list(params),
                },
            },
        },
    }


def tool(description, requires_approval=False):
    """Register a function as an agent tool. Schema comes from the signature."""
    def wrap(fn):
        params = {n: _JSON_TYPES.get(p.annotation, "string")
                  for n, p in inspect.signature(fn).parameters.items()}
        register_tool(fn.__name__, description, fn, params, requires_approval)
        return fn
    return wrap


def register_webhook_tool(name, description, url, params,
                          requires_approval=True, timeout=15, allow_private=False):
    """Bring-your-own-tool over HTTP: args POSTed as JSON, response text returned."""
    _check_webhook_url(url, allow_private)

    def call(**kwargs):
        req = urllib.request.Request(
            url, data=json.dumps(kwargs).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(65536).decode("utf-8", "replace")[:8000]

    register_tool(name, description, call, params, requires_approval)


def _check_webhook_url(url, allow_private):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise WebhookURLError(f"webhook url must be http(s): {url}")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or 80)
    except OSError as e:
        raise WebhookURLError(f"cannot resolve {p.hostname}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise WebhookURLError(f"webhook host resolves to non-public address {ip}")
```

(`json` and `inspect` are already imported in tools.py; add the three new imports at top. Note this checks at registration time; hosted M2 must re-check at call time against DNS rebinding.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_webhook_tools.py tests/ -q`
Expected: all PASS (existing suite proves `tool()` still works)

- [ ] **Step 5: Commit**

```bash
git add agentloop/tools.py tests/test_webhook_tools.py
git commit -m "feat: register_tool API and SSRF-guarded webhook tools"
```

---

### Task 2: per-run approval overrides

**Files:**
- Modify: `agentloop/agent.py` (the `needs approval` branch in `run()`)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: existing `run(..., tools=None)` scoping.
- Produces: `run(..., require_approval=None)` — iterable of tool names that need approval this run even if registered with `requires_approval=False`. Callers must pass the same value when resuming, as with `tools`/`system`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_agent.py`)

```python
def test_require_approval_override_pauses_ungated_tool(fake_openai, tmp_path):
    responses, _ = fake_openai
    responses.append(tool_call_resp("calculate", '{"expression": "1+1"}'))
    with pytest.raises(ApprovalNeeded) as exc_info:
        run("calc", approve=None, require_approval=["calculate"],
            runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert exc_info.value.tool == "calculate"

    responses.append(answer_resp("2"))
    answer, _ = run(state=exc_info.value.state, decision=True,
                    require_approval=["calculate"],
                    runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agent.py::test_require_approval_override_pauses_ungated_tool -v`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'require_approval'`

- [ ] **Step 3: Implement**

In `run()`'s signature add `require_approval=None`; in the pending loop replace

```python
in_scope = name in REGISTRY and (tools is None or name in tools)
if in_scope and REGISTRY[name]["requires_approval"]:
```

with

```python
in_scope = name in REGISTRY and (tools is None or name in tools)
gated = in_scope and (REGISTRY[name]["requires_approval"]
                      or (require_approval and name in require_approval))
if gated:
```

Also document the new kwarg in the `run()` docstring: `require_approval: extra tool names to approval-gate for this run.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add agentloop/agent.py tests/test_agent.py
git commit -m "feat: per-run approval overrides via require_approval"
```

---

### Task 3: RunStore interface

**Files:**
- Create: `agentloop/stores.py`
- Modify: `agentloop/agent.py` (all `_log`/`run_file` plumbing), `api/index.py` (`_events`), `tests/test_agent.py`, `tests/test_guardrails.py`
- Test: `tests/test_stores.py`

**Interfaces:**
- Produces: `JsonlRunStore(path)` and `MemoryRunStore()`, both with `.append(entry: dict) -> None`, `.load() -> list[dict]`, and `.name: str` (jsonl: file basename; memory: `"memory"`).
- Changes: `run()` returns `(answer, store)` instead of `(answer, run_file_path)`; `ApprovalNeeded.store` replaces `.run_file`; `_execute(name, args, approved, store, tools=None)`; `run()` gains `store=None` (default: `JsonlRunStore` under `runs_dir`, resuming the file named by `state["run_name"]`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stores.py
from agentloop.stores import JsonlRunStore, MemoryRunStore


def test_jsonl_store_roundtrip(tmp_path):
    s = JsonlRunStore(str(tmp_path / "r.jsonl"))
    s.append({"event": "task"})
    s.append({"event": "answer"})
    assert [e["event"] for e in s.load()] == ["task", "answer"]
    assert s.name == "r.jsonl"


def test_memory_store_roundtrip():
    s = MemoryRunStore()
    s.append({"event": "task"})
    assert s.load() == [{"event": "task"}]
    assert s.name == "memory"


def test_missing_file_loads_empty(tmp_path):
    assert JsonlRunStore(str(tmp_path / "nope.jsonl")).load() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_stores.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentloop.stores'`

- [ ] **Step 3: Implement `agentloop/stores.py`**

```python
"""Run event storage. The engine appends; frontends load."""

import json
import os


class JsonlRunStore:
    """One run, one .jsonl file. The default store."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def append(self, entry):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []


class MemoryRunStore:
    """In-process store for tests and embedding."""

    name = "memory"

    def __init__(self):
        self.events = []

    def append(self, entry):
        self.events.append(dict(entry))

    def load(self):
        return list(self.events)
```

- [ ] **Step 4: Rewire `agentloop/agent.py`**

- `_log(store, entry)`: set `entry["ts"]` then `store.append(entry)`.
- In `run()`: replace the `run_name`/`run_file` block with

```python
    if store is None:
        os.makedirs(runs_dir, exist_ok=True)
        run_name = os.path.basename((state or {}).get("run_name") or f"run-{int(time.time())}.jsonl")
        store = JsonlRunStore(os.path.join(runs_dir, run_name))
```

- `ApprovalNeeded(tool, args, state, store)` with `self.store = store`; the raise site puts `"run_name": store.name` into state.
- Every `_log(run_file, ...)` becomes `_log(store, ...)`; both `return` sites return `(answer, store)`; `_execute` takes `store`.
- `from .stores import JsonlRunStore` at top.

In `api/index.py`: `_events(store)` becomes `return store.load() if store else []`; the two call sites pass `exc.store` / `store` (rename the `run()` result variable accordingly). In `agentloop/__main__.py`: print `store.path`.

In tests: `test_agent.py` replaces `open(run_file)` reads with `store.load()` (`events = [e["event"] for e in store.load()]`); `test_guardrails.py` passes `MemoryRunStore()` instead of `"log.jsonl"` to `_execute` and imports it from `agentloop.stores`.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check .`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agentloop/stores.py agentloop/agent.py agentloop/__main__.py api/index.py tests/
git commit -m "feat: pluggable RunStore (jsonl + memory)"
```

---

### Task 4: BYOK credentials per run

**Files:**
- Modify: `agentloop/agent.py` (`run()`, `_make_caller`), `api/index.py`
- Test: `tests/test_anthropic_backend.py`, `tests/test_api.py` (append)

**Interfaces:**
- Produces: `run(..., credentials=None)` where credentials is `{"api_key": str}`; `_make_caller(model, credentials=None)`. When set, `api_key` overrides env/OAuth resolution for whichever provider the model selects. Web API reads it from the `X-Provider-Key` request header and never logs or stores it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anthropic_backend.py`:

```python
def test_credentials_override_env(fake_anthropic, monkeypatch, tmp_path):
    from agentloop.agent import run
    responses, calls = fake_anthropic
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-should-lose")
    responses.append(a_answer("hi"))
    run("hello", model="claude-opus-5", credentials={"api_key": "sk-byok"},
        runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert calls[0]["client_kwargs"]["api_key"] == "sk-byok"
    assert "auth_token" not in calls[0]["client_kwargs"]
```

Append to `tests/test_api.py`:

```python
def test_provider_key_header_forwarded(client, fake_openai, monkeypatch):
    seen = {}
    import api.index as api_mod
    real_run = api_mod.run
    def spy(**kw):
        seen.update(kw)
        return real_run(**kw)
    monkeypatch.setattr(api_mod, "run", spy)
    fake_openai[0].append(answer_resp("hi"))
    client.post("/api/run", json={"task": "hi"}, headers={"X-Provider-Key": "sk-byok"})
    assert seen["credentials"] == {"api_key": "sk-byok"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_anthropic_backend.py tests/test_api.py -v`
Expected: the two new tests FAIL (`TypeError: run() got an unexpected keyword argument 'credentials'` / KeyError)

- [ ] **Step 3: Implement**

`_make_caller(model, credentials=None)`; `key = (credentials or {}).get("api_key")`. Anthropic branch: if `key`, build `anthropic.Anthropic(api_key=key, max_retries=3, timeout=60)` and skip the OAuth-token branch. OpenAI branch: `OpenAI(api_key=key or os.environ["OPENAI_API_KEY"], ...)`. `run()` passes `credentials` through; docstring line: `credentials: {"api_key": ...} overriding server env for this run.`

`api/index.py`: add `x_provider_key: str | None = Header(default=None)` to `run_task`, pass `credentials={"api_key": x_provider_key} if x_provider_key else None` into `run()`, and let a present header satisfy the credential 503 check for either provider.

- [ ] **Step 4: Run full suite**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check .`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add agentloop/agent.py api/index.py tests/
git commit -m "feat: BYOK provider credentials per run"
```
