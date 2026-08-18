"""Vercel serverless entry: web chat UI over the agent loop.

Serverless FS is read-only outside /tmp. Every run gets its own scratch
directory for file tools and its run log, passed to the agent as an explicit
sandbox root (never os.chdir - instances handle concurrent requests).

Approval flow: a run that hits a tool marked requires_approval pauses and
returns status=approval_needed plus an opaque state blob. The client posts
the blob back with decision=true/false to resume the run.
"""

import hmac
import os
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agentloop import shop  # noqa: E402
from agentloop.agent import ApprovalNeeded, run  # noqa: E402

app = FastAPI(title="agentloop")

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")

MODEL = os.environ.get("MODEL", "gpt-4o-mini")

RATE_LIMIT = 20  # requests per IP per minute
_hits = defaultdict(list)  # ponytail: in-memory, per-instance; use KV if this ever needs to be exact


def check_token(x_demo_token):
    """If DEMO_TOKEN is set, require it - keeps a public demo from burning your API key."""
    required = os.environ.get("DEMO_TOKEN")
    if required and not hmac.compare_digest(x_demo_token or "", required):
        raise HTTPException(401, "missing or wrong X-Demo-Token header")


def check_rate(request):
    fwd = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")
    ip = fwd.split(",")[0].strip()
    now = time.time()
    _hits[ip] = [t for t in _hits[ip] if now - t < 60]
    if len(_hits[ip]) >= RATE_LIMIT:
        raise HTTPException(429, "rate limit: try again in a minute")
    _hits[ip].append(now)


def _valid_work(path):
    """Client-echoed state may only point at an agent scratch dir, nothing else."""
    if not path:
        return False
    full = os.path.realpath(path)
    prefix = os.path.join(os.path.realpath(tempfile.gettempdir()), "agent-")
    return full.startswith(prefix) and os.path.isdir(full)


def _events(store):
    return store.load() if store else []


class RunRequest(BaseModel):
    task: str | None = None
    auto_approve: bool = False
    state: dict | None = None
    decision: bool | None = None


@app.get("/", response_class=HTMLResponse)
def ui():
    with open(_WEB, encoding="utf-8") as f:
        return f.read()


@app.get("/api/orders")
def orders():
    """The seed order book, so the UI can show what there is to ask about."""
    return {"store": shop.STORE, "orders": shop.SEED}


@app.post("/api/run")
def run_task(req: RunRequest, request: Request, x_demo_token: str | None = Header(default=None),
             x_provider_key: str | None = Header(default=None)):
    check_token(x_demo_token)
    check_rate(request)
    if MODEL.startswith("claude"):
        cred_ok = (x_provider_key or os.environ.get("ANTHROPIC_API_KEY")
                   or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                   or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
        if not cred_ok:
            raise HTTPException(503, "no Anthropic credentials configured on the server")
    elif not (x_provider_key or os.environ.get("OPENAI_API_KEY")):
        raise HTTPException(503, "OPENAI_API_KEY not configured on the server")

    if req.state is not None:
        if req.decision is None:
            raise HTTPException(400, "resume needs a decision")
        work = req.state.pop("work", None)
        if not _valid_work(work):  # cold instance or tampered blob: fresh sandbox
            work = tempfile.mkdtemp(prefix="agent-")
        kwargs = {"state": req.state, "decision": req.decision}
    else:
        task = (req.task or "").strip()
        if not task:
            raise HTTPException(400, "task is empty")
        if len(task) > 2000:
            raise HTTPException(400, "task too long")
        work = tempfile.mkdtemp(prefix="agent-")
        kwargs = {"task": task}

    approve = (lambda name, args: True) if req.auto_approve else None
    credentials = {"api_key": x_provider_key} if x_provider_key else None
    try:
        answer, store = run(
            approve=approve, model=MODEL, runs_dir=work, work_dir=work,
            system=shop.SUPPORT_SYSTEM, tools=shop.SUPPORT_TOOLS,
            credentials=credentials, **kwargs)
    except ApprovalNeeded as exc:
        return {
            "status": "approval_needed",
            "tool": exc.tool,
            "args": exc.tool_args,
            "state": dict(exc.state, work=work),
            "events": _events(exc.store),
        }
    return {"status": "done", "answer": answer, "events": _events(store)}
