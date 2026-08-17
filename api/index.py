"""Vercel serverless entry: web chat UI over the agent loop.

Serverless FS is read-only outside /tmp. Every request gets its own scratch
directory for file tools and its own runs dir, so nothing leaks between
requests or users sharing a warm instance.
"""
import json
import hmac
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agentloop.agent import run  # noqa: E402

app = FastAPI(title="agentloop")

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")


def check_token(x_demo_token):
    """If DEMO_TOKEN is set, require it - keeps a public demo from burning your API key."""
    required = os.environ.get("DEMO_TOKEN")
    if required and not hmac.compare_digest(x_demo_token or "", required):
        raise HTTPException(401, "missing or wrong X-Demo-Token header")


class RunRequest(BaseModel):
    task: str
    auto_approve: bool = False


@app.get("/", response_class=HTMLResponse)
def ui():
    with open(_WEB, encoding="utf-8") as f:
        return f.read()


@app.post("/api/run")
def run_task(req: RunRequest, x_demo_token: str | None = Header(default=None)):
    check_token(x_demo_token)
    task = req.task.strip()
    if not task:
        raise HTTPException(400, "task is empty")
    if len(task) > 2000:
        raise HTTPException(400, "task too long")
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY not configured on the server")

    # per-request sandbox: file tools and run log are isolated to this request
    work = tempfile.mkdtemp(prefix="agent-")
    # ponytail: os.chdir is process-global; fine for one-request-per-instance
    # serverless, use per-request sandbox paths in tools if running multi-threaded
    os.chdir(work)

    approve = (lambda name, args: True) if req.auto_approve else (lambda name, args: False)
    answer, run_file = run(task, approve=approve, runs_dir=work)

    with open(run_file, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    return {"answer": answer, "events": events}
