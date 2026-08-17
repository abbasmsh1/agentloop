"""Vercel serverless entry: web chat UI over the agent loop.

Serverless FS is read-only outside /tmp, so runs are logged to /tmp/runs and
the agent's file tools operate in a per-request /tmp sandbox.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RUNS_DIR", "/tmp/runs")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agentloop.agent import run  # noqa: E402

app = FastAPI(title="agentloop")

_WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")


class RunRequest(BaseModel):
    task: str
    auto_approve: bool = False


@app.get("/", response_class=HTMLResponse)
def ui():
    with open(_WEB, encoding="utf-8") as f:
        return f.read()


@app.post("/api/run")
def run_task(req: RunRequest):
    task = req.task.strip()
    if not task:
        raise HTTPException(400, "task is empty")
    if len(task) > 2000:
        raise HTTPException(400, "task too long")
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY not configured on the server")

    # file tools sandbox to cwd: point cwd at a scratch dir on serverless
    work = "/tmp/agent-work"
    os.makedirs(work, exist_ok=True)
    os.chdir(work)

    approve = (lambda name, args: True) if req.auto_approve else (lambda name, args: False)
    answer = run(task, approve=approve)

    events = []
    files = sorted(glob.glob(os.path.join(os.environ["RUNS_DIR"], "*.jsonl")))
    if files:
        with open(files[-1], encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
    return {"answer": answer, "events": events}
