import pytest
from fastapi.testclient import TestClient

import api.index as api
from tests.conftest import answer_resp, tool_call_resp


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    api._hits.clear()
    return TestClient(api.app)


def test_token_gate(client, monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "secret")
    assert client.post("/api/run", json={"task": "hi"}).status_code == 401
    r = client.post("/api/run", json={"task": ""}, headers={"X-Demo-Token": "secret"})
    assert r.status_code == 400  # right token gets past the gate


def test_empty_and_long_task_rejected(client):
    assert client.post("/api/run", json={"task": "  "}).status_code == 400
    assert client.post("/api/run", json={"task": "x" * 2001}).status_code == 400


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 2)
    client.post("/api/run", json={"task": ""})
    client.post("/api/run", json={"task": ""})
    assert client.post("/api/run", json={"task": ""}).status_code == 429


def test_approval_roundtrip(client, fake_openai):
    responses, _ = fake_openai
    responses.append(tool_call_resp("issue_refund", '{"order_id": "A-1042", "amount": 89.0}'))
    r = client.post("/api/run", json={"task": "refund A-1042 in full"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "approval_needed"
    assert j["tool"] == "issue_refund"
    assert j["args"] == {"order_id": "A-1042", "amount": 89.0}
    assert j["state"]["work"].startswith("/tmp")

    responses.append(answer_resp("done, refunded"))
    r = client.post("/api/run", json={"state": j["state"], "decision": True})
    j2 = r.json()
    assert j2["status"] == "done"
    assert j2["answer"] == "done, refunded"
    events = {e["event"]: e for e in j2["events"]}
    assert "resume" in events
    assert "refunded 89.00 EUR" in events["tool"]["result"]


def test_resume_rejects_bad_work_dir(client, fake_openai):
    responses, _ = fake_openai
    responses.append(answer_resp("fine"))
    state = {"messages": [], "pending": [], "step": 0, "work": "/etc"}
    r = client.post("/api/run", json={"state": state, "decision": True})
    assert r.status_code == 200  # served, but from a fresh scratch dir, not /etc
    assert r.json()["status"] == "done"


def test_auto_approve_skips_pause(client, fake_openai):
    responses, _ = fake_openai
    responses += [tool_call_resp("issue_refund", '{"order_id": "A-1043", "amount": 10}'),
                  answer_resp("done")]
    r = client.post("/api/run", json={"task": "refund", "auto_approve": True})
    assert r.json()["status"] == "done"


def test_out_of_scope_tool_blocked_not_paused(client, fake_openai):
    responses, _ = fake_openai
    responses += [tool_call_resp("write_file", '{"path": "x.txt", "content": "hi"}'),
                  answer_resp("cannot do that")]
    r = client.post("/api/run", json={"task": "write a file"})
    j = r.json()
    assert j["status"] == "done"  # never pauses for a tool outside the web toolset
    assert "blocked" in [e["event"] for e in j["events"]]


def test_orders_endpoint(client):
    j = client.get("/api/orders").json()
    assert j["store"] == "Aster & Pine"
    assert "A-1042" in j["orders"]


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
