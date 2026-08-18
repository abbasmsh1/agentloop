import http.server
import json
import threading

import pytest

from agentloop.tools import REGISTRY, WebhookURLError, register_tool, register_webhook_tool


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
