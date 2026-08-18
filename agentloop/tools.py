"""Tool registry with allowlist, approval flags, and a path sandbox."""

import ast
import contextvars
import http.client
import inspect
import ipaddress
import json
import operator
import os
import socket
import ssl
from urllib.parse import urlparse

REGISTRY = {}

# File tools resolve paths against this instead of the process-global cwd, so
# concurrent requests on one server instance each get their own sandbox.
BASE_DIR = contextvars.ContextVar("agentloop_base_dir", default=None)

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


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
    """Bring-your-own-tool over HTTP: args POSTed as JSON, response text returned.

    URL is validated at registration time (fail-fast). At call time, the URL is
    re-validated and connection is pinned to the validated IP (prevents DNS rebinding).
    Redirects are never followed.
    """
    _check_webhook_url(url, allow_private)

    def call(**kwargs):
        return _webhook_post(url, kwargs, timeout, allow_private)

    register_tool(name, description, call, params, requires_approval)


def _check_webhook_url(url, allow_private):
    """Validate webhook URL and return the first resolved IP (re-validated at call time).

    Raises WebhookURLError on bad scheme, hostname, resolution, or non-global address
    (unless allow_private=True).
    """
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise WebhookURLError(f"webhook url must be http(s): {url}")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or 80, proto=socket.IPPROTO_TCP)
    except OSError as e:
        raise WebhookURLError(f"cannot resolve {p.hostname}: {e}") from e
    if not infos:
        raise WebhookURLError(f"webhook host resolution returned no addresses: {p.hostname}")
    ip_str = infos[0][4][0]
    ip = ipaddress.ip_address(ip_str)
    if not allow_private and not ip.is_global:
        raise WebhookURLError(f"webhook host resolves to non-public address {ip}")
    return ip_str


def _webhook_post(url, payload, timeout, allow_private):
    """POST JSON to a webhook. Re-validates the URL at call time, connects to
    the validated IP (no DNS-rebinding window), and never follows redirects."""
    p = urlparse(url)
    ip = _check_webhook_url(url, allow_private)
    port = p.port or (443 if p.scheme == "https" else 80)
    if p.scheme == "https":
        conn = _PinnedHTTPSConnection(ip, port, p.hostname, timeout)
    else:
        conn = http.client.HTTPConnection(ip, port, timeout=timeout)
    path = (p.path or "/") + (f"?{p.query}" if p.query else "")
    try:
        conn.request("POST", path, body=json.dumps(payload).encode(),
                     headers={"Content-Type": "application/json", "Host": p.netloc})
        r = conn.getresponse()
        if 300 <= r.status < 400:
            return f"error: webhook redirected ({r.status}); redirects are not followed"
        body = r.read(65536).decode("utf-8", "replace")[:8000]
        if r.status >= 400:
            return f"error: webhook returned {r.status}: {body[:200]}"
        return body
    finally:
        conn.close()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects to a pre-validated IP while keeping TLS SNI and certificate
    checks against the original hostname."""

    def __init__(self, ip, port, hostname, timeout):
        super().__init__(ip, port, timeout=timeout, context=ssl.create_default_context())
        self._tls_hostname = hostname

    def connect(self):
        sock = socket.create_connection((self.host, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self._tls_hostname)


def openai_specs(names=None):
    return [t["spec"] for n, t in REGISTRY.items() if names is None or n in names]


def anthropic_specs(names=None):
    return [{"name": n,
             "description": t["spec"]["function"]["description"],
             "input_schema": t["spec"]["function"]["parameters"]}
            for n, t in REGISTRY.items() if names is None or n in names]


# ---- built-in tools ----

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@tool(description="Evaluate an arithmetic expression, e.g. '(2+3)*4'.")
def calculate(expression: str) -> str:
    result = _safe_eval(ast.parse(expression, mode="eval").body)
    return str(result)


@tool(description="Read a UTF-8 text file inside the working directory.")
def read_file(path: str) -> str:
    full = _sandboxed(path)
    with open(full, encoding="utf-8") as f:
        return f.read()[:8000]


@tool(description="Write text to a file inside the working directory.", requires_approval=True)
def write_file(path: str, content: str) -> str:
    full = _sandboxed(path)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


def _sandboxed(path):
    """Resolve path inside the sandbox base dir; refuse anything that escapes it."""
    base = os.path.realpath(BASE_DIR.get() or os.getcwd())
    full = os.path.realpath(os.path.join(base, path))
    if not full.startswith(base + os.sep) and full != base:
        raise PermissionError(f"path escapes working directory: {path}")
    return full
