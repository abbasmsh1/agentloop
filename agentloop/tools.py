"""Tool registry with allowlist, approval flags, and a path sandbox."""

import ast
import contextvars
import inspect
import operator
import os

REGISTRY = {}

# File tools resolve paths against this instead of the process-global cwd, so
# concurrent requests on one server instance each get their own sandbox.
BASE_DIR = contextvars.ContextVar("agentloop_base_dir", default=None)

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(description, requires_approval=False):
    """Register a function as an agent tool. Schema comes from the signature."""

    def wrap(fn):
        params = {}
        for name, p in inspect.signature(fn).parameters.items():
            params[name] = {"type": _JSON_TYPES.get(p.annotation, "string")}
        REGISTRY[fn.__name__] = {
            "fn": fn,
            "requires_approval": requires_approval,
            "spec": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": list(params),
                    },
                },
            },
        }
        return fn

    return wrap


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
