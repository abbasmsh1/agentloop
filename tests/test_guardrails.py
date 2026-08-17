import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentloop.agent import _execute
from agentloop.tools import REGISTRY, calculate, _sandboxed


def test_calculator():
    assert calculate("(17 * 43) + 12") == "743"
    assert calculate("-5 + 2") == "-3"
    with pytest.raises(ValueError):
        calculate("__import__('os').system('id')")


def test_sandbox_blocks_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _sandboxed("ok.txt").startswith(str(tmp_path))
    with pytest.raises(PermissionError):
        _sandboxed("../outside.txt")
    with pytest.raises(PermissionError):
        _sandboxed("/etc/passwd")


def test_allowlist_blocks_unknown_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = _execute("delete_database", {}, approve=lambda n, a: True, run_file="log.jsonl")
    assert "does not exist" in out


def test_approval_gate_denies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert REGISTRY["write_file"]["requires_approval"] is True
    out = _execute("write_file", {"path": "x.txt", "content": "hi"},
                   approve=lambda n, a: False, run_file="log.jsonl")
    assert "denied" in out
    assert not os.path.exists(tmp_path / "x.txt")


def test_approval_gate_allows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = _execute("write_file", {"path": "x.txt", "content": "hi"},
                   approve=lambda n, a: True, run_file="log.jsonl")
    assert "wrote" in out
    assert (tmp_path / "x.txt").read_text() == "hi"
