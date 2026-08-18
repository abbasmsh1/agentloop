"""Shared fake OpenAI client: scripted responses, no network."""

from types import SimpleNamespace

import openai
import pytest


def tool_call_resp(name, arguments, tc_id="tc1"):
    tc = SimpleNamespace(model_dump=lambda: {
        "id": tc_id, "type": "function",
        "function": {"name": name, "arguments": arguments},
    })
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def answer_resp(text):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


@pytest.fixture
def fake_openai(monkeypatch):
    """Patch openai.OpenAI with a client that replays a scripted response list.

    Returns (responses, calls): append responses before running; calls records
    every kwargs dict passed to chat.completions.create.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses, calls = [], []

    def create(**kw):
        calls.append(kw)
        return responses.pop(0)

    class Client:
        def __init__(self, **kw):
            calls.append({"client_kwargs": kw})
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", Client)
    return responses, calls
