from types import SimpleNamespace

import pytest


def a_tool_call(name, input_dict, tc_id="tu1"):
    blocks = [SimpleNamespace(type="tool_use", id=tc_id, name=name, input=input_dict)]
    return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=20, output_tokens=8))


def a_answer(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=20, output_tokens=8))


@pytest.fixture
def fake_anthropic(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    responses, calls = [], []

    def create(**kw):
        calls.append(kw)
        return responses.pop(0)

    class Client:
        def __init__(self, **kw):
            calls.append({"client_kwargs": kw})
            self.messages = SimpleNamespace(create=create)

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    return responses, calls


def test_claude_tool_loop(fake_anthropic, tmp_path, monkeypatch):
    from agentloop.agent import run
    responses, calls = fake_anthropic
    responses += [a_tool_call("calculate", {"expression": "2+2"}),
                  a_answer("It is 4.")]
    answer, _ = run("math", model="claude-opus-5",
                    runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "It is 4."
    second = calls[-1]  # request kwargs of the follow-up call
    assert second["model"] == "claude-opus-5"
    # tool result went back in Anthropic format
    tool_results = [b for m in second["messages"] if isinstance(m["content"], list)
                    for b in m["content"] if b.get("type") == "tool_result"]
    assert tool_results == [{"type": "tool_result", "tool_use_id": "tu1", "content": "4"}]
    # assistant turn carries the tool_use block
    tool_uses = [b for m in second["messages"] if m["role"] == "assistant"
                 for b in m["content"] if b["type"] == "tool_use"]
    assert tool_uses[0]["name"] == "calculate"
    assert second["system"].startswith("You are a careful assistant")


def test_claude_uses_subscription_token(fake_anthropic, monkeypatch, tmp_path):
    from agentloop.agent import run
    responses, calls = fake_anthropic
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-test")
    responses.append(a_answer("hi"))
    run("hello", model="claude-opus-5", runs_dir=str(tmp_path), work_dir=str(tmp_path))
    ckw = calls[0]["client_kwargs"]
    assert ckw["auth_token"] == "sk-ant-oat-test"
    assert ckw["default_headers"] == {"anthropic-beta": "oauth-2025-04-20"}


def test_credentials_override_env(fake_anthropic, monkeypatch, tmp_path):
    from agentloop.agent import run
    responses, calls = fake_anthropic
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-should-lose")
    responses.append(a_answer("hi"))
    run("hello", model="claude-opus-5", credentials={"api_key": "sk-byok"},
        runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert calls[0]["client_kwargs"]["api_key"] == "sk-byok"
    assert "auth_token" not in calls[0]["client_kwargs"]


def test_parallel_tool_results_merge_into_one_user_message(fake_anthropic, tmp_path):
    from agentloop.agent import run
    responses, calls = fake_anthropic
    blocks = [SimpleNamespace(type="tool_use", id="tu1", name="calculate", input={"expression": "1+1"}),
              SimpleNamespace(type="tool_use", id="tu2", name="calculate", input={"expression": "2+2"})]
    responses.append(SimpleNamespace(content=blocks,
                                     usage=SimpleNamespace(input_tokens=20, output_tokens=8)))
    responses.append(a_answer("4 and 2"))
    run("math", model="claude-opus-5", runs_dir=str(tmp_path), work_dir=str(tmp_path))
    second = calls[-1]
    result_msgs = [m for m in second["messages"]
                   if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(result_msgs) == 1
    assert [b["tool_use_id"] for b in result_msgs[0]["content"]] == ["tu1", "tu2"]
