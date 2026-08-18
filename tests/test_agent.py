import pytest

from agentloop.agent import MAX_STEPS, ApprovalNeeded, run
from tests.conftest import answer_resp, tool_call_resp


def test_tool_result_fed_back_to_model(fake_openai, tmp_path):
    responses, calls = fake_openai
    responses += [tool_call_resp("calculate", '{"expression": "(17*43)+12"}'),
                  answer_resp("It is 743.")]
    answer, store = run("math", runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "It is 743."
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "tc1", "content": "743"}]
    events = [e["event"] for e in store.load()]
    assert events == ["task", "model", "tool", "model", "answer"]


def test_pause_then_approve_resumes(fake_openai, tmp_path):
    responses, _ = fake_openai
    responses.append(tool_call_resp("write_file", '{"path": "x.txt", "content": "hi"}'))
    with pytest.raises(ApprovalNeeded) as exc_info:
        run("write it", approve=None, runs_dir=str(tmp_path), work_dir=str(tmp_path))
    exc = exc_info.value
    assert exc.tool == "write_file"
    assert exc.tool_args == {"path": "x.txt", "content": "hi"}
    assert not (tmp_path / "x.txt").exists()

    responses.append(answer_resp("done"))
    answer, store = run(state=exc.state, decision=True,
                        runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "done"
    assert (tmp_path / "x.txt").read_text() == "hi"
    events = [e["event"] for e in store.load()]
    assert "resume" in events  # same log file spans both phases
    assert events[0] == "task"


def test_pause_then_deny_blocks_write(fake_openai, tmp_path):
    responses, calls = fake_openai
    responses.append(tool_call_resp("write_file", '{"path": "x.txt", "content": "hi"}'))
    with pytest.raises(ApprovalNeeded) as exc_info:
        run("write it", approve=None, runs_dir=str(tmp_path), work_dir=str(tmp_path))

    responses.append(answer_resp("ok, not writing"))
    answer, _ = run(state=exc_info.value.state, decision=False,
                    runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "ok, not writing"
    assert not (tmp_path / "x.txt").exists()
    tool_msgs = [m for m in calls[-1]["messages"] if m["role"] == "tool"]
    assert "denied" in tool_msgs[0]["content"]


def test_step_budget_aborts(fake_openai, tmp_path):
    responses, _ = fake_openai
    responses += [tool_call_resp("calculate", '{"expression": "1+1"}')] * MAX_STEPS
    answer, _ = run("loop forever", runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert "step budget" in answer


def test_unknown_tool_reported_to_model(fake_openai, tmp_path):
    responses, calls = fake_openai
    responses += [tool_call_resp("drop_tables", "{}"), answer_resp("cannot")]
    answer, _ = run("hack", runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "cannot"
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert "does not exist" in tool_msgs[0]["content"]


def test_require_approval_override_pauses_ungated_tool(fake_openai, tmp_path):
    responses, _ = fake_openai
    responses.append(tool_call_resp("calculate", '{"expression": "1+1"}'))
    with pytest.raises(ApprovalNeeded) as exc_info:
        run("calc", approve=None, require_approval=["calculate"],
            runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert exc_info.value.tool == "calculate"

    responses.append(answer_resp("2"))
    answer, _ = run(state=exc_info.value.state, decision=True,
                    require_approval=["calculate"],
                    runs_dir=str(tmp_path), work_dir=str(tmp_path))
    assert answer == "2"
