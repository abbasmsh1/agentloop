from agentloop.stores import JsonlRunStore, MemoryRunStore


def test_jsonl_store_roundtrip(tmp_path):
    s = JsonlRunStore(str(tmp_path / "r.jsonl"))
    s.append({"event": "task"})
    s.append({"event": "answer"})
    assert [e["event"] for e in s.load()] == ["task", "answer"]
    assert s.name == "r.jsonl"


def test_memory_store_roundtrip():
    s = MemoryRunStore()
    s.append({"event": "task"})
    assert s.load() == [{"event": "task"}]
    assert s.name == "memory"


def test_missing_file_loads_empty(tmp_path):
    assert JsonlRunStore(str(tmp_path / "nope.jsonl")).load() == []
