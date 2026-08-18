import pytest

from agentloop.shop import add_note, find_orders, issue_refund, lookup_order
from agentloop.tools import REGISTRY


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_refund_needs_approval_flag():
    assert REGISTRY["issue_refund"]["requires_approval"] is True
    assert REGISTRY["add_note"]["requires_approval"] is False


def test_lookup_and_find():
    assert "Stagg pour-over kettle" in lookup_order("A-1042")
    assert "error: no order" in lookup_order("A-9999")
    found = find_orders("sara@example.com")
    assert "A-1042" in found and "A-1043" in found and "A-1044" not in found


def test_refund_happy_path_and_limits():
    assert "refunded 89.00 EUR" in issue_refund("A-1042", 89.0)
    assert "only 0.00 EUR remains" in issue_refund("A-1042", 1.0)
    assert "error: amount must be positive" in issue_refund("A-1043", -5)
    assert "error: no order" in issue_refund("A-9999", 10)


def test_partial_refunds_accumulate():
    assert "refunded 20.00 EUR" in issue_refund("A-1044", 20)
    assert "20.00 of 74.00" in lookup_order("A-1044") or "20.0" in lookup_order("A-1044")
    assert "only 54.00 EUR remains" in issue_refund("A-1044", 60)


def test_note_persists():
    add_note("A-1043", "customer called")
    assert "customer called" in lookup_order("A-1043")
    assert "error: no order" in add_note("A-9999", "x")


def test_runs_are_isolated(tmp_path, monkeypatch):
    issue_refund("A-1042", 89.0)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)  # a different sandbox starts from the seed again
    assert '"refunded": 0.0' in lookup_order("A-1042")
