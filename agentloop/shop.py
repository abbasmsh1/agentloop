"""Demo use case: customer-support tools for a small coffee-gear store.

Order state lives in orders.json inside the per-run sandbox: every run starts
from the same seed, refunds persist across an approval pause on a warm
instance, and nothing leaks between runs.
"""

import json
from datetime import date, timedelta

from .tools import _sandboxed, tool

STORE = "Aster & Pine"


def _day(n):
    return (date.today() - timedelta(days=n)).isoformat()


SEED = {
    "A-1042": {"customer": "Sara Lindqvist", "email": "sara@example.com",
               "items": ["Stagg pour-over kettle"], "total": 89.00,
               "delivered": _day(6), "refunded": 0.0, "notes": ["gift order"]},
    "A-1043": {"customer": "Sara Lindqvist", "email": "sara@example.com",
               "items": ["Filter papers x100", "Ceramic dripper"], "total": 31.50,
               "delivered": _day(21), "refunded": 0.0, "notes": []},
    "A-1044": {"customer": "Marco Duarte", "email": "marco@example.com",
               "items": ["Hand grinder"], "total": 74.00,
               "delivered": _day(58), "refunded": 0.0, "notes": []},
    "A-1045": {"customer": "Ines Fournier", "email": "ines@example.com",
               "items": ["Espresso scale"], "total": 52.00,
               "delivered": None, "refunded": 0.0, "notes": ["still in transit"]},
}

SUPPORT_SYSTEM = (
    f"You are the customer support agent for {STORE}, a small coffee-gear store. "
    f"Today is {date.today().isoformat()}. Policy: refunds only within 30 days of delivery, "
    "and never more than what remains refundable on the order. Always look an order up "
    "before acting on it. issue_refund moves real money and needs human approval: state the "
    "exact amount and reason before calling it. If policy blocks a refund, say why and what "
    "the customer can do instead. Keep answers short and concrete."
)

SUPPORT_TOOLS = ["lookup_order", "find_orders", "add_note", "issue_refund", "calculate"]

_FILE = "orders.json"


def _orders():
    path = _sandboxed(_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        _save(SEED)
        return json.loads(json.dumps(SEED))


def _save(orders):
    with open(_sandboxed(_FILE), "w", encoding="utf-8") as f:
        json.dump(orders, f)


@tool(description="Look up one order by id, e.g. 'A-1042'. Returns the order as JSON.")
def lookup_order(order_id: str) -> str:
    o = _orders().get(order_id)
    return json.dumps({order_id: o}) if o else f"error: no order {order_id}"


@tool(description="List all of a customer's orders by email address.")
def find_orders(email: str) -> str:
    found = {k: v for k, v in _orders().items() if v["email"].lower() == email.strip().lower()}
    return json.dumps(found) if found else f"no orders for {email}"


@tool(description="Append an internal note to an order's file.")
def add_note(order_id: str, note: str) -> str:
    orders = _orders()
    if order_id not in orders:
        return f"error: no order {order_id}"
    orders[order_id]["notes"].append(note)
    _save(orders)
    return f"note added to {order_id}"


@tool(description="Refund an amount in EUR to the customer's original payment method. Final.",
      requires_approval=True)
def issue_refund(order_id: str, amount: float) -> str:
    orders = _orders()
    o = orders.get(order_id)
    if not o:
        return f"error: no order {order_id}"
    amount = round(float(amount), 2)
    if amount <= 0:
        return "error: amount must be positive"
    remaining = round(o["total"] - o["refunded"], 2)
    if amount > remaining:
        return f"error: only {remaining:.2f} EUR remains refundable on {order_id}"
    o["refunded"] = round(o["refunded"] + amount, 2)
    _save(orders)
    return f"refunded {amount:.2f} EUR on {order_id} ({o['refunded']:.2f} of {o['total']:.2f} total)"
