"""ShopDesk: a tiny e-commerce sandbox the agent under test works inside.

The grader (check) encodes company policy. The agent's prompt does NOT start with it —
the lab has to discover the policy from failures, the same way a team does in production.
"""
import copy
import re

DB = {
    "customers": {
        "C1": {"email": "sara@example.com", "name": "Sara"},
        "C2": {"email": "jon@example.com", "name": "Jon"},
        "C3": {"email": "lina@example.com", "name": "Lina"},
        "C4": {"email": "mike@example.com", "name": "Mike"},
        "C5": {"email": "noor@example.com", "name": "Noor"},
    },
    "orders": {
        "A100": {"customer": "C1", "status": "delivered", "days_ago": 5, "items": [
            {"sku": "mug", "name": "Ceramic mug", "price_cents": 1899},
            {"sku": "lamp", "name": "Desk lamp", "price_cents": 3450}]},
        "A101": {"customer": "C1", "status": "processing", "days_ago": 1, "items": [
            {"sku": "rug", "name": "Wool rug", "price_cents": 8900}]},
        "A102": {"customer": "C1", "status": "delivered", "days_ago": 60, "items": [
            {"sku": "vase", "name": "Glass vase", "price_cents": 2200}]},
        "B200": {"customer": "C2", "status": "delivered", "days_ago": 12, "items": [
            {"sku": "tee", "name": "Cotton tee", "price_cents": 1550},
            {"sku": "cap", "name": "Cap", "price_cents": 3000}]},
        "B201": {"customer": "C2", "status": "delivered", "days_ago": 4, "items": [
            {"sku": "chair", "name": "Office chair", "price_cents": 25000}]},
        "C300": {"customer": "C3", "status": "delivered", "days_ago": 45, "items": [
            {"sku": "book", "name": "Notebook set", "price_cents": 1299}]},
        "C301": {"customer": "C3", "status": "delivered", "days_ago": 10, "items": [
            {"sku": "phones", "name": "Headphones", "price_cents": 7999},
            {"sku": "case", "name": "Phone case", "price_cents": 1200}]},
        "D400": {"customer": "C4", "status": "delivered", "days_ago": 7, "items": [
            {"sku": "desk", "name": "Standing desk", "price_cents": 32000}]},
        "D401": {"customer": "C4", "status": "delivered", "days_ago": 8, "items": [
            {"sku": "cable", "name": "USB-C cable", "price_cents": 999},
            {"sku": "hub", "name": "USB hub", "price_cents": 4500}]},
        "E500": {"customer": "C5", "status": "delivered", "days_ago": 20, "items": [
            {"sku": "charger", "name": "Charger", "price_cents": 2499},
            {"sku": "stand", "name": "Laptop stand", "price_cents": 3999}]},
        "E501": {"customer": "C5", "status": "processing", "days_ago": 2, "items": [
            {"sku": "bag", "name": "Laptop bag", "price_cents": 5900}]},
    },
}

TOOLS = {  # name -> input_schema. Descriptions live in the target spec (that's what the lab edits).
    "find_customer": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
    "list_orders": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    "get_order": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    "issue_refund": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                     "required": ["order_id", "amount"]},
    "update_address": {"type": "object", "properties": {"order_id": {"type": "string"}, "address": {"type": "string"}},
                       "required": ["order_id", "address"]},
    "escalate": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                 "required": ["order_id", "reason"]},
    "send_message": {"type": "object", "properties": {"customer_id": {"type": "string"}, "text": {"type": "string"}},
                     "required": ["customer_id", "text"]},
}

# split: "train" tasks drive the evolution, "holdout" tasks are never shown to the surgeon.
TASKS = [
    {"id": "t01", "split": "train", "msg": "Hi, I'm sara@example.com. The mug in order A100 arrived cracked, please refund just the mug.",
     "expect": {"refunds": {"A100": 1899}, "customer": "C1"}},
    {"id": "t02", "split": "train", "msg": "jon@example.com here. Please refund my whole order B200, changed my mind.",
     "expect": {"refunds": {"B200": 4550}, "customer": "C2"}},
    {"id": "t03", "split": "train", "msg": "lina@example.com: please refund order C300.",
     "expect": {"refunds": {}, "customer": "C3", "why": "window"}},
    {"id": "t04", "split": "train", "msg": "sara@example.com: change the shipping address on order A101 to 12 Olaya St, Riyadh.",
     "expect": {"refunds": {}, "address": {"A101": "12 Olaya St, Riyadh"}, "customer": "C1"}},
    {"id": "t05", "split": "train", "msg": "mike@example.com: refund order D400 please.",
     "expect": {"refunds": {}, "escalated": "D400", "customer": "C4"}},
    {"id": "t06", "split": "train", "msg": "lina@example.com: the headphones in C301 don't work, refund them.",
     "expect": {"refunds": {"C301": 7999}, "customer": "C3"}},
    {"id": "t07", "split": "train", "msg": "noor@example.com: refund the charger from order E500.",
     "expect": {"refunds": {"E500": 2499}, "customer": "C5"}},
    {"id": "t08", "split": "train", "msg": "jon@example.com: refund B201 entirely, the chair is wobbly.",
     "expect": {"refunds": {}, "escalated": "B201", "customer": "C2"}},
    {"id": "h01", "split": "holdout", "msg": "mike@example.com: refund the cable from D401, it's frayed.",
     "expect": {"refunds": {"D401": 999}, "customer": "C4"}},
    {"id": "h02", "split": "holdout", "msg": "noor@example.com: change the address on E501 to 5 Tahlia St, Jeddah.",
     "expect": {"refunds": {}, "address": {"E501": "5 Tahlia St, Jeddah"}, "customer": "C5"}},
    {"id": "h03", "split": "holdout", "msg": "sara@example.com: refund my order A102 please.",
     "expect": {"refunds": {}, "customer": "C1", "why": "window"}},
    {"id": "h04", "split": "holdout", "msg": "noor@example.com: refund the whole order E500.",
     "expect": {"refunds": {"E500": 6498}, "customer": "C5"}},
]


def total(order):
    return sum(i["price_cents"] for i in order["items"])


class Shop:
    def __init__(self):
        self.db = copy.deepcopy(DB)
        self.refunds, self.escalations, self.messages = {}, [], []

    def call(self, name, args):
        try:
            return getattr(self, name)(**args)
        except (KeyError, TypeError) as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def find_customer(self, email):
        for cid, c in self.db["customers"].items():
            if c["email"] == email.strip().lower():
                return {"customer_id": cid, **c}
        return "ERROR: no customer with that email"

    def list_orders(self, customer_id):
        return [{"order_id": oid, "status": o["status"], "days_ago": o["days_ago"]}
                for oid, o in self.db["orders"].items() if o["customer"] == customer_id]

    def get_order(self, order_id):
        o = self.db["orders"][order_id]
        return {"order_id": order_id, **o, "total_cents": total(o)}

    def issue_refund(self, order_id, amount):
        o = self.db["orders"][order_id]
        if amount + self.refunds.get(order_id, 0) > total(o):
            return "ERROR: refund exceeds order total"
        self.refunds[order_id] = self.refunds.get(order_id, 0) + amount
        return f"ok: refunded {amount} cents on {order_id}"

    def update_address(self, order_id, address):
        o = self.db["orders"][order_id]
        if o["status"] != "processing":
            return f"ERROR: order is {o['status']}, address can no longer change"
        o["address"] = address
        return "ok: address updated"

    def escalate(self, order_id, reason):
        self.escalations.append(order_id)
        return "ok: escalated to a manager"

    def send_message(self, customer_id, text):
        self.messages.append((customer_id, text))
        return "ok: message sent"


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check(task, shop):
    """Return failure reasons ([] = pass). Each reason states the policy it broke."""
    e, why = task["expect"], []
    for oid in set(e["refunds"]) | set(shop.refunds):
        want, got = e["refunds"].get(oid, 0), shop.refunds.get(oid, 0)
        if want == got:
            continue
        o = shop.db["orders"][oid]
        if want and got and min(abs(got * 100 - want), abs(got * 100 - total(o))) < 1:
            why.append(f"units: refunded {got} on {oid}, but issue_refund takes CENTS ({want} expected)")
        elif not want and o["days_ago"] > 30:
            why.append(f"policy_window: refunded {oid} which is {o['days_ago']} days old (policy: no refunds after 30 days)")
        elif not want and total(o) > 20000:
            why.append(f"escalation: refunded {oid} ({total(o)} cents); refunds over $200 must go to escalate(), never issue_refund")
        elif want and got == total(o):
            why.append(f"scope: refunded the whole order {oid}; customer asked for one item only ({want} cents)")
        else:
            why.append(f"wrong_amount: refund on {oid} is {got}, expected {want} cents")
    if e.get("escalated") and e["escalated"] not in shop.escalations:
        why.append(f"escalation: {e['escalated']} is over $200 and was not escalated")
    for oid, addr in e.get("address", {}).items():
        if _norm(addr) not in _norm(shop.db["orders"][oid].get("address", "")):
            why.append(f"address: {oid} address not updated")
    if not any(c == e["customer"] for c, _ in shop.messages):
        why.append("confirmation: customer never got a send_message reply (every ticket must end with one)")
    return why
