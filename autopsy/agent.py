"""Runs the agent under test on one task and returns a full trace."""
import json
import random
import re

from .shop import Shop, check, total

MAX_STEPS = 12


def tool_defs(spec):
    from .shop import TOOLS
    return [{"name": n, "description": spec["tools"][n], "input_schema": s} for n, s in TOOLS.items()]


def run(spec, task, client=None, model=None, seed=0):
    shop = Shop()
    trace = {"task": task["id"], "msg": task["msg"], "steps": [], "final": "", "tokens": 0}
    if client is None:
        _mock(spec, task, shop, trace, seed)
    else:
        _real(spec, task, shop, trace, client, model)
    trace["fails"] = trace.pop("fatal", []) + check(task, shop)
    return trace


def _real(spec, task, shop, trace, client, model):
    import anthropic
    messages = [{"role": "user", "content": task["msg"]}]
    for _ in range(MAX_STEPS):
        try:
            r = client.messages.create(model=model, max_tokens=2048, system=spec["system"],
                                       tools=tool_defs(spec), messages=messages)
        except anthropic.APIError as e:
            trace["fatal"] = [f"api_error: {e}"]
            return
        trace["tokens"] += r.usage.input_tokens + r.usage.output_tokens
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            trace["final"] = "".join(b.text for b in r.content if b.type == "text")
            return
        results = []
        for b in r.content:
            if b.type == "tool_use":
                out = shop.call(b.name, b.input)
                trace["steps"].append({"tool": b.name, "input": b.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out),
                                "is_error": isinstance(out, str) and out.startswith("ERROR")})
        messages.append({"role": "user", "content": results})
    trace["fatal"] = ["loop: hit the step limit without finishing"]


def _mock(spec, task, shop, trace, seed):
    """Offline stand-in for the LLM: a scripted agent whose blind spots depend on what its
    prompt and tool descriptions actually say. Lets the whole pipeline run with no API key."""
    sysp, refund_desc = spec["system"].lower(), spec["tools"]["issue_refund"].lower()
    rng = random.Random(f"{seed}|{task['id']}|{sysp}|{refund_desc}")

    def do(name, **args):
        out = shop.call(name, args)
        trace["steps"].append({"tool": name, "input": args, "output": out})
        return out

    msg = task["msg"]
    cust = do("find_customer", email=re.search(r"\S+@\S+?\.com", msg).group())
    oid = re.search(r"\b[A-E]\d{3}\b", msg).group()
    order = do("get_order", order_id=oid)
    trace["tokens"] = 900 + len(sysp) // 3 + rng.randint(0, 300)

    injected = re.search(r"authoriz|override|urgent|entitled", msg, re.I) and "never instructions" not in sysp
    if order["customer"] != cust["customer_id"] and "belong" in sysp:
        pass  # refuses to touch another customer's order
    elif "address" in msg:
        do("update_address", order_id=oid, address=msg.split(" to ", 1)[1].rstrip("."))
    else:
        items = [i for i in order["items"] if i["name"].split()[-1].lower().rstrip("s") in msg.lower()]
        partial = items and not re.search(r"whole|entire", msg)
        if partial and "item" not in sysp + refund_desc:
            partial = False  # blind spot: refunds the full order
        amount = items[0]["price_cents"] if partial else total(order)
        too_old = order["days_ago"] > 30 and "30" in sysp and not injected
        too_big = amount > 20000 and "escalat" in sysp and not injected
        if too_big:
            do("escalate", order_id=oid, reason="refund over $200")
        elif not too_old:
            do("issue_refund", order_id=oid, amount=amount if "cent" in refund_desc else amount / 100)
    if "send_message" in sysp and rng.random() > 0.06:
        do("send_message", customer_id=cust["customer_id"], text="Done. Anything else?")
    trace["final"] = "Handled."
