"""The lab: Runner -> Coroner -> Surgeon -> Arena, repeated for N generations."""
import copy
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from . import agent
from .shop import TASKS


# ---------- Runner / Arena ----------

def evaluate(spec, split, runs, client=None, model=None):
    jobs = [(t, r) for t in TASKS if t["split"] == split for r in range(runs)]
    with ThreadPoolExecutor(8 if client else 1) as pool:
        traces = list(pool.map(lambda j: agent.run(spec, j[0], client, model, seed=j[1]), jobs))
    passed = sum(not t["fails"] for t in traces)
    return {"score": passed / len(traces), "tokens": sum(t["tokens"] for t in traces) // len(traces),
            "traces": traces}


# ---------- Coroner ----------

class Cause(BaseModel):
    category: str
    root_cause: str
    fix: str


class Diagnosis(BaseModel):
    causes: list[Cause]


MOCK_CAUSES = {
    "units": ("The issue_refund description never says amount is in cents, so the agent passes dollars.",
              "State the unit (integer cents) in the issue_refund description."),
    "policy_window": ("The 30-day refund window exists only in the grader, never in the prompt.",
                      "Add the 30-day refund window to the system prompt."),
    "escalation": ("Nothing tells the agent large refunds need a human.",
                   "Require escalate() instead of issue_refund for refunds over $200."),
    "scope": ("The agent refunds whole orders when the customer names one item.",
              "Tell the agent to refund only the item the customer names."),
    "confirmation": ("The prompt never asks for a closing reply to the customer.",
                     "Require a send_message reply to end every ticket."),
    "loop": ("The agent keeps calling tools without converging.", "Add a plan-then-act instruction and a stop rule."),
}


def diagnose(traces, client=None, model=None):
    clusters = defaultdict(list)
    for t in traces:
        for reason in t["fails"]:
            clusters[reason.split(":")[0]].append((t, reason))
    if not clusters:
        return []
    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    out = [{"category": c, "count": len(v), "tasks": sorted({t["task"] for t, _ in v}),
            "evidence": v[0][1], "example": v[0][0]} for c, v in ranked]
    if client is None:
        for o in out:
            o["root_cause"], o["fix"] = MOCK_CAUSES.get(o["category"], ("Unknown.", "Inspect traces."))
        return out
    brief = [{k: o[k] for k in ("category", "count", "evidence")} | {"trace": o["example"]["steps"][-8:]} for o in out]
    r = client.messages.parse(
        model=model, max_tokens=16000, output_format=Diagnosis,
        messages=[{"role": "user", "content":
                   "You are the coroner for a failing customer-support agent. For each failure cluster below, "
                   "give the root cause in the agent's instructions or tool descriptions (not in the model) and "
                   "a one-line fix. Return one cause per category, same category names.\n\n"
                   + json.dumps(brief, default=str, indent=1)}])
    found = {c.category: c for c in r.parsed_output.causes}
    for o in out:
        c = found.get(o["category"])
        o["root_cause"], o["fix"] = (c.root_cause, c.fix) if c else ("(not analysed)", "")
    return out


# ---------- Surgeon ----------

class ToolDesc(BaseModel):
    name: str
    description: str


class Variant(BaseModel):
    strategy: str
    rationale: str
    system: str
    tools: list[ToolDesc]


class Variants(BaseModel):
    variants: list[Variant]


MOCK_PATCHES = {  # category -> (system addition, {tool: new description})
    "units": ("", {"issue_refund": "Refund part or all of an order. `amount` is an integer in CENTS "
                                   "(e.g. $18.99 -> 1899). Read prices from get_order first."}),
    "policy_window": ("Refund policy: orders older than 30 days cannot be refunded; explain this kindly.", {}),
    "escalation": ("Refunds over $200 must be escalated with escalate(), never issued directly.", {}),
    "scope": ("If the customer names a specific item, refund only that item's price, not the whole order.", {}),
    "confirmation": ("Always finish every ticket with send_message to the customer summarising what you did.", {}),
    "loop": ("Plan first, then act; stop once the request is handled.", {}),
}


def _patch(spec, cats):
    s = copy.deepcopy(spec)
    for c in cats:
        add, tools = MOCK_PATCHES.get(c, ("", {}))
        s["system"] += ("\n" + add) if add and add not in s["system"] else ""
        s["tools"].update(tools)
    return s


def operate(spec, diag, client=None, model=None):
    """Return 3 competing repaired specs."""
    if client is None:  # mock surgeon: cautious, fixes at most 1/2/3 clusters per variant
        cats = [d["category"] for d in diag]
        return [{"strategy": name, "rationale": f"patch {', '.join(cats[:k])}", "spec": _patch(spec, cats[:k])}
                for k, name in ((1, "scalpel"), (2, "targeted"), (3, "broad"))]
    brief = [{k: d[k] for k in ("category", "count", "evidence", "root_cause", "fix")} for d in diag]
    r = client.messages.parse(
        model=model, max_tokens=32000, output_format=Variants,
        messages=[{"role": "user", "content":
                   "You are the surgeon. Repair this support agent by rewriting its system prompt and tool "
                   "descriptions. Produce exactly 3 variants with genuinely different strategies (e.g. minimal "
                   "patch, policy-first rewrite, tool-description-first). Fix the root causes as general rules: "
                   "never mention specific customers, emails or order IDs (it is scored on unseen tickets). "
                   "Return every tool with its full description.\n\n"
                   f"CURRENT SPEC:\n{json.dumps(spec, indent=1)}\n\nDIAGNOSIS:\n{json.dumps(brief, indent=1)}"}])
    out = []
    for v in r.parsed_output.variants[:3]:
        s = copy.deepcopy(spec)
        s["system"] = v.system
        s["tools"].update({t.name: t.description for t in v.tools if t.name in s["tools"]})
        out.append({"strategy": v.strategy, "rationale": v.rationale, "spec": s})
    return out


# ---------- Evolution ----------

def evolve(spec, gens=3, runs=3, client=None, agent_model=None, lab_model=None, log=print):
    best = evaluate(spec, "train", runs, client, agent_model) | {"spec": spec, "strategy": "baseline"}
    history = [{"gen": 0, "best": best["score"], "tokens": best["tokens"], "candidates": []}]
    seen = {}
    log(f"gen 0  baseline            train {best['score']:.0%}")
    for g in range(1, gens + 1):
        diag = diagnose(best["traces"], client, lab_model)
        for d in diag:
            seen.setdefault(d["category"], d)
        if not diag:
            log("no failures left, stopping early")
            break
        log("       autopsy: " + ", ".join(f"{d['category']}x{d['count']}" for d in diag))
        cands = []
        for v in operate(best["spec"], diag, client, lab_model):
            res = evaluate(v["spec"], "train", runs, client, agent_model) | v
            cands.append(res)
            log(f"gen {g}  {v['strategy'][:18]:<18}  train {res['score']:.0%}  ~{res['tokens']} tok/run")
        win = max(cands, key=lambda c: (c["score"], -c["tokens"]))
        improved = (win["score"], -win["tokens"]) > (best["score"], -best["tokens"])
        best = win if improved else best
        history.append({"gen": g, "best": best["score"], "tokens": best["tokens"], "diagnosis": _slim(diag),
                        "candidates": [{"strategy": c["strategy"], "score": c["score"], "rationale": c["rationale"]}
                                       for c in cands]})
        if not improved:
            log("       no variant beat the parent, stopping (plateau)")
            break
    log("holdout (never seen by the surgeon):")
    base_h = evaluate(spec, "holdout", runs, client, agent_model)
    best_h = evaluate(best["spec"], "holdout", runs, client, agent_model)
    log(f"       baseline {base_h['score']:.0%}  ->  evolved {best_h['score']:.0%}")
    base_tr = history[0]["best"]
    return {"baseline_spec": spec, "best_spec": best["spec"], "best_strategy": best["strategy"],
            "train": {"before": base_tr, "after": best["score"]},
            "holdout": {"before": base_h["score"], "after": best_h["score"]},
            "history": history, "diagnosis": _slim(seen.values()),
            "fail_counts": Counter(r.split(":")[0] for t in base_h["traces"] for r in t["fails"]),
            "sample": _sample(base_h["traces"], best_h["traces"])}


def _slim(diag):
    return [{k: v for k, v in d.items() if k != "example"} for d in diag]


def _sample(before, after):
    """First holdout task that failed before and passes after: the headline before/after trace."""
    fixed = {t["task"]: t for t in after if not t["fails"]}
    for t in before:
        if t["fails"] and t["task"] in fixed:
            return {"before": t, "after": fixed[t["task"]]}
    return None
