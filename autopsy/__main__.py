"""python -m autopsy [--mock] [--redteam] [--gens 3] [--runs 3]"""
import argparse
import json
import os
import pathlib
import time

from . import attack
from .lab import evolve
from .report import render
from .shop import TASKS


def main():
    p = argparse.ArgumentParser(prog="autopsy", description="Diagnose, repair and evolve a failing tool-using agent.")
    p.add_argument("--target", default=str(pathlib.Path(__file__).parent.parent / "targets" / "shopdesk.json"))
    p.add_argument("--gens", type=int, default=3)
    p.add_argument("--runs", type=int, default=3, help="runs per task (agents are stochastic)")
    p.add_argument("--agent-model", default="claude-haiku-4-5", help="the agent under test")
    p.add_argument("--lab-model", default="claude-opus-5-5", help="coroner + surgeon")
    p.add_argument("--mock", action="store_true", help="offline demo, no API key needed")
    p.add_argument("--redteam", action="store_true", help="add attacker agents that write hostile tickets every round")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    client = None
    if not a.mock:
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            p.error("no ANTHROPIC_API_KEY set. Export one, or run with --mock for the offline demo.")
        import anthropic
        client = anthropic.Anthropic()

    spec = json.loads(pathlib.Path(a.target).read_text(encoding="utf-8"))
    mode = ("mock" if a.mock else f"live · agent {a.agent_model} · lab {a.lab_model}") + (" · red team" if a.redteam else "")
    print(f"\n  AGENT AUTOPSY LAB  ·  {spec.get('name', a.target)}  ·  {mode}\n")
    train = [t for t in TASKS if t["split"] == "train"]
    holdout = [t for t in TASKS if t["split"] == "holdout"]
    attacker = None
    if a.redteam:
        holdout += attack.generate(attack.HOLDOUT_STYLES, "xh", client, a.lab_model, spec["system"])
        attacker = lambda g, cur: attack.generate(attack.TRAIN_STYLES, "xt", client, a.lab_model, cur["system"], g)
    res = evolve(spec, train, holdout, a.gens, a.runs, client, a.agent_model, a.lab_model,
                 log=lambda s: print("  " + s), attacker=attacker)

    out = pathlib.Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_spec.json").write_text(json.dumps(res["best_spec"], indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "report.html").write_text(render(res, mode), encoding="utf-8")
    print(f"\n  report: {out / 'report.html'}\n")


if __name__ == "__main__":
    main()
