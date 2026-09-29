import json
import pathlib

from autopsy.lab import evolve
from autopsy.shop import TASKS, Shop, check

SPEC = json.loads((pathlib.Path(__file__).parent.parent / "targets" / "shopdesk.json").read_text())


def test_grader_explains_policy():
    s = Shop()
    s.issue_refund("A100", 18.99)  # dollars instead of cents
    assert any(r.startswith("units:") for r in check(TASKS[0], s))
    s = Shop()
    s.issue_refund("A100", 1899)
    s.send_message("C1", "done")
    assert check(TASKS[0], s) == []


def test_lab_repairs_agent_and_generalises():
    res = evolve(SPEC, gens=4, runs=2, log=lambda _: None)
    assert res["train"]["after"] > res["train"]["before"] + 0.5
    assert res["holdout"]["after"] > res["holdout"]["before"] + 0.5
    assert "cents" in res["best_spec"]["tools"]["issue_refund"].lower()
