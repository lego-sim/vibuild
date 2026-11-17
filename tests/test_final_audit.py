"""Final quality cannot pass from old local scores or a partial view split."""

from pathlib import Path

import pytest

from vibuild.assessment.final import audit
from vibuild.resources import save, load
from vibuild.runtime.layout import initialize_task


def test_final_audit_uses_current_fit_and_held_out_results(tmp_path, monkeypatch):
    root = tmp_path / "project"
    initialize_task(root, ["R01"])
    plan = load(root / "plan.json")
    plan["steps"][0]["state"] = "accepted"
    save(root / "plan.json", plan)
    save(root / "status.json", {"active_region": None})
    (root / "scene.blend").write_bytes(b"BLENDERcurrent")
    save(root / "partition.json", {"regions": [{"id": "R01"}], "interfaces": []})
    commands = []

    def score(args):
        commands.append(args)
        if args[0] == "run":
            output = Path(args[args.index("--output") + 1])
            save(output / "scores.json", {"passed": "held_out" not in output.name})
        return 0

    monkeypatch.setattr("vibuild.assessment.final.visual.main", score)
    result = audit(root, tmp_path / "config.json")
    assert (
        not result["passed"]
        and load(root / "status.json")["global_acceptance"] == "failed"
    )
    assert len(commands) == 4 and "--held-out" in commands[2]


def test_final_audit_rejects_deferred_or_unfinished_stages(tmp_path):
    root = tmp_path / "project"
    initialize_task(root, ["R01"])
    with pytest.raises(ValueError, match="must be accepted"):
        audit(root, tmp_path / "config.json")
