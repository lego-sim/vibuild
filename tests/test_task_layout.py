"""Dated global tasks separate authoritative scene state from regional evidence."""

import json

import pytest

from vibuild.runtime.layout import initialize_task, task_paths
from vibuild.runtime.monitor import main as monitor_main, summary


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_task_paths_and_initial_plan(tmp_path):
    root = tmp_path / "modeling_20261109_175519"
    initialize_task(root, ["R11", "R02", "L01"])
    paths = task_paths(root, "R02", 8)
    assert paths["scene"] == root / "scene.blend"
    assert paths["iteration"] == root / "R02/iterations/008"
    assert paths["score"] == root / "R02/scores/008"
    assert json.loads(paths["plan"].read_text())["execution_order"] == [
        "R11",
        "R02",
        "L01",
    ]
    assert all((root / step / ".runtime").is_dir() for step in ["R11", "R02", "L01"])
    with pytest.raises(ValueError, match="already exists"):
        initialize_task(root, ["R11", "R02", "L01"])
    with pytest.raises(ValueError, match="Unknown"):
        task_paths(root, "../R11")
    with pytest.raises(ValueError, match="positive"):
        task_paths(root, "R11", 0)


def test_monitor_reports_all_steps_and_cannot_complete_task(tmp_path):
    root = tmp_path / "modeling_20261109_175519"
    initialize_task(root, ["R11", "R02", "L01"])
    put(
        root / "status.json",
        {"state": "reopened_for_R11_seam_repair", "execution_paused": True},
    )
    result = summary(root)
    assert len(result["plan"]["steps"]) == 3
    assert result["plan"]["global_acceptance"] == "not_assessed"
    with pytest.raises(ValueError, match="paused"):
        monitor_main(root, ["run", "tool_execution", "R02", "1", "test", "--", "false"])
    with pytest.raises(ValueError, match="cannot grant acceptance"):
        monitor_main(root, ["mark", "completed", "R11", "7", "done"])


@pytest.mark.parametrize(
    "code,command,expected",
    [
        (0, ["iterate"], "inspecting_result"),
        (1, ["iterate"], "awaiting_error_repair"),
        (2, ["joints"], "check_failed"),
        (2, ["iterate"], "awaiting_error_repair"),
    ],
)
def test_monitor_distinguishes_failed_check_from_tool_error(
    tmp_path, monkeypatch, code, command, expected
):
    root = tmp_path / "modeling_20261109_175519"
    initialize_task(root, ["R11", "R02", "L01"])
    monkeypatch.setattr("vibuild.runtime.monitor.subprocess.call", lambda cmd: code)
    assert (
        monitor_main(
            root,
            ["run", "tool_execution", "R11", "11", "contact repair", "--", *command],
        )
        == code
    )
    state = json.loads((root / "status.json").read_text())
    assert state["agents"]["builder"]["state"] == expected
    assert ("Completed:" in state["current_task"]) == (code == 0)


def test_monitor_preserves_unknown_interrupted_elapsed(tmp_path):
    root = tmp_path / "modeling_20261109_175519"
    initialize_task(root, ["R11", "R02", "L01"])
    event = {
        "event": "end",
        "time": 1,
        "category": "evidence_analysis",
        "seconds": None,
        "exit_code": 130,
        "elapsed_upper_bound_seconds": 95.2,
    }
    (root / ".runtime/timing.jsonl").write_text(json.dumps(event) + "\n")
    result = summary(root)
    assert result["timed_command_seconds"] == {}
    assert result["commands_with_unknown_elapsed"] == [event]
