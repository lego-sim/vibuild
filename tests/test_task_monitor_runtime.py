"""Monitor reports independent actors and feedback without inventing acceptance."""

import hashlib
import json
import os
from pathlib import Path

import pytest

import vibuild.runtime.monitor as monitor
from vibuild.runtime.layout import initialize_task


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "modeling_20261110_120000"
    initialize_task(path, ["R11", "R02", "R05"])
    put(
        path / "status.json",
        {
            "state": "building",
            "region": "R05",
            "round": 16,
            "current_task": "rail repair",
            "started": 10,
            "current_candidate": {"region": "R05", "round": 15, "sha256": "latest"},
        },
    )
    return path


@pytest.mark.parametrize(
    "role", ["analyst", "scorer", "checker", "coverage", "r08_planner"]
)
@pytest.mark.parametrize("action", ["mark", "run"])
def test_readonly_actor_does_not_replace_main_state_or_plan(
    root, monkeypatch, role, action
):
    monkeypatch.setenv("VIBUILD_MONITOR_AGENT", role)
    monkeypatch.setattr(monitor.subprocess, "call", lambda command: 0)
    original_plan = (root / "plan.json").read_bytes()
    original = json.loads((root / "status.json").read_text())
    argv = [action, "source_analysis", "R08", "12", "independent source check"]
    if action == "run":
        argv += ["--", "inspect"]
    assert monitor.main(root, argv) == 0
    state = json.loads((root / "status.json").read_text())
    assert all(state[key] == value for key, value in original.items())
    assert state["agents"][role]["region"] == "R08"
    assert (root / "plan.json").read_bytes() == original_plan


def test_writer_does_not_fabricate_new_candidate(root, monkeypatch):
    monkeypatch.setenv("VIBUILD_MONITOR_AGENT", "builder")
    monkeypatch.setattr(monitor.subprocess, "call", lambda command: 0)
    monitor.main(
        root, ["run", "tool_execution", "R08", "12", "new edit", "--", "iterate"]
    )
    result = monitor.summary(root)
    assert result["status"]["region"] == "R08"
    assert result["current_candidate"] == {
        "region": "R05",
        "round": 15,
        "sha256": "latest",
    }


def test_pending_guidance_is_early_explicit_and_digest_bound(root, monkeypatch, capsys):
    notice = root / ".runtime/main/user_guidance_pending.txt"
    notice.parent.mkdir(parents=True)
    notice.write_text("Keep the landing straight.\n")
    digest = hashlib.sha256(notice.read_bytes()).hexdigest()
    calls = []

    def readonly(command):
        assert str(notice) in capsys.readouterr().out
        calls.append(command)
        return 0

    monkeypatch.setattr(monitor.subprocess, "call", readonly)
    monkeypatch.setenv("VIBUILD_MONITOR_AGENT", "analyst")
    assert (
        monitor.main(
            root, ["run", "analysis", "R05", "16", "read frames", "--", "inspect"]
        )
        == 0
    )
    assert len(calls) == 1
    assert monitor.main(root, ["mark", "analysis", "R05", "16", "analyze"]) == 0
    assert digest not in capsys.readouterr().out
    with pytest.raises(ValueError, match="Pending guidance blocks"):
        monitor.main(
            root, ["run", "tool_execution", "R05", "16", "write", "--", "iterate"]
        )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="digest does not match"):
        monitor.record_guidance(root, "old", "/root/main")
    monitor.record_guidance(root, digest, "/root/main")
    ack = json.loads((notice.parent / "user_guidance_ack.json").read_text())
    assert ack["session_id"] == "/root/main" and ack["sha256"] == digest
    monkeypatch.setattr(monitor.subprocess, "call", lambda command: 0)
    assert (
        monitor.main(
            root, ["run", "tool_execution", "R05", "16", "write", "--", "iterate"]
        )
        == 0
    )
    notice.write_text("New instruction: preserve the right rail.\n")
    with pytest.raises(ValueError, match="Pending guidance blocks"):
        monitor.main(
            root, ["run", "tool_execution", "R05", "17", "write", "--", "iterate"]
        )


def test_explicit_custom_writer_cannot_skip_guidance(root, monkeypatch):
    notice = root / ".runtime/main/user_guidance_pending.txt"
    notice.parent.mkdir(parents=True)
    notice.write_text("New boundary correction")
    monkeypatch.setenv("VIBUILD_MONITOR_MUTATES_SCENE", "1")
    with pytest.raises(ValueError, match="Pending guidance blocks"):
        monitor.main(
            root,
            ["run", "tool_execution", "R05", "17", "custom writer", "--", "custom.py"],
        )


def test_j_task_assessments_distinguish_output_recording_and_cancellation(
    root, monkeypatch
):
    plan = json.loads((root / "plan.json").read_text())
    plan["steps"].append({"id": "J01", "state": "planned"})
    put(root / "plan.json", plan)
    for number in range(1, 5):
        rt = root / f"J01/scores/{number:03d}/J01/.runtime"
        rt.mkdir(parents=True)
        (rt / "judge_started").write_text("100")
        put(
            rt / "handoff.json",
            {"state": "running", "session_id": f"/root/assessor{number}"},
        )
        if number != 3:
            put(rt / "final.json", {"score": 75})
            os.utime(rt / "final.json", (150, 150))
        if number == 2:
            put(
                rt / "scored.json",
                {
                    "result_sha256": hashlib.sha256(
                        (rt / "final.json").read_bytes()
                    ).hexdigest(),
                    "seconds": 50,
                },
            )
            put(rt.parent / "score.json", {"score": 75})
        if number == 3:
            put(rt / "handoff.json", {"state": "cancelled", "cancelled_at": 125})
        if number == 4:
            os.utime(rt / "final.json", (90, 90))
    monkeypatch.setattr(monitor.time, "time", lambda: 200)
    assessments = monitor.summary(root)["independent_assessment_wall_seconds"]
    assert [row["state"] for row in assessments] == [
        "result_unrecorded",
        "recorded",
        "cancelled",
        "result_unrecorded",
    ]
    assert [row["seconds"] for row in assessments] == [50, 50, 25, None]


def test_parallel_events_do_not_become_exclusive_phase_or_model_latency(
    root, monkeypatch
):
    rows = [
        {"event": "start", "time": 20, "category": "render"},
        {"event": "start", "time": 21, "category": "analysis"},
        {"event": "end", "time": 30, "category": "render", "seconds": 10},
        {"event": "end", "time": 31, "category": "analysis", "seconds": 10},
        {"event": "end", "time": 32, "category": "bad_clock", "seconds": -1},
    ]
    (root / ".runtime/timing.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    monkeypatch.setattr(monitor.time, "time", lambda: 35)
    result = monitor.summary(root)
    assert result["timed_command_seconds"] == {"render": 10, "analysis": 10}
    assert result["commands_with_unknown_elapsed"] == [rows[-1]]
    assert result["wall_seconds"] == 25
    assert "phase_wall_seconds" not in result and "active_seconds" not in result
    assert result["timing_attribution"].startswith("unavailable")


def test_summary_cli_does_not_echo_full_history(root, capsys):
    assert monitor.main(root, ["summary"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "plan" not in result and "independent_assessment_wall_seconds" not in result
    stored = json.loads(Path(result["report"]).read_text())
    assert "plan" in stored and "independent_assessment_wall_seconds" in stored
