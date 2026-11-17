"""Region-aware task progress; reports state without granting acceptance."""

from contextlib import contextmanager
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from vibuild.runtime.layout import task_paths


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


@contextmanager
def state_lock(root):
    with (root / ".runtime/monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def guidance(root):
    notice = root / ".runtime/main/user_guidance_pending.txt"
    if not notice.exists():
        return None
    content = notice.read_bytes()
    if not content.strip():
        return None
    digest = hashlib.sha256(content).hexdigest()
    ack = read(root / ".runtime/main/user_guidance_ack.json", {})
    return {
        "sha256": digest,
        "text": content.decode(),
        "pending": ack.get("sha256") != digest,
        "acknowledgment": ack,
    }


def record_guidance(root, sha256, session_id):
    if not session_id.strip():
        raise ValueError("Actual receiving session ID is required")
    with state_lock(root):
        notice = guidance(root)
        if notice is None or sha256 != notice["sha256"]:
            raise ValueError(
                "Guidance digest does not match the current notice; read it before acknowledging"
            )
        receipt = {
            "sha256": sha256,
            "session_id": session_id,
            "acknowledged_at": time.time(),
        }
        write(root / ".runtime/main/user_guidance_ack.json", receipt)
    return receipt


def mutates_scene(command):
    # Managed writes use these commands. Custom writers must declare the flag;
    # a readonly flag cannot disguise a recognized managed write.
    return (
        bool({"iterate", "patch"} & set(command))
        or os.environ.get("VIBUILD_MONITOR_MUTATES_SCENE") == "1"
    )


def elapsed(start, end):
    if (
        not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end))
        or end < start
    ):
        return None
    return round(end - start, 1)


def summary(root):
    paths = task_paths(root)
    root = paths["root"]
    state = read(paths["status"], {})
    plan = read(paths["plan"], {})
    events = root / ".runtime/timing.jsonl"
    records = (
        [json.loads(line) for line in events.read_text().splitlines()]
        if events.exists()
        else []
    )
    now = state.get("finished", time.time())
    totals = {}
    assessments = []
    scores = []
    untimed = []
    for entry in records:
        if entry.get("event") == "end":
            if (
                isinstance(entry.get("seconds"), (int, float))
                and math.isfinite(entry["seconds"])
                and entry["seconds"] >= 0
            ):
                key = entry["category"]
                totals[key] = totals.get(key, 0) + entry["seconds"]
            else:
                untimed.append(entry)
    regions = [step["id"] for step in plan.get("steps", [])]
    for region in regions:
        child = task_paths(root, region)
        scores.extend(
            {"region": region, **row}
            for row in read(child["region"] / "scores.json", [])
        )
        for start in sorted(
            child["scores"].glob("*/" + region + "/.runtime/judge_started")
        ):
            final = start.parent / "final.json"
            handoff = read(start.parent / "handoff.json", {})
            scored = read(start.parent / "scored.json", {})
            recorded = (
                final.exists()
                and (start.parent.parent / "score.json").exists()
                and scored.get("result_sha256")
                == hashlib.sha256(final.read_bytes()).hexdigest()
            )
            status = (
                "cancelled"
                if handoff.get("state") == "cancelled"
                else "recorded"
                if recorded
                else "result_unrecorded"
                if final.exists()
                else "running"
            )
            end = (
                handoff.get("cancelled_at")
                if status == "cancelled"
                else final.stat().st_mtime
                if final.exists()
                else now
            )
            duration = elapsed(float(start.read_text()), end)
            if recorded and elapsed(0, scored.get("seconds")) is None:
                duration = None
            assessments.append(
                {
                    "region": region,
                    "round": start.parents[2].name,
                    "seconds": duration,
                    "state": status,
                    "session_id": handoff.get("session_id"),
                }
            )
    notice = guidance(root)
    result = {
        "status": state,
        "plan": plan,
        "wall_seconds": elapsed(state.get("started", now), now),
        "current_candidate": state.get("current_candidate"),
        "timed_command_seconds": totals,
        "independent_assessment_wall_seconds": assessments,
        "commands_with_unknown_elapsed": untimed,
        "guidance": {key: value for key, value in notice.items() if key != "text"}
        if notice
        else None,
        "timing_attribution": "unavailable: parallel event ordering does not identify exclusive phases or model latency",
        "note": "Historical timings are retained. Command sums and assessment intervals overlap and are not exclusive wall time. "
        "Unmonitored time cannot be attributed to model latency, waiting or tool construction. "
        "A completed local score does not complete the task or its seam gates.",
    }
    write(root / ".runtime/monitor_summary.json", result)
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(step['id'])}</td><td>{esc(step['state'])}</td></tr>"
        for step in plan.get("steps", [])
    )
    score_rows = "".join(
        "<tr><td>"
        + esc(row["region"])
        + "</td><td>"
        + esc(str(row.get("round")))
        + "</td><td>"
        + esc(str(row.get("score")))
        + (
            " (historical)"
            if row.get("provenance", "").startswith("historical")
            else ""
        )
        + '</td><td><a href="'
        + esc(os.path.relpath(row["report"], root))
        + '">Details</a></td></tr>'
        for row in scores
    )
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Modeling task progress</title>'
    page += "<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto}td,th{padding:8px;text-align:left}pre{white-space:pre-wrap}</style>"
    page += (
        "<h1>Shared global project progress</h1><p>"
        + esc(state.get("state", "planned"))
        + "</p><p>"
        + esc(state.get("current_task", ""))
        + "</p>"
    )
    page += (
        "<p>Global acceptance: "
        + esc(plan.get("global_acceptance", "not_assessed"))
        + '. Current scene: <a href="scene.blend">scene.blend</a>.</p>'
    )
    page += (
        "<table><tr><th>Session</th><th>State</th></tr>"
        + rows
        + "</table><h2>Regional assessment history</h2><table><tr><th>Region</th><th>Round</th><th>Total</th><th>Report</th></tr>"
        + score_rows
        + "</table>"
    )
    page += (
        "<p>"
        + esc(result["note"])
        + "</p><p>Task wall time: "
        + str(result["wall_seconds"])
        + " s (includes pauses).</p><h2>Command timing (overlapping sums)</h2><pre>"
        + esc(json.dumps(totals, indent=2))
        + "</pre><h2>Agent states</h2><pre>"
        + esc(json.dumps(state.get("agents", {}), indent=2))
        + "</pre>"
    )
    if untimed:
        page += (
            "<h2>Interrupted commands with unknown elapsed time</h2><pre>"
            + esc(json.dumps(untimed, indent=2))
            + "</pre>"
        )
    paths["progress"].write_text(page)
    return result


def main(root, argv=None):
    root = Path(root).resolve()
    action, *args = sys.argv[1:] if argv is None else argv
    if action == "summary":
        value = summary(root)
        print(
            json.dumps(
                {
                    "status": {
                        k: value["status"].get(k)
                        for k in (
                            "state",
                            "execution_paused",
                            "active_region",
                            "round",
                            "current_task",
                        )
                    },
                    "wall_seconds": value["wall_seconds"],
                    "timed_command_seconds": value["timed_command_seconds"],
                    "assessment_count": len(
                        value["independent_assessment_wall_seconds"]
                    ),
                    "report": str(root / ".runtime/monitor_summary.json"),
                },
                separators=(",", ":"),
            )
        )
        return 0
    if action not in ("run", "mark"):
        raise ValueError("Expected run, mark or summary")
    category, region, round_number, label, *cmd = args
    paths = task_paths(root, region, int(round_number))
    state = read(paths["status"], {})
    plan = read(paths["plan"], {})
    role = os.environ.get(
        "VIBUILD_MONITOR_AGENT", "main" if "score" in cmd else "builder"
    )
    active_owner = role in ("main", "builder")
    notice = guidance(root)
    if notice and notice["pending"]:
        print(
            json.dumps(
                {
                    "state": "guidance_pending",
                    "path": str(root / ".runtime/main/user_guidance_pending.txt"),
                }
            ),
            flush=True,
        )
        if action == "run" and mutates_scene(cmd):
            raise ValueError(
                "Pending guidance blocks geometry writes; submit through the current decision context"
            )
    if category in ("completed", "accepted", "global_accepted"):
        raise ValueError(
            "Monitor cannot grant acceptance; record verified gates separately"
        )
    if action == "run" and state.get("execution_paused", False):
        raise ValueError("Task execution is paused for workflow repairs")

    def update():
        with state_lock(root):
            state = read(paths["status"], {})
            plan = read(paths["plan"], {})
            agent = state.setdefault("agents", {}).setdefault(role, {})
            agent.update(
                state="running_tools" if action == "run" else category,
                task=label,
                region=region,
                round=int(round_number),
                updated=time.time(),
            )
            if active_owner:
                state.update(
                    region=region,
                    round=int(round_number),
                    state=category,
                    current_task=label,
                    updated=time.time(),
                )
                state.pop("finished", None)
                for step in plan.get("steps", []):
                    if step["id"] == region:
                        step.update(state=category, round=int(round_number))
                plan["state"] = "in_progress"
                write(paths["plan"], plan)
            write(paths["status"], state)

    def event(value):
        with (root / ".runtime/timing.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "region": region,
                        "round": int(round_number),
                        "category": category,
                        "actor": role,
                        **value,
                    }
                )
                + "\n"
            )

    if action == "mark":
        if cmd:
            raise ValueError("Unexpected mark arguments")
        update()
        event({"event": "mark", "label": label})
        return 0
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("Command is required")
    if "iterate" in cmd and "--candidate" in cmd:
        raise ValueError("Use iterate --script to avoid duplicate full scene copies")
    update()
    ident = uuid.uuid4().hex
    start = time.time()
    code = 1
    event({"event": "start", "id": ident, "label": label})
    try:
        code = subprocess.call(cmd)
    finally:
        event(
            {
                "event": "end",
                "id": ident,
                "label": label,
                "seconds": time.time() - start,
                "exit_code": code,
            }
        )
        with state_lock(root):
            current = read(paths["status"], {})
            agent = current.get("agents", {}).get(role, {})
            check_failed = code == 2 and "joints" in cmd
            outcome = (
                "Completed: "
                if code == 0
                else "Check failed: "
                if check_failed
                else f"Failed (exit {code}): "
            )
            next_state = (
                "inspecting_result"
                if code == 0
                else "check_failed"
                if check_failed
                else "awaiting_error_repair"
            )
            if agent.get("task") == label:
                agent.update(
                    state=next_state, task=outcome + label, updated=time.time()
                )
            if (
                active_owner
                and not current.get("execution_paused")
                and current.get("current_task") == label
            ):
                current.update(
                    state="awaiting_session_action",
                    current_task=outcome + label,
                    updated=time.time(),
                )
            write(paths["status"], current)
        summary(root)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
