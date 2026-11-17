import json
import os
from pathlib import Path
import sys

import pytest

import vibuild.runtime.jobs as jobs
import vibuild.runtime.service as task_service
from vibuild.runtime.layout import initialize_task


def setup(tmp_path, steps=None):
    root = tmp_path / "modeling_20261111_120000"
    initialize_task(root, ["R11", "R02", "R05", "R07", "R08"])
    (root / "scene.blend").write_bytes(b"BLENDER current")
    jobs.write(root / "status.json", {"execution_paused": False})
    out = root / "R07/scores/001"
    request = {
        "region": "R07",
        "round": 1,
        "steps": steps
        or [
            {"tool": "score", "args": ["run", "--region", "R07", "--output", str(out)]}
        ],
    }
    manifest = tmp_path / "job.json"
    jobs.write(manifest, request)
    (tmp_path / ".runtime").mkdir(exist_ok=True)
    jobs.write(
        tmp_path / ".runtime/binding.json",
        {
            "task": str(root),
            "scene_sha256": jobs.digest(root / "scene.blend"),
            "guidance_sha256": None,
        },
    )
    config = tmp_path / "config.json"
    jobs.write(config, {"runtime": {}})
    return root, manifest, config


def fake_executor(calls, outcome="completed"):
    def run(root, command, log, env, **kwargs):
        calls.append(command)
        log.write_text("bounded failure detail" if outcome != "completed" else "ok")
        return outcome, 0 if outcome == "completed" else 1

    return run


def test_submit_does_not_execute_or_launch_model(tmp_path, monkeypatch):
    root, manifest, _ = setup(tmp_path)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: pytest.fail("submission must not run")
    )
    result = jobs.submit(root, manifest)
    assert result["state"] == "queued"
    assert not Path(result["receipt"]).exists()


def test_worker_completes_and_does_not_repeat(tmp_path, monkeypatch):
    root, manifest, config = setup(tmp_path)
    calls = []
    monkeypatch.setattr(jobs, "execute", fake_executor(calls))
    result = jobs.submit(root, manifest)
    assert jobs.work(root, config)[0]["state"] == "completed"
    assert Path(result["receipt"]).exists()
    assert jobs.work(root, config) == []
    assert len(calls) == 1
    assert len((jobs.queue(root) / "notifications.jsonl").read_text().splitlines()) == 1


def test_failure_stops_batch_and_preserves_saved_root(tmp_path, monkeypatch):
    root, manifest, config = setup(tmp_path)
    request = jobs.read(manifest, {})
    request["steps"] *= 2
    jobs.write(manifest, request)

    def fail(root, command, log, env):
        (root / "scene.blend").write_bytes(b"BLENDER newer saved geometry")
        log.write_text("render failed after root save")
        return "failed", 1

    monkeypatch.setattr(jobs, "execute", fail)
    jobs.submit(root, manifest)
    result = jobs.work(root, config)[0]
    assert result["state"] == "failed" and len(result["steps"]) == 1
    assert (root / "scene.blend").read_bytes() == b"BLENDER newer saved geometry"
    assert result["current_scene_sha256"] == jobs.digest(root / "scene.blend")


def test_stale_scene_or_changed_input_never_executes(tmp_path, monkeypatch):
    root, manifest, config = setup(tmp_path)
    monkeypatch.setattr(jobs, "execute", lambda *a: pytest.fail("stale job ran"))
    jobs.submit(root, manifest)
    (root / "scene.blend").write_bytes(b"new scene")
    assert jobs.work(root, config)[0]["state"] == "stale"
    script = tmp_path / "edit.py"
    script.write_text("# original")
    request = jobs.read(manifest, {})
    jobs.write(
        tmp_path / ".runtime/binding.json",
        {"task": str(root), "scene_sha256": jobs.digest(root / "scene.blend")},
    )
    request["steps"] = [
        {
            "tool": "scene",
            "args": [
                "iterate",
                "--region",
                "R07",
                "--baseline",
                str(root / "scene.blend"),
                "--script",
                str(script),
                "--output",
                str(root / "R07/iterations/001"),
                "--width",
                "960",
                "--samples",
                "24",
                "--threads",
                "4",
            ],
        }
    ]
    jobs.write(manifest, request)
    jobs.submit(root, manifest)
    script.write_text("# modified")
    assert "Input changed" in jobs.work(root, config)[0]["error"]


def test_pause_blocks_submission_and_leaves_queue_untouched(tmp_path, monkeypatch):
    root, manifest, config = setup(tmp_path)
    jobs.submit(root, manifest)
    jobs.write(root / "status.json", {"execution_paused": True})
    with pytest.raises(ValueError, match="paused"):
        jobs.submit(root, manifest)
    assert jobs.work(root, config) == []
    assert (
        jobs.read(next(jobs.queue(root).glob("*/state.json")), {})["state"] == "queued"
    )


def test_interrupted_job_not_reexecuted(tmp_path, monkeypatch):
    root, manifest, config = setup(tmp_path)
    jobs.submit(root, manifest)
    state = next(jobs.queue(root).glob("*/state.json"))
    value = jobs.read(state, {})
    value["state"] = "running"
    jobs.write(state, value)
    monkeypatch.setattr(
        jobs, "execute", lambda *a: pytest.fail("interrupted job repeated")
    )
    assert jobs.work(root, config)[0]["state"] == "interrupted"
    assert jobs.work(root, config) == []


def test_rejects_shells_external_assessors_and_output_escape(tmp_path):
    root, manifest, _ = setup(tmp_path)
    request = jobs.read(manifest, {})
    request["steps"] = [{"tool": "score", "args": "run; arbitrary shell"}]
    with pytest.raises(ValueError):
        jobs.validate(root, request)
    for tool, action in [("score", "export-assessor"), ("scene", "joints")]:
        request["steps"] = [{"tool": tool, "args": [action, "--region", "R07"]}]
        with pytest.raises(ValueError):
            jobs.validate(root, request)
    request["steps"] = [
        {
            "tool": "score",
            "args": ["run", "--region", "R07", "--output", str(tmp_path / "escape")],
        }
    ]
    with pytest.raises(ValueError, match="outputs"):
        jobs.validate(root, request)


def test_local_wait_timeout_and_pause(tmp_path):
    root, _, _ = setup(tmp_path)
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    outcome, code = jobs.execute(
        root, command, tmp_path / "idle.log", dict(os.environ), idle_seconds=0.2
    )
    assert outcome == "idle_timeout" and code != 0
    jobs.write(root / "status.json", {"execution_paused": True})
    outcome, code = jobs.execute(
        root, command, tmp_path / "pause.log", dict(os.environ)
    )
    assert outcome == "cancelled" and code != 0


def test_service_one_decision_then_program_then_fresh_decision(tmp_path, monkeypatch):
    root, manifest, config = setup(tmp_path)
    calls = []
    decisions = []
    monkeypatch.setattr(jobs, "execute", fake_executor(calls))

    def decision(directory, prompt, config, **kwargs):
        decisions.append((directory, prompt))
        if len(decisions) == 1:
            jobs.submit(root, manifest)
        return {"state": "completed", "thread_id": str(len(decisions))}

    monkeypatch.setattr(task_service.sessions, "run", decision)
    monkeypatch.setenv("VIBUILD_PARENT_STOPS", "[]")
    result = task_service.run(root, config, "http://127.0.0.1:18897")
    assert result["state"] == "idle"
    assert len(calls) == 1 and len(decisions) == 2
    assert decisions[0][0] != decisions[1][0]
    terminal = decisions[1][1].split("LAST_EXECUTION=")[1].splitlines()[0]
    assert json.loads(terminal)[0]["state"] == "completed"


def test_paused_service_launches_no_model(tmp_path, monkeypatch):
    root, _, config = setup(tmp_path)
    jobs.write(root / "status.json", {"execution_paused": True})
    monkeypatch.setattr(
        task_service.sessions,
        "run",
        lambda *a, **k: pytest.fail("model launched while paused"),
    )
    assert task_service.run(root, config, "http://127.0.0.1:18897")["state"] == "paused"


def diagnostic_job(tmp_path):
    root, manifest, config = setup(tmp_path)
    script = tmp_path / "edit.py"
    script.write_text("# managed edit")
    out = root / "R07/iterations/001"
    score_out = root / "R07/scores/001"
    out.mkdir(parents=True)
    (score_out / "R07").mkdir(parents=True)
    request = jobs.read(manifest, {})
    request["steps"] = [
        {
            "tool": "scene",
            "args": [
                "iterate",
                "--region",
                "R07",
                "--baseline",
                str(root / "scene.blend"),
                "--script",
                str(script),
                "--output",
                str(out),
                "--width",
                "960",
                "--samples",
                "16",
                "--threads",
                "4",
            ],
        },
        {
            "tool": "score",
            "args": [
                "prepare",
                "--scene",
                str(root / "scene.blend"),
                "--region",
                "R07",
                "--output",
                str(score_out),
                "--width",
                "960",
                "--samples",
                "16",
                "--threads",
                "4",
            ],
        },
        {
            "tool": "score",
            "args": ["supplement", "--region", "R07", "--output", str(score_out)],
            "diagnostics": {
                "facts": [
                    {"role": "Doorway geometry", "path": str(out / "geometry.json")}
                ]
            },
        },
        {
            "tool": "score",
            "args": ["run", "--region", "R07", "--output", str(score_out)],
        },
    ]
    jobs.write(manifest, request)
    return root, manifest, config, out / "geometry.json", score_out


@pytest.mark.parametrize(
    "fault", [None, "missing", "changed_evidence", "changed_scene", "stale_packet"]
)
def test_generated_diagnostics_bind_after_save_in_one_job(tmp_path, monkeypatch, fault):
    root, manifest, config, fact, score_out = diagnostic_job(tmp_path)
    calls = []

    def execute(root, command, log, env):
        action = command[command.index("tools") + 4]
        calls.append(action)
        log.write_text("ok")
        if action == "iterate":
            (root / "scene.blend").write_bytes(b"new saved scene")
            if fault != "missing":
                jobs.write(fact, {"door_depth": 1.5})
        elif action == "prepare":
            if fault == "changed_evidence":
                jobs.write(fact, {"door_depth": 2})
            if fault == "changed_scene":
                (root / "scene.blend").write_bytes(b"another scene")
            jobs.write(
                score_out / "R07/packet.json",
                {
                    "scene_sha256": "old"
                    if fault == "stale_packet"
                    else jobs.digest(root / "scene.blend")
                },
            )
        elif action == "supplement":
            bound = jobs.read(Path(jobs.option(command, "--diagnostics")), {})
            assert bound["scene_sha256"] == jobs.digest(root / "scene.blend")
            assert bound["facts"][0]["path"] == str(fact)
        return "completed", 0

    monkeypatch.setattr(jobs, "execute", execute)
    monkeypatch.setattr(
        task_service.sessions,
        "run",
        lambda *a, **k: pytest.fail("No decision is needed to bind hashes"),
    )
    jobs.submit(root, manifest)
    result = jobs.work(root, config)[0]
    assert result["state"] == ("completed" if fault is None else "failed"), result
    if fault is None:
        assert calls == ["iterate", "prepare", "supplement", "run"]
    else:
        assert "run" not in calls
        assert {
            "missing": "geometry.json",
            "changed_evidence": "changed after",
            "changed_scene": "changed after",
            "stale_packet": "does not match",
        }[fault] in result["error"]
    assert jobs.work(root, config) == []


def test_generated_diagnostics_reject_old_external_or_malformed_files(tmp_path):
    root, manifest, config, fact, _ = diagnostic_job(tmp_path)
    request = jobs.read(manifest, {})
    jobs.write(fact, {"old": True})
    with pytest.raises(ValueError, match="already exists"):
        jobs.submit(root, manifest)
    fact.unlink()
    item = request["steps"][2]["diagnostics"]["facts"][0]
    item["path"] = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="earlier stage"):
        jobs.validate(root, request)
    item["path"] = str(fact)
    item["unused"] = "not part of the contract"
    with pytest.raises(ValueError, match="exactly"):
        jobs.validate(root, request)


def test_generated_evidence_appearing_while_queued_is_not_rebound(
    tmp_path, monkeypatch
):
    root, manifest, config, fact, _ = diagnostic_job(tmp_path)
    jobs.submit(root, manifest)
    jobs.write(fact, {"old": True})
    monkeypatch.setattr(jobs, "execute", lambda *a: pytest.fail("Unexpected execution"))
    assert "before execution" in jobs.work(root, config)[0]["error"]


@pytest.mark.parametrize("with_diagnostics", [False, True])
def test_real_job_process_runs_standalone_score_without_runner_agent(
    tmp_path, monkeypatch, with_diagnostics
):
    from helpers import config as fake_config
    from PIL import Image
    from vibuild.assessment.scoring import rubric_hash

    root, manifest, _ = setup(tmp_path)
    out = root / "R07/scores/001"
    work = out / "R07"
    work.mkdir(parents=True)
    views = []
    for ident in ["1", "2"]:
        card = work / f"view_{ident}.jpg"
        Image.new("RGB", (40, 40)).save(card)
        views.append(
            {
                "id": ident,
                "width": 20,
                "height": 20,
                "card": str(card),
                "card_sha256": jobs.digest(card),
            }
        )
    packet = {
        "region": "R07",
        "scene_sha256": jobs.digest(root / "scene.blend"),
        "rubric_hash": rubric_hash(),
        "views": views,
        "objects": [],
    }
    jobs.write(work / "packet.json", packet)
    (out / ".runtime").mkdir(parents=True, exist_ok=True)
    jobs.write(
        out / ".runtime/protocol.json",
        {
            "scene": str(root / "scene.blend"),
            "scene_sha256": packet["scene_sha256"],
            "rubric_hash": rubric_hash(),
            "all_regions": ["R07"],
            "regions": [
                {
                    "region": "R07",
                    "views": [
                        {"id": v["id"], "width": 20, "height": 20} for v in views
                    ],
                }
            ],
        },
    )
    cfg = fake_config(
        tmp_path,
        """
packet=json.JSONDecoder().raw_decode(prompt.split('PACKET=',1)[1])[0]
assert '--ephemeral' in sys.argv and 'resume' not in sys.argv
value={'region':'R07','views':[{'id':v,'scores':{k:80 for k in ['structure','completeness','alignment','appearance','connectivity']},'confidence':.9,'notes':'Visible match'} for v in packet['views']],'findings':[],'uncertainties':[]}
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(value))
print(json.dumps({'type':'thread.started','thread_id':'independent-scorer'}),flush=True)
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}),flush=True)
""",
    )
    data = jobs.read(cfg, {})
    data["runtime"]["analysis_python"] = sys.executable
    jobs.write(cfg, data)
    if with_diagnostics:
        generated = root / "R07/evidence/generated"
        request = jobs.read(manifest, {})
        request["steps"] = [
            {
                "tool": "scene",
                "args": [
                    "inspect",
                    "--blend",
                    str(root / "scene.blend"),
                    "--output",
                    str(generated),
                ],
            },
            {
                "tool": "score",
                "args": ["supplement", "--region", "R07", "--output", str(out)],
                "diagnostics": {
                    "images": [
                        {
                            "id": "detail",
                            "role": "Doorway",
                            "path": str(generated / "detail.jpg"),
                        }
                    ],
                    "facts": [
                        {"role": "Bounds", "path": str(generated / "geometry.json")}
                    ],
                },
            },
            *request["steps"],
        ]
        jobs.write(manifest, request)
        real_execute = jobs.execute

        def execute(root, command, log, env):
            if command[command.index("tools") + 3 : command.index("tools") + 5] == [
                "scene",
                "inspect",
            ]:
                generated.mkdir(parents=True)
                Image.new("RGB", (40, 40)).save(generated / "detail.jpg")
                jobs.write(generated / "geometry.json", {"depth": 1.5})
                log.write_text("Generated evidence")
                return "completed", 0
            return real_execute(root, command, log, env)

        monkeypatch.setattr(jobs, "execute", execute)
    jobs.submit(root, manifest)
    result = jobs.work(root, cfg)[0]
    assert result["state"] == "completed", result
    assert result["score"]["score"] == 80
    assert jobs.read(work / ".runtime/scored.json", {})["tool_calls"] == 0
    assert not (work / ".runtime/assessor_instruction.txt").exists()
    if with_diagnostics:
        packet = jobs.read(work / "packet.json", {})
        assert len(packet["diagnostic_views"]) == len(packet["diagnostic_facts"]) == 1
        assert packet["diagnostic_facts"][0]["value"] == {"depth": 1.5}
        assert [s["action"] for s in result["steps"]] == [
            "inspect",
            "supplement",
            "run",
        ]


def test_submit_combines_guidance_and_syntax_without_execution(tmp_path, monkeypatch):
    root, manifest, _ = setup(tmp_path)
    notice = root / ".runtime/main/user_guidance_pending.txt"
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text("Preserve the source door.")
    monkeypatch.setenv("CODEX_THREAD_ID", "actual-session")
    from vibuild.agents.context import material_index

    with pytest.raises(ValueError, match="guidance"):
        jobs.submit(root, manifest)
    material_index(root, tmp_path / "config.json", manifest.parent)
    jobs.submit(root, manifest)
    assert jobs.guidance(root)["acknowledgment"]["session_id"] == "actual-session"
    script = tmp_path / "bad.py"
    script.write_text("if invalid syntax")
    request = jobs.read(manifest, {})
    request["steps"] = [
        {
            "tool": "scene",
            "args": [
                "iterate",
                "--region",
                "R07",
                "--baseline",
                str(root / "scene.blend"),
                "--script",
                str(script),
                "--output",
                str(root / "R07/iterations/001"),
                "--width",
                "960",
                "--samples",
                "24",
                "--threads",
                "4",
            ],
        }
    ]
    jobs.write(manifest, request)
    with pytest.raises(ValueError, match=r"bad.py:1:"):
        jobs.submit(root, manifest)
    assert len(list(jobs.queue(root).glob("*/request.json"))) == 1


def test_failure_reports_inner_exception_and_exact_log(tmp_path):
    outer = tmp_path / "step.log"
    outer.write_text("RuntimeError: Edit failed\n")
    inner = tmp_path / "output/.runtime/tools.log"
    inner.parent.mkdir(parents=True)
    inner.write_text(
        "noise\n" * 10000
        + '  File "edit.py", line 93, in <module>\n    cap_points[i]-v\nAttributeError: Vector - int\n'
    )
    result = jobs.failure_detail(outer, tmp_path / "output")
    assert "Vector - int" in result["error_tail"] and "line 93" in result["error_tail"]
    assert result["error_log"] == str(inner)
    assert len(result["error_tail"]) < 1000


def test_private_binding_rejects_changed_scene_and_guidance_without_model_hashes(
    tmp_path, monkeypatch
):
    from vibuild.agents.context import material_index

    root, manifest, config = setup(tmp_path)
    notice = root / ".runtime/main/user_guidance_pending.txt"
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text("Preserve the source door.")
    request = jobs.read(manifest, {})
    request.pop("expected_scene_sha256", None)
    jobs.write(manifest, request)
    index = material_index(root, config, manifest.parent)
    assert "sha256" not in json.dumps(index)
    monkeypatch.setenv("CODEX_THREAD_ID", "actual-receiver")
    notice.write_text("Preserve both source doors.")
    with pytest.raises(ValueError, match="guidance changed"):
        jobs.submit(root, manifest)
    material_index(root, config, manifest.parent)
    original = (root / "scene.blend").read_bytes()
    (root / "scene.blend").write_bytes(b"changed")
    with pytest.raises(ValueError, match="another scene"):
        jobs.submit(root, manifest)
    (root / "scene.blend").write_bytes(original)
    result = jobs.submit(root, manifest)
    stored = jobs.read(Path(result["receipt"]).parent / "request.json", {})
    assert stored["expected_scene_sha256"] == jobs.digest(root / "scene.blend")
    assert set(result) == {"state", "receipt"}


def test_service_continues_explicit_owner_handoff_without_job(tmp_path, monkeypatch):
    root, _, config = setup(tmp_path)
    jobs.write(root / "status.json", {"active_region": "R10"})
    decisions = []

    def decision(directory, prompt, config, **kwargs):
        decisions.append(prompt)
        if len(decisions) == 1:
            jobs.write(root / "status.json", {"active_region": "J05"})
        return {"state": "completed"}

    monkeypatch.setattr(task_service.sessions, "run", decision)
    monkeypatch.setenv("VIBUILD_PARENT_STOPS", "[]")
    assert task_service.run(root, config, "http://127.0.0.1:18897")["state"] == "idle"
    assert len(decisions) == 2 and '"active_region":"J05"' in decisions[1]


def test_owner_handoff_blocks_old_region_submission(tmp_path):
    root, manifest, _ = setup(tmp_path)
    jobs.write(root / "status.json", {"active_region": "J05"})
    with pytest.raises(ValueError, match="Active task is J05"):
        jobs.submit(root, manifest)
    assert not list(jobs.queue(root).glob("*/request.json"))
