import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import vibuild.runtime.cache as cache
import vibuild.assessment.metrics as evaluate
import vibuild.assessment.comparison as comparison
import vibuild.blender.scene as scene
from vibuild.assessment.comparison import prepare_protocol, compare_reports


def protocol(tmp_path):
    data = dict(
        depth=np.full((4, 6), 3.0, np.float32),
        K=np.eye(3),
        camera_to_world=np.eye(4),
        confidence=np.ones((4, 6), np.float32),
        opaque=np.ones((4, 6), bool),
        semantic=np.ones((4, 6), np.int32),
        instance=np.ones((4, 6), np.int32),
    )
    np.savez(tmp_path / "ref.npz", **data)
    np.savez(tmp_path / "pred.npz", **data)
    return {
        "frames": [{"id": "one", "reference": "ref.npz", "prediction": "pred.npz"}],
        "classes": {"wall": 1},
    }


def test_cached_scores_match_and_invalidate_on_content(tmp_path):
    m = protocol(tmp_path)
    before = evaluate.evaluate_cached(
        m, tmp_path, tmp_path / "one.json", tmp_path / "cache"
    )
    after = evaluate.evaluate_cached(
        m, tmp_path, tmp_path / "two.json", tmp_path / "cache"
    )
    assert not before["cache"]["hit"] and after["cache"]["hit"]
    assert before["frames"] == after["frames"]
    assert not compare_reports(before, after)["measured_improvement"]
    # Prediction changes keep the protocol but invalidate the cached score.
    p = evaluate.load_npz(tmp_path / "pred.npz")
    p["depth"] *= 2
    np.savez(tmp_path / "pred.npz", **p)
    changed = evaluate.evaluate_cached(
        m, tmp_path, tmp_path / "three.json", tmp_path / "cache"
    )
    assert not changed["cache"]["hit"]
    assert changed["protocol_sha256"] == before["protocol_sha256"]
    assert compare_reports(before, changed)["regressions"]
    # Reference changes invalidate both score and protocol.
    p = evaluate.load_npz(tmp_path / "ref.npz")
    p["opaque"][0] = False
    np.savez(tmp_path / "ref.npz", **p)
    ref = evaluate.evaluate_cached(
        m, tmp_path, tmp_path / "four.json", tmp_path / "cache"
    )
    assert (
        not ref["cache"]["hit"] and ref["protocol_sha256"] != before["protocol_sha256"]
    )


def test_cache_corruption_and_copy_isolation(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "a").write_text("original")
    root = tmp_path / "cache"
    with cache.entry_lock(root, "key") as entry:
        cache.publish(entry, output, ["a"])
        assert cache.verified(entry)
        cache.restore(entry, tmp_path / "copy")
        (tmp_path / "copy/a").write_text("modified session artifact")
        assert cache.verified(entry)
        (entry / "a").write_text("corrupt")
        assert not cache.verified(entry)


def test_concurrent_cache_writers_only_compute_once(tmp_path):
    computed = []

    def run(_):
        with cache.entry_lock(tmp_path / "cache", "key") as entry:
            if not cache.verified(entry):
                out = tmp_path / "output"
                out.mkdir(exist_ok=True)
                (out / "result").write_text("done")
                computed.append(1)
                cache.publish(entry, out, ["result"])
            return cache.verified(entry)

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(pool.map(run, range(3)))
    assert len(computed) == 1


def test_compare_preflight_preserves_evidence_and_calibration(tmp_path):
    m = protocol(tmp_path)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(m))
    fixed, render = prepare_protocol(path)
    assert render["frames"][0]["width"] == 6 and render["frames"][0]["height"] == 4
    assert Path(fixed["frames"][0]["reference"]).is_absolute()
    assert "opaque" not in fixed["frames"][0]  # Never invent masks.
    m["frames"].append(dict(m["frames"][0]))
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="unique"):
        prepare_protocol(path)
    m["frames"].pop()
    m["frames"][0]["reference_rgb"] = "rgb.npy"
    np.save(tmp_path / "rgb.npy", np.zeros((8, 12, 3), np.uint8))
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="no implicit resize"):
        prepare_protocol(path)


def test_completed_artifacts_are_not_overwritten(tmp_path):
    m = protocol(tmp_path)
    out = tmp_path / "scores.json"
    out.write_text("frozen")
    with pytest.raises(FileExistsError):
        evaluate.evaluate_cached(m, tmp_path, out)
    with pytest.raises(SystemExit):
        evaluate.main(
            [
                "--reference",
                str(tmp_path / "ref.npz"),
                "--prediction",
                str(tmp_path / "pred.npz"),
                "--output",
                str(out),
            ]
        )
    assert out.read_text() == "frozen"


def test_renderer_protocol_change_is_not_scene_improvement(tmp_path):
    m = protocol(tmp_path)
    m["prediction_protocol_sha256"] = "renderer-v1"
    before = evaluate.evaluate_manifest(m, tmp_path)
    m["prediction_protocol_sha256"] = "renderer-v2"
    after = evaluate.evaluate_manifest(m, tmp_path)
    with pytest.raises(ValueError, match="protocols differ"):
        compare_reports(before, after)


def test_changed_candidate_does_not_inherit_baseline_review(tmp_path, monkeypatch):
    m = protocol(tmp_path)
    (tmp_path / "protocol.json").write_text(json.dumps(m))
    (tmp_path / "before.blend").write_bytes(b"before")
    (tmp_path / "after.blend").write_bytes(b"after")
    (tmp_path / "evidence.txt").write_text("baseline visual review")
    (tmp_path / "review.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "reviewer": "tester",
                "evidence": ["evidence.txt"],
                "checks": dict.fromkeys(
                    ["dimensions", "topology", "openings", "occlusion"], "passed"
                ),
            }
        )
    )

    def render(args):
        out = Path(args[args.index("--output") + 1])
        out.mkdir()
        (out / "one.npz").write_bytes((tmp_path / "pred.npz").read_bytes())
        (out / "render_manifest.json").write_text(
            json.dumps(
                {
                    "render_protocol_sha256": "fixed",
                    "request_seconds": 0,
                    "cache": {"hit": False},
                }
            )
        )
        return 0

    monkeypatch.setattr(comparison.scene, "main", render)
    assert (
        comparison.main(
            [
                "--manifest",
                str(tmp_path / "protocol.json"),
                "--baseline",
                str(tmp_path / "before.blend"),
                "--candidate",
                str(tmp_path / "after.blend"),
                "--baseline-review",
                str(tmp_path / "review.json"),
                "--output",
                str(tmp_path / "result"),
            ]
        )
        == 0
    )
    before = json.loads((tmp_path / "result/baseline_scores.json").read_text())
    after = json.loads((tmp_path / "result/candidate_scores.json").read_text())
    assert (
        before["plausibility"]["status"] == "passed"
        and after["plausibility"]["status"] == "blocked"
    )
    assert before["protocol_sha256"] == after["protocol_sha256"]


def test_missing_depth_keeps_all_frames_and_independent_rgb(tmp_path):
    m = protocol(tmp_path)
    np.save(tmp_path / "rgb.npy", np.zeros((8, 12, 3), np.uint8))
    rgb = {
        "reference_rgb": "rgb.npy",
        "appearance_calibration": {
            "reference": {
                "K": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                "camera_to_world": np.eye(4).tolist(),
            }
        },
    }
    m["frames"][0].update(rgb)
    m["frames"].append({"id": "missing", "reference": "missing.npz", **rgb})
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(m))
    fixed, render = prepare_protocol(path)
    assert len(fixed["frames"]) == 2 and len(render["frames"]) == 3
    assert [f["id"] for f in render["preflight"]] == ["one", "missing"]
    assert render["preflight"][1]["geometry"] == "blocked"
    assert render["preflight"][1]["appearance"] == "renderable"
    rgb_id = render["bindings"]["one"]["appearance"]
    frame = next(f for f in render["frames"] if f["id"] == rgb_id)
    assert (frame["width"], frame["height"]) == (12, 8) and frame["geometry"] is False
    assert render["frames"][0]["width"] == 6  # Native depth is unchanged.


def test_blocked_report_success_exit_is_distinct_from_input_failure(tmp_path):
    m = protocol(tmp_path)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(m))
    assert (
        evaluate.main(
            ["--manifest", str(manifest), "--output", str(tmp_path / "default.json")]
        )
        == 0
    )
    assert (
        evaluate.main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "strict.json"),
                "--require-passed",
            ]
        )
        == 2
    )
    assert (
        evaluate.main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
        == 2
    )


def test_build_failure_never_starts_render(tmp_path, monkeypatch):
    m = protocol(tmp_path)
    (tmp_path / "protocol.json").write_text(json.dumps(m))
    (tmp_path / "baseline.blend").write_bytes(b"base")
    (tmp_path / "edit.py").write_text("raise RuntimeError()")
    calls = []

    def fail(args):
        calls.append(args[0])
        return 1

    monkeypatch.setattr(comparison.scene, "main", fail)
    with pytest.raises(RuntimeError, match="build failed"):
        comparison.main(
            [
                "--manifest",
                str(tmp_path / "protocol.json"),
                "--baseline",
                str(tmp_path / "baseline.blend"),
                "--build-script",
                str(tmp_path / "edit.py"),
                "--output",
                str(tmp_path / "result"),
            ]
        )
    assert calls == ["patch"] and not (tmp_path / "result/comparison.json").exists()


def test_empty_appearance_mask_blocks_without_wasting_rgb_render(tmp_path):
    m = protocol(tmp_path)
    np.save(tmp_path / "rgb.npy", np.zeros((4, 6, 3), np.uint8))
    np.save(tmp_path / "mask.npy", np.zeros((4, 6), bool))
    m["frames"][0].update(
        reference_rgb="rgb.npy",
        appearance_mask="mask.npy",
        appearance_calibration={
            "reference": {
                "K": np.eye(3).tolist(),
                "camera_to_world": np.eye(4).tolist(),
            }
        },
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(m))
    fixed, render = prepare_protocol(path)
    assert len(render["frames"]) == 1 and render["frames"][0]["appearance"] is False
    assert render["preflight"][0]["appearance"] == "blocked"
    assert "contain scored pixels" in render["preflight"][0]["issues"][0]
    assert (
        evaluate.appearance_frame(fixed["frames"][0], tmp_path)["status"] == "blocked"
    )


def resumable_args(tmp_path, action="render"):
    blend = tmp_path / "scene.blend"
    blend.write_bytes(b"fixed scene")
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({"frames": [{"id": "1"}, {"id": "2"}]}))
    args = [
        action,
        "--blend",
        str(blend),
        "--blender",
        sys.executable,
        "--output",
        str(tmp_path / "output"),
    ]
    if action == "render":
        args += ["--manifest", str(manifest)]
    return args


def test_interrupted_render_only_replaces_uncommitted_frames(tmp_path, monkeypatch):
    args = resumable_args(tmp_path)
    rendered = []
    attempts = []

    def export(options, forwarded):
        progress = scene.output_progress(
            options,
            json.loads((options.output / cache.OutputProgress.marker).read_text())[
                "request"
            ],
        )
        progress.bind_inputs([])
        attempts.append(1)
        frames = []
        for ident in ("1", "2"):
            done = progress.get("frame:" + ident)
            if done is None:
                path = options.output / (ident + ".npz")
                if len(attempts) == 1 and ident == "2":
                    path.write_bytes(b"partial bytes")
                    return 1
                path.write_bytes(b"complete " + ident.encode())
                rendered.append(ident)
                done = {"id": ident, "prediction": path.name}
                progress.commit("frame:" + ident, [path.name], done)
            frames.append(done)
        (options.output / "render_manifest.json").write_text(
            json.dumps({"external_files": [], "frames": frames})
        )
        return 0

    monkeypatch.setattr(scene, "export", export)
    assert scene.main(args) == 1
    first = tmp_path / "output/1.npz"
    original = (cache.digest(first), first.stat().st_mtime_ns)
    marker = json.loads((tmp_path / "output" / cache.OutputProgress.marker).read_text())
    assert set(marker["receipts"]) == {"frame:1"}
    assert scene.main(args) == 0 and rendered == ["1", "2"] and len(attempts) == 2
    assert (cache.digest(first), first.stat().st_mtime_ns) == original
    assert (tmp_path / "output/2.npz").read_bytes() == b"complete 2"
    monkeypatch.setattr(
        scene, "export", lambda *a: pytest.fail("completed output relaunched export")
    )
    assert scene.main(args) == 0
    first.write_bytes(b"altered completed frame")
    with pytest.raises(ValueError, match="artifact changed"):
        scene.main(args)
    assert first.read_bytes() == b"altered completed frame"


def test_inspect_retry_and_request_identity_preserve_unknown_outputs(
    tmp_path, monkeypatch
):
    args = resumable_args(tmp_path, "inspect")
    calls = []

    def export(options, forwarded):
        progress = scene.output_progress(
            options,
            json.loads((options.output / cache.OutputProgress.marker).read_text())[
                "request"
            ],
        )
        progress.bind_inputs([])
        calls.append(1)
        path = options.output / "inventory.json"
        path.write_text(
            "{partial"
            if len(calls) == 1
            else json.dumps({"external_files": [], "objects": []})
        )
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(scene, "export", export)
    assert scene.main(args) == 1 and scene.main(args) == 0 and scene.main(args) == 0
    assert len(calls) == 2
    with pytest.raises(ValueError, match="request or source changed"):
        scene.main(args + ["--object", "other"])
    (tmp_path / "scene.blend").write_bytes(b"new source")
    with pytest.raises(ValueError, match="request or source changed"):
        scene.main(args)
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "note.txt").write_text("preserve")
    changed = list(args)
    changed[changed.index("--output") + 1] = str(unknown)
    with pytest.raises(ValueError, match="Unknown nonempty"):
        scene.main(changed)
    assert (unknown / "note.txt").read_text() == "preserve"


def test_resume_rejects_changed_external_asset_before_export(tmp_path, monkeypatch):
    args = resumable_args(tmp_path, "inspect")
    asset = tmp_path / "texture.bin"
    asset.write_bytes(b"original")

    def export(options, forwarded):
        progress = scene.output_progress(
            options,
            json.loads((options.output / cache.OutputProgress.marker).read_text())[
                "request"
            ],
        )
        inputs = [
            {"path": str(asset), "resolved": str(asset), "sha256": cache.digest(asset)}
        ]
        progress.bind_inputs(inputs)
        return 1

    monkeypatch.setattr(scene, "export", export)
    assert scene.main(args) == 1
    asset.write_bytes(b"changed")
    monkeypatch.setattr(
        scene,
        "export",
        lambda *a: pytest.fail("changed external input reached exporter"),
    )
    with pytest.raises(ValueError, match="external inputs changed"):
        scene.main(args)


def test_same_output_concurrent_requests_export_only_once(tmp_path, monkeypatch):
    import time

    args = resumable_args(tmp_path, "inspect")
    calls = []

    def export(options, forwarded):
        progress = scene.output_progress(
            options,
            json.loads((options.output / cache.OutputProgress.marker).read_text())[
                "request"
            ],
        )
        progress.bind_inputs([])
        calls.append(1)
        time.sleep(0.05)
        (options.output / "inventory.json").write_text(
            json.dumps({"external_files": [], "objects": []})
        )
        return 0

    monkeypatch.setattr(scene, "export", export)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: scene.main(args), range(2))) == [0, 0]
    assert len(calls) == 1
    assert not list((tmp_path / "output").glob("*.lock"))


@pytest.mark.parametrize("reason", ["stop", "deadline"])
def test_waiting_output_lock_obeys_stop_and_deadline(tmp_path, monkeypatch, reason):
    from threading import Event
    import time

    started = Event()
    stop = tmp_path / "stop"
    monkeypatch.setenv("VIBUILD_PARENT_STOPS", json.dumps([str(stop)]))
    monkeypatch.setenv("VIBUILD_DEADLINE", str(time.time() + 30))

    def waiter():
        started.set()
        with cache.entry_lock(tmp_path, ".output.scene-output"):
            pytest.fail("cancelled waiter acquired lock")

    with cache.entry_lock(tmp_path, ".output.scene-output"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiter)
            assert started.wait(timeout=1)
            if reason == "stop":
                stop.touch()
            else:
                monkeypatch.setenv("VIBUILD_DEADLINE", "1")
            with pytest.raises(
                (RuntimeError, TimeoutError), match="cancelled|deadline"
            ):
                future.result(timeout=1)


def test_writer_inherits_lock_after_launcher_handle_closes(tmp_path):
    import fcntl
    import subprocess

    with cache.entry_lock(tmp_path, "output", with_handle=True) as (_, handle):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(.3)"],
            pass_fds=(handle.fileno(),),
        )
    try:
        with (tmp_path / "output.lock").open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.wait(timeout=2)
        with cache.entry_lock(tmp_path, "output"):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
