"""Real Blender tests; enable with VIBUILD_TEST_BLENDER=/path/to/blender."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import signal
import time

import numpy as np
import pytest

import vibuild.blender.scene as scene
import vibuild.assessment.comparison as comparison


@pytest.fixture
def fixture_scene(tmp_path):
    blender = os.environ.get("VIBUILD_TEST_BLENDER")
    if not blender:
        pytest.skip(
            "set VIBUILD_TEST_BLENDER for real geometry/visibility/cache regression"
        )
    script = tmp_path / "fixture.py"
    script.write_text("""import bpy,sys,os
from pathlib import Path
out=Path(sys.argv[-1])
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene
def cube(name,location,scale,cid):
 bpy.ops.mesh.primitive_cube_add(size=2,location=location)
 o=bpy.context.object;o.name=name;o.scale=scale;o['semantic_class']=cid
 return o
cube('wall',(0,0,6),(8,8,.1),'wall')
pane=cube('pane',(-.9,0,3),(.45,.8,.02),'glass')
m=bpy.data.materials.new('glass');m.use_nodes=True
bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Transmission Weight'].default_value=1
pane.data.materials.append(m)
cube('mirror_scale',(1.0,.5,4),(-.4,.5,.2),'wall')
hidden=cube('render_hidden',(0,0,1),(5,5,.1),'hidden');hidden.hide_render=True
hidden=cube('object_hidden',(0,0,1.3),(5,5,.1),'hidden');hidden.hide_set(True)
c=bpy.data.collections.new('retired');s.collection.children.link(c);c.hide_render=True
hidden=cube('collection_hidden',(0,0,1.1),(5,5,.1),'hidden')
for old in list(hidden.users_collection):old.objects.unlink(hidden)
c.objects.link(hidden)
# Membership in a hidden collection must not hide another visible membership.
c.objects.link(pane)
src=bpy.data.collections.new('instance_source')
inst=cube('instance_mesh',(0,0,0),(.3,.3,.1),'wall')
for old in list(inst.users_collection):old.objects.unlink(inst)
src.objects.link(inst)
empty=bpy.data.objects.new('instance',None);empty.instance_type='COLLECTION';empty.instance_collection=src
s.collection.objects.link(empty);empty.location=(1,-.6,3.5)
animated=cube('moving',(-1.2,1.2,4),(.25,.25,.1),'wall')
animated.keyframe_insert('location',frame=1);animated.location.x=1.2;animated.keyframe_insert('location',frame=2)
s.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(out))
sys.stdout.flush();sys.stderr.flush();os._exit(0)
""")
    blend = tmp_path / "fixture.blend"
    subprocess.run(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            str(blend),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    manifest = tmp_path / "frames.json"
    frames = [
        {
            "id": str(i),
            "width": 48,
            "height": 32,
            "K": [[35, 0, 23.5], [0, 35, 15.5], [0, 0, 1]],
            "camera_to_world": np.eye(4).tolist(),
            "blender_frame": i,
        }
        for i in (1, 2)
    ]
    manifest.write_text(
        json.dumps({"frames": frames, "classes": {"wall": 1, "glass": 2, "hidden": 3}})
    )
    return blender, blend, manifest


def test_bvh_matches_blender_hidden_collections_glass_instances_animation(
    tmp_path, fixture_scene
):
    blender, blend, manifest = fixture_scene
    original = scene.digest(blend)
    for backend in ("scene", "bvh"):
        assert (
            scene.main(
                [
                    "render",
                    "--blend",
                    str(blend),
                    "--manifest",
                    str(manifest),
                    "--blender",
                    blender,
                    "--ray-backend",
                    backend,
                    "--validate-rays",
                    "256",
                    "--output",
                    str(tmp_path / backend),
                ]
            )
            == 0
        )
    for i in (1, 2):
        with (
            np.load(tmp_path / f"scene/{i}.npz") as a,
            np.load(tmp_path / f"bvh/{i}.npz") as b,
        ):
            assert np.allclose(a["depth"], b["depth"], atol=1e-5, equal_nan=True)
            assert np.array_equal(a["semantic"], b["semantic"])
            assert np.array_equal(a["instance"], b["instance"])
            assert np.allclose(a["normal_world"], b["normal_world"], atol=1e-5)
            assert 2 in b["semantic"] and 3 not in b["semantic"]
            assert b["depth"].dtype == np.float32 and b["depth"].min() > 2
    assert scene.digest(blend) == original


def test_render_and_comparison_reuse_without_blender_on_cache_hit(
    tmp_path, fixture_scene, monkeypatch
):
    blender, blend, manifest = fixture_scene
    args = [
        "render",
        "--blend",
        str(blend),
        "--manifest",
        str(manifest),
        "--blender",
        blender,
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    assert scene.main([*args, "--output", str(tmp_path / "first")]) == 0
    with monkeypatch.context() as m:
        m.setattr(scene, "export", lambda *a: pytest.fail("cache hit launched Blender"))
        assert scene.main([*args, "--output", str(tmp_path / "second")]) == 0
    for i in (1, 2):
        assert (tmp_path / f"first/{i}.npz").read_bytes() == (
            tmp_path / f"second/{i}.npz"
        ).read_bytes()
    frames = []
    for i in (1, 2):
        with np.load(tmp_path / f"first/{i}.npz") as a:
            data = dict(a)
        data.update(
            opaque=np.ones(data["depth"].shape, bool),
            confidence=np.ones(data["depth"].shape, np.float32),
        )
        np.savez(tmp_path / f"ref{i}.npz", **data)
        frames.append({"id": str(i), "reference": f"ref{i}.npz", "blender_frame": i})
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"frames": frames, "classes": {"wall": 1, "glass": 2}})
    )
    copied = tmp_path / "copy.blend"
    shutil.copy2(blend, copied)
    assert (
        comparison.main(
            [
                "--baseline",
                str(blend),
                "--candidate",
                str(copied),
                "--manifest",
                str(protocol),
                "--blender",
                blender,
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output",
                str(tmp_path / "compare"),
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "compare/comparison.json").read_text())
    assert (
        result["unchanged_blend"] and result["timings"]["candidate"]["render_cache_hit"]
    )
    assert (
        result["timings"]["candidate"]["evaluate_cache_hit"]
        and not result["regressions"]
    )
    # A different class map must miss even for the same source scene and cameras.
    value = json.loads(manifest.read_text())
    value["classes"]["wall"] = 9
    manifest.write_text(json.dumps(value))
    assert scene.main([*args, "--output", str(tmp_path / "changed")]) == 0
    assert not json.loads((tmp_path / "changed/render_manifest.json").read_text())[
        "cache"
    ]["hit"]


def test_external_asset_change_or_missing_asset_disables_reuse(tmp_path):
    asset = tmp_path / "texture.png"
    asset.write_bytes(b"original")
    report = {
        "external_files": [
            {
                "path": "//texture.png",
                "resolved": str(asset),
                "sha256": scene.digest(asset),
            }
        ]
    }
    assert scene.dependencies_match(report, tmp_path / "scene.blend")
    asset.write_bytes(b"changed")
    assert not scene.dependencies_match(report, tmp_path / "scene.blend")
    asset.unlink()
    assert not scene.dependencies_match(report, tmp_path / "scene.blend")


def test_interrupted_real_render_resumes_without_repeating_completed_frame(
    tmp_path, fixture_scene
):
    blender, blend, manifest = fixture_scene
    value = json.loads(manifest.read_text())
    # Leave a predictable window for cancellation after the first committed frame.
    value["frames"][1].update(
        width=512, height=384, K=[[350, 0, 255.5], [0, 350, 191.5], [0, 0, 1]]
    )
    manifest.write_text(json.dumps(value))
    out = tmp_path / "resumed"
    args = [
        "render",
        "--blend",
        str(blend),
        "--manifest",
        str(manifest),
        "--blender",
        blender,
        "--output",
        str(out),
    ]
    marker = out / ".scene_resume.json"
    with (tmp_path / "interrupted.log").open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(scene.__file__)), *args],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            until = time.monotonic() + 30
            while time.monotonic() < until:
                state = json.loads(marker.read_text()) if marker.exists() else {}
                if "frame:1" in state.get("receipts", {}):
                    break
                assert proc.poll() is None, (tmp_path / "interrupted.log").read_text()
                time.sleep(0.01)
            else:
                pytest.fail("first real frame did not commit before timeout")
            assert proc.poll() is None
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
    before = (scene.digest(out / "1.npz"), (out / "1.npz").stat().st_mtime_ns)
    committed = json.loads(marker.read_text())["receipts"]
    assert "complete" not in committed and "frame:2" not in committed
    (out / "2.npz").write_bytes(b"uncommitted interrupted frame")
    assert scene.main(args) == 0
    assert (scene.digest(out / "1.npz"), (out / "1.npz").stat().st_mtime_ns) == before
    with np.load(out / "2.npz", allow_pickle=False) as frame:
        assert frame["depth"].shape == (384, 512)
    report = json.loads((out / "render_manifest.json").read_text())
    assert [f["id"] for f in report["frames"]] == ["1", "2"]
    assert report["registered_cameras_unchanged"]
    assert set(json.loads(marker.read_text())["receipts"]) == {
        "frame:1",
        "frame:2",
        "complete",
    }


def test_completed_real_inspection_reuses_exact_output(
    tmp_path, fixture_scene, monkeypatch
):
    blender, blend, _ = fixture_scene
    out = tmp_path / "inspect_reused"
    args = [
        "inspect",
        "--blend",
        str(blend),
        "--object",
        "wall",
        "--mesh-detail",
        "--blender",
        blender,
        "--output",
        str(out),
    ]
    assert scene.main(args) == 0
    original = (
        scene.digest(out / "inventory.json"),
        (out / "inventory.json").stat().st_mtime_ns,
    )
    monkeypatch.setattr(
        scene,
        "export",
        lambda *a: pytest.fail("inspection unnecessarily relaunched Blender"),
    )
    assert scene.main(args) == 0
    assert (
        scene.digest(out / "inventory.json"),
        (out / "inventory.json").stat().st_mtime_ns,
    ) == original


def test_patch_inventory_and_native_mixed_grid_pipeline(tmp_path, fixture_scene):
    blender, blend, manifest = fixture_scene
    script = tmp_path / "edit.py"
    script.write_text(
        "import bpy\nbpy.data.objects['wall']['provenance']='reviewed fixture'\n"
    )
    original = scene.digest(blend)
    assert (
        scene.main(
            [
                "patch",
                "--blend",
                str(blend),
                "--script",
                str(script),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "patch"),
            ]
        )
        == 0
    )
    candidate = tmp_path / "patch/candidate.blend"
    assert candidate.is_file() and scene.digest(blend) == original
    assert not (tmp_path / "patch/.candidate.pending.blend").exists()
    assert (
        scene.main(
            [
                "inspect",
                "--blend",
                str(candidate),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "inspect"),
            ]
        )
        == 0
    )
    inv = json.loads((tmp_path / "inspect/inventory.json").read_text())
    assert any(m["name"] == "glass" for m in inv["materials"])
    assert (
        next(o for o in inv["objects"] if o["name"] == "wall")["provenance"][
            "provenance"
        ]
        == "reviewed fixture"
    )
    K = [[20, 0, 11.5], [0, 20, 7.5], [0, 0, 1]]
    pose = np.eye(4).tolist()
    np.savez(
        tmp_path / "ref.npz",
        depth=np.ones((16, 24), np.float32) * 6,
        K=K,
        camera_to_world=pose,
    )
    np.save(tmp_path / "rgb.npy", np.zeros((8, 12, 3), np.uint8))
    rgb = {
        "reference_rgb": "rgb.npy",
        "appearance_calibration": {
            "reference": {
                "K": [[10, 0, 5.5], [0, 10, 3.5], [0, 0, 1]],
                "camera_to_world": pose,
            }
        },
    }
    proto = {
        "classes": {"wall": 1},
        "frames": [
            {"id": "native", "reference": "ref.npz", **rgb},
            {"id": "missing", "reference": "missing.npz", **rgb},
        ],
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(proto))
    assert (
        comparison.main(
            [
                "--manifest",
                str(protocol),
                "--baseline",
                str(blend),
                "--build-script",
                str(script),
                "--output",
                str(tmp_path / "compare"),
                "--blender",
                blender,
                "--samples",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((tmp_path / "compare/candidate_scores.json").read_text())
    assert len(report["frames"]) == 2 and report["frames"][1]["status"] == "blocked"
    assert (
        report["appearance"]["scored_frames"] == 2
        and report["structure_status"] == "blocked"
    )
    with np.load(tmp_path / "compare/candidate/native.npz") as arrays:
        assert arrays["depth"].shape == (16, 24) and arrays["depth"].dtype == np.float32
    assert not (tmp_path / "compare/candidate/missing.npz").exists()
    assert scene.digest(blend) == original


def test_patch_error_does_not_publish_candidate(tmp_path, fixture_scene):
    blender, blend, _ = fixture_scene
    script = tmp_path / "fail.py"
    script.write_text('raise RuntimeError("expected")')
    assert (
        scene.main(
            [
                "patch",
                "--blend",
                str(blend),
                "--script",
                str(script),
                "--blender",
                blender,
                "--output",
                str(tmp_path / "fail"),
            ]
        )
        != 0
    )
    assert not (tmp_path / "fail/candidate.blend").exists()


def test_explicit_labels_are_reviewed_cached_and_do_not_mutate_scene(
    tmp_path, fixture_scene
):
    blender, blend, manifest = fixture_scene
    original = scene.digest(blend)
    review = tmp_path / "labels-review.json"
    review.write_text('{"review":"fixture pane identification"}')
    value = json.loads(manifest.read_text())
    value["object_labels"] = {
        "reviewer": "test reviewer",
        "evidence": [str(review)],
        "objects": {"pane": {"semantic_class": "wall", "instance_id": "reviewed-pane"}},
    }
    manifest.write_text(json.dumps(value))
    args = [
        "inspect",
        "--blend",
        str(blend),
        "--manifest",
        str(manifest),
        "--blender",
        blender,
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    assert scene.main([*args, "--output", str(tmp_path / "mapped")]) == 0
    report = json.loads((tmp_path / "mapped/inventory.json").read_text())
    assert (
        "pane" in report["label_coverage"]["wall"]
        and "glass" in report["missing_class_labels"]
    )
    assert scene.digest(blend) == original
    assert scene.main([*args, "--output", str(tmp_path / "reused")]) == 0
    assert json.loads((tmp_path / "reused/inventory.json").read_text())["cache"]["hit"]
    review.write_text('{"review":"changed evidence"}')
    assert scene.main([*args, "--output", str(tmp_path / "changed")]) == 0
    changed = json.loads((tmp_path / "changed/inventory.json").read_text())
    assert (
        not changed["cache"]["hit"]
        and changed["render_protocol_sha256"] != report["render_protocol_sha256"]
    )
    review.unlink()
    assert scene.main([*args, "--output", str(tmp_path / "missing-review")]) != 0
