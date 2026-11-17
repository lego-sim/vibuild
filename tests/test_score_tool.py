"""Fast contracts for regional evidence preparation; no Blender/model launch."""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.cwd()
import vibuild.assessment.visual as score


def frame(ident, x=0, split="model", coverage=1.0):
    return {
        "index": ident,
        "scene_center": [x, 0, 0],
        "camera_to_scene_rotation": np.eye(3).tolist(),
        "evaluation_split": split,
        "visible_regions": [
            {"region": "S01", "fraction_of_registered_observations": coverage}
        ],
    }


def test_selection_excludes_audit_and_spreads_camera_positions():
    frames = [
        frame(1),
        frame(2, 0.01),
        frame(3, 5),
        frame(4, -5),
        frame(5, 100, "audit"),
    ]
    scope = {"id": "S01", "suggested_first_frames": [5, 1, 2, 3, 4]}
    original = list(frames)
    chosen = score.select_views(frames, scope, 3)
    assert [f["index"] for f in chosen] == [1, 3, 4]
    assert frames == original
    assert score.select_views(frames, scope, 3) == chosen


def test_selection_prefers_local_cameras_over_distant_neighbor_views():
    frames = [
        {**frame(1, 100), "core_region": "S02"},
        *[{**frame(i, x), "core_region": "S01"} for i, x in [(2, 0), (3, 5), (4, -5)]],
    ]
    chosen = score.select_views(frames, {"id": "S01"}, 3)
    assert {f["index"] for f in chosen} == {2, 3, 4}


def test_selection_never_pads_two_good_views_with_weak_local_camera():
    frames = [
        {**frame(1), "core_region": "S01"},
        {**frame(2, 5), "core_region": "S01"},
        {**frame(3, 100, coverage=0.19), "core_region": "S01"},
    ]
    assert {f["index"] for f in score.select_views(frames, {"id": "S01"}, 3)} == {1, 2}
    assert len(score.select_views(frames[1:], {"id": "S01"}, 3)) == 1


def mesh(name, lower, upper):
    bounds = [
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]
    return {
        "name": name,
        "type": "MESH",
        "hide_render": False,
        "world_bounds": bounds,
        "materials": ["neutral"],
        "polygons": 12,
    }


def test_context_includes_latest_new_geometry_and_shared_context():
    inventory = {
        "objects": [
            mesh("new_local", [-0.2, -0.2, 2], [0.2, 0.2, 3]),
            mesh("shared_remote", [9, 9, 2], [10, 10, 3]),
            mesh("unrelated", [19, 19, 2], [20, 20, 3]),
        ]
    }
    scope = {"polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]], "core_z_interval": [1, 4]}
    view = {
        "id": "1",
        "camera_to_world": np.eye(4).tolist(),
        "K": [[50, 0, 49.5], [0, 50, 49.5], [0, 0, 1]],
        "width": 100,
        "height": 100,
    }
    result = score.object_context(
        inventory, scope, {"shared_read_only": [{"name": "shared_remote"}]}, [view]
    )
    assert {o["name"] for o in result} == {"new_local", "shared_remote"}
    local = next(o for o in result if o["name"] == "new_local")
    assert local["bounds"] == [-0.2, -0.2, 2.0, 0.2, 0.2, 3.0]
    assert local["projected_bounds"]["1"] == [0.445, 0.445, 0.545, 0.545]
    assert (
        next(o for o in result if o["name"] == "shared_remote")["projected_bounds"]
        == {}
    )


def test_compound_region_context_does_not_claim_the_empty_aabb_corner():
    scope = {
        "core_boxes": [
            {"bounds_min": [0, 0, 0], "bounds_max": [1, 3, 3]},
            {"bounds_min": [1, 0, 0], "bounds_max": [3, 1, 3]},
        ]
    }
    inventory = {
        "objects": [
            mesh("first", [0.1, 2, 1], [0.5, 2.5, 2]),
            mesh("second", [2, 0.1, 1], [2.5, 0.5, 2]),
            mesh("neighbor_in_envelope", [2, 2, 1], [2.5, 2.5, 2]),
        ]
    }
    assert {o["name"] for o in score.object_context(inventory, scope, {}, [])} == {
        "first",
        "second",
    }


def test_scaled_view_preserves_pixel_center_mapping(tmp_path):
    reference = tmp_path / "reference.jpg"
    Image.new("RGB", (100, 50)).save(reference)
    f = {
        **frame(1),
        "original_rgb": str(reference),
        "intrinsics_original": [[80, 0, 49.5], [0, 80, 24.5], [0, 0, 1]],
        "resolution": [100, 50],
    }
    view = score.scaled_view(f, tmp_path, 40)
    assert (view["width"], view["height"]) == (40, 20)
    np.testing.assert_allclose(view["K"], [[32, 0, 19.5], [0, 32, 9.5], [0, 0, 1]])
    assert view["geometry"] is False and view["appearance"] is True


def test_prepare_uses_explicit_partition_and_raw_calibration_only(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    record = tmp_path / "record"
    res = record / "resources"
    res.mkdir(parents=True)
    scene = tmp_path / "region.blend"
    scene.write_bytes(b"explicit local scene")
    raw = []
    associations = []
    for ident in range(4):
        image = res / f"{ident}.jpg"
        Image.new("RGB", (64, 32), "gray").save(image)
        split = "audit" if ident == 2 else "model"
        raw.append(
            {
                "name": str(ident),
                "index": ident,
                "center": [ident, 0, 0],
                "rotation": np.eye(3).tolist(),
                "K": [[40, 0, 31.5], [0, 40, 15.5], [0, 0, 1]],
                "width": 64,
                "height": 32,
                "original_rgb": str(image),
                "evaluation_split": split,
                "camera_zone": "OLD_ZONE_MUST_NOT_ENTER_PACKET",
            }
        )
        associations.append(
            {
                "name": str(ident),
                "core_region": "R01",
                "visible_regions": [
                    {
                        "region": "R01",
                        "fraction_of_registered_observations": 0.1
                        if ident == 3
                        else 0.8,
                    }
                ],
            }
        )
    score.save(res / "frames.json", raw)
    score.save(res / "coordinates.json", {"fixed": True})
    score.save(
        res / "index.json",
        {
            "frames": "frames.json",
            "coordinates": "coordinates.json",
            "partition": "missing-old-partition",
        },
    )
    partition = tmp_path / "partition.json"
    score.save(
        partition,
        {
            "schema_version": "partition_v1",
            "regions": [
                {
                    "id": "R01",
                    "name": "local",
                    "bounds_min": [-1, -1, 0],
                    "bounds_max": [3, 1, 4],
                }
            ],
            "frames": associations,
            "provenance": {
                "source_sha256": {
                    name: score.digest(res / name)
                    for name in ("frames.json", "coordinates.json")
                }
            },
        },
    )
    cfg = tmp_path / "config.json"
    score.save(cfg, {"runtime": {}})

    def tool(args, config, log):
        dest = Path(args[args.index("--output") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        if args[0] == "inspect":
            score.save(
                dest / "inventory.json",
                {"source_sha256": score.digest(scene), "objects": []},
            )
        else:
            views = score.load(args[args.index("--manifest") + 1])["frames"]
            for view in views:
                Image.new("RGB", (view["width"], view["height"])).save(
                    dest / f"{view['id']}.png"
                )

    monkeypatch.setattr(score, "run_tool", tool)
    monkeypatch.setattr(score.resources, "record_path", lambda _: record)
    args = SimpleNamespace(
        record="test",
        output=tmp_path / "scores",
        config=cfg,
        scene=scene,
        partition=partition,
        region=["R01"],
        views=2,
        width=64,
        samples=1,
        threads=1,
        workers=1,
    )
    score.prepare(args)
    packet = score.load(args.output / "R01/packet.json")
    assert {v["id"] for v in packet["views"]} == {"0", "1"}
    assert packet["scene_sha256"] == score.digest(scene)
    assert "OLD_ZONE" not in (args.output / "R01/packet.json").read_text()
    args.output = tmp_path / "frozen_scores"
    args.frame = ["1", "0"]
    score.prepare(args)
    assert [v["id"] for v in score.load(args.output / "R01/packet.json")["views"]] == [
        "1",
        "0",
    ]
    import pytest

    args.output = tmp_path / "invalid_frozen"
    args.frame = ["1", "2"]
    with pytest.raises(ValueError, match="non-audit"):
        score.prepare(args)
    args.frame = ["1", "3"]
    with pytest.raises(ValueError, match="coverage"):
        score.prepare(args)
    score.save(res / "coordinates.json", {"changed": True})
    with pytest.raises(ValueError, match="calibration/frames"):
        score.prepare(args)


def test_curved_region_uses_centerline_radius_when_polygon_is_null():
    inventory = {
        "objects": [
            mesh("new_curved", [1.5, -0.1, 2], [1.6, 0.1, 3]),
            mesh("outside", [4, 4, 2], [5, 5, 3]),
        ]
    }
    scope = {
        "polygon": None,
        "curved_centerline": [[0, 0], [1, 0]],
        "curved_radius": 0.5,
        "core_z_interval": [1, 4],
    }
    result = score.object_context(inventory, scope, {}, [])
    assert [o["name"] for o in result] == ["new_curved"]


def scored_fixture(tmp_path, completed=("S01", "S02")):
    from vibuild.assessment.scoring import RUBRIC, rubric_hash

    scene = tmp_path / "scene.blend"
    scene.write_bytes(b"frozen scene")
    out = tmp_path / "scores"
    protocol = {
        "scene": str(scene),
        "scene_sha256": score.digest(scene),
        "rubric_hash": rubric_hash(),
        "regions": [],
    }
    for region in ("S01", "S02"):
        work = out / region
        work.mkdir(parents=True)
        frozen_views = [
            {"id": "1", "width": 20, "height": 10},
            {"id": "2", "width": 20, "height": 10},
        ]
        views = []
        for view in frozen_views:
            card = work / f"view_{view['id']}.jpg"
            Image.new("RGB", (40, 74)).save(card)
            views.append({**view, "card": str(card), "card_sha256": score.digest(card)})
        packet = {
            "region": region,
            "scene_sha256": protocol["scene_sha256"],
            "rubric_hash": rubric_hash(),
            "views": views,
            "objects": [],
        }
        score.save(work / "packet.json", packet)
        protocol["regions"].append({"region": region, "views": frozen_views})
        if region in completed:
            result = {
                "region": region,
                "views": [
                    {
                        "id": v["id"],
                        "scores": {d: 90 for d in RUBRIC["dimensions"]},
                        "confidence": 0.9,
                        "notes": "Visible scope matches with minor differences",
                    }
                    for v in views
                ],
                "findings": [],
                "uncertainties": [],
            }
            score.save(work / ".runtime/final.json", result)
            score.save(
                work / ".runtime/scored.json",
                {
                    "packet_sha256": score.signature(packet),
                    "result_sha256": score.digest(work / ".runtime/final.json"),
                },
            )
    score.save(out / ".runtime/protocol.json", protocol)
    return out, protocol


def test_score_receipt_binds_actual_cards_packet_and_result(tmp_path):
    import pytest

    out, protocol = scored_fixture(tmp_path)
    work = out / "S01"
    assert score.verified_score(work, protocol)["score"] == 90
    card = work / "view_1.jpg"
    original = card.read_bytes()
    card.write_bytes(b"changed")
    with pytest.raises(ValueError, match="card has changed"):
        score.verified_packet(work, protocol)
    card.write_bytes(original)
    packet = score.load(work / "packet.json")
    packet["objects"].append({"name": "new_unreviewed"})
    score.save(work / "packet.json", packet)
    with pytest.raises(ValueError, match="receipt"):
        score.verified_score(work, protocol)
    packet["objects"] = []
    score.save(work / "packet.json", packet)
    result = score.load(work / ".runtime/final.json")
    result["views"][0]["scores"]["structure"] = 95
    score.save(work / ".runtime/final.json", result)
    with pytest.raises(ValueError, match="receipt"):
        score.verified_score(work, protocol)


def test_report_recomputes_raw_results_and_never_gives_partial_global_score(tmp_path):
    out, _ = scored_fixture(tmp_path, completed=("S01",))
    score.save(out / "S01/score.json", {"score": 999, "passed": True})
    score.report(out)
    summary = score.load(out / "scores.json")
    assert summary["score"] is None and summary["partial_score"] == 90
    assert summary["missing_regions"] == ["S02"] and not summary["passed"]
    assert score.load(out / "S01/score.json")["score"] == 90
    assert "No global score yet" in (out / "report.html").read_text()


def test_score_rejects_changed_scene_and_historical_rubric(tmp_path):
    import pytest

    out, protocol = scored_fixture(tmp_path)
    Path(protocol["scene"]).write_bytes(b"modified after scoring")
    with pytest.raises(ValueError, match="Scene changed"):
        score.verified_score(out / "S01", protocol)
    protocol["rubric_hash"] = "historical-language-version"
    with pytest.raises(ValueError, match="Historical rubric"):
        score.verified_packet(out / "S01", protocol)


def test_supplemental_diagnostic_card_bytes_are_verified(tmp_path):
    import pytest

    out, protocol = scored_fixture(tmp_path)
    work = out / "S01"
    card = work / "diagnostic.jpg"
    Image.new("RGB", (40, 74)).save(card)
    packet = score.load(work / "packet.json")
    packet["diagnostic_views"] = [
        {"card": str(card), "card_sha256": score.digest(card)}
    ]
    score.save(work / "packet.json", packet)
    score.verified_packet(work, protocol)
    card.write_bytes(b"changed supplemental evidence")
    with pytest.raises(ValueError, match="card has changed"):
        score.verified_packet(work, protocol)


def test_report_preserves_unknown_tool_call_count(tmp_path):
    out, _ = scored_fixture(tmp_path)
    score.report(out)
    assert score.load(out / "scores.json")["performance"]["tool_calls"] is None


def test_supplement_attaches_evidence_without_external_assessor(tmp_path):
    import pytest
    from types import SimpleNamespace

    out, protocol = scored_fixture(tmp_path, completed=())
    work = out / "S01"
    image = tmp_path / "detail.jpg"
    Image.new("RGB", (40, 74)).save(image)
    manifest = tmp_path / "diagnostics.json"
    score.save(
        manifest,
        {
            "scene_sha256": protocol["scene_sha256"],
            "images": [
                {"id": "detail", "path": "detail.jpg", "role": "Supplement only"}
            ],
        },
    )
    args = SimpleNamespace(output=out, region=["S01"], diagnostics=manifest)
    score.supplement(args)
    assert len(score.load(work / "packet.json")["diagnostic_views"]) == 1
    assert not (work / ".runtime/handoff.json").exists()
    assert not (work / ".runtime/assessor_instruction.txt").exists()
    score.save(work / ".runtime/state.json", {"state": "running"})
    with pytest.raises(ValueError, match="already started"):
        score.supplement(args)


def test_two_region_fake_model_cli_attaches_images_and_rebuilds_cached_summary(
    tmp_path,
):
    from helpers import config

    out, _ = scored_fixture(tmp_path, completed=())
    cfg = config(
        tmp_path,
        """
packet=json.JSONDecoder().raw_decode(prompt.split('PACKET=',1)[1])[0]
assert sys.argv.count('--image')==len(packet['views'])==2
assert 'objects' not in packet and 'object_details' not in packet and 'scene_sha256' not in packet
assert sys.argv[sys.argv.index('--sandbox')+1]=='read-only'
for i,arg in enumerate(sys.argv):
 if arg=='--image':assert Path(sys.argv[i+1]).is_file()
work=Path(sys.argv[sys.argv.index('--cd')+1])
marker=work/'fake_invocations'
marker.write_text(str(int(marker.read_text())+1) if marker.exists() else '1')
value={'region':packet['region'],'views':[{'id':v,'scores':{k:90 for k in ['structure','completeness','alignment','appearance','connectivity']},'confidence':.9,'notes':'Visible elements match'} for v in packet['views']],'findings':[],'uncertainties':[]}
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(value))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':100}}),flush=True)
""",
        session_minutes=1,
    )
    args = ["run", "--output", str(out), "--config", str(cfg), "--workers", "2"]
    assert score.main(args) == 0
    assert score.load(out / "scores.json")["passed"]
    for region in ("S01", "S02"):
        assert score.load(out / region / ".runtime/scored.json")["tool_calls"] == 0
        assert (out / region / "fake_invocations").read_text() == "1"
        (out / region / "score.json").unlink()
    assert score.main(args) == 0
    for region in ("S01", "S02"):
        assert score.load(out / region / "score.json")["score"] == 90
        assert (out / region / "fake_invocations").read_text() == "1"


def test_standalone_judge_attaches_supplements_and_completes_prepared_handoff(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    out, _ = scored_fixture(tmp_path)
    work = out / "S01"
    rt = work / ".runtime"
    packet = score.load(work / "packet.json")
    card = work / "supplement.jpg"
    Image.new("RGB", (20, 20)).save(card)
    packet["diagnostic_views"] = [
        {"id": "detail", "card": str(card), "card_sha256": score.digest(card)}
    ]
    score.save(work / "packet.json", packet)
    score.save(rt / "handoff.json", {"state": "prepared"})

    def run(*args, **kwargs):
        assert kwargs["read_only"] is True
        assert kwargs["images"] == [v["card"] for v in packet["views"]] + [str(card)]
        (rt / "events.jsonl").write_text("")
        return {
            "state": "completed",
            "thread_id": "fresh-cli",
            "started": 1,
            "finished": 2,
        }

    monkeypatch.setattr(score.sessions, "run", run)
    args = SimpleNamespace(
        output=out,
        region=["S01"],
        model="gpt-6-astra",
        reasoning_effort="high",
        config=tmp_path / "config.json",
    )
    score.judge_assessment(args)
    assert score.load(rt / "handoff.json")["context_mode"] == "standalone_fresh"
    assert score.load(rt / "handoff.json")["state"] == "completed"
    assert score.load(work / "score.json")["score"] == 90


def test_standalone_judge_cannot_overwrite_active_external_assessor(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    import pytest

    out, _ = scored_fixture(tmp_path)
    args = SimpleNamespace(output=out, region=["S01"])
    for state in ("running", "cancelled"):
        score.save(out / "S01/.runtime/handoff.json", {"state": state})
        with pytest.raises(ValueError, match="External assessor"):
            score.judge_assessment(args)


def test_whole_scene_selects_calibrated_views_without_sparse_tracks():
    frames = [{**frame(i, i), "visible_regions": []} for i in range(1, 5)]
    frames[-1]["evaluation_split"] = "audit"
    scope = {"id": "R01", "whole_scene": True}
    assert len(score.select_views(frames, scope, 3)) == 3
    assert score.select_views(frames, {"id": "R01"}, 3) == []
    assert [
        f["index"] for f in score.select_views(frames, scope, 3, held_out=True)
    ] == [4]
    assert all(f["visible_regions"] == [] for f in frames)
