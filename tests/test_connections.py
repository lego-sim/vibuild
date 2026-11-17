"""Connection evidence must cover routes and approaches without inventing adjacency."""

import copy
import json
import pytest
from vibuild.registration.connections import (
    prepare,
    authoritative_partition,
    digest,
    write_packet,
)


def frame(name, time, core="C", split="model"):
    return {
        "name": name,
        "video": "v",
        "time": time,
        "center": [time, 0, 0],
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "K": [[10, 0, 5], [0, 10, 5], [0, 0, 1]],
        "width": 10,
        "height": 10,
        "original_rgb": "/tmp/example.jpg",
        "core_region": core,
        "visible_regions": [
            {"region": r, "observed_sparse_points": 30} for r in ["A", "B"]
        ],
        "evaluation_split": split,
        "split": "model",
    }


def edge(ident, a, b, status="observed_transition"):
    return {
        "id": ident,
        "regions": [a, b],
        "status": status,
        "bounds_min": [0, 0, 0],
        "bounds_max": [0, 1, 1],
        "evidence": [
            {
                "video": "v",
                "from": a,
                "to": b,
                "time_range": [0, 1],
                "frames": ["zero", "one"],
            }
        ],
    }


def partition():
    return {
        "regions": [{"id": r} for r in "ABCD"],
        "interfaces": [
            edge("I1", "A", "B"),
            edge("I2", "B", "C"),
            edge("I3", "C", "D", "candidate"),
        ],
        "frames": [
            frame("before", -3),
            frame("cross", 0.5),
            frame("after", 4),
            frame("context", 20),
        ],
        "validation": {},
    }


def test_route_expands_known_intermediate_and_keeps_unknown_adjacency():
    p = prepare(partition(), ["A", "C"], "J01")
    assert p["regions"] == ["A", "B", "C"] and p["intermediate_regions"] == ["B"]
    assert p["relations"] == [
        {
            "regions": ["A", "C"],
            "direct_edge": "not_observed",
            "known_route": ["A", "B", "C"],
            "status": "observed_route",
        }
    ]
    assert p["interfaces"] == ["I1", "I2"]
    assert p["writer_reserved"] is False and p["write_scope"]["objects"] == []
    assert p["envelope"]["boxes"][0]["bounds_min"][0] < -3
    assert p["envelope"]["boxes"][0]["bounds_max"][0] > 4
    assert len(p["mandatory_components"]) == 12
    assert all(c["status"] == "unreviewed" for c in p["mandatory_components"])


def test_candidate_edge_and_disconnected_request_stay_unresolved():
    p = prepare(partition(), ["A", "D"], "J02")
    assert p["status"] == "needs_evidence" and p["interfaces"] == []
    assert p["relations"][0]["known_route"] == []


def test_sources_include_outside_camera_visible_context_and_temporal_approaches():
    p = prepare(partition(), ["A", "B"], "J01")
    assert {e["role"] for e in p["evidence"]} == {
        "crossing",
        "approach_from",
        "approach_to",
        "context",
    }
    assert all(e["frame"]["core_region"] == "C" for e in p["evidence"])
    assert p["coverage"][0]["missing_roles"] == []
    assert p["coverage"][0]["visual_confirmation"] == "pending"


def test_holdouts_excluded_even_if_split_fields_disagree_or_partition_marks_them():
    data = partition()
    data["frames"][0]["split"] = "audit"
    data["frames"][1]["evaluation_split"] = "holdout"
    data["validation"]["excluded_audit_holdout_frames"] = ["context"]
    p = prepare(data, ["A", "B"], "J01")
    assert {e["source_frame"] for e in p["evidence"]} == {"after"}
    assert set(p["coverage"][0]["missing_roles"]) == {
        "crossing",
        "approach_from",
        "context",
    }


def test_missing_reverse_view_does_not_claim_coverage_and_selection_is_deterministic():
    data = partition()
    data["frames"] = data["frames"][:2]
    for f in data["frames"]:
        f["visible_regions"] = []
    p = prepare(data, ["A", "B"], "J01")
    assert set(p["coverage"][0]["missing_roles"]) == {"approach_to", "context"}
    assert p == prepare(
        {**data, "frames": list(reversed(data["frames"]))}, ["B", "A"], "J01"
    )


def test_explicit_scope_rejects_unbounded_or_unrelated_region_or_uncovered_edge():
    box = {
        "bounds_min": [-5, -5, -5],
        "bounds_max": [5, 5, 5],
        "regions": ["A", "B"],
        "interfaces": ["I1"],
    }
    p = prepare(partition(), ["A", "B"], "J01", {"boxes": [box]})
    assert p["envelope"]["status"] == "explicit"
    for key, value in [
        ("bounds_max", [float("inf"), 5, 5]),
        ("regions", ["D"]),
        ("interfaces", ["I3"]),
    ]:
        bad = {**box, key: value}
        with pytest.raises(ValueError):
            prepare(partition(), ["A", "B"], "J01", {"boxes": [bad]})
    with pytest.raises(ValueError):
        prepare(partition(), ["A", "C"], "J01", {"boxes": [box]})


def test_raw_calibration_overrides_partition_and_source_hash_mismatch_fails(tmp_path):
    root = tmp_path / "resources"
    (root / "alignment").mkdir(parents=True)
    (root / "index.json").write_text(
        json.dumps(
            {
                "frames": "alignment/frames.json",
                "coordinates": "alignment/coordinates.json",
            }
        )
    )
    data = partition()
    raw = copy.deepcopy(data["frames"])
    raw[0]["center"] = [99, 98, 97]
    (root / "alignment/frames.json").write_text(json.dumps(raw))
    (root / "alignment/coordinates.json").write_text("{}")
    data["provenance"] = {
        "source_sha256": {
            p: digest(root / p)
            for p in ["alignment/frames.json", "alignment/coordinates.json"]
        }
    }
    assert authoritative_partition(data, root)["frames"][0]["center"] == [99, 98, 97]
    (root / "alignment/frames.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        authoritative_partition(data, root)


def test_packet_views_keep_registered_cameras_and_fixed_source_hashes(tmp_path):
    data = partition()
    rgb = tmp_path / "rgb.jpg"
    rgb.write_bytes(b"fixture")
    for f in data["frames"]:
        f["original_rgb"] = str(rgb)
    part = tmp_path / "partition.json"
    part.write_text(json.dumps(data))
    packet = prepare(data, ["A", "B"], "J01")
    write_packet(packet, tmp_path / "packet", part)
    views = json.loads((tmp_path / "packet/views.json").read_text())["frames"]
    assert views[0]["camera_to_world"][0][-1] == 0.5
    assert views[0]["K"] == data["frames"][1]["K"]
    saved = json.loads((tmp_path / "packet/connection.json").read_text())
    assert saved["evidence"][0]["frame"]["original_sha256"] == digest(rgb)
    assert saved["partition_sha256"] == digest(part)


def test_intermediate_volume_preserves_exact_core_union_not_enclosing_aabb():
    data = partition()
    data["regions"][1].update(
        bounds_min=[0, 0, 0],
        bounds_max=[10, 10, 10],
        core_boxes=[
            {"bounds_min": [0, 0, 0], "bounds_max": [2, 2, 2]},
            {"bounds_min": [8, 8, 8], "bounds_max": [10, 10, 10]},
        ],
    )
    p = prepare(data, ["A", "C"], "J01")
    intermediate = [b for b in p["envelope"]["boxes"] if b.get("reason")]
    assert [b["bounds_max"] for b in intermediate] == [[2, 2, 2], [10, 10, 10]]
    assert p["write_scope"]["status"] == "needs_visual_scope_confirmation"
    assert all(e["primary"] for e in p["evidence"])
    assert len({c["id"] for c in p["mandatory_components"]}) == len(
        p["mandatory_components"]
    )


def test_approach_prefers_visible_structure_over_nearest_occluded_camera():
    data = partition()
    data["frames"][0]["visible_regions"] = [
        {"region": "A", "observed_sparse_points": 2}
    ]
    data["frames"].append(frame("visible_approach", -5))
    p = prepare(data, ["A", "B"], "J01")
    assert (
        next(e["source_frame"] for e in p["evidence"] if e["role"] == "approach_from")
        == "visible_approach"
    )
