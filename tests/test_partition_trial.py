"""Spatial ownership and serial planning must preserve regional scope and connection gates."""

import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
from vibuild.registration.partition import graph_analysis, build, connection_references

sys.path.insert(0, str(ROOT / "examples"))
from indoor_partition import assign, apply_spatial_review, SPECS, shared_face
from vibuild.cli import parser


def graph(names, pairs):
    return (
        [{"id": n} for n in names],
        [
            {"id": f"I{i}", "regions": list(pair), "status": "observed_transition"}
            for i, pair in enumerate(pairs)
        ],
    )


def test_branching_regions_and_connections_have_separate_ownership():
    regions, edges = graph(
        ["R01", "R02", "R03", "R04"], [("R01", "R02"), ("R01", "R03"), ("R03", "R04")]
    )
    plan = graph_analysis(regions, edges, start="R01")["execution_plan"]
    steps = plan["chains"][0]["steps"]
    assert [s["id"] for s in steps] == ["R01", "R02", "R03", "R04", "J01", "J02", "J03"]
    assert [s["kind"] for s in steps] == ["region"] * 4 + ["connection"] * 3
    assert all(s["check_interfaces"] == [] for s in steps[:4])
    assert sorted(i for s in steps[4:] for i in s["check_interfaces"]) == sorted(
        e["id"] for e in edges
    )
    assert all(
        set(s["acceptance_dependencies"]) == set(s["regions"]) for s in steps[4:]
    )


def test_loop_has_ordinary_connections_without_special_stages():
    regions, edges = graph(
        ["R01", "R02", "R03"], [("R01", "R02"), ("R02", "R03"), ("R03", "R01")]
    )
    plan = graph_analysis(regions, edges, start="R01")["execution_plan"]
    steps = plan["chains"][0]["steps"]
    assert [s["id"] for s in steps] == ["R01", "R02", "R03", "J01", "J02", "J03"]
    assert plan["region_agent_count"] == 3 and plan["connection_agent_count"] == 3
    assert all(set(s["regions"]) == set(e["regions"]) for s, e in zip(steps[3:], edges))
    assert plan["global_audit"]["check_interfaces"] == [e["id"] for e in edges]
    assert (
        graph_analysis(list(reversed(regions)), list(reversed(edges)), start="R01")[
            "execution_plan"
        ]
        == plan
    )


def test_disconnected_and_candidate_edges_do_not_create_connections():
    regions, edges = graph(
        ["R01", "R02", "R03", "R04"], [("R01", "R02"), ("R03", "R04")]
    )
    edges.append({"id": "candidate", "regions": ["R02", "R03"], "status": "candidate"})
    result = graph_analysis(regions, edges, start="R01")
    steps = result["execution_plan"]["chains"][0]["steps"]
    assert result["connected_components"] == 2 and result["observed_edges"] == 2
    assert len([s for s in steps if s["kind"] == "connection"]) == 2
    assert next(s for s in steps if s["id"] == "R03")["check_dependencies"] == []
    assert not any(set(s["regions"]) == {"R02", "R03"} for s in steps)


def test_each_step_continues_the_same_scene_in_dependency_order():
    regions, edges = graph(["R01", "R02"], [("R01", "R02")])
    plan = graph_analysis(regions, edges)["execution_plan"]
    steps = plan["chains"][0]["steps"]
    assert (
        plan["scene"]["mode"] == "single_global_project"
        and plan["scene"]["self_contained"]
    )
    for index, step in enumerate(steps):
        assert step["depends_on"] == ([steps[index - 1]["id"]] if index else [])
        assert step["scene_after"] == (steps[index - 1]["id"] if index else None)
    assert plan["global_audit"]["depends_on"] == [s["id"] for s in steps]


def test_connection_references_preserve_time_offsets_and_exclude_audit():
    data = {
        "frames": [
            {"name": "before", "video": "v", "time": 1, "split": "model"},
            {"name": "after", "video": "v", "time": 3, "split": "model"},
            {"name": "audit", "video": "v", "time": 2, "evaluation_split": "audit"},
            {"name": "holdout", "video": "v", "time": 2.1, "split": "holdout"},
            {"name": "other", "video": "w", "time": 2, "split": "model"},
        ],
        "interfaces": [
            {"id": "I1", "evidence": [{"video": "v", "time_range": [1.2, 2.1]}]},
            {"id": "I2", "evidence": [{"video": "missing", "time_range": [0, 1]}]},
        ],
    }
    connection_references(data)
    assert data["interfaces"][0]["reference_frames"] == [
        {"frame": "before", "crossing_time": 1.2, "time_offset_seconds": -0.2},
        {"frame": "after", "crossing_time": 2.1, "time_offset_seconds": 0.9},
    ]
    assert data["interfaces"][1]["reference_frames"] == []


def test_half_open_boxes_have_unique_3d_ownership():
    regions = [
        {"bounds_min": [0, 0, -2], "bounds_max": [2, 2, 0]},
        {"bounds_min": [0, 0, 0], "bounds_max": [2, 2, 2]},
    ]
    np.testing.assert_array_equal(
        assign(np.array([[1, 1, -1], [1, 1, 0], [1, 1, 1], [2, 1, 1]]), regions),
        [0, 1, 1, -1],
    )


def test_reviewed_box_union_keeps_lobbies_separate_and_preserves_real_loop():
    regions = [
        {
            "id": f"R{i + 1:02d}",
            "name": spec[0],
            "bounds_min": spec[1],
            "bounds_max": spec[2],
            "color": spec[3],
            "point_count": 1,
            "robust_scope_point_count": 1,
            "camera_count": 1,
            "camera_center_frame_count": 1,
            "evidence": [],
            "uncertainties": [],
        }
        for i, spec in enumerate(SPECS)
    ]
    lookup = {r["id"]: r for r in regions}
    pairs = [
        ("I07", "R05", "R06"),
        ("I08", "R05", "R08"),
        ("I11", "R06", "R07"),
        ("I12", "R06", "R08"),
        ("I13", "R07", "R09"),
        ("I14", "R08", "R09"),
        ("I10", "R05", "R16"),
    ]
    interfaces = []
    for ident, a, b in pairs:
        axis, lo, hi = shared_face(lookup[a], lookup[b])
        interfaces.append(
            {
                "id": ident,
                "regions": [a, b],
                "status": "observed_transition",
                "bounds_min": lo.tolist(),
                "bounds_max": hi.tolist(),
                "plane": {"axis": int(axis), "coordinate": float(lo[axis])},
                "evidence": [{"video": "v", "time_range": [1, 2]}],
                "crossing_count": 1,
                "patches": [
                    {
                        "id": "P01",
                        "bounds_min": lo.tolist(),
                        "bounds_max": hi.tolist(),
                        "evidence_indices": [0],
                    }
                ],
            }
        )
    cameras = [
        {
            "name": r["id"],
            "video": "v",
            "time": i,
            "core_region": r["id"],
            "center": ((np.array(r["bounds_min"]) + r["bounds_max"]) / 2).tolist(),
        }
        for i, r in enumerate(regions)
    ]
    cameras[1]["time"] = 20  # An active camera after an excluded exterior excursion.
    frames = [
        {
            "name": r["id"],
            "core_region": r["id"],
            "registered_sparse_observations": 1,
            "visible_regions": [{"region": r["id"], "observed_sparse_points": 1}],
        }
        for r in regions
    ]
    data = {
        "regions": regions,
        "interfaces": interfaces,
        "cameras": cameras,
        "frames": frames,
        "point_cloud": {
            "sampling": "fixture",
            "positions": [c["center"] for c in cameras],
            "colors": [[0, 0, 0]] * 16,
            "region_ids": [r["id"] for r in regions],
        },
        "validation": {
            "camera_count": 16,
            "reference_frame_count": 16,
            "assigned_raw_points": 16,
            "robust_scope_points": 16,
        },
    }
    result = apply_spatial_review(data)
    active = result["regions"]
    ids = [r["id"] for r in active]
    assert (
        len(active) == 10
        and "R06" not in ids
        and not set(ids) & {f"R{i}" for i in range(12, 17)}
    )
    centers = np.array(
        [c["center"] for c in cameras if c["name"] in ("R05", "R06", "R07", "R08")]
    )
    assert [ids[i] for i in assign(centers, active)] == ["R05", "R05", "R07", "R08"]
    edges = {frozenset(e["regions"]): e for e in result["interfaces"]}
    assert frozenset(["R01", "R02"]) not in edges
    for a, b in zip(["R05", "R08", "R09", "R07"], ["R08", "R09", "R07", "R05"]):
        assert frozenset([a, b]) in edges
    assert edges[frozenset(["R07", "R09"])]["id"] == "I13"
    combined = edges[frozenset(["R05", "R08"])]
    assert len(combined["patches"]) == 2 and combined["plane"] is None
    assert [p["evidence_indices"] for p in combined["patches"]] == [[0], [1]]
    assert (
        result["validation"]["camera_count"] == 11
        and result["validation"]["excluded_scope_cameras"] == 5
    )
    camera_map = {c["name"]: c for c in result["cameras"]}
    assert camera_map["R01"]["segment"] != camera_map["R02"]["segment"]
    assert result["validation"]["active_core_box_count"] == 11
    assert set(result["point_cloud"]["region_ids"]) <= set(ids)


def test_report_is_self_contained_and_escapes_embedded_data(tmp_path):
    regions, edges = graph(["R01"], [])
    regions[0].update(
        name="</script><script>bad()</script>",
        bounds_min=[0, 0, 0],
        bounds_max=[1, 1, 1],
        color="#123456",
    )
    source = tmp_path / "partition.json"
    source.write_text(
        json.dumps({"regions": regions, "interfaces": edges, "frames": []})
    )
    output = tmp_path / "review.html"
    build(source, output)
    html = output.read_text()
    assert "</script><script>bad()" not in html and "\\u003c/script" in html
    assert 'src="https:' not in html and 'src="http:' not in html
    assert "__PARTITION_DATA__" not in html


@pytest.mark.parametrize("command", ["start", "workflow", "workflow-task"])
def test_retired_orchestration_has_no_cli_alias(command):
    with pytest.raises(SystemExit):
        parser().parse_args([command, "--record", "record"])


def test_partition_entry_requires_a_project_and_score_is_an_advanced_tool():
    args = parser().parse_args(
        ["partition", "demo", "--spec", "scope.json", "--reviewed"]
    )
    assert args.command == "partition" and args.reviewed
    assert (
        parser().parse_args(["tools", "score", "report", "--output", "scores"]).name
        == "score"
    )
