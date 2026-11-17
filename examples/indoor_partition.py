#!/usr/bin/env python3
"""Independent vidmap-only planning experiment; never reads a scene or old partition.

Example recipe for one reviewed indoor capture; edit SPECS for another site.
Region constants are explicit architectural hypotheses from raw reference inspection.
They are NOT wall measurements, room segmentation, or a verified metric survey.
"""

import argparse
import copy
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


import numpy as np
from PIL import Image


# Absolute registered coordinates. Axis-aligned half-open cores share no volume.
SPECS = [
    (
        "Upper exhibition corridor",
        [-1.5, -2.2, -1.7],
        [38.5, 5, 3.5],
        "#e6a23c",
        "121floor",
        26,
    ),
    (
        "Lower exhibition corridor",
        [-1.5, -2.2, -6.8],
        [38.5, 5, -1.7],
        "#d36c52",
        "121floor",
        109,
    ),
    (
        "West corridor return stair",
        [-1.5, -11, -6.8],
        [4.5, -2.2, 3.5],
        "#eac85b",
        "121floor",
        141,
    ),
    (
        "East corridor return stair",
        [32.5, -11, -6.8],
        [38.5, -2.2, 3.5],
        "#cb8549",
        "121floor",
        80,
    ),
    (
        "Ground atrium and corridor approach",
        [-27, -10, -6.8],
        [-1.5, 3, -2],
        "#47a8bd",
        "spiral_staircase",
        31,
    ),
    (
        "Feature stair and lower approach",
        [-19, -23, -6.8],
        [-1.5, -10, 3.5],
        "#8cb369",
        "spiral_staircase",
        53,
    ),
    (
        "Upper rear landing",
        [-27, -23, -2],
        [-19, -10, 3.5],
        "#ac7cc4",
        "spiral_staircase",
        85,
    ),
    (
        "Ground rear lobby",
        [-27, -23, -6.8],
        [-19, -10, -2],
        "#5b8eab",
        "spiral_staircase",
        136,
    ),
    (
        "Rear enclosed stair",
        [-35, -23, -7.5],
        [-27, -10, 3.5],
        "#c18fc5",
        "spiral_staircase",
        117,
    ),
    (
        "Front entrance forecourt",
        [-35, 3, -8.5],
        [-1.5, 17, 10],
        "#709a73",
        "spiral_staircase",
        180,
    ),
    (
        "East glazed corridor return",
        [38.5, -26, -7.5],
        [52, 5, 4],
        "#69a99d",
        "outside",
        114,
    ),
    (
        "East entrance portico and apron",
        [52, -42, -10],
        [70, 5, 10],
        "#a39b72",
        "outside",
        142,
    ),
    (
        "Garden-facing exterior route",
        [-1, -60, -10],
        [52, -26, 10],
        "#82a65c",
        "outside",
        207,
    ),
    ("South exterior route", [-70, -78, -11], [52, -60, 12], "#8c9a56", "outside", 248),
    ("West exterior route", [-70, -60, -11], [-35, 3, 12], "#719177", "outside", 297),
    ("West entrance apron", [-35, -10, -8.5], [-27, 3, 10], "#749cac", "outside", 340),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qrotation(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_points(path):
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        ids = np.empty(n, dtype=np.uint64)
        points = np.empty((n, 3))
        colors = np.empty((n, 3), dtype=np.uint8)
        errors = np.empty(n)
        tracks = np.empty(n, dtype=np.int32)
        for i in range(n):
            row = struct.unpack("<QdddBBBd", f.read(43))
            ids[i], points[i], colors[i], errors[i] = row[0], row[1:4], row[4:7], row[7]
            (tracks[i],) = struct.unpack("<Q", f.read(8))
            f.seek(int(tracks[i]) * 8, 1)
    return ids, points, colors, errors, tracks


def read_images(path, wanted):
    out = []
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            row = struct.unpack("<idddddddi", f.read(64))
            chars = bytearray()
            while (ch := f.read(1)) != b"\0":
                if not ch:
                    raise ValueError("Truncated COLMAP image name")
                chars.extend(ch)
            name = chars.decode()
            (count,) = struct.unpack("<Q", f.read(8))
            data = f.read(count * 24)
            obs = np.frombuffer(data, dtype=[("xy", "<f8", (2,)), ("id", "<i8")])
            R = qrotation(row[1:5])
            out.append(
                {
                    "name": name,
                    "center": -R.T @ np.array(row[5:8]),
                    "rotation": R.T,
                    "point_ids": obs["id"].copy() if name in wanted else None,
                }
            )
    return out


def assign(points, regions):
    labels = np.full(len(points), -1, dtype=np.int16)
    for i, r in enumerate(regions):
        mask = np.zeros(len(points), dtype=bool)
        for box in r.get("core_boxes", [r]):
            mask |= np.all(
                (points >= box["bounds_min"]) & (points < box["bounds_max"]), axis=1
            )
        assert not np.any(mask & (labels >= 0)), "Core overlap"
        labels[mask] = i
    return labels


def shared_face(a, b):
    lo = np.maximum(a["bounds_min"], b["bounds_min"])
    hi = np.minimum(a["bounds_max"], b["bounds_max"])
    zero = np.flatnonzero(np.isclose(lo, hi, atol=1e-9))
    if len(zero) == 1 and np.all(hi >= lo) and np.all(np.delete(hi - lo, zero) > 0):
        return int(zero[0]), lo, hi
    return None


def graph_summary(regions, edges):
    adj = {r["id"]: set() for r in regions}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = set()
    components = []
    for v in adj:
        if v in seen:
            continue
        todo = [v]
        comp = []
        seen.add(v)
        while todo:
            u = todo.pop()
            comp.append(u)
            for w in adj[u] - seen:
                seen.add(w)
                todo.append(w)
        components.append(sorted(comp))
    return {
        "nodes": list(adj),
        "edges": [list(e) for e in edges],
        "degrees": {k: len(v) for k, v in adj.items()},
        "connected_components": components,
        "cycle_rank": len(edges) - len(adj) + len(components),
        "is_chain": len(components) == 1
        and len(edges) == len(adj) - 1
        and max(map(len, adj.values())) <= 2,
        "interpretation": "Observed camera transitions across explicit shared patches only; no edges inferred merely from overlap.",
    }


def apply_spatial_review(data):
    """Apply explicit user-reviewed scope/grouping without inventing trajectory edges.

    R05 is a two-box union, never its enclosing AABB. R06 remains a source-cell ID.
    Outside-video observations of indoor geometry remain useful evidence; only the
    excluded spatial regions are removed from modeling scope.
    """
    original = {r["id"]: r for r in data["regions"]}
    excluded = {f"R{i:02d}" for i in range(12, 17)}
    remap = {
        rid: ("R05" if rid == "R06" else rid) for rid in original if rid not in excluded
    }
    regions = [
        copy.deepcopy(r) for r in data["regions"] if r["id"] not in excluded | {"R06"}
    ]
    merged = next(r for r in regions if r["id"] == "R05")
    merged.update(
        name="Circular hall and spiral staircase",
        merged_from=["R05", "R06"],
        core_boxes=[
            {
                "id": rid,
                "bounds_min": original[rid]["bounds_min"],
                "bounds_max": original[rid]["bounds_max"],
            }
            for rid in ("R05", "R06")
        ],
        bounds_semantics="bounds_min/max are display envelopes only; core ownership is the union of core_boxes",
        grouping_source="User spatial review: keep circular hall and spiral staircase in one local modeling region",
    )
    merged["bounds_min"] = np.min(
        [b["bounds_min"] for b in merged["core_boxes"]], axis=0
    ).tolist()
    merged["bounds_max"] = np.max(
        [b["bounds_max"] for b in merged["core_boxes"]], axis=0
    ).tolist()
    merged["evidence"] += copy.deepcopy(original["R06"]["evidence"])
    for key in (
        "point_count",
        "robust_scope_point_count",
        "camera_count",
        "camera_center_frame_count",
    ):
        merged[key] = sum(original[rid].get(key, 0) for rid in ("R05", "R06"))
    merged["uncertainties"].append(
        "The enclosing AABB contains other regions; only the two original boxes are owned."
    )
    names = {
        "R07": "Level 2 elevator lobby",
        "R08": "Level 1 elevator lobby",
        "R09": "Stair between elevator lobbies",
    }
    for r in regions:
        if r["id"] in names:
            r.update(name=names[r["id"]], semantic_source="User spatial review")
    grouped = {}
    internal = []
    removed = []
    for edge in data["interfaces"]:
        if any(r in excluded for r in edge["regions"]):
            removed.append(edge["id"])
            continue
        pair = tuple(sorted({remap[r] for r in edge["regions"]}))
        if len(pair) == 1:
            internal.append(copy.deepcopy(edge))
            continue
        if pair not in grouped:
            grouped[pair] = {
                **copy.deepcopy(edge),
                "regions": list(pair),
                "evidence": [],
                "patches": [],
                "source_interface_ids": [],
                "crossing_count": 0,
            }
        target = grouped[pair]
        offset = len(target["evidence"])
        target["source_interface_ids"].append(edge["id"])
        target["evidence"] += copy.deepcopy(edge["evidence"])
        target["crossing_count"] += edge["crossing_count"]
        for patch in edge.get("patches", [edge]):
            target["patches"].append(
                {
                    "id": edge["id"] + "/" + patch.get("id", "P01"),
                    "bounds_min": patch["bounds_min"],
                    "bounds_max": patch["bounds_max"],
                    "source_interface": edge["id"],
                    "source_regions": edge["regions"],
                    "plane": edge.get("plane"),
                    "evidence_indices": [
                        offset + i
                        for i in patch.get(
                            "evidence_indices", range(len(edge["evidence"]))
                        )
                    ],
                }
            )
    for edge in grouped.values():
        edge["bounds_min"] = np.min(
            [p["bounds_min"] for p in edge["patches"]], axis=0
        ).tolist()
        edge["bounds_max"] = np.max(
            [p["bounds_max"] for p in edge["patches"]], axis=0
        ).tolist()
        # A compound region can meet the same neighbor on different planes.
        planes = {json.dumps(p["plane"], sort_keys=True) for p in edge["patches"]}
        edge["plane"] = edge["patches"][0]["plane"] if len(planes) == 1 else None
        edge["patch_semantics"] = (
            "Use individual patches and their planes. The envelope is for display only, never a continuous portal."
        )
    merged["internal_interfaces"] = internal
    frames = []
    for f in data["frames"]:
        counts = Counter()
        for v in f.get("visible_regions", []):
            if v["region"] in remap:
                counts[remap[v["region"]]] += v["observed_sparse_points"]
        if f["core_region"] not in remap and not counts:
            continue
        f["source_core_region"] = f["core_region"]
        f["core_region"] = remap.get(f["core_region"])
        f["visible_regions"] = [
            {
                "region": r,
                "observed_sparse_points": n,
                "fraction_of_registered_observations": round(
                    n / max(1, f["registered_sparse_observations"]), 5
                ),
            }
            for r, n in counts.most_common()
        ]
        frames.append(f)
    cameras = []
    for video in sorted({c["video"] for c in data["cameras"]}):
        segment = 0
        was_active = False
        for c in sorted(
            (c for c in data["cameras"] if c["video"] == video), key=lambda c: c["time"]
        ):
            if c["core_region"] not in remap:
                was_active = False
                continue
            if not was_active:
                segment += 1
            c["source_core_region"] = c["core_region"]
            c["core_region"] = remap[c["core_region"]]
            c["segment"] = segment
            cameras.append(c)
            was_active = True
    for r in regions:
        r["frame_ids"] = [f["name"] for f in frames if f["core_region"] == r["id"]]
        r["visible_frame_count"] = sum(
            any(v["region"] == r["id"] for v in f["visible_regions"]) for f in frames
        )
    cloud = data["point_cloud"]
    keep = [i for i, rid in enumerate(cloud["region_ids"]) if rid in remap]
    cloud = {
        **cloud,
        "positions": [cloud["positions"][i] for i in keep],
        "colors": [cloud["colors"][i] for i in keep],
        "region_ids": [remap[cloud["region_ids"][i]] for i in keep],
        "sampling": cloud["sampling"]
        + "; display restricted to active core unions after user scope review",
    }
    contacts = []
    for contact in data.get("geometric_contacts_without_transition", []):
        if any(r not in remap for r in contact["regions"]):
            continue
        pair = sorted({remap[r] for r in contact["regions"]})
        if len(pair) < 2:
            continue
        contacts.append(
            {**contact, "regions": pair, "source_regions": contact["regions"]}
        )
    validation = data["validation"]
    validation.update(
        source_camera_count=validation["camera_count"],
        source_reference_frame_count=validation["reference_frame_count"],
        source_assigned_raw_points=validation["assigned_raw_points"],
        excluded_scope_cameras=sum(
            original[r].get("camera_count", 0) for r in excluded
        ),
        excluded_scope_raw_points=sum(
            original[r].get("point_count", 0) for r in excluded
        ),
        camera_count=len(cameras),
        assigned_cameras=len(cameras),
        reference_frame_count=sum(f["core_region"] is not None for f in frames),
        assigned_camera_center_frames=sum(f["core_region"] is not None for f in frames),
        evidence_frame_count=len(frames),
        evidence_frames_with_camera_outside_scope=sum(
            f["core_region"] is None for f in frames
        ),
        assigned_raw_points=sum(r.get("point_count", 0) for r in regions),
        assigned_robust_scope_points=sum(
            r.get("robust_scope_point_count", 0) for r in regions
        ),
        coverage_note="Global raw-point denominator includes excluded outdoor regions. Do not compare this scoped coverage to the former all-region coverage.",
        core_geometry="Disjoint half-open boxes; R05 owns exactly the R05/R06 union, not its enclosing AABB",
    )
    cells = [(r["id"], box) for r in regions for box in r.get("core_boxes", [r])]
    overlaps = []
    for i, (rid, a) in enumerate(cells):
        for sid, b in cells[i + 1 :]:
            if rid != sid and np.all(
                np.minimum(a["bounds_max"], b["bounds_max"])
                > np.maximum(a["bounds_min"], b["bounds_min"])
            ):
                overlaps.append([rid, sid])
    if overlaps:
        raise ValueError(f"Active core unions overlap: {overlaps}")
    validation.update(
        active_core_box_count=len(cells), positive_volume_overlap_pairs=overlaps
    )
    validation["unassigned_robust_scope_points"] = (
        validation["robust_scope_points"] - validation["assigned_robust_scope_points"]
    )
    data.update(
        regions=regions,
        interfaces=list(grouped.values()),
        frames=frames,
        cameras=cameras,
        point_cloud=cloud,
        geometric_contacts_without_transition=contacts,
    )
    data["topology"] = graph_summary(
        regions, [e["regions"] for e in data["interfaces"]]
    )
    data["topology"].update(
        start_region="R11",
        reviewed_loop=["R05", "R08", "R09", "R07", "R05"],
        start_reason="Start with each reviewed region, then handle observed interfaces with ordinary J connection sessions. Provisional until local evidence acceptance.",
        layout={
            "R01": [90, 65],
            "R04": [235, 65],
            "R02": [235, 180],
            "R03": [90, 180],
            "R11": [435, 65],
            "R05": [235, 300],
            "R07": [435, 300],
            "R08": [235, 400],
            "R09": [435, 400],
            "R10": [65, 300],
        },
    )
    data["spatial_review"] = {
        "source": "User visual verification",
        "merged_regions": {"R05": ["R05", "R06"]},
        "excluded_regions": sorted(excluded),
        "excluded_interfaces": removed,
        "R01_R02": "No traversable edge. Shared horizontal box face only; use R03 or R04 stairs.",
        "R07_R09": "Existing observed edge I13, spiral_staircase video 97.56–97.80 seconds.",
        "loop": "R05 circular hall + spiral staircase → R08 level 1 elevator lobby → R09 stair → R07 level 2 elevator lobby → R05",
    }
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resources", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--point-budget", type=int, default=24000)
    args = ap.parse_args()
    root = args.resources.resolve()
    out = args.output.resolve()
    if out.is_relative_to(root):
        raise ValueError("Output must not modify frozen resources")
    if args.point_budget < 1:
        raise ValueError("Positive point budget required")
    out.mkdir(parents=True, exist_ok=True)
    input_files = [
        "index.json",
        "alignment/frames.json",
        "alignment/cameras.json",
        "alignment/coordinates.json",
        "reconstruction/vidmap/combined/rec/points3D.bin",
        "reconstruction/vidmap/combined/rec/images.bin",
    ]
    co = json.loads((root / input_files[3]).read_text())
    B = np.array(co["basis"])
    origin = np.array(co["origin"])
    scale = co["scale"]
    # Deliberately ignore all inherited partition, cache, refined depth, and visibility fields.
    fields = [
        "name",
        "video",
        "time",
        "center",
        "rotation",
        "K",
        "width",
        "height",
        "index",
        "image",
        "original_rgb",
        "evaluation_split",
        "split",
    ]
    frames = [
        {k: f[k] for k in fields if k in f}
        for f in json.loads((root / input_files[1]).read_text())
    ]
    excluded_frames = [
        f
        for f in frames
        if f.get("evaluation_split", f.get("split")) in ("audit", "holdout")
    ]
    frames = [f for f in frames if f not in excluded_frames]
    cameras = [
        {
            k: c[k]
            for k in [
                "name",
                "video",
                "time",
                "center",
                "rotation",
                "K",
                "width",
                "height",
            ]
        }
        for c in json.loads((root / input_files[2]).read_text())
    ]
    ids, raw, colors, errors, tracks = read_points(root / input_files[4])
    points = (raw - origin) @ B.T * scale
    raw_images = read_images(root / input_files[5], {f["name"] for f in frames})
    byname = {c["name"]: c for c in cameras}
    residuals = []
    rotation_residuals = []
    for im in raw_images:
        if im["name"] in byname:
            c = byname[im["name"]]
            residuals.append(
                np.linalg.norm((im["center"] - origin) @ B.T * scale - c["center"])
            )
            rotation_residuals.append(
                np.max(np.abs(B @ im["rotation"] - c["rotation"]))
            )
    assert max(residuals) < 1e-6, "Raw/aligned camera mismatch"
    reviewed = []
    for video in sorted({f["video"] for f in frames}):
        a = [f for f in frames if f["video"] == video]
        reviewed.extend(a[int(i)]["name"] for i in np.linspace(0, len(a) - 1, 18))
    regions = []
    for i, (name, lo, hi, color, video, time) in enumerate(SPECS):
        f = min(
            (f for f in frames if f["video"] == video),
            key=lambda f: abs(f["time"] - time),
        )
        regions.append(
            {
                "id": f"R{i + 1:02d}",
                "name": name,
                "bounds_min": lo,
                "bounds_max": hi,
                "color": color,
                "boundary_status": "heuristic_architectural_planning_cut",
                "evidence": [
                    {
                        "type": "raw_reference",
                        "frame": f["name"],
                        "video": video,
                        "time": f["time"],
                    }
                ],
                "uncertainties": [
                    "Box encloses a work region, not a measured room solid; slanted or curved envelopes are approximated.",
                    "Scene units have no verified conversion to metres.",
                ],
            }
        )
        reviewed.append(f["name"])
    labels = assign(points, regions)
    cp = np.array([c["center"] for c in cameras])
    cl = assign(cp, regions)
    fp = np.array([f["center"] for f in frames])
    fl = assign(fp, regions)
    # Robust scope means a reproducible central quantile crop, NOT verified inliers.
    quantile_lo, quantile_hi = np.quantile(points, [0.005, 0.995], axis=0)
    scope = np.all((points >= quantile_lo) & (points <= quantile_hi), axis=1)
    finite = np.isfinite(points).all(axis=1)
    lookup = {int(pid): i for i, pid in enumerate(ids)}
    obs_by_name = {
        im["name"]: im["point_ids"] for im in raw_images if im["point_ids"] is not None
    }
    thumbs = out / "thumbnails"
    thumbs.mkdir(exist_ok=True)
    reference_hashes = {}
    for i, f in enumerate(frames):
        f["core_region"] = regions[fl[i]]["id"] if fl[i] >= 0 else None
        seen = [
            lookup[int(pid)]
            for pid in obs_by_name.get(f["name"], [])
            if pid >= 0 and int(pid) in lookup
        ]
        counts = Counter(int(labels[p]) for p in seen)
        f["visible_regions"] = [
            {
                "region": regions[j]["id"],
                "observed_sparse_points": n,
                "fraction_of_registered_observations": round(n / len(seen), 5),
            }
            for j, n in counts.most_common()
            if j >= 0 and n >= 3
        ]
        f["registered_sparse_observations"] = len(seen)
        f["observations_outside_cores"] = counts.get(-1, 0)
        f["visibility_method"] = (
            "COLMAP observed point tracks; sparse feature support, not dense surface visibility or proof of occlusion-free coverage."
        )
        # Keep the fixed render calibration with each original RGB reference.
        f.update(
            {
                key: byname[f["name"]][key]
                for key in ("center", "rotation", "K", "width", "height")
            }
        )
        if f["name"] in reviewed:
            p = Path(f.get("original_rgb", f["image"]))
            image = Image.open(p)
            image.thumbnail((480, 270))
            target = thumbs / f"{f['video']}_{f['time']:.6f}.jpg"
            image.save(target, quality=84)
            f["thumbnail"] = target.relative_to(out).as_posix()
            f["inspected_reference"] = True
            reference_hashes[str(p.relative_to(root.resolve()))] = sha(p)
    for i, c in enumerate(cameras):
        c["core_region"] = regions[cl[i]]["id"] if cl[i] >= 0 else None
        for k in ["rotation", "K", "width", "height"]:
            c.pop(k)
    for i, r in enumerate(regions):
        r["point_count"] = int((labels == i).sum())
        r["robust_scope_point_count"] = int(((labels == i) & scope).sum())
        r["camera_count"] = int((cl == i).sum())
        r["camera_center_frame_count"] = int((fl == i).sum())
        r["visible_frame_count"] = sum(
            any(v["region"] == r["id"] for v in f["visible_regions"]) for f in frames
        )
        r["frame_ids"] = [f["name"] for f in frames if f["core_region"] == r["id"]]
        if r["camera_count"] == 0:
            r["uncertainties"].append(
                "No sampled camera center falls inside this core."
            )
    crossing_groups = defaultdict(list)
    unresolved = []
    for video in sorted({c["video"] for c in cameras}):
        seq = sorted(
            (c for c in cameras if c["video"] == video), key=lambda c: c["time"]
        )
        for a, b in zip(seq, seq[1:]):
            ra, rb = a["core_region"], b["core_region"]
            if ra == rb:
                continue
            event = {
                "video": video,
                "from": ra,
                "to": rb,
                "frames": [a["name"], b["name"]],
                "time_range": [a["time"], b["time"]],
                "sample_distance": float(
                    np.linalg.norm(np.array(a["center"]) - b["center"])
                ),
            }
            if not ra or not rb:
                event["reason"] = "Unassigned camera sample adjacent to core boundary"
                unresolved.append(event)
                continue
            ia = int(ra[1:]) - 1
            ib = int(rb[1:]) - 1
            face = shared_face(regions[ia], regions[ib])
            if face is None:
                event["reason"] = (
                    "Consecutive samples change cores without a shared finite face"
                )
                unresolved.append(event)
                continue
            axis, lo, hi = face
            p = np.array(a["center"])
            q = np.array(b["center"])
            t = (lo[axis] - p[axis]) / (q[axis] - p[axis])
            cross = p + t * (q - p)
            if (
                not (
                    0 <= t <= 1
                    and np.all(cross >= lo - 1e-8)
                    and np.all(cross <= hi + 1e-8)
                )
                or event["sample_distance"] > 5
            ):
                event["reason"] = (
                    "Interpolated crossing misses shared face or sample spacing exceeds 5 scene units"
                )
                unresolved.append(event)
                continue
            event["crossing_position"] = cross.tolist()
            crossing_groups[tuple(sorted((ia, ib)))].append(event)
    interfaces = []
    for (ia, ib), events in sorted(crossing_groups.items()):
        axis, lo, hi = shared_face(regions[ia], regions[ib])
        cross = np.array([e["crossing_position"] for e in events])
        padding = np.array([0.7, 0.7, 0.8])
        padding[axis] = 0
        patch_lo = np.maximum(lo, cross.min(0) - padding)
        patch_hi = np.minimum(hi, cross.max(0) + padding)
        # Do not bridge disconnected observed crossing sites with one alleged opening.
        clusters = []
        remaining = set(range(len(events)))
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            group = [seed]
            todo = [seed]
            while todo:
                current = todo.pop()
                near = [
                    j
                    for j in sorted(remaining)
                    if np.linalg.norm(cross[current] - cross[j]) <= 2.5
                ]
                for j in near:
                    remaining.remove(j)
                    group.append(j)
                    todo.append(j)
            clusters.append(group)
        patches = [
            {
                "id": f"P{j + 1:02d}",
                "bounds_min": np.maximum(lo, cross[group].min(0) - padding).tolist(),
                "bounds_max": np.minimum(hi, cross[group].max(0) + padding).tolist(),
                "evidence_indices": group,
            }
            for j, group in enumerate(clusters)
        ]
        normal = np.zeros(3)
        normal[axis] = 1
        interfaces.append(
            {
                "id": f"I{len(interfaces) + 1:02d}",
                "regions": [regions[ia]["id"], regions[ib]["id"]],
                "bounds_min": patch_lo.tolist(),
                "bounds_max": patch_hi.tolist(),
                "patches": patches,
                "plane": {
                    "axis": "xyz"[axis],
                    "coordinate": float(lo[axis]),
                    "normal": normal.tolist(),
                },
                "status": "observed_transition",
                "confidence": "observed_camera_path; provisional_physical_portal",
                "evidence": events,
                "crossing_count": len(events),
                "patch_semantics": "Use patches as finite crossing sites. Outer bounds are their envelope only and may bridge unsupported space. Sites cluster within 2.5 scene units and are padded by 0.7 horizontal / 0.8 vertical scene units; not measured doorway extents.",
                "uncertainties": [
                    "Linearly interpolated between registered samples; exact physical threshold and clearance unmeasured."
                ],
            }
        )
    contacts = []
    for i, a in enumerate(regions):
        for j, b in enumerate(regions[i + 1 :], i + 1):
            lo = np.maximum(a["bounds_min"], b["bounds_min"])
            hi = np.minimum(a["bounds_max"], b["bounds_max"])
            assert not np.all(hi > lo), "Positive-volume core overlap"
            face = shared_face(a, b)
            if face and (i, j) not in crossing_groups:
                contacts.append(
                    {
                        "regions": [a["id"], b["id"]],
                        "bounds_min": face[1].tolist(),
                        "bounds_max": face[2].tolist(),
                        "status": "unsupported",
                        "reason": "Geometric face contact only; excluded from topology.",
                    }
                )
    edges = [e["regions"] for e in interfaces]
    topology = graph_summary(regions, edges)
    topology["start_region"] = "R01"
    topology["start_reason"] = (
        "Strong repeated non-audit views and simple long corridor geometry; graph has cycles, so this is not claimed to be an endpoint."
    )
    topology["interpretation_cautions"] = [
        "The R05-R06-R08 triangle is partly induced by heuristic cuts through an open ground lobby; it does not demonstrate three separately bounded rooms.",
        "The two corridor end stairs and the feature/rear atrium stairs establish distinct observed routes between levels.",
        "A region cycle is a repeated route through work boxes, not automatically a physical room or corridor cycle.",
        "Contracting all overlapping cycles can collapse most of the building into one node; inspect articulation-preserving block quotients before recommending a serial order.",
    ]
    rng = np.random.default_rng(20261109)
    eligible = np.flatnonzero(scope & finite)
    take = np.sort(
        rng.choice(eligible, min(len(eligible), args.point_budget), replace=False)
    )
    # Include all-source checksums and separately list exactly which RGB references were opened.
    provenance = {
        "builder": "examples/indoor_partition.py",
        "builder_sha256": sha(Path(__file__)),
        "source_sha256": {p: sha(root / p) for p in input_files},
        "inspected_reference_sha256": reference_hashes,
        "excluded_inputs": [
            "All existing partitions, scenes, evaluations, scores, workflow runs, prior reports, refined depths, inherited zone fields"
        ],
        "reference_guard": "No audit or holdout RGB opened. Raw reconstruction is frozen and may have used all source frames upstream; no holdout fitting performed here.",
    }
    validation = {
        "positive_volume_overlap_pairs": [],
        "core_boundary_rule": "min <= coordinate < max; shared faces have zero volume and unique membership.",
        "camera_count": len(cameras),
        "assigned_cameras": int((cl >= 0).sum()),
        "unassigned_cameras": [c["name"] for c in cameras if c["core_region"] is None],
        "reference_frame_count": len(frames),
        "assigned_camera_center_frames": int((fl >= 0).sum()),
        "excluded_audit_holdout_frames": [f["name"] for f in excluded_frames],
        "raw_point_count": len(points),
        "assigned_raw_points": int((labels >= 0).sum()),
        "robust_scope_definition": "Joint per-axis 0.5%-99.5% quantile crop of raw registered sparse cloud; no statistical or geometric inlier guarantee.",
        "robust_scope_bounds_min": quantile_lo.tolist(),
        "robust_scope_bounds_max": quantile_hi.tolist(),
        "robust_scope_points": int(scope.sum()),
        "assigned_robust_scope_points": int(((labels >= 0) & scope).sum()),
        "outside_robust_scope_points": int((~scope).sum()),
        "unassigned_robust_scope_points": int((scope & (labels < 0)).sum()),
        "unresolved_trajectory_changes": unresolved,
    }
    result = {
        "schema_version": "partition_v1",
        "coordinate_system": dict(
            co,
            transform_equation="scene_column = scale * basis @ (raw_colmap_column - origin)",
            verification={
                "matched_camera_count": len(residuals),
                "max_center_residual": max(residuals),
                "max_rotation_element_residual": max(rotation_residuals),
                "basis_determinant": float(np.linalg.det(B)),
            },
        ),
        "regions": regions,
        "interfaces": interfaces,
        "geometric_contacts_without_transition": contacts,
        "cameras": cameras,
        "frames": frames,
        "point_cloud": {
            "source": "reconstruction/vidmap/combined/rec/points3D.bin",
            "sampling": "Seeded uniform sample from explicit central quantile scope",
            "positions": np.round(points[take], 4).tolist(),
            "colors": colors[take].tolist(),
            "region_ids": [regions[j]["id"] if j >= 0 else None for j in labels[take]],
        },
        "topology": topology,
        "validation": validation,
        "provenance": provenance,
    }
    result = apply_spatial_review(result)
    (out / "partition.json").write_text(
        json.dumps(result, separators=(",", ":")) + "\n"
    )
    active = result["regions"]
    review = result["spatial_review"]
    v = result["validation"]
    lines = [
        "# User-reviewed spatial partition",
        "",
        "Raw vidmap calibration is unchanged. Boundaries remain provisional; metric scale is unverified.",
        f"Active regions: {len(active)}. Observed region-pair interfaces: {len(result['interfaces'])}. Cycle rank: {result['topology']['cycle_rank']}.",
        "R05 owns two boxes: former R05 circular hall and R06 spiral staircase. Its AABB is not its core.",
        "R12–R16 are excluded from modeling. Outside-video evidence observing active interiors may still be used.",
        review["R01_R02"],
        review["R07_R09"],
        "Reviewed loop: " + review["loop"],
        f"Active camera coverage: {v['assigned_cameras']}/{v['camera_count']}; {v['excluded_scope_cameras']} source cameras outside requested scope.",
        "",
        "| Region | Name | Core boxes |",
        "|---|---|---|",
    ]
    for r in active:
        boxes = r.get(
            "core_boxes",
            [{"bounds_min": r["bounds_min"], "bounds_max": r["bounds_max"]}],
        )
        lines.append(f"| {r['id']} | {r['name']} | {json.dumps(boxes)} |")
    (out / "findings.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "regions": len(active),
                "interfaces": len(result["interfaces"]),
                "topology": result["topology"],
                "validation": v,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
