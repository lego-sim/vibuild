"""Export fixed camera evidence and a reviewed finite partition from COLMAP."""

from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

from .colmap import read, rotation, undistort
from .partition import graph_analysis, build
from vibuild.resources import load, save, digest


def import_reconstruction(project, reconstruction, images, sources=None):
    project = Path(project)
    meta = load(project / "project.json")
    if meta.get("inputs_frozen"):
        raise ValueError(
            "Inputs are frozen; start another project for a new registration"
        )
    cameras, registered, points = read(reconstruction)
    images = Path(images).resolve()
    source_map = {f["name"]: f for f in sources or []}
    res = project / "inputs/resources"
    frames = []
    for index, item in enumerate(registered):
        relative = Path(item["name"])
        path = (images / relative).resolve()
        if not path.is_relative_to(images) or not path.is_file():
            raise ValueError(f"Registered image missing: {item['name']}")
        camera = cameras[item["camera"]]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (camera["width"], camera["height"]):
                raise ValueError(f"Camera/image resolution differs: {item['name']}")
            corrected, K = undistort(image, camera)
            original = res / "input/original_frames" / f"{index:06d}.png"
            original.parent.mkdir(parents=True, exist_ok=True)
            image.save(original)
            rgb = res / "input/frames" / f"{index:06d}.png"
            rgb.parent.mkdir(parents=True, exist_ok=True)
            corrected.save(rgb)
        source = source_map.get(item["name"])
        if source is None:
            try:
                stamp = float(relative.stem)
            except ValueError:
                raise ValueError(
                    "Supply frame sources with name/video/time for non-timestamp image names"
                ) from None
            source = {"video": relative.parent.as_posix(), "time": stamp}
        R = rotation(item["q"]).T
        center = -R @ np.asarray(item["t"])
        frames.append(
            dict(
                name=item["name"],
                camera=f"Camera_{index}",
                index=index,
                video=source["video"],
                time=source["time"],
                width=camera["width"],
                height=camera["height"],
                K=K.tolist(),
                rotation=R.tolist(),
                center=center.tolist(),
                original_rgb=str(rgb.relative_to(res)),
                raw_rgb=str(original.relative_to(res)),
                reference_mapping="COLMAP pinhole grid; distortion resampled from raw_rgb; pixel-center origin is (0,0)",
                evaluation_split=source.get(
                    "evaluation_split", "audit" if index % 10 == 9 else "fit"
                ),
                point_ids=item["point_ids"],
            )
        )
    raw = project / ".runtime/registration.json"
    save(raw, {"frames": frames, "points": {str(k): v for k, v in points.items()}})
    save(
        res / "index.json",
        {
            "version": 1,
            "frames": "alignment/frames.json",
            "cameras": "alignment/cameras.json",
            "coordinates": "alignment/coordinates.json",
            "videos": [
                str((project / p).relative_to(res))
                for p in meta["videos"]
                if (project / p).is_relative_to(res)
            ],
        },
    )
    meta.update(state="registered", registration=str(Path(reconstruction).resolve()))
    save(project / "project.json", meta)
    return {"registered_frames": len(frames), "next": f"vibuild partition {project}"}


def boxes(region):
    return region.get("core_boxes", [region])


def labels(points, regions):
    result = np.full(len(points), -1, dtype=int)
    for i, region in enumerate(regions):
        mask = np.zeros(len(points), dtype=bool)
        for box in boxes(region):
            mask |= np.all(
                (points >= box["bounds_min"]) & (points < box["bounds_max"]), axis=1
            )
        if np.any(mask & (result >= 0)):
            raise ValueError("Regional core boxes overlap")
        result[mask] = i
    return result


def validate_regions(regions):
    import re

    if not regions or len({r["id"] for r in regions}) != len(regions):
        raise ValueError("Regions must be nonempty and unique")
    occupied = []
    for region in regions:
        if not re.fullmatch(r"R\d{2,}", region["id"]):
            raise ValueError("Region IDs must be R01, R02, ...")
        for box in boxes(region):
            lo = np.array(box["bounds_min"], float)
            hi = np.array(box["bounds_max"], float)
            if (
                lo.shape != (3,)
                or hi.shape != (3,)
                or not np.isfinite([lo, hi]).all()
                or not np.all(hi > lo)
            ):
                raise ValueError("Each core needs finite increasing 3D bounds")
            for owner, a, b in occupied:
                if owner != region["id"] and np.all(
                    np.minimum(hi, b) > np.maximum(lo, a)
                ):
                    raise ValueError("Regional core boxes overlap")
            occupied.append((region["id"], lo, hi))
        region["bounds_min"] = np.min(
            [b["bounds_min"] for b in boxes(region)], axis=0
        ).tolist()
        region["bounds_max"] = np.max(
            [b["bounds_max"] for b in boxes(region)], axis=0
        ).tolist()
        region.setdefault("name", region["id"])
        region.setdefault("color", f"#{(len(occupied) * 483191) % 0xFFFFFF:06x}")


def prepare_partition(project, spec=None, reviewed=False):
    project = Path(project)
    meta = load(project / "project.json")
    if meta.get("inputs_frozen"):
        raise ValueError(
            "Inputs are already frozen; partition changes require a new project"
        )
    raw = load(project / ".runtime/registration.json")
    points = raw["points"]
    frames = raw["frames"]
    if spec is None:
        coordinates = np.array(
            [p["xyz"] for p in points.values()] + [f["center"] for f in frames]
        )
        lo = coordinates.min(axis=0) - 0.5
        hi = coordinates.max(axis=0) + 0.5
        spec = project / "partition.review.json"
        if not spec.exists():
            save(
                spec,
                {
                    "transform": {
                        "basis": np.eye(3).tolist(),
                        "origin": [0, 0, 0],
                        "scale": 1,
                    },
                    "regions": [
                        {
                            "id": "R01",
                            "name": "Review and split this scope",
                            "bounds_min": lo.tolist(),
                            "bounds_max": hi.tolist(),
                            "color": "#47a8bd",
                        }
                    ],
                },
            )
        if reviewed:
            raise ValueError(
                "Read the review template and pass it with --spec FILE --reviewed"
            )
    spec = Path(spec)
    value = load(spec)
    transform = value.get(
        "transform", {"basis": np.eye(3).tolist(), "origin": [0, 0, 0], "scale": 1}
    )
    B = np.array(transform["basis"], float)
    origin = np.array(transform["origin"], float)
    scale = float(transform["scale"])
    if (
        B.shape != (3, 3)
        or origin.shape != (3,)
        or not np.isfinite([*B.ravel(), *origin, scale]).all()
        or scale <= 0
        or not np.allclose(B @ B.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(B), 1)
    ):
        raise ValueError(
            "Transform needs a proper rotation, finite origin, and positive scale"
        )
    xyz = np.array([p["xyz"] for p in points.values()], float).reshape(-1, 3)
    xyz = (xyz - origin) @ B.T * scale
    for frame in frames:
        frame["center"] = ((np.array(frame["center"]) - origin) @ B.T * scale).tolist()
        frame["rotation"] = (B @ frame["rotation"]).tolist()
    regions = value["regions"]
    validate_regions(regions)
    point_labels = labels(xyz, regions)
    camera_labels = labels(np.array([f["center"] for f in frames]), regions)
    lookup = {int(ident): int(label) for ident, label in zip(points, point_labels)}
    for frame, label in zip(frames, camera_labels):
        seen = [lookup[ident] for ident in frame.pop("point_ids") if ident in lookup]
        counts = Counter(seen)
        frame["core_region"] = regions[label]["id"] if label >= 0 else None
        frame["visible_regions"] = [
            {
                "region": regions[i]["id"],
                "fraction_of_registered_observations": n / len(seen),
                "observed_sparse_points": n,
            }
            for i, n in counts.items()
            if i >= 0
        ]
    edges = value.get("interfaces", [])
    # Infer graph candidates only from actual trajectory crossings through finite shared faces.
    if not edges:
        by_video = {
            name: sorted(
                (f for f in frames if f["video"] == name), key=lambda f: f["time"]
            )
            for name in {f["video"] for f in frames}
        }
        pairs = {}
        by_id = {r["id"]: r for r in regions}
        for video, sequence in by_video.items():
            for a, b in zip(sequence, sequence[1:]):
                ra, rb = a["core_region"], b["core_region"]
                if not ra or not rb or ra == rb:
                    continue
                crossing = None
                for ba in boxes(by_id[ra]):
                    for bb in boxes(by_id[rb]):
                        lo = np.maximum(ba["bounds_min"], bb["bounds_min"])
                        hi = np.minimum(ba["bounds_max"], bb["bounds_max"])
                        zero = np.flatnonzero(np.isclose(lo, hi))
                        if len(zero) != 1 or np.any(hi < lo):
                            continue
                        axis = int(zero[0])
                        p = np.array(a["center"])
                        q = np.array(b["center"])
                        denom = q[axis] - p[axis]
                        if not denom:
                            continue
                        t = (lo[axis] - p[axis]) / denom
                        c = p + t * (q - p)
                        if (
                            0 <= t <= 1
                            and np.all(c >= lo - 1e-8)
                            and np.all(c <= hi + 1e-8)
                        ):
                            crossing = (lo, hi, c)
                if crossing is None:
                    continue
                key = tuple(sorted((ra, rb)))
                lo, hi, c = crossing
                edge = pairs.setdefault(
                    key,
                    {
                        "id": f"I{len(pairs) + 1:02d}",
                        "regions": list(key),
                        "status": "observed_transition",
                        "bounds_min": lo.tolist(),
                        "bounds_max": hi.tolist(),
                        "patches": [],
                        "evidence": [],
                    },
                )
                padding = np.array([0.7, 0.7, 0.8])
                padding[np.isclose(lo, hi)] = 0
                edge["patches"].append(
                    {
                        "bounds_min": np.maximum(lo, c - padding).tolist(),
                        "bounds_max": np.minimum(hi, c + padding).tolist(),
                    }
                )
                edge["evidence"].append(
                    {
                        "video": video,
                        "from": ra,
                        "to": rb,
                        "time_range": [a["time"], b["time"]],
                        "frames": [a["name"], b["name"]],
                        "crossing_position": c.tolist(),
                    }
                )
        edges = list(pairs.values())
    data = {
        "schema_version": "partition_v1",
        "regions": regions,
        "frames": [],
        "interfaces": edges,
        "topology": {"start_region": value.get("start_region", regions[0]["id"])},
        "point_cloud": {
            "positions": xyz[:: max(1, len(xyz) // 12000)].tolist(),
            "colors": [
                p["color"] for p in list(points.values())[:: max(1, len(xyz) // 12000)]
            ],
        },
        "cameras": [
            {k: f[k] for k in ("name", "video", "time", "center", "core_region")}
            for f in frames
        ],
    }
    for f in frames:
        data["frames"].append(
            {
                **f,
                "original_rgb": str(
                    (project / "inputs/resources" / f["original_rgb"]).resolve()
                ),
            }
        )
    graph = graph_analysis(regions, edges, data["topology"]["start_region"])
    data["execution_plan"] = value.get("execution_plan", graph["execution_plan"])
    if reviewed and data["execution_plan"]["status"] != "planned":
        raise ValueError(
            "Review the topology and provide an explicit execution_plan for branching or complex cycles"
        )
    if reviewed:
        steps = [
            step
            for chain in data["execution_plan"]["chains"]
            for step in chain["steps"]
        ]
        ids = [step["id"] for step in steps]
        if len(set(ids)) != len(ids) or {
            s["id"] for s in steps if s.get("kind", "region") == "region"
        } != {r["id"] for r in regions}:
            raise ValueError("Execution plan must include each region exactly once")
        visited = set()
        for step in steps:
            if not set(step.get("depends_on", [])) <= visited:
                raise ValueError("Execution dependencies must refer to earlier stages")
            visited.add(step["id"])
    partition = project / "partition.json"
    report = project / "partition.html"
    if reviewed:
        res = project / "inputs/resources"
        save(res / "alignment/frames.json", frames)
        save(res / "alignment/cameras.json", frames)
        save(res / "alignment/coordinates.json", transform)
        data["provenance"] = {
            "source_sha256": {
                str(p.relative_to(res)): digest(p)
                for p in [
                    res / "alignment/frames.json",
                    res / "alignment/coordinates.json",
                ]
            }
        }
    save(partition, data)
    build(partition, report)
    if not reviewed:
        return {
            "review": str(report),
            "spec": str(spec),
            "next": f"vibuild partition {project} --spec {spec} --reviewed",
        }
    return {
        "partition": str(partition),
        "review": str(report),
        "frames": frames,
        "execution_plan": data["execution_plan"],
    }
