#!/usr/bin/env python3
"""Build an offline spatial/topology review; never modify reconstruction inputs."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path


def graph_analysis(regions, interfaces, start=None):
    """Order regional work, then assign each observed interface to a J connection."""
    ids = sorted(r["id"] for r in regions)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Regions must be nonempty and unique")
    graph = {ident: set() for ident in ids}
    observed = []
    edge_ids, pairs = set(), set()
    for edge in sorted(interfaces, key=lambda item: item["id"]):
        a, b = edge["regions"]
        if a not in graph or b not in graph or a == b:
            raise ValueError("Interface must connect two distinct known regions")
        if edge["id"] in edge_ids:
            raise ValueError("Interface IDs must be unique")
        edge_ids.add(edge["id"])
        if edge["status"] != "observed_transition":
            continue
        pair = frozenset((a, b))
        if pair in pairs:
            raise ValueError("Combine interface patches for each observed region pair")
        pairs.add(pair)
        graph[a].add(b)
        graph[b].add(a)
        observed.append(edge)
    if start is not None and start not in graph:
        raise ValueError("Unknown start region")
    start = start or next((ident for ident in ids if len(graph[ident]) == 1), ids[0])
    order, seen, components = [], set(), 0
    for root in [start, *ids]:
        if root in seen:
            continue
        components += 1
        pending = [root]
        while pending:
            ident = pending.pop()
            if ident in seen:
                continue
            seen.add(ident)
            order.append(ident)
            pending.extend(sorted(graph[ident] - seen, reverse=True))
    steps, built = [], set()
    previous = None
    for ident in order:
        steps.append(
            {
                "id": ident,
                "kind": "region",
                "regions": [ident],
                "depends_on": [previous] if previous else [],
                "scene_after": previous,
                "check_dependencies": sorted(graph[ident] & built),
                "check_interfaces": [],
                "result": "accepted",
            }
        )
        built.add(ident)
        previous = ident
    for index, edge in enumerate(observed, 1):
        ident = f"J{index:02d}"
        steps.append(
            {
                "id": ident,
                "kind": "connection",
                "regions": sorted(edge["regions"]),
                "depends_on": [previous],
                "scene_after": previous,
                "acceptance_dependencies": sorted(edge["regions"]),
                "connection_interfaces": [edge["id"]],
                "check_interfaces": [edge["id"]],
                "result": "accepted",
            }
        )
        previous = ident
    return {
        "start_region": start,
        "observed_edges": len(observed),
        "connected_components": components,
        "execution_plan": {
            "status": "planned",
            "chains": [{"id": "chain_01", "steps": steps}],
            "region_agent_count": len(ids),
            "connection_agent_count": len(observed),
            "execution_agent_count": len(steps),
            "max_active_modeling_agents": 1,
            "scene": {
                "mode": "single_global_project",
                "initial_geometry": "empty",
                "self_contained": True,
            },
            "global_audit": {
                "depends_on": [step["id"] for step in steps],
                "check_interfaces": [edge["id"] for edge in observed],
            },
        },
    }


def thumbnails(data, limit=3):
    """Embed a few original, non-audit reference images per region."""
    from PIL import Image

    frames = data.get("frames", [])
    for region in data["regions"]:

        def visible(f):
            return [
                v.get("region") if isinstance(v, dict) else v
                for v in f.get("visible_regions", [])
            ]

        relevant = [
            f
            for f in frames
            if f.get("split", f.get("evaluation_split")) not in ("audit", "holdout")
            and (f.get("core_region") == region["id"] or region["id"] in visible(f))
        ]
        # Prefer cameras inside this core before distant feature-track observations.
        relevant.sort(
            key=lambda f: (f.get("core_region") != region["id"], f.get("time", 0))
        )
        core = [f for f in relevant if f.get("core_region") == region["id"]]
        relevant = core if core else relevant
        chosen = []
        for n in range(min(limit, len(relevant))):
            f = relevant[
                round(n * (len(relevant) - 1) / max(1, min(limit, len(relevant)) - 1))
            ]
            path = Path(f.get("image", f.get("original_rgb", "")))
            if not path.is_file():
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((480, 270))
                stream = BytesIO()
                image.save(stream, format="JPEG", quality=70)
            chosen.append(
                {
                    "id": f.get("id", f.get("index", f.get("name"))),
                    "name": f.get("name", ""),
                    "video": f.get("video", ""),
                    "time": f.get("time"),
                    "core_region": f.get("core_region"),
                    "src": "data:image/jpeg;base64,"
                    + base64.b64encode(stream.getvalue()).decode(),
                }
            )
        region["thumbnails"] = chosen


def connection_references(data):
    """Index original calibrated frames near each crossing, without exporting geometry.

    Nearest sampled frames are leads for visual checking, not proof of seam coverage.
    Preserve time offsets so missing or distant evidence cannot appear exact.
    """
    frames = [
        f
        for f in data.get("frames", [])
        if f.get("evaluation_split") not in ("audit", "holdout")
        and f.get("split") not in ("audit", "holdout")
    ]
    for edge in data["interfaces"]:
        references = []
        for event in edge.get("evidence", []):
            candidates = [f for f in frames if f.get("video") == event.get("video")]
            for time in event.get("time_range", []):
                if not candidates:
                    continue
                frame = min(
                    candidates, key=lambda f: (abs(f["time"] - time), f["name"])
                )
                references.append(
                    {
                        "frame": frame["name"],
                        "crossing_time": time,
                        "time_offset_seconds": round(frame["time"] - time, 6),
                    }
                )
        edge["reference_frames"] = references
        edge["reference_policy"] = (
            "Look up original RGB and fixed center/rotation/K in frames; render the current global project. Inspect actual seam visibility; nearest timestamps alone do not establish coverage."
        )


def build(source, output, start=None):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output == source:
        raise ValueError("Output must not overwrite partition inputs")
    data = json.loads(source.read_text())
    connection_references(data)
    data["graph_analysis"] = graph_analysis(
        data["regions"],
        data["interfaces"],
        start
        or data.get("start_region")
        or data.get("topology", {}).get("start_region"),
    )
    if data.get("execution_plan"):
        data["graph_analysis"]["execution_plan"] = data["execution_plan"]
    thumbnails(data)
    template = Path(__file__).with_suffix(".html").read_text()
    serialized = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    serialized = (
        serialized.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__PARTITION_DATA__", serialized))
    print(json.dumps({"report": str(output), "regions": len(data["regions"])}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start")
    args = parser.parse_args(argv)
    build(args.input, args.output, args.start)


if __name__ == "__main__":
    main()
