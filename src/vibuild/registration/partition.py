#!/usr/bin/env python3
"""Build an offline spatial/topology review; never modify reconstruction inputs."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path


def graph_analysis(regions, interfaces, start=None):
    """Keep the physical graph intact; bridge blocks are planning diagnostics only."""
    ids = sorted(r["id"] for r in regions)
    if len(ids) != len(set(ids)):
        raise ValueError("Region IDs must be unique")
    graph = {u: set() for u in ids}
    observed = []
    edge_ids = set()
    pairs = set()
    for edge in interfaces:
        a, b = edge["regions"]
        if a not in graph or b not in graph or a == b:
            raise ValueError("Interface must connect two distinct known regions")
        if edge["id"] in edge_ids:
            raise ValueError("Interface IDs must be unique")
        edge_ids.add(edge["id"])
        if edge["status"] == "observed_transition":
            pair = tuple(sorted((a, b)))
            if pair in pairs:
                raise ValueError(
                    "Combine interface patches for each observed region pair"
                )
            pairs.add(pair)
            graph[a].add(b)
            graph[b].add(a)
            observed.append(edge)
    if start is not None and start not in graph:
        raise ValueError("Unknown start region")
    start = start or next(
        (u for u in ids if len(graph[u]) == 1), ids[0] if ids else None
    )
    entered = {}
    low = {}
    bridges = set()

    def visit(u, parent=None):
        entered[u] = low[u] = len(entered)
        for v in sorted(graph[u]):
            if v == parent:
                continue
            if v not in entered:
                visit(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > entered[u]:
                    bridges.add(tuple(sorted((u, v))))
            else:
                low[u] = min(low[u], entered[v])

    for u in ids:
        if u not in entered:
            visit(u)
    blocks = []
    owners = {}
    for u in ids:
        if u in owners:
            continue
        bid = f"C{len(blocks) + 1:02d}"
        members = []
        pending = [u]
        owners[u] = bid
        while pending:
            v = pending.pop()
            members.append(v)
            for w in sorted(graph[v]):
                if w not in owners and tuple(sorted((v, w))) not in bridges:
                    owners[w] = bid
                    pending.append(w)
        blocks.append(
            {
                "id": bid,
                "members": sorted(members),
                "cycle_contracted": len(members) > 1,
            }
        )
    tree = {b["id"]: set() for b in blocks}
    for a, b in bridges:
        tree[owners[a]].add(owners[b])
        tree[owners[b]].add(owners[a])
    components = []
    seen = set()
    for root in ([owners[start]] if start else []) + sorted(tree):
        if root in seen:
            continue
        members = []
        todo = [root]
        seen.add(root)
        while todo:
            u = todo.pop()
            members.append(u)
            for v in sorted(tree[u]):
                if v not in seen:
                    seen.add(v)
                    todo.append(v)
        components.append(sorted(members))
    result = {
        "start_region": start,
        "observed_edges": len(observed),
        "cycle_blocks": blocks,
        "quotient_edges": [
            list(pair)
            for pair in sorted({tuple(sorted((a, b))) for a in tree for b in tree[a]})
        ],
        "branch_blocks": [u for u in tree if len(tree[u]) > 2],
        "connected_components": len(components),
        "cycles_alone_form_one_chain": len(components) == 1
        and all(len(v) <= 2 for v in tree.values()),
        "policy": "Observed edges only. Each region gets a clean serial session. Each simple cycle gets an additional bounded closure session. Contraction is diagnostic, never a modeling assignment.",
    }
    result["execution_plan"] = execution_plan(
        blocks, tree, components, owners, observed, start
    )
    return result


def execution_plan(blocks, tree, components, owners, observed, start):
    """Plan paths of simple cycles and singletons; never absorb regions or invent edges.

    Every stage edits the same evolving global Blender project. Connection IDs select
    video/camera evidence for checks against its existing geometry; they are not
    transferable geometry. Branches and non-simple cycles need explicit ordering.
    """
    block_map = {b["id"]: b for b in blocks}
    by_pair = {frozenset(e["regions"]): e for e in observed}
    errors = []
    paths = []
    for component in components:
        root = (
            owners[start]
            if start and owners[start] in component
            else next((b for b in component if len(tree[b]) <= 1), component[0])
        )
        if any(len(tree[b]) > 2 for b in component):
            errors.append(
                "Branched quotient requires an explicit execution order: "
                + ", ".join(component)
            )
            continue
        if len(tree[root]) > 1:
            errors.append("Choose an endpoint start block for a single chain: " + root)
            continue
        path = []
        previous = None
        current = root
        while current is not None:
            path.append(current)
            nxt = sorted(tree[current] - ({previous} if previous else set()))
            previous, current = current, nxt[0] if nxt else None
        paths.append(path)
    for block in blocks:
        members = set(block["members"])
        internal = [e for e in observed if set(e["regions"]) <= members]
        if len(members) > 1 and (
            len(internal) != len(members)
            or any(sum(r in e["regions"] for e in internal) != 2 for r in members)
        ):
            errors.append(
                "Non-simple cycle requires explicit decomposition and cut constraints: "
                + block["id"]
            )
    base = {
        "status": "needs_plan" if errors else "planned",
        "errors": errors,
        "controller_implemented": True,
        "scene": {
            "mode": "single_global_project",
            "initial_geometry": "empty",
            "self_contained": True,
            "write_policy": "serial",
            "continuation": "Open the current project including all completed geometry; save changes to the same project.",
            "comparison": "Render the current project using original connection video frames and their fixed camera poses/intrinsics.",
        },
        "chains": [],
        "region_agent_count": 0,
        "closure_agent_count": 0,
        "execution_agent_count": 0,
        "max_active_modeling_agents": 1,
        "global_audit": None,
    }
    if errors:
        return base
    chains = []
    closure_count = 0
    acceptance = {}
    previous_stage = None
    for path in paths:
        steps = []
        bridge = None
        previous_exit = None
        for k, bid in enumerate(path):
            members = set(block_map[bid]["members"])
            internal = [e for e in observed if set(e["regions"]) <= members]
            entry = (
                next(r for r in bridge["regions"] if r in members)
                if bridge
                else (start if start in members else min(members))
            )
            outgoing = (
                next(
                    (
                        e
                        for e in observed
                        if {owners[r] for r in e["regions"]} == {bid, path[k + 1]}
                    ),
                    None,
                )
                if k + 1 < len(path)
                else None
            )
            order = [entry]
            while len(order) < len(members):
                candidates = [
                    (e["id"], next(r for r in e["regions"] if r != order[-1]))
                    for e in internal
                    if order[-1] in e["regions"]
                    and any(r not in order for r in e["regions"])
                ]
                order.append(min(candidates)[1])
            cut = (
                by_pair[frozenset([order[0], order[-1]])] if len(members) > 1 else None
            )
            cycle_steps = []
            for j, region in enumerate(order):
                incoming = by_pair[frozenset([order[j - 1], region])] if j else bridge
                source = order[j - 1] if j else previous_exit
                step = {
                    "id": region,
                    "kind": "region",
                    "regions": [region],
                    "cycle": bid if cut else None,
                    "depends_on": [previous_stage] if previous_stage else [],
                    "scene_after": previous_stage,
                    "connection_interfaces": [incoming["id"]] if incoming else [],
                    "check_dependencies": [source] if source else [],
                    "check_interfaces": [incoming["id"]] if incoming else [],
                    "deferred_interfaces": [cut["id"]]
                    if cut and region in cut["regions"]
                    else [],
                    "result": "locally_passed_closure_pending" if cut else "accepted",
                }
                steps.append(step)
                cycle_steps.append(region)
                previous_stage = region
            if cut:
                closure_count += 1
                closure_id = f"L{closure_count:02d}"
                steps.append(
                    {
                        "id": closure_id,
                        "kind": "closure",
                        "cycle": bid,
                        "regions": cut["regions"],
                        "cycle_regions": order,
                        "depends_on": [previous_stage],
                        "scene_after": previous_stage,
                        "check_dependencies": cycle_steps
                        + ([previous_exit] if bridge else []),
                        "connection_interfaces": [cut["id"]],
                        "write_interfaces": [cut["id"]],
                        "check_interfaces": sorted(e["id"] for e in internal)
                        + ([bridge["id"]] if bridge else []),
                        "deferred_interfaces": [],
                        "result": "cycle_accepted",
                    }
                )
                previous_stage = closure_id
            for region in members:
                acceptance[region] = steps[-1]["id"]
            if outgoing:
                previous_exit = next(r for r in outgoing["regions"] if r in members)
            bridge = outgoing
        chains.append({"id": f"chain_{len(chains) + 1:02d}", "steps": steps})
    base.update(
        chains=chains,
        region_agent_count=len(owners),
        closure_agent_count=closure_count,
        execution_agent_count=len(owners) + closure_count,
        global_audit={
            "depends_on": sorted(set(acceptance.values())),
            "accepted_region_stages": acceptance,
            "check_interfaces": sorted(e["id"] for e in observed),
            "requires": "The current global project, all regional and seam checks, fixed gates and held-out audit evidence. Unobserved mandatory connections remain unresolved.",
        },
    )
    return base


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
