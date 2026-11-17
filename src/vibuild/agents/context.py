"""Task-specific material index and one reusable job template; no model or tool execution."""

import sys
from vibuild.runtime.cache import digest
from vibuild.config import settings
from vibuild.runtime.jobs import absolute, option, queue, failure_detail
from vibuild.runtime.monitor import read, write, guidance


def material_index(root, config, directory=None):
    status = read(root / "status.json", {})
    region = status.get("active_region") or status.get("region")
    session = settings(config).get("session", {})
    score_options = []
    for flag, key in [("--model", "model"), ("--reasoning-effort", "reasoning_effort")]:
        if session.get(key):
            score_options += [flag, session[key]]
    scene_hash = digest(root / "scene.blend")
    index = {"region": region, "analysis_python": sys.executable}
    notice = guidance(root)
    if directory:
        (directory / ".runtime").mkdir(parents=True, exist_ok=True)
        write(
            directory / ".runtime/binding.json",
            {
                "task": str(root),
                "scene_sha256": scene_hash,
                "guidance_sha256": notice["sha256"] if notice else None,
            },
        )
    if notice:
        index["guidance"] = {k: notice[k] for k in ("pending", "text")}
    plan = read(root / "plan.json", {})
    index["inputs"] = {k: plan[k] for k in ("record", "partition") if k in plan}
    from vibuild.paths import PACKAGE

    index["geometry_guide"] = str(PACKAGE / "agents/prompts/curves.md")
    index["geometry_library"] = str(PACKAGE / "blender/curves.py")
    if not region:
        return index
    base = root / region
    materials = sorted(
        str(p.relative_to(base))
        for p in (base / "evidence").rglob("*")
        if p.is_file() and p.suffix in {".json", ".py", ".md"}
    )
    index["source_materials"] = {
        "base": str(base),
        "files": materials[:24],
        "remaining": max(0, len(materials) - 24),
    }
    index["scope_files"] = [
        str(p) for p in (base / "stage.json", base / "region.json") if p.is_file()
    ]
    index["scope"] = read(base / "region.json", {}).get("region", {})
    stage = read(base / "stage.json", {})
    index["dependencies"] = {
        k: stage[k]
        for k in (
            "kind",
            "regions",
            "depends_on",
            "check_interfaces",
            "deferred_interfaces",
        )
        if k in stage
    }
    step = next(
        (s for s in read(root / "plan.json", {}).get("steps", []) if s["id"] == region),
        {},
    )
    if step.get("closeout"):
        index["closeout"] = {
            k: v for k, v in step["closeout"].items() if "sha256" not in k
        }
    index["routing"] = {
        k: step[k]
        for k in (
            "work_ready",
            "work_mode",
            "blocked_by",
            "resume_region",
            "work_objective",
            "acceptance_dependencies",
        )
        if k in step
    }
    if step.get("context_packet"):
        packet = read(absolute(step["context_packet"]), {})
        index["reference_evidence"] = {
            "region": packet.get("region"),
            "matches_current_scene": packet.get("scene_sha256") == scene_hash,
            "views": [
                {k: v[k] for k in ("id", "card")} for v in packet.get("views", [])
            ],
        }
    # Examine receipts locally, never load whole histories into a model prompt.
    build = prepare = None
    build_hash = None
    latest = None
    next_round = 1
    for path in sorted(queue(root).glob("*/request.json"), reverse=True):
        request = read(path, {})
        if request.get("region") != region:
            continue
        next_round = max(next_round, int(request.get("round") or 0) + 1)
        result = read(path.parent / "result.json", {})
        if latest is None and result.get("steps"):
            latest = (path, request, result)
        for step, receipt in zip(request.get("steps", []), result.get("steps", [])):
            if receipt.get("state") != "completed":
                continue
            key = (step["tool"], step["args"][0])
            if key == ("scene", "iterate") and build is None:
                build = step
                build_hash = result.get("current_scene_sha256")
            if key == ("score", "prepare") and prepare is None:
                prepare = step
    for folder in ("iterations", "scores"):
        for p in (base / folder).glob("*"):
            if p.name.isdigit():
                next_round = max(next_round, int(p.name) + 1)
    if latest:
        path, request, result = latest
        index["last_execution"] = {
            "receipt": str(path.parent / "result.json"),
            "state": result["state"],
            "outputs": list(
                dict.fromkeys(
                    option(s["args"], "--output")
                    for s in request["steps"]
                    if option(s["args"], "--output")
                )
            ),
        }
        if result["state"] != "completed":
            index["last_execution"].update(
                {
                    k: result[k]
                    for k in ("error", "error_tail", "error_log")
                    if k in result
                }
            )
            for step, receipt in zip(request["steps"], result["steps"]):
                if receipt.get("state") == "completed":
                    continue
                index["last_execution"]["script"] = option(step["args"], "--script")
                log = absolute(receipt.get("log", ""))
                output = option(step["args"], "--output")
                if log.is_file():
                    index["last_execution"].update(
                        failure_detail(
                            log,
                            absolute(output) if output else None,
                            result.get("started", 0),
                        )
                    )
                break
    if build:
        args = build["args"]
        index["edit"] = {
            "script": option(args, "--script"),
            "views": option(args, "--views"),
            "matches_current_scene": build_hash == scene_hash,
            "comparisons": sorted(
                str(p)
                for p in absolute(option(args, "--output")).glob("comparison_*.jpg")
            ),
            "what": option(args, "--what"),
            "why": option(args, "--why"),
        }
    score_paths = sorted((base / "scores").glob(f"*/{region}/score.json"))
    if score_paths:
        path = score_paths[-1]
        score = read(path, {})
        index["score"] = {k: score.get(k) for k in ("score", "status", "confidence")}
        index["score"]["dimensions"] = {
            k: {f: v[f] for f in ("score", "coverage") if f in v}
            for k, v in score.get("dimensions", {}).items()
        }
        index["score"].update(
            path=str(path),
            matches_current_scene=score.get("scene_sha256") == scene_hash,
        )
        findings = sorted(
            score.get("findings", []), key=lambda f: f.get("priority", 0), reverse=True
        )
        index["score"]["findings"] = [
            {
                **{k: v for k, v in f.items() if k != "missing_element"},
                "element": f.get("element")
                or f.get("missing_element")
                or ", ".join(f.get("objects", [])),
            }
            for f in findings[:3]
        ]
        index["score"]["remaining_findings"] = max(0, len(findings) - 3)
        index["score"]["evidence_requests"] = score.get("evidence_requests", [])[:3]
        index["score"]["remaining_evidence_requests"] = max(
            0, len(score.get("evidence_requests", [])) - 3
        )
        index["score"]["uncertainty_count"] = len(score.get("uncertainties", []))
    packet_paths = sorted((base / "scores").glob(f"*/{region}/packet.json"))
    if packet_paths:
        path = packet_paths[-1]
        packet = read(path, {})
        index["evidence"] = {
            "packet": str(path),
            "matches_current_scene": packet.get("scene_sha256") == scene_hash,
            "views": [
                {k: v.get(k) for k in ("id", "card")} for v in packet.get("views", [])
            ],
        }
        from vibuild.assessment.scoring import object_candidates

        findings = index.get("score", {}).get("findings", [])
        for finding in findings:
            if "bbox" in finding and "frame_id" in finding:
                finding["candidate_objects"] = object_candidates(finding, packet)
        names = {
            n
            for f in findings
            for n in [*f.get("objects", []), *f.get("candidate_objects", [])]
        }
        selected = [o for o in packet.get("objects", []) if o["name"] in names]
        index["evidence"]["affected_objects"] = [
            {k: o.get(k) for k in ("name", "bounds", "materials")}
            for o in selected[:12]
        ]
        index["evidence"]["remaining_affected_objects"] = max(0, len(selected) - 12)
    seed = read(base / "job_template.json", {})
    for step in seed.get("steps", []):
        if step["tool"] == "scene" and step["args"][0] == "iterate" and build is None:
            build = step
        if step["tool"] == "score" and step["args"][0] == "prepare" and prepare is None:
            prepare = step
    if directory and prepare and build is None:
        protocol = read(
            absolute(option(prepare["args"], "--output")) / ".runtime/protocol.json", {}
        )
        if protocol and protocol.get("scene_sha256") == scene_hash:
            calibration = next(
                s["views"] for s in protocol["regions"] if s["region"] == region
            )
            views_path = directory / "views.json" if directory else base / "views.json"
            write(views_path, {"frames": calibration[:3]})
            options = protocol["settings"]
            build = {
                "tool": "scene",
                "args": [
                    "iterate",
                    "--baseline",
                    str(root / "scene.blend"),
                    "--script",
                    str(directory / "edit.py"),
                    "--views",
                    str(views_path),
                    "--region",
                    region,
                    "--what",
                    "DESCRIBE_THIS_EDIT",
                    "--why",
                    "CITE_CURRENT_FINDING",
                    "--output",
                    str(base / "iterations" / f"{next_round:03}"),
                    "--width",
                    str(options["width"]),
                    "--samples",
                    str(options["samples"]),
                    "--threads",
                    str(options["threads"]),
                ],
            }
    if directory and prepare and not build:
        score_out = base / "scores" / f"{next_round:03}"
        args = list(prepare["args"])
        args[args.index("--output") + 1] = str(score_out)
        args[args.index("--scene") + 1] = str(root / "scene.blend")
        write(
            directory / "job.json",
            {
                "region": region,
                "round": next_round,
                "steps": [
                    {"tool": "score", "args": args},
                    {
                        "tool": "score",
                        "args": [
                            "run",
                            "--output",
                            str(score_out),
                            "--region",
                            region,
                            *score_options,
                        ],
                    },
                ],
            },
        )
        index["job_template"] = str(directory / "job.json")
    if directory and build and prepare:
        directory.mkdir(parents=True, exist_ok=True)
        iteration = base / "iterations" / f"{next_round:03}"
        score_out = base / "scores" / f"{next_round:03}"

        def changed(step, replacements):
            args = list(step["args"])
            for key, value in replacements.items():
                if key in args:
                    args[args.index(key) + 1] = str(value)
                else:
                    args += [key, str(value)]
            if "--notes" in args:
                i = args.index("--notes")
                del args[i : i + 2]
            return {"tool": step["tool"], "args": args}

        template = {
            "region": region,
            "round": next_round,
            "steps": [
                changed(
                    build,
                    {
                        "--baseline": root / "scene.blend",
                        "--script": directory / "edit.py",
                        "--output": iteration,
                        "--what": "DESCRIBE_THIS_EDIT",
                        "--why": "CITE_CURRENT_FINDING",
                    },
                ),
                changed(
                    prepare, {"--scene": root / "scene.blend", "--output": score_out}
                ),
                {
                    "tool": "score",
                    "args": [
                        "run",
                        "--output",
                        str(score_out),
                        "--region",
                        region,
                        *score_options,
                    ],
                },
            ],
        }
        # One source of calibrated views for both edits and scoring: the prepared protocol.
        protocol = read(
            absolute(option(prepare["args"], "--output")) / ".runtime/protocol.json", {}
        )
        if protocol:
            views = next(
                s["views"] for s in protocol["regions"] if s["region"] == region
            )
            if "--connection" in prepare["args"]:
                preview = read(absolute(option(build["args"], "--views")), {})["frames"]
                ids = {str(v["id"]) for v in preview}
                views = [v for v in views if str(v["id"]) in ids]
        else:
            views = read(absolute(option(build["args"], "--views")), {})["frames"]
        write(directory / "views.json", {"frames": views})
        args = template["steps"][0]["args"]
        if "--views" in args:
            args[args.index("--views") + 1] = str(directory / "views.json")
        else:
            args += ["--views", str(directory / "views.json")]
        if "--connection" not in prepare["args"]:
            args = template["steps"][1]["args"]
            for flag in ("--frame", "--views"):
                while flag in args:
                    i = args.index(flag)
                    del args[i : i + 2]
            for view in views:
                args += ["--frame", str(view["id"])]
        write(directory / "job.json", template)
        index["job_template"] = str(directory / "job.json")
    return index


def decision_images(index):
    """Attach the current evidence once; never label historical evidence as current."""
    evidence = index.get("evidence") or index.get("reference_evidence", {})
    paths = (
        [v["card"] for v in evidence.get("views", [])]
        if evidence.get("matches_current_scene")
        else []
    )
    edit = index.get("edit", {})
    if not paths and edit.get("matches_current_scene"):
        paths = edit.get("comparisons", [])
    return [p for p in paths if p and absolute(p).is_file()]
