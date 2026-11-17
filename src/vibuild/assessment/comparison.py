"""Render and score a fixed reference protocol with reusable baseline artifacts."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from vibuild.runtime.cache import digest, fingerprint
from vibuild.assessment.metrics import (
    load_npz,
    validate_grid,
    evaluate_arrays,
    evaluate_cached,
    _read_rgb,
    appearance_mask,
    DEFAULT_CLASSES,
)
import vibuild.blender.scene as scene


def prepare_protocol(path):
    """Resolve evidence once. Never infer masks, resize RGB, or change thresholds."""
    protocol = json.loads(path.read_text())
    base = path.resolve().parent
    frames = protocol.get("frames", [])
    if not frames:
        raise ValueError("comparison needs a nonempty fixed frame set")
    render_frames, ids, bindings, capabilities = [], set(), {}, []
    for f in frames:
        fid = str(f["id"])
        if fid in ids or Path(fid).name != fid or fid in ("", ".", ".."):
            raise ValueError("frame IDs must be unique safe file names")
        ids.add(fid)
    for f in frames:
        fid = f["id"] = str(f["id"])
        bindings[fid] = {}
        capability = {
            "id": fid,
            "geometry": "blocked",
            "appearance": "not_requested",
            "issues": [],
        }
        for field in ("prediction", "prediction_rgb", "baseline_rgb"):
            f.pop(field, None)
        for field in ("reference", "reference_rgb", "appearance_mask"):
            if field in f:
                f[field] = str((base / f[field]).resolve())
        frame = None
        try:
            ref = load_npz(f["reference"])
            (h, w), K = validate_grid(ref, "reference")
            if "camera_to_world" not in ref:
                raise ValueError(
                    "reference needs calibrated camera_to_world or extrinsics"
                )
            if K[0, 1] != 0 or K[1, 0] != 0:
                raise ValueError("renderer requires zero-skew pinhole K")
            measured = evaluate_arrays(
                ref,
                ref,
                classes=protocol.get("classes"),
                limits=protocol.get("limits"),
                min_confidence=protocol.get("min_confidence", 0.5),
            )
            frame = {
                "id": fid,
                "width": w,
                "height": h,
                "K": K.tolist(),
                "camera_to_world": ref["camera_to_world"].tolist(),
                "appearance": False,
            }
            frame.update({k: f[k] for k in ("camera", "blender_frame") if k in f})
            render_frames.append(frame)
            bindings[fid]["geometry"] = fid
            capability.update(geometry="renderable", reference_gates=measured)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            capability["issues"].append("geometry: " + str(exc))
        if "reference_rgb" in f:
            capability["appearance"] = "blocked"
            calibration = f.get("appearance_calibration", {}).get("reference", {})
            # Invalid/missing explicit calibration is a protocol error, not a resize instruction.
            K = np.asarray(calibration.get("K", []), float)
            pose = np.asarray(calibration.get("camera_to_world", []), float)
            if K.shape != (3, 3) or pose.shape != (4, 4):
                raise ValueError(
                    f"{fid}: RGB needs explicit native calibration; no implicit resize"
                )
            validate_grid(
                {"depth": np.ones((1, 1), np.float32), "K": K, "camera_to_world": pose},
                "RGB calibration",
            )
            if K[0, 1] != 0 or K[1, 0] != 0:
                raise ValueError(f"{fid}: RGB requires zero-skew pinhole K")
            try:
                h, w = _read_rgb(Path(f["reference_rgb"])).shape[:2]
                appearance_mask(f, base, (h, w))
                same = (
                    frame is not None
                    and (frame["height"], frame["width"]) == (h, w)
                    and np.allclose(frame["K"], K, atol=1e-6, rtol=0)
                    and np.allclose(frame["camera_to_world"], pose, atol=1e-6, rtol=0)
                )
                if same:
                    frame["appearance"] = True
                    rgb_id = fid
                else:
                    rgb_id = "__rgb_" + fid
                    while rgb_id in ids:
                        rgb_id = "_" + rgb_id
                    ids.add(rgb_id)
                    rgb_frame = {
                        "id": rgb_id,
                        "width": w,
                        "height": h,
                        "K": K.tolist(),
                        "camera_to_world": pose.tolist(),
                        "geometry": False,
                        "appearance": True,
                    }
                    rgb_frame.update(
                        {k: f[k] for k in ("camera", "blender_frame") if k in f}
                    )
                    render_frames.append(rgb_frame)
                bindings[fid]["appearance"] = rgb_id
                capability["appearance"] = "renderable"
                capability["separate_rgb_grid"] = not same
            except (OSError, ValueError) as exc:
                capability["issues"].append("appearance: " + str(exc))
        capabilities.append(capability)
    if isinstance(protocol.get("plausibility"), dict):
        protocol["plausibility"]["evidence"] = [
            str((base / p).resolve())
            for p in protocol["plausibility"].get("evidence", [])
        ]
    render = {
        "frames": render_frames,
        "classes": protocol.get("classes", DEFAULT_CLASSES),
        "bindings": bindings,
        "preflight": capabilities,
    }
    if "object_labels" in protocol:
        protocol["object_labels"]["evidence"] = [
            str((base / p).resolve())
            for p in protocol["object_labels"].get("evidence", [])
        ]
        render["object_labels"] = protocol["object_labels"]
    return protocol, render


def compare_reports(before, after):
    # Use exactly the supervisor's comparison rules; no second scoring definition.
    from vibuild.assessment.reports import score_values, coverage_floors, compare_scores

    if before["protocol_sha256"] != after["protocol_sha256"]:
        raise ValueError("comparison protocols differ")
    a, b = score_values(before), score_values(after)
    return {
        **compare_scores(a, b, coverage_floors(after)),
        "structure_status_before": before["structure_status"],
        "structure_status_after": after["structure_status"],
        "overall_acceptance": "not_assessed",
    }


def review(argv=None):
    """Re-review existing paired scores without rendering or rewriting evidence."""
    p = argparse.ArgumentParser(description=review.__doc__)
    p.add_argument(
        "--baseline",
        type=Path,
        nargs="+",
        required=True,
        help="ordered baseline score reports",
    )
    p.add_argument(
        "--candidate",
        type=Path,
        nargs="+",
        required=True,
        help="same ordered candidate score reports",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if len(args.baseline) != len(args.candidate):
        p.error("baseline/candidate report counts must match")
    groups = []
    for before, after in zip(args.baseline, args.candidate):
        result = compare_reports(
            json.loads(before.read_text()), json.loads(after.read_text())
        )
        result["inputs"] = {
            role: {"path": str(path.resolve()), "sha256": digest(path)}
            for role, path in [("baseline", before), ("candidate", after)]
        }
        groups.append(result)
    recommended = any(g["measured_improvement"] for g in groups) and not any(
        g["regressions"] for g in groups
    )
    result = {
        "groups": groups,
        "candidate_recommended": recommended,
        "decision": "advance_candidate"
        if recommended
        else "needs_revision_or_measured_gain",
        "overall_acceptance": "not_assessed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="fixed frames with reference NPZs, optional calibrated RGB, classes and thresholds",
    )
    p.add_argument("--baseline", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument(
        "--build-script",
        type=Path,
        help="synchronously patch baseline, save, then render/score; script modifies bpy in memory",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for renders, score reports and comparison",
    )
    p.add_argument("--blender", default=None)
    p.add_argument(
        "--cache-dir", type=Path, default=os.environ.get("VIBUILD_CACHE_DIR")
    )
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--baseline-review", type=Path, help="manual plausibility review of baseline"
    )
    p.add_argument(
        "--candidate-review",
        type=Path,
        help="manual plausibility review of changed candidate",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="validate inputs without launching Blender",
    )
    args = p.parse_args(argv)
    started = time.monotonic()
    protocol, render_manifest = prepare_protocol(args.manifest)
    hashes = {"baseline": digest(args.baseline)}
    if args.candidate:
        hashes["candidate"] = digest(args.candidate)
    else:
        digest(args.build_script)  # Fail before allocating output or launching Blender.
    if args.check_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "frames": len(protocol["frames"]),
                    "blend_sha256": hashes,
                    "capabilities": render_manifest["preflight"],
                },
                indent=2,
            )
        )
        return 0
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "comparison output must be empty; preserve versioned artifacts"
        )
    out.mkdir(parents=True, exist_ok=True)
    scene.dump(out / "preflight.json", {"frames": render_manifest["preflight"]})
    if args.build_script:
        if scene.main(
            [
                "patch",
                "--blend",
                str(args.baseline.resolve()),
                "--script",
                str(args.build_script.resolve()),
                "--output",
                str(out / "build"),
                "--blender",
                args.blender,
            ]
        ):
            raise RuntimeError("candidate build failed; render was not started")
        args.candidate = out / "build" / "candidate.blend"
        hashes["candidate"] = digest(args.candidate)

    reviews = {}
    for role in ("baseline", "candidate"):
        path = getattr(args, role + "_review")
        if path:
            value = json.loads(path.read_text())
            value["evidence"] = [
                str((path.resolve().parent / p).resolve())
                for p in value.get("evidence", [])
            ]
            reviews[role] = value
    if hashes["baseline"] == hashes["candidate"] and "baseline" in reviews:
        reviews.setdefault("candidate", reviews["baseline"])
    cache = args.cache_dir or out / ".cache"
    scene.dump(out / "render_input.json", render_manifest)
    reports, render_stats = {}, {}
    for role in ("baseline", "candidate"):
        options = [
            "render",
            "--blend",
            str(getattr(args, role).resolve()),
            "--manifest",
            str(out / "render_input.json"),
            "--output",
            str(out / role),
            "--cache-dir",
            str(cache),
            "--blender",
            args.blender,
            "--samples",
            str(args.samples),
            "--threads",
            str(args.threads),
        ]
        if render_manifest["frames"]:
            if scene.main(options):
                raise RuntimeError(f"{role} render failed; no comparison published")
            render_stats[role] = json.loads(
                (out / role / "render_manifest.json").read_text()
            )
        else:
            render_stats[role] = {
                "render_protocol_sha256": fingerprint(render_manifest),
                "request_seconds": 0,
                "cache": {"hit": False},
            }
        manifest = json.loads(json.dumps(protocol))
        manifest["prediction_protocol_sha256"] = render_stats[role][
            "render_protocol_sha256"
        ]
        if role in reviews:
            manifest["plausibility"] = reviews[role]
        elif role == "candidate" and hashes["baseline"] != hashes["candidate"]:
            # A baseline review cannot certify newly changed geometry.
            manifest.pop("plausibility", None)
        by_id = {f["id"]: f for f in render_manifest["frames"]}
        for f in manifest["frames"]:
            binding = render_manifest["bindings"][f["id"]]
            if "geometry" in binding:
                f["prediction"] = str(out / role / (binding["geometry"] + ".npz"))
            if "appearance" in binding:
                rgb_id = binding["appearance"]
                rf = by_id[rgb_id]
                f["prediction_rgb"] = str(out / role / (rgb_id + ".png"))
                f["baseline_rgb"] = str(out / "baseline" / (rgb_id + ".png"))
                f["appearance_calibration"] = {
                    **f["appearance_calibration"],
                    **{
                        k: {n: rf[n] for n in ("K", "camera_to_world")}
                        for k in ("prediction", "baseline")
                    },
                }
        scene.dump(out / (role + "_evaluation.json"), manifest)
        reports[role] = evaluate_cached(
            manifest, out, out / (role + "_scores.json"), cache
        )
    result = compare_reports(reports["baseline"], reports["candidate"])
    result.update(
        blend_sha256=hashes,
        unchanged_blend=hashes["baseline"] == hashes["candidate"],
        protocol_sha256=reports["candidate"]["protocol_sha256"],
        reports=["baseline_scores.json", "candidate_scores.json"],
        report_sha256={
            role: digest(out / (role + "_scores.json"))
            for role in ("baseline", "candidate")
        },
        elapsed_seconds=time.monotonic() - started,
        timings={
            k: {
                "render_request_seconds": v["request_seconds"],
                "render_cache_hit": v["cache"]["hit"],
                "evaluate_seconds": reports[k]["elapsed_seconds"],
                "evaluate_cache_hit": reports[k]["cache"]["hit"],
            }
            for k, v in render_stats.items()
        },
    )
    scene.dump(out / "comparison.json", result)
    print(json.dumps(result, indent=2))
    # A completed comparison is successful even when measured structure is blocked.
    # Only the evaluator reports and the supervisor decide readiness.
    return 0
