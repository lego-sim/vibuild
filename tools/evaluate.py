#!/usr/bin/env python3
"""Evaluate native reference pixels; structure and appearance remain separate.

NPZ arrays: depth (float scene units), K (3x3), confidence (raw finite), opaque (bool),
semantic (integer class IDs), instance (integer instance IDs, 0=unlabelled).
Optional scalar depth_convention is 'z' (default) or 'range'; camera_to_world
is OpenCV (x right, y down, z forward). Both inputs must represent the same
camera; K maps pixel coordinates, never a guessed image resize. Pixel indices
refer to pixel centres, with the upper-left centre at (0, 0).

Optional frame reference_rgb/prediction_rgb paths score appearance independently.
appearance_calibration must specify reference/prediction K and camera_to_world;
baseline_rgb requires baseline calibration too. Images are never resized.
appearance_mask optionally selects valid material pixels (NPY 0/1 or image).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import os
import time

import numpy as np

DEFAULT_CLASSES = {"window_frame": 1, "window_pane": 2, "tile": 3}
DEFAULT_LIMITS = {"depth_abs_rel": .1, "depth_coverage": .95,
                  "semantic_iou": .5, "instance_iou": .5,
                  "instance_precision": .8, "instance_recall": .8,
                  "instance_count_relative_error": .2}


def gate(status, **details):
    return {"status": status, **details}


def combine(items):
    statuses = [v["status"] for v in items]
    if not statuses or "blocked" in statuses:
        return "blocked"
    return "failed" if "failed" in statuses else "passed"


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if "K" not in data and "intrinsics" in data:
        data["K"] = data["intrinsics"]
    if "camera_to_world" not in data and "extrinsics" in data:
        extrinsics = np.asarray(data["extrinsics"], dtype=float)
        if extrinsics.shape == (3, 4):
            extrinsics = np.vstack((extrinsics, [0, 0, 0, 1]))
        if extrinsics.shape != (4, 4):
            raise ValueError("extrinsics must be a 3x4 or 4x4 world-to-camera transform")
        data["camera_to_world"] = np.linalg.inv(extrinsics)
    return data


def validate_grid(data, name):
    if "depth" not in data or "K" not in data:
        raise ValueError(f"{name} needs depth and K")
    depth = np.asarray(data["depth"])
    if depth.ndim != 2 or not depth.size or depth.dtype.kind != "f":
        raise ValueError(f"{name}.depth must be a nonempty 2D float array in scene units")
    K = np.asarray(data["K"], dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(np.linalg.det(K)) < 1e-12):
        raise ValueError(f"{name}.K is not a valid pinhole calibration")
    for key in ("confidence", "opaque", "semantic", "instance"):
        if key in data and np.asarray(data[key]).shape != depth.shape:
            raise ValueError(f"{name}.{key} must match native depth shape")
    for key in ("semantic", "instance"):
        if key in data and (data[key].dtype.kind not in "iu" or np.any(data[key] < 0)):
            raise ValueError(f"{name}.{key} must contain nonnegative integer IDs")
    if "confidence" in data:
        c = data["confidence"]
        if not np.isfinite(c).all():
            raise ValueError(f"{name}.confidence must be finite")
    if "opaque" in data and not np.isin(data["opaque"], [0, 1]).all():
        raise ValueError(f"{name}.opaque must be a boolean mask")
    if "camera_to_world" in data:
        pose = np.asarray(data["camera_to_world"], dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
            raise ValueError(f"{name}.camera_to_world is not a rigid transform")
    return depth.shape, K


def z_depth(data, K):
    depth = np.asarray(data["depth"], dtype=np.float64)
    convention = str(np.asarray(data.get("depth_convention", "z")).item())
    if convention == "z":
        return depth
    if convention != "range":
        raise ValueError("depth_convention must be z or range")
    y, x = np.indices(depth.shape)
    rays = np.linalg.inv(K) @ np.stack((x.ravel(), y.ravel(), np.ones(x.size)))
    return depth / np.linalg.norm(rays, axis=0).reshape(depth.shape)


def align_prediction(reference, prediction):
    """Nearest sample using K_pred @ inverse(K_ref); preserve reference grid."""
    shape, kr = validate_grid(reference, "reference")
    pshape, kp = validate_grid(prediction, "prediction")
    if "camera_to_world" in reference or "camera_to_world" in prediction:
        if not all("camera_to_world" in d for d in (reference, prediction)):
            raise ValueError("both camera poses are required when either is supplied")
        if not np.allclose(reference["camera_to_world"], prediction["camera_to_world"], atol=1e-6, rtol=0):
            raise ValueError("camera poses differ; reprojection with visibility is required")
    y, x = np.indices(shape)
    mapping = kp @ np.linalg.inv(kr)
    pixels = mapping @ np.stack((x.ravel(), y.ravel(), np.ones(x.size)))
    px = np.floor(pixels[0] / pixels[2] + .5).astype(int).reshape(shape)
    py = np.floor(pixels[1] / pixels[2] + .5).astype(int).reshape(shape)
    inside = (px >= 0) & (px < pshape[1]) & (py >= 0) & (py < pshape[0])
    px, py = np.clip(px, 0, pshape[1]-1), np.clip(py, 0, pshape[0]-1)
    aligned = {key: prediction[key][py, px] for key in ("semantic", "instance") if key in prediction}
    aligned["depth"] = z_depth(prediction, kp)[py, px]
    return z_depth(reference, kr), aligned, inside, mapping


def instance_metrics(ref, pred, ref_mask, pred_mask, threshold):
    rvalues, rcounts = np.unique(ref[ref_mask & (ref > 0)], return_counts=True)
    pvalues, pcounts = np.unique(pred[pred_mask & (pred > 0)], return_counts=True)
    ref_counts = dict(zip(rvalues.tolist(), rcounts.tolist()))
    pred_counts = dict(zip(pvalues.tolist(), pcounts.tolist()))
    rids, pids = list(ref_counts), list(pred_counts)
    overlap = ref_mask & pred_mask & (ref > 0) & (pred > 0)
    pairs, counts = np.unique(np.stack((ref[overlap], pred[overlap]), axis=1), axis=0, return_counts=True)
    edges = {rid: [] for rid in rids}
    scores = {}
    for (rid, pid), intersection in zip(pairs.tolist(), counts.tolist()):
        score = intersection / (ref_counts[rid] + pred_counts[pid] - intersection)
        if score >= threshold:
            edges[rid].append(pid)
            scores[(rid, pid)] = score
    for rid, options in edges.items():
        options.sort(key=lambda pid: -scores[(rid, pid)])
    # Maximum cardinality one-to-one matching; do not double-count split/merged objects.
    matched = {}
    def augment(rid, seen):
        for pid in edges[rid]:
            if pid in seen:
                continue
            seen.add(pid)
            if pid not in matched or augment(matched[pid], seen):
                matched[pid] = rid
                return True
        return False
    for rid in rids:
        augment(int(rid), set())
    count = len(matched)
    return {"reference_count": len(rids), "prediction_count": len(pids),
            "matched_count": count, "precision": count / len(pids) if len(pids) else 0.,
            "recall": count / len(rids) if len(rids) else 0.,
            "count_relative_error": abs(len(pids)-len(rids)) / max(len(rids), 1),
            "matches": [{"reference": rid, "prediction": pid, "iou": scores[(rid, pid)]}
                        for pid, rid in matched.items()]}


def evaluate_arrays(reference, prediction, classes=None, limits=None, min_confidence=.5):
    classes = DEFAULT_CLASSES if classes is None else classes
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    if not np.isfinite(min_confidence):
        raise ValueError("min_confidence must be finite in the model's raw units")
    for key, value in limits.items():
        if key not in DEFAULT_LIMITS or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid limit: {key}")
        if key != "instance_count_relative_error" and value > 1:
            raise ValueError(f"limit {key} must be in [0,1]")
    rd, pred, inside, mapping = align_prediction(reference, prediction)
    components = {}
    if not all(k in reference for k in ("confidence", "opaque")):
        components["depth"] = gate("blocked", reason="reference confidence and opaque masks are required")
    else:
        mask = np.isfinite(rd) & (rd > 0) & reference["opaque"].astype(bool) & (reference["confidence"] >= min_confidence)
        valid = mask & inside & np.isfinite(pred["depth"]) & (pred["depth"] > 0)
        n, count = int(mask.sum()), int(valid.sum())
        if not n:
            components["depth"] = gate("blocked", reason="no confident opaque reference depth pixels")
        elif not count:
            components["depth"] = gate("failed", reference_pixels=n, valid_prediction_pixels=0, coverage=0.)
        else:
            error = pred["depth"][valid] - rd[valid]
            rel = float(np.mean(np.abs(error) / rd[valid]))
            coverage = count/n
            components["depth"] = gate("passed" if rel <= limits["depth_abs_rel"] and coverage >= limits["depth_coverage"] else "failed",
                reference_pixels=n, valid_prediction_pixels=count, coverage=coverage,
                abs_rel=rel, mae_scene_units=float(np.mean(np.abs(error))), rmse_scene_units=float(np.sqrt(np.mean(error**2))),
                p95_abs_error_scene_units=float(np.quantile(np.abs(error), .95)))
    if not classes:
        components["segmentation"] = gate("blocked", reason="no required structural classes configured")
    for name, cid in classes.items():
        if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
            raise ValueError("class IDs must be positive integers")
        if not all("semantic" in d and "instance" in d for d in (reference, pred)):
            components[name] = gate("blocked", reason="both semantic and instance labels are required")
            continue
        rm = reference["semantic"] == cid
        pm = (pred["semantic"] == cid) & inside
        if not rm.any():
            components[name] = gate("blocked", reason="required class absent from reference; supply a view that observes it")
            continue
        iou = float(np.count_nonzero(rm & pm) / np.count_nonzero(rm | pm))
        if np.any(rm & (reference["instance"] == 0)):
            components[name] = gate("blocked", reason="reference class pixels lack instance IDs", semantic_iou=iou)
            continue
        metrics = instance_metrics(reference["instance"], pred["instance"], rm, pm, limits["instance_iou"])
        passed = (iou >= limits["semantic_iou"] and metrics["precision"] >= limits["instance_precision"]
                  and metrics["recall"] >= limits["instance_recall"]
                  and metrics["count_relative_error"] <= limits["instance_count_relative_error"]
                  and not np.any(pm & (pred["instance"] == 0)))
        components[name] = gate("passed" if passed else "failed", semantic_iou=iou, **metrics)
    return {"status": combine(components.values()), "components": components, "limits": limits,
            "reference_shape": list(rd.shape), "prediction_to_reference_sampling": "nearest using K_pred @ inv(K_ref)",
            "reference_pixel_to_prediction_pixel": mapping.tolist(), "min_confidence": min_confidence,
            "confidence_note": "threshold uses raw model confidence units; not a calibrated probability",
            "depth_units": "original scene units; metric scale not inferred"}


def plausibility_gate(review, base):
    result = _plausibility_gate(review, base)
    required = ("dimensions", "topology", "openings", "occlusion")
    checks = review.get("checks", {}) if isinstance(review, dict) else {}
    checks = checks if isinstance(checks, dict) else {}
    passed = sum(checks.get(k) == "passed" for k in required)
    unknown = sum(checks.get(k) not in ("passed", "failed") for k in required)
    return {**result, "passed_checks": passed, "required_checks": 4,
            "unknown_checks": unknown, "passed_fraction": passed / 4,
            "score_0_100": 100 * passed / 4,
            "score_note": "fraction of four manual checks marked passed; evidence validity still gates status"}


def _plausibility_gate(review, base):
    if not isinstance(review, dict) or review.get("status") not in ("passed", "failed"):
        return gate("blocked", reason="manual plausibility review is missing or unknown")
    evidence = review.get("evidence", [])
    if (not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or not isinstance(evidence, list) or not evidence
            or not all(isinstance(p, str) and (base / p).is_file() for p in evidence)):
        return gate("blocked", reason="review requires a named reviewer and existing evidence files")
    required = ("dimensions", "topology", "openings", "occlusion")
    checks = review.get("checks", {})
    if not isinstance(checks, dict) or any(checks.get(k) not in ("passed", "failed") for k in required):
        return gate("blocked", reason="review must assess dimensions, topology, openings and occlusion")
    status = "failed" if "failed" in checks.values() or review["status"] == "failed" else "passed"
    return gate(status, reviewer=review["reviewer"], evidence=evidence, checks=checks)


def _read_rgb(path):
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        from PIL import Image
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.
    elif array.dtype.kind != "f":
        raise ValueError("RGB arrays must be uint8 or normalized floating sRGB")
    if array.ndim != 3 or array.shape[2] != 3 or not array.size:
        raise ValueError("RGB must be a nonempty HxWx3 array")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("floating RGB must be finite and in [0,1]")
    return array


def _lab(rgb):
    """sRGB D65 to CIE Lab; ΔE76 is a colour distance, not perceptual acceptance."""
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]]).T
    xyz /= np.array([.95047, 1., 1.08883])
    f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz / (3*(6/29)**2) + 4/29)
    return np.stack((116*f[:, 1]-16, 500*(f[:, 0]-f[:, 1]), 200*(f[:, 1]-f[:, 2])), axis=1)


def appearance_metrics(reference, prediction, mask):
    if reference.shape != prediction.shape:
        raise ValueError("appearance image grids differ; implicit resizing is forbidden")
    if mask.shape != reference.shape[:2] or not mask.any():
        raise ValueError("appearance mask must match the native grid and contain scored pixels")
    a, b = reference[mask], prediction[mask]
    error = a - b
    mae = float(np.mean(np.abs(error), dtype=np.float64))
    delta_e = np.linalg.norm(_lab(a) - _lab(b), axis=1)
    return {"scored_pixels": int(mask.sum()), "rgb_mae": mae,
            "rgb_rmse": float(np.sqrt(np.mean(error**2, dtype=np.float64))),
            "cie_delta_e76_mean": float(delta_e.mean()),
            "cie_delta_e76_p95": float(np.quantile(delta_e, .95)),
            "score_0_100": float(100*(1-mae))}


def appearance_mask(frame, base, shape):
    mask = np.ones(shape, dtype=bool)
    if "appearance_mask" in frame:
        path = base / frame["appearance_mask"]
        if path.suffix.lower() == ".npy":
            raw = np.load(path, allow_pickle=False)
            if not np.isin(raw, [0, 1]).all():
                raise ValueError("NPY appearance mask must contain only 0/1")
        else:
            from PIL import Image
            with Image.open(path) as image:
                raw = np.asarray(image.convert("L"))
        mask = raw > 0
    if mask.shape != shape or not mask.any():
        raise ValueError("appearance mask must match the native grid and contain scored pixels")
    return mask


def appearance_frame(frame, base):
    requested = any(k in frame for k in ("reference_rgb", "prediction_rgb", "baseline_rgb", "appearance_mask"))
    if not requested:
        return gate("not_evaluated", reason="no appearance images supplied")
    try:
        reference = _read_rgb(base / frame["reference_rgb"])
        mask = appearance_mask(frame, base, reference.shape[:2])
        prediction = _read_rgb(base / frame["prediction_rgb"])
        calibration = frame["appearance_calibration"]
        roles = ["reference", "prediction"] + (["baseline"] if "baseline_rgb" in frame else [])
        for role in roles:
            validate_grid({"depth": np.ones((1, 1)), **calibration[role]}, f"appearance.{role}")
            if "camera_to_world" not in calibration[role]:
                raise ValueError("appearance calibration needs K and camera_to_world for each image")
            if role != "reference" and (not np.allclose(calibration[role]["K"], calibration["reference"]["K"], atol=1e-6, rtol=0)
                or not np.allclose(calibration[role]["camera_to_world"], calibration["reference"]["camera_to_world"], atol=1e-6, rtol=0)):
                raise ValueError("appearance images must share identical calibrated K and camera pose")
        metrics = appearance_metrics(reference, prediction, mask)
        result = gate("scored", **metrics,
            metric_note="RGB errors use normalized display sRGB; colour error uses CIE Lab D65 ΔE76; score=100*(1-RGB_MAE), a proxy, not an acceptance probability")
        if "baseline_rgb" in frame:
            baseline = appearance_metrics(reference, _read_rgb(base / frame["baseline_rgb"]), mask)
            delta = metrics["score_0_100"] - baseline["score_0_100"]
            result["baseline"] = baseline
            result["score_delta"] = delta
            result["regressed"] = bool(delta < -1e-8)
        return result
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return gate("blocked", reason=str(exc))


def protocol_hash(manifest, base):
    """Freeze reference content, masks, frame set, classes and thresholds, not predictions."""
    frames = []
    for frame in manifest.get("frames", []):
        refs = {}
        for key in ("reference", "reference_rgb", "appearance_mask", "baseline_rgb"):
            if key in frame:
                path = base / frame[key]
                h = hashlib.sha256()
                try:
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(8*1024*1024), b""): h.update(chunk)
                    refs[key] = h.hexdigest()
                except OSError: refs[key] = "missing"
        frames.append({"id": frame.get("id"), "references": refs,
                       "appearance_reference": frame.get("appearance_calibration", {}).get("reference")})
    value = dict(frames=frames, classes=manifest.get("classes", DEFAULT_CLASSES),
                 limits=manifest.get("limits"), min_confidence=manifest.get("min_confidence", .5),
                 prediction_protocol_sha256=manifest.get('prediction_protocol_sha256'))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate_manifest(manifest, base=Path(".")):
    if not isinstance(manifest, dict):
        raise ValueError("evaluation manifest must be a JSON object")
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or any(not isinstance(f, dict) for f in frames):
        raise ValueError("frames must be a list of objects")
    results = []
    appearance_results = []
    for index, frame in enumerate(frames):
        try:
            result = evaluate_arrays(load_npz(base / frame["reference"]), load_npz(base / frame["prediction"]),
                classes=manifest.get("classes"), limits=manifest.get("limits"),
                min_confidence=manifest.get("min_confidence", .5))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result = gate("blocked", reason=str(exc))
        results.append({"id": frame.get("id", str(index)), **result})
        appearance_results.append({"id": frame.get("id", str(index)), **appearance_frame(frame, base)})
    plausibility = plausibility_gate(manifest.get("plausibility"), base)
    status = combine(results + [plausibility]) if results else "blocked"
    requested = [r for r in appearance_results if r["status"] != "not_evaluated"]
    appearance_status = ("not_evaluated" if not requested else
                         "blocked" if any(r["status"] == "blocked" for r in requested) else "scored")
    appearance = gate(appearance_status, frames=appearance_results,
        requested_frames=len(requested), scored_frames=sum(r["status"] == "scored" for r in requested),
        unscored_frames=sum(r["status"] == "not_evaluated" for r in appearance_results),
        score_note="per-frame appearance proxies include lighting/exposure effects; no averaging across views or with structural scores")
    return {"schema_version": 1, "protocol_sha256": protocol_hash(manifest, base), "structure_status": status, "structure_accepted": status == "passed",
            "frames": results, "plausibility": plausibility,
            "appearance": appearance,
            "overall_acceptance": "not_assessed",
            "reason": "no frames supplied" if not results else "every required component and plausibility gate must pass"}


def evaluate_cached(manifest, base, output, cache_dir=None):
    from artifact_cache import digest, file_hash, fingerprint, entry_lock, verified, publish
    started = time.monotonic()
    normalized = json.loads(json.dumps(manifest))
    for f in normalized.get('frames', []):
        for field in ('reference', 'prediction', 'reference_rgb', 'prediction_rgb', 'baseline_rgb', 'appearance_mask'):
            if field in f:
                f[field] = file_hash(base / f[field])
    if isinstance(normalized.get('plausibility'), dict):
        normalized['plausibility']['evidence'] = [file_hash(base / p) for p in normalized['plausibility'].get('evidence', [])]
    key = fingerprint({'manifest': normalized, 'evaluator': digest(__file__),
                       'cache_code': digest(Path(__file__).with_name('artifact_cache.py'))})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Preserve versioned score report: {output}')
    hit = False
    if cache_dir:
        import tempfile
        with entry_lock(Path(cache_dir) / 'evaluate', key) as entry:
            if verified(entry):
                report = json.loads((entry / 'scores.json').read_text())
                hit = True
            else:
                report = evaluate_manifest(manifest, base)
                with tempfile.TemporaryDirectory(dir=entry.parent) as temp:
                    temp = Path(temp)
                    (temp / 'scores.json').write_text(json.dumps(report, allow_nan=False))
                    publish(entry, temp, ['scores.json'])
    else:
        report = evaluate_manifest(manifest, base)
    # Cached scores may have been produced from equivalent evidence at another path.
    report['plausibility'] = plausibility_gate(manifest.get('plausibility'), base)
    report['cache'] = {'key': key, 'hit': hit}
    report['elapsed_seconds'] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('compare', 'review'):
        from comparison import main as compare, review
        try:
            return (review if argv[0] == 'review' else compare)(argv[1:])
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            print(f'evaluate {argv[0]}: {exc}', file=sys.stderr)
            return 2
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="JSON frames [{id,reference,prediction}], classes, limits, plausibility")
    source.add_argument("--reference", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--review", type=Path, help="manual plausibility JSON; paths relative to review file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=os.environ.get('VIBUILD_CACHE_DIR'))
    parser.add_argument('--require-passed', action='store_true', help='exit 2 unless structure passes; default exit status describes report generation')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; preserve versioned score reports')
    if args.reference and not args.prediction:
        parser.error("--prediction is required with --reference")
    try:
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
            base = args.manifest.resolve().parent
        else:
            manifest = {"frames": [{"reference": str(args.reference.resolve()), "prediction": str(args.prediction.resolve())}]}
            base = Path.cwd()
        if args.review:
            review = json.loads(args.review.read_text())
            review["evidence"] = [str((args.review.resolve().parent / p).resolve()) for p in review.get("evidence", [])]
            manifest["plausibility"] = review
        report = evaluate_cached(manifest, base, args.output, args.cache_dir)
    except (ValueError, OSError, TypeError) as exc:
        report = {"structure_status": "blocked", "structure_accepted": False, "overall_acceptance": "not_assessed", "error": str(exc)}
    if 'error' in report:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"structure_status": report["structure_status"], "output": str(args.output)}))
    return 2 if 'error' in report or (args.require_passed and report['structure_status'] != 'passed') else 0


if __name__ == "__main__":
    sys.exit(main())
