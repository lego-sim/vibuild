#!/usr/bin/env python3
"""Offline, pose-conditioned DA3 depth and SAM3.1 concept tracking.

Model dependencies live in the configured backend Python, not this CLI's Python.
Run ``perceive.py check --config configs/local.json`` before inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid


class PerceptionError(ValueError):
    pass


DEFAULT_CONCEPTS = ["wall", "floor", "ceiling", "window", "window frame",
                    "glass pane", "floor tile", "door", "door leaf", "door handle"]
ALLOWED_SPLITS = {"model", "fit", "train"}


def local_path(value, base):
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise PerceptionError(f"Expected a local path, got {value!r}")
    path = Path(os.path.expandvars(value)).expanduser()
    # Do not resolve symlinks: venv/bin/python must retain its environment.
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def read_config(path):
    path = Path(path).absolute()
    config = json.loads(path.read_text())
    if not isinstance(config.get("runtime"), dict):
        raise PerceptionError("Config requires runtime.da3 and/or runtime.sam31 objects")
    return config, path.parent


def backend_config(config, base, backend):
    value = config["runtime"].get(backend)
    if not isinstance(value, dict):
        raise PerceptionError(f"Missing runtime.{backend}")
    result = dict(value)
    for key in ("python", "source", "checkpoint"):
        result[key] = str(local_path(value.get(key), base))
    result.setdefault("device", "cuda")
    return result


def matrix(value, shape, label):
    try:
        if len(value) != shape[0]:
            raise ValueError()
        rows = [[float(x) for x in row] for row in value]
        if any(len(row) != shape[1] for row in rows):
            raise ValueError()
        if not all(math.isfinite(x) for row in rows for x in row):
            raise ValueError()
        return rows
    except (TypeError, ValueError):
        raise PerceptionError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix") from None


def load_frames(path, selected_ids=None, audit_evidence=False):
    """Select exact IDs in manifest order; never select evaluation images implicitly."""
    if audit_evidence and not selected_ids:
        raise PerceptionError("Audit evidence requires an explicit nonempty --ids allowlist")
    path = Path(path).absolute()
    document = json.loads(path.read_text())
    rows = document.get("frames") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise PerceptionError("Frame manifest must contain a nonempty frames list")
    requested = None if selected_ids is None else {str(i) for i in selected_ids}
    if requested is not None and len(requested) != len(selected_ids):
        raise PerceptionError("Duplicate requested frame IDs")
    seen, output = set(), []
    for raw in rows:
        if not isinstance(raw, dict) or raw.get("id", raw.get("index")) is None:
            raise PerceptionError("Every frame requires id or index")
        key = str(raw.get("id", raw.get("index")))
        if key in seen:
            raise PerceptionError(f"Duplicate manifest frame ID: {key}")
        seen.add(key)
        if requested is not None and key not in requested:
            continue
        splits = [str(raw[k]).lower() for k in ("split", "evaluation_split") if k in raw]
        if audit_evidence:
            if not splits or any(s not in {"audit", "holdout"} for s in splits):
                raise PerceptionError(f"Frame {key}: rejected split {splits}; audit evidence requires original audit/holdout tags")
        elif not splits or any(s not in ALLOWED_SPLITS for s in splits):
            raise PerceptionError(f"Frame {key}: rejected split {splits}; only model/fit/train inputs are allowed")
        image = local_path(raw.get("original_rgb", raw.get("image")), path.parent)
        if not image.is_file():
            raise PerceptionError(f"Frame {key}: image missing: {image}")
        k = matrix(raw.get("intrinsics_original", raw.get("K")), (3, 3), f"Frame {key} intrinsics")
        r = matrix(raw.get("camera_to_scene_rotation", raw.get("rotation")), (3, 3), f"Frame {key} rotation")
        center = raw.get("scene_center", raw.get("center"))
        c = matrix([center], (1, 3), f"Frame {key} center")[0]
        if k[0][0] <= 0 or k[1][1] <= 0 or any(abs(k[2][j] - [0, 0, 1][j]) > 1e-6 for j in range(3)):
            raise PerceptionError(f"Frame {key}: invalid camera intrinsics")
        dot = lambda a, b: sum(x*y for x, y in zip(a, b))
        det = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
               - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
               + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
        if abs(det - 1) > 1e-3 or any(abs(dot(r[i], r[j]) - (i == j)) > 1e-3 for i in range(3) for j in range(3)):
            raise PerceptionError(f"Frame {key}: rotation must be a proper camera-to-scene rotation")
        output.append({**raw, "id": key, "image": str(image), "K": k,
                       "center": c, "rotation": r, "split": raw.get("split", splits[0])})
    if requested is not None and requested - seen:
        raise PerceptionError(f"Unknown frame IDs: {sorted(requested - seen)}")
    if not output:
        raise PerceptionError("No frames selected")
    return output


def offline_environment():
    result = dict(os.environ)
    result.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return result


def prepare_backend(runtime, backend):
    os.environ.update({k: v for k, v in offline_environment().items() if k.endswith("OFFLINE")})
    source = Path(runtime["source"])
    package_root = source / "src" if backend == "da3" and (source / "src").is_dir() else source
    sys.path.insert(0, str(package_root))
    return package_root


def readiness(runtime, backend):
    """Import API and inspect CUDA; never load weights or call inference."""
    result = {"backend": backend, "runtime": runtime, "inference_verified": False,
              "weights_loaded": False, "ready": False, "errors": []}
    source = prepare_backend(runtime, backend)
    checkpoint = Path(runtime["checkpoint"])
    if not source.is_dir():
        result["errors"].append(f"Source directory missing: {source}")
    required = [checkpoint / "config.json", checkpoint / "model.safetensors"] if backend == "da3" else [checkpoint]
    for path in required:
        if not path.is_file():
            result["errors"].append(f"Local model file missing: {path}")
    try:
        import torch
        module = importlib.import_module("depth_anything_3.api" if backend == "da3" else "sam3.model_builder")
        attr = "DepthAnything3" if backend == "da3" else "build_sam3_multiplex_video_predictor"
        if not hasattr(module, attr):
            raise PerceptionError(f"Installed backend lacks {attr}; no fallback is permitted")
        result.update(import_ok=True, api_source=str(Path(module.__file__).absolute()),
                      torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            result["errors"].append("Imported API did not come from configured source")
        if runtime["device"].startswith("cuda") and not result["cuda_available"]:
            result["errors"].append("CUDA is unavailable in this process")
        elif runtime["device"].startswith("cuda"):
            result["device_name"] = torch.cuda.get_device_name(runtime["device"])
        if backend == "sam31" and runtime["device"] != "cuda":
            result["errors"].append("This SAM3.1 multiplex API requires device='cuda'; use CUDA_VISIBLE_DEVICES to choose GPU")
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["ready"] = not result["errors"]
    return result


def image_shapes(frames):
    from PIL import Image
    shapes = []
    for frame in frames:
        with Image.open(frame["image"]) as image:
            # EXIF orientation changes require transformed calibration, not an implicit transpose.
            if image.getexif().get(274, 1) != 1:
                raise PerceptionError(f"Frame {frame['id']}: normalize EXIF orientation and calibration before inference")
            shapes.append([image.height, image.width])
    return shapes


def validate_plan(backend, runtime, frames, shapes, args):
    """Cheap invariants shared by dry runs and real runs, before loading weights."""
    if backend == "da3":
        if len(frames) < 2 or max(math.dist(frames[0]["center"], f["center"]) for f in frames) < 1e-6:
            raise PerceptionError("Pose-conditioned DA3 requires at least two distinct camera centers for input-scale alignment")
        if (args.resolution or int(runtime.get("process_res", 1008))) < 14:
            raise PerceptionError("DA3 resolution must be >= 14")
    else:
        if len({tuple(shape) for shape in shapes}) != 1:
            raise PerceptionError("SAM3.1 video sessions require equal image sizes; group frames by native size")
        videos = {f.get("video", f.get("video_id")) for f in frames} - {None}
        if len(videos) > 1:
            raise PerceptionError("One SAM3.1 tracking session must contain one video; select its frame IDs")
        times = [f.get("time") for f in frames]
        if all(t is not None for t in times) and any(float(b) <= float(a) for a, b in zip(times, times[1:])):
            raise PerceptionError("SAM3.1 tracking frames must be in strictly increasing timestamp order")
        concepts = args.concepts or DEFAULT_CONCEPTS
        if any(not c.strip() for c in concepts) or len(set(concepts)) != len(concepts):
            raise PerceptionError("Concepts must be unique, nonempty text strings")
        if args.max_objects < 1:
            raise PerceptionError("--max-objects must be positive")


def camera_arrays(frames):
    import numpy as np
    intrinsics = np.asarray([f["K"] for f in frames], dtype=np.float32)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], len(frames), axis=0)
    for i, frame in enumerate(frames):
        rotation = np.asarray(frame["rotation"], dtype=np.float32).T
        extrinsics[i, :3, :3] = rotation
        extrinsics[i, :3, 3] = -rotation @ np.asarray(frame["center"], dtype=np.float32)
    return intrinsics, extrinsics


def depth_arrays(depth, confidence, intrinsics, extrinsics, original_intrinsics):
    """No decimation, confidence threshold, half precision conversion, or pose refit."""
    import numpy as np
    depth, confidence = np.asarray(depth, np.float32), np.asarray(confidence, np.float32)
    k, e = np.asarray(intrinsics, np.float32), np.asarray(extrinsics, np.float32)
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise PerceptionError("DA3 depth/confidence must be matching HxW arrays")
    if k.shape != (3, 3) or e.shape not in ((3, 4), (4, 4)) or not np.isfinite(k).all() or not np.isfinite(e).all():
        raise PerceptionError("DA3 returned invalid camera matrices")
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    uv = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    camera_xyz = (uv @ np.linalg.inv(k).T) * depth[..., None]
    xyz = (camera_xyz - e[:3, 3]) @ e[:3, :3]
    transform = k @ np.linalg.inv(np.asarray(original_intrinsics, np.float32))
    return {"depth": depth, "confidence": confidence, "xyz": xyz.astype(np.float32),
            "valid": np.isfinite(depth) & np.isfinite(confidence) & (depth > 0),
            "intrinsics": k, "extrinsics": e,
            "original_to_output_pixels": transform.astype(np.float32)}


def frame_metadata(frame, shape):
    return {"id": frame["id"], "index": frame.get("index"), "image": frame["image"],
            "split": frame["split"], "original_shape": shape,
            "intrinsics_original": frame["K"], "scene_center": frame["center"],
            "camera_to_scene_rotation": frame["rotation"],
            **{k: frame[k] for k in ("video", "video_id", "time", "timestamp_seconds", "evaluation_split") if k in frame}}


def run_depth(runtime, frames, shapes, output, args):
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3
    k, e = camera_arrays(frames)
    resolution = args.resolution or int(runtime.get("process_res", 1008))
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(runtime["checkpoint"], local_files_only=True).to(runtime["device"]).eval()
    if runtime["device"].startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.inference([f["image"] for f in frames], extrinsics=e, intrinsics=k,
                                     align_to_input_ext_scale=True, process_res=resolution,
                                     process_res_method="upper_bound_resize", infer_gs=False)
    if runtime["device"].startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_started
    if prediction.conf is None or len(prediction.depth) != len(frames):
        raise PerceptionError("DA3 returned missing confidence or incomplete frames")
    records = []
    for i, frame in enumerate(frames):
        arrays = depth_arrays(prediction.depth[i], prediction.conf[i], prediction.intrinsics[i], prediction.extrinsics[i], k[i])
        if not np.allclose(arrays["extrinsics"][:3], e[i, :3], atol=1e-4):
            raise PerceptionError("DA3 did not preserve registered world-to-camera extrinsics")
        arrays.update(intrinsics_original=k[i], extrinsics_input=e[i])
        file = f"{i:06d}.npz"
        np.savez_compressed(output / file, **arrays)
        records.append({**frame_metadata(frame, shapes[i]), "depth_file": file,
                        "shape": list(arrays["depth"].shape),
                        "original_to_output_pixels": arrays["original_to_output_pixels"].tolist()})
    return {"frames": records, "align_to_input_ext_scale": True, "processing_resolution": resolution,
            "processing_method": "upper_bound_resize", "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "pixel_convention": "OpenCV integer pixel centers; output_pixel = original_to_output_pixels @ original_pixel",
            "geometry_convention": "camera +x right, +y down, +z forward; xyz in input scene coordinates",
            "depth_units": "input camera translation units; not independently verified metric ground truth",
            "confidence_note": "Raw DA3 confidence, not calibrated probability; no threshold filtering"}


def as_numpy(value):
    import numpy as np
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def mask_arrays(outputs, expected_shape):
    import numpy as np
    ids = as_numpy(outputs["out_obj_ids"]).astype(np.int64)
    masks = as_numpy(outputs["out_binary_masks"])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] != len(ids) or tuple(masks.shape[1:]) != tuple(expected_shape):
        raise PerceptionError(f"SAM3.1 output shape {masks.shape} does not match original grid {expected_shape}; refusing an invented pixel transform")
    if len(set(ids.tolist())) != len(ids):
        raise PerceptionError("Duplicate SAM3.1 local object IDs")
    result = {"masks": masks.astype(bool), "object_ids": ids,
              "original_to_output_pixels": np.eye(3, dtype=np.float32)}
    for key in ("out_probs", "out_scores", "out_boxes_xywh"):
        if key in outputs:
            result[key] = as_numpy(outputs[key]).astype(np.float32)
    return result


def run_segment(runtime, frames, shapes, output, args):
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    concepts = args.concepts or DEFAULT_CONCEPTS
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=runtime["checkpoint"], use_fa3=False, use_rope_real=False,
        compile=False, warm_up=False, async_loading_frames=False,
        max_num_objects=args.max_objects)
    load_seconds, sessions = time.perf_counter() - started, []
    # Lossless staging preserves every original pixel and makes input ordering explicit.
    with tempfile.TemporaryDirectory(prefix="vibuild-sam31-") as stage:
        for i, frame in enumerate(frames):
            with Image.open(frame["image"]) as image:
                image.convert("RGB").save(Path(stage) / f"{i:06d}.png")
        for concept_index, concept in enumerate(concepts):
            folder = output / f"concept_{concept_index:03d}"
            folder.mkdir()
            sid = predictor.handle_request({"type": "start_session", "resource_path": stage})["session_id"]
            session_key = f"{args.run_id}/{concept_index}/{sid}"
            try:
                predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": concept})
                records, seen = [], set()
                for response in predictor.propagate_in_video(session_id=sid, propagation_direction="forward", is_last_batch=True):
                    i = int(response["frame_index"])
                    if i in seen or not 0 <= i < len(frames):
                        raise PerceptionError(f"Invalid or duplicate SAM3.1 frame index: {i}")
                    seen.add(i)
                    arrays = mask_arrays(response["outputs"], shapes[i])
                    file = f"concept_{concept_index:03d}/{i:06d}.npz"
                    np.savez_compressed(output / file, **arrays)
                    records.append({**frame_metadata(frames[i], shapes[i]), "frame_index": i,
                                    "mask_file": file, "shape": shapes[i],
                                    "original_to_output_pixels": arrays["original_to_output_pixels"].tolist(),
                                    "objects": [{"local_id": int(j), "track_id": f"{session_key}/{j}", "concept": concept} for j in arrays["object_ids"]]})
                if seen != set(range(len(frames))):
                    raise PerceptionError("SAM3.1 propagation returned incomplete frames")
                sessions.append({"concept": concept, "session_key": session_key,
                                 "frames": sorted(records, key=lambda r: r["frame_index"])})
            finally:
                predictor.handle_request({"type": "close_session", "session_id": sid})
    return {"sessions": sessions, "model_load_seconds": load_seconds,
            "identity_scope": "run/concept/session/local_id; never a cross-concept or global object identity",
            "pixel_convention": "Original image grid; identity pixel transform; bool masks stored losslessly",
            "short_clip_warning": ("Temporal track confirmation may suppress short clips. Empty outputs are not evidence of absence; use adjacent frames and inspect native detections." if len(frames) < 16 else None),
            "observed_object_counts": [len(f["objects"]) for session in sessions for f in session["frames"]],
            "semantic_accuracy_verified": False,
            "inference_parameters": {"max_num_objects": args.max_objects, "compile": False,
                                     "use_fa3": False, "use_rope_real": False}}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "depth", "segment"])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--backend", choices=["all", "da3", "sam31"], default="all", help="check only")
    p.add_argument("--frames", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--ids", nargs="+", help="exact IDs; input manifest order is retained")
    p.add_argument("--audit-evidence", action="store_true",
                   help="prepare independent audit references only; requires exact --ids and original audit/holdout split tags")
    p.add_argument("--concepts", nargs="+", help="SAM3.1 semantic or part prompts")
    p.add_argument("--resolution", type=int, help="DA3 longest-side input processing size; outputs are never decimated")
    p.add_argument("--max-objects", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="validate inputs and plan only; no weights or inference")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    config, base = read_config(args.config)
    backends = (["da3", "sam31"] if args.backend == "all" else [args.backend]) if args.command == "check" else ["da3" if args.command == "depth" else "sam31"]
    if not args.worker:
        status = 0
        for backend in backends:
            runtime = backend_config(config, base, backend)
            if not Path(runtime["python"]).is_file():
                raise PerceptionError(f"Backend Python missing: {runtime['python']}")
            command = [runtime["python"], str(Path(__file__).absolute()), *arguments, "--worker"]
            if args.command == "check":
                command += ["--backend", backend]
            status = max(status, subprocess.run(command, env=offline_environment(), check=False).returncode)
        return status
    backend = backends[0]
    runtime = backend_config(config, base, backend)
    if args.command == "check":
        report = readiness(runtime, backend)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 2
    if args.frames is None or args.output is None:
        raise PerceptionError("depth/segment require --frames and --output")
    frames = load_frames(args.frames, args.ids, audit_evidence=args.audit_evidence)
    shapes = image_shapes(frames)
    validate_plan(backend, runtime, frames, shapes, args)
    if args.dry_run:
        print(json.dumps({"backend": backend, "frame_ids": [f["id"] for f in frames],
                          "audit_evidence": args.audit_evidence,
                          "frame_splits": [{k: f[k] for k in ("id", "split", "evaluation_split") if k in f} for f in frames],
                          "original_shapes": shapes, "runtime": runtime, "output": str(args.output.absolute()),
                          "concepts": (args.concepts or DEFAULT_CONCEPTS) if backend == "sam31" else None,
                          "weights_loaded": False, "inference_verified": False}, indent=2))
        return 0
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise PerceptionError(f"Output directory must be empty to prevent stale artifacts: {args.output}")
    report = readiness(runtime, backend)
    if not report["ready"]:
        raise PerceptionError(json.dumps(report, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    args.run_id = uuid.uuid4().hex
    import torch
    if runtime["device"].startswith("cuda"):
        if ":" in runtime["device"]:
            torch.cuda.set_device(runtime["device"])
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    payload = (run_depth if backend == "da3" else run_segment)(runtime, frames, shapes, args.output, args)
    checkpoint = Path(runtime["checkpoint"])
    provenance = {"source": runtime["source"], "checkpoint": str(checkpoint),
                  "api_source": report["api_source"], "python": sys.executable}
    model_config = checkpoint / "config.json" if checkpoint.is_dir() else checkpoint.parent / "config.json"
    if model_config.is_file():
        provenance["model_config_sha256"] = hashlib.sha256(model_config.read_bytes()).hexdigest()
    device_info = {"requested_device": runtime["device"], "stored_float_dtype": "float32",
                   "model_compute_precision": "backend-managed autocast"}
    if runtime["device"].startswith("cuda"):
        torch.cuda.synchronize()
        device_info.update(device_name=torch.cuda.get_device_name(),
                           autocast_dtype="bfloat16" if backend == "sam31" or torch.cuda.is_bf16_supported() else "float16",
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    manifest = {"schema_version": 1, "backend": backend, "run_id": args.run_id,
                "audit_evidence": args.audit_evidence, "requested_ids": args.ids,
                "inference_verified": True, "elapsed_seconds": time.perf_counter() - started,
                "source_frames": str(args.frames.absolute()), "provenance": provenance,
                "device": device_info, "seed": args.seed, **payload}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str((args.output / "manifest.json").absolute()),
                      "backend": backend, "frames": len(frames), "inference_verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PerceptionError, FileNotFoundError, json.JSONDecodeError, ImportError) as exc:
        print(f"perceive: {exc}", file=sys.stderr)
        raise SystemExit(2)
