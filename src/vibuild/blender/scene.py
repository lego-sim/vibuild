#!/usr/bin/env python3
"""Read-only Blender inventory and calibrated native-grid geometry export.

Run with ordinary Python; --blender selects the external Blender executable.
Render manifest: {"frames":[{"id":"001","width":640,"height":480,
"K":[[500,0,319.5],[0,500,239.5],[0,0,1]], "camera_to_world": [...]}]}.
The 4x4 pose uses OpenCV camera axes in scene world coordinates (original scene units).
Optional camera names verify registered poses; optional blender_frame evaluates
animation in memory. No .blend is saved. Geometry is first-surface ray casting,
including transparent panes, not photometric depth inferred from RGB.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import time
import shutil
import runpy

# Blender runs this file with its bundled Python.
if __package__ in (None, ""):
    # Expose only this pure-Python package to Blender, not the host's site-packages.
    # NumPy extensions from another Python version cannot be loaded by Blender.
    import importlib.util

    package = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "vibuild", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vibuild"] = module
    spec.loader.exec_module(module)
from vibuild.runtime.cache import (
    digest,
    fingerprint,
    file_hash,
    entry_lock,
    verified,
    restore,
    publish,
    OutputProgress,
)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def external_files():
    import bpy
    from collections import Counter

    raw = bpy.utils.blend_paths(absolute=False, packed=False, local=False)
    absolute = bpy.utils.blend_paths(absolute=True, packed=False, local=False)
    # Appended IDs retain their source .blend as provenance, not a runtime input.
    # Count occurrences instead of removing a path wholesale: an actual asset can
    # use the same path, and a genuinely linked library must always remain pinned.
    linked = {
        str(Path(bpy.path.abspath(lib.filepath)).resolve())
        for lib in bpy.data.libraries
    }
    weak = Counter()
    for block in bpy.data.user_map():
        ref = block.library_weak_reference
        if ref:
            target = str(
                Path(bpy.path.abspath(ref.filepath, library=block.library)).resolve()
            )
            if target not in linked:
                weak[(ref.filepath, target)] += 1
    result = []
    seen = set()
    hashes = {}
    for path, target in zip(raw, absolute):
        target = str(Path(target).resolve())
        key = (path, target)
        if weak[key]:
            weak[key] -= 1
            continue
        if key in seen:
            continue
        seen.add(key)
        if target not in hashes:
            hashes[target] = file_hash(target)
        result.append({"path": path, "resolved": target, "sha256": hashes[target]})
    return result


def dependencies_match(report, blend, *, allow_missing=False):
    hashes = {}
    for asset in report.get("external_files", []):
        raw = asset["path"]
        target = (
            blend.resolve().parent / raw[2:]
            if raw.startswith("//")
            else Path(asset["resolved"])
        )
        # Unresolved sequences/UDIMs and missing files disable caching conservatively.
        if target not in hashes:
            hashes[target] = file_hash(target)
        if (asset["sha256"] is None and not allow_missing) or hashes[target] != asset[
            "sha256"
        ]:
            return False
    return "external_files" in report


def apply_labels(manifest):
    """Only apply explicitly reviewed mappings in the disposable Blender process."""
    import bpy

    mapping = manifest.get("object_labels")
    if mapping is None:
        return None
    if (
        not mapping.get("reviewer")
        or not mapping.get("evidence")
        or not isinstance(mapping.get("objects"), dict)
    ):
        raise ValueError(
            "object_labels needs reviewer, evidence files and explicit objects mapping"
        )
    for path in mapping["evidence"]:
        if not Path(path).is_file():
            raise ValueError(f"label mapping evidence missing: {path}")
    classes = manifest.get("classes", {})
    for name, labels in mapping["objects"].items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise ValueError(f"label mapping object is not a mesh: {name}")
        if (
            set(labels) != {"semantic_class", "instance_id"}
            or labels["semantic_class"] not in classes
            or not isinstance(labels["instance_id"], str)
            or not labels["instance_id"]
        ):
            raise ValueError(
                f"label mapping requires a declared class and nonempty instance_id: {name}"
            )
        for key, value in labels.items():
            obj[key] = value
    return mapping


def blender_main(args):
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    from vibuild.blender.geometry import (
        apply_visibility,
        build_bvh,
        canonical_cast,
        validate_rays,
    )

    started = time.monotonic()

    def pose(camera):
        return np.array(camera.matrix_world, dtype=float) @ np.diag(
            [1.0, -1.0, -1.0, 1.0]
        )

    def frame_pose(frame):
        if "camera_to_world" in frame:
            return np.asarray(frame["camera_to_world"], float)
        result = np.eye(4)
        result[:3, :3] = np.asarray(frame["rotation"], float)
        result[:3, 3] = np.asarray(frame["center"], float)
        return result

    def manifest_frames(manifest):
        records = (
            manifest
            if isinstance(manifest, list)
            else manifest.get("frames", manifest.get("views", []))
        )
        return [
            {**f, "id": str(f.get("id", f.get("index", i)))}
            for i, f in enumerate(records)
        ]

    def camera_record(camera):
        data = camera.data
        return {
            "name": camera.name,
            "source_image": str(camera.get("source_image", "")),
            "is_registered": bool(
                camera.get("source_image") or camera.get("registered", False)
            ),
            "camera_to_world": pose(camera).tolist(),
            "projection": data.type,
            "lens_mm": data.lens,
            "sensor_width_mm": data.sensor_width,
            "sensor_height_mm": data.sensor_height,
            "sensor_fit": data.sensor_fit,
            "shift_x": data.shift_x,
            "shift_y": data.shift_y,
            "clip_start": data.clip_start,
            "clip_end": data.clip_end,
        }

    def snapshot():
        return [
            camera_record(o)
            for o in sorted(bpy.data.objects, key=lambda o: o.name)
            if o.type == "CAMERA"
        ]

    def intrinsic(camera, width, height):
        d = camera.data
        if d.type != "PERSP":
            raise ValueError("only pinhole perspective cameras are supported")
        ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
        fit = d.sensor_fit
        if fit == "AUTO":
            fit = "HORIZONTAL" if width >= height * ratio else "VERTICAL"
        view = width if fit == "HORIZONTAL" else height * ratio
        sensor = d.sensor_height if d.sensor_fit == "VERTICAL" else d.sensor_width
        fx = d.lens / sensor * view
        return np.array(
            [
                [fx, 0, (width - 1) / 2 - d.shift_x * view],
                [0, fx / ratio, (height - 1) / 2 + d.shift_y * view / ratio],
                [0, 0, 1.0],
            ]
        )

    bpy.ops.wm.open_mainfile(filepath=str(Path(args.blend).resolve()))
    scene = bpy.context.scene
    before = snapshot()
    original_scale = scene.unit_settings.scale_length
    source_cameras = {o.name: o for o in bpy.data.objects if o.type == "CAMERA"}
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "joints":
        from vibuild.blender.joints import check

        report = check(json.loads(args.manifest.read_text()))
        report.update(
            source_blend=str(args.blend.resolve()),
            source_sha256=digest(args.blend),
            manifest_sha256=digest(args.manifest),
            external_files=external_files(),
            checker_sha256=digest(Path(__file__).with_name("joints.py")),
            wrapper_sha256=digest(Path(__file__)),
            registered_cameras_unchanged=snapshot() == before,
            scene_unit_scale=original_scale,
            saved_scene=False,
        )
        dump(out / "joints.json", report)
        return
    if args.action == "patch":
        # A bounded edit script only modifies the loaded scene. This wrapper owns save/exit.
        bpy.context.preferences.filepaths.save_version = 0
        runpy.run_path(
            str(args.script.resolve()),
            init_globals={"OUTPUT_DIR": str(out)},
            run_name="__main__",
        )
        if (
            snapshot() != before
            or bpy.context.scene.unit_settings.scale_length != original_scale
        ):
            raise ValueError("patch changed camera registration or scene scale")
        if args._single_scene_edit:
            bpy.ops.file.pack_all()
            if external_files():
                raise ValueError(
                    "Working scene must be self-contained; unresolved external dependencies remain"
                )
        candidate = out / ".candidate.pending.blend"
        bpy.ops.wm.save_as_mainfile(
            filepath=str(candidate), check_existing=False, compress=False
        )
        dump(
            out / "patch.json",
            {
                "source_blend": str(args.blend.resolve()),
                "script_sha256": digest(args.script) if args.script else None,
                "candidate": "candidate.blend",
                "registered_cameras_unchanged": True,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return
    progress = output_progress(
        args, json.loads((out / OutputProgress.marker).read_text())["request"]
    )
    input_files = external_files()
    progress.bind_inputs(input_files)
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
    mapping = apply_labels(manifest) if isinstance(manifest, dict) else None
    selected_collections = set(args.collection or [])
    if not selected_collections <= {c.name for c in bpy.data.collections}:
        raise ValueError("one or more requested collections do not exist")
    selected_objects = set()
    for name in selected_collections:
        selected_objects.update(o.name for o in bpy.data.collections[name].all_objects)
    requested = set(args.object or [])
    if requested and not requested <= {o.name for o in scene.objects}:
        raise ValueError("Requested inspection object does not exist")
    inventory = []
    for obj in sorted(scene.objects, key=lambda o: o.name):
        if requested and obj.name not in requested:
            continue
        if (
            selected_collections
            and obj.type != "CAMERA"
            and obj.name not in selected_objects
        ):
            continue
        if args.action != "inspect":
            inventory.append(
                {"name": obj.name, "semantic_class": str(obj.get("semantic_class", ""))}
            )
            continue
        entry = {
            "name": obj.name,
            "type": obj.type,
            "collections": [c.name for c in obj.users_collection],
            "hide_render": bool(obj.hide_render),
            "semantic_class": str(obj.get("semantic_class", "")),
            "instance_id": str(obj.get("instance_id", "")),
            "materials": [
                slot.material.name if slot.material else None
                for slot in obj.material_slots
            ],
            "provenance": {
                k: str(v)
                for k, v in obj.items()
                if k not in ("_RNA_UI", "semantic_class", "instance_id")
            },
            "matrix_world": [list(v) for v in obj.matrix_world],
        }
        if obj.type == "MESH":
            entry.update(
                vertices=len(obj.data.vertices),
                polygons=len(obj.data.polygons),
                world_bounds=[
                    list(obj.matrix_world @ Vector(v)) for v in obj.bound_box
                ],
            )
        if args.mesh_detail and obj.type == "MESH":
            entry["mesh"] = {
                "vertices_world": [
                    list(obj.matrix_world @ v.co) for v in obj.data.vertices
                ],
                "faces": [list(p.vertices) for p in obj.data.polygons],
            }
        inventory.append(entry)
    report = {
        "schema_version": 1,
        "source_blend": str(Path(args.blend).resolve()),
        "objects": inventory,
        "camera_counts": {
            "registered": sum(c["is_registered"] for c in before),
            "other": sum(not c["is_registered"] for c in before),
        },
        "scene_unit_scale": scene.unit_settings.scale_length,
        "saved_scene": False,
        "selected_collections": sorted(selected_collections),
        "external_files": input_files,
    }
    report["camera_signature"] = fingerprint(before)
    if args.camera_details:
        report["registered_cameras"] = [c for c in before if c["is_registered"]]
        report["review_or_unlabelled_cameras"] = [
            c for c in before if not c["is_registered"]
        ]
    report["materials"] = [
        {
            "name": m.name,
            "use_nodes": m.use_nodes,
            "diffuse_color": list(m.diffuse_color),
            "nodes": [
                {
                    "name": n.name,
                    "type": n.bl_idname,
                    "image": n.image.filepath if getattr(n, "image", None) else None,
                }
                for n in m.node_tree.nodes
            ]
            if m.node_tree
            else [],
        }
        for m in sorted(bpy.data.materials, key=lambda m: m.name)
        if args.action == "inspect"
        and (not requested or any(m.name in o["materials"] for o in inventory))
    ]
    classes = manifest.get("classes", {}) if isinstance(manifest, dict) else {}
    report["label_coverage"] = {
        c: [o["name"] for o in inventory if o["semantic_class"] == c] for c in classes
    }
    report["missing_class_labels"] = [
        c for c, names in report["label_coverage"].items() if not names
    ]
    report["object_labels"] = mapping
    if args.action == "inspect":
        if args.manifest:
            manifest = json.loads(Path(args.manifest).read_text())
            checks = []
            width = int(
                scene.render.resolution_x * scene.render.resolution_percentage / 100
            )
            height = int(
                scene.render.resolution_y * scene.render.resolution_percentage / 100
            )
            for f in manifest_frames(manifest):
                cam = source_cameras.get(f.get("camera"))
                check = {
                    "id": f.get("id"),
                    "camera": f.get("camera"),
                    "status": "unknown",
                }
                if cam is not None:
                    expected = frame_pose(f)
                    check["pose_matches"] = bool(
                        np.allclose(pose(cam), expected, atol=1e-6, rtol=0)
                    )
                    check["current_render_K"] = intrinsic(cam, width, height).tolist()
                    check["current_render_K_matches"] = bool(
                        width == f["width"]
                        and height == f["height"]
                        and np.allclose(
                            check["current_render_K"], f["K"], atol=1e-5, rtol=0
                        )
                    )
                    check["status"] = (
                        "passed"
                        if check["pose_matches"] and check["current_render_K_matches"]
                        else "failed"
                    )
                checks.append(check)
            report["registered_camera_checks"] = checks
        dump(out / "inventory.json", report)
        return

    # Rendering needs compact provenance, not a duplicate full scene inventory per group.
    report["object_count"] = len(report.pop("objects"))
    for field in ("materials", "registered_cameras", "review_or_unlabelled_cameras"):
        report.pop(field, None)

    manifest = json.loads(Path(args.manifest).read_text())
    frames = manifest_frames(manifest)
    if args.frame:
        selected = set(args.frame)
        available = {str(f["id"]) for f in frames}
        if not selected <= available:
            raise ValueError(f"unknown frame IDs: {sorted(selected - available)}")
        frames = [f for f in frames if str(f["id"]) in selected]
    if not frames:
        raise ValueError("no frames selected")
    grid_records = {}
    if args.depth_grid:
        grid_manifest = json.loads(args.depth_grid.read_text())
        grid_records = {f["id"]: f for f in manifest_frames(grid_manifest)}
        if any(f["id"] not in grid_records for f in frames):
            raise ValueError("depth-grid manifest must contain every selected frame ID")
    # Align evaluated geometry visibility with the requested render scope. These
    # changes live only in this disposable Blender process; the source is never saved.
    apply_visibility(
        scene,
        bpy.context.view_layer,
        selected_objects if selected_collections else None,
    )
    if args.appearance or any(f.get("appearance", False) for f in frames):
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = args.denoise
        if args.denoise:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.seed = args.seed
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.threads
    classes = (
        manifest.get("classes", {"window_frame": 1, "window_pane": 2, "tile": 3})
        if isinstance(manifest, dict)
        else {"window_frame": 1, "window_pane": 2, "tile": 3}
    )
    if not isinstance(classes, dict) or any(
        not isinstance(v, int) or v <= 0 for v in classes.values()
    ):
        raise ValueError("classes must map names to positive integer IDs")
    # Stable scene instance IDs; explicitly grouped objects share the custom ID.
    keys = sorted(
        {str(o.get("instance_id", o.name)) for o in scene.objects if o.type == "MESH"}
    )
    instances = {key: i + 1 for i, key in enumerate(keys)}
    output_frames = []
    baseline_frame = scene.frame_current
    for f in frames:
        frame_started = time.monotonic()
        frame_id = str(f["id"])
        if Path(frame_id).name != frame_id or frame_id in ("", ".", ".."):
            raise ValueError("frame IDs must be safe file names")
        completed = progress.get("frame:" + frame_id)
        if completed is not None:
            output_frames.append(completed)
            continue
        scene.frame_set(int(f.get("blender_frame", baseline_frame)))
        grid = grid_records.get(frame_id, f)
        if "depth_file" in grid:
            with np.load(
                args.depth_grid.resolve().parent / grid["depth_file"],
                allow_pickle=False,
            ) as buffers:
                h, w = buffers["depth"].shape
                K = np.asarray(
                    buffers["K"] if "K" in buffers else buffers["intrinsics"], float
                )
        else:
            w, h = int(grid["width"]), int(grid["height"])
            K = np.asarray(grid["K"], float)
        c2w = frame_pose(f)
        if (
            w <= 0
            or h <= 0
            or K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or K[0, 1] != 0
            or K[1, 0] != 0
        ):
            raise ValueError(
                "render requires positive native dimensions and zero-skew pinhole K"
            )
        if (
            c2w.shape != (4, 4)
            or not np.isfinite(c2w).all()
            or not np.allclose(c2w[3], [0, 0, 0, 1])
            or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError("camera_to_world must be a rigid OpenCV transform")
        registered = source_cameras.get(f.get("camera"))
        if f.get("camera") and registered is None:
            raise ValueError(f"registered camera {f['camera']} does not exist")
        if registered and not np.allclose(pose(registered), c2w, atol=1e-6, rtol=0):
            raise ValueError(f"registered camera pose differs for {frame_id}")
        # Separate temporary camera; source camera data and transforms stay intact.
        data = bpy.data.cameras.new("vibuild_export_camera")
        camera = bpy.data.objects.new("vibuild_export_camera", data)
        scene.collection.objects.link(camera)
        camera.matrix_world = Matrix(c2w @ np.diag([1.0, -1.0, -1.0, 1.0]))
        scene.camera = camera
        scene.render.resolution_x = w
        scene.render.resolution_y = h
        scene.render.resolution_percentage = 100
        ratio = K[0, 0] / K[1, 1]
        scene.render.pixel_aspect_x = max(1.0, 1.0 / ratio)
        scene.render.pixel_aspect_y = max(1.0, ratio)
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36.0
        data.lens = K[0, 0] * 36.0 / w
        data.shift_x = ((w - 1) / 2 - K[0, 2]) / w
        data.shift_y = (K[1, 2] - (h - 1) / 2) * ratio / w
        actual_K = intrinsic(camera, w, h)
        if not np.allclose(K, actual_K, atol=1e-4, rtol=1e-6):
            raise ValueError("Blender camera does not reproduce requested K")
        bpy.context.view_layer.update()
        item = {
            "id": frame_id,
            "width": w,
            "height": h,
            "K": actual_K.tolist(),
            "camera_to_world": c2w.tolist(),
            "registered_pose_verified": registered is not None,
        }
        if f.get("geometry", True):
            deps = bpy.context.evaluated_depsgraph_get()
            depth = np.full((h, w), np.nan, np.float32)
            normal = np.zeros((h, w, 3), np.float32)
            semantic = np.zeros((h, w), np.int32)
            instance = np.zeros((h, w), np.int32)
            origin = Vector(c2w[:3, 3])
            rot = c2w[:3, :3]
            invK = np.linalg.inv(K)
            # Precompute rays once; retain every native pixel and float32 optical Z.
            yy, xx = np.indices((h, w))
            directions = (
                rot @ invK @ np.stack((xx.ravel(), yy.ravel(), np.ones(w * h)))
            ).T
            if args.ray_backend == "bvh":
                cast, build_stats = build_bvh(deps)
                validation = validate_rays(
                    scene,
                    deps,
                    cast,
                    origin,
                    directions,
                    args.validate_rays,
                    args.max_distance,
                )
            else:
                cast = lambda origin, direction, distance: canonical_cast(
                    scene, deps, origin, direction, distance
                )
                build_stats, validation = {}, {"backend": "Blender scene.ray_cast"}
            geometry_started = time.monotonic()
            for y in range(h):
                for x in range(w):
                    result = cast(
                        origin,
                        Vector(directions[y * w + x]).normalized(),
                        args.max_distance,
                    )
                    if result is not None:
                        point, norm, original = result
                        depth[y, x] = float((rot.T @ (np.array(point) - c2w[:3, 3]))[2])
                        normal[y, x] = np.array(norm)
                        semantic[y, x] = classes.get(
                            str(original.get("semantic_class", "")), 0
                        )
                        instance[y, x] = instances.get(
                            str(original.get("instance_id", original.name)), 0
                        )
                if y % max(h // 10, 1) == 0:
                    print(f"geometry {frame_id}: {y}/{h} rows", flush=True)
            geometry_seconds = time.monotonic() - geometry_started
            np.savez_compressed(
                out / f"{frame_id}.npz",
                depth=depth,
                normal_world=normal,
                semantic=semantic,
                instance=instance,
                K=K,
                camera_to_world=c2w,
                depth_convention=np.array("z"),
                depth_units=np.array("scene"),
            )
            item.update(
                {
                    "id": frame_id,
                    "prediction": f"{frame_id}.npz",
                    "width": w,
                    "height": h,
                    "K": actual_K.tolist(),
                    "camera_to_world": c2w.tolist(),
                    "finite_depth_pixels": int(np.isfinite(depth).sum()),
                    "registered_pose_verified": registered is not None,
                    "grid_source": str(args.depth_grid.resolve())
                    if args.depth_grid
                    else "input frame calibration",
                    "original_image_K": f["K"],
                    "pixel_convention": "OpenCV integer centres",
                    "ray_validation": validation,
                    "bvh": build_stats,
                    "geometry_seconds": geometry_seconds,
                }
            )
        if f.get("appearance", args.appearance):
            rgb_started = time.monotonic()
            scene.render.use_border = False
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = str(out / f"{frame_id}.png")
            bpy.ops.render.render(write_still=True)
            item["appearance"] = f"{frame_id}.png"
            item["appearance_seconds"] = time.monotonic() - rgb_started
        item["elapsed_seconds"] = time.monotonic() - frame_started
        output_frames.append(item)
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(data)
        progress.commit(
            "frame:" + frame_id,
            [item[k] for k in ("prediction", "appearance") if k in item],
            item,
        )
    scene.frame_set(baseline_frame)
    after = snapshot()
    if before != after:
        raise RuntimeError("registered camera invariants changed")
    report.update(
        frames=output_frames,
        classes=classes,
        instances=instances,
        registered_cameras_unchanged=True,
        geometry_semantics="first surface including transparent panes; render-hidden objects excluded; evaluated active view layer",
        geometry_limitations=[
            "geometry instances require object semantic_class and instance_id labels",
            "procedural shader tiles are not separate geometry instances",
            "depth uses original scene units; metric scale is not inferred",
        ],
        max_ray_distance=args.max_distance,
        ray_backend=args.ray_backend,
        elapsed_seconds=time.monotonic() - started,
        appearance_settings={
            "engine": "CYCLES",
            "device": "CPU",
            "samples": args.samples,
            "seed": args.seed,
            "threads": args.threads,
            "denoise": args.denoise,
        }
        if args.appearance or any(f.get("appearance", False) for f in frames)
        else None,
    )
    dump(out / "render_manifest.json", report)


def render_key(args, protocol=False):
    sources = [
        Path(__file__),
        Path(__file__).with_name("geometry.py"),
        Path(__file__).parents[1] / "runtime/cache.py",
    ]
    inputs = {"blend": digest(args.blend), "tools": [digest(p) for p in sources]}
    for key in ("manifest", "depth_grid"):
        path = getattr(args, key)
        if path:
            value = json.loads(path.read_text())
            inputs[key] = value
            if isinstance(value, dict) and value.get("object_labels"):
                inputs["label_evidence"] = [
                    file_hash(p) for p in value["object_labels"].get("evidence", [])
                ]
            if key == "depth_grid":
                frames = (
                    value
                    if isinstance(value, list)
                    else value.get("frames", value.get("views", []))
                )
                inputs["depth_files"] = [
                    file_hash(path.resolve().parent / f["depth_file"])
                    for f in frames
                    if "depth_file" in f
                ]
    inputs["options"] = {
        k: getattr(args, k)
        for k in (
            "action",
            "frame",
            "collection",
            "object",
            "mesh_detail",
            "camera_details",
            "appearance",
            "max_distance",
            "ray_backend",
            "validate_rays",
            "samples",
            "denoise",
            "threads",
            "seed",
        )
    }
    if not args.blender or not shutil.which(args.blender):
        raise ValueError(
            "Configure runtime.blender or supply --blender with an installed executable"
        )
    executable = Path(shutil.which(args.blender)).resolve()
    inputs["blender"] = [
        str(executable),
        executable.stat().st_size,
        executable.stat().st_mtime_ns,
    ]
    if protocol:
        del inputs["blend"]
    return fingerprint(inputs)


def export(args, forwarded):
    from contextlib import nullcontext

    if not args.blender or not shutil.which(args.blender):
        raise ValueError(
            "Configure runtime.blender or supply --blender with an installed executable"
        )
    before = digest(args.blend)
    command = [
        args.blender,
        "--background",
        "-noaudio",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        *forwarded,
        "--_inside-blender",
    ]
    remaining = float(os.environ.get("VIBUILD_DEADLINE", "inf")) - time.time()
    try:
        sys.stdout.fileno()
        sys.stderr.fileno()
        output = {"stdout": sys.stdout, "stderr": sys.stderr}
    except (AttributeError, OSError):
        output = {}
    reservation = nullcontext()
    config = os.environ.get("VIBUILD_CONFIG")
    if args.action == "render" and config:
        from vibuild.blender.capacity import render_reservation
        from vibuild.runtime.scene import task_root

        root = task_root(args.blend)
        reservation = render_reservation(
            root / ".runtime" if root else args.output / ".runtime",
            config,
            deadline=float(os.environ.get("VIBUILD_DEADLINE", "inf")),
            stop_paths=json.loads(os.environ.get("VIBUILD_PARENT_STOPS", "[]")),
        )
    with reservation:
        remaining = float(os.environ.get("VIBUILD_DEADLINE", "inf")) - time.time()
        # The writer retains the output lock if its Python launcher is interrupted.
        # Otherwise an orphaned Blender could race a retry in the same directory.
        inherited = (args._output_lock_fd,) if hasattr(args, "_output_lock_fd") else ()
        completed = subprocess.run(
            command,
            check=False,
            timeout=max(0.01, min(args.timeout_seconds, remaining)),
            pass_fds=inherited,
            **output,
        )
    if digest(args.blend) != before:
        raise RuntimeError("source blend content changed during export")
    return completed.returncode


def output_progress(args, request):
    names = {"inventory.json"} if args.action == "inspect" else {"render_manifest.json"}
    if args.action == "render":
        manifest = json.loads(args.manifest.read_text())
        frames = (
            manifest
            if isinstance(manifest, list)
            else manifest.get("frames", manifest.get("views", []))
        )
        ids = [str(f.get("id", f.get("index", i))) for i, f in enumerate(frames)]
        if len(set(ids)) != len(ids):
            raise ValueError("Render frame IDs must be unique")
        for ident in ids:
            if Path(ident).name != ident or ident in ("", ".", ".."):
                raise ValueError("Render frame IDs must be safe file names")
            if not args.frame or ident in args.frame:
                names.update((ident + ".npz", ident + ".png"))
    return OutputProgress(args.output, request, names)


def read_output(args, forwarded, started):
    """Resume a known request while holding its output lock, outside that directory."""
    report_name = (
        "render_manifest.json" if args.action == "render" else "inventory.json"
    )
    key = render_key(args)
    request = dict(
        key=key,
        source_blend=str(args.blend.resolve()),
        source_sha256=digest(args.blend),
    )
    progress = output_progress(args, request)
    if progress.state["inputs"] is not None and not dependencies_match(
        {"external_files": progress.state["inputs"]}, args.blend
    ):
        raise ValueError(
            "Output external inputs changed or cannot be verified; use a new output directory"
        )
    if progress.get("complete") is not None:
        print(json.dumps({"output": str(args.output)}))
        return 0

    def finish(cache_hit):
        # Blender commits each completed frame. Reload those receipts after it exits.
        state = output_progress(args, request)
        report = json.loads((args.output / report_name).read_text())
        state.bind_inputs(report["external_files"])
        if render_key(args) != key or not dependencies_match(
            report, args.blend, allow_missing=True
        ):
            raise ValueError("Scene, request or external inputs changed during export")
        report.update(
            source_blend=str(args.blend.resolve()),
            source_sha256=request["source_sha256"],
            render_protocol_sha256=render_key(args, protocol=True),
            cache={"key": key, "hit": cache_hit},
            request_seconds=time.monotonic() - started,
        )
        dump(args.output / report_name, report)
        names = [report_name]
        for frame in report.get("frames", []):
            names += [frame[k] for k in ("prediction", "appearance") if k in frame]
        # Missing/sequence assets can still be diagnosed by a fresh inspection,
        # but their unverified contents must not authorize later output reuse.
        if dependencies_match(report, args.blend):
            state.commit("complete", names, {"report": report_name})
        return report, names

    cache_hit = False
    if args.cache_dir:
        with entry_lock(args.cache_dir.resolve() / "scene", key) as entry:
            # Do not overwrite already committed frames while restoring a whole cache entry.
            if (
                not progress.state["receipts"]
                and verified(entry)
                and dependencies_match(
                    json.loads((entry / report_name).read_text()), args.blend
                )
            ):
                restore(entry, args.output)
                cache_hit = True
            else:
                code = export(args, forwarded)
                if code:
                    return code
            report, names = finish(cache_hit)
            if not cache_hit and dependencies_match(report, args.blend):
                publish(entry, args.output, names)
    else:
        code = export(args, forwarded)
        if code:
            return code
        report, _ = finish(False)
    print(json.dumps({"output": str(args.output)}))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "iterate":
        from vibuild.blender.iteration import main as iterate

        return iterate(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "render", "patch", "joints"))
    parser.add_argument(
        "--script",
        type=Path,
        help="patch only: bounded bpy edit script; wrapper saves candidate synchronously",
    )
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--depth-grid",
        type=Path,
        help="explicit grid manifest matched by frame ID: depth_file NPZ or width/height/K",
    )
    parser.add_argument(
        "--frame", action="append", help="select an ID; repeat for multiple frames"
    )
    parser.add_argument(
        "--collection",
        action="append",
        help="restrict mesh scope to a collection and descendants; repeatable",
    )
    parser.add_argument(
        "--object", action="append", help="inspect only: select exact object names"
    )
    parser.add_argument(
        "--mesh-detail",
        action="store_true",
        help="inspect only: include selected mesh world vertices and faces",
    )
    parser.add_argument(
        "--camera-details",
        action="store_true",
        help="inspect only: include full camera records; default is counts and integrity signature",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for reports and buffers"
    )
    parser.add_argument(
        "--appearance",
        action="store_true",
        help="also render deterministic CPU Cycles RGB with source materials",
    )
    parser.add_argument("--max-distance", type=float, default=10000.0)
    parser.add_argument("--ray-backend", choices=["bvh", "scene"], default="bvh")
    parser.add_argument(
        "--validate-rays",
        type=int,
        default=128,
        help="fixed-seed Blender oracle checks per frame (minimum 1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="deterministic CPU Cycles samples for RGB",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="RGB visual preview only; alters appearance protocol",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600,
        help="per-export timeout, capped by the session deadline",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=os.environ.get("VIBUILD_CACHE_DIR"),
        help="shared content-addressed cache; defaults to session tool cache",
    )
    parser.add_argument(
        "--_inside-blender", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_single-scene-edit", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    from vibuild.runtime.scene import task_root

    managed = task_root(args.blend)
    if managed and args.action == "patch" and not args._single_scene_edit:
        parser.error(
            "Use scene iterate --script for the single working scene; direct candidate copies are disabled"
        )
    if (
        args.object or args.mesh_detail or args.camera_details
    ) and args.action != "inspect":
        parser.error("object/mesh-detail/camera-details are for inspection only")
    if args.mesh_detail and not args.object:
        parser.error("mesh-detail requires an explicit object selection")
    if args.action == "patch" and (not args.script or not args.script.is_file()):
        parser.error("patch requires an existing --script")
    if args.action in ("render", "joints") and not args.manifest:
        parser.error(f"{args.action} requires --manifest")
    if (
        args.max_distance <= 0
        or args.validate_rays < 1
        or args.samples < 1
        or args.threads < 1
        or args.timeout_seconds <= 0
    ):
        parser.error("distance, validation rays, samples and threads must be positive")
    if args._inside_blender:
        blender_main(args)
        return 0
    args.output = args.output.resolve()
    if (
        args.action not in ("render", "inspect")
        and args.output.exists()
        and any(args.output.iterdir())
    ):
        parser.error("output directory must be empty; preserve versioned artifacts")
    started = time.monotonic()
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if args.action == "joints":
        expected_manifest = digest(args.manifest)
        code = export(args, forwarded)
        if code:
            return code
        path = args.output / "joints.json"
        report = json.loads(path.read_text())
        if (
            digest(args.manifest) != expected_manifest
            or report["manifest_sha256"] != expected_manifest
        ):
            raise ValueError("Joint manifest changed during check")
        print(json.dumps({"joints": str(path), "passed": report["passed"]}))
        return 0 if report["passed"] else 2
    if args.action == "patch":
        code = export(args, forwarded)
        if code:
            return code
        report = json.loads((args.output / "patch.json").read_text())
        pending = args.output / ".candidate.pending.blend"
        with pending.open("rb") as stream:
            if stream.read(7) != b"BLENDER":
                raise ValueError(
                    "patch did not save a complete uncompressed Blender file"
                )
        # Only publish the candidate after the save process completed successfully.
        pending.rename(args.output / "candidate.blend")
        report.update(
            candidate_sha256=digest(args.output / "candidate.blend"),
            source_sha256=digest(args.blend),
            request_seconds=time.monotonic() - started,
        )
        dump(args.output / "patch.json", report)
        print(json.dumps({"result": str(args.output / "patch.json")}))
        return 0
    with entry_lock(
        args.output.parent, "." + args.output.name + ".scene-output", with_handle=True
    ) as (_, handle):
        args._output_lock_fd = handle.fileno()
        return read_output(args, forwarded, started)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    if arguments and "--_inside-blender" in arguments:
        # This is a disposable worker, with all outputs synchronously written.
        # Bypass Blender's audio teardown, which can hang in headless sandboxes.
        code = 0
        try:
            code = main(arguments)
        except Exception:
            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(main(arguments))
