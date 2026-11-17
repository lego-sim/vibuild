"""One local edit, a few visual comparisons, and one compact deliverable."""

import argparse
import contextlib
import html
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vibuild.runtime.cache import digest
import vibuild.blender.scene as scene
from vibuild.runtime.scene import task_root, scene_lock, commit


def preview_frames(path, selected=None):
    """Read only selected RGB calibration; no masks, scoring, or depth loads for RGB views."""
    value = json.loads(path.read_text())
    base = path.resolve().parent
    frames = value["frames"]
    if selected:
        frames = [f for f in frames if str(f["id"]) in set(selected)]
    else:
        frames = frames[:3]
    result = []
    for f in frames:
        frame = {k: f[k] for k in ("id", "camera", "blender_frame") if k in f}
        if f.get("reference_rgb"):
            reference = (base / f["reference_rgb"]).resolve()
            if reference.suffix == ".npy":
                shape = np.load(reference, mmap_mode="r", allow_pickle=False).shape
                height, width = shape[:2]
            else:
                with Image.open(reference) as image:
                    width, height = image.size
            calibration = f.get("appearance_calibration", {}).get("reference", {})
            if "K" not in calibration or "camera_to_world" not in calibration:
                raise ValueError("RGB preview requires explicit reference calibration")
            frame.update(
                width=width,
                height=height,
                K=calibration["K"],
                camera_to_world=calibration["camera_to_world"],
                reference_rgb=str(reference),
            )
        else:
            # Geometry-only evidence can supply calibration without evaluating its arrays.
            with np.load(base / f["reference"], allow_pickle=False) as ref:
                height, width = ref["depth"].shape
                frame.update(
                    width=width,
                    height=height,
                    K=ref["K"].tolist(),
                    camera_to_world=ref["camera_to_world"].tolist(),
                )
        result.append(frame)
    return result


def difference_image(before, after):
    """Fixed-scale absolute RGB difference, independent of each image's extrema."""
    a = np.asarray(before, dtype=np.int16)
    b = np.asarray(after, dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError("Before/after preview grids differ")
    delta = np.max(np.abs(a - b), axis=2)
    # Ignore <=2 levels of rounding/noise. Fixed anchors make views comparable.
    anchors = [0, 2, 3, 16, 32, 64, 255]
    colors = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [20, 50, 180],
            [135, 30, 200],
            [235, 50, 25],
            [255, 230, 60],
            [255, 255, 255],
        ]
    )
    heat = np.stack(
        [np.interp(delta, anchors, colors[:, i]) for i in range(3)], axis=-1
    ).astype(np.uint8)
    return Image.fromarray(heat), int((delta > 2).sum())


def cards(frames, before, after, output, what, why, region="MAIN"):
    outputs = []
    changed = 0
    for index, f in enumerate(frames, 1):
        fid = f["id"]
        images = []
        labels = []
        reference = f.get("reference_rgb")
        if reference and Path(reference).is_file():
            if Path(reference).suffix == ".npy":
                a = np.load(reference, allow_pickle=False)
                if a.dtype.kind == "f":
                    a = np.uint8(np.clip(a, 0, 1) * 255)
                image = Image.fromarray(a).convert("RGB")
            else:
                image = Image.open(reference).convert("RGB")
            images.append(image)
            labels.append("Reference")
        a = Image.open(before / f"{fid}.png").convert("RGB")
        b = Image.open(after / f"{fid}.png").convert("RGB")
        if a.size != b.size:
            raise ValueError("Before/after preview grids differ")
        difference, count = difference_image(a, b)
        changed += count
        images.extend([a, b, difference])
        labels.extend(
            ["Before", "After", "RGB difference | blue: low, yellow/white: high"]
        )
        width = 640
        height = round(a.height * width / a.width)
        board = Image.new("RGB", (width * len(images), height + 98), "#f4f4f4")
        draw = ImageDraw.Draw(board)
        draw.rectangle((0, 0, board.width, 43), fill="#163a5f")
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        draw.text((14, 8), f"REGION {region}  |  VIEW {fid}", font=font, fill="white")
        for i, (image, label) in enumerate(zip(images, labels)):
            board.paste(
                image.resize((width, height), Image.Resampling.LANCZOS), (i * width, 88)
            )
            draw.text((i * width + 12, 58), f"{label} | view {fid}", fill="black")
        path = output / f"comparison_{index:02}.jpg"
        board.save(path, quality=90)
        outputs.append(path)
    return outputs, changed


def publish_html(
    output,
    images,
    what,
    why,
    notes="",
    scene_path="scene.blend",
    scene_label="Open this iteration scene",
):
    body = "".join(
        f'<figure><a href="{html.escape(p.name, quote=True)}"><img src="{html.escape(p.name, quote=True)}" alt="Reference, before, after and RGB difference"></a><figcaption>View {i} · Open full resolution</figcaption></figure>'
        for i, p in enumerate(images, 1)
    )
    (output / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:17px/1.55 system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#222}h1{font-size:28px}img{width:100%}figure{margin:24px 0}figcaption,small{color:#666}a{color:#1463bd}</style>"""
        + f"<h1>{html.escape(what)}</h1><p><b>Why this changed: </b>{html.escape(why)}</p>"
        + "<p>The difference map compares before/after renders from the same camera: black means every RGB channel differs by ≤2; blue is a small change, red moderate, yellow/white large (fixed 0–255 scale). Lighting and reflection changes also appear; color does not indicate improvement or regression.</p>"
        + body
        + f'<p>{html.escape(notes)}</p><p><a href="{html.escape(scene_path, quote=True)}">{html.escape(scene_label)}</a></p><small>Local visual iteration; complete-scene acceptance has not been performed.</small>',
        encoding="utf-8",
    )


def main(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--baseline", type=Path)
    probe.add_argument("--output", type=Path)
    known, _ = probe.parse_known_args(argv)
    root = task_root(known.output) if known.output else None
    if root is None and known.baseline:
        root = task_root(known.baseline)
    with scene_lock(root, writing=True):
        try:
            return iterate(argv, root)
        finally:
            if root and known.output and known.output.resolve().is_relative_to(root):
                for name in (
                    "candidate.blend",
                    ".candidate.pending.blend",
                    ".candidate.pending.blend1",
                ):
                    (known.output / ".runtime/build" / name).unlink(missing_ok=True)


def iterate(argv=None, root=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument(
        "--script",
        type=Path,
        required=True,
        help="bounded bpy edit; the tool saves atomically",
    )
    views = p.add_mutually_exclusive_group(required=True)
    views.add_argument("--views", type=Path, help="calibrated frame manifest")
    views.add_argument(
        "--protocol",
        type=Path,
        help="reuse selected views from an evaluation protocol, without scoring",
    )
    p.add_argument(
        "--frame",
        action="append",
        help="choose up to three views that show the edit and its boundary",
    )
    p.add_argument(
        "--region", help="Region ID; sessions inherit TASK.unit automatically"
    )
    p.add_argument("--what", required=True)
    p.add_argument("--why", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blender", default=None)
    p.add_argument(
        "--cache-dir", type=Path, default=os.environ.get("VIBUILD_CACHE_DIR")
    )
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--width", type=int, default=960)
    args = p.parse_args(argv)
    if args.width < 16 or args.samples < 1:
        p.error("positive samples and width >= 16 required")
    task = os.environ.get("VIBUILD_TASK")
    task_value = json.loads(Path(task).read_text()) if task else None
    if task_value and task_value.get("mode") in ("score", "audit"):
        p.error(
            "Scoring/audit is read-only; return evidence instead of creating an edited candidate"
        )
    if task_value and args.region and args.region != task_value["unit"]:
        p.error("region must match TASK.unit")
    region = task_value["unit"] if task_value else (args.region or "MAIN")
    if region != "MAIN" and (
        region[:1] not in ("R", "S", "L", "J") or not region[1:].isdigit()
    ):
        p.error("region must be an assigned R, S, L or J number")
    started = time.monotonic()
    out = args.output.resolve()
    if out.exists() and any(f.name != ".runtime" for f in out.iterdir()):
        p.error("output must be empty except for runner .runtime files")
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / ".runtime"
    runtime.mkdir(exist_ok=True)
    if (runtime / "views.json").exists():
        p.error("iteration already started here; use a new versioned output")
    baseline = args.baseline.resolve()
    baseline_sha = digest(baseline)
    if root:
        if baseline != root / "scene.blend":
            raise ValueError(
                "Use the latest task-root scene.blend; per-iteration scene paths are retired"
            )
        if not out.is_relative_to(root) or out == root or not args.script:
            raise ValueError(
                "Managed iterations require an evidence subdirectory and --script; no candidate copies or merges"
            )
    if task_value and task_value.get("baseline", {}).get("sha256") not in (
        None,
        baseline_sha,
    ):
        raise ValueError("Worker iteration must use its assigned baseline")
    if args.protocol:
        frames = preview_frames(args.protocol, args.frame)
    else:
        manifest = json.loads(args.views.read_text())
        frames = manifest["frames"]
        for f in frames:
            if "reference_rgb" in f:
                f["reference_rgb"] = str(
                    (args.views.resolve().parent / f["reference_rgb"]).resolve()
                )
    ids = [str(f["id"]) for f in frames]
    if len(set(ids)) != len(ids) or any(
        Path(fid).name != fid or fid in ("", ".", "..") for fid in ids
    ):
        raise ValueError("Preview frame IDs must be unique safe file names")
    if args.frame:
        wanted = set(args.frame)
        available = {str(f["id"]) for f in frames}
        if not wanted <= available:
            raise ValueError(
                "Requested preview view has no calibrated image or geometry grid"
            )
        frames = [f for f in frames if str(f["id"]) in wanted]
        if len(frames) > 3:
            raise ValueError(
                "Local iterations use at most three views; use an audit for wider coverage"
            )
    else:
        frames = frames[:3]
    if not frames:
        raise ValueError("No calibrated preview views")
    for f in frames:
        width = min(args.width, int(f["width"]))
        height = max(1, round(int(f["height"]) * width / int(f["width"])))
        sx, sy = width / int(f["width"]), height / int(f["height"])
        A = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.0]])
        f.update(
            id=str(f["id"]),
            K=(A @ np.asarray(f["K"])).tolist(),
            width=width,
            height=height,
            geometry=False,
            appearance=True,
        )
    scene.dump(runtime / "views.json", {"frames": frames})
    candidate = baseline if root else out / "scene.blend"
    with (
        (runtime / "tools.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        stats = {}

        def render(role, blend):
            options = [
                "render",
                "--blend",
                str(blend),
                "--manifest",
                str(runtime / "views.json"),
                "--output",
                str(runtime / role),
                "--blender",
                args.blender,
                "--denoise",
                "--samples",
                str(args.samples),
                "--threads",
                str(args.threads),
            ]
            if args.cache_dir:
                options += ["--cache-dir", str(args.cache_dir)]
            if scene.main(options):
                raise RuntimeError(f"{role} preview failed")
            stats[role] = json.loads(
                (runtime / role / "render_manifest.json").read_text()
            )

        if root:
            render("before", baseline)
        options = [
            "patch",
            "--blend",
            str(baseline),
            "--output",
            str(runtime / "build"),
            "--blender",
            args.blender,
            "--script",
            str(args.script.resolve()),
        ]
        if root:
            options += ["--_single-scene-edit"]
        if scene.main(options):
            raise RuntimeError("Edit failed; no preview published")
        if root:
            commit(
                root,
                runtime / "build/candidate.blend",
                baseline_sha,
                region,
                int(out.name) if out.name.isdigit() else out.name,
            )
        else:
            (runtime / "build/candidate.blend").rename(candidate)
        build = json.loads((runtime / "build/patch.json").read_text())
        build["candidate"] = str(candidate)
        scene.dump(runtime / "build/patch.json", build)
        for role, blend in [("before", baseline), ("after", candidate)]:
            if role not in stats:
                render(role, blend)
    if (
        stats["before"]["camera_signature"] != stats["after"]["camera_signature"]
        or stats["before"]["scene_unit_scale"] != stats["after"]["scene_unit_scale"]
    ):
        raise ValueError("Candidate changed registered cameras or scene scale")
    images, changed = cards(
        frames, runtime / "before", runtime / "after", out, args.what, args.why, region
    )
    publish_html(
        out,
        images,
        args.what,
        args.why,
        args.notes,
        scene_path=os.path.relpath(candidate, out),
        scene_label="Open current global scene (may be newer than these images)"
        if root
        else "Open scene",
    )
    if not root and digest(baseline) != baseline_sha:
        raise ValueError("Source scene changed during iteration")
    receipt = {
        "schema_version": 1,
        "kind": "visual_iteration",
        "baseline": str(baseline),
        "baseline_sha256": baseline_sha,
        "candidate": str(candidate),
        "candidate_sha256": digest(candidate),
        "what": args.what,
        "why": args.why,
        "frames": [f["id"] for f in frames],
        "changed_pixels": changed,
        "cameras_unchanged": True,
        "scale_unchanged": True,
        "finding_ids": task_value.get("finding_ids", []) if task_value else [],
        "region": region,
        "comparison_format": "reference_before_after_difference_v1",
        "image_regions": {str(f): region for f in images},
        "preview_settings": {
            "samples": args.samples,
            "threads": args.threads,
            "width": args.width,
        },
        "artifacts": {str(f): digest(f) for f in [*images, out / "review.html"]},
        "elapsed_seconds": time.monotonic() - started,
        "overall_acceptance": "not_assessed",
        "timings": {role: stat["request_seconds"] for role, stat in stats.items()},
    }
    scene.dump(runtime / "iteration.json", receipt)
    # Only delivery artifacts stay in this round. Reusable raw previews belong to the cache.
    shutil.rmtree(runtime / "before")
    shutil.rmtree(runtime / "after")
    print(
        json.dumps(
            {
                "review": str(out / "review.html"),
                "scene": str(candidate),
                "receipt": str(runtime / "iteration.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0
