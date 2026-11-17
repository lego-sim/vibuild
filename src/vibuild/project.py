"""The project lifecycle; source inputs freeze before the single-scene run starts."""

from pathlib import Path
import subprocess

from .config import default_config, settings, tool
from .paths import PACKAGE
from .resources import load, save, copy, digest
from .runtime.layout import initialize_task
from .runtime.scene import file_lock
from .runtime.monitor import state_lock


def initialize(root, videos=None, config=None):
    root = Path(root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty project directory")
    selected = settings(config or default_config())
    root.mkdir(parents=True, exist_ok=True)
    save(root / "vibuild.json", selected)
    destination = root / "inputs/resources/input/videos"
    destination.mkdir(parents=True)
    files = []
    if videos:
        path = Path(videos).expanduser().resolve()
        files = (
            [path]
            if path.is_file()
            else sorted(
                p
                for p in path.iterdir()
                if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")
            )
        )
        if not files:
            raise ValueError("No videos found")
        for path in files:
            copy(path, destination / path.name)
    save(
        root / "inputs/record.json",
        {"version": 1, "id": root.name, "resources": "resources/index.json"},
    )
    save(
        root / "project.json",
        {
            "version": 2,
            "name": root.name,
            "state": "new",
            "inputs_frozen": False,
            "videos": [str((destination / p.name).relative_to(root)) for p in files],
        },
    )
    return {"project": str(root), "next": f"vibuild register {root} --device cuda"}


def configuration(root, override=None):
    return Path(override or Path(root) / "vibuild.json").resolve()


def register(
    root,
    config=None,
    device=None,
    fps=2,
    reconstruction=None,
    images=None,
    sources=None,
):
    from .registration.prepare import import_reconstruction

    root = Path(root).resolve()
    meta = load(root / "project.json")
    if meta["inputs_frozen"]:
        raise ValueError(
            "Project inputs are frozen; use another project for new registration"
        )
    if reconstruction:
        if not images:
            raise ValueError("--images is required when importing a reconstruction")
        return import_reconstruction(
            root, reconstruction, images, load(sources) if sources else None
        )
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose --device cpu or --device cuda")
    if not meta["videos"]:
        raise ValueError(
            "No source videos; import an existing reconstruction with --from and --images"
        )
    from .registration.capture import prepare
    from .registration.vidmap import execute

    capture, frames = prepare(root, configuration(root, config), fps)
    output = root / ".runtime/vidmap"
    execute(configuration(root, config), "run", capture, output, device)
    return import_reconstruction(root, output / "rec", capture, frames)


def partition(root, spec=None, reviewed=False, config=None):
    from .registration.prepare import prepare_partition

    root = Path(root).resolve()
    result = prepare_partition(root, spec, reviewed)
    if not reviewed:
        return result
    if (root / "scene.blend").exists():
        raise ValueError("The project already has a scene")
    frames = result.pop("frames")
    execution = result.pop("execution_plan")
    runtime = root / ".runtime"
    runtime.mkdir(exist_ok=True)
    camera_file = runtime / "cameras.json"
    save(camera_file, frames)
    pending = runtime / "scene.pending.blend"
    log = runtime / "initialize.log"
    command = [
        tool(configuration(root, config), "blender"),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(PACKAGE / "blender/initialize.py"),
        "--",
        str(camera_file),
        str(pending),
    ]
    with log.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise ValueError(f"Blender initialization failed; see {log}")
    with pending.open("rb") as stream:
        if stream.read(7) != b"BLENDER":
            raise ValueError(f"Blender produced no complete scene; see {log}")
    pending.replace(root / "scene.blend")
    steps = [s for chain in execution["chains"] for s in chain["steps"]]
    initialize_task(root, steps)
    plan = load(root / "plan.json")
    plan.update(record=str(root / "inputs"), partition=str(root / "partition.json"))
    save(root / "plan.json", plan)
    region_map = {r["id"]: r for r in load(root / "partition.json")["regions"]}
    for step in plan["steps"]:
        directory = root / step["id"]
        save(directory / "stage.json", step)
        if step["id"] in region_map:
            save(directory / "region.json", {"region": region_map[step["id"]]})
            prepare = [
                "prepare",
                "--record",
                str(root / "inputs"),
                "--partition",
                str(root / "partition.json"),
                "--scene",
                str(root / "scene.blend"),
                "--region",
                step["id"],
                "--output",
                str(directory / "scores/001"),
                "--width",
                "640",
                "--samples",
                "8",
                "--threads",
                str(
                    settings(configuration(root, config))["rendering"][
                        "threads_per_worker"
                    ]
                ),
            ]
        else:
            from .registration.connections import (
                prepare as connection_prepare,
                write_packet,
            )

            data = load(root / "partition.json")
            packet = connection_prepare(data, step["regions"], step["id"])
            write_packet(packet, directory / "evidence/v001", root / "partition.json")
            prepare = [
                "prepare",
                "--record",
                str(root / "inputs"),
                "--partition",
                str(root / "partition.json"),
                "--scene",
                str(root / "scene.blend"),
                "--region",
                step["id"],
                "--connection",
                str(directory / "evidence/v001/connection.json"),
                "--output",
                str(directory / "scores/001"),
                "--width",
                "640",
                "--samples",
                "8",
                "--threads",
                "4",
            ]
        save(
            directory / "job_template.json",
            {"steps": [{"tool": "score", "args": prepare}]},
        )
    save(
        root / "status.json",
        {
            "state": "ready",
            "execution_paused": False,
            "active_region": plan["execution_order"][0],
            "global_acceptance": "not_assessed",
        },
    )
    meta = load(root / "project.json")
    meta.update(inputs_frozen=True, state="ready")
    save(root / "project.json", meta)
    # Input inventory is written once; modeling never refreshes it or changes its files.
    res = root / "inputs/resources"
    save(
        root / "inputs/inventory.json",
        {
            "files": [
                {
                    "path": str(p.relative_to(res)),
                    "bytes": p.stat().st_size,
                    "sha256": digest(p),
                }
                for p in sorted(res.rglob("*"))
                if p.is_file()
            ]
        },
    )
    result.update(scene=str(root / "scene.blend"), next=f"vibuild run {root}")
    return result


def status(root):
    root = Path(root).resolve()
    meta = load(root / "project.json")
    state = load(root / "status.json") if (root / "status.json").exists() else {}
    service = (
        load(root / ".runtime/main/state.json")
        if (root / ".runtime/main/state.json").exists()
        else {}
    )
    plan = load(root / "plan.json") if (root / "plan.json").exists() else {}
    return {
        "project": str(root),
        "state": state.get("state", meta["state"]),
        "paused": state.get("execution_paused", False)
        or (root / ".runtime/main/STOP").exists(),
        "active_region": state.get("active_region"),
        "service": service.get("state", "stopped"),
        "stages": [
            {k: s.get(k) for k in ("id", "state")} for s in plan.get("steps", [])
        ],
        "scene": str(root / "scene.blend") if (root / "scene.blend").exists() else None,
    }


def pause(root):
    root = Path(root).resolve()
    runtime = root / ".runtime/main"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "STOP").touch()
    with state_lock(root):
        state = load(root / "status.json") if (root / "status.json").exists() else {}
        state.update(execution_paused=True, state="paused")
        save(root / "status.json", state)
    return {"state": "paused", "project": str(root)}


def resume(root):
    root = Path(root).resolve()
    if not (root / "plan.json").is_file():
        raise ValueError("Complete the reviewed partition before resuming")
    with file_lock(root / ".runtime/main/main.lock"), state_lock(root):
        state = load(root / "status.json")
        state.update(execution_paused=False, state="ready")
        save(root / "status.json", state)
        (root / ".runtime/main/STOP").unlink(missing_ok=True)
    return {"state": "ready", "next": f"vibuild run {root}"}
