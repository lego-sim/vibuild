"""Prepare one ordered capture from source videos, preserving decoded timestamps."""

from pathlib import Path
import re
import shutil
import subprocess

from vibuild.config import tool
from vibuild.resources import load, save


def prepare(project, config, fps=2, images=None):
    if fps <= 0:
        raise ValueError("Sampling fps must be positive")
    project = Path(project)
    meta = load(project / "project.json")
    output = project / ".runtime/capture"
    manifest = output / "sources.json"
    if manifest.exists():
        value = load(manifest)
        if value.get("images") != (str(Path(images).resolve()) if images else None):
            raise ValueError(
                "Capture source is fixed; use a new project for different images"
            )
        if value["fps"] != fps:
            raise ValueError(
                "Capture sampling is fixed; use a new project for a different fps"
            )
        return output, value["frames"]
    if images:
        from vibuild.resources import copy

        source = Path(images).resolve()
        files = sorted(
            (
                p
                for p in source.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ),
            key=lambda p: [
                int(part) if part.isdigit() else part
                for part in re.split(r"(\d+)", p.name)
            ],
        )
        if len(files) < 2:
            raise ValueError("--images requires at least two ordered images")
        frames = []
        for index, path in enumerate(files):
            name = "cam0/" + path.name
            copy(path, output / name)
            frames.append({"name": name, "video": source.name, "time": index / fps})
        save(
            manifest,
            {
                "fps": fps,
                "frames": frames,
                "images": str(source),
                "timestamp_status": "sequence order divided by supplied fps",
            },
        )
        return output, frames
    images = output / "cam0"
    images.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, video in enumerate(meta["videos"]):
        source = project / video
        directory = output / f"clip_{i:02d}"
        directory.mkdir(exist_ok=True)
        log = directory / "extract.log"
        command = [
            tool(config, "ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{1 / fps:.12g})',showinfo",
            "-vsync",
            "vfr",
            str(directory / "%08d.jpg"),
        ]
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(f"Frame extraction failed; see {log}")
        times = re.findall(r"\bn:\s*\d+\s+pts:.*?\bpts_time:([^\s]+)", log.read_text())
        decoded = sorted(directory.glob("*.jpg"))
        if len(decoded) != len(times) or not decoded:
            raise ValueError(f"Frame timestamp mapping failed; see {log}")
        for original, stamp in zip(decoded, times):
            name = f"cam0/{len(frames) / fps:.9f}.jpg"
            target = output / name
            shutil.copyfile(original, target)
            frames.append(dict(name=name, video=source.stem, time=float(stamp)))
    if len(frames) < 2:
        raise ValueError("At least two sampled source frames are required")
    save(manifest, {"fps": fps, "frames": frames})
    return output, frames
