"""Explicit installers and stage-specific environment checks; never called by a model."""

import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from .config import settings, tool, vidmap_runtime, setup
from .resources import digest, save

VIDMAP_URL = "https://github.com/cvg/vidmap.git"
VIDMAP_REVISION = "effb06d5a518208a18a51667d91eda5dac22ac40"
BLENDER_VERSION = "4.5.13"
BLENDER_ARCHIVE = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL = f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}"
BLENDER_SHA256 = "da4e69b06b75b9e642d106496c50e7e240218b411d2f6e18271c1d1d819cef91"


def doctor(config, stage="model", deep=False):
    cfg = settings(config)
    checks = []

    def check(name, operation, fix):
        try:
            detail = operation()
            checks.append({"name": name, "status": "ok", "detail": detail})
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            checks.append(
                {"name": name, "status": "missing", "reason": str(exc), "fix": fix}
            )

    def probe(command, env=None):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=45 if deep else 15, env=env
        )
        if result.returncode:
            raise ValueError(
                (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"exit {result.returncode}"
                ).splitlines()[-1]
            )
        return result.stdout.strip()

    def codex():
        executable = tool(config, "codex")
        env = dict(os.environ)
        if cfg["session"].get("codex_home"):
            env["CODEX_HOME"] = cfg["session"]["codex_home"]
        help_text = probe([executable, "exec", "--help"], env)
        required = [
            "--json",
            "--ephemeral",
            "--image",
            "--output-schema",
            "--output-last-message",
            "--approve-for-me",
        ]
        missing = [flag for flag in required if flag not in help_text]
        if missing:
            raise ValueError("Unsupported Codex CLI flags: " + ", ".join(missing))
        probe([executable, "login", "status"], env)
        return probe([executable, "--version"], env).splitlines()[0]

    if stage == "model":
        check(
            "blender",
            lambda: probe([tool(config, "blender"), "--version"]).splitlines()[0],
            f"vibuild install blender --config {config}",
        )
        check(
            "codex", codex, "Install Codex CLI, run codex login, and set runtime.codex"
        )
        check(
            "model",
            lambda: (
                cfg["session"]["model"]
                or (_ for _ in ()).throw(ValueError("No model selected"))
            ),
            f"vibuild setup --config {config} --model MODEL",
        )
        if deep:
            check(
                "blender_python",
                lambda: probe(
                    [
                        tool(config, "blender"),
                        "--background",
                        "--factory-startup",
                        "--python-expr",
                        'import bpy,numpy; print("Blender Python ready")',
                    ]
                ),
                f"vibuild install blender --config {config}",
            )
    else:
        check(
            "ffmpeg",
            lambda: probe([tool(config, "ffmpeg"), "-version"]).splitlines()[0],
            f"vibuild install ffmpeg --config {config}",
        )
        check(
            "vidmap",
            lambda: vidmap_runtime(config),
            f"vibuild install vidmap --config {config} --python PYTHON",
        )
        if deep:
            check(
                "vidmap_native",
                lambda: probe(
                    [
                        vidmap_runtime(config)["python"],
                        "-c",
                        "import vidmap,vidmap_native,torch,pycolmap;print(vidmap.__file__)",
                    ]
                ),
                "Install VidMap native dependencies in its configured environment; see the upstream setup guide",
            )
    return {
        "ok": all(c["status"] == "ok" for c in checks),
        "stage": stage,
        "checks": checks,
    }


def install(component, config, python=None, destination=None):
    config = Path(config).absolute()
    if not config.exists():
        setup(config)
    cfg = settings(config)
    destination = Path(destination or config.parent / ".vibuild/tools").absolute()
    destination.mkdir(parents=True, exist_ok=True)
    log = destination / f"{component}.log"

    def run(command):
        with log.open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(f"Install failed: {component}; see {log}")

    if component == "blender":
        if platform.system() != "Linux" or platform.machine() not in (
            "x86_64",
            "amd64",
        ):
            raise ValueError(
                "Automatic Blender install supports Linux x86_64; configure another installed Blender explicitly"
            )
        binary = destination / BLENDER_ARCHIVE.removesuffix(".tar.xz") / "blender"
        if not binary.is_file():
            with tempfile.TemporaryDirectory(dir=destination) as directory:
                archive = Path(directory) / BLENDER_ARCHIVE
                urllib.request.urlretrieve(BLENDER_URL, archive)
                if digest(archive) != BLENDER_SHA256:
                    raise ValueError(
                        "Blender download checksum differs from the pinned release"
                    )
                with tarfile.open(archive) as tar:
                    for member in tar.getmembers():
                        target = (destination / member.name).resolve()
                        if not target.is_relative_to(destination):
                            raise ValueError("Unsafe archive member")
                        if member.issym() or member.islnk():
                            linked = (
                                (target.parent / member.linkname).resolve()
                                if member.issym()
                                else (destination / member.linkname).resolve()
                            )
                            if not linked.is_relative_to(destination):
                                raise ValueError("Unsafe archive link")
                        if member.isdev():
                            raise ValueError("Unexpected device in archive")
                    tar.extractall(destination)
        cfg["runtime"]["blender"] = str(binary)
    elif component == "ffmpeg":
        run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg==0.6.0"])
        result = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())",
            ],
            text=True,
        )
        cfg["runtime"]["ffmpeg"] = result.strip()
    else:
        if not python:
            raise ValueError(
                "Pass --python PYTHON for a separate environment with PyTorch and COLMAP installed; see https://github.com/cvg/vidmap#setup"
            )
        source = Path(cfg["runtime"]["vidmap"]["source"])
        if not source.exists():
            run(["git", "clone", "--no-checkout", VIDMAP_URL, str(source)])
            run(["git", "-C", str(source), "checkout", "--detach", VIDMAP_REVISION])
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != VIDMAP_REVISION:
            raise ValueError(
                "VidMap revision differs from the pinned release; use a clean pinned checkout"
            )
        run(["git", "-C", str(source), "submodule", "update", "--init", "--recursive"])
        run([str(python), "-c", "import torch,pycolmap"])
        run([str(python), "-m", "pip", "install", "-e", str(source)])
        cfg["runtime"]["vidmap"] = {
            "python": str(Path(python).absolute()),
            "source": str(source),
        }
    save(config, cfg)
    return {
        "component": component,
        "config": str(config),
        "next": f"vibuild doctor --config {config} --stage "
        + ("register" if component in ("vidmap", "ffmpeg") else "model"),
    }
