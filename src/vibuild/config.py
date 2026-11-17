"""Project configuration and explicit external runtimes."""

import json
import os
from pathlib import Path
import shutil

from .paths import PACKAGE

DEFAULT_CONFIG = PACKAGE / "defaults.json"


def default_config():
    if os.environ.get("VIBUILD_CONFIG"):
        return Path(os.environ["VIBUILD_CONFIG"])
    for path in (Path.cwd() / "vibuild.json", Path.cwd() / "vibuild.local.json"):
        if path.is_file():
            return path
    return DEFAULT_CONFIG


def settings(path=None):
    path = Path(path or default_config()).absolute()
    value = json.loads(DEFAULT_CONFIG.read_text())
    if path != DEFAULT_CONFIG:
        supplied = json.loads(path.read_text())
        for key, item in supplied.items():
            if isinstance(item, dict) and isinstance(value.get(key), dict):
                value[key].update(item)
            else:
                value[key] = item
    base = Path.cwd() if path == DEFAULT_CONFIG else path.parent
    runtime = value["runtime"]
    for name in ("blender", "codex", "ffmpeg"):
        command = runtime.get(name)
        if (
            isinstance(command, str)
            and command
            and ("/" in command or command.startswith("~"))
        ):
            runtime[name] = os.path.abspath(base / Path(command).expanduser())
    for name in ("vidmap", "da3", "sam31"):
        backend = runtime.get(name, {})
        for key in ("source", "python", "checkpoint"):
            command = backend.get(key)
            if (
                isinstance(command, str)
                and command
                and (key != "python" or "/" in command or command.startswith("~"))
            ):
                backend[key] = os.path.abspath(base / Path(command).expanduser())
    if value["session"].get("codex_home"):
        value["session"]["codex_home"] = os.path.abspath(
            base / Path(value["session"]["codex_home"]).expanduser()
        )
    return value


def tool(config, name):
    command = settings(config)["runtime"]
    for key in name.split("."):
        command = command.get(key) if isinstance(command, dict) else None
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"Configure runtime.{name} in {config}")
    if not shutil.which(command):
        raise ValueError(f"Not executable: runtime.{name}={command}")
    return command


def vidmap_runtime(config):
    runtime = settings(config)["runtime"]["vidmap"]
    source = runtime.get("source")
    if not source or not (Path(source) / "vidmap/run.py").is_file():
        raise ValueError(
            "VidMap source missing. Run: vibuild install vidmap --python PYTHON"
        )
    return {"source": source, "python": tool(config, "vidmap.python")}


def setup(path, **overrides):
    from .resources import save

    path = Path(path).absolute()
    config = settings(path) if path.exists() else json.loads(DEFAULT_CONFIG.read_text())
    for name in ("blender", "codex", "ffmpeg"):
        config["runtime"][name] = (
            overrides.get(name)
            or config["runtime"].get(name)
            or shutil.which(name)
            or ""
        )
    config["session"]["model"] = overrides.get("model") or config["session"].get(
        "model", ""
    )
    config["runtime"]["vidmap"]["source"] = os.path.abspath(
        path.parent
        / Path(config["runtime"]["vidmap"].get("source") or "third_party/vidmap")
    )
    if overrides.get("vidmap_python"):
        config["runtime"]["vidmap"]["python"] = overrides["vidmap_python"]
    save(path, config)
    return {
        "config": str(path),
        "next": f"vibuild doctor --config {path} --stage model",
    }
