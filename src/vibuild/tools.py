"""The fixed tool catalogue used by the CLI and execution jobs."""

from importlib import import_module
from pathlib import Path
import sys

MODULES = {
    "vidmap": "vibuild.registration.vidmap",
    "connections": "vibuild.registration.connections",
    "perception": "vibuild.registration.perception",
    "scene": "vibuild.blender.scene",
    "score": "vibuild.assessment.visual",
    "audit": "vibuild.assessment.metrics",
    "jobs": "vibuild.runtime.jobs",
    "materials": "vibuild.agents.materials",
    "usage": "vibuild.runtime.usage",
    "monitor": "vibuild.runtime.monitor",
}


def invoke(name, arguments, config):
    from .config import tool

    args = list(arguments)
    if "--help" not in args:
        if name in ("vidmap", "score", "jobs", "perception") and "--config" not in args:
            args = ["--config", str(config), *args]
        if (
            name == "scene" or name == "audit" and args[:1] == ["compare"]
        ) and "--blender" not in args:
            args += ["--blender", tool(config, "blender")]
    module = import_module(MODULES[name])
    if name == "monitor":
        return module.main(Path(args[0]).resolve(), args[1:])
    previous = sys.argv[0]
    sys.argv[0] = f"vibuild tools {name}"
    try:
        return module.main(args) or 0
    finally:
        sys.argv[0] = previous
