"""Video-guided Blender modeling."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import default_config, settings, setup, tool


def parser():
    root = argparse.ArgumentParser(prog="vibuild", description=__doc__)
    from . import __version__

    root.add_argument("--version", action="version", version=__version__)
    root.add_argument("--config", type=Path)
    commands = root.add_subparsers(dest="command", required=True)

    def command(name, help, project=False):
        p = commands.add_parser(name, help=help)
        p.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        if project:
            p.add_argument("project", type=Path)
        return p

    p = command("setup", "create or update local tool configuration")
    for name in ("blender", "codex", "ffmpeg", "model", "vidmap-python"):
        p.add_argument("--" + name)
    p = command("doctor", "check requirements and print repairs")
    p.add_argument("--stage", choices=("model", "register"), default="model")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--json", action="store_true")
    p = command("install", "explicitly install an external tool")
    p.add_argument("component", choices=("blender", "ffmpeg", "vidmap"))
    p.add_argument("--python", type=Path)
    p.add_argument("--destination", type=Path)
    p = command("init", "create a project", True)
    p.add_argument("--videos", type=Path)
    p = command("register", "register source video or import COLMAP cameras", True)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--fps", type=float, default=2)
    p.add_argument("--from", dest="reconstruction", type=Path)
    p.add_argument("--images", type=Path)
    p.add_argument("--sources", type=Path)
    p = command(
        "partition",
        "review spatial scope and create the registered Blender scene",
        True,
    )
    p.add_argument("--spec", type=Path)
    p.add_argument("--reviewed", action="store_true")
    p = command("run", "start modeling from current project state", True)
    p.add_argument("--proxy")
    p = command("status", "show project and stage progress", True)
    p.add_argument("--json", action="store_true")
    command("pause", "stop execution and prevent further writes", True)
    p = command("resume", "resume modeling", True)
    p.add_argument("--no-run", action="store_true")
    p = command(
        "stage", "record a review, acceptance, deferral, or owner handoff", True
    )
    p.add_argument("action", choices=("review", "accept", "defer", "handoff"))
    p.add_argument("--target")
    p.add_argument("--return-to")
    p.add_argument("--score", type=Path)
    p.add_argument("--reason")
    p.add_argument("--view", action="append", default=[])
    p.add_argument("--begin", action="store_true")
    p.add_argument("--finish", action="store_true")
    p.add_argument("--corrected", action="store_true")
    p = command("audit", "assess the final scene on fit and held-out views", True)
    from .tools import MODULES

    p = command("tools", "advanced scene, evidence, and execution tools")
    p.add_argument("name", choices=tuple(MODULES))
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    from . import project

    config = args.config
    if config is None:
        config = (
            (args.project / "vibuild.json")
            if hasattr(args, "project") and args.command != "init"
            else default_config()
        )
    config = Path(config).absolute()
    try:
        if args.command == "setup":
            path = config if args.config else Path.cwd() / "vibuild.local.json"
            result = setup(
                path,
                **{
                    k: getattr(args, k)
                    for k in ("blender", "codex", "ffmpeg", "model", "vidmap_python")
                },
            )
        elif args.command == "doctor":
            from .install import doctor

            result = doctor(config, args.stage, args.deep)
        elif args.command == "install":
            from .install import install

            result = install(
                args.component,
                config if args.config else Path.cwd() / "vibuild.local.json",
                args.python,
                args.destination,
            )
        elif args.command == "init":
            result = project.initialize(args.project, args.videos, config)
        elif args.command == "register":
            result = project.register(
                args.project,
                config,
                args.device,
                args.fps,
                args.reconstruction,
                args.images,
                args.sources,
            )
        elif args.command == "partition":
            result = project.partition(args.project, args.spec, args.reviewed, config)
        elif args.command == "status":
            result = project.status(args.project)
        elif args.command == "pause":
            result = project.pause(args.project)
        elif args.command == "stage":
            from .runtime.stages import transition

            result = transition(
                args.project,
                args.action,
                args.target,
                args.score,
                args.reason,
                args.view,
                args.begin,
                args.finish,
                args.corrected,
                args.return_to,
            )
        elif args.command == "tools":
            from .tools import invoke

            previous = os.environ.get("VIBUILD_CONFIG")
            os.environ["VIBUILD_CONFIG"] = str(config)
            try:
                return invoke(args.name, args.arguments, config)
            finally:
                if previous is None:
                    os.environ.pop("VIBUILD_CONFIG", None)
                else:
                    os.environ["VIBUILD_CONFIG"] = previous
        elif args.command == "audit":
            from .assessment.final import audit

            result = audit(args.project, config)
        else:
            if args.command == "resume":
                result = project.resume(args.project)
                if args.no_run:
                    print(json.dumps(result))
                    return 0
            state = project.status(args.project)
            if state["paused"]:
                raise ValueError("Project is paused. Use: vibuild resume PROJECT")
            if not (args.project / "plan.json").is_file():
                raise ValueError(
                    "Run vibuild partition PROJECT --spec FILE --reviewed first"
                )
            if not state["active_region"]:
                raise ValueError(
                    "Local stages are complete. Use: vibuild audit PROJECT"
                )
            for name in ("blender", "codex"):
                tool(config, name)
            if not settings(config)["session"].get("model"):
                raise ValueError("Choose a model with vibuild setup --model MODEL")
            from .runtime.service import run

            result = run(args.project.resolve(), config, getattr(args, "proxy", None))
        if args.command == "doctor" and not args.json:
            for check in result["checks"]:
                detail = check.get("reason", check.get("detail", ""))
                print(f"{check['status'].upper():7} {check['name']}: {detail}")
                if check.get("fix"):
                    print(f"        Fix: {check['fix']}")
        elif args.command == "status" and not args.json:
            print(
                f"{result['project']}\nState: {result['state']}  Paused: {result['paused']}  Active: {result['active_region'] or '-'}"
            )
            if result["scene"]:
                print(f"Scene: {result['scene']}")
            for stage in result["stages"]:
                print(f"{stage['id']:6} {stage['state']}")
        else:
            print(json.dumps(result, separators=(",", ":")))
        return int(
            result.get("ok") is False
            or args.command == "audit"
            and not result["passed"]
            or args.command in ("run", "resume")
            and result.get("state") in ("blocked", "failed")
        )
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f"vibuild: {exc}", file=sys.stderr)
        return 1
