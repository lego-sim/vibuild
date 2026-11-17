"""Task execution details always live under .runtime."""

from pathlib import Path
import json
import re


def closeout_policy():
    """Planning limits; the orchestrator records usage in the existing step receipt."""
    from vibuild.assessment.scoring import RUBRIC

    return {
        **RUBRIC["closeout"],
        "budget_scope": "stage_across_restarts",
        "max_check_batches": 1,
        "full_rescore_for_seam_only": False,
        "reference": "docs/workflow.md",
    }


def stage_kind(ident):
    """The workflow has two stage types: regional work and connection work."""
    if not re.fullmatch(r"[RJ][0-9]{2,}", ident):
        raise ValueError(
            "Unknown stage ID; use Rxx for a region or Jxx for a connection"
        )
    return "region" if ident.startswith("R") else "connection"


def task_paths(root, region=None, round_number=None):
    """Paths for one evolving global scene and per-session evidence."""
    root = Path(root).resolve()
    result = {
        "root": root,
        "scene": root / "scene.blend",
        "runtime": root / ".runtime",
        "cache": root / ".runtime/cache",
        "plan": root / "plan.json",
        "status": root / "status.json",
        "progress": root / "progress.html",
    }
    if region is not None:
        stage_kind(region)
        child = root / region
        result.update(
            region=child,
            region_runtime=child / ".runtime",
            iterations=child / "iterations",
            scores=child / "scores",
        )
        if round_number is not None:
            if type(round_number) is not int or round_number < 1:
                raise ValueError("Round must be a positive integer")
            result.update(
                iteration=child / "iterations" / f"{round_number:03d}",
                score=child / "scores" / f"{round_number:03d}",
            )
    elif round_number is not None:
        raise ValueError("Round requires a region or connection stage")
    return result


def initialize_task(root, steps):
    """Create the dated layout and a plan, without launching or completing work."""
    paths = task_paths(root)
    if paths["plan"].exists():
        raise ValueError("Task plan already exists")
    steps = [dict(step) if isinstance(step, dict) else {"id": step} for step in steps]
    identifiers = [step["id"] for step in steps]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Task steps must be nonempty and unique")
    for step in steps:
        step["kind"] = stage_kind(step["id"])
    paths["cache"].mkdir(parents=True, exist_ok=True)
    for region in identifiers:
        child = task_paths(root, region)
        for key in ("region_runtime", "iterations", "scores"):
            child[key].mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "scene": "scene.blend",
        "scene_storage": "single_working_scene",
        "state": "planned",
        "closeout_policy": closeout_policy(),
        "global_acceptance": "not_assessed",
        "execution_order": identifiers,
        "steps": [
            {
                **step,
                "state": "planned",
                "directory": step["id"],
            }
            for step in steps
        ],
    }
    paths["plan"].write_text(json.dumps(plan, indent=2) + "\n")
    return paths


def runtime_file(work, name):
    return Path(work) / ".runtime" / name


def workspace_of(path):
    return Path(path).parent.parent


def session_files(root, name):
    yield from Path(root).rglob(".runtime/" + name)
