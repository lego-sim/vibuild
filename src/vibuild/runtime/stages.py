"""Validated stage transitions; modeling decisions do not edit workflow JSON."""

from pathlib import Path
import time

from .monitor import read, write, state_lock
from .scene import active_assessments
from .cache import digest


def transition(
    root,
    action,
    target=None,
    score=None,
    reason=None,
    views=(),
    begin=False,
    finish=False,
    corrected=False,
    return_to=None,
):
    root = Path(root).resolve()
    with state_lock(root):
        state = read(root / "status.json", {})
        plan = read(root / "plan.json", {})
        if state.get("execution_paused") or (root / ".runtime/main/STOP").exists():
            raise ValueError("Project is paused")
        active = state.get("active_region")
        step = next((s for s in plan.get("steps", []) if s["id"] == active), None)
        if step is None:
            raise ValueError("No active stage")
        if list(active_assessments(root)):
            raise ValueError(
                "Finish or cancel the active assessment before a stage transition"
            )
        from vibuild.assessment.scoring import RUBRIC

        budget = RUBRIC["closeout"]
        if action == "handoff":
            owner = next((s for s in plan["steps"] if s["id"] == target), None)
            if owner is None or owner.get("state") == "deferred" or target == active:
                raise ValueError("Choose another non-deferred owner stage")
            if not reason:
                raise ValueError("Handoff requires --reason")
            step["blocked_by"] = target
            owner.update(
                work_objective=reason,
                resume_region=return_to or active,
                state="planned",
            )
            state["active_region"] = target
        elif action == "review":
            review = step.get("closeout")
            if begin or not finish:
                if review:
                    raise ValueError(
                        "This stage already used its closeout inspection; finish that review"
                    )
                if not score:
                    raise ValueError(
                        "Review requires --score DIRECTORY with a passing current assessment"
                    )
                from vibuild.assessment.visual import verified_score

                path = Path(score).resolve()
                if not path.is_relative_to(root / active / "scores"):
                    raise ValueError("Use the active stage scoring directory")
                protocol = read(path / ".runtime/protocol.json", {})
                summary = verified_score(path / active, protocol)
                if summary["status"] != "passed":
                    raise ValueError("Visual gate has not passed")
                review = {
                    "score": str(path),
                    "scene_sha256": digest(root / "scene.blend"),
                    "started": time.time(),
                    "last_iteration": max(
                        (p.name for p in (root / active / "iterations").iterdir()),
                        default="",
                    ),
                    "state": "inspecting",
                    "max_reviews": budget["max_reviews"],
                    "max_corrections": budget["max_corrections"],
                }
                step["closeout"] = review
            if finish or not begin:
                if not review or review["state"] != "inspecting":
                    raise ValueError(
                        "Start the closeout inspection before finishing it"
                    )
                if not 1 <= len(set(views)) <= budget["max_views"]:
                    raise ValueError("Review requires one or two existing --view IDs")
                protocol = read(Path(review["score"]) / ".runtime/protocol.json", {})
                known = {
                    v["id"]
                    for s in protocol["regions"]
                    if s["region"] == active
                    for v in s["views"]
                }
                if not set(views) <= known:
                    raise ValueError("Review views must be from the passing assessment")
                current = digest(root / "scene.blend")
                if current != review["scene_sha256"]:
                    revisions = [
                        p
                        for p in (root / active / "iterations").glob(
                            "*/.runtime/iteration.json"
                        )
                        if p.parents[1].name > review["last_iteration"]
                    ]
                    if not corrected or len(revisions) != 1:
                        raise ValueError(
                            "Closeout permits exactly one recorded correction; further work needs explicit deferral"
                        )
                elif corrected:
                    raise ValueError("No scene correction was recorded")
                review.update(
                    state="reviewed",
                    views=list(views),
                    corrections=int(corrected),
                    reviews=1,
                    current_scene_sha256=current,
                    note=reason or "",
                )
        elif action in ("accept", "defer"):
            if action == "accept":
                pending = [
                    dep
                    for dep in step.get("acceptance_dependencies", [])
                    if next(s for s in plan["steps"] if s["id"] == dep).get("state")
                    != "accepted"
                ]
                if pending:
                    raise ValueError(
                        "Acceptance dependencies unresolved: " + ", ".join(pending)
                    )
                review = step.get("closeout", {})
                if review.get("state") != "reviewed" or review.get(
                    "current_scene_sha256"
                ) != digest(root / "scene.blend"):
                    raise ValueError(
                        "Complete the bounded visual review against the current scene before acceptance"
                    )
                step.update(
                    state="accepted", accepted_scene_sha256=digest(root / "scene.blend")
                )
            else:
                if not reason:
                    raise ValueError("Deferral requires --reason")
                step.update(state="deferred", deferred_reason=reason)
            step.pop("blocked_by", None)
            next_id = step.pop("resume_region", None)
            if next_id is None:
                terminal = {"accepted", "deferred"}
                next_id = next(
                    (
                        s["id"]
                        for s in plan["steps"]
                        if s.get("state") not in terminal
                        and all(
                            next(x for x in plan["steps"] if x["id"] == dependency).get(
                                "state"
                            )
                            in terminal
                            for dependency in s.get("depends_on", [])
                        )
                    ),
                    None,
                )
            state.update(
                active_region=next_id,
                state="ready" if next_id else "awaiting_global_audit",
                global_acceptance="not_assessed",
            )
        else:
            raise ValueError("Unknown stage action")
        write(root / "plan.json", plan)
        write(root / "status.json", state)
    return {
        "stage": active,
        "action": action,
        "next_stage": state.get("active_region"),
        "state": state.get("state"),
    }
