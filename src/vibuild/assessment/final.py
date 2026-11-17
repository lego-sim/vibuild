"""Explicit final visual audit on the current scene, including reserved views."""

from pathlib import Path
import time

from . import visual
from vibuild.resources import load, save, digest
from vibuild.runtime.scene import file_lock


def audit(root, config):
    root = Path(root).resolve()
    plan = load(root / "plan.json")
    if any(s.get("state") != "accepted" for s in plan["steps"]):
        raise ValueError(
            "All region/connection stages must be accepted before the final audit"
        )
    results = {}
    snapshot = digest(root / "scene.blend")
    ident = str(time.time_ns())
    with file_lock(root / ".runtime/main/main.lock"):
        for label in ("fit", "held_out"):
            output = root / "GLOBAL/scores" / f"{ident}_{label}"
            arguments = [
                "prepare",
                "--record",
                str(root / "inputs"),
                "--partition",
                str(root / "partition.json"),
                "--scene",
                str(root / "scene.blend"),
                "--output",
                str(output),
                "--config",
                str(config),
            ]
            if label == "held_out":
                arguments += ["--held-out"]
            if visual.main(arguments):
                raise ValueError(f"Final {label} evidence preparation failed")
            if visual.main(["run", "--output", str(output), "--config", str(config)]):
                raise ValueError(f"Final {label} assessment failed; see {output}")
            summary = load(output / "scores.json")
            results[label] = summary
        from vibuild.registration.connections import prepare, write_packet

        partition = load(root / "partition.json")
        for interface in partition["interfaces"]:
            if interface["status"] != "observed_transition":
                continue
            packet = prepare(partition, interface["regions"], "J01")
            evidence = root / "GLOBAL/evidence" / f"{ident}_{interface['id']}"
            write_packet(packet, evidence, root / "partition.json")
            output = root / "GLOBAL/scores" / f"{ident}_{interface['id']}"
            arguments = [
                "prepare",
                "--record",
                str(root / "inputs"),
                "--partition",
                str(root / "partition.json"),
                "--scene",
                str(root / "scene.blend"),
                "--output",
                str(output),
                "--region",
                "J01",
                "--connection",
                str(evidence / "connection.json"),
                "--config",
                str(config),
            ]
            if visual.main(arguments) or visual.main(
                ["run", "--output", str(output), "--config", str(config)]
            ):
                raise ValueError(f"Final connection assessment failed; see {output}")
            results[interface["id"]] = load(output / "scores.json")
        if digest(root / "scene.blend") != snapshot:
            raise ValueError("Scene changed during final audit")
        passed = all(result.get("passed") for result in results.values())
        state = load(root / "status.json")
        state.update(
            global_acceptance="passed" if passed else "failed",
            state="completed" if passed else "audit_failed",
        )
        save(root / "status.json", state)
        report = root / "audit.json"
        save(report, {"scene_sha256": snapshot, "passed": passed, "results": results})
    return {"passed": passed, "report": str(report)}
