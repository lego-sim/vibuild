"""Compact visual iteration receipts; full architectural acceptance remains separate."""

from pathlib import Path
from vibuild.resources import load, digest


def validate(path, root, expected_region=None):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Iteration receipt escapes its workspace/record")
    value = load(path)
    if value.get("kind") != "visual_iteration" or value.get("schema_version") != 1:
        raise ValueError("Not a visual iteration receipt")
    candidate = Path(value["candidate"]).resolve()
    if (
        not candidate.is_relative_to(root)
        or digest(candidate) != value["candidate_sha256"]
    ):
        raise ValueError("Iteration candidate changed or escapes workspace")
    if not value.get("cameras_unchanged") or not value.get("scale_unchanged"):
        raise ValueError("Iteration violates camera/scale invariants")
    if (
        not value.get("what")
        or not value.get("why")
        or not 1 <= len(value.get("frames", [])) <= 3
    ):
        raise ValueError("Iteration needs change, reason and one to three views")
    if type(value.get("changed_pixels")) is not int or value["changed_pixels"] < 0:
        raise ValueError("Invalid visual change count")
    if "preview_settings" in value:
        settings = value["preview_settings"]
        if (
            not isinstance(settings, dict)
            or set(settings) != {"samples", "threads", "width"}
            or any(
                type(number) is not int or number < (16 if key == "width" else 1)
                for key, number in settings.items()
            )
        ):
            raise ValueError("Invalid preview settings")
    artifacts = value.get("artifacts", {})
    if not artifacts or not any(p.endswith(".jpg") for p in artifacts):
        raise ValueError("Iteration needs visual comparisons")
    for name, sha in artifacts.items():
        p = Path(name).resolve()
        if not p.is_relative_to(root) or digest(p) != sha:
            raise ValueError("Iteration comparison changed or escapes workspace")
    if expected_region is not None:
        if (
            value.get("region") != expected_region
            or expected_region[:1] not in ("R", "S", "L", "J")
            or not expected_region[1:].isdigit()
        ):
            raise ValueError("Visual comparison region differs from assigned region")
        if value.get("comparison_format") != "reference_before_after_difference_v1":
            raise ValueError(
                "Worker needs labeled before/after/difference comparison images"
            )
        comparisons = [p for p in artifacts if p.endswith(".jpg")]
        if any(
            value.get("image_regions", {}).get(p) != expected_region
            for p in comparisons
        ):
            raise ValueError("Comparison image missing its assigned region label")
    return value
