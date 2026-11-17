"""Shared score-report contract and comparison metrics for tools and supervisor."""

import math

# Candidate iteration policy, independent of frozen measurement/acceptance gates.
REVIEW_POLICY = {
    "version": 2,
    "max_coverage_drop": 0.01,
    "default_coverage_floor": 0.95,
}


def validate_score_report(report, name="score report"):
    if (
        not isinstance(report, dict)
        or not isinstance(report.get("protocol_sha256"), str)
        or not report["protocol_sha256"]
    ):
        raise ValueError(
            f"{name}: expected an evaluate score report with protocol_sha256; put explanations in notes, not score reports"
        )
    if (
        report.get("structure_status") not in ("blocked", "failed", "passed")
        or not isinstance(report.get("frames"), list)
        or not isinstance(report.get("plausibility"), dict)
        or not isinstance(report.get("appearance"), dict)
    ):
        raise ValueError(f"{name}: incomplete evaluate score report")
    for frame in report["frames"]:
        if (
            not isinstance(frame, dict)
            or "id" not in frame
            or frame.get("status") not in ("blocked", "failed", "passed")
            or not isinstance(frame.get("components", {}), dict)
        ):
            raise ValueError(f"{name}: malformed frame score")
        if any(not isinstance(v, dict) for v in frame.get("components", {}).values()):
            raise ValueError(f"{name}: malformed component score")
    appearance = report["appearance"].get("frames", [])
    if not isinstance(appearance, list) or any(
        not isinstance(f, dict) or "id" not in f for f in appearance
    ):
        raise ValueError(f"{name}: malformed appearance scores")
    for score in [report["plausibility"], *appearance]:
        value = score.get("score_0_100", 0)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError(f"{name}: invalid score_0_100")
    ids = [str(f["id"]) for f in report["frames"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate evaluation frame IDs")
    return report


def score_values(report):
    validate_score_report(report)
    scores = {}
    signs = dict(
        abs_rel=-1,
        coverage=1,
        semantic_iou=1,
        precision=1,
        recall=1,
        count_relative_error=-1,
    )
    gates = {"blocked": -1, "failed": 0, "passed": 1}
    for frame in report["frames"]:
        key = str(frame["id"])
        scores[key + "/gate"] = gates.get(frame.get("status"), -1)
        for component, item in frame.get("components", {}).items():
            prefix = f"{key}/{component}"
            scores[prefix + "/gate"] = gates.get(item.get("status"), -1)
            for metric, sign in signs.items():
                value = item.get(metric)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    scores[prefix + "/" + metric] = sign * value
    p = report["plausibility"]
    scores["plausibility/gate"] = gates.get(p.get("status"), -1)
    scores["plausibility"] = p.get("score_0_100", 0) / 100
    for frame in report["appearance"].get("frames", []):
        if frame.get("status") == "scored":
            scores[f"{frame['id']}/appearance"] = frame.get("score_0_100", 0) / 100
    return scores


def coverage_floors(report):
    return {
        str(f["id"]) + "/depth/coverage": (f.get("limits") or {}).get(
            "depth_coverage", REVIEW_POLICY["default_coverage_floor"]
        )
        for f in report["frames"]
    }


def compare_scores(before, after, floors=None):
    """Review a candidate without turning every small metric loss into a veto.

    All raw losses remain visible. Missing metrics and loss of a passed gate still
    block selection. Unknown manual reviews and unmatched instance counts cannot
    establish numerical improvement or regression, but their failed gates remain.
    """
    floors = floors or {}

    def advisory(key):
        if key.endswith("plausibility"):
            if before.get(key + "/gate") == after.get(key + "/gate") == -1:
                return "both plausibility reviews are blocked; checklist fractions are not comparable quality scores"
        if key.endswith("/count_relative_error"):
            prefix = key.rsplit("/", 1)[0]
            if all(
                values.get(prefix + "/gate") == 0
                and all(
                    values.get(prefix + "/" + k) == 0
                    for k in ("semantic_iou", "precision", "recall")
                )
                for values in (before, after)
            ):
                return "both component assessments have zero overlap and zero instance matches; unmatched counts alone do not rank candidates"
        return None

    raw = [k for k in before if k not in after or after[k] < before[k] - 1e-6]
    blocked, tolerated = [], []
    for key in raw:
        reason = advisory(key) if key in after else None
        if key in after and key.endswith("/depth/coverage"):
            error = (
                key.rsplit("/", 1)[0] + "/abs_rel"
            )  # Signed score: higher is better.
            floor = floors.get(key, REVIEW_POLICY["default_coverage_floor"])
            if (
                before[key] - after[key] <= REVIEW_POLICY["max_coverage_drop"] + 1e-6
                and after[key] >= floor
                and error in before
                and error in after
                and after[error] >= before[error] - 1e-6
            ):
                reason = "coverage loss is within 1 percentage point, remains above its frozen floor, and depth error does not worsen"
        if reason:
            tolerated.append(
                {
                    "metric": key,
                    "before_score": before[key],
                    "after_score": after[key],
                    "reason": reason,
                }
            )
        else:
            blocked.append(key)
    gains = [
        k
        for k in after
        if (k not in before or after[k] > before[k] + 1e-4) and not advisory(k)
    ]
    return {
        "review_policy": dict(REVIEW_POLICY),
        "raw_regressions": raw,
        "regressions": blocked,
        "tolerated_regressions": tolerated,
        "gains": gains,
        "measured_improvement": bool(gains) and not blocked,
        "candidate_recommended": bool(gains) and not blocked,
    }
