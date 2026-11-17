"""Validated time limits for one model session."""

import math
from vibuild.resources import load

DEFAULTS = dict(session_minutes=60, idle_minutes=20)


def limits(config):
    result = {**DEFAULTS, **load(config).get("budget", {})}
    for key, value in result.items():
        if (
            key not in DEFAULTS
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid budget: {key}={value}")
    return result
