"""Bounded, cancellable render admission shared by local tools."""

from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import time
from vibuild.resources import load


def policy(config):
    value = {
        "max_workers": 3,
        "threads_per_worker": 4,
        **load(config).get("rendering", {}),
    }
    if set(value) != {"max_workers", "threads_per_worker"}:
        raise ValueError("Unknown render limit")
    if any(type(n) is not int or n <= 0 for n in value.values()):
        raise ValueError("Render limits must be positive integers")
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    value["effective_workers"] = min(
        value["max_workers"], max(1, cores // value["threads_per_worker"])
    )
    return value


@contextmanager
def render_reservation(root, config, *, deadline=None, stop_paths=(), poll_seconds=0.2):
    """Acquire one bounded render slot, separately from model-session admission.

    Waits are cancellation/deadline-aware. An acquired OS lock releases on process exit;
    stable slot files are intentionally not unlinked, avoiding inode replacement races.
    """
    if poll_seconds <= 0:
        raise ValueError("Positive render poll interval required")
    directory = Path(root) / "cache/render_slots"
    directory.mkdir(parents=True, exist_ok=True)
    count = policy(config)["effective_workers"]
    handle = None
    stops = [Path(p) for p in stop_paths]
    try:
        while handle is None:
            if any(path.exists() for path in stops):
                raise RuntimeError("Render admission cancelled")
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("Render admission deadline exceeded")
            for number in range(count):
                candidate = (directory / f"{number}.lock").open("a+")
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                handle = candidate
                break
            if handle is None:
                time.sleep(
                    min(poll_seconds, max(0, deadline - time.time()))
                    if deadline is not None
                    else poll_seconds
                )
        yield
    finally:
        if handle:
            handle.close()
