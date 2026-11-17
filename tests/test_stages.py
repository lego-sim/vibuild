"""Bounded reviews survive restarts and only the program advances stage state."""

import pytest

from vibuild.resources import load, save
from vibuild.runtime.layout import initialize_task
from vibuild.runtime.stages import transition


@pytest.fixture
def stage(tmp_path, monkeypatch):
    root = tmp_path / "project"
    initialize_task(root, [{"id": "R01"}, {"id": "R02", "depends_on": ["R01"]}])
    (root / "scene.blend").write_bytes(b"BLENDERcurrent")
    save(root / "status.json", {"active_region": "R01", "execution_paused": False})
    output = root / "R01/scores/001"
    save(
        output / ".runtime/protocol.json",
        {
            "scene": str(root / "scene.blend"),
            "regions": [{"region": "R01", "views": [{"id": "1"}, {"id": "2"}]}],
        },
    )
    monkeypatch.setattr(
        "vibuild.assessment.visual.verified_score", lambda *args: {"status": "passed"}
    )
    return root, output


def test_one_review_advances_to_next_stage(stage):
    root, score = stage
    with pytest.raises(ValueError, match="bounded visual"):
        transition(root, "accept")
    transition(root, "review", score=score, views=["1"])
    with pytest.raises(ValueError, match="already used"):
        transition(root, "review", score=score, views=["1"])
    result = transition(root, "accept")
    assert result["next_stage"] == "R02"
    with pytest.raises(ValueError, match="requires"):
        transition(root, "defer")
    result = transition(root, "defer", reason="Explicitly deferred by user")
    assert result["next_stage"] is None
    assert load(root / "status.json")["global_acceptance"] == "not_assessed"


def test_score_failure_and_invalid_views_never_advance(stage, monkeypatch):
    root, score = stage
    monkeypatch.setattr(
        "vibuild.assessment.visual.verified_score", lambda *args: {"status": "failed"}
    )
    with pytest.raises(ValueError, match="not passed"):
        transition(root, "review", score=score, views=["1"])
    monkeypatch.setattr(
        "vibuild.assessment.visual.verified_score", lambda *args: {"status": "passed"}
    )
    with pytest.raises(ValueError, match="passing assessment"):
        transition(root, "review", score=score, views=["wrong"])
    assert load(root / "status.json")["active_region"] == "R01"


def test_one_correction_is_recorded_without_full_rescoring(stage):
    root, score = stage
    transition(root, "review", score=score, begin=True)
    (root / "scene.blend").write_bytes(b"BLENDERrepaired")
    save(
        root / "R01/iterations/001/.runtime/iteration.json", {"what": "seam correction"}
    )
    transition(root, "review", finish=True, corrected=True, views=["1"])
    transition(root, "accept")
    assert load(root / "plan.json")["steps"][0]["closeout"]["corrections"] == 1


def test_second_correction_and_pause_block_advance(stage):
    root, score = stage
    transition(root, "review", score=score, begin=True)
    (root / "scene.blend").write_bytes(b"BLENDERrepaired")
    for n in (1, 2):
        save(root / f"R01/iterations/{n:03}/.runtime/iteration.json", {})
    with pytest.raises(ValueError, match="exactly one"):
        transition(root, "review", finish=True, corrected=True, views=["1"])
    state = load(root / "status.json")
    state["execution_paused"] = True
    save(root / "status.json", state)
    with pytest.raises(ValueError, match="paused"):
        transition(root, "accept")
