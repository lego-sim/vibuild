"""Registration-to-modeling smoke tests, using tiny real COLMAP and Blender inputs."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from vibuild import project
from vibuild.resources import load, save, digest
from vibuild.registration.colmap import undistort
from vibuild.registration.prepare import prepare_partition


@pytest.fixture
def registered_project(tmp_path):
    root = tmp_path / "project"
    project.initialize(root)
    reconstruction = tmp_path / "rec"
    reconstruction.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    (reconstruction / "cameras.txt").write_text("1 PINHOLE 64 32 45 45 32 16\n")
    rows = []
    for i in range(20):
        name = f"{i / 2:.1f}.png"
        Image.new("RGB", (64, 32), (100 + i, 120, 140)).save(images / name)
        rows += [f"{i + 1} 1 0 0 0 {-i * 0.01} 0 0 1 {name}", "32 16 1 33 16 2 32 17 3"]
    (reconstruction / "images.txt").write_text("\n".join(rows) + "\n")
    (reconstruction / "points3D.txt").write_text(
        "1 -.1 0 4 255 0 0 0\n2 .1 0 4 0 255 0 0\n3 0 .1 4 0 0 255 0\n"
    )
    result = project.register(root, reconstruction=reconstruction, images=images)
    assert result["registered_frames"] == 20
    return root


def test_reviewable_registration_keeps_original_images_and_reserved_views(
    registered_project,
):
    root = registered_project
    preview = prepare_partition(root)
    assert (root / "partition.html").is_file() and Path(preview["spec"]).is_file()
    raw = load(root / ".runtime/registration.json")["frames"]
    assert len(raw) == 20 and sum(f["evaluation_split"] == "audit" for f in raw) == 2
    frame = raw[0]
    res = root / "inputs/resources"
    assert (res / frame["raw_rgb"]).is_file() and (
        res / frame["original_rgb"]
    ).is_file()
    assert np.allclose(frame["K"], [[45, 0, 31.5], [0, 45, 15.5], [0, 0, 1]])
    assert (
        not (root / "scene.blend").exists()
        and not load(root / "project.json")["inputs_frozen"]
    )


def test_review_rejects_overlapping_regions_and_invalid_alignment(registered_project):
    root = registered_project
    spec = root / "spec.json"
    save(
        spec,
        {
            "regions": [
                {"id": name, "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]}
                for name in ("R01", "R02")
            ]
        },
    )
    with pytest.raises(ValueError, match="overlap"):
        prepare_partition(root, spec, True)
    save(
        spec,
        {
            "transform": {
                "basis": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                "origin": [0, 0, 0],
                "scale": 1,
            },
            "regions": [
                {"id": "R01", "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]}
            ],
        },
    )
    with pytest.raises(ValueError, match="proper rotation"):
        prepare_partition(root, spec, True)


def test_review_creates_registered_scene_and_one_initial_assessment(registered_project):
    blender = os.environ.get("VIBUILD_TEST_BLENDER")
    if not blender:
        pytest.skip("set VIBUILD_TEST_BLENDER for lifecycle smoke")
    root = registered_project
    from vibuild.blender.scene import main as scene

    config = load(root / "vibuild.json")
    config["runtime"]["blender"] = blender
    save(root / "vibuild.json", config)
    prepare_partition(root)
    result = project.partition(root, root / "partition.review.json", True)
    assert load(root / "project.json")["inputs_frozen"] and result["scene"] == str(
        root / "scene.blend"
    )
    assert list(root.rglob("*.blend")) == [root / "scene.blend"]
    plan = load(root / "plan.json")
    assert plan["execution_order"] == ["R01"]
    task = load(root / "R01/job_template.json")
    assert task["steps"][0]["args"][0] == "prepare"
    assert (
        scene(
            [
                "inspect",
                "--blend",
                str(root / "scene.blend"),
                "--output",
                str(root / ".runtime/inspect"),
                "--blender",
                blender,
                "--camera-details",
            ]
        )
        == 0
    )
    inventory = load(root / ".runtime/inspect/inventory.json")
    assert len(inventory["registered_cameras"]) == 20
    assert all(camera["is_registered"] for camera in inventory["registered_cameras"])
    # Initialize and render using the exact same calibration; this verifies camera export.
    from vibuild.assessment.visual import main as score

    assert (
        score(
            task["steps"][0]["args"]
            + [
                "--config",
                str(root / "vibuild.json"),
                "--samples",
                "1",
                "--workers",
                "1",
            ]
        )
        == 0
    )
    packet = load(root / "R01/scores/001/R01/packet.json")
    assert len(packet["views"]) >= 2
    assert all(Path(v["reference_rgb"]).is_file() for v in packet["views"])
    assert not any(v["id"] in {"9", "19"} for v in packet["views"])
    source_hash = digest(root / "inputs/resources/alignment/frames.json")
    scene_hash = digest(root / "scene.blend")
    project.pause(root)
    assert project.status(root)["paused"]
    project.resume(root)
    assert not project.status(root)["paused"]
    assert (
        digest(root / "scene.blend") == scene_hash
        and digest(root / "inputs/resources/alignment/frames.json") == source_hash
    )
    with pytest.raises(ValueError, match="frozen"):
        project.register(root, reconstruction=root, images=root)


def test_undistortion_identity_and_distorted_center():
    image = Image.fromarray(np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3))
    same, K = undistort(image, {"model": "PINHOLE", "params": [3, 3, 2.5, 2.5]})
    assert same is image and K[0, 2] == 2
    warped, _ = undistort(
        image, {"model": "SIMPLE_RADIAL", "params": [3, 2.5, 2.5, 0.1]}
    )
    assert np.array_equal(np.asarray(warped)[2, 2], np.asarray(image)[2, 2])
