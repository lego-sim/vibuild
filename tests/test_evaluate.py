"""Regression checks for evidence gates, calibrated mapping and instance matching."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parents[1] / "tools/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def sample():
    return {"depth": np.array([[2., 3., 4.], [2., 3., 4.]], np.float32),
            "K": np.eye(3), "confidence": np.ones((2, 3)), "opaque": np.ones((2, 3), bool),
            "semantic": np.array([[1, 2, 3], [1, 2, 3]], np.int32),
            "instance": np.array([[10, 20, 30], [10, 20, 30]], np.int32)}


def test_native_intrinsic_mapping_and_range_depth():
    ref = sample()
    pred = {"depth": np.full((2, 6), 99., np.float64), "K": np.diag([2., 1., 1.])}
    pred["depth"][:, ::2] = ref["depth"]
    _, aligned, inside, mapping = evaluation.align_prediction(ref, pred)
    assert np.array_equal(aligned["depth"], ref["depth"])
    assert inside.all()
    assert mapping[0, 0] == 2
    y, x = np.indices(ref["depth"].shape)
    ranged = dict(ref, depth=ref["depth"] * np.sqrt(1+x*x+y*y), depth_convention=np.array("range"))
    assert np.allclose(evaluation.z_depth(ranged, ref["K"]), ref["depth"])


def test_depth_masks_and_float_precision():
    ref = sample(); pred = sample()
    pred["depth"] = pred["depth"].astype(np.float64)
    pred["depth"][0, 0] += .000001
    pred["depth"][0, 1:] = 1000
    ref["opaque"][0, 1] = False
    ref["confidence"][0, 2] = 0
    result = evaluation.evaluate_arrays(ref, pred)
    assert result["status"] == "passed"
    assert 0 < result["components"]["depth"]["mae_scene_units"] < 1e-6
    assert result["components"]["depth"]["reference_pixels"] == 4


def test_missing_prediction_depth_cannot_hide_behind_overlap():
    ref = sample(); pred = sample()
    pred["depth"][:, 1:] = np.nan
    depth = evaluation.evaluate_arrays(ref, pred)["components"]["depth"]
    assert depth["status"] == "failed"
    assert depth["abs_rel"] == 0
    assert depth["coverage"] == pytest.approx(1/3)


def test_missing_required_component_blocks():
    ref = sample(); pred = sample()
    ref["semantic"][ref["semantic"] == 3] = 0
    result = evaluation.evaluate_arrays(ref, pred)
    assert result["status"] == "blocked"
    assert result["components"]["tile"]["status"] == "blocked"
    ref = sample(); del ref["opaque"]
    assert evaluation.evaluate_arrays(ref, pred)["components"]["depth"]["status"] == "blocked"


def test_one_to_one_instance_matching_rejects_merge():
    ref = sample(); pred = sample()
    ref["semantic"][:] = 1; pred["semantic"][:] = 1
    pred["instance"][:] = 100
    result = evaluation.evaluate_arrays(ref, pred, classes={"window_frame": 1})
    assert result["status"] == "failed"
    assert result["components"]["window_frame"]["reference_count"] == 3
    assert result["components"]["window_frame"]["prediction_count"] == 1


def test_unknown_plausibility_and_self_report_do_not_accept(tmp_path):
    assert evaluation.plausibility_gate(None, tmp_path)["status"] == "blocked"
    assert evaluation.plausibility_gate({"status": "passed"}, tmp_path)["status"] == "blocked"
    np.savez(tmp_path / "ref.npz", **sample())
    np.savez(tmp_path / "pred.npz", **sample())
    manifest = {"frames": [{"reference": "ref.npz", "prediction": "pred.npz"}]}
    result = evaluation.evaluate_manifest(manifest, tmp_path)
    assert result["frames"][0]["status"] == "passed"
    assert result["structure_status"] == "blocked"
    assert result["overall_acceptance"] == "not_assessed"
    assert not result["structure_accepted"]


def test_complete_structure_evidence_stays_separate_from_appearance(tmp_path):
    np.savez(tmp_path / "ref.npz", **sample())
    np.savez(tmp_path / "pred.npz", **sample())
    (tmp_path / "review.txt").write_text("Manual inspection evidence")
    review = {"status": "passed", "reviewer": "human", "evidence": ["review.txt"],
              "checks": dict.fromkeys(("dimensions", "topology", "openings", "occlusion"), "passed")}
    manifest = {"frames": [{"reference": "ref.npz", "prediction": "pred.npz"}], "plausibility": review}
    result = evaluation.evaluate_manifest(manifest, tmp_path)
    assert result["structure_accepted"]
    assert result["appearance"]["status"] == "not_evaluated"
    assert result["overall_acceptance"] == "not_assessed"
    review["checks"]["topology"] = "failed"
    assert evaluation.evaluate_manifest(manifest, tmp_path)["structure_status"] == "failed"


def test_empty_inputs_block(tmp_path):
    assert evaluation.evaluate_manifest({}, tmp_path)["structure_status"] == "blocked"
    with pytest.raises(ValueError, match="nonempty"):
        evaluation.evaluate_arrays({"depth": np.empty((0, 0)), "K": np.eye(3)}, sample())
    ref = sample(); ref["opaque"][:] = False
    assert evaluation.evaluate_arrays(ref, sample())["components"]["depth"]["status"] == "blocked"


def test_pose_mismatch_and_integer_depth_rejected():
    ref = sample(); pred = sample()
    ref["camera_to_world"] = np.eye(4); pred["camera_to_world"] = np.eye(4)
    pred["camera_to_world"][0, 3] = 1
    with pytest.raises(ValueError, match="camera poses differ"):
        evaluation.evaluate_arrays(ref, pred)
    pred = sample(); pred["depth"] = pred["depth"].astype(np.uint16)
    with pytest.raises(ValueError, match="float"):
        evaluation.evaluate_arrays(sample(), pred)


def test_raw_model_confidence_and_perception_intrinsics(tmp_path):
    ref = sample()
    ref["confidence"][:] = 12.
    assert evaluation.evaluate_arrays(ref, sample(), min_confidence=10)["components"]["depth"]["status"] == "passed"
    ref["intrinsics"] = ref.pop("K")
    ref["extrinsics"] = np.eye(4)[:3]
    np.savez(tmp_path / "depth.npz", **ref)
    loaded = evaluation.load_npz(tmp_path / "depth.npz")
    assert np.array_equal(loaded["K"], np.eye(3))
    assert np.array_equal(loaded["camera_to_world"], np.eye(4))
    ref = sample(); ref["confidence"][:] = -2
    assert evaluation.evaluate_arrays(ref, sample(), min_confidence=-3)["components"]["depth"]["status"] == "passed"


def rgb_frame(tmp_path):
    np.save(tmp_path / "reference.npy", np.zeros((3, 4, 3), dtype=np.float32))
    np.save(tmp_path / "prediction.npy", np.zeros((3, 4, 3), dtype=np.float32))
    calibration = {"K": np.eye(3).tolist(), "camera_to_world": np.eye(4).tolist()}
    return {"reference_rgb": "reference.npy", "prediction_rgb": "prediction.npy",
            "appearance_calibration": {"reference": calibration, "prediction": calibration}}


def test_appearance_missing_images_empty_mask_and_calibration(tmp_path):
    frame = rgb_frame(tmp_path)
    frame["prediction_rgb"] = "missing.png"
    assert evaluation.appearance_frame(frame, tmp_path)["status"] == "blocked"
    frame["prediction_rgb"] = "prediction.npy"
    np.save(tmp_path / "mask.npy", np.zeros((3, 4), dtype=bool))
    frame["appearance_mask"] = "mask.npy"
    assert evaluation.appearance_frame(frame, tmp_path)["status"] == "blocked"
    del frame["appearance_mask"]
    del frame["appearance_calibration"]
    assert evaluation.appearance_frame(frame, tmp_path)["status"] == "blocked"


def test_appearance_does_not_resize_or_hide_baseline_regression(tmp_path):
    frame = rgb_frame(tmp_path)
    frame["baseline_rgb"] = "reference.npy"
    frame["appearance_calibration"]["baseline"] = frame["appearance_calibration"]["reference"]
    np.save(tmp_path / "prediction.npy", np.ones((3, 4, 3), dtype=np.float32))
    result = evaluation.appearance_frame(frame, tmp_path)
    assert result["score_0_100"] == 0
    assert result["rgb_rmse"] == 1
    assert result["cie_delta_e76_mean"] == pytest.approx(100, abs=.001)
    assert result["regressed"] and result["score_delta"] == -100
    np.save(tmp_path / "prediction.npy", np.ones((6, 8, 3), dtype=np.float32))
    assert evaluation.appearance_frame(frame, tmp_path)["status"] == "blocked"


def test_appearance_and_structure_are_never_averaged(tmp_path):
    frame = rgb_frame(tmp_path)
    # Perfect appearance cannot compensate for missing structure evidence.
    report = evaluation.evaluate_manifest({"frames": [frame]}, tmp_path)
    assert report["appearance"]["frames"][0]["score_0_100"] == 100
    assert report["structure_status"] == "blocked"
    assert report["overall_acceptance"] == "not_assessed"
    # Poor appearance cannot change the independently measured structure gate.
    np.savez(tmp_path / "ref.npz", **sample()); np.savez(tmp_path / "pred.npz", **sample())
    frame.update(reference="ref.npz", prediction="pred.npz")
    np.save(tmp_path / "prediction.npy", np.ones((3, 4, 3), dtype=np.float32))
    (tmp_path / "review.txt").write_text("review evidence")
    review = {"status": "passed", "reviewer": "human", "evidence": ["review.txt"],
              "checks": dict.fromkeys(("dimensions", "topology", "openings", "occlusion"), "passed")}
    report = evaluation.evaluate_manifest({"frames": [frame], "plausibility": review}, tmp_path)
    assert report["structure_status"] == "passed"
    assert report["appearance"]["frames"][0]["score_0_100"] == 0
    assert report["overall_acceptance"] == "not_assessed"
    assert report["plausibility"]["passed_fraction"] == 1
    del review["checks"]["occlusion"]
    plausibility = evaluation.plausibility_gate(review, tmp_path)
    assert plausibility["score_0_100"] == 75
    assert plausibility["unknown_checks"] == 1
    assert plausibility["status"] == "blocked"


def test_protocol_hash_freezes_evidence_not_predictions(tmp_path):
    (tmp_path / "reference.npz").write_bytes(b"fixed native depth")
    manifest = {"frames": [{"id": "24", "reference": "reference.npz", "prediction": "a.npz"}],
                "classes": {"glass": 1}, "limits": {"depth_abs_rel": .1}}
    baseline = evaluation.protocol_hash(manifest, tmp_path)
    manifest["frames"][0]["prediction"] = "b.npz"
    assert evaluation.protocol_hash(manifest, tmp_path) == baseline
    manifest["limits"]["depth_abs_rel"] = .2
    assert evaluation.protocol_hash(manifest, tmp_path) != baseline
    manifest["limits"]["depth_abs_rel"] = .1
    (tmp_path / "reference.npz").write_bytes(b"changed reference")
    assert evaluation.protocol_hash(manifest, tmp_path) != baseline
