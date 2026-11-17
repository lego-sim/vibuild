"""Selection, calibration and serialization contracts; no model or GPU needed."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import contextlib
import io

import numpy as np

SPEC = importlib.util.spec_from_file_location("perceive", Path(__file__).parents[1] / "tools/perceive.py")
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class PerceptionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "frame.png").write_bytes(b"image placeholder")
        self.frame = {"index": 7, "image": "frame.png", "split": "model",
                      "K": [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
                      "rotation": np.eye(3).tolist(), "center": [3, 4, 5]}

    def manifest(self, frames):
        path = self.root / "frames.json"
        path.write_text(json.dumps({"frames": frames}))
        return path

    def test_relative_images_and_exact_selection_preserve_manifest_order(self):
        path = self.manifest([self.frame, {**self.frame, "index": 8}, {**self.frame, "index": 9, "split": "audit"}])
        frames = p.load_frames(path, ["8", "7"])
        self.assertEqual([f["id"] for f in frames], ["7", "8"])
        self.assertEqual(frames[0]["image"], str(self.root / "frame.png"))

    def test_selection_rejects_audit_and_unknown_ids(self):
        path = self.manifest([self.frame, {**self.frame, "index": 9, "split": "audit"}])
        with self.assertRaisesRegex(p.PerceptionError, "rejected split"):
            p.load_frames(path)
        with self.assertRaisesRegex(p.PerceptionError, "rejected split"):
            p.load_frames(path, ["9"])
        with self.assertRaisesRegex(p.PerceptionError, "Unknown frame"):
            p.load_frames(path, ["70"])

    def test_missing_or_conflicting_split_is_not_training(self):
        frame = dict(self.frame)
        del frame["split"]
        for row in [frame, {**self.frame, "evaluation_split": "holdout"}, {**frame, "evaluation_split": "development"}]:
            with self.assertRaisesRegex(p.PerceptionError, "rejected split"):
                p.load_frames(self.manifest([row]))
        self.assertEqual(p.load_frames(self.manifest([{**frame, "evaluation_split": "fit"}]))[0]["split"], "fit")

    def test_explicit_audit_evidence_preserves_split_ids_and_calibration(self):
        audit = {**self.frame, "id": "audit_9", "split": "holdout", "evaluation_split": "audit"}
        path = self.manifest([self.frame, audit])
        with self.assertRaisesRegex(p.PerceptionError, "rejected split"):
            p.load_frames(path, ["audit_9"])
        selected = p.load_frames(path, ["audit_9"], audit_evidence=True)
        self.assertEqual(len(selected), 1)
        for key in ("id", "index", "split", "evaluation_split", "K", "rotation", "center"):
            self.assertEqual(selected[0][key], audit[key])
        metadata = p.frame_metadata(selected[0], [10, 20])
        self.assertEqual(metadata["split"], "holdout")
        self.assertEqual(metadata["evaluation_split"], "audit")
        holdout = {**audit, "evaluation_split": "holdout"}
        self.assertEqual(p.load_frames(self.manifest([holdout]), ["audit_9"], audit_evidence=True)[0]["evaluation_split"], "holdout")

    def test_audit_evidence_requires_exact_ids_and_original_audit_tags(self):
        audit = {**self.frame, "split": "holdout", "evaluation_split": "audit"}
        path = self.manifest([audit])
        for ids in (None, []):
            with self.assertRaisesRegex(p.PerceptionError, "explicit nonempty"):
                p.load_frames(path, ids, audit_evidence=True)
        with self.assertRaisesRegex(p.PerceptionError, "Unknown frame"):
            p.load_frames(path, ["unknown"], audit_evidence=True)
        with self.assertRaisesRegex(p.PerceptionError, "Duplicate requested"):
            p.load_frames(path, ["7", "7"], audit_evidence=True)
        no_split = {k: v for k, v in self.frame.items() if k != "split"}
        for row in [self.frame, {**audit, "split": "model"},
                    {**audit, "evaluation_split": "development"}, no_split]:
            with self.assertRaisesRegex(p.PerceptionError, "original audit/holdout tags"):
                p.load_frames(self.manifest([row]), ["7"], audit_evidence=True)

    def test_audit_dry_run_routes_opt_in_without_loading_models(self):
        from PIL import Image
        Image.new("RGB", (20, 10)).save(self.root / "frame.png")
        audit = {**self.frame, "split": "holdout", "evaluation_split": "audit"}
        path = self.manifest([audit, {**audit, "index": 8, "center": [4, 4, 5]}])
        config = self.root / "config.json"
        config.write_text(json.dumps({"runtime": {"da3": {
            "python": "/usr/bin/python3", "source": str(self.root), "checkpoint": str(self.root)}}}))
        args = ["depth", "--worker", "--config", str(config), "--frames", str(path),
                "--output", str(self.root / "depth"), "--ids", "8", "7", "--dry-run"]
        with self.assertRaisesRegex(p.PerceptionError, "rejected split"):
            p.main(args)
        output = io.StringIO()
        with patch.object(p, "readiness", side_effect=AssertionError("dry-run must not probe models")), \
             patch.object(p, "run_depth", side_effect=AssertionError("dry-run must not infer")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(p.main([*args, "--audit-evidence"]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["frame_ids"], ["7", "8"])
        self.assertTrue(report["audit_evidence"])
        self.assertEqual(report["frame_splits"][0], {"id": "7", "split": "holdout", "evaluation_split": "audit"})
        self.assertFalse(report["weights_loaded"])
        self.assertFalse((self.root / "depth").exists())

    def test_duplicate_ids_are_not_silently_deduplicated(self):
        with self.assertRaisesRegex(p.PerceptionError, "Duplicate manifest"):
            p.load_frames(self.manifest([self.frame, self.frame]))
        with self.assertRaisesRegex(p.PerceptionError, "Duplicate requested"):
            p.load_frames(self.manifest([self.frame]), ["7", "7"])

    def test_invalid_calibration_is_rejected(self):
        for patch in [{"center": [0, float("nan"), 0]}, {"rotation": np.diag([1, 1, -1]).tolist()}, {"K": [[0, 0, 1], [0, 2, 1], [0, 0, 1]]}]:
            with self.assertRaises(p.PerceptionError):
                p.load_frames(self.manifest([{**self.frame, **patch}]))

    def test_tracking_rejects_reverse_time(self):
        args = SimpleNamespace(concepts=["window"], max_objects=16)
        with self.assertRaisesRegex(p.PerceptionError, "timestamp"):
            p.validate_plan("sam31", {}, [{"video": "v", "time": 2}, {"video": "v", "time": 1}], [[10,20],[10,20]], args)

    def test_camera_roundtrip_uses_scene_coordinates(self):
        frame = {**self.frame, "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]]}
        k, e = p.camera_arrays([frame])
        world_point = np.array([3, 5, 7], np.float32)
        camera = e[0, :3, :3] @ world_point + e[0, :3, 3]
        np.testing.assert_allclose(camera, [1, 0, 2])
        output = p.depth_arrays(np.full((3, 3), 2), np.ones((3, 3)), k[0], e[0], k[0])
        np.testing.assert_allclose(output["xyz"][1, 2], world_point)

    def test_depth_keeps_every_pixel_raw_confidence_and_float32(self):
        depth = np.arange(35, dtype=np.float32).reshape(5, 7) / 7
        depth[0, 0] = np.nan
        confidence = np.linspace(-10, 10, 35).reshape(5, 7)
        original_k = np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]])
        transform = np.array([[.5, 0, -1], [0, .5, -2], [0, 0, 1]])
        output = p.depth_arrays(depth, confidence, transform @ original_k, np.eye(4), original_k)
        self.assertEqual(output["xyz"].shape, (5, 7, 3))
        np.testing.assert_array_equal(output["depth"], depth)
        np.testing.assert_array_equal(output["confidence"], confidence.astype(np.float32))
        np.testing.assert_allclose(output["original_to_output_pixels"], transform)
        self.assertFalse(output["valid"][0, 0])
        self.assertTrue(output["valid"][0, 1])  # Negative confidence is preserved, not thresholded.
        for key in ("depth", "confidence", "xyz", "intrinsics", "extrinsics"):
            self.assertEqual(output[key].dtype, np.float32)

    def test_mask_grid_and_confidence_are_lossless_at_arbitrary_size(self):
        mask = np.zeros((2, 1, 13, 19), bool)
        mask[0, 0, 2, 3] = True
        output = p.mask_arrays({"out_obj_ids": np.array([40, 900]), "out_binary_masks": mask,
                                "out_probs": np.array([.2, .75])}, [13, 19])
        self.assertEqual(output["masks"].shape, (2, 13, 19))
        self.assertTrue(output["masks"][0, 2, 3])
        np.testing.assert_array_equal(output["object_ids"], [40, 900])
        np.testing.assert_allclose(output["original_to_output_pixels"], np.eye(3))
        self.assertEqual(output["out_probs"].dtype, np.float32)

    def test_empty_detection_is_a_valid_native_grid(self):
        output = p.mask_arrays({"out_obj_ids": [], "out_binary_masks": np.zeros((0, 13, 19), bool)}, [13, 19])
        self.assertEqual(output["masks"].shape, (0, 13, 19))

    def test_unknown_mask_transform_is_rejected(self):
        with self.assertRaisesRegex(p.PerceptionError, "invented pixel transform"):
            p.mask_arrays({"out_obj_ids": [1], "out_binary_masks": np.zeros((1, 13, 19))}, [26, 38])

    def test_venv_python_symlink_is_not_resolved(self):
        path = self.root / "python"
        path.symlink_to("/usr/bin/python3")
        self.assertEqual(p.local_path("python", self.root), path)
        with self.assertRaises(p.PerceptionError):
            p.local_path("https://example.invalid/model", self.root)

    def test_dry_run_rejects_unobservable_scale_and_invalid_video_batch(self):
        args = SimpleNamespace(resolution=504, concepts=["window frame"], max_objects=16)
        with self.assertRaisesRegex(p.PerceptionError, "two distinct camera centers"):
            p.validate_plan("da3", {}, [self.frame, self.frame], [[13, 19]] * 2, args)
        with self.assertRaisesRegex(p.PerceptionError, "equal image sizes"):
            p.validate_plan("sam31", {}, [self.frame] * 2, [[13, 19], [15, 20]], args)
        with self.assertRaisesRegex(p.PerceptionError, "one video"):
            p.validate_plan("sam31", {}, [{**self.frame, "video": "a"}, {**self.frame, "video": "b"}], [[13, 19]] * 2, args)


if __name__ == "__main__":
    unittest.main()
