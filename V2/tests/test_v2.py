import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from asteroid_validator_v2.dataset import validate_splits, write_json
from asteroid_validator_v2.demo import generate_case, demo_dataset
from asteroid_validator_v2.features import extract_case
from asteroid_validator_v2.geometry import geometric_error, load_mesh, surface_distances, sample_surface, diameter, LABEL_METRIC
from asteroid_validator_v2.model import (ErrorEvaluator, forward, loss_gradient, train,
                                         FEATURE_VERSION)
from asteroid_validator_v2.physics.synthetic import Mesh
from asteroid_validator_v2.pipeline import fit_dataset, predict_features


def square(z=0, subdivided=False):
    vertices = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    if subdivided:
        vertices = np.vstack((vertices, [.5, .5, z]))
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return Mesh(vertices, faces)


class GeometryTests(unittest.TestCase):
    def test_exact_point_to_triangle_not_vertex_distance(self):
        np.testing.assert_allclose(surface_distances(np.array([[.25, .25, 1], [2, .5, 0], [2, 2, 1]]),
                                                     square()), [1, 1, np.sqrt(3)], atol=1e-12)

    def test_retriangulated_identical_surface_has_zero_error(self):
        result = geometric_error(square(), square(subdivided=True), samples=128)
        self.assertLess(result["geometric_error"], 1e-12)

    def test_parallel_surfaces_have_analytic_error_and_preserve_translation(self):
        result = geometric_error(square(.2), square(), samples=128)
        self.assertAlmostEqual(result["geometric_error"], .2 / np.sqrt(2))
        self.assertLess(result["sampling_standard_error"], 1e-12)

    def test_normalized_error_is_invariant_to_joint_unit_change(self):
        a, b = square(.2), square()
        original = geometric_error(a, b, samples=128)["geometric_error"]
        scaled = geometric_error(Mesh(a.vertices * 17, a.faces), Mesh(b.vertices * 17, b.faces),
                                 samples=128)["geometric_error"]
        self.assertAlmostEqual(original, scaled)

    def test_area_sampling_does_not_weight_faces_equally(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [13, 0, 0], [10, 3, 0.]])
        p = sample_surface(Mesh(v, np.array([[0, 1, 2], [3, 4, 5]])), 5000, seed=9)
        self.assertAlmostEqual(float((p[:, 0] > 5).mean()), .9, delta=.025)

    def test_loader_preserves_frame_and_supports_obj_negative_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            path.write_text("v 10 0 0\nv 11 0 0\nv 10 1 0\nf -3 -2 -1\n")
            mesh = load_mesh(path)
            self.assertEqual(mesh.vertices[0, 0], 10)
            np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])

    def test_invalid_indices_and_degenerate_surface_fail(self):
        with self.assertRaises(ValueError):
            sample_surface(Mesh(square().vertices, np.array([[0, 1, 99]])), 8)
        with self.assertRaises(ValueError):
            sample_surface(Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]])), 8)

    def test_diameter_ignores_unused_vertices_and_degenerate_faces(self):
        base = square()
        mesh = Mesh(np.vstack((base.vertices, [100, 100, 100])),
                    np.vstack((base.faces, [4, 4, 4])))
        self.assertAlmostEqual(diameter(mesh), np.sqrt(2))


class RegressionTests(unittest.TestCase):
    def test_all_chain_rule_gradients_match_finite_differences(self):
        rng = np.random.default_rng(3)
        x, y = rng.uniform(0, 3, (12, 3)), rng.uniform(0, .1, 12)
        parameters = np.array([.2, -.3, .1, -1.4, -3.])
        _, gradient = loss_gradient(parameters, x, y)
        approximate = []
        for index in range(5):
            delta = np.zeros(5)
            delta[index] = 1e-5
            plus = loss_gradient(parameters + delta, x, y)[0]
            minus = loss_gradient(parameters - delta, x, y)[0]
            approximate.append((plus - minus) / 2e-5)
        np.testing.assert_allclose(gradient, approximate, rtol=1e-5, atol=1e-10)

    def test_adam_learns_continuous_labels_and_saved_predictions_roundtrip(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(.1, 4, (100, 3))
        y = forward(np.array([1., -1., 0., -1., -4.]), x)[0]
        model, history = train(x[:70], y[:70], x[70:], y[70:], epochs=1500, seed=2)
        self.assertLess(model.metadata["best_validation_mse"], history[0]["validation_mse"] * .1)
        np.testing.assert_allclose(model.scales, np.median(x[:70], axis=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = ErrorEvaluator.load(path)
            np.testing.assert_allclose(model.predict(x), loaded.predict(x), atol=0, rtol=0)
        self.assertTrue(np.all(model.predict(x) >= 0))
        self.assertTrue(np.all(model.predict(x + 1) >= model.predict(x)))

    def test_nonfinite_and_missing_inputs_fail(self):
        model = ErrorEvaluator(np.zeros(5), np.ones(3))
        for x in ([[0, 0]], [[0, np.nan, 1]], [[-1, 1, 1]]):
            with self.assertRaises(ValueError):
                model.predict(np.asarray(x))


class PipelineTests(unittest.TestCase):
    def test_split_leakage_and_duplicate_reference_files_are_rejected(self):
        rows = [{"asteroid_id": "a", "candidate_id": "one", "split": "train"},
                {"asteroid_id": "a", "candidate_id": "two", "split": "test"}]
        data = {"schema_version": 1, "feature_version": FEATURE_VERSION,
                "label_metric": LABEL_METRIC, "records": rows}
        with self.assertRaisesRegex(ValueError, "object leakage"):
            validate_splits(data)
        rows[1]["asteroid_id"] = "b"
        for row in rows:
            row["reference_sha256"] = "same"
        with self.assertRaisesRegex(ValueError, "identical reference"):
            validate_splits(data)

    def test_observation_extraction_works_after_reference_mesh_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_case(root, 30, 3, 32)
            (root / "ground_truth/asteroid.npz").unlink()
            report = extract_case(root / "manifest.json")
            self.assertEqual(len(report["records"]), 3)
            self.assertFalse(report["reference_geometry_used_for_features"])
            self.assertEqual(report["failures"], [])

    def test_end_to_end_train_then_predict_without_labels_or_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = demo_dataset(root, train_objects=3, validation_objects=1, test_objects=2,
                                   candidates=3, resolution=32, samples=64, seed=72)
            report = fit_dataset(dataset, root / "experiment", epochs=100, seed=72)
            self.assertEqual(report["split_object_counts"], {"train": 3, "validation": 1, "test": 2})
            case = root / "cases/synthetic_76"
            (case / "ground_truth/asteroid.npz").unlink()
            features = extract_case(case / "manifest.json")
            self.assertTrue(all("geometric_error" not in row for row in features["records"]))
            write_json(root / "features.json", features)
            prediction = predict_features(root / "experiment/model.json", root / "features.json", root / "prediction.json")
            self.assertFalse(prediction["reference_geometry_required"])
            self.assertEqual(len(prediction["candidates"]), 3)
            errors = [row["predicted_geometric_error"] for row in prediction["candidates"]]
            self.assertEqual(errors, sorted(errors))

    def test_wrong_label_units_and_alignment_policy_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "geometric-error metric"):
            validate_splits({"schema_version": 1, "feature_version": FEATURE_VERSION,
                             "label_metric": "vertex_rmse", "records": []})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ErrorEvaluator(np.zeros(5), np.ones(3), {"alignment_policy": "supplied-camera"})
            model.save(root / "model.json")
            write_json(root / "features.json", {"schema_version": 1, "feature_version": FEATURE_VERSION,
                        "alignment_policy": "outline-fixed", "records": []})
            with self.assertRaisesRegex(ValueError, "alignment policy"):
                predict_features(root / "model.json", root / "features.json", root / "output.json")


if __name__ == "__main__":
    unittest.main()
