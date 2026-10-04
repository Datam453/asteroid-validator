import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from asteroid_validator.calibration import (
    calibrate_weights,
    load_calibration_case,
    spearman_rank_correlation,
)


class CalibrationTests(unittest.TestCase):
    def test_spearman_identifies_matching_and_reversed_orders(self) -> None:
        values = np.asarray([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(spearman_rank_correlation(values, values), 1.0)
        self.assertAlmostEqual(spearman_rank_correlation(values, values[::-1]), -1.0)

    def test_calibration_prefers_the_informative_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "candidates": [
                    {"id": f"candidate_{index:03d}", "benchmark_only_vertex_rmse": value}
                    for index, value in enumerate((0.1, 0.2, 0.3, 0.4))
                ]
            }
            report = {
                "meshes": [
                    {
                        "mesh_path": f"candidate_{index:03d}.npz",
                        "silhouette_score": 1.0,
                        "photometric_score": truth,
                        "thermal_score": 0.4 - 0.1 * index,
                    }
                    for index, truth in enumerate((0.1, 0.2, 0.3, 0.4))
                ]
            }
            manifest_path = root / "manifest.json"
            report_path = root / "report.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            case = load_calibration_case(manifest_path, report_path)
            result = calibrate_weights([case], grid_steps=10)

            self.assertAlmostEqual(result["mean_training_spearman"], 1.0)
            self.assertGreater(
                result["normalized_weights"]["photometric_score"],
                result["normalized_weights"]["thermal_score"],
            )
            self.assertTrue(result["cases"][0]["top1_correct"])


if __name__ == "__main__":
    unittest.main()
