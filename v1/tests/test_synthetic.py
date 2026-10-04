import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from asteroid_validator.synthetic import (
    bolometric_thermal_flux,
    equilibrium_thermal_radiance,
    generate_benchmark,
    make_asteroid,
)


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_generation_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            manifest = generate_benchmark(output, seed=11, candidates=3, resolution=64)

            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(len(manifest["candidates"]), 3)
            self.assertTrue((output / "ground_truth" / "asteroid.npz").exists())
            self.assertTrue((output / "observations" / "optical.png").exists())
            self.assertTrue((output / "observations" / "silhouette.png").exists())
            self.assertTrue((output / "observations" / "infrared_10um_preview.png").exists())
            thermal_image = np.load(output / "observations" / "infrared_10um.npy")
            self.assertEqual(thermal_image.shape, (64, 64))
            self.assertGreater(float(thermal_image.max()), 0.0)
            self.assertEqual(
                json.loads((output / "manifest.json").read_text()) ["seed"], 11
            )

            optical = np.asarray(Image.open(output / "observations" / "optical.png"))
            silhouette = np.asarray(Image.open(output / "observations" / "silhouette.png"))
            self.assertEqual(optical.shape, (64, 64))
            self.assertEqual(silhouette.shape, (64, 64))
            self.assertGreater(int(np.count_nonzero(silhouette)), 0)
            self.assertGreater(
                manifest["observations"]["thermal"]["bolometric_flux_w_m2"], 0.0
            )
            self.assertEqual(manifest["observations"]["thermal"]["wavelength_um"], 10.0)

    def test_spectral_ir_radiance_is_not_scaled_optical_brightness(self) -> None:
        values = equilibrium_thermal_radiance(
            np.array([0.25, 0.5, 1.0]),
            bond_albedo=0.08,
            solar_flux_w_m2_at_1au=1361.0,
            heliocentric_distance_au=1.0,
            wavelength_um=10.0,
        )
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertNotAlmostEqual(values[2] / values[1], 2.0, places=2)

    def test_thermal_flux_scales_with_projected_area_and_distance(self) -> None:
        mesh = make_asteroid(5)
        common = {
            "sun_direction": (0.0, 0.0, 1.0),
            "observer_direction": (0.0, 0.0, 1.0),
            "observer_distance_m": 1000.0,
        }
        base = bolometric_thermal_flux(mesh, mesh_unit_m=1.0, **common)
        larger = bolometric_thermal_flux(mesh, mesh_unit_m=2.0, **common)
        farther = bolometric_thermal_flux(
            mesh,
            mesh_unit_m=1.0,
            sun_direction=(0.0, 0.0, 1.0),
            observer_direction=(0.0, 0.0, 1.0),
            observer_distance_m=2000.0,
        )
        self.assertGreater(base, 0.0)
        self.assertAlmostEqual(larger / base, 4.0, places=10)
        self.assertAlmostEqual(farther / base, 0.25, places=10)

    def test_generation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = generate_benchmark(root / "first", seed=3, candidates=2, resolution=48)
            second = generate_benchmark(root / "second", seed=3, candidates=2, resolution=48)
            self.assertEqual(first, second)
            self.assertEqual(
                (root / "first" / "observations" / "optical.png").read_bytes(),
                (root / "second" / "observations" / "optical.png").read_bytes(),
            )
            np.testing.assert_array_equal(
                np.load(root / "first" / "observations" / "infrared_10um.npy"),
                np.load(root / "second" / "observations" / "infrared_10um.npy"),
            )


if __name__ == "__main__":
    unittest.main()
