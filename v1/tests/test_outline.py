import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from asteroid_validator.outline import (
    _distance_map_to_points,
    OutlineMatcherConfig,
    PhotometricConfig,
    ThermalConfig,
    load_grayscale_image,
    match_files,
    match_mesh_outline,
    outline_metrics,
    photometric_normal_metrics,
    render_silhouette,
    segment_silhouette,
    thermal_flux_metrics,
    thermal_image_metrics,
)
from asteroid_validator.synthetic import bolometric_thermal_flux, generate_benchmark, make_asteroid


class OutlineMatcherTests(unittest.TestCase):
    def test_distance_map_matches_brute_force_at_high_resolution(self) -> None:
        shape = (256, 256)
        points = np.asarray([[12, 27], [100, 95], [238, 241]], dtype=np.int32)
        actual = _distance_map_to_points(points, shape)
        yy, xx = np.indices(shape)
        expected = np.sqrt(
            np.min(
                np.stack([(yy - y) ** 2 + (xx - x) ** 2 for y, x in points]),
                axis=0,
            )
        )
        np.testing.assert_allclose(actual, expected)

    def test_synthetic_image_segmentation_recovers_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            generate_benchmark(output, seed=7, candidates=2, resolution=64)
            image = load_grayscale_image(output / "observations" / "optical.png")
            mask, diagnostics = segment_silhouette(image)
            self.assertEqual(mask.shape, (64, 64))
            self.assertGreater(mask.sum(), 100)
            self.assertLess(mask.sum(), 64 * 64 * 0.9)
            self.assertEqual(diagnostics["method"], "automatic_largest_component")

    def test_outline_metrics_is_symmetric_and_rewards_overlap(self) -> None:
        observed = np.zeros((32, 32), dtype=bool)
        observed[8:24, 10:22] = True
        same = outline_metrics(observed, observed.copy())
        shifted = np.roll(observed, shift=4, axis=1)
        shifted_metrics = outline_metrics(observed, shifted)
        self.assertAlmostEqual(same["score"], 0.0, places=7)
        self.assertGreater(shifted_metrics["score"], same["score"])
        self.assertGreater(shifted_metrics["symmetric_chamfer_px"], 0.0)

    def test_photometric_normal_cone_rewards_an_admissible_normal(self) -> None:
        image = np.full((2, 2), 0.5, dtype=np.float64)
        mask = np.ones((2, 2), dtype=bool)
        matching_normals = np.zeros((2, 2, 3), dtype=np.float64)
        matching_normals[..., 0] = np.sqrt(0.75)
        matching_normals[..., 2] = 0.5
        wrong_normals = np.zeros((2, 2, 3), dtype=np.float64)
        wrong_normals[..., 2] = 1.0
        config = PhotometricConfig(
            light_direction=(0.0, 0.0, 1.0),
            albedo=1.0,
        )

        matching = photometric_normal_metrics(
            image, mask, matching_normals, mask, config
        )
        wrong = photometric_normal_metrics(image, mask, wrong_normals, mask, config)

        self.assertAlmostEqual(matching["photometric_score"], 0.0, places=7)
        self.assertGreater(wrong["photometric_score"], matching["photometric_score"])
        self.assertAlmostEqual(matching["photometric_coverage"], 1.0)

    def test_matcher_reports_photometric_score_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            manifest = generate_benchmark(output, seed=7, candidates=2, resolution=64)
            observation = manifest["observations"]
            report = match_files(
                output / observation["optical"],
                [output / manifest["candidates"][0]["path"]],
                mask_path=output / observation["silhouette"],
                output_path=output / "photometric_report.json",
                config=OutlineMatcherConfig(
                    coarse_yaw_step_deg=90.0,
                    coarse_pitch_angles_deg=(-45.0, 0.0, 45.0),
                    coarse_roll_step_deg=90.0,
                    fine_pose_radius_deg=15.0,
                    fine_pose_step_deg=15.0,
                ),
                photometric_config=PhotometricConfig(
                    light_direction=tuple(observation["light_direction"]),
                    albedo=observation["albedo"],
                    ambient=observation["ambient"],
                    exposure=observation["exposure"],
                ),
            )
            candidate = report["meshes"][0]
            self.assertEqual(report["schema_version"], 2)
            self.assertIn("silhouette_score", candidate)
            self.assertIn("photometric_score", candidate)
            self.assertGreater(candidate["photometric_valid_pixels"], 0)
            self.assertGreater(candidate["photometric_coverage"], 0.8)
            self.assertAlmostEqual(
                candidate["score"],
                candidate["silhouette_score"]
                + 0.5 * candidate["photometric_score"],
            )

    def test_matcher_reports_thermal_score_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            manifest = generate_benchmark(output, seed=7, candidates=2, resolution=64)
            observation = manifest["observations"]
            thermal = observation["thermal"]
            report = match_files(
                output / observation["optical"],
                [output / manifest["candidates"][0]["path"]],
                mask_path=output / observation["silhouette"],
                output_path=output / "thermal_report.json",
                config=OutlineMatcherConfig(
                    coarse_yaw_step_deg=90.0,
                    coarse_pitch_angles_deg=(-45.0, 0.0, 45.0),
                    coarse_roll_step_deg=90.0,
                    fine_pose_radius_deg=15.0,
                    fine_pose_step_deg=15.0,
                ),
                thermal_config=ThermalConfig(
                    observed_flux_w_m2=thermal["bolometric_flux_w_m2"],
                    sun_direction=tuple(thermal["sun_direction"]),
                    observer_direction=tuple(thermal["observer_direction"]),
                    bond_albedo=thermal["bond_albedo"],
                    solar_flux_w_m2_at_1au=thermal["solar_flux_w_m2_at_1au"],
                    heliocentric_distance_au=thermal["heliocentric_distance_au"],
                    observer_distance_m=thermal["observer_distance_m"],
                    mesh_unit_m=thermal["mesh_unit_m"],
                    wavelength_um=thermal["wavelength_um"],
                ),
                thermal_image_path=output / thermal["radiance_image"],
            )
            candidate = report["meshes"][0]
            self.assertEqual(report["schema_version"], 3)
            self.assertIn("thermal_score", candidate)
            self.assertIn("thermal_image_score", candidate)
            self.assertGreater(candidate["thermal_image_coverage"], 0.8)
            self.assertGreater(candidate["predicted_thermal_flux_w_m2"], 0.0)
            self.assertAlmostEqual(
                candidate["thermal_score"],
                0.5 * candidate["thermal_flux_score"]
                + 0.5 * candidate["thermal_image_score"],
            )
            self.assertAlmostEqual(
                candidate["score"],
                candidate["silhouette_score"] + 0.1 * candidate["thermal_score"],
            )

    def test_thermal_score_is_zero_for_matching_flux(self) -> None:
        mesh = make_asteroid(9)
        flux = bolometric_thermal_flux(
            mesh,
            sun_direction=(0.2, 0.3, 0.9),
            observer_distance_m=1000.0,
        )
        metrics = thermal_flux_metrics(
            mesh,
            ThermalConfig(
                observed_flux_w_m2=flux,
                sun_direction=(0.2, 0.3, 0.9),
                observer_distance_m=1000.0,
            ),
        )
        self.assertAlmostEqual(metrics["thermal_score"], 0.0, places=12)

    def test_thermal_image_rewards_matching_radiance_pattern(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        normals = np.zeros((2, 2, 3), dtype=np.float64)
        normals[..., 2] = 1.0
        config = ThermalConfig(
            observed_flux_w_m2=1.0,
            sun_direction=(0.0, 0.0, 1.0),
        )
        observed = np.full(
            (2, 2),
            (1.0 - config.bond_albedo) * config.solar_flux_w_m2_at_1au / np.pi,
        )
        matching = thermal_image_metrics(observed, mask, normals, mask, config)
        wrong_normals = normals.copy()
        wrong_normals[0, 0] = (1.0, 0.0, 0.0)
        wrong = thermal_image_metrics(observed, mask, wrong_normals, mask, config)
        self.assertAlmostEqual(matching["thermal_image_score"], 0.0)
        self.assertAlmostEqual(matching["thermal_image_coverage"], 1.0)
        self.assertGreater(wrong["thermal_image_score"], matching["thermal_image_score"])

    def test_matcher_reports_alignment_and_ranks_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            generate_benchmark(output, seed=7, candidates=2, resolution=64)
            report = match_files(
                output / "observations" / "optical.png",
                sorted((output / "candidates").glob("*.npz")),
                mask_path=output / "observations" / "silhouette.png",
                output_path=output / "report.json",
                overlay_dir=output / "overlays",
                config=OutlineMatcherConfig(
                    coarse_yaw_step_deg=90.0,
                    coarse_pitch_angles_deg=(-45.0, 0.0, 45.0),
                    coarse_roll_step_deg=90.0,
                    fine_pose_radius_deg=15.0,
                    fine_pose_step_deg=15.0,
                ),
            )
            self.assertEqual(len(report["meshes"]), 2)
            self.assertIn("translation_px", report["meshes"][0])
            self.assertIn("search_score_margin", report["meshes"][0])
            self.assertIn("final_alignment_margin", report["meshes"][0])
            self.assertTrue((output / "overlays").exists())
            self.assertLess(report["meshes"][0]["score"], report["meshes"][-1]["score"])

    def test_matcher_recovers_deliberate_translation_and_scale(self) -> None:
        mesh = make_asteroid(13)
        observed = render_silhouette(
            mesh,
            (80, 80),
            roll_deg=20.0,
            scale_px_per_unit=30.0,
            translation_px=(7.0, -5.0),
        )
        result = match_mesh_outline(
            mesh,
            observed,
            config=OutlineMatcherConfig(
                coarse_yaw_step_deg=90.0,
                coarse_pitch_angles_deg=(-45.0, 0.0, 45.0),
                coarse_roll_step_deg=90.0,
                fine_pose_radius_deg=15.0,
                fine_pose_step_deg=15.0,
            ),
        )
        # A silhouette alone can admit multiple 3D poses. The test therefore
        # checks registration quality and ambiguity reporting, not recovery of
        # one arbitrary Euler-angle representation.
        self.assertLess(result["score"], 0.04)
        self.assertLess(abs(result["translation_px"][0] - 7.0), 4.5)
        self.assertLess(abs(result["translation_px"][1] + 5.0), 4.5)
        self.assertIn("translation_seed_px", result)
        self.assertIn("translation_seed_score", result)
        self.assertGreaterEqual(result["final_alignment_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
