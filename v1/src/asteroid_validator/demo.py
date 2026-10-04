"""Run a small, reproducible end-to-end asteroid ranking demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import calibrate_weights, load_calibration_case, spearman_rank_correlation
from .outline import OutlineMatcherConfig, PhotometricConfig, ThermalConfig, match_files
from .synthetic import generate_benchmark
from .viewer import build_viewer


def _score_case(
    case_dir: Path,
    manifest: dict,
    *,
    weights: dict[str, float],
    matcher_config: OutlineMatcherConfig,
    report_name: str,
) -> tuple[dict, Path]:
    observation = manifest["observations"]
    thermal = observation["thermal"]
    report_path = case_dir / report_name
    config = OutlineMatcherConfig(
        **{
            **matcher_config.__dict__,
            "silhouette_term_weight": weights["silhouette_term_weight"],
        }
    )
    report = match_files(
        case_dir / observation["optical"],
        [case_dir / item["path"] for item in manifest["candidates"]],
        mask_path=case_dir / observation["silhouette"],
        output_path=report_path,
        overlay_dir=case_dir / "overlays",
        config=config,
        photometric_config=PhotometricConfig(
            light_direction=tuple(observation["light_direction"]),
            albedo=observation["albedo"],
            ambient=observation["ambient"],
            exposure=observation["exposure"],
            weight=weights["photometric_weight"],
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
            weight=weights["thermal_weight"],
        ),
        thermal_image_path=case_dir / thermal["radiance_image"],
    )
    return report, report_path


def _evaluate(manifest: dict, report: dict) -> dict:
    truth = {
        item["id"]: item["benchmark_only_vertex_rmse"]
        for item in manifest["candidates"]
    }
    rows = [
        (Path(item["mesh_path"]).stem, item["score"])
        for item in report["meshes"]
    ]
    predicted_order = [name for name, _ in rows]
    truth_order = sorted(truth, key=truth.get)
    scores = np.asarray([score for _, score in rows])
    errors = np.asarray([truth[name] for name, _ in rows])
    return {
        "predicted_order": predicted_order,
        "ground_truth_order": truth_order,
        "top1_correct": predicted_order[0] == truth_order[0],
        "spearman": spearman_rank_correlation(scores, errors),
        "candidate_vertex_rmse": truth,
    }


def run_demo(
    output: Path,
    *,
    train_seeds: tuple[int, ...] = (7, 11, 19),
    test_seeds: tuple[int, ...] = (13, 17),
    candidates: int = 4,
    resolution: int = 256,
) -> dict:
    """Generate distinct cases, fit weights, and score held-out shapes."""

    if not train_seeds or not test_seeds or len(set(train_seeds + test_seeds)) != len(train_seeds + test_seeds):
        raise ValueError("training and held-out seeds must be nonempty and distinct")
    if candidates < 3:
        raise ValueError("at least three candidates are needed for rank calibration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matcher_config = OutlineMatcherConfig(
        coarse_yaw_step_deg=120.0,
        coarse_pitch_angles_deg=(-45.0, 0.0, 45.0),
        coarse_roll_step_deg=120.0,
        fine_pose_radius_deg=15.0,
        fine_pose_step_deg=15.0,
        translation_seed_fractions=(-0.125, 0.0, 0.125),
        final_rerank_top_k=12,
    )
    initial_weights = {
        "silhouette_term_weight": 1.0,
        "photometric_weight": 0.5,
        "thermal_weight": 0.1,
    }
    training_cases = []
    calibration_cases = []
    for seed in train_seeds:
        case_dir = output / "train" / f"seed_{seed}"
        manifest = generate_benchmark(
            case_dir, seed=seed, candidates=candidates, resolution=resolution
        )
        report, report_path = _score_case(
            case_dir,
            manifest,
            weights=initial_weights,
            matcher_config=matcher_config,
            report_name="ranking.json",
        )
        training_cases.append({"seed": seed, "report": str(report_path), **_evaluate(manifest, report)})
        calibration_cases.append(load_calibration_case(case_dir / "manifest.json", report_path))
    calibration = calibrate_weights(calibration_cases)
    raw_weights = calibration["cli_weights"]
    total = sum(raw_weights.values())
    normalized_weights = {key: value / total for key, value in raw_weights.items()}
    calibration["sum_to_one_weights"] = normalized_weights
    (output / "weights.json").write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )

    held_out_cases = []
    for seed in test_seeds:
        case_dir = output / "held_out" / f"seed_{seed}"
        manifest = generate_benchmark(
            case_dir, seed=seed, candidates=candidates, resolution=resolution
        )
        report, report_path = _score_case(
            case_dir,
            manifest,
            weights=normalized_weights,
            matcher_config=matcher_config,
            report_name="ranking.json",
        )
        viewer_path = case_dir / "viewer.html"
        build_viewer(report_path, viewer_path)
        held_out_cases.append(
            {
                "seed": seed,
                "report": str(report_path),
                "viewer": str(viewer_path),
                "thermal_observation": str(
                    case_dir / manifest["observations"]["thermal"]["preview_image"]
                ),
                **_evaluate(manifest, report),
            }
        )
    summary = {
        "schema_version": 1,
        "candidate_source": "procedural perturbations of each reference mesh",
        "reconstruction_models_run": [],
        "training_seeds": list(train_seeds),
        "held_out_seeds": list(test_seeds),
        "candidates_per_case": candidates,
        "resolution": resolution,
        "weights_sum_to_one": normalized_weights,
        "training_cases": training_cases,
        "held_out_cases": held_out_cases,
        "held_out_top1_accuracy": float(np.mean([case["top1_correct"] for case in held_out_cases])),
        "held_out_mean_spearman": float(np.mean([case["spearman"] for case in held_out_cases])),
        "caveat": (
            "This demonstrates the complete scoring workflow with synthetic candidate "
            "meshes and a 10-micrometre equilibrium IR proxy. It does not establish performance "
            "on TRELLIS, Hunyuan3D, Ouroboros3D, or real instrument images."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Asteroid ranking meeting demo",
        "",
        f"This run generated {len(train_seeds) + len(test_seeds)} different reference asteroids. Candidate meshes are",
        "procedural perturbations of each reference, not outputs from TRELLIS,",
        "Hunyuan3D, or Ouroboros3D.",
        "",
        "## Pipeline",
        "",
        "Reference mesh → optical image, silhouette, thermal image and flux →",
        "candidate ranking without ground truth → weight fitting on the training",
        "asteroids → locked weights on held-out asteroids → ground-truth check.",
        "",
        "## Held-out results",
        "",
        f"- Top-1 accuracy: {summary['held_out_top1_accuracy']:.0%}",
        f"- Mean Spearman rank correlation: {summary['held_out_mean_spearman']:.3f}",
        "- Weights (sum to 1): "
        + ", ".join(f"{key}={value:.3f}" for key, value in normalized_weights.items()),
        "",
        "| Held-out seed | Predicted order | Ground-truth order | Top-1 correct | Spearman |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for case in held_out_cases:
        lines.append(
            f"| {case['seed']} | {', '.join(case['predicted_order'])} | "
            f"{', '.join(case['ground_truth_order'])} | "
            f"{case['top1_correct']} | {case['spearman']:.3f} |"
        )
    first_seed = test_seeds[0]
    lines.extend([
        "",
        "## Show these files",
        "",
        f"- [Input optical image](held_out/seed_{first_seed}/observations/optical.png)",
        f"- [10 µm IR preview](held_out/seed_{first_seed}/observations/infrared_10um_preview.png)",
        f"- [Interactive candidate viewer](held_out/seed_{first_seed}/viewer.html)",
        f"- [Full candidate scores](held_out/seed_{first_seed}/ranking.json)",
        "- [Learned weights](weights.json)",
        "",
        "The IR preview displays physical spectral radiance stored",
        f"in `held_out/seed_{first_seed}/observations/infrared_10um.npy`. The scorer reads that array.",
        "This is a monochromatic 10 µm equilibrium proxy, not a calibrated IR-camera simulation.",
        "",
        "## What this proves",
        "",
        "The code connects generation, observations, three scoring terms, calibration,",
        "a held-out ranking, and ground-truth evaluation in one reproducible run.",
        "This small synthetic sample is too small to establish scientific accuracy.",
        "",
        "## A short way to present it",
        "",
        "I start from known asteroid shapes and render optical, silhouette, and",
        "10 µm IR observations. The ranking code sees those observations and four",
        "candidate meshes per case, but not the true shape. It combines contour,",
        "brightness/normal, and thermal evidence. I fit non-negative weights on",
        "three cases and freeze them before testing two new shapes. The held-out",
        "result is one correct best-mesh pick out of two, with mean rank correlation",
        f"{summary['held_out_mean_spearman']:.3f}. This is a working evaluation harness, not a claim",
        "that any named 3D generator has been validated yet.",
        "",
        "## Next milestone",
        "",
        "Run an accessible reconstruction model on held-out renders, export its",
        "meshes in OBJ or NPZ, and replace the procedural candidates. Then expand",
        "to many independent shapes, views, lighting conditions, and realistic",
        "instrument-specific IR observations before making performance claims.",
        "",
    ])
    (output / "MEETING_DEMO.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark_runs/meeting_demo_5case_ir_256"))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[13, 17])
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args(argv)
    summary = run_demo(
        args.output,
        train_seeds=tuple(args.train_seeds),
        test_seeds=tuple(args.test_seeds),
        candidates=args.candidates,
        resolution=args.resolution,
    )
    print(args.output / "MEETING_DEMO.md")
    print(
        f"Held-out top-1 accuracy: {summary['held_out_top1_accuracy']:.0%}; "
        f"mean Spearman: {summary['held_out_mean_spearman']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
