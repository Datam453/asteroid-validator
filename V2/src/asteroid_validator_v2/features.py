"""Observation-only extraction; evaluator parameters never choose the pose."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from .geometry import load_mesh
from .model import FEATURES, FEATURE_VERSION
from .physics.synthetic import Mesh
from .physics.outline import (
    OutlineMatcherConfig, PhotometricConfig, ThermalConfig, load_grayscale_image,
    load_mask, match_mesh_outline, render_normal_map, outline_metrics,
    photometric_normal_metrics, thermal_flux_metrics, thermal_image_metrics,
    _project_vertices, _save_overlay,
)


def extract_case(manifest_path: Path, output_dir: Path | None = None) -> dict:
    """Read only observations/candidates from the manifest, never reference data.

    Geometry is in the declared mesh frame. Supplied-camera mode consumes camera
    metadata; outline-fixed mode chooses pose using silhouette only. Both freeze
    their extraction independently of trained regression weights.
    """
    manifest_path = Path(manifest_path).resolve()
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, obs = manifest_path.parent, spec["observations"]
    image = load_grayscale_image(base / obs["optical"])
    mask = load_mask(base / obs["silhouette"])
    if image.shape != mask.shape or not np.any(mask):
        raise ValueError("reviewed mask must be nonempty and match the optical image")
    thermal = obs["thermal"]
    radiance = np.load(base / thermal["radiance_image"], allow_pickle=False)
    photo = PhotometricConfig(tuple(obs["light_direction"]), obs["albedo"],
                              obs.get("ambient", 0), obs.get("exposure", 1), weight=0)
    heat = ThermalConfig(
        thermal["bolometric_flux_w_m2"], tuple(thermal["sun_direction"]),
        tuple(thermal.get("observer_direction", [0, 0, 1])),
        bond_albedo=thermal["bond_albedo"],
        solar_flux_w_m2_at_1au=thermal["solar_flux_w_m2_at_1au"],
        heliocentric_distance_au=thermal["heliocentric_distance_au"],
        observer_distance_m=thermal["observer_distance_m"],
        mesh_unit_m=thermal["mesh_unit_m"], wavelength_um=thermal.get("wavelength_um"), weight=0)
    policy = spec.get("alignment_policy", "outline-fixed")
    if policy not in ("supplied-camera", "outline-fixed"):
        raise ValueError("alignment_policy must be supplied-camera or outline-fixed")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    config = OutlineMatcherConfig(coarse_yaw_step_deg=120, coarse_pitch_angles_deg=(-45., 0., 45.),
        coarse_roll_step_deg=120, fine_pose_radius_deg=0, fine_pose_step_deg=15,
        translation_seed_fractions=(-.125, 0., .125), final_rerank_top_k=1)
    for item in spec["candidates"]:
        candidate_id = item["id"]
        try:
            mesh = load_mesh(base / item["path"])
            if policy == "supplied-camera":
                projection = obs["projection"]
                pose = dict(yaw_deg=float(projection.get("yaw_deg", 0)),
                            pitch_deg=float(projection.get("pitch_deg", 0)),
                            roll_deg=float(projection.get("roll_deg", 0)),
                            scale_px_per_unit=float(projection["scale_px_per_unit"]),
                            translation_px=tuple(projection.get("translation_px", [0, 0])))
                if not all(np.isfinite(v) for k, v in pose.items() if k != "translation_px"):
                    raise ValueError("camera projection must be finite")
                if pose["scale_px_per_unit"] <= 0 or not np.isfinite(pose["translation_px"]).all():
                    raise ValueError("camera scale must be positive and translation finite")
            else:
                aligned = match_mesh_outline(mesh, mask, config=config)
                pose = {key: aligned[key] for key in ("yaw_deg", "pitch_deg", "roll_deg",
                                                      "scale_px_per_unit", "translation_px")}
            normals, predicted_mask = render_normal_map(mesh, image.shape, **pose)
            silhouette = outline_metrics(mask, predicted_mask)
            photometry = photometric_normal_metrics(image, mask, normals, predicted_mask, photo)
            posed = Mesh(_project_vertices(mesh.vertices, pose["yaw_deg"], pose["pitch_deg"],
                                           pose["roll_deg"]), mesh.faces)
            thermal_scores = thermal_flux_metrics(posed, heat)
            thermal_scores.update(thermal_image_metrics(radiance, mask, normals, predicted_mask, heat))
            thermal_scores["thermal_score"] = .5 * (thermal_scores["thermal_flux_score"] +
                                                    thermal_scores["thermal_image_score"])
            values = [silhouette["score"], photometry["photometric_score"], thermal_scores["thermal_score"]]
            if not np.isfinite(values).all():
                raise ValueError("candidate has no usable finite observation consistency")
            row = {"candidate_id": candidate_id, "mesh_path": str((base / item["path"]).resolve()),
                   **dict(zip(FEATURES, values)), "alignment": pose,
                   "diagnostics": {"iou": silhouette["iou"], **photometry, **thermal_scores}}
            if output_dir:
                path = Path(output_dir) / f"{Path(candidate_id).name}_overlay.png"
                _save_overlay(path, mask, predicted_mask)
                row["overlay_path"] = str(path.resolve())
            records.append(row)
        except (ValueError, OSError, KeyError) as error:
            failures.append({"candidate_id": candidate_id, "error": str(error)})
    return {"schema_version": 1, "feature_version": FEATURE_VERSION, "alignment_policy": policy,
            "extraction_config": asdict(config) if policy == "outline-fixed" else obs["projection"],
            "manifest": str(manifest_path), "records": records, "failures": failures,
            "reference_geometry_used_for_features": False}
