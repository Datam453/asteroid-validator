"""CPU engineering demo: procedural objects, not generator validation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from .dataset import write_json, build_dataset
from .physics.synthetic import Mesh, make_asteroid, perturb_mesh, bolometric_thermal_flux, equilibrium_thermal_radiance
from .physics.outline import render_normal_map


def generate_case(root: Path, seed: int, candidates: int, resolution: int) -> dict:
    if candidates < 2 or resolution < 16:
        raise ValueError("demo needs at least two candidates and resolution >= 16")
    rng = np.random.default_rng(seed)
    mesh = make_asteroid(seed, latitude_bands=6, longitude_bands=12, roughness=float(rng.uniform(.04, .22)))
    mesh = Mesh(mesh.vertices * rng.uniform(.65, 1.4, size=3), mesh.faces)
    for name in ("observations", "candidates", "ground_truth"):
        (root / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "ground_truth/asteroid.npz", vertices=mesh.vertices, faces=mesh.faces)
    # Projection is observing metadata supplied to every candidate, not fitted using truth at scoring time.
    projection = {"yaw_deg": 0., "pitch_deg": 0., "roll_deg": 0.,
                  "scale_px_per_unit": (resolution - 1) / 4., "translation_px": [0., 0.]}
    normals, mask = render_normal_map(mesh, (resolution, resolution), **projection)
    light = np.array([rng.uniform(-.5, .5), rng.uniform(.15, .6), 1.])
    light /= np.linalg.norm(light)
    albedo, ambient, exposure = .62, .06, .92
    mu = np.maximum(0, normals @ light)
    image = np.where(mask, exposure * (ambient + albedo * mu), 0.)
    image = np.clip(image + rng.normal(0, .004, size=image.shape) * mask, 0, 1)
    heat = {"sun_direction": light.tolist(), "observer_direction": [0., 0., 1.],
            "bond_albedo": .08, "solar_flux_w_m2_at_1au": 1361., "heliocentric_distance_au": 1.2,
            "observer_distance_m": 20000., "mesh_unit_m": 500., "wavelength_um": 10.}
    flux_keys = {key: heat[key] for key in heat if key != "wavelength_um"}
    flux = bolometric_thermal_flux(mesh, **flux_keys)
    radiance = equilibrium_thermal_radiance(mu, bond_albedo=heat["bond_albedo"],
        solar_flux_w_m2_at_1au=heat["solar_flux_w_m2_at_1au"],
        heliocentric_distance_au=heat["heliocentric_distance_au"], wavelength_um=heat["wavelength_um"])
    radiance[~mask] = 0
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(root / "observations/optical.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(root / "observations/silhouette.png")
    Image.fromarray(np.rint(radiance / max(float(radiance.max()), 1e-12) * 255).astype(np.uint8)).save(
        root / "observations/infrared_10um_preview.png")
    np.save(root / "observations/infrared_10um.npy", radiance)
    rows = []
    for i in range(candidates):
        severity = float(rng.uniform(.25, 3.0))
        candidate = perturb_mesh(mesh, rng, severity)
        candidate_id = f"candidate_{i:03d}"
        filename = f"candidates/{candidate_id}.npz"
        np.savez_compressed(root / filename, vertices=candidate.vertices, faces=candidate.faces)
        rows.append({"id": candidate_id, "path": filename})
    manifest = {"schema_version": 1, "generator": "procedural_perturbation_engineering_demo",
                "alignment_policy": "supplied-camera", "ground_truth_mesh": "ground_truth/asteroid.npz",
                "seed": seed, "candidates": rows,
                "observations": {"optical": "observations/optical.png", "silhouette": "observations/silhouette.png",
                    "projection": projection, "light_direction": light.tolist(), "albedo": albedo,
                    "ambient": ambient, "exposure": exposure, "thermal": {**heat,
                        "bolometric_flux_w_m2": flux, "radiance_image": "observations/infrared_10um.npy"}}}
    write_json(root / "manifest.json", manifest)
    return manifest


def demo_dataset(output: Path, *, train_objects: int = 16, validation_objects: int = 4,
                 test_objects: int = 4, candidates: int = 4, resolution: int = 64,
                 samples: int = 512, seed: int = 2026) -> Path:
    if min(train_objects, validation_objects, test_objects) < 1:
        raise ValueError("every demo split needs at least one object")
    output = Path(output).resolve()
    cases, index = [], 0
    for split, count in (("train", train_objects), ("validation", validation_objects), ("test", test_objects)):
        for _ in range(count):
            object_id = f"synthetic_{seed + index}"
            root = output / "cases" / object_id
            generate_case(root, seed + index, candidates, resolution)
            cases.append({"asteroid_id": object_id, "split": split,
                          "manifest": f"cases/{object_id}/manifest.json"})
            index += 1
    write_json(output / "cases.json", {"cases": cases,
        "scope": "Procedural same-family smoke test; not evidence about real generators or mission transfer."})
    build_dataset(output / "cases.json", output / "dataset.json", samples=samples, seed=seed)
    return output / "dataset.json"
