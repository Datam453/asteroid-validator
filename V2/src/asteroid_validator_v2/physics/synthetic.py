"""Generate deterministic synthetic asteroid reconstruction benchmarks.

The renderer is intentionally small and dependency-light. It is not the final
physics validator. Its job is to provide controlled observations and known
geometry for the first ranking experiments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def _unit_vector(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(vector), dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("direction vectors must be non-zero")
    return value / norm


def make_asteroid(
    seed: int,
    *,
    latitude_bands: int = 18,
    longitude_bands: int = 36,
    roughness: float = 0.08,
) -> Mesh:
    """Create an irregular ellipsoid with a deterministic low-frequency surface."""

    if latitude_bands < 4 or longitude_bands < 8:
        raise ValueError("the mesh resolution is too small")

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=4)
    axes = np.array([1.10, 0.86, 0.72], dtype=np.float64)
    vertices: list[list[float]] = []

    for latitude in range(latitude_bands + 1):
        theta = np.pi * latitude / latitude_bands
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for longitude in range(longitude_bands):
            phi = 2 * np.pi * longitude / longitude_bands
            radial = 1.0 + roughness * (
                0.55 * np.sin(3 * phi + phase[0]) * sin_theta**2
                + 0.28 * np.cos(2 * theta + phase[1])
                + 0.17 * np.sin(5 * phi + theta + phase[2]) * sin_theta
            )
            direction = np.array(
                [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
            )
            vertices.append((axes * radial * direction).tolist())

    faces: list[list[int]] = []
    for latitude in range(latitude_bands):
        for longitude in range(longitude_bands):
            next_longitude = (longitude + 1) % longitude_bands
            a = latitude * longitude_bands + longitude
            b = latitude * longitude_bands + next_longitude
            c = (latitude + 1) * longitude_bands + longitude
            d = (latitude + 1) * longitude_bands + next_longitude
            faces.extend(([a, b, c], [b, d, c]))

    mesh = Mesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
    )
    return _orient_faces_outward(mesh)


def _orient_faces_outward(mesh: Mesh) -> Mesh:
    vertices = mesh.vertices
    faces = mesh.faces.copy()
    for index, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        center = (v0 + v1 + v2) / 3.0
        if np.dot(normal, center) < 0:
            faces[index] = [face[0], face[2], face[1]]
    return Mesh(vertices=vertices, faces=faces)


def perturb_mesh(mesh: Mesh, rng: np.random.Generator, severity: float) -> Mesh:
    """Create a candidate with controlled scale and surface-shape errors."""

    vertices = mesh.vertices.copy()
    scale = 1.0 + rng.normal(0.0, 0.045 * severity, size=3)
    vertices *= scale

    radii = np.linalg.norm(vertices, axis=1)
    directions = vertices / np.maximum(radii[:, None], 1e-12)
    radial_error = rng.normal(0.0, 0.035 * severity, size=len(vertices))
    vertices += directions * (radii * radial_error)[:, None]
    return _orient_faces_outward(Mesh(vertices=vertices, faces=mesh.faces.copy()))


def _face_normals(mesh: Mesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    return cross / np.maximum(lengths, 1e-12)


def bolometric_thermal_flux(
    mesh: Mesh,
    *,
    sun_direction: Iterable[float],
    observer_direction: Iterable[float] = (0.0, 0.0, 1.0),
    bond_albedo: float = 0.08,
    solar_flux_w_m2_at_1au: float = 1361.0,
    heliocentric_distance_au: float = 1.0,
    observer_distance_m: float,
    mesh_unit_m: float = 1.0,
) -> float:
    """Return disk-integrated bolometric thermal flux in W/m^2.

    This is the inexpensive equilibrium model from the presentation: each
    facet instantly reradiates all absorbed sunlight as a diffuse emitter.
    It intentionally omits thermal inertia, rotation, roughness, self-heating,
    and wavelength-dependent temperature fitting.
    """

    if not 0.0 <= bond_albedo < 1.0:
        raise ValueError("bond_albedo must be in [0, 1)")
    for name, value in (
        ("solar_flux_w_m2_at_1au", solar_flux_w_m2_at_1au),
        ("heliocentric_distance_au", heliocentric_distance_au),
        ("observer_distance_m", observer_distance_m),
        ("mesh_unit_m", mesh_unit_m),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")

    sun = _unit_vector(sun_direction)
    observer = _unit_vector(observer_direction)
    triangles = mesh.vertices[mesh.faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_areas = np.linalg.norm(area_vectors, axis=1)
    normals = area_vectors / np.maximum(double_areas[:, None], 1e-12)
    areas_m2 = 0.5 * double_areas * mesh_unit_m**2
    mu_sun = np.maximum(0.0, normals @ sun)
    mu_observer = np.maximum(0.0, normals @ observer)
    incident_flux = solar_flux_w_m2_at_1au / heliocentric_distance_au**2
    absorbed_power = (1.0 - bond_albedo) * incident_flux * mu_sun * areas_m2
    return float(
        np.sum(absorbed_power * mu_observer)
        / (np.pi * observer_distance_m**2)
    )


def equilibrium_thermal_radiance(
    mu_sun: np.ndarray | float,
    *,
    bond_albedo: float,
    solar_flux_w_m2_at_1au: float,
    heliocentric_distance_au: float,
    wavelength_um: float | None = None,
) -> np.ndarray:
    """Surface radiance: bolometric, or spectral at one IR wavelength.

    Unit emissivity and zero thermal inertia are assumed. Spectral output is
    W m^-2 sr^-1 um^-1; bolometric output is W m^-2 sr^-1.
    """

    if wavelength_um is not None and (not np.isfinite(wavelength_um) or wavelength_um <= 0):
        raise ValueError("wavelength_um must be positive and finite")
    absorbed = (
        (1.0 - bond_albedo)
        * solar_flux_w_m2_at_1au
        / heliocentric_distance_au**2
        * np.maximum(0.0, np.asarray(mu_sun, dtype=np.float64))
    )
    if wavelength_um is None:
        return absorbed / np.pi
    stefan_boltzmann = 5.670374419e-8
    planck = 6.62607015e-34
    light_speed = 299792458.0
    boltzmann = 1.380649e-23
    wavelength_m = wavelength_um * 1e-6
    temperature = np.power(absorbed / stefan_boltzmann, 0.25)
    exponent = planck * light_speed / (
        wavelength_m * boltzmann * np.maximum(temperature, 1e-12)
    )
    spectral_per_m = 2.0 * planck * light_speed**2 / wavelength_m**5
    return np.where(
        temperature > 0.0,
        spectral_per_m / np.expm1(np.minimum(exponent, 700.0)) * 1e-6,
        0.0,
    )


def render_orthographic(
    mesh: Mesh,
    *,
    resolution: int,
    light_direction: Iterable[float],
    albedo: float = 0.62,
    ambient: float = 0.06,
    exposure: float = 1.0,
    noise_sigma: float = 0.0,
    seed: int = 0,
    thermal_bond_albedo: float | None = None,
    thermal_wavelength_um: float | None = None,
    solar_flux_w_m2_at_1au: float = 1361.0,
    heliocentric_distance_au: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a grayscale Lambertian image and a binary silhouette.

    The camera is orthographic at positive z and looks toward the origin. A
    small face-level z-buffer is sufficient for this controlled benchmark.
    """

    if resolution < 16:
        raise ValueError("resolution must be at least 16 pixels")

    light = _unit_vector(light_direction)
    image = np.zeros((resolution, resolution), dtype=np.float64)
    depth = np.full((resolution, resolution), -np.inf, dtype=np.float64)
    silhouette = np.zeros((resolution, resolution), dtype=bool)
    thermal_radiance = np.zeros((resolution, resolution), dtype=np.float64)
    if thermal_bond_albedo is not None:
        if not 0.0 <= thermal_bond_albedo < 1.0:
            raise ValueError("thermal_bond_albedo must be in [0, 1)")
        if solar_flux_w_m2_at_1au <= 0 or heliocentric_distance_au <= 0:
            raise ValueError("solar flux and heliocentric distance must be positive")
    normals = _face_normals(mesh)
    extent = max(float(np.max(np.abs(mesh.vertices[:, :2]))), 1e-9) * 1.08

    for face, normal in zip(mesh.faces, normals):
        triangle = mesh.vertices[face]
        screen_x = (triangle[:, 0] / extent + 1.0) * 0.5 * (resolution - 1)
        screen_y = (-triangle[:, 1] / extent + 1.0) * 0.5 * (resolution - 1)
        min_x = max(0, int(np.floor(np.min(screen_x))))
        max_x = min(resolution - 1, int(np.ceil(np.max(screen_x))))
        min_y = max(0, int(np.floor(np.min(screen_y))))
        max_y = min(resolution - 1, int(np.ceil(np.max(screen_y))))
        if min_x > max_x or min_y > max_y:
            continue

        denominator = (
            (screen_y[1] - screen_y[2]) * (screen_x[0] - screen_x[2])
            + (screen_x[2] - screen_x[1]) * (screen_y[0] - screen_y[2])
        )
        if abs(denominator) < 1e-12:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        bary_a = (
            (screen_y[1] - screen_y[2]) * (xx - screen_x[2])
            + (screen_x[2] - screen_x[1]) * (yy - screen_y[2])
        ) / denominator
        bary_b = (
            (screen_y[2] - screen_y[0]) * (xx - screen_x[2])
            + (screen_x[0] - screen_x[2]) * (yy - screen_y[2])
        ) / denominator
        bary_c = 1.0 - bary_a - bary_b
        inside = (bary_a >= -1e-7) & (bary_b >= -1e-7) & (bary_c >= -1e-7)
        if not np.any(inside):
            continue

        triangle_depth = (
            bary_a * triangle[0, 2] + bary_b * triangle[1, 2] + bary_c * triangle[2, 2]
        )
        current_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        visible = inside & (triangle_depth > current_depth)
        if not np.any(visible):
            continue

        brightness = ambient + albedo * max(0.0, float(np.dot(normal, light)))
        current_depth[visible] = triangle_depth[visible]
        depth[min_y : max_y + 1, min_x : max_x + 1] = current_depth
        current_image = image[min_y : max_y + 1, min_x : max_x + 1]
        current_image[visible] = brightness * exposure
        image[min_y : max_y + 1, min_x : max_x + 1] = current_image
        if thermal_bond_albedo is not None:
            radiance = equilibrium_thermal_radiance(
                float(np.dot(normal, light)),
                bond_albedo=thermal_bond_albedo,
                solar_flux_w_m2_at_1au=solar_flux_w_m2_at_1au,
                heliocentric_distance_au=heliocentric_distance_au,
                wavelength_um=thermal_wavelength_um,
            )
            current_thermal = thermal_radiance[min_y : max_y + 1, min_x : max_x + 1]
            current_thermal[visible] = radiance
            thermal_radiance[min_y : max_y + 1, min_x : max_x + 1] = current_thermal
        current_silhouette = silhouette[min_y : max_y + 1, min_x : max_x + 1]
        current_silhouette[visible] = True
        silhouette[min_y : max_y + 1, min_x : max_x + 1] = current_silhouette

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        image += rng.normal(0.0, noise_sigma, size=image.shape)
    image = np.clip(image, 0.0, 1.0)
    if thermal_bond_albedo is not None:
        return image, silhouette, thermal_radiance
    return image, silhouette


def _save_mesh(path: Path, mesh: Mesh) -> None:
    np.savez_compressed(path, vertices=mesh.vertices, faces=mesh.faces)


def generate_benchmark(
    output: Path,
    *,
    seed: int = 7,
    candidates: int = 8,
    resolution: int = 160,
) -> dict:
    """Generate one benchmark directory and return its manifest."""

    if candidates < 2:
        raise ValueError("at least two candidates are required")

    output = Path(output)
    ground_truth_dir = output / "ground_truth"
    candidates_dir = output / "candidates"
    observations_dir = output / "observations"
    for directory in (ground_truth_dir, candidates_dir, observations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    truth = make_asteroid(seed)
    _save_mesh(ground_truth_dir / "asteroid.npz", truth)

    light_direction = _unit_vector((0.35, 0.42, 0.84))
    thermal_parameters = {
        "model": "instantaneous_equilibrium_bolometric",
        "sun_direction": light_direction.tolist(),
        "observer_direction": [0.0, 0.0, 1.0],
        "bond_albedo": 0.08,
        "solar_flux_w_m2_at_1au": 1361.0,
        "heliocentric_distance_au": 1.2,
        "observer_distance_m": 20000.0,
        "mesh_unit_m": 500.0,
        "wavelength_um": 10.0,
    }
    thermal_flux_w_m2 = bolometric_thermal_flux(
        truth,
        sun_direction=thermal_parameters["sun_direction"],
        observer_direction=thermal_parameters["observer_direction"],
        bond_albedo=thermal_parameters["bond_albedo"],
        solar_flux_w_m2_at_1au=thermal_parameters["solar_flux_w_m2_at_1au"],
        heliocentric_distance_au=thermal_parameters["heliocentric_distance_au"],
        observer_distance_m=thermal_parameters["observer_distance_m"],
        mesh_unit_m=thermal_parameters["mesh_unit_m"],
    )
    image, silhouette, thermal_radiance = render_orthographic(
        truth,
        resolution=resolution,
        light_direction=light_direction,
        albedo=0.62,
        ambient=0.06,
        exposure=0.92,
        noise_sigma=0.008,
        seed=seed + 1,
        thermal_bond_albedo=thermal_parameters["bond_albedo"],
        thermal_wavelength_um=thermal_parameters["wavelength_um"],
        solar_flux_w_m2_at_1au=thermal_parameters["solar_flux_w_m2_at_1au"],
        heliocentric_distance_au=thermal_parameters["heliocentric_distance_au"],
    )
    Image.fromarray(np.rint(image * 255).astype(np.uint8), mode="L").save(
        observations_dir / "optical.png"
    )
    Image.fromarray((silhouette.astype(np.uint8) * 255), mode="L").save(
        observations_dir / "silhouette.png"
    )
    np.save(observations_dir / "infrared_10um.npy", thermal_radiance)
    preview_scale = max(float(np.max(thermal_radiance)), 1e-12)
    Image.fromarray(
        np.rint(255.0 * thermal_radiance / preview_scale).astype(np.uint8), mode="L"
    ).save(observations_dir / "infrared_10um_preview.png")

    candidate_records = []
    candidate_rng = np.random.default_rng(seed + 2)
    for index in range(candidates):
        severity = 0.35 + 0.18 * index
        candidate = perturb_mesh(truth, candidate_rng, severity)
        filename = f"candidate_{index:03d}.npz"
        _save_mesh(candidates_dir / filename, candidate)
        vertex_rmse = float(np.sqrt(np.mean((candidate.vertices - truth.vertices) ** 2)))
        candidate_records.append(
            {
                "id": filename.removesuffix(".npz"),
                "path": f"candidates/{filename}",
                "benchmark_only_vertex_rmse": vertex_rmse,
                "perturbation_severity": severity,
            }
        )

    manifest = {
        "schema_version": 2,
        "seed": seed,
        "resolution": resolution,
        "ground_truth_mesh": "ground_truth/asteroid.npz",
        "observations": {
            "optical": "observations/optical.png",
            "silhouette": "observations/silhouette.png",
            "camera": "orthographic_positive_z",
            "light_direction": light_direction.tolist(),
            "albedo": 0.62,
            "ambient": 0.06,
            "exposure": 0.92,
            "noise_sigma": 0.008,
            "thermal": {
                **thermal_parameters,
                "bolometric_flux_w_m2": thermal_flux_w_m2,
                "radiance_image": "observations/infrared_10um.npy",
                "preview_image": "observations/infrared_10um_preview.png",
                "radiance_units": "W m^-2 sr^-1 um^-1 (at 10 um)",
            },
        },
        "candidates": candidate_records,
        "ground_truth_metrics_are_for_evaluation_only": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate a synthetic benchmark")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--candidates", type=int, default=8)
    generate.add_argument("--resolution", type=int, default=160)
    args = parser.parse_args(argv)

    if args.command == "generate":
        manifest = generate_benchmark(
            args.output,
            seed=args.seed,
            candidates=args.candidates,
            resolution=args.resolution,
        )
        print(
            f"Generated {len(manifest['candidates'])} candidates in "
            f"{Path(args.output).resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
