"""Robust outline matching between an image and candidate 3D meshes.

The matcher treats image placement and camera pose as nuisance parameters. It
searches bounded scale/translation plus full yaw/pitch/roll pose, then reports
the fitted transform and both search and final score margins.

Current mesh formats: the project's ``.npz`` format and triangular ``.obj``.
Automatic segmentation remains a proposal, not ground truth: reports include
confidence and review warnings, and serious runs should supply a reviewed mask
or use the strict segmentation option.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .synthetic import Mesh, bolometric_thermal_flux, equilibrium_thermal_radiance


@dataclass(frozen=True)
class OutlineMatcherConfig:
    """Search and scoring controls, kept explicit for reproducibility."""

    coarse_yaw_step_deg: float = 60.0
    coarse_pitch_angles_deg: tuple[float, ...] = (-45.0, 0.0, 45.0)
    coarse_roll_step_deg: float = 60.0
    fine_pose_radius_deg: float = 15.0
    fine_pose_step_deg: float = 10.0
    scale_factors: tuple[float, ...] = (0.93, 1.0, 1.07)
    translation_fractions: tuple[float, ...] = (-0.06, 0.0, 0.06)
    # The centroid alignment is only a starting point.  This broader search
    # uses the observed contour distance field to propose a translation before
    # the normal local scale/translation search.  The fraction is relative to
    # the larger image dimension, not to the candidate mesh.
    translation_seed_fractions: tuple[float, ...] = (-0.25, -0.125, 0.0, 0.125, 0.25)
    fine_scale_factors: tuple[float, ...] = (0.97, 1.0, 1.03)
    fine_translation_pixels: tuple[int, ...] = (-4, 0, 4)
    max_mesh_area_fraction: float = 0.95
    chamfer_weight: float = 1.0
    iou_weight: float = 0.35
    area_weight: float = 0.15
    silhouette_term_weight: float = 1.0
    final_rerank_top_k: int = 32


@dataclass(frozen=True)
class PhotometricConfig:
    """Known Lambertian observation parameters for the photometric term.

    The observed image is modeled as
    ``exposure * (ambient + albedo * max(0, normal dot light))``.
    Supplying this configuration enables photometric scoring; omitting it
    preserves the original silhouette-only behavior.
    """

    light_direction: tuple[float, float, float]
    albedo: float
    ambient: float = 0.0
    exposure: float = 1.0
    weight: float = 0.5

    def __post_init__(self) -> None:
        light = np.asarray(self.light_direction, dtype=np.float64)
        if light.shape != (3,) or not np.isfinite(light).all():
            raise ValueError("light_direction must contain three finite values")
        if float(np.linalg.norm(light)) <= 1e-12:
            raise ValueError("light_direction must be non-zero")
        if not np.isfinite(self.albedo) or self.albedo <= 0:
            raise ValueError("albedo must be positive")
        if not np.isfinite(self.exposure) or self.exposure <= 0:
            raise ValueError("exposure must be positive")
        if not np.isfinite(self.ambient) or self.ambient < 0:
            raise ValueError("ambient must be non-negative")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("photometric weight must be non-negative")


@dataclass(frozen=True)
class ThermalConfig:
    """Parameters for the inexpensive equilibrium bolometric thermal term."""

    observed_flux_w_m2: float
    sun_direction: tuple[float, float, float]
    observer_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    bond_albedo: float = 0.08
    solar_flux_w_m2_at_1au: float = 1361.0
    heliocentric_distance_au: float = 1.0
    observer_distance_m: float = 1.0
    mesh_unit_m: float = 1.0
    wavelength_um: float | None = None
    weight: float = 0.1

    def __post_init__(self) -> None:
        for name, direction in (
            ("sun_direction", self.sun_direction),
            ("observer_direction", self.observer_direction),
        ):
            value = np.asarray(direction, dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain three finite values")
            if float(np.linalg.norm(value)) <= 1e-12:
                raise ValueError(f"{name} must be non-zero")
        if not np.isfinite(self.observed_flux_w_m2) or self.observed_flux_w_m2 <= 0:
            raise ValueError("observed_flux_w_m2 must be positive")
        if not 0.0 <= self.bond_albedo < 1.0:
            raise ValueError("bond_albedo must be in [0, 1)")
        for name in (
            "solar_flux_w_m2_at_1au",
            "heliocentric_distance_au",
            "observer_distance_m",
            "mesh_unit_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("thermal weight must be non-negative")
        if self.wavelength_um is not None and (
            not np.isfinite(self.wavelength_um) or self.wavelength_um <= 0
        ):
            raise ValueError("wavelength_um must be positive and finite")


def thermal_flux_metrics(mesh: Mesh, config: ThermalConfig) -> dict:
    """Compare a posed candidate's bolometric equilibrium flux to observation."""

    predicted = bolometric_thermal_flux(
        mesh,
        sun_direction=config.sun_direction,
        observer_direction=config.observer_direction,
        bond_albedo=config.bond_albedo,
        solar_flux_w_m2_at_1au=config.solar_flux_w_m2_at_1au,
        heliocentric_distance_au=config.heliocentric_distance_au,
        observer_distance_m=config.observer_distance_m,
        mesh_unit_m=config.mesh_unit_m,
    )
    absolute_error = abs(predicted - config.observed_flux_w_m2)
    relative_error = absolute_error / config.observed_flux_w_m2
    return {
        "predicted_thermal_flux_w_m2": float(predicted),
        "thermal_absolute_error_w_m2": float(absolute_error),
        "thermal_flux_score": float(relative_error),
        "thermal_score": float(relative_error),
    }


def thermal_image_metrics(
    observed_radiance: np.ndarray,
    observed_mask: np.ndarray,
    normal_map: np.ndarray,
    normal_valid: np.ndarray,
    config: ThermalConfig,
) -> dict:
    """Compare physical radiance images at the configured wavelength."""

    if observed_radiance.shape != observed_mask.shape:
        raise ValueError("thermal image and silhouette shapes must match")
    if not np.isfinite(observed_radiance).all() or np.any(observed_radiance < 0):
        raise ValueError("thermal radiance must be finite and non-negative")
    valid = observed_mask & normal_valid
    if not np.any(valid):
        return {"thermal_image_score": float("inf"), "thermal_image_coverage": 0.0}
    sun = np.asarray(config.sun_direction, dtype=np.float64)
    sun /= np.linalg.norm(sun)
    predicted = equilibrium_thermal_radiance(
        normal_map @ sun,
        bond_albedo=config.bond_albedo,
        solar_flux_w_m2_at_1au=config.solar_flux_w_m2_at_1au,
        heliocentric_distance_au=config.heliocentric_distance_au,
        wavelength_um=config.wavelength_um,
    )
    reference = max(float(np.mean(observed_radiance[observed_mask])), 1e-12)
    return {
        "thermal_image_score": float(
            np.mean(np.abs(predicted[valid] - observed_radiance[valid])) / reference
        ),
        "thermal_image_coverage": float(np.count_nonzero(valid) / np.count_nonzero(observed_mask)),
    }


def load_mesh(path: Path) -> Mesh:
    """Load a candidate mesh without changing its coordinate system."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as archive:
            vertices = np.asarray(archive["vertices"], dtype=np.float64)
            faces = np.asarray(archive["faces"], dtype=np.int32)
    elif suffix == ".obj":
        vertices, faces = _load_obj(path)
    else:
        raise ValueError(f"unsupported mesh format {suffix!r}; use .npz or .obj")

    _validate_mesh(vertices, faces, path)
    return _prepare_mesh(vertices, faces)


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f" and len(fields) >= 4:
            indices = []
            for field in fields[1:]:
                raw_index = int(field.split("/")[0])
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                indices.append(index)
            for index in range(1, len(indices) - 1):
                faces.append([indices[0], indices[index], indices[index + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path | None = None) -> None:
    label = f" in {path}" if path else ""
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"mesh must contain an Nx3 vertex array{label}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"mesh must contain an Mx3 triangular face array{label}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"mesh vertices contain non-finite values{label}")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError(f"mesh faces contain invalid vertex indices{label}")


def _prepare_mesh(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Remove export noise and normalize only translation, not physical scale."""

    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    centered_vertices = vertices - center
    triangles = centered_vertices[faces]
    area_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    keep = np.linalg.norm(area_vectors, axis=1) > 1e-12
    cleaned_faces = faces[keep].copy()
    cleaned_triangles = centered_vertices[cleaned_faces]
    cleaned_normals = np.cross(
        cleaned_triangles[:, 1] - cleaned_triangles[:, 0],
        cleaned_triangles[:, 2] - cleaned_triangles[:, 0],
    )
    face_centers = cleaned_triangles.mean(axis=1)
    outward = np.sum(cleaned_normals * face_centers, axis=1) >= 0.0
    cleaned_faces[~outward] = cleaned_faces[~outward][:, [0, 2, 1]]
    if len(cleaned_faces) == 0:
        raise ValueError("mesh has no non-degenerate triangular faces")
    return Mesh(vertices=centered_vertices, faces=cleaned_faces)


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    rgb = rgba[..., :3]
    # Luminance is preferable to a single channel for arbitrary RGB inputs.
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    alpha = rgba[..., 3]
    if np.any(alpha < 0.999):
        # Transparent pixels should behave as background.
        gray = np.where(alpha > 0, gray * alpha, 0.0)
    return gray


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = values > 0
    return _postprocess_mask(mask)


def segment_silhouette(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Extract one foreground component and report the segmentation decision.

    This automatic mode is intended for controlled images with a reasonably
    separable object/background. For real data, supplying a reviewed mask is
    preferable and is supported by the CLI.
    """

    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("the image must be a 2D array at least 8x8 pixels")

    height, width = image.shape
    border = np.concatenate(
        [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
    )
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    robust_sigma = max(1.4826 * mad, 0.005)
    threshold = _otsu_threshold(image)

    threshold_gap = max(4.0 * robust_sigma, 0.025)
    proposals = [
        ("bright", image > max(threshold, background + threshold_gap)),
        ("dark", image < min(threshold, background - threshold_gap)),
    ]
    # If Otsu is too close to a uniform background, use a background-relative
    # threshold as a fallback rather than returning a full-frame mask.
    proposals.extend(
        [
            ("bright_background_relative", image > background + threshold_gap),
            ("dark_background_relative", image < background - threshold_gap),
        ]
    )

    scored: list[tuple[float, str, np.ndarray, float, bool]] = []
    for polarity, proposal in proposals:
        component, touches_border = _largest_component(proposal, prefer_non_border=True)
        if not np.any(component):
            continue
        component = _postprocess_mask(component)
        area_fraction = float(component.mean())
        contrast = abs(float(image[component].mean()) - background)
        contrast_factor = min(1.0, contrast / max(4.0 * robust_sigma, 0.025))
        # Border-touching components are usually background. Keep them only as
        # a last resort, and penalize them heavily.
        quality = area_fraction * contrast_factor * (0.05 if touches_border else 1.0)
        scored.append((quality, polarity, component, area_fraction, touches_border))

    if not scored:
        raise ValueError("could not identify a foreground silhouette automatically")
    scored.sort(key=lambda item: item[0], reverse=True)
    _, polarity, mask, area_fraction, touches_border = scored[0]
    if area_fraction > 0.95:
        raise ValueError("automatic segmentation selected nearly the whole image; supply --mask")

    # Threshold stability is a useful warning signal. It is not a proof that
    # the mask is correct, but unstable masks should never be treated as hard
    # research labels without review.
    stability_masks = []
    for factor in (0.8, 1.0, 1.2):
        gap = threshold_gap * factor
        if polarity.startswith("bright"):
            proposal = image > max(threshold, background + gap)
        else:
            proposal = image < min(threshold, background - gap)
        stable_component, _ = _largest_component(proposal, prefer_non_border=True)
        stability_masks.append(_postprocess_mask(stable_component))
    stability = float(
        np.mean([_mask_iou(mask, other) for other in stability_masks if np.any(other)])
    ) if any(np.any(other) for other in stability_masks) else 0.0
    contrast = abs(float(image[mask].mean()) - background)
    contrast_score = min(1.0, contrast / max(4.0 * robust_sigma, 0.025))
    quality_margin = (
        scored[0][0] / max(scored[0][0] + scored[1][0], 1e-12)
        if len(scored) > 1
        else 1.0
    )
    confidence = float(0.45 * stability + 0.35 * contrast_score + 0.20 * quality_margin)
    warnings = []
    if touches_border:
        warnings.append("selected component touches the image border")
    if confidence < 0.65:
        warnings.append("automatic segmentation is unstable; provide a reviewed --mask")
    if area_fraction < 0.002 or area_fraction > 0.80:
        warnings.append("foreground area is outside the conservative object-size range")

    return mask, {
        "method": "automatic_largest_component",
        "polarity": polarity,
        "background_median": background,
        "background_mad": mad,
        "otsu_threshold": threshold,
        "foreground_area_fraction": area_fraction,
        "confidence": confidence,
        "threshold_stability": stability,
        "quality_margin": quality_margin,
        "warnings": warnings,
        "requires_manual_review": bool(warnings),
    }


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def _otsu_threshold(image: np.ndarray) -> float:
    values = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0.5
    indices = np.arange(256, dtype=np.float64)
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * indices)
    denominator = cumulative_weight * (total - cumulative_weight)
    numerator = (
        total * cumulative_mean - cumulative_weight * cumulative_mean[-1]
    ) ** 2
    between = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(np.argmax(between) / 255.0)


def _largest_component(
    mask: np.ndarray,
    *,
    prefer_non_border: bool = False,
) -> tuple[np.ndarray, bool]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask)
    best: list[tuple[int, int]] = []
    best_touches_border = False
    best_interior: list[tuple[int, int]] = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        component: list[tuple[int, int]] = []
        touches_border = False
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            if cy in (0, height - 1) or cx in (0, width - 1):
                touches_border = True
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component
            best_touches_border = touches_border
        if not touches_border and len(component) > len(best_interior):
            best_interior = component

    if prefer_non_border and best_interior:
        best = best_interior
        best_touches_border = False
    result = np.zeros_like(mask)
    if best:
        ys, xs = zip(*best)
        result[np.asarray(ys), np.asarray(xs)] = True
    return result, best_touches_border


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = _binary_close(np.asarray(mask, dtype=bool), radius=1)
    mask, _ = _largest_component(mask, prefer_non_border=True)
    return _fill_holes(mask)


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, radius), radius)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.zeros_like(mask)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = max(0, dy)
            target_y1 = min(height, height + dy)
            target_x0 = max(0, dx)
            target_x1 = min(width, width + dx)
            result[target_y0:target_y1, target_x0:target_x1] |= mask[
                source_y0:source_y1, source_x0:source_x1
            ]
    return result


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.ones_like(mask)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = max(0, dy)
            target_y1 = min(height, height + dy)
            target_x0 = max(0, dx)
            target_x1 = min(width, width + dx)
            result[target_y0:target_y1, target_x0:target_x1] &= mask[
                source_y0:source_y1, source_x0:source_x1
            ]
    # Pixels without a full neighborhood should not become foreground.
    if radius:
        result[:radius, :] = False
        result[-radius:, :] = False
        result[:, :radius] = False
        result[:, -radius:] = False
    return result


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    inverse = ~mask
    exterior, _ = _largest_component(inverse)
    # The largest inverse component is the exterior in the normal case. If the
    # mask touches the border, this remains conservative rather than filling
    # arbitrary image regions.
    return mask | (inverse & ~exterior)


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return Rz(roll) Ry(yaw) Rx(pitch) for row-vector vertices."""

    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _project_vertices(
    vertices: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    return vertices @ _rotation_matrix(yaw_deg, pitch_deg, roll_deg).T


def render_silhouette(
    mesh: Mesh,
    image_shape: tuple[int, int],
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float,
    scale_px_per_unit: float,
    translation_px: tuple[float, float],
) -> np.ndarray:
    """Render a z-buffered binary silhouette under a 2D alignment transform."""

    height, width = image_shape
    transformed = _project_vertices(mesh.vertices, yaw_deg, pitch_deg, roll_deg)
    raster = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(raster)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    tx, ty = translation_px

    # Painter ordering makes the opaque projection depth-aware without
    # treating all back-facing triangles as visible. It is exact for the
    # non-intersecting triangle projections used by ordinary closed meshes and
    # is substantially safer than an unordered union of all triangles.
    transformed_faces = transformed[mesh.faces]
    face_vectors = np.cross(
        transformed_faces[:, 1] - transformed_faces[:, 0],
        transformed_faces[:, 2] - transformed_faces[:, 0],
    )
    # The camera is at positive z looking toward the origin. Back-facing
    # triangles cannot appear on the visible opaque surface. Keeping grazing
    # faces avoids eroding the digitized outer contour.
    facing = face_vectors[:, 2] >= -1e-10
    visible_faces = mesh.faces[facing]
    visible_transformed_faces = transformed_faces[facing]
    if len(visible_faces) == 0:
        visible_faces = mesh.faces
        visible_transformed_faces = transformed_faces
    face_depth = visible_transformed_faces[:, :, 2].mean(axis=1)
    face_order = np.argsort(face_depth)
    for face in visible_faces[face_order]:
        triangle = transformed[face]
        screen_x = center_x + tx + scale_px_per_unit * triangle[:, 0]
        screen_y = center_y + ty - scale_px_per_unit * triangle[:, 1]
        min_x = max(0, int(np.floor(np.min(screen_x))))
        max_x = min(width - 1, int(np.ceil(np.max(screen_x))))
        min_y = max(0, int(np.floor(np.min(screen_y))))
        max_y = min(height - 1, int(np.ceil(np.max(screen_y))))
        if min_x > max_x or min_y > max_y:
            continue

        draw.polygon(
            [
                (int(round(screen_x[0])), int(round(screen_y[0]))),
                (int(round(screen_x[1])), int(round(screen_y[1]))),
                (int(round(screen_x[2])), int(round(screen_y[2]))),
            ],
            fill=255,
        )
    return np.asarray(raster, dtype=np.uint8) > 0


def render_normal_map(
    mesh: Mesh,
    image_shape: tuple[int, int],
    *,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float,
    scale_px_per_unit: float,
    translation_px: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize visible candidate normals with a per-pixel z-buffer."""

    height, width = image_shape
    rotation = _rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    transformed = mesh.vertices @ rotation.T
    triangles = transformed[mesh.faces]
    face_vectors = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(face_vectors, axis=1, keepdims=True)
    normals = face_vectors / np.maximum(lengths, 1e-12)
    normal_map = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.full((height, width), -np.inf, dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    tx, ty = translation_px

    for triangle, normal in zip(triangles, normals):
        if normal[2] < -1e-10:
            continue
        screen_x = center_x + tx + scale_px_per_unit * triangle[:, 0]
        screen_y = center_y + ty - scale_px_per_unit * triangle[:, 1]
        min_x = max(0, int(np.floor(np.min(screen_x))))
        max_x = min(width - 1, int(np.ceil(np.max(screen_x))))
        min_y = max(0, int(np.floor(np.min(screen_y))))
        max_y = min(height - 1, int(np.ceil(np.max(screen_y))))
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
        triangle_depth = (
            bary_a * triangle[0, 2] + bary_b * triangle[1, 2] + bary_c * triangle[2, 2]
        )
        current_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        visible = inside & (triangle_depth > current_depth)
        if not np.any(visible):
            continue
        current_depth[visible] = triangle_depth[visible]
        depth[min_y : max_y + 1, min_x : max_x + 1] = current_depth
        current_normals = normal_map[min_y : max_y + 1, min_x : max_x + 1]
        current_normals[visible] = normal
        normal_map[min_y : max_y + 1, min_x : max_x + 1] = current_normals
        current_valid = valid[min_y : max_y + 1, min_x : max_x + 1]
        current_valid[visible] = True
        valid[min_y : max_y + 1, min_x : max_x + 1] = current_valid
    return normal_map, valid


def photometric_normal_metrics(
    observed_image: np.ndarray,
    observed_mask: np.ndarray,
    normal_map: np.ndarray,
    predicted_mask: np.ndarray,
    config: PhotometricConfig,
) -> dict:
    """Measure minimum angular inconsistency with Lambertian normal cones."""

    if observed_image.shape != observed_mask.shape:
        raise ValueError("observed image and mask shapes must match")
    if normal_map.shape != (*observed_image.shape, 3):
        raise ValueError("normal map shape must be image_shape + (3,)")
    if predicted_mask.shape != observed_mask.shape:
        raise ValueError("predicted and observed mask shapes must match")

    light = np.asarray(config.light_direction, dtype=np.float64)
    light /= np.linalg.norm(light)
    candidate_cosine = np.clip(normal_map @ light, -1.0, 1.0)
    raw_cosine = (observed_image / config.exposure - config.ambient) / config.albedo
    observed_cosine = np.clip(raw_cosine, 0.0, 1.0)
    valid = observed_mask & predicted_mask & np.isfinite(raw_cosine)
    valid_count = int(np.count_nonzero(valid))
    observed_count = max(int(np.count_nonzero(observed_mask)), 1)
    if valid_count == 0:
        return {
            "photometric_score": float("inf"),
            "photometric_rmse_cosine": float("inf"),
            "photometric_valid_pixels": 0,
            "photometric_coverage": 0.0,
            "photometric_clipped_fraction": 0.0,
        }

    a = candidate_cosine[valid]
    c = observed_cosine[valid]
    illuminated = raw_cosine[valid] > 0.0
    cone_cosine = np.ones_like(a)
    if np.any(illuminated):
        ai = a[illuminated]
        ci = c[illuminated]
        cone_cosine[illuminated] = (
            ai * ci
            + np.sqrt(np.maximum(0.0, 1.0 - ai * ai))
            * np.sqrt(np.maximum(0.0, 1.0 - ci * ci))
        )
    shadow = ~illuminated
    if np.any(shadow):
        ashadow = a[shadow]
        cone_cosine[shadow] = np.where(
            ashadow <= 0.0,
            1.0,
            np.sqrt(np.maximum(0.0, 1.0 - ashadow * ashadow)),
        )
    inconsistency = 1.0 - np.clip(cone_cosine, -1.0, 1.0)
    predicted_lambert = np.maximum(0.0, a)
    clipped = (raw_cosine[valid] < 0.0) | (raw_cosine[valid] > 1.0)
    return {
        "photometric_score": float(np.mean(inconsistency)),
        "photometric_rmse_cosine": float(np.sqrt(np.mean((predicted_lambert - c) ** 2))),
        "photometric_valid_pixels": valid_count,
        "photometric_coverage": float(valid_count / observed_count),
        "photometric_clipped_fraction": float(np.mean(clipped)),
    }


def _warp_alignment_mask(
    base_mask: np.ndarray,
    *,
    scale_factor: float,
    base_translation: tuple[float, float],
    target_translation: tuple[float, float],
) -> np.ndarray:
    """Apply scale and translation around the image centre using Pillow."""

    if scale_factor <= 0:
        return np.zeros_like(base_mask)
    height, width = base_mask.shape
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    base_tx, base_ty = base_translation
    target_tx, target_ty = target_translation
    inverse_scale = 1.0 / scale_factor
    coefficients = (
        inverse_scale,
        0.0,
        center_x + base_tx - (center_x + target_tx) * inverse_scale,
        0.0,
        inverse_scale,
        center_y + base_ty - (center_y + target_ty) * inverse_scale,
    )
    source = Image.fromarray(base_mask.astype(np.uint8) * 255, mode="L")
    warped = source.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return np.asarray(warped, dtype=np.uint8) > 0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~_binary_erode(mask, radius=1)


def _distance_map_to_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Exact Euclidean distance map in linear time and bounded memory."""

    height, width = shape
    if len(points) == 0:
        return np.full(shape, np.inf, dtype=np.float64)
    sites = np.full(shape, np.inf, dtype=np.float64)
    sites[points[:, 0], points[:, 1]] = 0.0
    row_pass = np.empty(shape, dtype=np.float64)
    for row in range(height):
        row_pass[row] = _squared_distance_transform_1d(sites[row])
    squared = np.empty(shape, dtype=np.float64)
    for column in range(width):
        squared[:, column] = _squared_distance_transform_1d(row_pass[:, column])
    return np.sqrt(squared)


def _squared_distance_transform_1d(costs: np.ndarray) -> np.ndarray:
    """Lower envelope of squared-distance parabolas for one image axis."""

    length = len(costs)
    finite = np.flatnonzero(np.isfinite(costs))
    if len(finite) == 0:
        return np.full(length, np.inf, dtype=np.float64)
    vertices = np.empty(length, dtype=np.int32)
    breaks = np.empty(length + 1, dtype=np.float64)
    count = 0
    vertices[0] = int(finite[0])
    breaks[0] = -np.inf
    breaks[1] = np.inf
    for site in finite[1:]:
        site = int(site)
        while True:
            previous = int(vertices[count])
            crossing = (
                (costs[site] + site * site)
                - (costs[previous] + previous * previous)
            ) / (2 * (site - previous))
            if crossing > breaks[count]:
                break
            count -= 1
        count += 1
        vertices[count] = site
        breaks[count] = crossing
        breaks[count + 1] = np.inf
    result = np.empty(length, dtype=np.float64)
    count = 0
    for position in range(length):
        while breaks[count + 1] < position:
            count += 1
        site = int(vertices[count])
        result[position] = (position - site) ** 2 + costs[site]
    return result


def _nearest_point_distances(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Boundary-only distances avoid building a full map for final scores."""

    nearest = np.empty(len(points), dtype=np.float64)
    target_y = targets[:, 0][None, :]
    target_x = targets[:, 1][None, :]
    for start in range(0, len(points), 128):
        chunk = points[start : start + 128]
        dy = chunk[:, 0, None] - target_y
        dx = chunk[:, 1, None] - target_x
        nearest[start : start + len(chunk)] = np.sqrt(np.min(dy * dy + dx * dx, axis=1))
    return nearest


def outline_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    chamfer_weight: float = 1.0,
    iou_weight: float = 0.35,
    area_weight: float = 0.15,
) -> dict:
    """Compute a symmetric contour score with overlap and area safeguards."""

    observed_boundary_mask = _mask_boundary(observed)
    predicted_boundary_mask = _mask_boundary(predicted)
    observed_boundary = np.argwhere(observed_boundary_mask)
    predicted_boundary = np.argwhere(predicted_boundary_mask)
    diagonal = float(np.hypot(*observed.shape))
    if len(observed_boundary) == 0 or len(predicted_boundary) == 0:
        return {
            "score": float("inf"),
            "symmetric_chamfer_px": float("inf"),
            "symmetric_chamfer_normalized": float("inf"),
            "iou": 0.0,
            "area_ratio": 0.0,
            "area_log_error": float("inf"),
        }

    observed_to_predicted = _nearest_point_distances(observed_boundary, predicted_boundary)
    predicted_to_observed = _nearest_point_distances(predicted_boundary, observed_boundary)
    chamfer = float(0.5 * (observed_to_predicted.mean() + predicted_to_observed.mean()))
    intersection = np.count_nonzero(observed & predicted)
    union = np.count_nonzero(observed | predicted)
    iou = float(intersection / union) if union else 0.0
    observed_area = max(int(np.count_nonzero(observed)), 1)
    predicted_area = max(int(np.count_nonzero(predicted)), 1)
    area_ratio = float(predicted_area / observed_area)
    area_log_error = float(abs(np.log(area_ratio)))
    # The contour term follows the presentation's outline-distance idea. IoU
    # and area terms prevent a thin or displaced contour from looking good by
    # matching only one side of the outline.
    score = (
        chamfer_weight * chamfer / diagonal
        + iou_weight * (1.0 - iou)
        + area_weight * min(area_log_error, 2.0)
    )
    return {
        "score": float(score),
        "symmetric_chamfer_px": chamfer,
        "symmetric_chamfer_normalized": float(chamfer / diagonal),
        "iou": iou,
        "area_ratio": area_ratio,
        "area_log_error": area_log_error,
    }


def _initial_alignment(
    mesh: Mesh,
    observed: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[float, tuple[float, float]]:
    projected = _project_vertices(mesh.vertices, yaw_deg, pitch_deg, roll_deg)
    observed_points = np.argwhere(observed)
    if len(observed_points) == 0:
        raise ValueError("the observed silhouette is empty")
    observed_height, observed_width = observed.shape
    observed_height_px = float(np.ptp(observed_points[:, 0]) + 1)
    observed_width_px = float(np.ptp(observed_points[:, 1]) + 1)
    mesh_width = max(float(np.ptp(projected[:, 0])), 1e-12)
    mesh_height = max(float(np.ptp(projected[:, 1])), 1e-12)
    scale = 0.98 * min(observed_width_px / mesh_width, observed_height_px / mesh_height)
    observed_center_yx = observed_points.mean(axis=0)
    mesh_center_xy = projected[:, :2].mean(axis=0)
    center_x, center_y = (observed_width - 1) / 2.0, (observed_height - 1) / 2.0
    tx = float(observed_center_yx[1] - (center_x + scale * mesh_center_xy[0]))
    ty = float(observed_center_yx[0] - (center_y - scale * mesh_center_xy[1]))
    return scale, (tx, ty)


def _search_outline_score(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_distance_map: np.ndarray,
    *,
    config: OutlineMatcherConfig,
) -> float:
    """Fast alignment proposal score.

    The observed contour distance map is precomputed once. IoU and area terms
    keep the one-way distance from preferring a contour that only covers a
    small part of the target. The final winning transform is re-scored with
    the fully symmetric contour metric.
    """

    predicted_boundary = _mask_boundary(predicted)
    if not np.any(predicted_boundary):
        return float("inf")
    predicted_to_observed = float(observed_distance_map[predicted_boundary].mean())
    intersection = np.count_nonzero(observed & predicted)
    union = np.count_nonzero(observed | predicted)
    iou = float(intersection / union) if union else 0.0
    area_ratio = np.count_nonzero(predicted) / max(np.count_nonzero(observed), 1)
    area_error = abs(float(np.log(max(area_ratio, 1e-12))))
    diagonal = float(np.hypot(*observed.shape))
    return (
        config.chamfer_weight * predicted_to_observed / diagonal
        + config.iou_weight * (1.0 - iou)
        + config.area_weight * min(area_error, 2.0)
    )


def _chamfer_translation_seed(
    base_mask: np.ndarray,
    observed: np.ndarray,
    observed_distance_map: np.ndarray,
    *,
    base_translation: tuple[float, float],
    config: OutlineMatcherConfig,
) -> tuple[float, float, float]:
    """Propose translation by searching the observed contour distance field.

    The centroid-based alignment is not a reliable registration rule for
    asymmetric silhouettes: the mesh centroid, silhouette centroid, and
    image-object centroid can all differ.  Here we treat translation as a
    nuisance parameter and minimize the same fast chamfer/overlap objective
    used by the search.  This is a proposal stage only; the selected result is
    later evaluated with symmetric contour distance and IoU.
    """

    height, width = observed.shape
    image_scale = float(max(height, width))
    base_tx, base_ty = base_translation
    best_translation = base_translation
    best_score = float("inf")
    for dx_fraction in config.translation_seed_fractions:
        for dy_fraction in config.translation_seed_fractions:
            tx = base_tx + dx_fraction * image_scale
            ty = base_ty + dy_fraction * image_scale
            shifted = _warp_alignment_mask(
                base_mask,
                scale_factor=1.0,
                base_translation=base_translation,
                target_translation=(tx, ty),
            )
            score = _search_outline_score(
                observed,
                shifted,
                observed_distance_map,
                config=config,
            )
            if score < best_score:
                best_score = score
                best_translation = (tx, ty)
    return float(best_translation[0]), float(best_translation[1]), float(best_score)


def match_mesh_outline(
    mesh: Mesh,
    observed_mask: np.ndarray,
    *,
    config: OutlineMatcherConfig | None = None,
    observed_image: np.ndarray | None = None,
    photometric_config: PhotometricConfig | None = None,
    thermal_config: ThermalConfig | None = None,
    observed_thermal_image: np.ndarray | None = None,
) -> dict:
    """Fit bounded alignment and full camera pose, then return an audit report."""

    config = config or OutlineMatcherConfig()
    if photometric_config is not None and observed_image is None:
        raise ValueError("observed_image is required when photometric scoring is enabled")
    if observed_image is not None and observed_image.shape != observed_mask.shape:
        raise ValueError("observed image and silhouette shapes must match")
    if observed_thermal_image is not None and thermal_config is None:
        raise ValueError("thermal_config is required with an observed thermal image")
    if observed_thermal_image is not None and observed_thermal_image.shape != observed_mask.shape:
        raise ValueError("thermal image and silhouette shapes must match")
    observed_mask = _postprocess_mask(observed_mask)
    if not np.any(observed_mask):
        raise ValueError("the observed silhouette is empty")
    height, width = observed_mask.shape
    observed_area = np.count_nonzero(observed_mask)
    observed_boundary = _mask_boundary(observed_mask)
    observed_boundary_points = np.argwhere(observed_boundary)
    observed_distance_map = _distance_map_to_points(observed_boundary_points, observed_mask.shape)
    best: dict | None = None
    second_best: dict | None = None
    evaluated_records: list[dict] = []
    trials = 0

    def evaluate(
        predicted: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float,
        scale: float,
        tx: float,
        ty: float,
        translation_seed_px: tuple[float, float] | None = None,
        translation_seed_score: float | None = None,
    ) -> None:
        nonlocal best, second_best, trials
        if scale <= 0:
            return
        predicted_area = np.count_nonzero(predicted)
        if predicted_area == 0 or predicted_area > config.max_mesh_area_fraction * height * width:
            return
        search_score = _search_outline_score(
            observed_mask,
            predicted,
            observed_distance_map,
            config=config,
        )
        trials += 1
        record = {
            "search_score": float(search_score),
            "yaw_deg": float(yaw % 360.0),
            "pitch_deg": float(np.clip(pitch, -90.0, 90.0)),
            "roll_deg": float(roll % 360.0),
            "scale_px_per_unit": float(scale),
            "translation_px": [float(tx), float(ty)],
        }
        if translation_seed_px is not None:
            record["translation_seed_px"] = [
                float(translation_seed_px[0]),
                float(translation_seed_px[1]),
            ]
        if translation_seed_score is not None:
            record["translation_seed_score"] = float(translation_seed_score)
        evaluated_records.append(record)
        if best is None or record["search_score"] < best["search_score"]:
            second_best = best
            best = record
        elif second_best is None or record["search_score"] < second_best["search_score"]:
            second_best = record

    def evaluate_pose(yaw: float, pitch: float, roll: float, *, fine: bool) -> None:
        scale0, (tx0, ty0) = _initial_alignment(
            mesh, observed_mask, float(yaw), float(pitch), float(roll)
        )
        base_mask = render_silhouette(
            mesh,
            observed_mask.shape,
            yaw_deg=float(yaw),
            pitch_deg=float(pitch),
            roll_deg=float(roll),
            scale_px_per_unit=scale0,
            translation_px=(tx0, ty0),
        )
        seed_tx, seed_ty, seed_score = _chamfer_translation_seed(
            base_mask,
            observed_mask,
            observed_distance_map,
            base_translation=(tx0, ty0),
            config=config,
        )
        scale_factors = config.fine_scale_factors if fine else config.scale_factors
        max_offset = max(height, width) * max(abs(value) for value in config.translation_fractions)
        translation_values = (
            config.fine_translation_pixels if fine else config.translation_fractions
        )
        for scale_factor in scale_factors:
            scale = scale0 * scale_factor if not fine else scale_factor
            for dx_value in translation_values:
                for dy_value in translation_values:
                    if fine:
                        tx = seed_tx + dx_value
                        ty = seed_ty + dy_value
                        warp_factor = scale / scale0
                    else:
                        tx = seed_tx + dx_value * max_offset
                        ty = seed_ty + dy_value * max_offset
                        warp_factor = scale_factor
                    evaluate(
                        _warp_alignment_mask(
                            base_mask,
                            scale_factor=warp_factor,
                            base_translation=(tx0, ty0),
                            target_translation=(tx, ty),
                        ),
                        float(yaw),
                        float(pitch),
                        float(roll),
                        scale,
                        tx,
                        ty,
                        (seed_tx - tx0, seed_ty - ty0),
                        seed_score,
                    )

    coarse_yaws = np.arange(0.0, 360.0, config.coarse_yaw_step_deg)
    coarse_rolls = np.arange(0.0, 360.0, config.coarse_roll_step_deg)
    for yaw in coarse_yaws:
        for pitch in config.coarse_pitch_angles_deg:
            for roll in coarse_rolls:
                evaluate_pose(float(yaw), float(pitch), float(roll), fine=False)

    if best is None:
        raise ValueError("no valid mesh alignment was found")

    coarse_best = dict(best)
    fine_radius = config.fine_pose_radius_deg
    fine_step = config.fine_pose_step_deg
    fine_yaws = np.arange(
        coarse_best["yaw_deg"] - fine_radius,
        coarse_best["yaw_deg"] + fine_radius + 0.1,
        fine_step,
    )
    fine_pitches = np.arange(
        max(-90.0, coarse_best["pitch_deg"] - fine_radius),
        min(90.0, coarse_best["pitch_deg"] + fine_radius) + 0.1,
        fine_step,
    )
    fine_rolls = np.arange(
        coarse_best["roll_deg"] - fine_radius,
        coarse_best["roll_deg"] + fine_radius + 0.1,
        fine_step,
    )
    for yaw in fine_yaws:
        for pitch in fine_pitches:
            for roll in fine_rolls:
                scale0, (tx0, ty0) = _initial_alignment(
                    mesh, observed_mask, float(yaw), float(pitch), float(roll)
                )
                target_scale = coarse_best["scale_px_per_unit"]
                base_mask = render_silhouette(
                    mesh,
                    observed_mask.shape,
                    yaw_deg=float(yaw),
                    pitch_deg=float(pitch),
                    roll_deg=float(roll),
                    scale_px_per_unit=scale0,
                    translation_px=(tx0, ty0),
                )
                seed_tx, seed_ty, seed_score = _chamfer_translation_seed(
                    base_mask,
                    observed_mask,
                    observed_distance_map,
                    base_translation=(tx0, ty0),
                    config=config,
                )
                for scale_factor in config.fine_scale_factors:
                    for dx in config.fine_translation_pixels:
                        for dy in config.fine_translation_pixels:
                            tx, ty = seed_tx + dx, seed_ty + dy
                            scale = target_scale * scale_factor
                            evaluate(
                                _warp_alignment_mask(
                                    base_mask,
                                    scale_factor=scale / scale0,
                                    base_translation=(tx0, ty0),
                                    target_translation=(tx, ty),
                                ),
                                float(yaw),
                                float(pitch),
                                float(roll),
                                scale,
                                tx,
                                ty,
                                (seed_tx - tx0, seed_ty - ty0),
                                seed_score,
                            )

    ranked_search = sorted(evaluated_records, key=lambda record: record["search_score"])
    search_best = ranked_search[0]
    rerank_candidates = ranked_search[: max(1, config.final_rerank_top_k)]
    finalized: list[dict] = []
    for candidate in rerank_candidates:
        candidate_predicted = render_silhouette(
            mesh,
            observed_mask.shape,
            yaw_deg=candidate["yaw_deg"],
            pitch_deg=candidate["pitch_deg"],
            roll_deg=candidate["roll_deg"],
            scale_px_per_unit=candidate["scale_px_per_unit"],
            translation_px=tuple(candidate["translation_px"]),
        )
        final_metrics = outline_metrics(
            observed_mask,
            candidate_predicted,
            chamfer_weight=config.chamfer_weight,
            iou_weight=config.iou_weight,
            area_weight=config.area_weight,
        )
        silhouette_score = float(final_metrics["score"])
        combined_metrics: dict = {**final_metrics, "silhouette_score": silhouette_score}
        combined_score = config.silhouette_term_weight * silhouette_score
        normal_map = normal_valid = None
        if photometric_config is not None or observed_thermal_image is not None:
            normal_map, normal_valid = render_normal_map(
                mesh,
                observed_mask.shape,
                yaw_deg=candidate["yaw_deg"],
                pitch_deg=candidate["pitch_deg"],
                roll_deg=candidate["roll_deg"],
                scale_px_per_unit=candidate["scale_px_per_unit"],
                translation_px=tuple(candidate["translation_px"]),
            )
            if photometric_config is not None and observed_image is not None:
                photo_metrics = photometric_normal_metrics(
                    observed_image,
                    observed_mask,
                    normal_map,
                    candidate_predicted & normal_valid,
                    photometric_config,
                )
                combined_metrics.update(photo_metrics)
                combined_score += (
                    photometric_config.weight * photo_metrics["photometric_score"]
                )
        if thermal_config is not None:
            posed_mesh = Mesh(
                vertices=_project_vertices(
                    mesh.vertices,
                    candidate["yaw_deg"],
                    candidate["pitch_deg"],
                    candidate["roll_deg"],
                ),
                faces=mesh.faces,
            )
            thermal_metrics = thermal_flux_metrics(posed_mesh, thermal_config)
            if observed_thermal_image is not None:
                image_metrics = thermal_image_metrics(
                    observed_thermal_image,
                    observed_mask,
                    normal_map,
                    normal_valid & candidate_predicted,
                    thermal_config,
                )
                thermal_metrics.update(image_metrics)
                thermal_metrics["thermal_score"] = float(
                    0.5 * thermal_metrics["thermal_flux_score"]
                    + 0.5 * image_metrics["thermal_image_score"]
                )
            combined_metrics.update(thermal_metrics)
            combined_score += thermal_config.weight * thermal_metrics["thermal_score"]
        combined_metrics["score"] = float(combined_score)
        finalized.append(
            {**candidate, **combined_metrics, "predicted_mask": candidate_predicted}
        )
    finalized.sort(key=lambda candidate: candidate["score"])
    best = finalized[0]
    second_final = finalized[1] if len(finalized) > 1 else None
    predicted = best.pop("predicted_mask")
    search_score = float(best["search_score"])
    best["trials_evaluated"] = trials
    best["coarse_search_score"] = float(coarse_best["search_score"])
    best["alignment_refinement_gain"] = float(coarse_best["search_score"] - search_score)
    search_pose = (
        search_best["yaw_deg"],
        search_best["pitch_deg"],
        search_best["roll_deg"],
        search_best["scale_px_per_unit"],
        tuple(search_best["translation_px"]),
    )
    search_winner_final = next(
        (
            candidate
            for candidate in finalized
            if (
                candidate["yaw_deg"],
                candidate["pitch_deg"],
                candidate["roll_deg"],
                candidate["scale_px_per_unit"],
                tuple(candidate["translation_px"]),
            )
            == search_pose
        ),
        None,
    )
    best["search_winner_final_score"] = (
        float(search_winner_final["score"]) if search_winner_final else None
    )
    best["selected_by_final_rerank"] = (
        best["yaw_deg"], best["pitch_deg"], best["roll_deg"]
    ) != (
        search_best["yaw_deg"], search_best["pitch_deg"], search_best["roll_deg"]
    )
    best["second_best_search_score"] = (
        float(ranked_search[1]["search_score"]) if len(ranked_search) > 1 else None
    )
    best["search_score_margin"] = (
        float(ranked_search[1]["search_score"] - search_best["search_score"])
        if len(ranked_search) > 1
        else None
    )
    if second_final:
        best["second_best_final_score"] = float(second_final["score"])
        best["final_alignment_margin"] = float(second_final["score"] - best["score"])
    else:
        best["second_best_final_score"] = None
        best["final_alignment_margin"] = None
    best["observed_area_px"] = int(observed_area)
    best["predicted_area_px"] = int(np.count_nonzero(predicted))
    best["predicted_mask"] = predicted
    return best


def _save_overlay(path: Path, observed: np.ndarray, predicted: np.ndarray) -> None:
    image = np.zeros((*observed.shape, 3), dtype=np.uint8)
    observed_boundary = _mask_boundary(observed)
    predicted_boundary = _mask_boundary(predicted)
    image[observed_boundary] = [255, 50, 50]
    image[predicted_boundary & ~observed_boundary] = [0, 205, 0]
    overlap = observed_boundary & predicted_boundary
    image[overlap] = [255, 230, 0]
    Image.fromarray(image, mode="RGB").save(path)


def match_files(
    image_path: Path,
    mesh_paths: list[Path],
    *,
    mask_path: Path | None = None,
    output_path: Path,
    overlay_dir: Path | None = None,
    config: OutlineMatcherConfig | None = None,
    strict_segmentation: bool = False,
    photometric_config: PhotometricConfig | None = None,
    thermal_config: ThermalConfig | None = None,
    thermal_image_path: Path | None = None,
) -> dict:
    image = load_grayscale_image(image_path)
    thermal_image = None
    if thermal_image_path is not None:
        thermal_image = np.load(thermal_image_path, allow_pickle=False)
        if thermal_image.shape != image.shape:
            raise ValueError("thermal radiance image shape must match optical image")
    if mask_path:
        observed_mask = load_mask(mask_path)
        segmentation = {"method": "supplied_mask", "path": str(mask_path)}
    else:
        observed_mask, segmentation = segment_silhouette(image)
        if strict_segmentation and segmentation.get("requires_manual_review", False):
            warnings = "; ".join(segmentation.get("warnings", []))
            raise ValueError(
                "automatic segmentation failed the strict review gate"
                + (f": {warnings}" if warnings else "")
            )

    reports = []
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observed_mask.astype(np.uint8) * 255, mode="L").save(
            overlay_dir / "observed_mask.png"
        )
    for mesh_path in sorted(mesh_paths):
        mesh = load_mesh(mesh_path)
        result = match_mesh_outline(
            mesh,
            observed_mask,
            config=config,
            observed_image=image,
            photometric_config=photometric_config,
            thermal_config=thermal_config,
            observed_thermal_image=thermal_image,
        )
        predicted_mask = result.pop("predicted_mask")
        if overlay_dir:
            overlay_path = overlay_dir / f"{mesh_path.stem}_overlay.png"
            _save_overlay(overlay_path, observed_mask, predicted_mask)
            result["overlay_path"] = str(overlay_path)
        reports.append({"mesh_path": str(mesh_path), **result})
    reports.sort(key=lambda report: report["score"])
    report = {
        "schema_version": (
            3 if thermal_config is not None else 2 if photometric_config is not None else 1
        ),
        "image_path": str(image_path),
        "image_shape": list(image.shape),
        "segmentation": segmentation,
        "alignment_model": (
            "orthographic_full_yaw_pitch_roll_with_chamfer_field_translation_seed"
            "_and_bounded_scale_translation"
        ),
        "score_definition": " + ".join(
            ["silhouette score"]
            + (["weighted Lambertian normal-cone inconsistency"] if photometric_config else [])
            + (["weighted equilibrium thermal flux/image error"] if thermal_config else [])
        ) + "; final rerank after silhouette search",
        "config": asdict(config or OutlineMatcherConfig()),
        "photometric_config": (
            asdict(photometric_config) if photometric_config is not None else None
        ),
        "thermal_config": asdict(thermal_config) if thermal_config is not None else None,
        "thermal_image_path": str(thermal_image_path) if thermal_image_path else None,
        "meshes": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _expand_mesh_inputs(inputs: list[Path]) -> list[Path]:
    result: list[Path] = []
    for value in inputs:
        if value.is_dir():
            result.extend(sorted(p for p in value.iterdir() if p.suffix.lower() in {".npz", ".obj"}))
        else:
            result.append(value)
    if not result:
        raise ValueError("no .npz or .obj meshes were found")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, nargs="+", required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument(
        "--light-direction",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="enable photometric scoring with this known light direction",
    )
    parser.add_argument("--albedo", type=float, default=0.62)
    parser.add_argument("--ambient", type=float, default=0.0)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--photometric-weight", type=float, default=0.5)
    parser.add_argument("--silhouette-term-weight", type=float, default=1.0)
    parser.add_argument(
        "--thermal-observed-flux",
        type=float,
        help="enable equilibrium thermal scoring with this bolometric flux in W/m^2",
    )
    parser.add_argument("--thermal-sun-direction", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--observer-direction", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--bond-albedo", type=float, default=0.08)
    parser.add_argument("--solar-flux-w-m2", type=float, default=1361.0)
    parser.add_argument("--heliocentric-distance-au", type=float, default=1.0)
    parser.add_argument("--observer-distance-m", type=float, default=1.0)
    parser.add_argument("--mesh-unit-m", type=float, default=1.0)
    parser.add_argument("--thermal-weight", type=float, default=0.1)
    parser.add_argument("--thermal-wavelength-um", type=float,
                        help="wavelength of a spectral IR image in micrometres; omit for bolometric")
    parser.add_argument(
        "--thermal-image",
        type=Path,
        help="optional physical thermal radiance image (.npy); set --thermal-wavelength-um for spectral IR",
    )
    parser.add_argument(
        "--strict-segmentation",
        action="store_true",
        help="fail instead of silently using an uncertain automatic silhouette",
    )
    args = parser.parse_args(argv)

    matcher_config = OutlineMatcherConfig(
        silhouette_term_weight=args.silhouette_term_weight
    )

    photometric_config = None
    if args.light_direction is not None:
        photometric_config = PhotometricConfig(
            light_direction=tuple(args.light_direction),
            albedo=args.albedo,
            ambient=args.ambient,
            exposure=args.exposure,
            weight=args.photometric_weight,
        )
    thermal_config = None
    if args.thermal_observed_flux is not None:
        thermal_sun_direction = args.thermal_sun_direction or args.light_direction
        if thermal_sun_direction is None:
            parser.error(
                "--thermal-observed-flux requires --thermal-sun-direction "
                "or --light-direction"
            )
        thermal_config = ThermalConfig(
            observed_flux_w_m2=args.thermal_observed_flux,
            sun_direction=tuple(thermal_sun_direction),
            observer_direction=tuple(args.observer_direction),
            bond_albedo=args.bond_albedo,
            solar_flux_w_m2_at_1au=args.solar_flux_w_m2,
            heliocentric_distance_au=args.heliocentric_distance_au,
            observer_distance_m=args.observer_distance_m,
            mesh_unit_m=args.mesh_unit_m,
            wavelength_um=args.thermal_wavelength_um,
            weight=args.thermal_weight,
        )

    report = match_files(
        args.image,
        _expand_mesh_inputs(args.mesh),
        mask_path=args.mask,
        output_path=args.output,
        overlay_dir=args.overlay_dir,
        config=matcher_config,
        strict_segmentation=args.strict_segmentation,
        photometric_config=photometric_config,
        thermal_config=thermal_config,
        thermal_image_path=args.thermal_image,
    )
    for index, mesh in enumerate(report["meshes"], start=1):
        photo = (
            f" photo={mesh['photometric_score']:.6f}"
            if "photometric_score" in mesh
            else ""
        )
        thermal = (
            f" thermal={mesh['thermal_score']:.6f}"
            if "thermal_score" in mesh
            else ""
        )
        print(f"{index:02d}. {mesh['mesh_path']} score={mesh['score']:.6f} "
              f"iou={mesh['iou']:.4f}{photo}{thermal} roll={mesh['roll_deg']:.1f}deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
