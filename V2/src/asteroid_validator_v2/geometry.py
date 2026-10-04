"""Topology-independent area-sampled point-to-triangle surface error.

Coordinate frame and scale are preserved. No ground-truth-assisted registration
or scale fitting takes place here. Large comparisons are bounded by chunk sizes.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .physics.synthetic import Mesh

LABEL_METRIC = "symmetric_area_mean_point_to_triangle_distance/reference_diameter"

def validate_mesh(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    raw_faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("vertices must be an Nx3 array with at least three vertices")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces):
        raise ValueError("faces must be a nonempty Mx3 integer array")
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError("face indices must be integers")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    faces = raw_faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face indices are outside the vertex array")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    keep = areas > 0
    if not np.isfinite(areas).all() or not np.any(keep):
        raise ValueError("mesh must have finite, nondegenerate surface area")
    return triangles[keep], areas[keep]


def load_mesh(path: Path) -> Mesh:
    """Load NPZ or OBJ without recentering, rotating, or rescaling the mesh."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            mesh = Mesh(np.asarray(archive["vertices"], dtype=float), archive["faces"].copy())
    elif path.suffix.lower() == ".obj":
        vertices, faces = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(v) for v in fields[1:4]])
            elif fields[0] == "f":
                indices = []
                for item in fields[1:]:
                    index = int(item.split("/")[0])
                    if index == 0:
                        raise ValueError("OBJ indices are one-based; zero is invalid")
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                if len(indices) < 3:
                    raise ValueError("OBJ faces need at least three vertices")
                faces.extend((indices[0], indices[i], indices[i + 1])
                             for i in range(1, len(indices) - 1))
        mesh = Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64))
    else:
        raise ValueError("use an NPZ mesh with vertices/faces or an OBJ mesh")
    validate_mesh(mesh)
    return mesh


def sample_surface(mesh: Mesh, count: int, seed: int = 0) -> np.ndarray:
    if count < 2:
        raise ValueError("at least two surface samples are required")
    triangles, areas = validate_mesh(mesh)
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    uv = rng.random((count, 2))
    r = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - r, r * (1 - uv[:, 1]), r * uv[:, 1]))
    return np.einsum("ni,nij->nj", bary, chosen)


def surface_distances(points: np.ndarray, mesh: Mesh, *,
                      point_chunk: int = 64, triangle_chunk: int = 512) -> np.ndarray:
    """Exact nearest triangle distance for each given point, not nearest vertex."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite Nx3 array")
    if point_chunk < 1 or triangle_chunk < 1:
        raise ValueError("chunk sizes must be positive")
    triangles, _ = validate_mesh(mesh)
    result = np.empty(len(points))
    for start in range(0, len(points), point_chunk):
        p = points[start:start + point_chunk, None, :]
        best = np.full(len(p), np.inf)
        for offset in range(0, len(triangles), triangle_chunk):
            t = triangles[offset:offset + triangle_chunk]
            a, b, c = t[:, 0], t[:, 1], t[:, 2]
            ab, ac = b - a, c - a
            normal = np.cross(ab, ac)
            n2 = np.sum(normal * normal, axis=1)
            pa = p - a
            signed = np.sum(pa * normal, axis=2)
            projection = pa - (signed / n2)[..., None] * normal
            d00 = np.sum(ab * ab, axis=1)
            d01 = np.sum(ab * ac, axis=1)
            d11 = np.sum(ac * ac, axis=1)
            d20 = np.sum(projection * ab, axis=2)
            d21 = np.sum(projection * ac, axis=2)
            # n2 equals the barycentric denominator, with better numerical behavior.
            u = (d11 * d20 - d01 * d21) / n2
            v = (d00 * d21 - d01 * d20) / n2
            inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
            distances2 = np.where(inside, signed * signed / n2, np.inf)
            for first, second in ((a, b), (b, c), (c, a)):
                edge = second - first
                delta = p - first
                fraction = np.clip(np.sum(delta * edge, axis=2) /
                                   np.sum(edge * edge, axis=1), 0, 1)
                residual = delta - fraction[..., None] * edge
                distances2 = np.minimum(distances2, np.sum(residual * residual, axis=2))
            best = np.minimum(best, distances2.min(axis=1))
        result[start:start + len(p)] = np.sqrt(np.maximum(best, 0))
    return result


def diameter(mesh: Mesh, chunk: int = 256) -> float:
    triangles, _ = validate_mesh(mesh)
    if chunk < 1:
        raise ValueError("diameter chunk size must be positive")
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    largest2 = 0.0
    for start in range(0, len(vertices), chunk):
        delta = vertices[start:start + chunk, None, :] - vertices[None, :, :]
        largest2 = max(largest2, float(np.max(np.sum(delta * delta, axis=2))))
    return float(np.sqrt(largest2))


def geometric_error(candidate: Mesh, reference: Mesh, *, samples: int = 2048,
                    seed: int = 0) -> dict:
    forward = surface_distances(sample_surface(candidate, samples, seed), reference)
    backward = surface_distances(sample_surface(reference, samples, seed + 1), candidate)
    size = diameter(reference)
    mean = (float(forward.mean()) + float(backward.mean())) / 2
    standard_error = np.sqrt(forward.var(ddof=1) / samples + backward.var(ddof=1) / samples) / 2
    return {
        "metric": LABEL_METRIC,
        "geometric_error": mean / size,
        "surface_distance_mesh_units": mean,
        "reference_diameter_mesh_units": size,
        "sampling_standard_error": float(standard_error / size),
        "symmetric_p95_mesh_units": float(np.quantile(np.concatenate((forward, backward)), .95)),
        "samples_per_surface": samples,
        "sampling_seed": seed,
        "registration": "none; preserve supplied common coordinate frame and scale",
    }
