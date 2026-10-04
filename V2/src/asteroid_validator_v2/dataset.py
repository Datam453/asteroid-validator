"""Feature/label assembly, explicit object splits and leakage checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .features import extract_case
from .geometry import geometric_error, load_mesh, LABEL_METRIC
from .model import FEATURES, FEATURE_VERSION


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def dataset_arrays(records: list[dict], *, labels: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    if not records:
        raise ValueError("split contains no usable candidates")
    x = np.array([[row[name] for name in FEATURES] for row in records], dtype=float)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("all three features must be finite and non-negative")
    if not labels:
        return x, None
    y = np.array([row["geometric_error"] for row in records], dtype=float)
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("all labels must be finite and non-negative")
    return x, y


def validate_splits(data: dict, *, require_test: bool = True) -> dict[str, list[dict]]:
    if data.get("schema_version") != 1 or data.get("feature_version") != FEATURE_VERSION:
        raise ValueError("unsupported dataset schema or feature version")
    if data.get("label_metric") != LABEL_METRIC:
        raise ValueError("dataset labels must use the declared V2 geometric-error metric")
    groups = {s: [] for s in ("train", "validation", "test")}
    owners, references, seen = {}, {}, set()
    for row in data["records"]:
        split, object_id = row["split"], row["asteroid_id"]
        if split not in groups:
            raise ValueError("split must be train, validation or test")
        if object_id in owners and owners[object_id] != split:
            raise ValueError(f"object leakage across splits: {object_id}")
        owners[object_id] = split
        ref = row.get("reference_sha256")
        if ref and ref in references and references[ref] != split:
            raise ValueError("identical reference file appears across splits")
        if ref:
            references[ref] = split
        key = (object_id, row.get("view_id", "default"), row["candidate_id"])
        if key in seen:
            raise ValueError(f"duplicate candidate record: {key}")
        seen.add(key)
        groups[split].append(row)
    for split in ("train", "validation", "test") if require_test else ("train", "validation"):
        dataset_arrays(groups[split])
    if len({row.get("alignment_policy", data.get("alignment_policy")) for row in data["records"]}) != 1:
        raise ValueError("mixing alignment policies requires a separately defined model")
    return groups


def build_dataset(cases_path: Path, output_path: Path, *, samples: int = 2048, seed: int = 0) -> dict:
    cases_path = Path(cases_path).resolve()
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    records, failures, extraction = [], [], []
    for i, case in enumerate(cases):
        path = (cases_path.parent / case["manifest"]).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        feature_report = extract_case(path, Path(output_path).parent / "overlays" / case["asteroid_id"])
        extraction.append({"manifest": str(path), "alignment_policy": feature_report["alignment_policy"],
                           "extraction_config": feature_report["extraction_config"]})
        reference_path = path.parent / manifest["ground_truth_mesh"]
        reference = load_mesh(reference_path)
        reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        for j, row in enumerate(feature_report["records"]):
            label = geometric_error(load_mesh(Path(row["mesh_path"])), reference,
                                    samples=samples, seed=seed + i * 10000 + j * 2)
            records.append({**row, "asteroid_id": case["asteroid_id"], "split": case["split"],
                            "view_id": case.get("view_id", "default"),
                            "alignment_policy": feature_report["alignment_policy"],
                            "reference_sha256": reference_hash,
                            "geometric_error": label["geometric_error"], "label_details": label,
                            "generator": manifest.get("generator", "external_or_unspecified")})
        failures.extend({**row, "asteroid_id": case["asteroid_id"], "split": case["split"]}
                        for row in feature_report["failures"])
        print(f"Features and labels: {i + 1}/{len(cases)} objects", flush=True)
    data = {"schema_version": 1, "feature_version": FEATURE_VERSION, "records": records,
            "failures": failures, "extraction": extraction, "seed": seed,
            "samples_per_surface": samples,
            "label_metric": LABEL_METRIC}
    validate_splits(data)
    write_json(output_path, data)
    return data
