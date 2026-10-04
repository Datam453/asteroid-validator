"""Build a self-contained HTML viewer for outline-matching diagnostics."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path


def _resolve_report_path(value: str, report_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    search_roots = [Path.cwd(), report_path.parent, report_path.parent.parent]
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return candidate


def _data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{payload}"


def _compact_mesh_name(value: str) -> str:
    return Path(value).stem


def build_viewer(report_path: Path, output_path: Path) -> None:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    image_path = _resolve_report_path(report["image_path"], report_path)
    if not image_path.exists():
        raise FileNotFoundError(f"input image not found: {report['image_path']}")

    image_uri = _data_uri(image_path)
    candidates: list[dict] = []
    observed_mask_uri: str | None = None
    for mesh in report.get("meshes", []):
        overlay_value = mesh.get("overlay_path")
        if not overlay_value:
            continue
        overlay_path = _resolve_report_path(overlay_value, report_path)
        if not overlay_path.exists():
            continue
        if observed_mask_uri is None:
            observed_mask_path = overlay_path.parent / "observed_mask.png"
            if observed_mask_path.exists():
                observed_mask_uri = _data_uri(observed_mask_path)
        candidates.append(
            {
                "name": _compact_mesh_name(mesh["mesh_path"]),
                "overlay": _data_uri(overlay_path),
                "score": mesh.get("score"),
                "silhouette_score": mesh.get("silhouette_score"),
                "photometric_score": mesh.get("photometric_score"),
                "photometric_rmse_cosine": mesh.get("photometric_rmse_cosine"),
                "photometric_coverage": mesh.get("photometric_coverage"),
                "thermal_score": mesh.get("thermal_score"),
                "predicted_thermal_flux_w_m2": mesh.get("predicted_thermal_flux_w_m2"),
                "thermal_absolute_error_w_m2": mesh.get("thermal_absolute_error_w_m2"),
                "symmetric_chamfer_px": mesh.get("symmetric_chamfer_px"),
                "iou": mesh.get("iou"),
                "area_ratio": mesh.get("area_ratio"),
                "yaw_deg": mesh.get("yaw_deg"),
                "pitch_deg": mesh.get("pitch_deg"),
                "roll_deg": mesh.get("roll_deg"),
                "scale_px_per_unit": mesh.get("scale_px_per_unit"),
                "translation_px": mesh.get("translation_px"),
                "translation_seed_px": mesh.get("translation_seed_px"),
                "final_alignment_margin": mesh.get("final_alignment_margin"),
                "search_score_margin": mesh.get("search_score_margin"),
            }
        )
    if not candidates:
        raise ValueError("report has no usable overlay images")

    payload = json.dumps(
        {
            "image": image_uri,
            "observed_mask": observed_mask_uri,
            "candidates": candidates,
            "segmentation": report.get("segmentation", {}),
            "alignment_model": report.get("alignment_model"),
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Asteroid outline inspection</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; padding: 24px; background: Canvas; color: CanvasText; }}
main {{ max-width: 1120px; margin: 0 auto; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
.subtle {{ color: GrayText; margin: 0 0 18px; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center; margin-bottom: 18px; }}
label {{ font-weight: 600; }}
select {{ font: inherit; padding: 6px 8px; }}
.views {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
figure {{ margin: 0; }}
figcaption {{ font-weight: 600; margin: 0 0 7px; }}
img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: contain; image-rendering: auto; background: #111; }}
.details {{ margin-top: 18px; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px 18px; }}
.metric {{ border-top: 1px solid GrayText; padding-top: 7px; }}
.metric span {{ display: block; color: GrayText; font-size: 12px; }}
.metric strong {{ font-variant-numeric: tabular-nums; }}
.legend {{ margin-top: 16px; color: GrayText; font-size: 13px; }}
.red {{ color: #e55; }} .green {{ color: #3c6; }} .yellow {{ color: #edc83c; }}
@media (max-width: 760px) {{ .views, .details {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>Asteroid outline inspection</h1>
<p class="subtle">See what the matcher is comparing before trusting the score.</p>
<div class="toolbar">
  <label for="candidate">Candidate</label>
  <select id="candidate" aria-label="Candidate mesh"></select>
  <span id="segmentation" class="subtle"></span>
</div>
<section class="views" aria-label="Alignment views">
  <figure><figcaption>1. Input image</figcaption><img id="image" alt="Input observation"></figure>
  <figure><figcaption>2. Observed silhouette</figcaption><img id="mask" alt="Observed binary silhouette"></figure>
  <figure><figcaption>3. Fitted outline overlay</figcaption><img id="overlay" alt="Observed and fitted outline overlay"></figure>
</section>
<section class="details" aria-live="polite">
  <div class="metric"><span>Final score</span><strong id="score">—</strong></div>
  <div class="metric"><span>Silhouette score</span><strong id="silhouetteScore">—</strong></div>
  <div class="metric"><span>Photometric score</span><strong id="photometricScore">—</strong></div>
  <div class="metric"><span>Photometric cosine RMSE</span><strong id="photometricRmse">—</strong></div>
  <div class="metric"><span>Photometric coverage</span><strong id="photometricCoverage">—</strong></div>
  <div class="metric"><span>Thermal relative error</span><strong id="thermalScore">—</strong></div>
  <div class="metric"><span>Predicted thermal flux</span><strong id="thermalFlux">—</strong></div>
  <div class="metric"><span>Thermal absolute error</span><strong id="thermalError">—</strong></div>
  <div class="metric"><span>Symmetric chamfer</span><strong id="chamfer">—</strong></div>
  <div class="metric"><span>IoU</span><strong id="iou">—</strong></div>
  <div class="metric"><span>Area ratio</span><strong id="area">—</strong></div>
  <div class="metric"><span>Yaw / pitch / roll</span><strong id="pose">—</strong></div>
  <div class="metric"><span>Scale (px / unit)</span><strong id="scale">—</strong></div>
  <div class="metric"><span>Translation (px)</span><strong id="translation">—</strong></div>
  <div class="metric"><span>Chamfer seed correction</span><strong id="seed">—</strong></div>
  <div class="metric"><span>Final margin</span><strong id="finalMargin">—</strong></div>
  <div class="metric"><span>Search margin</span><strong id="searchMargin">—</strong></div>
</section>
<p class="legend"><span class="red">Red</span> = observed contour · <span class="green">green</span> = fitted mesh contour · <span class="yellow">yellow</span> = overlap</p>
</main>
<script>
const DATA = {payload};
const select = document.getElementById('candidate');
const fmt = (value, digits = 4) => value == null ? '—' : Number(value).toFixed(digits);
const fmtPair = (value) => value == null ? '—' : value.map((x) => fmt(x, 2)).join(', ');
DATA.candidates.forEach((item, index) => {{
  const option = document.createElement('option');
  option.value = index;
  option.textContent = item.name;
  select.appendChild(option);
}});
document.getElementById('image').src = DATA.image;
document.getElementById('mask').src = DATA.observed_mask || DATA.image;
const update = () => {{
  const item = DATA.candidates[Number(select.value)];
  document.getElementById('overlay').src = item.overlay;
  document.getElementById('score').textContent = fmt(item.score, 6);
  document.getElementById('silhouetteScore').textContent = fmt(item.silhouette_score, 6);
  document.getElementById('photometricScore').textContent = fmt(item.photometric_score, 6);
  document.getElementById('photometricRmse').textContent = fmt(item.photometric_rmse_cosine, 6);
  document.getElementById('photometricCoverage').textContent = item.photometric_coverage == null ? '—' : `${{fmt(100 * item.photometric_coverage, 1)}}%`;
  document.getElementById('thermalScore').textContent = item.thermal_score == null ? '—' : `${{fmt(100 * item.thermal_score, 2)}}%`;
  document.getElementById('thermalFlux').textContent = item.predicted_thermal_flux_w_m2 == null ? '—' : `${{Number(item.predicted_thermal_flux_w_m2).toExponential(5)}} W/m²`;
  document.getElementById('thermalError').textContent = item.thermal_absolute_error_w_m2 == null ? '—' : `${{Number(item.thermal_absolute_error_w_m2).toExponential(5)}} W/m²`;
  document.getElementById('chamfer').textContent = item.symmetric_chamfer_px == null ? '—' : `${{fmt(item.symmetric_chamfer_px, 3)}} px`;
  document.getElementById('iou').textContent = fmt(item.iou, 4);
  document.getElementById('area').textContent = fmt(item.area_ratio, 4);
  document.getElementById('pose').textContent = [item.yaw_deg, item.pitch_deg, item.roll_deg].map((x) => `${{fmt(x, 1)}}°`).join(' / ');
  document.getElementById('scale').textContent = fmt(item.scale_px_per_unit, 2);
  document.getElementById('translation').textContent = fmtPair(item.translation_px);
  document.getElementById('seed').textContent = fmtPair(item.translation_seed_px);
  document.getElementById('finalMargin').textContent = fmt(item.final_alignment_margin, 6);
  document.getElementById('searchMargin').textContent = fmt(item.search_score_margin, 6);
}};
select.addEventListener('change', update);
const segmentation = DATA.segmentation || {{}};
document.getElementById('segmentation').textContent = `Segmentation: ${{segmentation.method || 'unknown'}}${{segmentation.requires_manual_review ? ' · manual review required' : ''}}`;
update();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_viewer(args.report, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
