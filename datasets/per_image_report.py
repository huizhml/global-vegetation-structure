"""
Report folder for per-image (pre-median-aggregation) debug predictions.

Layout written by `write_per_image_report`:

    <YYYY-MM-DD>_per-image-predictions_<tile>_<year>/
    ├── README.md      purpose, inputs, conclusions (to fill in)
    ├── index.html     browse all figures
    ├── summary.csv    per-image statistics
    ├── figures/       <tile>_<date>_img<idx>.png  (RGB | RH98 quicklook)
    └── tifs/          <tile>_<date>_img<idx>.tif  (full-res RH98, decimeters)
"""
import html
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUICKLOOK_SIZE = 1024  # px, longest side of the downsampled quicklook
RH_MAX_M = 50


def per_image_dir_name(tile_id: str, year) -> str:
    return f'{date.today().isoformat()}_per-image-predictions_{tile_id}_{year}'


def _stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) S2 DN -> (H, W, 3) in [0, 1] with a 2-98 percentile stretch, nodata (0) kept black."""
    rgb = rgb.astype(np.float32)
    valid = rgb.sum(axis=0) > 0
    if not valid.any():
        return np.zeros(rgb.shape[1:] + (3,), dtype=np.float32)
    lo, hi = np.percentile(rgb[:, valid], [2, 98])
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb[:, ~valid] = 0
    return np.moveaxis(rgb, 0, -1)


def _plot_quicklook(rgb, rh98_m, title, out_fp: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(_stretch_rgb(rgb))
    axes[0].set_title('S2 RGB')
    im = axes[1].imshow(rh98_m, cmap='viridis', vmin=0, vmax=RH_MAX_M, interpolation='nearest')
    axes[1].set_title('RH98 (m)')
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(out_fp, dpi=100)
    plt.close(fig)


def _write_index_html(out_dir: Path, summary: pd.DataFrame, title: str):
    cards = []
    for row in summary.itertuples():
        caption = (f'<b>{html.escape(row.date)}</b> · img {row.img_idx}<br>'
                   f'<code>{html.escape(row.id)}</code><br>'
                   f'valid {row.valid_frac:.1%} · RH98 median {row.rh98_median_m:.1f} m')
        cards.append(f'<figure><a href="figures/{row.name}.png"><img src="figures/{row.name}.png" loading="lazy"></a>'
                     f'<figcaption>{caption}</figcaption></figure>')
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
figure {{ margin: 0; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 13px; margin-top: 6px; line-height: 1.4; }}
code {{ font-size: 11px; word-break: break-all; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p><a href="README.md">README.md</a> · <a href="summary.csv">summary.csv</a> · {len(summary)} images</p>
<div class="grid">
{chr(10).join(cards)}
</div></body></html>
"""
    (out_dir / 'index.html').write_text(page)


def _write_readme(out_dir: Path, summary: pd.DataFrame, inputs: dict, title: str):
    input_lines = '\n'.join(f'- **{k}**: `{v}`' for k, v in inputs.items())
    readme = f"""# {title}

## Purpose

Debug per-image predictions *before* the post-inference median aggregation
(`save_intermediate_tif=True`). Each input image's RH98 (Q1) prediction is kept
separately, with the per-image SCL cloud/snow and ESA snow masks applied; the
water / built-up masks are the majority vote over all images, as in production.

## Inputs

{input_lines}
- **n_images**: {len(summary)}
- **dates**: {summary['date'].min()} – {summary['date'].max()}

## Outputs

- `tifs/` — full-resolution RH98 GeoTIFF per image, int16 decimeters, nodata = {inputs.get('nodata_value')}
- `figures/` — RGB | RH98 quicklook per image
- `summary.csv` — per-image valid fraction and RH98 statistics
- `index.html` — browse all figures

## Conclusions

_TODO_
"""
    (out_dir / 'README.md').write_text(readme)


def write_per_image_report(out_dir: Path, meta: pd.DataFrame, tiff_writers, s2_array, nodata_value: int,
                           inputs: dict, title: str):
    """
    Args:
        out_dir: report folder; `tifs/` must already hold the per-image GeoTIFFs.
        meta: one row per output image (columns: img_idx, date, id, name), same order as `tiff_writers`.
        tiff_writers: open GDAL datasets of the per-image tifs.
        s2_array: zarr array (n_images, bands, H, W); bands 1:4 are B04, B03, B02.
        inputs: key/value pairs listed under "Inputs" in the README.
    """
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    height, width = s2_array.shape[2:]
    step = max(1, int(np.ceil(max(height, width) / QUICKLOOK_SIZE)))

    rows = []
    for row, writer in zip(meta.itertuples(index=False), tiff_writers):
        writer.FlushCache()
        pred = writer.GetRasterBand(1).ReadAsArray()
        valid = pred != nodata_value
        rh98_m = pred[valid].astype(np.float32) / 10
        stats = dict(
            valid_frac=float(valid.mean()),
            rh98_mean_m=float(rh98_m.mean()) if rh98_m.size else np.nan,
            rh98_median_m=float(np.median(rh98_m)) if rh98_m.size else np.nan,
            rh98_p95_m=float(np.percentile(rh98_m, 95)) if rh98_m.size else np.nan,
        )
        quicklook = np.where(valid, pred / 10, np.nan)[::step, ::step]
        rgb = s2_array[row.img_idx, 1:4, ::step, ::step]
        _plot_quicklook(rgb, quicklook, f'{row.name}  ·  {row.id}  ·  valid {stats["valid_frac"]:.1%}',
                        fig_dir / f'{row.name}.png')
        rows.append({**row._asdict(), **stats, 'tif': f'tifs/{row.name}.tif', 'figure': f'figures/{row.name}.png'})
        print(f'[per-image report] {row.name}: valid {stats["valid_frac"]:.1%}, '
              f'RH98 median {stats["rh98_median_m"]:.1f} m')

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'summary.csv', index=False)
    _write_index_html(out_dir, summary, title)
    _write_readme(out_dir, summary, inputs, title)
    print(f'[per-image report] written to {out_dir}')
