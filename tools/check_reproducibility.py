from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def check_reproducibility(
        ref_dir: str,
        new_dir: str,
        save_dir: str,
        ref_label: str = 'ref',
        new_label: str = 'new',
        nodata: int = 32767,
        block_rows: int = 1024,
        **kwargs) -> pd.DataFrame:
    """Check that a new version of the deploy code reproduces the reference version's
    per-RH prediction GeoTIFFs (e.g. RH98_Q1.tif) on the same tile, file by file.

    ref_dir holds the output of the reference (old) code, new_dir the output of
    the code under test. ref_label / new_label name them in the report, e.g. the
    git refs that produced them. Rasters are read in row blocks so a full
    10980x10980 tile never sits in memory twice.

    Writes <save_dir>/per_file_diff.csv with one row per file:
      n_diff_pixels      pixels whose values differ (nodata included)
      n_nodata_mismatch  pixels that are nodata in one version but not the other
      max_abs_diff       largest |new - ref| where both are valid (stored units, i.e. dm)
      valid_frac_ref/new fraction of valid pixels in each
    and prints a verdict line ending in "-> REPRODUCED" or "-> NOT REPRODUCED".
    """
    ref_dir, new_dir = Path(ref_dir).expanduser(), Path(new_dir).expanduser()
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reference ({ref_label}): {ref_dir}')
    print(f'New       ({new_label}): {new_dir}')
    names_ref = {p.name for p in ref_dir.glob('*.tif')}
    names_new = {p.name for p in new_dir.glob('*.tif')}
    rows = []
    for name in sorted(names_ref | names_new):
        row = {'file': name, 'in_ref': name in names_ref, 'in_new': name in names_new}
        if row['in_ref'] and row['in_new']:
            row.update(_compare_pair(ref_dir / name, new_dir / name, nodata, block_rows))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_fp = save_dir / 'per_file_diff.csv'
    df.to_csv(out_fp, index=False)

    n_missing = int((~df['in_ref'] | ~df['in_new']).sum()) if len(df) else 0
    both = df[df['in_ref'] & df['in_new']] if len(df) else df
    n_files_diff = int((both['n_diff_pixels'] > 0).sum()) if len(both) else 0
    reproduced = n_missing == 0 and n_files_diff == 0 and len(both) > 0
    print(f'{len(both)} files compared, {n_missing} missing in one version, '
          f'{n_files_diff} with differing pixels -> {"REPRODUCED" if reproduced else "NOT REPRODUCED"}')
    if n_files_diff:
        print(both.sort_values('n_diff_pixels', ascending=False).head(10).to_string(index=False))
    print(f'Per-file differences written to {out_fp}')
    return df


def _compare_pair(fp_ref, fp_new, nodata, block_rows):
    with rasterio.open(fp_ref) as ref, rasterio.open(fp_new) as new:
        if (ref.width, ref.height) != (new.width, new.height):
            return {'shape_mismatch': True}
        n_diff = n_nodata_mismatch = n_valid_ref = n_valid_new = 0
        max_abs_diff = 0
        for row_off in range(0, ref.height, block_rows):
            win = Window(0, row_off, ref.width, min(block_rows, ref.height - row_off))
            v_ref = ref.read(1, window=win).astype(np.int32)
            v_new = new.read(1, window=win).astype(np.int32)
            valid_ref, valid_new = v_ref != nodata, v_new != nodata
            n_diff += int((v_ref != v_new).sum())
            n_nodata_mismatch += int((valid_ref != valid_new).sum())
            n_valid_ref += int(valid_ref.sum())
            n_valid_new += int(valid_new.sum())
            both = valid_ref & valid_new
            if both.any():
                max_abs_diff = max(max_abs_diff, int(np.abs(v_new[both] - v_ref[both]).max()))
        n_pixels = ref.width * ref.height
    return {'shape_mismatch': False, 'n_diff_pixels': n_diff, 'n_nodata_mismatch': n_nodata_mismatch,
            'max_abs_diff': max_abs_diff,
            'valid_frac_ref': n_valid_ref / n_pixels, 'valid_frac_new': n_valid_new / n_pixels}
