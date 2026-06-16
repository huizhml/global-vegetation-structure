"""Generate meta_lvis.csv for the Gabon2016 LVIS evaluation.

Lists every <tile>.parquet under the by-tile dir and writes one row per tile
with Country/Tile name/Year/Ours columns, matching the format on_als.evaluate
consumes via `meta_<ref_col>.csv`.
"""
from pathlib import Path
import pandas as pd

BY_TILE_DIR = Path(
    '~/data/gvs/evaluation/with_airborne_lidar/LVIS2_Gabon2016_by_s2_tile'
).expanduser()
OUT_FILE = BY_TILE_DIR.parent / 'meta_lvis_profile.csv'

tiles = sorted(p.stem for p in BY_TILE_DIR.glob('*.parquet'))
df = pd.DataFrame({
    'Country': 'Gabon',
    'Tile name': tiles,
    'Year': 2016,
    'Ours': 2017,
})
df.to_csv(OUT_FILE, index=False)
print(f'Wrote {len(df)} rows to {OUT_FILE}')
