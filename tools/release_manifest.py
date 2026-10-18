"""Write the manifest of a released VSM product year, one row per tile.

A release picks, per tile, where its files live under the product tree. For
2020 that choice lives only in the STAC collection published to Source Coop
(vsm_local_masked): its items point at three roots (original/tiles/cog,
original/tiles/geotiff, masked/tiles/geotiff) and, after the Q1-first
stac_retarget, a tile's Q1 sits in cog/ while its Q0/Q2 are still in geotiff/.
2024 masked is in no collection (masked/tiles/cog/<TILE>/RH*_Q*.tif). The
manifest flattens both into one table, so a download or an analysis never has
to read the catalog:

    <products_root>/<year>/<release>.tsv

    tile_id  q0_root                    q1_root                q2_root  file_suffix  n_files  size_bytes
    32UNG    2020/masked/tiles/geotiff  2020/masked/tiles/cog  ...      .tif         303      53412345678

The 101 files of quantile q are <q{q}_root>/<tile_id>/RH{0..100}_Q{q}<file_suffix>,
roots relative to products_root. A root names the product version and format,
so counting the q*_root column counts tiles per version. That every RH of one
quantile shares a dir, named after the tile, is checked for every tile; a tile
breaking it stops the run, nothing is written. n_files counts the files that
exist (303 when complete), size_bytes sums them.

The 2020 manifest goes stale whenever stac_retarget repoints hrefs: regenerate
it after every `stac_retarget run.dry_run=false`.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

FILE_URI = "file://"
RHS = range(101)
QS = (0, 1, 2)
N_FILES = len(RHS) * len(QS)
RH_Q = re.compile(r"^RH(\d+)_Q(\d)(\..+)$")
MGRS_TILE = re.compile(r"\d{2}[A-Z]{3}")
COLUMNS = ["tile_id", "q0_root", "q1_root", "q2_root", "file_suffix", "n_files", "size_bytes"]


def _paths_from_item(item_dir: Path) -> Tuple[str, List[str]]:
    item_id = item_dir.name
    with open(item_dir / f"{item_id}.json") as fh:
        assets = json.load(fh)["assets"]
    paths = [a["href"][len(FILE_URI):] if a["href"].startswith(FILE_URI) else a["href"]
             for a in assets.values()]
    return item_id.split("_", 1)[0], paths


def _paths_from_dir(tile_dir: Path) -> Tuple[str, List[str]]:
    return tile_dir.name, [str(f) for f in tile_dir.glob("RH*_Q*.tif")]


def _tile_row(tile_id: str, paths: List[str], products_root: Path) -> Tuple[List[str], str]:
    """Return (row, error); error is "" when the tile fits one dir per quantile."""
    if not MGRS_TILE.fullmatch(tile_id):
        return [], f"{tile_id}: not an MGRS tile id (old <TILE>_cog dir? see run=rename_tile_dirs)"
    dirs = {q: set() for q in QS}
    suffixes, names = set(), set()
    for path in paths:
        m = RH_Q.match(os.path.basename(path))
        if not m or int(m[2]) not in dirs:
            return [], f"{tile_id}: unexpected file {path}"
        dirs[int(m[2])].add(os.path.relpath(os.path.dirname(path), products_root))
        suffixes.add(m[3])
        names.add((int(m[1]), int(m[2])))
    expected = {(rh, q) for rh in RHS for q in QS}
    if names != expected:
        return [], f"{tile_id}: {len(expected - names)} RH/Q missing, {len(names - expected)} unexpected"
    spread = {f"Q{q}": sorted(d) for q, d in dirs.items() if len(d) != 1}
    if spread:
        return [], f"{tile_id}: quantile spread over several dirs {spread}"
    if len(suffixes) != 1:
        return [], f"{tile_id}: mixed file suffixes {sorted(suffixes)}"

    q_dirs = [dirs[q].pop() for q in QS]
    if any(os.path.basename(d) != tile_id for d in q_dirs):
        return [], f"{tile_id}: tile dir not named after the tile {q_dirs}"
    q_roots = [os.path.dirname(d) for d in q_dirs]
    suffix = suffixes.pop()
    n_files = n_bytes = 0
    for q, d in zip(QS, q_dirs):
        for rh in RHS:
            try:
                n_bytes += os.stat(products_root / d / f"RH{rh}_Q{q}{suffix}").st_size
                n_files += 1
            except FileNotFoundError:
                pass
    return [tile_id, *q_roots, suffix, str(n_files), str(n_bytes)], ""


def write_release_manifest(
    year: int = None,
    release: str = "released_v0",
    sources: Mapping = None,
    products_root: str = None,
    out_fp: str = None,
    workers: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Collect where each tile of one release year lives into a TSV manifest.

    Args:
        year: product year; picks the entry of `sources`.
        release: release name, only used in out_fp.
        sources: {year: {catalog: <STAC collection dir>} or {dir: <tiles dir>}}.
            catalog: the asset hrefs of the items <TILE>_<year>.
            dir: the RH*_Q*.tif in each tile subdir <TILE>/.
        products_root: dir the q*_root columns are relative to,
            e.g. ~/data/gvs/products/vsm.
        out_fp: manifest to write (replaced atomically).
        workers: threads for item reads and stats (NFS latency-bound).

    Returns:
        Tile and byte counts, plus the tiles with missing files.
    """
    source = (sources or {}).get(str(year))
    if not source:
        raise RuntimeError(f"no source for year {year} in sources: {list(sources or {})}")
    products_root = Path(products_root).expanduser()
    out_fp = Path(out_fp).expanduser()

    if source.get("catalog"):
        catalog = Path(source["catalog"]).expanduser()
        units = sorted(d for d in catalog.iterdir()
                       if d.is_dir() and d.name.endswith(f"_{year}"))
        read = _paths_from_item
        print(f"{len(units)} items of {year} in {catalog}")
    else:
        tiles_dir = Path(source["dir"]).expanduser()
        units = sorted(d for d in tiles_dir.iterdir() if d.is_dir())
        read = _paths_from_dir
        print(f"{len(units)} tile dirs in {tiles_dir}")
    if not units:
        raise RuntimeError(f"nothing to read for {year}: {source}")

    def one(unit: Path):
        return _tile_row(*read(unit), products_root)

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, units))
    errors = [e for _, e in results if e]
    if errors:
        raise RuntimeError(f"{len(errors)} tiles failed the layout check, "
                           f"nothing written:\n  " + "\n  ".join(errors[:10]))
    rows = sorted(r for r, _ in results)
    duplicated = [t for t, n in Counter(r[0] for r in rows).items() if n > 1]
    if duplicated:
        raise RuntimeError(f"tiles listed twice: {duplicated[:10]}")

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fp.with_suffix(".tsv.tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        fh.writelines("\t".join(r) + "\n" for r in rows)
    os.replace(tmp, out_fp)

    incomplete = [r[0] for r in rows if int(r[5]) != N_FILES]
    n_bytes = sum(int(r[6]) for r in rows)
    print(f"\n{len(rows)} tiles, {n_bytes / 1e12:.2f} TB -> {out_fp}")
    for q in QS:
        print(f"Q{q} tiles by root:")
        for root, n in Counter(r[1 + q] for r in rows).most_common():
            print(f"  {root:<35s} {n:7d}")
    if incomplete:
        print(f"WARNING: {len(incomplete)} tiles with files missing on disk "
              f"(n_files < {N_FILES}), e.g. {incomplete[:5]}")

    return {"tiles": len(rows), "bytes": n_bytes,
            "incomplete_tiles": incomplete, "out_fp": str(out_fp)}
