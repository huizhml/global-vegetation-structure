"""Audit a STAC collection against the COG copies that exist on disk.

The 2020 VSM collection is not homogeneous: its items point at three different
roots under the product tree (original/tiles/cog, original/tiles/geotiff,
masked/tiles/geotiff), one root per item. Only the item JSON knows where a tile
actually lives, so anything that walks the product directory instead of the
catalog will pick the wrong copy.

This op reports how the items are distributed across roots and, for the items
that point at `src_root`, whether a complete COG already sits under `cog_root`.
It writes two tile worklists so the SLURM array can be sized off real numbers:

    <prefix>_need_cog.txt   tiles whose COG is missing or incomplete
    <prefix>_have_cog.txt   tiles whose COG has every asset the item lists

"Complete" means every asset filename in the item exists in the COG dir. It is
deliberately a filename check, not a content check -- a half-finished
conversion leaves an empty tile dir, which is exactly what this catches.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _read_item(catalog: Path, item_id: str,
               product_root: Path) -> Tuple[str, str, Set[str]]:
    """Return (item_id, href root relative to product_root, asset filenames)."""
    path = catalog / item_id / f"{item_id}.json"
    try:
        with open(path) as fh:
            assets = json.load(fh)["assets"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return item_id, f"<unreadable: {type(exc).__name__}>", set()

    roots, names = set(), set()
    for asset in assets.values():
        href = asset.get("href", "")
        if href.startswith("file://"):
            href = href[len("file://"):]
        roots.add(os.path.relpath(os.path.dirname(os.path.dirname(href)),
                                  product_root))
        names.add(os.path.basename(href))
    # Items mixing roots would break the one-root-per-tile assumption below.
    return item_id, "|".join(sorted(roots)), names


def audit_stac_cog(
    catalog: str = None,
    product_root: str = None,
    out_dir: str = None,
    src_root: str = "masked/tiles/geotiff",
    cog_root: str = "masked/tiles/cog",
    asset_glob: str = "*.tif",
    prefix: str = "masked",
    workers: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Classify STAC items by href root and check COG availability.

    Args:
        catalog: STAC collection dir, one <ITEM_ID>/<ITEM_ID>.json per tile.
        product_root: the dir the href roots are reported relative to, e.g.
            ~/data/gvs/products/vsm/2020.
        out_dir: where the two tile worklists are written.
        src_root: items pointing here are the ones that need a COG.
        cog_root: where the COG copy of a `src_root` tile is expected to live.
        asset_glob: only these asset filenames count, e.g. RH*_Q1.tif to size
            a median-quantile-first pass. A tile is "complete" when every
            matching asset exists under cog_root.
        prefix: filename prefix for the two worklists.
        workers: threads for reading item JSONs (NFS latency-bound, not CPU).

    Returns:
        Counts per root plus the size of each worklist.
    """
    catalog = Path(catalog).expanduser()
    product_root = Path(product_root).expanduser()
    out_dir = Path(out_dir).expanduser()
    if not catalog.is_dir():
        raise RuntimeError(f"STAC collection dir not found: {catalog}")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(d.name for d in catalog.iterdir() if d.is_dir())
    if not items:
        raise RuntimeError(f"no items under {catalog}")

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(
            lambda i: _read_item(catalog, i, product_root), items))

    by_root = Counter(root for _, root, _ in rows)
    print(f"{len(rows)} items in {catalog.name}, by href root:")
    for root, n in by_root.most_common():
        print(f"  {root:<30s} {n:6d}")

    def cog_names(tile: str) -> Set[str]:
        try:
            return {f for f in os.listdir(product_root / cog_root / tile)
                    if fnmatch(f, asset_glob)}
        except OSError:
            return set()

    need: List[str] = []
    have: List[str] = []
    targets = [(i, {n for n in names if fnmatch(n, asset_glob)})
               for i, root, names in rows if root == src_root]
    with ThreadPoolExecutor(workers) as pool:
        for (item_id, names), on_disk in zip(
                targets, pool.map(lambda t: cog_names(t[0].split("_", 1)[0]),
                                  targets)):
            tile = item_id.split("_", 1)[0]
            (have if names and names <= on_disk else need).append(tile)

    need_fp = out_dir / f"{prefix}_need_cog.txt"
    have_fp = out_dir / f"{prefix}_have_cog.txt"
    need_fp.write_text("\n".join(sorted(need)) + "\n" if need else "")
    have_fp.write_text("\n".join(sorted(have)) + "\n" if have else "")

    print(f"\nitems pointing at {src_root} (assets matching {asset_glob}): "
          f"{len(targets)}")
    print(f"  COG complete : {len(have):6d}  -> {have_fp}")
    print(f"  COG missing  : {len(need):6d}  -> {need_fp}")

    return {"items": len(rows), "by_root": dict(by_root),
            "need_cog": len(need), "have_cog": len(have),
            "need_cog_file": str(need_fp), "have_cog_file": str(have_fp)}
