"""Repoint STAC item assets from one product root to another.

Used after a COG conversion: the item still says the asset lives in
masked/tiles/geotiff, the COG now sits in masked/tiles/cog, and the publishing
path (deploy/source_coop, src.kind=stac) reads hrefs, so nothing reaches
source.coop until the catalog is moved over.

Rewrites href only, and only when the destination file actually exists -- a
half-converted collection is the normal state during a quantile-by-quantile
rollout, so an item can legitimately end up with Q1 pointing at cog/ while
Q0/Q2 still point at geotiff/. Anything else in the item is left alone.

Defaults to dry_run=true: it prints what it would change and touches nothing.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Tuple

FILE_URI = "file://"


def _retarget_item(item_path: Path, product_root: Path, src_root: str,
                   dst_root: str, asset_glob: str, dry_run: bool,
                   backup_dir: Path) -> Tuple[str, int, int]:
    """Rewrite one item. Returns (outcome, n_moved, n_target_missing)."""
    try:
        with open(item_path) as fh:
            item = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable: {type(exc).__name__}", 0, 0

    src_prefix = str(product_root / src_root) + os.sep
    moved = missing = 0

    for asset in item.get("assets", {}).values():
        href = asset.get("href", "")
        path = href[len(FILE_URI):] if href.startswith(FILE_URI) else href
        if not path.startswith(src_prefix):
            continue
        if not fnmatch(os.path.basename(path), asset_glob):
            continue
        target = product_root / dst_root / os.path.relpath(path, src_prefix)
        # An asset with no COG yet stays where it is: pointing the catalog at
        # a file that does not exist would break the upload, not delay it.
        if not target.is_file() or target.stat().st_size == 0:
            missing += 1
            continue
        asset["href"] = FILE_URI + str(target)
        moved += 1

    if not moved:
        return ("nothing to move" if not missing else "target missing"), 0, missing

    if dry_run:
        return "would rewrite", moved, missing

    # Back up the original once, then swap atomically so a crash mid-write
    # cannot leave a truncated item JSON behind.
    backup = backup_dir / item_path.name
    if not backup.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item_path, backup)
    tmp = item_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(item, fh, indent=2)
    os.replace(tmp, item_path)
    return "rewritten", moved, missing


def retarget_stac_assets(
    catalog: str = None,
    product_root: str = None,
    src_root: str = "masked/tiles/geotiff",
    dst_root: str = "masked/tiles/cog",
    asset_glob: str = "RH*_Q1.tif",
    backup_dir: str = None,
    dry_run: bool = True,
    workers: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Point matching assets at their converted copies.

    Args:
        catalog: STAC collection dir, one <ITEM_ID>/<ITEM_ID>.json per tile.
        product_root: dir the roots below are relative to.
        src_root: only hrefs under here are candidates.
        dst_root: where the converted copy lives; the path tail is preserved.
        asset_glob: filename filter, e.g. RH*_Q1.tif for a median-only pass.
        backup_dir: originals of every rewritten item land here.
        dry_run: true (default) reports without writing.
        workers: threads for item IO.

    Returns:
        Per-outcome item counts plus assets moved and assets still missing.
    """
    catalog = Path(catalog).expanduser()
    product_root = Path(product_root).expanduser()
    backup_dir = Path(backup_dir).expanduser() if backup_dir else \
        catalog.parent / f"{catalog.name}_backup"
    if not catalog.is_dir():
        raise RuntimeError(f"STAC collection dir not found: {catalog}")
    if not (product_root / dst_root).is_dir():
        raise RuntimeError(f"destination root not found: {product_root / dst_root}")

    items = sorted(d.name for d in catalog.iterdir() if d.is_dir())
    print(f"{len(items)} items in {catalog.name}: {src_root} -> {dst_root} "
          f"[{asset_glob}]{'  (DRY RUN)' if dry_run else ''}")

    def one(item_id: str):
        return _retarget_item(catalog / item_id / f"{item_id}.json",
                              product_root, src_root, dst_root, asset_glob,
                              dry_run, backup_dir)

    outcomes: Counter = Counter()
    moved = missing = 0
    with ThreadPoolExecutor(workers) as pool:
        for outcome, n_moved, n_missing in pool.map(one, items):
            outcomes[outcome] += 1
            moved += n_moved
            missing += n_missing

    for outcome, n in outcomes.most_common():
        print(f"  {outcome:<20s} {n:6d} items")
    print(f"  assets repointed     {moved:6d}")
    print(f"  assets with no COG   {missing:6d}  (left pointing at {src_root})")
    if dry_run:
        print("  dry run — nothing written; re-run with run.dry_run=false")
    else:
        print(f"  originals backed up to {backup_dir}")

    return {"items": len(items), "outcomes": dict(outcomes),
            "assets_repointed": moved, "assets_missing_cog": missing,
            "backup_dir": str(backup_dir), "dry_run": dry_run}
