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

from deploy.source_coop.common import norm_patterns
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _read_item(catalog: Path, item_id: str,
               product_root: Path) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """Return (item_id, {asset filename: href root}, {asset filename: href}).

    Per asset, not per item. A partial rollout leaves an item straddling two
    roots -- Q1 already moved to cog/ while Q0/Q2 still sit on geotiff/ -- and
    collapsing that into one label per item made every such item unclassifiable
    (it matched neither "…/cog" nor "…/geotiff", so the worklists came out
    empty). What decides whether a tile can be published is where each asset we
    are about to publish points, so that is what gets recorded.
    """
    path = catalog / item_id / f"{item_id}.json"
    try:
        with open(path) as fh:
            assets = json.load(fh)["assets"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return item_id, {f"<unreadable: {type(exc).__name__}>": "?"}, {}

    roots, hrefs = {}, {}
    for asset in assets.values():
        href = asset.get("href", "")
        if href.startswith("file://"):
            href = href[len("file://"):]
        if not href:
            continue
        name = os.path.basename(href)
        roots[name] = os.path.relpath(
            os.path.dirname(os.path.dirname(href)), product_root)
        hrefs[name] = href
    return item_id, roots, hrefs


def audit_stac_cog(
    catalog: str = None,
    product_root: str = None,
    out_dir: str = None,
    cog_dirname: str = "cog",
    raw_dirname: str = "geotiff",
    asset_glob="*.tif",
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
        cog_dirname: the path segment marking a converted root, e.g. the
            "cog" in original/tiles/cog. Both product branches (original/ and
            masked/) are handled -- an asset is publishable from whichever
            branch its own href names.
        raw_dirname: the unconverted counterpart, e.g. "geotiff". Swapping this
            segment for cog_dirname gives where the COG of a raw asset belongs,
            which is what separates "needs stac_retarget" from "needs
            translate".
        asset_glob: only these asset filenames count -- a single pattern or a
            list (one per RH for the key-RH rollout). Use RH<n>_Q*.tif, not a
            bare prefix: RH10* also matches RH100. A tile counts as done only
            when every matching asset, all quantiles included, has a COG. E.g. to size
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

    asset_pats = norm_patterns(asset_glob)
    items = sorted(d.name for d in catalog.iterdir() if d.is_dir())
    if not items:
        raise RuntimeError(f"no items under {catalog}")

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(
            lambda i: _read_item(catalog, i, product_root), items))

    # Per-asset classification. Only the assets we are about to publish count;
    # the other ~250 RHs per tile are irrelevant to this pass.
    by_root: Counter = Counter()
    ready: List[str] = []
    need_retarget: List[str] = []      # COG exists, href still on raw
    need_translate: List[str] = []     # no COG on disk yet
    no_match: List[str] = []           # item carries none of the wanted assets

    def classify(row) -> Tuple[str, str]:
        item_id, roots, hrefs = row
        tile = item_id.split("_", 1)[0]
        names = [n for n in roots if any(fnmatch(n, p) for p in asset_pats)]
        if not names:
            return tile, "no_match"

        verdict = "ready"
        for name in names:
            root = roots[name]
            by_root[root] += 1
            if f"/{cog_dirname}" in f"/{root}" and os.path.exists(hrefs[name]):
                continue
            # Not publishable as-is. Does the COG merely need pointing at?
            cog_href = hrefs[name].replace(f"/{raw_dirname}/", f"/{cog_dirname}/")
            if cog_href != hrefs[name] and os.path.exists(cog_href):
                verdict = "retarget" if verdict != "translate" else verdict
            else:
                verdict = "translate"
        return tile, verdict

    with ThreadPoolExecutor(workers) as pool:
        for tile, verdict in pool.map(classify, rows):
            {"ready": ready, "retarget": need_retarget,
             "translate": need_translate, "no_match": no_match}[verdict].append(tile)

    need = sorted(set(need_retarget) | set(need_translate))
    have = sorted(set(ready))

    print(f"\nassets matching the {len(asset_pats)} pattern(s), by href root:")
    for root, n in by_root.most_common():
        flag = "" if f"/{cog_dirname}" in f"/{root}" else "   <- not a COG root"
        print(f"  {root:<30s} {n:8d}{flag}")

    need_fp = out_dir / f"{prefix}_need_cog.txt"
    have_fp = out_dir / f"{prefix}_have_cog.txt"
    need_fp.write_text("\n".join(need) + "\n" if need else "")
    have_fp.write_text("\n".join(have) + "\n" if have else "")

    print(f"\ntiles ({len(rows)} items):")
    print(f"  publishable now  : {len(have):6d}  -> {have_fp}")
    print(f"  need stac_retarget: {len(set(need_retarget)):6d}  "
          f"(COG on disk, href still on {raw_dirname}/)")
    print(f"  need translate    : {len(set(need_translate)):6d}  "
          f"(no COG on disk)      -> {need_fp}")
    if no_match:
        print(f"  no matching asset : {len(no_match):6d}  "
              f"(item carries none of the wanted RHs)")
    if need_retarget:
        print("\n  -> python -m tools.run run=stac_retarget run.dry_run=false")

    return {"items": len(rows), "by_root": dict(by_root),
            "have_cog": len(have), "need_cog": len(need),
            "need_retarget": len(set(need_retarget)),
            "need_translate": len(set(need_translate)),
            "no_match": len(no_match),
            "need_cog_file": str(need_fp), "have_cog_file": str(have_fp)}
