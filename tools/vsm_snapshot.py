"""Write what each VSM product version holds right now.

A version is a dir <products_root>/<year>/<version>/ with a tiles/ or mosaic/
(original, masked, blended, ...). Paths inside a version follow one layout
(tiles/<fmt>/<TILE>/RH*_Q*.tif, mosaic/...), so the tables record what exists,
not where:

    <year>/snapshot_versions.tsv        one row per version: what it holds
    <year>/<version>/snapshot.tsv       one row per tile
    <year>/<version>/snapshot_mosaic.tsv  one row per RH (if mosaic/ exists)

snapshot_versions.tsv
    version  tiles  cog_tiles  cog_q0  cog_q1  cog_q2  geotiff_tiles  geotiff_q0  geotiff_q1
    geotiff_q2  vrt_tiles  multiband_tiles  tile_bytes  mosaic_files  mosaic_variables
    mosaic_quantiles  mosaic_formats  mosaic_bytes  generated
<fmt>_tiles: tiles with any file in tiles/<fmt>/; <fmt>_q<q>: tiles with all
101 RH of Q<q> there. mosaic_*: the distinct values of the mosaic table
columns, comma-separated, RH runs as ranges (RH0-2,RH4-98: RH3 missing).
generated: UTC time this version was scanned; a run over some versions only
replaces their rows.

snapshot.tsv
    tile_id  cog_q0  cog_q1  cog_q2  geotiff_q0  geotiff_q1  geotiff_q2  multiband  vrt  other_files  size_bytes
    32UNG    101     101     101     0           0           0           0          0    0            ...
<fmt>_q<q> counts the RH*_Q<q> rasters (.tif or .cog.tif) in tiles/<fmt>/<TILE>/,
101 when complete; an old-style <TILE>_cog/ dir counts for <TILE>. multiband
counts tiles/multibands/<TILE>_*.tif, vrt the <TILE>*.vrt in any vrt* dir of
the version or of tiles/. other_files are the other files in the tile dirs
(.aux.xml sidecars not counted), size_bytes sums everything counted. The
columns are the same for every version, so tables of different versions join
on tile_id.

snapshot_mosaic.tsv
    variable  global_mosaic_Q1.cog.tif  global_mosaic_Q1.tif  ...  tmp_tiles_resampled_1km_Q1/  n_files  size_bytes  modified
    RH25      1                         1                          18143                        ...
    CR        ...
A mosaic name splits as <kind>[_<year>]_<variable>[_<quantile>].<format>,
variable RH<n> or CR. Each variable is a row and each <kind>[_<quantile>].<format>
a column (prefixed cog/ or geotiff/ for files in mosaic/<fmt>/), the cell the
number of such files. Other subdirs of mosaic/ (tmp_*) are columns ending in
"/", the cell the number of files inside. Names that do not split (README.md,
run.log, vrt_tmp/) are their own columns in a last row "other". n_files,
size_bytes and modified (latest, UTC) sum up the row.

A snapshot reports the layout as it is: entries it does not recognise are
listed in the printed summary, never an error. It goes stale as soon as a
prediction, translation or retarget touches the version; regenerate it then.
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

QS = (0, 1, 2)
N_RHS = 101
RH_Q = re.compile(r"^RH(\d+)_Q(\d)\.(?:cog\.)?tif$")
TILE_DIR = re.compile(r"^(\d{2}[A-Z]{3})(?:_cog)?$")
TILE_FILE = re.compile(r"^(\d{2}[A-Z]{3})(?:[_.].*)?$")
MOSAIC_NAME = re.compile(
    r"^(?P<kind>[A-Za-z0-9_]+?)(?:_\d{4})?_(?P<variable>RH\d+|CR)"
    r"(?:_(?P<quantile>Q[A-Za-z0-9-]+))?(?:\.(?P<format>.+))?$")
SIDECAR = ".aux.xml"
MULTIBAND_DIR = "multibands"
OTHER = "other"


def _version_columns(formats: Sequence[str]) -> List[str]:
    return (["version", "tiles"]
            + [c for fmt in formats for c in [f"{fmt}_tiles"] + [f"{fmt}_q{q}" for q in QS]]
            + ["vrt_tiles", "multiband_tiles", "tile_bytes", "mosaic_files",
               "mosaic_variables", "mosaic_quantiles", "mosaic_formats",
               "mosaic_bytes", "generated"])


def _columns(formats: Sequence[str]) -> List[str]:
    return (["tile_id"] + [f"{fmt}_q{q}" for fmt in formats for q in QS]
            + ["multiband", "vrt", "other_files", "size_bytes"])


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scan_tile_dir(tile_dir: str) -> Tuple[Counter, int, int]:
    """(rasters per Q, other files, bytes) of one tiles/<fmt>/<TILE>/."""
    per_q, rhs, other, n_bytes = Counter(), set(), 0, 0
    with os.scandir(tile_dir) as it:
        for e in it:
            if not e.is_file() or e.name.endswith(SIDECAR):
                continue
            n_bytes += e.stat().st_size
            m = RH_Q.match(e.name)
            if (m and int(m[1]) < N_RHS and int(m[2]) in QS
                    and (int(m[1]), int(m[2])) not in rhs):
                rhs.add((int(m[1]), int(m[2])))
                per_q[int(m[2])] += 1
            else:
                other += 1
    return per_q, other, n_bytes


def _tile_files(d: Path, suffix: str, unknown: List[str]) -> List[Tuple[str, int]]:
    """(tile_id, bytes) of the per-tile files <TILE>*<suffix> directly in d."""
    out = []
    with os.scandir(d) as it:
        for e in it:
            m = TILE_FILE.match(e.name)
            if e.is_file() and e.name.endswith(suffix) and m:
                out.append((m[1], e.stat().st_size))
            elif not e.name.endswith(SIDECAR):
                unknown.append(str(d / e.name))
    return out


def _snapshot_tiles(version_dir: Path, formats: Sequence[str], workers: int,
                    unknown: List[str]) -> List[List[str]]:
    rows: Dict[str, Counter] = {}

    def row(tile: str) -> Counter:
        return rows.setdefault(tile, Counter())

    tiles_dir = version_dir / "tiles"
    jobs = []                               # (fmt, tile_id, tile dir)
    vrt_dirs = [d for d in version_dir.glob("vrt*") if d.is_dir()]
    if tiles_dir.is_dir():
        for sub in sorted(tiles_dir.iterdir()):
            if sub.is_dir() and sub.name in formats:
                for d in os.scandir(sub):
                    m = TILE_DIR.match(d.name)
                    if d.is_dir() and m:
                        jobs.append((sub.name, m[1], d.path))
                    else:
                        unknown.append(d.path)
            elif sub.is_dir() and sub.name.startswith("vrt"):
                vrt_dirs.append(sub)
            elif sub.is_dir() and sub.name == MULTIBAND_DIR:
                for tile, n in _tile_files(sub, ".tif", unknown):
                    row(tile)["multiband"] += 1
                    row(tile)["size_bytes"] += n
            else:
                unknown.append(str(sub))

    with ThreadPoolExecutor(workers) as pool:
        scans = pool.map(lambda j: _scan_tile_dir(j[2]), jobs)
        for (fmt, tile, _), (per_q, other, n_bytes) in zip(jobs, scans):
            r = row(tile)
            for q, n in per_q.items():
                r[f"{fmt}_q{q}"] += n
            r[f"{fmt}_any"] += 1
            r["other_files"] += other
            r["size_bytes"] += n_bytes
    for d in vrt_dirs:
        for tile, n in _tile_files(d, ".vrt", unknown):
            row(tile)["vrt"] += 1
            row(tile)["size_bytes"] += n

    cols = _columns(formats)[1:]
    out = [[tile] + [str(rows[tile][c]) for c in cols] for tile in sorted(rows)]
    has = {fmt: sum(rows[t][f"{fmt}_any"] > 0 for t in rows) for fmt in formats}
    return out, has


def _mosaic_entry(rel: str, name: str, n_files: int, n_bytes: int, mtime: float,
                  is_dir: bool) -> Dict[str, Any]:
    """One mosaic file or dir: its row (variable), column, and counts."""
    prefix = rel.rstrip("/")[:-len(name)]          # "cog/" for mosaic/cog/<name>, else ""
    m = MOSAIC_NAME.match(name)
    if m:
        variable, quantile, fmt = m["variable"], m["quantile"] or "", m["format"] or ""
        column = m["kind"] + (f"_{quantile}" if quantile else "") + ("/" if is_dir else f".{fmt}" if fmt else "")
    else:
        variable, quantile = OTHER, ""
        fmt = name.split(".", 1)[1] if "." in name else ""
        column = name + ("/" if is_dir else "")
    return {"variable": variable, "quantile": quantile, "format": "dir" if is_dir else fmt,
            "column": prefix + column, "n_files": n_files, "bytes": n_bytes, "mtime": mtime}


def _rh_order(v: str):
    m = re.fullmatch(r"RH(\d+)", v)
    return (0, int(m[1]), "") if m else (2, 0, "") if v == OTHER else (1, 0, v)


def _mosaic_by_variable(entries: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    """Pivot the entries: one row per variable, one column per kind/quantile/format."""
    cols = sorted({e["column"] for e in entries}, key=lambda c: (c.endswith("/"), c))
    cells: Dict[str, Counter] = {}
    for e in entries:
        c = cells.setdefault(e["variable"], Counter())
        c[e["column"]] += e["n_files"]
        c["n_files"] += e["n_files"]
        c["bytes"] += e["bytes"]
        c["mtime"] = max(c["mtime"], e["mtime"])
    rows = [[v] + [str(c[col]) for col in cols]
            + [str(c["n_files"]), str(c["bytes"]), _utc(c["mtime"])]
            for v, c in sorted(cells.items(), key=lambda kv: _rh_order(kv[0]))]
    return ["variable"] + cols + ["n_files", "size_bytes", "modified"], rows


def _snapshot_mosaic(mosaic_dir: Path, formats: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []

    def file_row(e: os.DirEntry, rel: str):
        st = e.stat()
        rows.append(_mosaic_entry(rel, e.name, 1, st.st_size, st.st_mtime, False))

    def dir_row(d: os.DirEntry, rel: str):
        n = n_bytes = 0
        latest = 0.0
        for root, _, files in os.walk(d.path):
            for f in files:
                st = os.stat(os.path.join(root, f))
                n, n_bytes = n + 1, n_bytes + st.st_size
                latest = max(latest, st.st_mtime)
        rows.append(_mosaic_entry(rel + "/", d.name, n, n_bytes,
                                  latest or d.stat().st_mtime, True))

    with os.scandir(mosaic_dir) as it:
        for e in sorted(it, key=lambda e: e.name):
            if e.is_file():
                file_row(e, e.name)
            elif e.is_dir() and e.name in formats:
                with os.scandir(e.path) as sub:
                    for f in sorted(sub, key=lambda f: f.name):
                        rel = f"{e.name}/{f.name}"
                        file_row(f, rel) if f.is_file() else dir_row(f, rel)
            elif e.is_dir():
                dir_row(e, e.name)
    return rows


def _distinct(values) -> str:
    """Sorted distinct values, runs of RH<n> collapsed: RH0-2,RH4-98,CR."""
    values = {v for v in values if v}
    rhs = sorted(int(v[2:]) for v in values if re.fullmatch(r"RH\d+", v))
    runs = []
    for n in rhs:
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    out = [f"RH{a}" if a == b else f"RH{a}-{b}" for a, b in runs]
    return ",".join(out + sorted(v for v in values if not re.fullmatch(r"RH\d+", v)))


def _write_tsv(fp: Path, columns: List[str], rows: List[List[str]]):
    tmp = fp.with_suffix(".tsv.tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        fh.writelines("\t".join(r) + "\n" for r in rows)
    os.replace(tmp, fp)


def _read_tsv(fp: Path) -> List[Dict[str, str]]:
    if not fp.exists():
        return []
    with open(fp, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_vsm_snapshot(
    year: int = None,
    products_root: str = None,
    versions: Sequence[str] = None,
    formats: Sequence[str] = ("cog", "geotiff"),
    out_name: str = "snapshot.tsv",
    mosaic_out_name: str = "snapshot_mosaic.tsv",
    versions_out_name: str = "snapshot_versions.tsv",
    workers: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Write the snapshot tables of every version of a year, plus the year overview.

    Args:
        year: product year, <products_root>/<year>/.
        products_root: e.g. ~/data/gvs/products/vsm.
        versions: version dir names to snapshot; null -> every subdir of the
            year with a tiles/ or mosaic/.
        formats: tiles/<fmt>/ dirs holding per-tile RH*_Q* rasters; also the
            mosaic/<fmt>/ subdirs listed file by file. Fixes the columns.
        out_name, mosaic_out_name: file names written into each version dir.
        versions_out_name: the per-version overview in the year dir.
        workers: threads for the tile dir scans (NFS latency-bound).

    Returns:
        The overview rows, plus the entries that were not recognised per version.
    """
    year_dir = Path(products_root).expanduser() / str(year)
    if not year_dir.is_dir():
        raise RuntimeError(f"no such year dir: {year_dir}")
    formats = list(formats)
    subdirs = sorted(d for d in year_dir.iterdir() if d.is_dir())
    all_versions = [d for d in subdirs if (d / "tiles").is_dir() or (d / "mosaic").is_dir()]
    if versions:
        version_dirs = [year_dir / v for v in versions]
        missing = [str(d) for d in version_dirs if not d.is_dir()]
        if missing:
            raise RuntimeError(f"version dirs not found: {missing}")
    else:
        version_dirs = all_versions
        skipped = [d.name for d in subdirs if d not in all_versions]
        if skipped:
            print(f"not a version (no tiles/ or mosaic/), skipped: {skipped}")
    if not version_dirs:
        raise RuntimeError(f"no version dirs in {year_dir}")

    columns = _columns(formats)
    overview, unrecognised = {}, {}
    for vdir in version_dirs:
        name = f"{year}/{vdir.name}"
        generated = _utc(datetime.now().timestamp())
        unknown: List[str] = []
        rows, has = _snapshot_tiles(vdir, formats, workers, unknown)
        _write_tsv(vdir / out_name, columns, rows)

        idx = {c: i for i, c in enumerate(columns)}
        complete = {f"{fmt}_q{q}": sum(int(r[idx[f"{fmt}_q{q}"]]) == N_RHS for r in rows)
                    for fmt in formats for q in QS}
        tile_bytes = sum(int(r[idx["size_bytes"]]) for r in rows)
        entries = []
        if (vdir / "mosaic").is_dir():
            entries = _snapshot_mosaic(vdir / "mosaic", formats)
            _write_tsv(vdir / mosaic_out_name, *_mosaic_by_variable(entries))
        mosaic_files = [e for e in entries if e["format"] != "dir"]

        ov = {"version": vdir.name, "tiles": len(rows),
              "vrt_tiles": sum(int(r[idx["vrt"]]) > 0 for r in rows),
              "multiband_tiles": sum(int(r[idx["multiband"]]) > 0 for r in rows),
              "tile_bytes": tile_bytes, "mosaic_files": len(mosaic_files),
              "mosaic_variables": _distinct(e["variable"] for e in mosaic_files
                                            if e["variable"] != OTHER),
              "mosaic_quantiles": _distinct(e["quantile"] for e in mosaic_files),
              "mosaic_formats": _distinct(e["format"] for e in mosaic_files),
              "mosaic_bytes": sum(e["bytes"] for e in entries), "generated": generated}
        ov.update(complete)
        for fmt in formats:
            ov[f"{fmt}_tiles"] = has[fmt]
        overview[vdir.name] = ov

        print(f"\n== {name}: {len(rows)} tiles, {tile_bytes / 1e12:.2f} TB -> {vdir / out_name}")
        for fmt in formats:
            if has[fmt]:
                full = ", ".join(f"Q{q} {complete[f'{fmt}_q{q}']}" for q in QS)
                print(f"  {fmt:<8s} {has[fmt]:7d} tiles; all 101 RH: {full}")
        if entries:
            print(f"  mosaic   {len(mosaic_files)} files, variables {ov['mosaic_variables']}, "
                  f"formats {ov['mosaic_formats']} -> {vdir / mosaic_out_name}")
        if unknown:
            print(f"  NOTE: {len(unknown)} entries not recognised, e.g. {unknown[:5]}")
        unrecognised[name] = unknown

    # Rows of versions not scanned this time stay, if the version still exists.
    vcols = _version_columns(formats)
    existing = {d.name for d in all_versions}
    merged = {r["version"]: r for r in _read_tsv(year_dir / versions_out_name)
              if r.get("version") in existing}
    merged.update(overview)
    _write_tsv(year_dir / versions_out_name, vcols,
               [[str(merged[v].get(c, "")) for c in vcols] for v in sorted(merged)])
    print(f"\noverview of {len(merged)} versions -> {year_dir / versions_out_name}")

    return {"versions": overview, "unrecognised": unrecognised}
