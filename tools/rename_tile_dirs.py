"""Rename <TILE><dir_suffix>/RH*_Q*<old_ext> tile dirs to <TILE>/RH*_Q*<new_ext>.

2024 masked was written as masked/tiles/cog/<TILE>_cog/RH*_Q*.cog.tif, unlike
every other product (tiles/<fmt>/<TILE>/RH*_Q*.tif). Everything that reads a
product by year + version builds the second form, so this brings 2024 in line:

    <tiles_dir>/01GEL_cog/RH0_Q1.cog.tif          -> <tiles_dir>/01GEL/RH0_Q1.tif
    <tiles_dir>/01GEL_cog/RH0_Q1.cog.tif.aux.xml  -> <tiles_dir>/01GEL/RH0_Q1.tif.aux.xml

Per tile, every target name is checked before anything moves; a tile with a
clash (<TILE>/ already there, or both RH0_Q1.cog.tif and RH0_Q1.tif) is left
untouched and marked failed. Files are renamed first, the dir last, so an
interrupted run is resumed by running again: already renamed files no longer
match old_ext, and a tile whose dir is renamed counts as done. Files that are
not RH rasters or their sidecars are moved along with the dir, not renamed,
and listed. Renames only, no data is copied; dry_run (the default) only reports.

Progress goes to a TaskState (tools/task_state.py) in state_dir.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.task_state import TaskState

MGRS_TILE = re.compile(r"\d{2}[A-Z]{3}")
RH_Q = re.compile(r"^RH\d+_Q\d")


def _plan_tile(src: Path, dst: Path, old_ext: str, new_ext: str
               ) -> Tuple[List[Tuple[str, str]], List[str], str]:
    """(file renames, other files, error) for one tile dir; error "" if clean."""
    if dst.exists():
        return [], [], f"{dst.name}/ exists next to {src.name}/"
    names = set(os.listdir(src))
    renames, other = [], []
    for name in sorted(names):
        stem, sep, tail = name.partition(old_ext)
        # RH0_Q1.cog.tif, or a sidecar of it: RH0_Q1.cog.tif.aux.xml
        if sep and RH_Q.match(stem) and (tail == "" or tail.startswith(".")):
            new = stem + new_ext + tail
            if new in names:
                return [], [], f"{src.name}/: both {name} and {new}"
            renames.append((name, new))
        elif not (RH_Q.match(name) and new_ext in name):
            other.append(name)
    return renames, other, ""


def _rename_tile(src: Path, dst: Path, renames: List[Tuple[str, str]]) -> int:
    n_before = len(os.listdir(src))
    for old, new in renames:
        os.rename(src / old, src / new)
    os.rename(src, dst)
    n_after = len(os.listdir(dst))
    if n_after != n_before:
        raise RuntimeError(f"{dst.name}: {n_before} files before, {n_after} after")
    return n_after


def rename_tile_dirs(
    tiles_dir: str = None,
    dir_suffix: str = "_cog",
    old_ext: str = ".cog.tif",
    new_ext: str = ".tif",
    state_dir: str = None,
    dry_run: bool = True,
    workers: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Rename every <TILE><dir_suffix>/ in tiles_dir and the RH files in it.

    Args:
        tiles_dir: e.g. ~/data/gvs/products/vsm/2024/masked/tiles/cog.
        dir_suffix: dir name = <TILE><dir_suffix>; the target is <TILE>.
        old_ext, new_ext: RH*_Q*<old_ext>[.sidecar] -> RH*_Q*<new_ext>[.sidecar].
        state_dir: TaskState dir for the per-tile progress (not written on a dry run).
        dry_run: only check and report.
        workers: threads (NFS latency-bound).

    Returns:
        Counts of renamed, already done and failed tiles, plus the failures.
    """
    tiles_dir = Path(tiles_dir).expanduser()
    entries = sorted(os.listdir(tiles_dir))
    srcs = [e for e in entries if e.endswith(dir_suffix)
            and MGRS_TILE.fullmatch(e[:-len(dir_suffix)]) and (tiles_dir / e).is_dir()]
    done_already = [e for e in entries if MGRS_TILE.fullmatch(e)
                    and e + dir_suffix not in entries and (tiles_dir / e).is_dir()]
    unknown = sorted(set(entries) - set(srcs) - set(done_already))
    print(f"{tiles_dir}: {len(srcs)} dirs to rename, {len(done_already)} already "
          f"renamed, {len(unknown)} other entries {unknown[:5]}")

    def plan(name: str):
        src = tiles_dir / name
        return (name, src, tiles_dir / name[:-len(dir_suffix)],
                *_plan_tile(src, tiles_dir / name[:-len(dir_suffix)], old_ext, new_ext))

    with ThreadPoolExecutor(workers) as pool:
        plans = list(pool.map(plan, srcs))
    failed = {name: err for name, _, _, _, _, err in plans if err}
    clean = [p for p in plans if not p[5]]
    with_other = {name: other for name, _, _, _, other, err in plans if other and not err}
    n_files = sum(len(p[3]) for p in clean)
    print(f"{len(clean)} tiles clean, {n_files} files to rename "
          f"({n_files / max(len(clean), 1):.1f} per tile), {len(failed)} with clashes")
    if with_other:
        print(f"{len(with_other)} tiles hold other files (moved, not renamed), e.g. "
              f"{dict(list(with_other.items())[:3])}")
    for name, err in list(failed.items())[:10]:
        print(f"  CLASH {err}")
    if clean:
        name, src, dst, renames, _, _ = clean[0]
        print(f"e.g. {src}/{renames[0][0]}\n  -> {dst}/{renames[0][1]}")

    result = {"to_rename": len(clean), "files": n_files, "already_done": len(done_already),
              "failed": failed, "other_files": with_other, "dry_run": dry_run}
    if dry_run:
        print("\ndry run: nothing renamed; rerun with run.dry_run=false")
        return result

    state = TaskState(state_dir)
    for name, err in failed.items():
        state.mark_failed(name[:-len(dir_suffix)], err)

    def run(p):
        name, src, dst, renames, _, _ = p
        try:
            return name, _rename_tile(src, dst, renames), ""
        except Exception as e:      # one bad tile must not stop the rest
            return name, 0, repr(e)

    n_ok = 0
    with ThreadPoolExecutor(workers) as pool:
        for name, n, err in pool.map(run, clean):
            tile = name[:-len(dir_suffix)]
            if err:
                failed[name] = err
                state.mark_failed(tile, err)
            else:
                n_ok += 1
                state.mark_done(tile, f"{n} files")
    print(f"\nrenamed {n_ok} tiles, {len(failed)} failed -> state {state_dir}")
    for name, err in list(failed.items())[:10]:
        print(f"  FAILED {name}: {err}")
    result.update(renamed=n_ok, failed=failed)
    return result
