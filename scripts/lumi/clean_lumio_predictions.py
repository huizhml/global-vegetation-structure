#!/usr/bin/env python3
"""
Clean redundant raw predictions on LUMI-O  (scaled for ~18k tiles).

For every bucket named {zone}-{year} (e.g. 60g-2024) on the private LUMI-O
remote, this verifies the *corrected* per-tile output and, ONLY when that output
is complete and valid, deletes the now-redundant raw predictions:

    lumi-465002698-private:60g-2024/60GTA                         <- VERIFY
    lumi-465002698-private:60g-2024/predictions_GTiff_2024/60GTA  <- DELETE

A tile passes (its predictions become deletable) when:
  1. {bucket}/{tile} holds exactly --expected (303) .tif files,
  2. none of those .tif files is zero-byte, and
  3. every .tif opens cleanly with GDAL over /vsis3 (header read only).

  NOTE: the header open does not decompress pixels, so a truncated-yet-openable
  COG is NOT caught. Upload to LUMI-O is an atomic S3 PUT, so a short object is
  unlikely unless the COG was already truncated at translation time.

Scaling notes for ~18k tiles (~5.5M COGs):
  * one recursive `rclone lsf` per bucket gives every file's size/path in a
    single listing -- the count + zero-byte checks need no per-tile calls;
  * the header check runs in an in-process GDAL/rasterio thread pool that reuses
    HTTP keep-alive connections (no per-file process spawn), so the ~5.5M opens
    are cheap range reads -- roughly 1-2h at --threads 64-96;
  * purges run with tile-level parallelism.

SAFETY: dry-run by default. Nothing is deleted unless you pass --apply.
        A tif that errors transiently fails the check -> the tile is KEPT, never
        wrongly deleted (failures err on the safe side).

Usage:
    python clean_lumio_predictions.py                     # dry-run, all buckets
    python clean_lumio_predictions.py --apply             # actually delete
    python clean_lumio_predictions.py --year 2024         # only *-2024 buckets
    python clean_lumio_predictions.py --apply 60g-2024    # one bucket only
    python clean_lumio_predictions.py --no-deep           # count + zero-byte only
    python clean_lumio_predictions.py --threads 96        # more open workers

Run on a compute node with rclone + a GDAL/rasterio python env, e.g.:
    srun -A project_465002698 -p small -c 32 --mem=16G -t 03:00:00 \
         python scripts/lumi/clean_lumio_predictions.py --apply --threads 96
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BUCKET_RE = re.compile(r"^[a-z0-9]+-[0-9]{4}$")  # {zone}-{year}
GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
#  rclone helpers
# --------------------------------------------------------------------------- #
def rclone(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["rclone", *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def list_buckets(remote: str) -> list[str]:
    out = rclone("lsf", remote, "--dirs-only").stdout
    return [line.rstrip("/") for line in out.splitlines() if line.strip()]


def list_bucket_files(remote_bucket: str) -> list[tuple[str, int]]:
    """One recursive listing -> [(path_relative_to_bucket, size_bytes), ...]."""
    out = rclone(
        "lsf", remote_bucket, "-R", "--files-only", "--fast-list",
        "--format", "ps", "--separator", "\t",
    ).stdout
    rows: list[tuple[str, int]] = []
    for line in out.splitlines():
        if not line:
            continue
        path, _, size = line.rpartition("\t")
        if not path:
            continue
        try:
            rows.append((path, int(size)))
        except ValueError:
            continue
    return rows


def purge(remote_path: str) -> bool:
    return rclone("purge", remote_path, check=False).returncode == 0


# --------------------------------------------------------------------------- #
#  GDAL /vsis3 credentials + validator
# --------------------------------------------------------------------------- #
def vsis3_options(remote_name: str) -> dict[str, str]:
    """Read S3 creds/endpoint from the rclone remote and shape them for GDAL."""
    out = rclone("config", "show", remote_name).stdout
    cfg = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    ak, sk, ep = cfg.get("access_key_id"), cfg.get("secret_access_key"), cfg.get("endpoint")
    if not (ak and sk and ep):
        sys.exit(f"ERROR: missing access_key_id/secret_access_key/endpoint for {remote_name}")
    ep = re.sub(r"^https?://", "", ep).rstrip("/")
    return {
        "AWS_ACCESS_KEY_ID": ak,
        "AWS_SECRET_ACCESS_KEY": sk,
        "AWS_S3_ENDPOINT": ep,
        "AWS_HTTPS": "YES",
        "AWS_VIRTUAL_HOSTING": "FALSE",          # Ceph radosgw -> path-style
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "AWS_DEFAULT_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MAX_RETRY": "3",              # ride out transient blips
        "GDAL_HTTP_RETRY_DELAY": "1",
        "VSI_CACHE": "TRUE",
    }


def make_validator(opts: dict[str, str]):
    """Return (initializer, validate_fn, backend_name) for the thread pool.

    validate() opens each COG's header over /vsis3 (a few KB of HTTP range reads,
    no pixel decode) and confirms it is a readable GeoTIFF. Any error -> False ->
    the tile is KEPT (failures err on the safe side). NOTE: a header open does
    not read the pixels, so a truncated-yet-openable COG is not caught.
    """
    try:
        from osgeo import gdal  # noqa: F401

        def init():
            from osgeo import gdal
            gdal.DontUseExceptions()
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            for k, v in opts.items():
                gdal.SetConfigOption(k, v)

        def validate(vsipath: str) -> bool:
            from osgeo import gdal
            try:
                ds = gdal.Open(vsipath)
                ok = ds is not None
                ds = None
                return ok
            except Exception:
                return False

        return init, validate, "osgeo.gdal"
    except ImportError:
        pass

    try:
        import rasterio  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: the header check needs GDAL python bindings or rasterio.\n"
            "       Load a GDAL/rasterio env, or rerun with --no-deep."
        )

    def init():
        import rasterio
        rasterio.Env(**opts).__enter__()  # one Env per worker thread, kept open

    def validate(vsipath: str) -> bool:
        import rasterio
        try:
            with rasterio.open(vsipath) as src:
                _ = src.count
            return True
        except Exception:
            return False

    return init, validate, "rasterio"


# --------------------------------------------------------------------------- #
#  per-bucket processing
# --------------------------------------------------------------------------- #
def index_bucket(rows, year):
    """Split a bucket's file listing into verify-output and predictions maps."""
    preds_prefix = f"predictions_GTiff_{year}/"
    verify = defaultdict(list)   # tile -> [(path, size), ...] for *.tif
    preds = defaultdict(lambda: [0, 0])  # tile -> [total_bytes, file_count]
    for path, size in rows:
        if path.startswith(preds_prefix):
            rest = path[len(preds_prefix):]
            tile = rest.split("/", 1)[0]
            if tile:
                preds[tile][0] += size
                preds[tile][1] += 1
        elif "/" in path and path.endswith(".tif"):
            tile = path.split("/", 1)[0]
            verify[tile].append((path, size))
    return verify, preds, preds_prefix


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean redundant LUMI-O predictions.")
    ap.add_argument("buckets", nargs="*", help="restrict to these bucket name(s)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--project", default=os.environ.get("LUMI_PROJECT", "465002698"))
    ap.add_argument("--year", default=os.environ.get("YEAR", ""), help="restrict to *-YEAR buckets")
    ap.add_argument("--expected", type=int, default=int(os.environ.get("EXPECTED_TIF", "303")))
    ap.add_argument("--no-deep", action="store_true",
                    help="skip the header check (count + zero-byte only)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("DEEP_JOBS", "64")),
                    help="parallel header-check workers")
    ap.add_argument("--purge-jobs", type=int, default=int(os.environ.get("PURGE_JOBS", "16")),
                    help="parallel tile purges")
    ap.add_argument("--csv", default=f"clean_lumio_{datetime.now():%Y%m%d_%H%M%S}.csv")
    args = ap.parse_args()

    remote_name = f"lumi-{args.project}-private"
    remote = f"{remote_name}:"
    deep = not args.no_deep

    # validation pool (one shared pool for the whole run -> max connection reuse)
    pool = backend = None
    if deep:
        init, validate, backend = make_validator(vsis3_options(remote_name))
        pool = ThreadPoolExecutor(max_workers=args.threads, initializer=init)

    # discover buckets
    buckets = []
    for b in list_buckets(remote):
        if not BUCKET_RE.match(b):
            continue
        if args.year and not b.endswith(f"-{args.year}"):
            continue
        if args.buckets and b not in args.buckets:
            continue
        buckets.append(b)
    buckets.sort()

    print("=" * 66)
    print(" LUMI-O prediction cleanup (scaled)")
    print(f"   remote    : {remote}")
    print(f"   mode      : {'APPLY (will DELETE)' if args.apply else 'DRY-RUN (no deletions)'}")
    print(f"   buckets   : {len(buckets)} matched" + (f" (year={args.year})" if args.year else ""))
    rule = f"{args.expected} tifs, none empty" + (f", header OK via {backend}" if deep else "")
    print(f"   pass rule : {rule}")
    print(f"   results   : {args.csv}")
    print("=" * 66)
    if not buckets:
        print("No matching buckets. Nothing to do.")
        return

    n_pass = n_keep = n_del = n_delfail = 0
    n_dir_removed = n_dir_would = 0
    freed_bytes = 0

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bucket", "tile", "tif_count", "deep_failures", "decision", "freed_gib"])

        for bi, bucket in enumerate(buckets, 1):
            year = bucket.rsplit("-", 1)[-1]
            verify, preds, preds_prefix = index_bucket(list_bucket_files(f"{remote}{bucket}/"), year)
            print(f"\n>>> [{bi}/{len(buckets)}] {bucket}: {len(preds)} tile(s) with predictions")

            # ---- metadata gate (count + zero-byte), then collect deep work ----
            metadata_pass = []
            rows_out = []  # (tile, count, deepfails, decision, freed_gib)
            for tile in sorted(preds):
                vfiles = verify.get(tile, [])
                count = len(vfiles)
                if count == 0:
                    rows_out.append((tile, 0, "-", "KEEP:no-output", 0.0)); continue
                if count != args.expected:
                    rows_out.append((tile, count, "-", f"KEEP:count={count}", 0.0)); continue
                if any(sz == 0 for _, sz in vfiles):
                    n_empty = sum(1 for _, sz in vfiles if sz == 0)
                    rows_out.append((tile, count, "-", f"KEEP:{n_empty}-empty", 0.0)); continue
                metadata_pass.append((tile, vfiles))

            # ---- header check (shared thread pool) ----
            deep_fails = {}  # tile -> count of tifs that failed to open
            if deep and metadata_pass:
                jobs = {}  # future -> tile
                for tile, vfiles in metadata_pass:
                    for path, _ in vfiles:
                        jobs[pool.submit(validate, f"/vsis3/{bucket}/{path}")] = tile
                for fut, tile in jobs.items():
                    if not fut.result():
                        deep_fails[tile] = deep_fails.get(tile, 0) + 1

            # ---- decide which tiles' predictions to purge ----
            purge_targets = []  # (tile, remote_path)
            for tile, vfiles in metadata_pass:
                nfail = deep_fails.get(tile, 0)
                if nfail:
                    rows_out.append((tile, len(vfiles), nfail, f"KEEP:{nfail}-open-fail", 0.0))
                else:
                    gib = preds[tile][0] / GIB
                    purge_targets.append((tile, f"{remote}{bucket}/{preds_prefix}{tile}"))
                    rows_out.append((tile, len(vfiles), 0 if deep else "-",
                                     "WOULD_DELETE" if not args.apply else "PASS", gib))

            # ---- act (parallel purges) ----
            if args.apply and purge_targets:
                with ThreadPoolExecutor(max_workers=args.purge_jobs) as pp:
                    done = dict(pp.map(lambda t: (t[0], purge(t[1])), purge_targets))
                rows_out = [
                    (r[0], r[1], r[2], ("DELETED" if done[r[0]] else "DELETE_FAILED"), r[4])
                    if r[3] == "PASS" else r
                    for r in rows_out
                ]

            # ---- emit + tally ----
            for tile, count, dfail, decision, gib in rows_out:
                w.writerow([bucket, tile, count, dfail, decision, f"{gib:.2f}"])
                if decision.startswith("KEEP"):
                    n_keep += 1
                else:                                  # passed all checks
                    n_pass += 1
                    if decision in ("WOULD_DELETE", "DELETED"):
                        freed_bytes += preds[tile][0]
                    if decision == "DELETED":
                        n_del += 1
                    elif decision == "DELETE_FAILED":
                        n_delfail += 1
            fh.flush()
            ok_n = sum(1 for r in rows_out if not r[3].startswith("KEEP"))
            print(f"    pass={ok_n}  keep={len(rows_out) - ok_n}  "
                  f"frees={sum(r[4] for r in rows_out):.1f} GiB")

            # ---- remove the predictions_GTiff_{year} folder once it's empty ----
            # Empty after this run = no prediction tile survives (none kept / failed).
            survivors = sum(1 for r in rows_out
                            if r[3].startswith("KEEP") or r[3] == "DELETE_FAILED")
            if survivors == 0:
                preds_dir = f"{remote}{bucket}/predictions_GTiff_{year}"
                if preds and args.apply:                 # we just emptied a real folder
                    rclone("rmdir", preds_dir, check=False)   # drop any dir marker, best effort
                    n_dir_removed += 1
                    w.writerow([bucket, f"predictions_GTiff_{year}", "-", "-", "REMOVED_DIR", "0.00"])
                    print(f"    REMOVED_DIR (now empty): {preds_dir}")
                elif preds:                              # dry-run preview
                    n_dir_would += 1
                    w.writerow([bucket, f"predictions_GTiff_{year}", "-", "-", "WOULD_REMOVE_DIR", "0.00"])
                    print(f"    WOULD_REMOVE_DIR (would be empty): {preds_dir}")
                elif args.apply:                         # no files now; clean a leftover marker
                    if rclone("rmdir", preds_dir, check=False).returncode == 0:
                        n_dir_removed += 1
                        w.writerow([bucket, f"predictions_GTiff_{year}", "-", "-", "REMOVED_DIR", "0.00"])
                        print(f"    REMOVED_DIR (leftover empty): {preds_dir}")

    if pool:
        pool.shutdown(wait=True)

    print("\n" + "=" * 66)
    print(" SUMMARY")
    print(f"   passed check : {n_pass}")
    print(f"   kept (skip)  : {n_keep}")
    if args.apply:
        print(f"   deleted      : {n_del}   (failed: {n_delfail})")
        print(f"   empty dirs   : {n_dir_removed} predictions_GTiff folder(s) removed")
        print(f"   freed        : {freed_bytes / GIB:.1f} GiB")
    else:
        print(f"   would delete : {n_pass}  ->  would free {freed_bytes / GIB:.1f} GiB")
        print(f"   empty dirs   : {n_dir_would} predictions_GTiff folder(s) would be removed")
        print("   (dry-run: re-run with --apply to actually delete)")
    print(f"   CSV          : {args.csv}")
    print("=" * 66)


if __name__ == "__main__":
    main()
