#!/usr/bin/env python3
"""核对 Source Cooperative 上每个 tile 的 RH × Q 文件是否齐全。

只读、匿名(公开桶不需要 source-coop login),每个 tile 一次 list 请求
(303 个对象 < 单页 1000),18k 个 tile 几分钟扫完。tile 清单来自 release
manifest(docs/data/vsm_release_manifest.md)。

    python -m deploy.run run=sc_audit run.year=2020 \
        run.manifest=~/data/gvs/products/vsm/2020/released_v0.tsv \
        run.out_file=~/data/gvs/products/vsm/2020/source_coop_state/audit.tsv

输出 out_file,一个 tile 一行:
    tile_id  q0  q1  q2  zero_size  extra
- q<q>: gvsm/<year>/<tile>/ 下 RH{0..100}_Q<q>.tif 存在的个数,齐全是 101。
- zero_size: 0 字节的对象数(算在 q<q> 里,但需要重传)。
- extra: 不属于 RH*_Q*.tif 命名的对象数。
另写 <out_file 去掉 .tsv>_missing.txt:只缺一部分 RH 的 (tile, Q) 的缺失 key,
整个 Q 都没有的不展开(看 tsv 里的 0 就够了,展开会有几百万行)。
"""

from __future__ import annotations

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config

NAME_RE = re.compile(r"^RH(\d+)_Q(\d)\.tif$")


def _client(endpoint: str, workers: int):
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, max_pool_connections=workers,
                      retries={"max_attempts": 10, "mode": "adaptive"}))


def _list_keys(s3, bucket: str, prefix: str) -> List[Tuple[str, int]]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return out


def _list_tile_dirs(s3, bucket: str, prefix: str) -> set:
    tiles = set()
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix, Delimiter="/"):
        tiles |= {p["Prefix"][len(prefix):].rstrip("/") for p in page.get("CommonPrefixes", [])}
    return tiles


def _audit_tile(s3, bucket: str, prefix: str, tile: str, quantiles, n_rh: int) -> Tuple[Dict, List[str]]:
    have: Dict[int, set] = {q: set() for q in quantiles}
    zero = extra = 0
    for key, size in _list_keys(s3, bucket, f"{prefix}{tile}/"):
        m = NAME_RE.match(key.rsplit("/", 1)[-1])
        if not m or int(m.group(2)) not in have or int(m.group(1)) >= n_rh:
            extra += 1
            continue
        have[int(m.group(2))].add(int(m.group(1)))
        zero += size == 0
    row = {"tile_id": tile, **{f"q{q}": len(have[q]) for q in quantiles},
           "zero_size": zero, "extra": extra}
    missing = [f"{prefix}{tile}/RH{rh}_Q{q}.tif"
               for q in quantiles if 0 < len(have[q]) < n_rh
               for rh in range(n_rh) if rh not in have[q]]
    return row, missing


def audit_source_coop(
    year: int = None,
    manifest: str = None,
    out_file: str = None,
    bucket: str = "geoai-ucph",
    prefix: str = "gvsm/{year}/",
    endpoint: str = "https://data.source.coop",
    quantiles: List[int] = (0, 1, 2),
    n_rh: int = 101,
    tiles: Optional[List[str]] = None,
    workers: int = 32,
    **kwargs,
) -> Dict[str, int]:
    """Source Coop 上 manifest 里每个 tile 的 RH*_Q*.tif 计数 -> out_file。

    Args:
        year: 年份,填进 prefix。
        manifest: release manifest(released_v0.tsv),取 tile_id 列。
        out_file: 每个 tile 一行的结果 tsv。
        bucket / prefix / endpoint: 公开桶的位置,prefix 下一层是 <tile>/。
        quantiles: 要核对的分位数。
        n_rh: 每个 Q 的 RH 个数(RH0..RH100)。
        tiles: 只查这些 tile(调试用),None = manifest 全部。
        workers: 并发 list 请求数。

    Returns:
        各项计数(完整 tile 数、每个 Q 缺的 tile 数、manifest 外的 tile 数)。
    """
    prefix = prefix.format(year=year)
    quantiles = [int(q) for q in quantiles]
    with open(Path(manifest).expanduser()) as fh:
        expected = [r["tile_id"] for r in csv.DictReader(fh, delimiter="\t")]
    in_manifest = set(expected)
    if tiles:
        expected = [t for t in expected if t in set(tiles)]
    s3 = _client(endpoint, workers)

    on_remote = _list_tile_dirs(s3, bucket, prefix)
    unexpected = sorted(on_remote - in_manifest)
    print(f"[info] manifest {len(expected)} 个 tile;{prefix} 下有 {len(on_remote)} 个 tile 目录,"
          f"其中 {len(unexpected)} 个不在 manifest 里" + (f": {unexpected[:10]}" if unexpected else ""))

    rows, missing = [], []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_audit_tile, s3, bucket, prefix, t, quantiles, n_rh): t for t in expected}
        for i, fut in enumerate(as_completed(futs), 1):
            row, miss = fut.result()
            rows.append(row)
            missing += miss
            if i % 2000 == 0:
                print(f"[prog] {i}/{len(expected)}  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["tile_id"])

    out = Path(out_file).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tile_id"] + [f"q{q}" for q in quantiles] + ["zero_size", "extra"]
    with open(out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    miss_path = out.with_name(out.stem + "_missing.txt")
    miss_path.write_text("".join(k + "\n" for k in sorted(missing)))

    stats = {"tiles": len(rows),
             "complete": sum(all(r[f"q{q}"] == n_rh for q in quantiles) and not r["zero_size"]
                             for r in rows),
             "zero_size_files": sum(r["zero_size"] for r in rows),
             "extra_files": sum(r["extra"] for r in rows),
             "not_in_manifest": len(unexpected)}
    for q in quantiles:
        stats[f"q{q}_full"] = sum(r[f"q{q}"] == n_rh for r in rows)
        stats[f"q{q}_partial"] = sum(0 < r[f"q{q}"] < n_rh for r in rows)
        stats[f"q{q}_none"] = sum(r[f"q{q}"] == 0 for r in rows)
    print(f"[done] {time.time() - t0:.0f}s -> {out}  (部分缺失的 key -> {miss_path}, {len(missing)} 行)")
    for k, v in stats.items():
        print(f"  {k:16s} {v}")
    return stats
