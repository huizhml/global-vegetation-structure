#!/usr/bin/env python3
"""LUMI-O -> Source Cooperative 传输基准测试

依赖: pip install --user hydra-core boto3

用法(默认值见 config/deploy/config.yaml 的 sc_bench):
    # 1. 源端天花板(只读,不写目标端)
    python -m deploy.run run=sc_bench run.mode=read \
        run.src.bucket=32m-2024 run.src.prefix=32MND/

    # 2. 单进程扫 workers
    python -m deploy.run run=sc_bench run.mode=workers \
        run.src.bucket=32m-2024 run.src.prefix=32MND/ \
        run.dst.prefix=gvsm/_bench/ 'run.workers_grid=[16,32,64,128,256]'

    # 3. 固定 workers,扫进程数(模拟同节点多作业)
    python -m deploy.run run=sc_bench run.mode=procs \
        run.src.bucket=32m-2024 run.src.prefix=32MND/ \
        run.dst.prefix=gvsm/_bench/ run.workers=64 'run.procs_grid=[1,2,4,8]'

    # 4. 清理残留
    python -m deploy.run run=sc_bench run.mode=cleanup run.dst.prefix=gvsm/_bench/

安全限制: dst.prefix 必须包含 "_bench",防止写进真实数据路径。

判读:
    read 远快于 workers        -> 瓶颈在上传侧,继续加上传并发
    read 接近 workers          -> 源端才是瓶颈,加上传并发没用
    吞吐线性增长               -> 延迟受限,继续加
    吞吐平了 + 重试为 0        -> 撞到带宽或连接上限
    吞吐平了 + 重试上升        -> 撞到对方限流,退回上一档并写进邮件
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError
from omegaconf import DictConfig, OmegaConf

from deploy.source_coop.common import (
    build_cfg, build_dst, build_src, delete_prefix, list_source, retry_count,
)


# ================================================================ 统计
@dataclass
class Stats:
    label: str = ""
    objects: int = 0
    bytes_moved: int = 0
    wall_sec: float = 0.0
    errors: int = 0
    retries: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def mbps(self) -> float:
        return self.bytes_moved / self.wall_sec / 1e6 if self.wall_sec else 0.0

    @property
    def objps(self) -> float:
        return self.objects / self.wall_sec if self.wall_sec else 0.0

    def pct(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        i = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
        return s[i]

    def summary(self) -> Dict[str, Any]:
        return {"label": self.label, "objects": self.objects,
                "bytes": self.bytes_moved, "wall_sec": self.wall_sec,
                "mbps": self.mbps, "objps": self.objps,
                "p50": self.pct(0.5), "p95": self.pct(0.95),
                "retries": self.retries, "errors": self.errors}


# ================================================================ 工作进程
_G: Dict[str, Any] = {}       # boto3 client 不是 fork 安全的,必须在子进程里建


def _child_init(cfg_yaml: str, workers: int) -> None:
    cfg = OmegaConf.create(cfg_yaml)
    _G["cfg"] = cfg
    _G["src"] = build_src(cfg, workers)
    _G["dst"], _ = build_dst(cfg, workers)


def _move_one(args) -> Tuple[float, int, int, int]:
    """返回 (耗时, 字节数, 重试次数, 是否失败)。"""
    key, _size, dst_prefix, read_only = args
    cfg = _G["cfg"]
    t0 = time.time()
    try:
        got = _G["src"].get_object(Bucket=cfg.src.bucket, Key=key)
        retries = retry_count(got)
        body = got["Body"].read()
        if not read_only:
            base = key.rsplit("/", 1)[-1]
            put = _G["dst"].put_object(Bucket=cfg.dst.bucket,
                                       Key=dst_prefix + base, Body=body)
            retries += retry_count(put)
        return time.time() - t0, len(body), retries, 0
    except (ClientError, BotoCoreError, OSError):
        return time.time() - t0, 0, 0, 1


def _run_slice(payload) -> Dict[str, Any]:
    objs, workers, dst_prefix, read_only = payload
    tasks = [(k, s, dst_prefix, read_only) for k, s in objs]
    nbytes = nobj = nretry = nerr = 0
    lat: List[float] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dt, size, retries, err in pool.map(_move_one, tasks):
            lat.append(dt)
            nbytes += size
            nretry += retries
            nerr += err
            if not err:
                nobj += 1
    return {"objects": nobj, "bytes": nbytes, "errors": nerr,
            "retries": nretry, "latencies": lat}


def run_trial(cfg: DictConfig, objs: List[Tuple[str, int]], procs: int,
              workers: int, label: str, read_only: bool = False) -> Stats:
    """procs 个进程 x workers 个线程一起搬 objs。"""
    run_id = f"{int(time.time())}_{os.getpid()}"
    chunks: List[List[Tuple[str, int]]] = [[] for _ in range(procs)]
    for i, item in enumerate(objs):
        chunks[i % procs].append(item)

    payloads = [
        (chunk, workers, f"{cfg.dst.prefix}{run_id}/p{i}/", read_only)
        for i, chunk in enumerate(chunks) if chunk
    ]

    cfg_yaml = OmegaConf.to_yaml(cfg)
    st = Stats(label=label)
    t0 = time.time()
    if procs == 1:
        _child_init(cfg_yaml, workers)
        res = [_run_slice(payloads[0])]
    else:
        with ProcessPoolExecutor(max_workers=procs, initializer=_child_init,
                                 initargs=(cfg_yaml, workers)) as pool:
            res = list(pool.map(_run_slice, payloads))
    st.wall_sec = time.time() - t0

    for r in res:
        st.objects += r["objects"]
        st.bytes_moved += r["bytes"]
        st.errors += r["errors"]
        st.retries += r["retries"]
        st.latencies.extend(r["latencies"])

    if not read_only and not cfg.keep_objects:
        delete_prefix(cfg, f"{cfg.dst.prefix}{run_id}/")
    return st


# ================================================================ 报告
def report(rows: List[Stats]) -> None:
    print()
    print(f"{'配置':>18} {'MB/s':>9} {'obj/s':>8} {'p50':>7} {'p95':>7} "
          f"{'重试':>6} {'错误':>6} {'加速比':>8}")
    print("-" * 76)
    base = rows[0].mbps if rows else 1.0
    for st in rows:
        speedup = st.mbps / base if base else 0.0
        print(f"{st.label:>18} {st.mbps:9.0f} {st.objps:8.1f} "
              f"{st.pct(0.5):7.1f} {st.pct(0.95):7.1f} "
              f"{st.retries:6d} {st.errors:6d} {speedup:7.2f}x")
    print()

    if len(rows) >= 2:
        best = max(rows, key=lambda s: s.mbps)
        print(f"最快: {best.label}  ({best.mbps:.0f} MB/s)")
        for prev, cur in zip(rows, rows[1:]):
            gain = (cur.mbps / prev.mbps - 1) * 100 if prev.mbps else 0
            if gain < 15:
                print(f"拐点: {prev.label} -> {cur.label} 只涨了 {gain:.0f}%,"
                      "再加并发收益有限")
                break
        hot = [s for s in rows if s.objects and s.retries > s.objects * 0.02]
        if hot:
            print(f"注意: {', '.join(s.label for s in hot)} 的重试率超过 2%,"
                  "可能已触发对方限流")


# ================================================================ op 入口
def benchmark_transfer(
    src: Any = None,
    dst: Any = None,
    mode: str = "workers",              # read | workers | procs | cleanup
    workers: int = 32,                  # procs 模式下每进程的 workers
    workers_grid: Optional[List[int]] = None,
    procs_grid: Optional[List[int]] = None,
    trial_gb: float = 5.0,              # 每次试验搬多少数据
    settle_sec: float = 10.0,           # 试验间冷却,避免连接复用让后一轮虚高
    keep_objects: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """LUMI-O -> Source Cooperative 的并发调优基准。

    Args:
        src: 源端(LUMI-O)配置 —— bucket / prefix / rclone_remote / region。
        dst: 目标端(Source Cooperative)配置 —— bucket / prefix / endpoint /
            profile / region。prefix 必须包含 "_bench"。
        mode: read 只读源端(吞吐上限) | workers 扫线程数 | procs 扫进程数 |
            cleanup 只清理 dst.prefix 下的残留。
        workers: procs 模式下每个进程的线程数。
        workers_grid: read/workers 模式扫描的线程数档位。
        procs_grid: procs 模式扫描的进程数档位。
        trial_gb: 每档搬多少 GB 数据(源端按此预算采样对象)。
        settle_sec: 两档之间的冷却秒数。
        keep_objects: True 则保留写入目标端的基准对象(默认跑完即删)。

    Returns:
        {'mode': ..., 'rows': [每档的统计字典]}。
    """
    cfg = build_cfg(
        required=("src.bucket", "dst.prefix"),
        src=src, dst=dst, mode=mode, workers=workers,
        workers_grid=list(workers_grid or [16, 32, 64, 128]),
        procs_grid=list(procs_grid or [1, 2, 4, 8]),
        trial_gb=trial_gb, settle_sec=settle_sec, keep_objects=keep_objects,
    )

    if cfg.mode != "read" and "_bench" not in cfg.dst.prefix:
        raise ValueError(
            f"dst.prefix 必须包含 '_bench',收到 {cfg.dst.prefix!r}。"
            "这是防止基准数据写进真实路径的保护。")

    if cfg.mode == "cleanup":
        n = delete_prefix(cfg, cfg.dst.prefix)
        print(f"删除了 {n} 个对象 (前缀 {cfg.dst.prefix})")
        return {"mode": "cleanup", "deleted": n, "rows": []}

    if cfg.mode not in ("read", "workers", "procs"):
        raise ValueError(
            f"mode 只能是 read/workers/procs/cleanup,收到 {cfg.mode!r}")

    src_client = build_src(cfg, max(list(cfg.workers_grid) + [cfg.workers]))

    objs = list_source(src_client, cfg.src.bucket, cfg.src.prefix,
                       budget_bytes=int(cfg.trial_gb * 1e9))
    if not objs:
        raise RuntimeError(
            f"s3://{cfg.src.bucket}/{cfg.src.prefix} 下没有对象")

    total = sum(s for _, s in objs)
    mean_mb = total / len(objs) / 1e6
    print(f"[info] 样本 {len(objs)} 个对象 / {total/1e9:.2f} GB / "
          f"平均 {mean_mb:.1f} MB")
    if not 30 <= mean_mb <= 90:
        print(f"[warn] 平均对象 {mean_mb:.1f} MB 偏离全局平均 (58.7 MB),"
              "外推到全量会有偏差")
    del src_client   # 父进程的 client 不要被 fork 继承

    rows: List[Stats] = []

    if cfg.mode == "read":
        print("[info] 只读源端,不写目标端 —— 这是总吞吐的绝对上限")
        for w in cfg.workers_grid:
            st = run_trial(cfg, objs, 1, w, f"read w={w}", read_only=True)
            rows.append(st)
            print(f"  read w={w:<4} {st.mbps:6.0f} MB/s  errors={st.errors}",
                  flush=True)
            time.sleep(cfg.settle_sec)

    elif cfg.mode == "workers":
        for w in cfg.workers_grid:
            st = run_trial(cfg, objs, 1, w, f"w={w}")
            rows.append(st)
            print(f"  w={w:<4} {st.mbps:6.0f} MB/s  retries={st.retries} "
                  f"errors={st.errors}", flush=True)
            time.sleep(cfg.settle_sec)

    else:  # procs
        for p in cfg.procs_grid:
            st = run_trial(cfg, objs, p, cfg.workers, f"{p}p x {cfg.workers}w")
            rows.append(st)
            print(f"  procs={p:<3} (总并发 {p * cfg.workers:<5}) "
                  f"{st.mbps:6.0f} MB/s  retries={st.retries} "
                  f"errors={st.errors}", flush=True)
            time.sleep(cfg.settle_sec)

    report(rows)

    if cfg.mode == "procs":
        best = max(rows, key=lambda s: s.mbps)
        print(f"\n按 {best.mbps:.0f} MB/s (单节点) 估算 320 TB:")
        for nodes in (1, 2, 4, 8, 16):
            days = 320e12 / (best.mbps * 1e6 * nodes) / 86400
            print(f"  {nodes:2d} 个节点: {days:5.1f} 天")
        print("\n注意: 跨节点不会线性叠加 —— LUMI 出口和 Source Coop 的限流"
              "是全局共享的。上线后用 scontrol update ArrayTaskThrottle 逐步加。")

    if cfg.mode != "read" and not cfg.keep_objects:
        left = delete_prefix(cfg, cfg.dst.prefix)
        if left:
            print(f"[info] 补清理了 {left} 个残留对象")

    return {"mode": cfg.mode, "rows": [st.summary() for st in rows]}


if __name__ == "__main__":
    # 临时调试用;正式入口是 `python -m deploy.run run=sc_bench`。
    benchmark_transfer(
        src={"bucket": "32m-2024", "prefix": "32MND/",
             "rclone_remote": "lumi-465002698-private", "region": "us-east-1"},
        dst={"bucket": "geoai-ucph", "prefix": "gvsm/_bench/",
             "endpoint": "https://data.source.coop", "profile": "source-coop",
             "region": "us-east-1"},
        mode="read", workers_grid=[16, 32], trial_gb=1.0, settle_sec=2.0,
    )
