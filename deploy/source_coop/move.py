#!/usr/bin/env python3
"""Source Cooperative 内部搬迁:改前缀,不重传字节。

S3 没有 rename,但 source.coop 的网关支持服务端 CopyObject(实测 ETag 与源
完全一致,不经过本地),所以调整已发布的目录结构不需要从 hendrix 重传一遍 ——
copy 完确认 ETag 一致再删源。

    # 干跑,只看前 20 条映射和总量
    python -m deploy.run run=sc_move run.mode=dry \
        run.src_prefix=gvsm/2020/ run.dst_prefix=gvsm/2020/tiles/

    # 实搬(可中断,重跑续上)
    python -m deploy.run run=sc_move \
        run.src_prefix=gvsm/2020/ run.dst_prefix=gvsm/2020/tiles/ \
        run.state_prefix=$PWD/state/move_2020_tiles run.workers=128

    # 只重试失败的
    python -m deploy.run run=sc_move ... run.only_failed=true

凭证窗口和 upload.py 一样是硬约束:token 最长 12h 且从 login 起算,183 万个
对象搬不完一轮,所以状态文件必须留着 —— 重新 login 后原命令扔回去就接着走。

嵌套前缀是允许的(gvsm/2020/ -> gvsm/2020/tiles/):列举在前、搬迁在后,且每个
key 会跳过已经落在目标前缀下的那些,不会自己搬自己。
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError
from omegaconf import DictConfig

from deploy.source_coop.common import (
    NONRETRYABLE, StateWriter, build_cfg, build_dst, load_lines, map_key,
    normalize_prefix,
)
from deploy.source_coop.upload import check_credentials


def list_prefix(dst, bucket: str, prefix: str,
                skip_prefix: Optional[str] = None) -> List[Tuple[str, int, str]]:
    """列举前缀下所有对象,返回 (key, size, etag)。

    skip_prefix 下的对象直接跳过 —— 目标前缀嵌套在源前缀里时(gvsm/2020/ ->
    gvsm/2020/tiles/),已经搬过去的对象会再次出现在列举结果里,不排掉就会被
    反复搬运,而且第二轮的 dst_key 会变成 tiles/tiles/...。
    """
    out: List[Tuple[str, int, str]] = []
    paginator = dst.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if skip_prefix and key.startswith(skip_prefix):
                continue
            out.append((key, obj["Size"], obj.get("ETag", "").strip('"')))
        if len(out) % 200_000 == 0 and out:
            print(f"[info] 已列举 {len(out)} 个对象...", flush=True)
    return out


def move_one(dst, cfg: DictConfig, key: str, size: int, etag: str) -> bool:
    """copy -> 校验 -> delete。任何一步不过就不删源。"""
    dst_key = map_key(cfg.src_prefix, cfg.dst_prefix, key)

    resp = dst.copy_object(Bucket=cfg.dst_bucket, Key=dst_key,
                           CopySource={"Bucket": cfg.dst_bucket, "Key": key})
    new_etag = (resp.get("CopyObjectResult", {}).get("ETag") or "").strip('"')

    # ETag 对不上就把刚写的副本删掉,源保持原样 —— 宁可这一条没搬成,也不能
    # 删掉源之后留下一个内容存疑的副本。
    if etag and new_etag and new_etag != etag:
        try:
            dst.delete_object(Bucket=cfg.dst_bucket, Key=dst_key)
        except (ClientError, BotoCoreError):
            pass
        raise RuntimeError(f"ETag 不一致: 源 {etag} 副本 {new_etag}(已删副本)")

    if not cfg.keep_source:
        dst.delete_object(Bucket=cfg.dst_bucket, Key=key)
    return True


def move_with_retry(dst, cfg: DictConfig, state: StateWriter, key: str,
                    size: int, etag: str) -> bool:
    last = ""
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            move_one(dst, cfg, key, size, etag)
            state.ok(key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            last = f"{code}: {exc}"
            if code in NONRETRYABLE:
                break
        except (BotoCoreError, RuntimeError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < cfg.max_attempts:
            time.sleep(2 ** attempt)
    state.fail(key, last)
    return False


def run_move(cfg: DictConfig) -> Dict[str, Any]:
    dst, cred = build_dst(cfg, cfg.workers)
    if cfg.mode != "dry":
        check_credentials(cred, cfg.min_cred_minutes)

    nested = cfg.dst_prefix.startswith(cfg.src_prefix)
    print(f"[info] {cfg.dst_bucket}: {cfg.src_prefix} -> {cfg.dst_prefix}"
          f"{'  (目标嵌套在源下,列举时排除)' if nested else ''}")

    keys = list_prefix(dst, cfg.dst_bucket, cfg.src_prefix,
                       skip_prefix=cfg.dst_prefix if nested else None)
    total_bytes = sum(s for _, s, _ in keys)
    print(f"[info] 源端 {len(keys)} 个对象 / {total_bytes/1e9:.1f} GB")
    if not keys:
        return {"moved": 0, "failed": 0, "skipped_empty": True}

    if cfg.mode == "dry":
        for key, size, _ in keys[:20]:
            print(f"  {key}\n      -> {map_key(cfg.src_prefix, cfg.dst_prefix, key)}"
                  f"  ({size/1e6:.1f} MB)")
        if len(keys) > 20:
            print(f"  ... 另外 {len(keys) - 20} 个")
        return {"would_move": len(keys), "bytes": total_bytes}

    done_path = f"{cfg.state_prefix}.done"
    failed_path = f"{cfg.state_prefix}.failed"
    if cfg.only_failed:
        retry = load_lines(failed_path)
        keys = [k for k in keys if k[0] in retry]
        print(f"[info] only_failed: 只搬 {len(keys)} 个")
    else:
        already = load_lines(done_path)
        if already:
            keys = [k for k in keys if k[0] not in already]
            print(f"[info] 跳过已完成的 {len(already)} 个,待搬 {len(keys)} 个")

    state = StateWriter(done_path, failed_path)
    moved = failed = 0
    stopped_early = False
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            for start in range(0, len(keys), cfg.submit_chunk):
                if cfg.stop_before_cred_expiry:
                    left = cred.refresh()
                    if left is not None and left < cfg.stop_before_cred_expiry:
                        stopped_early = True
                        break
                batch = keys[start:start + cfg.submit_chunk]
                futures = {
                    pool.submit(move_with_retry, dst, cfg, state, k, s, e): k
                    for k, s, e in batch
                }
                for future in as_completed(futures):
                    if future.result():
                        moved += 1
                    else:
                        failed += 1
                    if (moved + failed) % cfg.progress_every == 0:
                        rate = (moved + failed) / max(time.time() - t0, 1e-9)
                        eta = (len(keys) - moved - failed) / max(rate, 1e-9) / 3600
                        print(f"  {moved + failed}/{len(keys)}  {rate:.0f} obj/s  "
                              f"eta {eta:.1f}h  {failed} failed", flush=True)
    finally:
        state.close()

    elapsed = time.time() - t0
    print(f"\n搬迁 {moved} 个,失败 {failed} 个,用时 {elapsed/3600:.2f} h")
    if stopped_early:
        print(f"[info] 凭证快到期(阈值 {cfg.stop_before_cred_expiry:.0f} 分钟),"
              "已干净退出。重新 login 后原命令重跑即可续上。")
    if failed:
        raise RuntimeError(
            f"{failed} 个对象搬迁失败,明细见 {failed_path};"
            "重试: 同样的命令加 run.only_failed=true")
    return {"moved": moved, "failed": failed, "elapsed_sec": elapsed,
            "stopped_early": stopped_early,
            "remaining": len(keys) - moved - failed}


# ================================================================ op 入口
def move_within_source_coop(
    dst_bucket: str = "geoai-ucph",
    src_prefix: str = None,
    dst_prefix: str = None,
    endpoint: str = "https://data.source.coop",
    profile: str = "source-coop",
    region: str = "us-east-1",
    state_prefix: str = None,
    mode: str = "run",              # run | dry
    workers: int = 128,
    submit_chunk: int = 5000,
    max_attempts: int = 3,
    only_failed: bool = False,
    progress_every: int = 1000,
    keep_source: bool = False,
    min_cred_minutes: float = 30.0,
    stop_before_cred_expiry: float = 20.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """把已发布对象搬到另一个前缀下(服务端 copy + 删源)。

    Args:
        dst_bucket: source.coop 的 bucket,源和目标是同一个。
        src_prefix / dst_prefix: 搬迁前后的前缀。目标可以嵌套在源下面。
        endpoint / profile / region: 同 sc_upload,凭证走 credential_process。
        state_prefix: 断点续传状态,会写 <prefix>.done 和 .failed。
        mode: run 实搬 | dry 只打印映射和总量。
        workers: 并发数。实测服务端 copy 随并发线性提升(64 并发约 37 obj/s)。
        submit_chunk: 每批提交给线程池的对象数;批与批之间检查凭证剩余时间。
        keep_source: true 只复制不删源(先验证一轮再决定删的时候用)。
        min_cred_minutes / stop_before_cred_expiry: 同 sc_upload。183 万个对象
            一个 12h 窗口搬不完,靠状态文件跨窗口续。

    Returns:
        {'moved': ..., 'failed': ...} 之类的统计字典。
    """
    cfg = build_cfg(
        required=("src_prefix", "dst_prefix") + (
            () if mode == "dry" else ("state_prefix",)),
        dst_bucket=dst_bucket, src_prefix=src_prefix, dst_prefix=dst_prefix,
        endpoint=endpoint, profile=profile, region=region,
        state_prefix=state_prefix, mode=mode, workers=workers,
        submit_chunk=submit_chunk, max_attempts=max_attempts,
        only_failed=only_failed, progress_every=progress_every,
        keep_source=keep_source, min_cred_minutes=min_cred_minutes,
        stop_before_cred_expiry=stop_before_cred_expiry,
    )
    # build_dst 按 cfg.dst.* 取值,这个 op 的源和目标是同一个 bucket,所以在
    # 这里补一个等价的 dst 节点,而不是让调用方写两遍。
    cfg.dst = {"bucket": cfg.dst_bucket, "prefix": cfg.dst_prefix,
               "endpoint": endpoint, "profile": profile, "region": region}
    cfg.src_prefix = normalize_prefix(cfg.src_prefix)
    cfg.dst_prefix = normalize_prefix(cfg.dst_prefix)

    if cfg.src_prefix == cfg.dst_prefix:
        raise ValueError("src_prefix 和 dst_prefix 相同,没有可搬的东西")
    if cfg.src_prefix.startswith(cfg.dst_prefix):
        raise ValueError(
            f"dst_prefix {cfg.dst_prefix!r} 是 src_prefix {cfg.src_prefix!r} 的"
            "父前缀,搬过去会把目录层级拍平并可能互相覆盖")

    if cfg.mode not in ("run", "dry"):
        raise ValueError(f"mode 只能是 run / dry,收到 {cfg.mode!r}")
    return run_move(cfg)
