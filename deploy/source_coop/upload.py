#!/usr/bin/env python3
"""LUMI-O -> Source Cooperative 传输 (正式)

依赖: pip install --user hydra-core boto3

用法(默认值见 config/deploy/config.yaml 的 sc_upload):
    # 自检:纯逻辑 + 实网
    python -m deploy.run run=sc_upload run.mode=selftest \
        run.src.bucket=32m-2024 run.src.prefix=32MNC/ \
        run.dst.prefix=gvsm/_selftest/ run.state_prefix=/tmp/st

    # 干跑,只看路径映射
    python -m deploy.run run=sc_upload run.mode=dry \
        run.src.bucket=32m-2024 run.src.prefix=32MNC/ \
        run.dst.prefix=gvsm/original/2024/32MNC/ \
        run.state_prefix=$PWD/state/32MNC

    # 实传(整桶,tile 层级由源 key 自带)
    python -m deploy.run run=sc_upload \
        run.src.bucket=32m-2024 run.src.prefix="" \
        run.dst.prefix=gvsm/original/2024/ \
        run.state_prefix=$PWD/state/32m-2024 run.workers=64

    # 只重试失败的
    python -m deploy.run run=sc_upload ... run.only_failed=true

失败即抛异常(进程退出码非 0):致命配置/连通性问题抛 RuntimeError/ValueError,
有对象传输失败也抛 RuntimeError —— 进度写在状态文件里,重跑会续上。

凭证:
    source-coop login --duration 12h     # 12h 是上限,且从 login 那刻起算
`source-coop creds` 只是打印这份缓存,不会重新签发,所以整个作业必须跑在这个
窗口里。开传前会检查剩余有效期(min_cred_minutes,默认 30 分钟),不够就直接
失败 —— 排队几小时才轮到的 SLURM 作业最容易踩这个。
"""

from __future__ import annotations

import io
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError
from omegaconf import DictConfig, OmegaConf

from deploy.source_coop.common import (
    NONRETRYABLE, CountingStream, StateWriter, build_cfg, build_dst, build_src,
    list_source, load_lines, map_key, normalize_prefix, transfer_config,
)


# ================================================================ 传输
def transfer_one(src, dst, cfg: DictConfig, key: str, size: Optional[int],
                 xfer) -> int:
    """源端流式读 -> 目标端 multipart 写,内存占用是分片大小而不是整个对象。

    走 upload_fileobj 而不是 put_object:source.coop 对单次 PutObject 有大小
    上限,大 COG 会被 413 掉。
    """
    body = src.get_object(Bucket=cfg.src.bucket, Key=key)["Body"]
    stream = CountingStream(body)
    dst_key = map_key(cfg.src.prefix, cfg.dst.prefix, key)
    dst.upload_fileobj(stream, cfg.dst.bucket, dst_key, Config=xfer)

    # 流式写完才知道读了多少,所以这个校验只能后置 —— 对不上就把刚写进去的
    # 半成品删掉,不能留在目标端冒充完整对象。
    if size is not None and stream.count != size:
        try:
            dst.delete_object(Bucket=cfg.dst.bucket, Key=dst_key)
        except (ClientError, BotoCoreError):
            pass
        raise RuntimeError(
            f"读取 {stream.count} 字节,列举时是 {size} 字节(已删除残留)")
    return stream.count


def move(src, dst, cfg: DictConfig, state: StateWriter, key: str,
         size: Optional[int], xfer) -> int:
    """返回传输字节数;失败返回 -1(已记录,不向上抛)。"""
    last = ""
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            nbytes = transfer_one(src, dst, cfg, key, size, xfer)
            state.ok(key)
            return nbytes
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            last = f"{code}: {exc}"
            if code in NONRETRYABLE:
                break
        except (BotoCoreError, RuntimeError, ValueError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt < cfg.max_attempts:
            time.sleep(2 ** attempt)
    state.fail(key, last)
    return -1


# ================================================================ 自检
def selftest(cfg: DictConfig) -> int:
    """纯逻辑 + 实网两层。返回失败项数。"""
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, fn) -> bool:
        """返回是否通过。注意不要用返回值判断成功,函数可能返回 None。"""
        try:
            fn()
            results.append((name, True, ""))
            return True
        except Exception as exc:            # noqa: BLE001 自检要抓全部
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            return False

    # ---------------- 第一层:纯逻辑
    def t_normalize():
        assert normalize_prefix("59VMG") == "59VMG/"
        assert normalize_prefix("59VMG/") == "59VMG/"
        assert normalize_prefix("") == ""
    check("前缀归一化", t_normalize)

    def t_map_key():
        assert map_key("59VMG/", "gvsm/o/2024/59VMG/",
                       "59VMG/RH11_Q2.tif") == "gvsm/o/2024/59VMG/RH11_Q2.tif"
        # 嵌套子目录要保留(整桶模式依赖这条)
        assert map_key("", "gvsm/original/2024/",
                       "32MNC/RH11_Q2.tif") == "gvsm/original/2024/32MNC/RH11_Q2.tif"
        assert map_key("a/", "b/", "a/x/y.tif") == "b/x/y.tif"
        # 前缀不匹配必须报错,而不是悄悄切错字符
        try:
            map_key("a/", "b/", "zzz/y.tif")
        except ValueError:
            pass
        else:
            raise AssertionError("前缀不匹配时没有报错")
    check("key 路径映射", t_map_key)

    def t_state():
        tmp = tempfile.mkdtemp()
        try:
            done_p = os.path.join(tmp, "s.done")
            fail_p = os.path.join(tmp, "s.failed")
            st = StateWriter(done_p, fail_p)
            st.ok("a/1.tif")
            st.fail("a/2.tif", "AccessDenied: boom\nsecond line")
            st.close()
            assert load_lines(done_p) == {"a/1.tif"}
            assert load_lines(fail_p) == {"a/2.tif"}
            assert len(open(fail_p).read().splitlines()) == 1, "换行没压平"
            assert load_lines(os.path.join(tmp, "nope")) == set()
        finally:
            shutil.rmtree(tmp)
    check("状态文件读写", t_state)

    def t_concurrent_state():
        tmp = tempfile.mkdtemp()
        try:
            done_p = os.path.join(tmp, "c.done")
            st = StateWriter(done_p, os.path.join(tmp, "c.failed"))
            keys = [f"k/{i}.tif" for i in range(500)]
            with ThreadPoolExecutor(max_workers=32) as pool:
                list(pool.map(st.ok, keys))
            st.close()
            assert load_lines(done_p) == set(keys), "并发写丢行"
        finally:
            shutil.rmtree(tmp)
    check("并发写状态文件", t_concurrent_state)

    print("\n--- 纯逻辑 ---")
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {msg}" if msg else ""))

    if any(not ok for _, ok, _ in results):
        print("\n[fatal] 纯逻辑测试失败,不继续联网测试", file=sys.stderr)
        return sum(1 for _, ok, _ in results if not ok)

    # ---------------- 第二层:实网
    results.clear()
    holder: dict = {}

    def t_src():
        holder["src"] = build_src(cfg, cfg.workers)
    check("读取 rclone 配置并建立源端 client", t_src)

    def t_dst():
        holder["dst"], holder["cred"] = build_dst(cfg, cfg.workers)
    check("建立目标端 client", t_dst)

    cred = holder.get("cred")
    if cred is not None:
        def t_window():
            assert cred.minutes_left is not None, (
                f"凭证类型 {cred.kind} 没有 Expiration,无法预判失效时间")
            assert cred.minutes_left >= cfg.min_cred_minutes, (
                f"只剩 {cred.minutes_left:.0f} 分钟,不够跑长任务;"
                "先 source-coop login --duration 12h")
        check(f"凭证有效期充足 ({cred.describe()})", t_window)

    src = holder.get("src")
    if src is not None and not OmegaConf.is_missing(cfg.src, "bucket"):
        def t_src_list():
            resp = src.list_objects_v2(Bucket=cfg.src.bucket,
                                       Prefix=cfg.src.prefix, MaxKeys=3)
            if resp.get("KeyCount", 0) == 0:
                raise AssertionError(
                    f"s3://{cfg.src.bucket}/{cfg.src.prefix} 下没有对象")
        check("源端可列举", t_src_list)

    dst = holder.get("dst")
    if dst is not None:
        nbytes = int(cfg.selftest_size_mb * 1_000_000)
        payload = random.randbytes(nbytes)
        probe = cfg.dst.prefix + f".selftest_{os.getpid()}_{int(time.time())}"

        def t_put():
            dst.put_object(Bucket=cfg.dst.bucket, Key=probe, Body=payload)

        if check(f"目标端可写入 ({nbytes/1e6:.1f} MB)", t_put):
            def t_roundtrip():
                got = dst.get_object(Bucket=cfg.dst.bucket,
                                     Key=probe)["Body"].read()
                assert len(got) == nbytes, f"读回 {len(got)} != 写入 {nbytes}"
                assert got == payload, "读回内容与写入不一致"
            check("字节级往返校验", t_roundtrip)

            def t_head():
                h = dst.head_object(Bucket=cfg.dst.bucket, Key=probe)
                assert h["ContentLength"] == nbytes
            check("head_object 大小一致", t_head)

            def t_delete():
                dst.delete_object(Bucket=cfg.dst.bucket, Key=probe)
            if not check("清理探测对象", t_delete):
                print(f"  [warn] 请手动删除 {cfg.dst.bucket}/{probe}",
                      file=sys.stderr)

        # 大对象走 multipart —— source.coop 对单次 PutObject 有大小上限,超了
        # 直接 413 Request Entity Too Large。上面那个小探测对象在阈值以下,
        # 测不到这条路径,而真实的 COG 恰恰都在阈值附近,所以必须单独测。
        large_bytes = int(cfg.selftest_large_mb * 1_000_000)
        large_probe = cfg.dst.prefix + f".selftest_mp_{os.getpid()}_{int(time.time())}"

        def t_multipart():
            xfer = transfer_config(cfg)
            payload = io.BytesIO(b"\0" * large_bytes)
            dst.upload_fileobj(payload, cfg.dst.bucket, large_probe, Config=xfer)
            got = dst.head_object(Bucket=cfg.dst.bucket, Key=large_probe)
            assert got["ContentLength"] == large_bytes, (
                f"写入 {large_bytes} 字节,回读 {got['ContentLength']}")

        label = (f"大对象 multipart 写入 ({cfg.selftest_large_mb:.0f} MB, "
                 f"阈值 {cfg.multipart_threshold_mb:.0f} MB)")
        if check(label, t_multipart):
            def t_delete_large():
                dst.delete_object(Bucket=cfg.dst.bucket, Key=large_probe)
            if not check("清理 multipart 探测对象", t_delete_large):
                print(f"  [warn] 请手动删除 {cfg.dst.bucket}/{large_probe}",
                      file=sys.stderr)

    print("\n--- 实网 ---")
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {msg}" if msg else ""))

    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'全部通过' if not failed else f'{failed} 项失败'}")
    return failed


# ================================================================ 主流程
def check_credentials(cred, min_minutes: float) -> None:
    """凭证剩余有效期不够就直接失败,别传到一半撞 403。

    source-coop 的 token 最长 12h,且是从 `source-coop login` 那一刻开始算的 ——
    作业在 SLURM 队列里排了几小时的话,真正能用的窗口就只剩这么点。早失败比传
    一半失败干净:进度都在状态文件里,重新 login 后原命令扔回去就续上了。
    """
    print(f"[info] 目标端凭证: {cred.describe()}")

    if cred.minutes_left is None:
        print(f"[warn] 拿不到凭证的 Expiration(类型 {cred.kind}),"
              "无法预判会不会中途失效", file=sys.stderr)
        return

    if cred.minutes_left < min_minutes:
        raise RuntimeError(
            f"目标端凭证只剩 {cred.minutes_left:.0f} 分钟"
            f"(阈值 {min_minutes:.0f}),现在开传大概率会中途 403。\n"
            "        先跑 `source-coop login --duration 12h` 再重新提交;\n"
            "        已传的对象记在状态文件里,不会重传。\n"
            "        确实想跑就调低 run.min_cred_minutes。")


def run_transfer(cfg: DictConfig) -> Dict[str, Any]:
    src = build_src(cfg, cfg.workers)
    dst, cred = build_dst(cfg, cfg.workers)

    if cfg.mode != "dry":
        check_credentials(cred, cfg.min_cred_minutes)

    if "Refreshable" not in cred.kind:
        print(f"[warn] 目标端凭证类型是 {cred.kind},botocore 不会重调 "
              "credential_process,进程开始时拿到什么就用到底", file=sys.stderr)

    # 启动前确认两端都通,避免跑到一半才发现 403
    try:
        src.list_objects_v2(Bucket=cfg.src.bucket, Prefix=cfg.src.prefix,
                            MaxKeys=1)
    except (ClientError, BotoCoreError) as exc:
        raise RuntimeError(f"源端不可读 ({cfg.src.bucket}): {exc}")

    if cfg.mode != "dry":
        probe = cfg.dst.prefix + ".transfer_probe"
        try:
            dst.put_object(Bucket=cfg.dst.bucket, Key=probe, Body=b"ok")
            dst.delete_object(Bucket=cfg.dst.bucket, Key=probe)
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(
                f"目标端不可写 ({cfg.dst.bucket}/{probe}): {exc}")

    keys = list_source(src, cfg.src.bucket, cfg.src.prefix)
    if not keys:
        msg = f"源前缀下没有任何对象: s3://{cfg.src.bucket}/{cfg.src.prefix}"
        if cfg.allow_empty:
            print(f"[warn] {msg}(allow_empty=true,按成功退出)")
            return {"sent": 0, "failed": 0, "bytes": 0, "skipped_empty": True}
        raise RuntimeError(
            f"{msg}\n"
            "        如果这是预期内的,加 run.allow_empty=true。\n"
            "        否则请检查桶名和前缀 —— 静默跳过会造成数据缺失。")

    total_objects = len(keys)
    total_bytes = sum(s for _, s in keys)

    done_path = cfg.state_prefix + ".done"
    failed_path = cfg.state_prefix + ".failed"
    done = load_lines(done_path)

    if cfg.only_failed:
        wanted = load_lines(failed_path) - done
        keys = [(k, s) for k, s in keys if k in wanted]
        open(failed_path, "w").close()      # 清空,避免旧记录累积
    else:
        keys = [(k, s) for k, s in keys if k not in done]

    pending_bytes = sum(s for _, s in keys)
    print(f"[info] 源端共 {total_objects} 个对象 / {total_bytes/1e9:.1f} GB")
    print(f"[info] 待传 {len(keys)} 个 / {pending_bytes/1e9:.1f} GB "
          f"(已完成 {len(done)})")

    if cfg.mode == "dry":
        for key, size in keys[:10]:
            print(f"  {key}\n      -> "
                  f"{map_key(cfg.src.prefix, cfg.dst.prefix, key)}"
                  f"  ({size/1e6:.1f} MB)")
        if len(keys) > 10:
            print(f"  ... 另外 {len(keys) - 10} 个")
        return {"sent": 0, "failed": 0, "bytes": 0, "dry_run": len(keys)}

    if not keys:
        print("[info] 没有待传对象,已全部完成")
        return {"sent": 0, "failed": 0, "bytes": 0}

    state = StateWriter(done_path, failed_path)
    xfer = transfer_config(cfg)
    sent = failed = 0
    sent_bytes = 0
    started = time.time()

    stopped_early = False
    # 窗口速率的基准点。只打累计均值的话,跑了几天之后速率掉一半也看不出来 ——
    # 分母太大,均值几乎不动。长任务必须看最近一段的瞬时值。
    win_t0, win_bytes0 = started, 0
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            for start in range(0, len(keys), cfg.submit_chunk):
                # 凭证快到期就停止提交新对象,等在途的传完再退出。撞上 403 的话
                # 会在 .failed 里留下一批本可避免的记录,还得再扫一遍;干净退出
                # 之后只要重新登录、原命令重跑,直接从 .done 之后接着走。
                if cfg.stop_before_cred_expiry:
                    left = cred.refresh()
                    if left is not None and left < cfg.stop_before_cred_expiry:
                        stopped_early = True
                        break

                batch = keys[start:start + cfg.submit_chunk]
                futures = [pool.submit(move, src, dst, cfg, state, k, s, xfer)
                           for k, s in batch]
                for fut in as_completed(futures):
                    nbytes = fut.result()
                    if nbytes < 0:
                        failed += 1
                    else:
                        sent += 1
                        sent_bytes += nbytes
                    if (sent + failed) % cfg.progress_every == 0:
                        now = time.time()
                        elapsed = now - started
                        rate = sent_bytes / elapsed / 1e6 if elapsed else 0
                        win_sec = now - win_t0
                        win_rate = ((sent_bytes - win_bytes0) / win_sec / 1e6
                                    if win_sec else 0)
                        win_t0, win_bytes0 = now, sent_bytes
                        left_bytes = max(0, pending_bytes - sent_bytes)
                        eta_h = (left_bytes / (win_rate * 1e6) / 3600
                                 if win_rate else 0)
                        print(f"[prog] {sent + failed}/{len(keys)}  "
                              f"ok={sent} fail={failed}  "
                              f"{sent_bytes/1e9:.1f}/{pending_bytes/1e9:.1f} GB  "
                              f"{win_rate:.0f} MB/s (近 {win_sec:.0f}s)  "
                              f"均值 {rate:.0f}  剩余 ~{eta_h:.1f}h",
                              flush=True)
    except KeyboardInterrupt:
        print(f"\n[warn] 收到中断,进度已保存在 {done_path},重跑会续上",
              file=sys.stderr)
        raise
    finally:
        state.close()

    elapsed = time.time() - started
    rate = sent_bytes / elapsed / 1e6 if elapsed else 0
    print(f"[done] 成功 {sent}  失败 {failed}  {sent_bytes/1e9:.2f} GB  "
          f"用时 {elapsed/60:.1f} 分钟  平均 {rate:.0f} MB/s")

    if stopped_early:
        remaining = len(keys) - sent - failed
        left = cred.minutes_left
        print(f"\n[stop] 凭证剩余 {left:.0f} 分钟(阈值 "
              f"{cfg.stop_before_cred_expiry:.0f}),已停止提交新对象并干净退出。\n"
              f"       本轮完成 {sent} 个,还剩 {remaining} 个未传。\n"
              f"       重新登录后原样重跑同一条命令即可续上(已完成的不会重传):\n"
              f"         source-coop login --duration 12h\n"
              f"         python -m deploy.run run=sc_upload ...(原命令)")

    if failed:
        raise RuntimeError(
            f"{failed} 个对象失败,明细见 {failed_path};"
            "重试: 同样的命令加 run.only_failed=true")
    return {"sent": sent, "failed": failed, "bytes": sent_bytes,
            "elapsed_sec": elapsed, "stopped_early": stopped_early,
            "remaining": len(keys) - sent - failed}


# ================================================================ op 入口
def upload_to_source_coop(
    src: Any = None,
    dst: Any = None,
    state_prefix: str = None,
    mode: str = "run",              # run | dry | selftest
    workers: int = 32,
    submit_chunk: int = 2000,       # 分批提交,避免一次建几百万个 future
    max_attempts: int = 3,          # 单对象外层重试次数
    only_failed: bool = False,
    allow_empty: bool = False,
    progress_every: int = 100,
    min_cred_minutes: float = 30.0,
    stop_before_cred_expiry: float = 20.0,
    multipart_threshold_mb: float = 32.0,
    multipart_chunk_mb: float = 32.0,
    selftest_size_mb: float = 1.0,
    selftest_large_mb: float = 128.0,
    **kwargs,
) -> Dict[str, Any]:
    """把 LUMI-O 上的 VSM 预测 COG 传到 Source Cooperative。

    Args:
        src: 源端(LUMI-O)配置 —— bucket / prefix / rclone_remote / region。
            凭证从 `rclone config dump` 读取。
        dst: 目标端(Source Cooperative)配置 —— bucket / prefix / endpoint /
            profile / region。凭证走 aws profile 的 credential_process。
        state_prefix: 断点续传状态文件前缀,会写 <prefix>.done 和
            <prefix>.failed;重跑时已在 .done 里的 key 直接跳过。
        mode: run 实传 | dry 只打印路径映射 | selftest 纯逻辑+实网自检。
        workers: 上传线程数(连接池按此自动放大)。
        submit_chunk: 每批提交给线程池的对象数。
        max_attempts: 单个对象的外层重试次数(NONRETRYABLE 错误不重试)。
        only_failed: 只重传 <state_prefix>.failed 里的 key。
        allow_empty: 源前缀为空时按成功退出而不是报错。
        progress_every: 每处理多少个对象打一行进度。
        min_cred_minutes: 目标端凭证剩余有效期低于这个值就直接失败,不开传
            (mode=dry 不检查)。source-coop 的 token 最长 12h 且从 login 起算。
        stop_before_cred_expiry: 跑到剩余有效期低于这个值时,停止提交新对象、
            等在途的传完、干净退出(不算失败)。0 表示关掉,一直跑到撞 403。
        multipart_threshold_mb: 超过这个大小走 multipart。source.coop 对单次
            PutObject 有上限,大对象直接 413,所以这个值不能设得太高。
        multipart_chunk_mb: multipart 的分片大小,也是单个线程的内存占用。
        selftest_size_mb: selftest 往返校验写入的探测对象大小 (MB)。
        selftest_large_mb: selftest 里 multipart 探测对象的大小 (MB),要设成
            大于实测的 413 阈值(~110 MB),否则测不到真正会挂的那条路径。

    Returns:
        {'sent': ..., 'failed': ..., 'bytes': ...} 之类的统计字典。
    """
    cfg = build_cfg(
        required=("src.bucket", "dst.prefix", "state_prefix"),
        src=src, dst=dst, state_prefix=state_prefix, mode=mode,
        workers=workers, submit_chunk=submit_chunk, max_attempts=max_attempts,
        only_failed=only_failed, allow_empty=allow_empty,
        progress_every=progress_every, min_cred_minutes=min_cred_minutes,
        stop_before_cred_expiry=stop_before_cred_expiry,
        multipart_threshold_mb=multipart_threshold_mb,
        multipart_chunk_mb=multipart_chunk_mb,
        selftest_size_mb=selftest_size_mb, selftest_large_mb=selftest_large_mb,
    )

    if cfg.mode == "selftest":
        failed = selftest(cfg)
        if failed:
            raise RuntimeError(f"自检有 {failed} 项失败")
        return {"selftest_failed": 0}

    if cfg.mode not in ("run", "dry"):
        raise ValueError(
            f"mode 只能是 run / dry / selftest,收到 {cfg.mode!r}")

    state_dir = os.path.dirname(cfg.state_prefix)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    return run_transfer(cfg)


if __name__ == "__main__":
    # 临时调试用;正式入口是 `python -m deploy.run run=sc_upload`。
    upload_to_source_coop(
        src={"bucket": "32m-2024", "prefix": "32MNC/",
             "rclone_remote": "lumi-465002698-private", "region": "us-east-1"},
        dst={"bucket": "geoai-ucph", "prefix": "gvsm/_selftest/",
             "endpoint": "https://data.source.coop", "profile": "source-coop",
             "region": "us-east-1"},
        state_prefix="/tmp/sc_selftest", mode="selftest",
    )
