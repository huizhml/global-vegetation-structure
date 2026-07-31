#!/usr/bin/env python3
"""LUMI-O -> Source Cooperative 共享层

upload.py(正式传输)和 bench.py(基准测试)都 import 这里的东西。
凡是会影响"数据落在哪里"的逻辑都必须放在这个文件里,不允许两边各写一份。

配置本身不在这里定义 —— 默认值全部下沉到 config/deploy/config.yaml,
这里只接收组装好的 DictConfig。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
from omegaconf import DictConfig, OmegaConf

# 重试没有意义的错误,直接判失败
NONRETRYABLE = {
    "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch",
    "NoSuchKey", "NoSuchBucket", "ExpiredToken",
}


# ================================================================ 配置
def build_cfg(required: Tuple[str, ...] = (), **fields: Any) -> DictConfig:
    """把 op 函数收到的 kwargs 组装成内部统一的 DictConfig。

    内部函数(build_src / delete_prefix / ...)一律按 cfg.src.bucket 这种路径
    取值,子进程之间也靠 OmegaConf.to_yaml 传递,所以入口处先归一成 DictConfig。

    YAML 里必填项一律留 null(而不是 ???),在这里统一报错 —— MISSING 会先在
    Hydra 的 resolve_args 里炸掉,报出来的错和真正的原因对不上。
    """
    cfg = OmegaConf.create({
        k: (OmegaConf.to_container(v, resolve=True)
            if isinstance(v, DictConfig) else v)
        for k, v in fields.items()
    })
    missing = [p for p in required
               if OmegaConf.select(cfg, p) in (None, "")]
    if missing:
        raise ValueError(
            f"以下配置项必须提供: {missing}。"
            f"例: python -m deploy.run run=... " +
            " ".join(f"run.{p}=<值>" for p in missing))
    normalize_cfg_prefixes(cfg)
    return cfg


# ================================================================ 纯逻辑
def normalize_prefix(prefix: str) -> str:
    """非空前缀必须以 / 结尾,否则拼出的 key 会带双斜杠。"""
    if prefix and not prefix.endswith("/"):
        return prefix + "/"
    return prefix


def map_key(src_prefix: str, dst_prefix: str, key: str) -> str:
    """源 key -> 目标 key。前缀必须已经归一化过。

    这个函数决定 545 万个对象落在哪里,改动前先跑
    `python -m deploy.run run=sc_upload run.mode=selftest`。
    """
    if not key.startswith(src_prefix):
        raise ValueError(f"key {key!r} 不在前缀 {src_prefix!r} 下")
    return dst_prefix + key[len(src_prefix):]


def normalize_cfg_prefixes(cfg: DictConfig) -> None:
    """就地归一化 src/dst 前缀。必须在任何 key 拼接之前调用。"""
    for node, name in ((cfg.get("src"), "src.prefix"),
                       (cfg.get("dst"), "dst.prefix")):
        if node is None or OmegaConf.is_missing(node, "prefix"):
            continue
        before = node.prefix
        node.prefix = normalize_prefix(before)
        if node.prefix != before:
            print(f"[info] {name} 补上了结尾斜杠: {before} -> {node.prefix}")


def retry_count(resp: Optional[dict]) -> int:
    """botocore 自动重试的次数。限流常常不报错,只表现为这个数字上升。"""
    if not resp:
        return 0
    return resp.get("ResponseMetadata", {}).get("RetryAttempts", 0) or 0


# ================================================================ 状态文件
def load_lines(path: str) -> set:
    """读状态文件。failed 文件是 key<TAB>reason,只取 key。"""
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {line.split("\t")[0].strip() for line in fh if line.strip()}


class StateWriter:
    """线程安全地追加写状态文件。"""

    def __init__(self, done_path: str, failed_path: str):
        self._lock = threading.Lock()
        self._done = open(done_path, "a")
        self._failed = open(failed_path, "a")

    def ok(self, key: str) -> None:
        with self._lock:
            self._done.write(key + "\n")
            self._done.flush()

    def fail(self, key: str, reason: str) -> None:
        # 压平换行,否则状态文件会错行
        reason = " ".join(str(reason).split())[:500]
        with self._lock:
            self._failed.write(f"{key}\t{reason}\n")
            self._failed.flush()

    def close(self) -> None:
        self._done.close()
        self._failed.close()


# ================================================================ 客户端
def boto_config(workers: int = 32) -> Config:
    """连接池必须 >= 并发数,否则线程会静默排队,调优结果全部失效。"""
    return Config(
        max_pool_connections=max(64, workers * 2),
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=30,
        read_timeout=300,
    )


def transfer_config(cfg: DictConfig) -> TransferConfig:
    """写目标端必须走 multipart。

    source.coop 的网关对单次 PutObject 有大小上限 —— 实测 ~110 MB 的 COG 会
    直接返回 `413 Request Entity Too Large`,而同一批里偏小的对象能过。
    `aws s3 sync` 之所以一直没暴露这个问题,是因为它超过 8 MB 就自动切片。

    use_threads=False:外层(bench 的 workers / upload 的线程池)已经有自己的
    并发了,再让 s3transfer 开一层,实际并发会变成 workers × max_concurrency,
    调优结果全部失真,也更容易撞对方限流。
    """
    return TransferConfig(
        multipart_threshold=int(cfg.multipart_threshold_mb * 1024 ** 2),
        multipart_chunksize=int(cfg.multipart_chunk_mb * 1024 ** 2),
        use_threads=False,
    )


class CountingStream:
    """透传读取并累计字节数。

    流式上传(upload_fileobj)不像 put_object 那样能先 len(body),但"读到的字节
    数必须等于列举时的大小"这条完整性检查不能丢,所以在流上数。
    """

    def __init__(self, raw):
        self._raw = raw
        self.count = 0

    def read(self, amt=None):
        chunk = self._raw.read() if amt is None else self._raw.read(amt)
        self.count += len(chunk)
        return chunk


def drain(body, chunk_bytes: int = 8 * 1024 * 1024) -> int:
    """只读不留 —— 返回读到的字节数,内存占用是 chunk_bytes 而不是整个对象。

    read 模式下 128 个 worker × 121 MB 的对象全读进内存就是 15 GB,直接 OOM。
    """
    total = 0
    while True:
        chunk = body.read(chunk_bytes)
        if not chunk:
            return total
        total += len(chunk)


def build_src(cfg: DictConfig, workers: int = 32):
    """源端凭证从 rclone 配置读取,不另外维护一份。"""
    try:
        dump = subprocess.check_output(["rclone", "config", "dump"], text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "rclone 不在 PATH 里,无法读取 LUMI-O 凭证。"
            "批处理作业中 PATH 常与登录 shell 不同,请用绝对路径或加载 module。")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"rclone config dump 失败: {exc}")

    conf = json.loads(dump)
    name = cfg.src.rclone_remote
    if name not in conf:
        raise RuntimeError(
            f"rclone 配置里没有 remote {name!r}。现有: {sorted(conf)}")

    remote = conf[name]
    missing = [k for k in ("endpoint", "access_key_id", "secret_access_key")
               if not remote.get(k)]
    if missing:
        raise RuntimeError(f"remote {name!r} 缺少字段: {missing}")

    return boto3.client(
        "s3",
        endpoint_url=remote["endpoint"],
        aws_access_key_id=remote["access_key_id"],
        aws_secret_access_key=remote["secret_access_key"],
        region_name=remote.get("region") or cfg.src.region,
        config=boto_config(workers),
    )


@dataclass
class CredInfo:
    """目标端凭证的状态。"""
    kind: str = "None"                      # botocore 的凭证类名
    minutes_left: Optional[float] = None    # None = 拿不到 Expiration
    creds: Any = None                       # botocore 凭证对象,用于重新查询

    def refresh(self) -> Optional[float]:
        """重新查剩余时间。

        长任务跑几小时后要重新问一次 —— botocore 会在快到期时重调
        credential_process,如果外面有东西(比如登录节点上的刷新循环)更新了
        source-coop 的缓存,这里就能看到延长后的过期时间。
        """
        self.minutes_left = _minutes_left(self.creds)
        return self.minutes_left

    def describe(self) -> str:
        if self.minutes_left is None:
            return f"{self.kind}(无 Expiration 信息)"
        return f"{self.kind},剩余 {self.minutes_left:.0f} 分钟"


def _minutes_left(creds) -> Optional[float]:
    """凭证还剩多少分钟。拿不到过期时间就返回 None。

    注意 source-coop CLI 不会自动续期 —— `source-coop creds` 只是打印
    `source-coop login` 缓存下来的那份 token(最长 12h),所以这个剩余时间是
    从 login 那一刻算起的硬上限,不会因为 botocore 重调 credential_process 而
    变长。作业排队几小时再启动的话,真正可用的窗口只剩这么多。
    """
    if creds is None:
        return None
    try:
        creds.get_frozen_credentials()   # 触发 credential_process,填充过期时间
    except Exception:                    # noqa: BLE001 取不到就当未知
        return None
    # _expiry_time 是 botocore 的私有属性,静态凭证上不存在。
    expiry = getattr(creds, "_expiry_time", None)
    if expiry is None:
        return None
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return (expiry - datetime.now(timezone.utc)).total_seconds() / 60


def build_dst(cfg: DictConfig, workers: int = 32) -> Tuple[Any, CredInfo]:
    """目标端凭证走 aws profile 的 credential_process。

    返回 (client, CredInfo)。类型名里没有 "Refreshable" 说明 botocore 连重调
    credential_process 都不会做,进程一开始拿到什么就用到底。
    """
    try:
        session = boto3.Session(profile_name=cfg.dst.profile)
        client = session.client("s3", endpoint_url=cfg.dst.endpoint,
                                region_name=cfg.dst.region,
                                config=boto_config(workers))
    except ProfileNotFound:
        raise RuntimeError(
            f"~/.aws/config 里没有 [profile {cfg.dst.profile}]。"
            "需要配置 credential_process = source-coop creds")

    creds = session.get_credentials()
    return client, CredInfo(
        kind=type(creds).__name__ if creds else "None",
        minutes_left=_minutes_left(creds),
        creds=creds,
    )


# ================================================================ 列举
def list_source(src, bucket: str, prefix: str,
                budget_bytes: Optional[int] = None) -> List[Tuple[str, int]]:
    """列举源端对象。budget_bytes 非空时,凑够这么多字节就停(基准采样用)。"""
    out: List[Tuple[str, int]] = []
    total = 0
    paginator = src.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/") or obj["Size"] == 0:
                continue
            if not obj["Key"].startswith(prefix):
                print(f"[warn] key 不在前缀下,跳过: {obj['Key']}",
                      file=sys.stderr)
                continue
            out.append((obj["Key"], obj["Size"]))
            total += obj["Size"]
            if budget_bytes is not None and total >= budget_bytes:
                return out
    return out


def delete_prefix(cfg: DictConfig, prefix: str) -> int:
    """删除某前缀下所有对象。基准测试清理用。

    先试批量 DeleteObjects,不行就逐个删 —— source.coop 的网关是自研的,批量
    删除会报 NoSuchBucket(错误码驴唇不对马嘴,单对象 delete_object 明明是通的)。
    同一个网关拒 rclone 也是这类实现差异,别指望它 S3 兼容得很完整。
    """
    dst, _ = build_dst(cfg)
    removed = 0
    bulk_ok = True
    paginator = dst.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=cfg.dst.bucket, Prefix=prefix):
            keys = [o["Key"] for o in page.get("Contents", [])]
            for i in range(0, len(keys), 1000):
                chunk = keys[i:i + 1000]
                if bulk_ok:
                    try:
                        dst.delete_objects(
                            Bucket=cfg.dst.bucket,
                            Delete={"Objects": [{"Key": k} for k in chunk]})
                        removed += len(chunk)
                        continue
                    except ClientError as exc:
                        code = exc.response.get("Error", {}).get("Code", "?")
                        print(f"[info] 批量删除不可用 ({code}),改为逐个删除",
                              file=sys.stderr)
                        bulk_ok = False
                for key in chunk:
                    dst.delete_object(Bucket=cfg.dst.bucket, Key=key)
                    removed += 1
    except (ClientError, BotoCoreError) as exc:
        print(f"[warn] 清理 {prefix} 失败(已删 {removed} 个): {exc}\n"
              f"       剩下的请手动删: aws --profile {cfg.dst.profile} "
              f"--endpoint-url {cfg.dst.endpoint} s3 rm "
              f"s3://{cfg.dst.bucket}/{prefix} --recursive",
              file=sys.stderr)
    return removed


__all__ = [
    "NONRETRYABLE", "CountingStream", "CredInfo", "StateWriter", "build_cfg",
    "build_dst", "build_src", "boto_config", "delete_prefix", "drain",
    "list_source", "load_lines", "map_key", "normalize_cfg_prefixes",
    "normalize_prefix", "retry_count", "transfer_config",
]
