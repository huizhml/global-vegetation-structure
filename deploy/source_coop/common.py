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
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

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


def src_kind(cfg: DictConfig) -> str:
    """源端类型:s3(LUMI-O)、local(本地目录)或 stac(STAC collection)。

    2020 年的数据在 hendrix 本地盘上,2024 年的在 LUMI-O 对象存储上。目标端和
    所有运维逻辑(凭证窗口、multipart、断点续传)完全一样,只有"从哪读"不同,
    所以做成一个开关而不是两套代码。

    stac 是 local 的一个变体:2020 年的 tile 分散在三个不同的 root 下
    (original/tiles/cog、original/tiles/geotiff、masked/tiles/geotiff),
    只有 item JSON 知道每个 tile 的文件到底在哪,所以不能用一个 dir + glob 去
    扫 —— 那样会挑错副本。
    """
    return cfg.src.get("kind", "s3") if "src" in cfg else "s3"


SRC_REQUIRED = {
    "s3": ("src.bucket",),
    "local": ("src.dir",),
    "stac": ("src.catalog",),
}


def src_required(src) -> Tuple[str, ...]:
    """按源端类型给出必填字段。

    每加一种 src.kind 都必须在这里登记 —— 否则会掉进"默认要 bucket"的分支,
    报一个和真实原因完全无关的错(本地/STAC 上传根本没有 bucket 这个概念)。
    未知 kind 直接报错,不静默退化。
    """
    kind = (src or {}).get("kind", "s3") or "s3"
    if kind not in SRC_REQUIRED:
        raise ValueError(
            f"src.kind 只能是 {sorted(SRC_REQUIRED)},收到 {kind!r}")
    return SRC_REQUIRED[kind]


def build_src(cfg: DictConfig, workers: int = 32):
    """建立源端客户端。本地目录 / STAC 不需要客户端,返回 None。"""
    kind = src_kind(cfg)
    if kind == "local":
        root = Path(cfg.src.dir).expanduser()
        if not root.is_dir():
            raise RuntimeError(f"源目录不存在: {root}")
        return None

    if kind == "stac":
        base = Path(cfg.src.catalog).expanduser()
        if not base.is_dir():
            raise RuntimeError(f"STAC collection 目录不存在: {base}")
        return None

    return _build_src_s3(cfg, workers)


def _build_src_s3(cfg: DictConfig, workers: int = 32):
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


def load_tile_filter(path: Optional[str]) -> Optional[set]:
    """读 tile 白名单(每行一个 tile id)。None/空路径表示不过滤。

    直接吃 stac_cog_audit 写出来的 <prefix>_have_cog.txt。Q1-first 的滚动发布
    下这个是必需的:没转完的 tile,其 item href 还指向 geotiff,而那些文件是
    存在的 —— 不过滤就会把转换前的原件一起发出去,而且全程不报错。
    """
    if not path:
        return None
    tiles = load_lines(str(Path(path).expanduser()))
    if not tiles:
        raise RuntimeError(f"tile 白名单是空的或不存在: {path}")
    return tiles


def norm_patterns(pattern) -> List[str]:
    """glob 可以是单个字符串,也可以是一组模式。

    分批发布要按 RH 挑文件,而 fnmatch 表达不了"这 15 个之一"。更要命的是
    `RH0*` 会把 `RH100` 也匹配进来、`RH10*` 同样 —— 所以模式必须写成
    `RH0_Q*.tif` 这种带下划线的形式,一个 RH 一条,合起来是个列表。
    """
    if isinstance(pattern, str):
        return [pattern]
    pats = [str(p) for p in pattern]
    if not pats:
        raise RuntimeError("glob 是空列表,匹配不到任何文件")
    return pats


def _matches(name: str, patterns: List[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


def _pattern_rank(name: str, patterns: List[str]) -> int:
    """命中的第一个模式的序号。order=asset 靠它排序。

    模式列表的顺序就是发布优先级 —— priority_rh_stac 是按 RH0..RH100 写的,
    所以"先发 RH0 的全部 tile,再发 RH10 的全部 tile"直接由列表顺序决定。
    """
    for i, pat in enumerate(patterns):
        if fnmatch(name, pat):
            return i
    return len(patterns)


def _walk_local(base: Path, pattern, tiles: Optional[set]
                ) -> Generator[Tuple[str, str], None, None]:
    """惰性产出 (相对路径, 绝对路径),目录和文件都按名字排序。

    不用 Path.glob("**/...") —— `sorted(glob(...))` 必须先把整棵树走完才返回,
    调用方的 budget 早停就完全失效了。实测 original/tiles/cog 那棵树在共享盘上
    要 47 分钟才列完,而基准只需要前 25 GB。

    "**/<文件名模式>" 这种形状(也就是实际用的那种)走 os.walk + fnmatch,可以
    边走边产出,还能在顶层按 tile 白名单剪枝 —— 白名单 2593 个 tile 而树下有
    一万八千个的话,省掉的是绝大部分目录。其它形状退回 glob。
    """
    patterns = norm_patterns(pattern)
    if all(p.startswith("**/") and "/" not in p[3:] for p in patterns):
        name_pats = [p[3:] for p in patterns]
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, base)
            if tiles is not None and rel_dir == ".":
                # 顶层就是 tile 目录,不在白名单里的整棵子树都不用进
                dirnames[:] = [d for d in dirnames if d in tiles]
            for name in sorted(filenames):
                if _matches(name, name_pats):
                    rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                    yield rel.replace(os.sep, "/"), os.path.join(dirpath, name)
    else:
        seen: Dict[str, str] = {}
        for pat in patterns:
            for path in base.glob(pat):
                if path.is_file():
                    seen[path.relative_to(base).as_posix()] = str(path)
        for rel in sorted(seen):
            yield rel, seen[rel]


def list_local(root: str, pattern: str = "**/*.tif",
               budget_bytes: Optional[int] = None,
               tiles: Optional[set] = None,
               order: str = "tile") -> List[Tuple[str, int]]:
    """列举本地目录。返回 (相对路径, 字节数),相对路径直接就是目标端的 key 尾部。

    排序是为了可重复:同一个目录两次列举顺序一致,断点续传和基准采样才可比。
    tiles 非空时只保留这些 tile(取相对路径的第一段)。
    order=asset 时按模式顺序排(先某个 RH 的全部 tile,再下一个 RH)—— 本地这
    条路径必须先全部收集再排序,所以 budget 的早停要等排完才生效;树只走一遍,
    在慢文件系统上比按模式走 15 遍划算得多。
    """
    base = Path(root).expanduser()
    if not base.is_dir():
        raise RuntimeError(f"源目录不存在: {base}")

    patterns = norm_patterns(pattern)
    out: List[Tuple[str, int]] = []
    ranked: List[Tuple[int, str, int]] = []
    total = 0
    for rel, abspath in _walk_local(base, patterns, tiles):
        if tiles is not None and rel.split("/", 1)[0] not in tiles:
            continue
        try:
            size = os.stat(abspath).st_size
        except OSError:
            continue
        if size == 0:
            continue
        if order == "asset":
            ranked.append((_pattern_rank(rel, patterns), rel, size))
            continue
        out.append((rel, size))
        total += size
        if budget_bytes is not None and total >= budget_bytes:
            return out

    if order == "asset":
        for _rank, rel, size in sorted(ranked):
            out.append((rel, size))
            total += size
            if budget_bytes is not None and total >= budget_bytes:
                break
    return out


@lru_cache(maxsize=4)
def _stac_item_ids(catalog: str) -> Dict[str, str]:
    """扫一遍 collection 目录,返回 tile -> item_id(按 tile 排序插入)。

    item 目录名就是 item id,形如 <TILE>_<YEAR>(42RUR_2020),里面放着同名的
    <item_id>.json。tile 取第一个下划线之前的部分 —— 目标端的布局是
    <dst.prefix>/<tile>/<文件名>,年份已经在 dst.prefix 里了,不能再带一次。

    缓存住是因为这个目录有一万八千个条目,而 open_source 每传一个对象都要用它
    把 key 还原成绝对路径。
    """
    base = Path(catalog).expanduser()
    if not base.is_dir():
        raise RuntimeError(f"STAC collection 目录不存在: {base}")

    # scandir 而不是 iterdir + is_dir():后者对一万八千个条目要发一万八千次
    # stat,在共享文件系统上是几十秒;DirEntry 的类型信息直接来自目录项本身,
    # 基本不额外发系统调用。这一步在每次启动时都要付一遍(12h 凭证窗口意味着
    # 整个上传会重启十几次),值得省。
    with os.scandir(base) as it:
        names = sorted(e.name for e in it if e.is_dir(follow_symlinks=False))

    out: Dict[str, str] = {}
    for name in names:
        tile = name.split("_", 1)[0]
        if tile in out:
            raise RuntimeError(
                f"同一个 tile 在 collection 里出现两次: {out[tile]} / {name};"
                "目标端 key 是 <tile>/<文件名>,会互相覆盖")
        out[tile] = name
    if not out:
        raise RuntimeError(f"STAC collection 目录下没有 item: {base}")
    return out


@lru_cache(maxsize=32768)
def _stac_assets(catalog: str, item_id: str) -> Dict[str, str]:
    """一个 item 的 <文件名> -> 绝对路径。

    按文件名而不是 asset key 建索引:key 是目标端 <tile>/<文件名>,这样即使某天
    asset key 和文件名不再一一对应也不会错位。href 是 file:// URI。

    缓存要能装下整次运行涉及的所有 item —— 每个条目只是几十个 <文件名, 路径>
    的小字典,三万个也就几十 MB,而一次 miss 是一次 open+parse,在共享文件系统
    上是毫秒级。之前设成 64 有两个问题:128 个并发 worker 同时在飞的 item 数就
    可能超过它,传输全程反复重读同一批 JSON;而 print_source_roots 的均匀抽样
    更是每次必 miss。
    """
    path = Path(catalog).expanduser() / item_id / f"{item_id}.json"
    try:
        with open(path) as fh:
            item = json.load(fh)
    except OSError as exc:
        raise RuntimeError(f"读不到 STAC item {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"STAC item 不是合法 JSON {path}: {exc}")

    assets: Dict[str, str] = {}
    for asset in item.get("assets", {}).values():
        href = asset.get("href", "")
        if not href:
            continue
        if href.startswith("file://"):
            href = href[len("file://"):]
        elif "://" in href:
            raise RuntimeError(
                f"{path} 里的 asset 不是本地文件: {href}(只支持 file://)")
        assets[os.path.basename(href)] = href
    return assets


def stac_path(catalog: str, key: str) -> Path:
    """把 <tile>/<文件名> 还原成磁盘上的绝对路径。"""
    tile, _, name = key.partition("/")
    item_id = _stac_item_ids(catalog).get(tile)
    if item_id is None:
        raise RuntimeError(f"STAC collection 里没有 tile {tile}({catalog})")
    assets = _stac_assets(catalog, item_id)
    if name not in assets:
        raise RuntimeError(f"STAC item {item_id} 里没有 asset {name}")
    return Path(assets[name])


def list_stac(catalog: str, pattern: str = "*.tif",
              budget_bytes: Optional[int] = None,
              tiles: Optional[set] = None,
              order: str = "tile") -> List[Tuple[str, int]]:
    """列举 STAC collection 里的 asset。返回 (<tile>/<文件名>, 字节数)。

    pattern 是对文件名(不是整条路径)做 fnmatch,分位数分批上传就靠它:
    RH*_Q1.tif 只挑中位数那一档。

    大小只能 stat 出来 —— item JSON 里没有 file:size。缺文件不中断整次列举
    (一万八千个 item,为一个坏 href 全盘停掉不划算),但会在结尾汇总告警,
    漏掉的东西必须让人看见。
    """
    patterns = norm_patterns(pattern)
    bad = [p for p in patterns if "/" in p]
    if bad:
        # kind=local 的默认值是 **/*.tif,直接拿来 fnmatch 文件名会一个都匹配
        # 不上,然后当成"源端是空的"退出 —— 这种静默的空跑必须拦住。
        raise RuntimeError(
            f"kind=stac 的 glob 匹配的是 asset 文件名,不能带 '/':{bad!r}"
            "(整档用 '*.tif',单档用 'RH98_Q1.tif',多档用列表)")

    # order=asset:一个模式扫一轮全部 tile,再下一个模式 —— 即"先发 RH0 的所有
    # tile,再发 RH10 的所有 tile"。item JSON 已经在 _stac_assets 里缓存住了,
    # 多扫几轮只是内存里的字典查找,不会重复读盘。
    groups = ([[p] for p in patterns] if order == "asset" else [patterns])

    out: List[Tuple[str, int]] = []
    total = 0
    missing: List[str] = []
    seen: set = set()
    for group in groups:
        for tile, item_id in _stac_item_ids(catalog).items():
            if tiles is not None and tile not in tiles:
                continue
            for name, path in sorted(_stac_assets(catalog, item_id).items()):
                if not _matches(name, group):
                    continue
                key = f"{tile}/{name}"
                if key in seen:      # 模式重叠时不能传两遍
                    continue
                try:
                    size = os.stat(path).st_size
                except OSError:
                    missing.append(path)
                    continue
                if size == 0:
                    missing.append(path)
                    continue
                seen.add(key)
                out.append((key, size))
                total += size
                if budget_bytes is not None and total >= budget_bytes:
                    return out

    if missing:
        print(f"[warn] STAC 里有 {len(missing)} 个 asset 在磁盘上缺失或为空,"
              f"已跳过。前 3 个: {missing[:3]}", file=sys.stderr)
    return out


def list_source_items(cfg: DictConfig, src,
                      budget_bytes: Optional[int] = None
                      ) -> List[Tuple[str, int]]:
    """按源端类型列举。上层不用关心是本地盘、STAC 还是 LUMI-O。"""
    kind = src_kind(cfg)
    tiles = load_tile_filter(cfg.src.get("tiles")) if "src" in cfg else None
    if tiles is not None:
        print(f"[info] tile 白名单: {len(tiles)} 个 ({cfg.src.tiles})", flush=True)

    # 列举可能要跑几十秒(STAC 要扫一万八千个 item 目录,整桶 list 要几千次
    # API 调用),不报时的话看着就是卡住了。
    t0 = time.time()
    order = cfg.src.get("order", "tile") if "src" in cfg else "tile"
    if order not in ("tile", "asset"):
        raise RuntimeError(f"src.order 只能是 tile / asset,收到 {order!r}")
    if kind == "local":
        items = list_local(cfg.src.dir, cfg.src.get("glob", "**/*.tif"),
                           budget_bytes, tiles, order)
    elif kind == "stac":
        items = list_stac(cfg.src.catalog, cfg.src.get("glob", "*.tif"),
                          budget_bytes, tiles, order)
    else:
        items = list_source(src, cfg.src.bucket, cfg.src.prefix, budget_bytes)

    elapsed = time.time() - t0
    if elapsed > 1:
        print(f"[info] 列举 {len(items)} 个对象耗时 {elapsed:.1f}s "
              f"({kind})", flush=True)
    return items


def source_label(cfg: DictConfig) -> str:
    """给日志和报错用的源端描述。"""
    kind = src_kind(cfg)
    if kind == "local":
        return f"{cfg.src.dir}/{cfg.src.get('glob', '**/*.tif')}"
    if kind == "stac":
        pats = norm_patterns(cfg.src.get("glob", "*.tif"))
        shown = pats[0] if len(pats) == 1 else f"{len(pats)} 个模式"
        return f"{cfg.src.catalog}[{shown}]"
    return f"s3://{cfg.src.bucket}/{cfg.src.prefix}"


def source_root_summary(cfg: DictConfig, keys: List[Tuple[str, int]],
                        sample: int = 200) -> List[Tuple[str, int, float]]:
    """按产品 root 统计将要读取的文件,返回 [(root, 个数, 平均MB)]。

    2020 的 item 分散在 original/tiles/cog、original/tiles/geotiff、
    masked/tiles/geotiff 三个 root 下,retarget 之后才会指向 masked/tiles/cog。
    发布前必须能一眼看出读的是哪一份 —— 传未压缩的原件不会报任何错,只会在
    对方那里多占几倍空间,而且事后很难分辨。

    抽样统计:解析每个 key 都要读一次 item JSON,几十万个对象全解一遍不值当。
    """
    kind = src_kind(cfg)
    if kind == "s3" or not keys:
        return []

    step = max(1, len(keys) // sample)
    picked = keys[::step][:sample]

    buckets: Dict[str, List[int]] = {}
    for key, size in picked:
        if kind == "stac":
            path = stac_path(cfg.src.catalog, key)
        else:
            path = Path(cfg.src.dir).expanduser() / key
        # .../<root>/<tile>/<文件名> —— 往上两级就是产品 root
        buckets.setdefault(str(path.parent.parent), []).append(size)

    return sorted(
        ((root, len(v), sum(v) / len(v) / 1e6) for root, v in buckets.items()),
        key=lambda r: -r[1])


def print_source_roots(cfg: DictConfig, keys: List[Tuple[str, int]]) -> None:
    """把 source_root_summary 打出来,并对疑似未转换的 root 给出告警。

    cfg.check_source_roots=false 可以关掉。发布前(mode=dry / 实传)不该关 ——
    传错副本不报任何错;基准测试里它只是个便利,嫌慢就关。
    """
    if not cfg.get("check_source_roots", True):
        return
    t0 = time.time()
    rows = source_root_summary(cfg, keys)
    if not rows:
        return
    elapsed = time.time() - t0
    if elapsed > 1:
        print(f"[info] root 抽查耗时 {elapsed:.1f}s"
              "(嫌慢: run.check_source_roots=false)")
    total = sum(n for _, n, _ in rows)
    print(f"[info] 源文件所在的 root(抽样 {total} 个):")
    for root, n, mb in rows:
        flag = "  <- 不是 COG?" if "/cog" not in root else ""
        print(f"         {n:6d} 个  平均 {mb:7.1f} MB  {root}{flag}")
    if any("/cog" not in root for root, _, _ in rows):
        print("[warn] 有文件不在 cog/ 下 —— 先跑 tools.run run=stac_retarget "
              "把 item href 指过去,否则发布的是转换前的原件", file=sys.stderr)


@contextmanager
def open_source(cfg: DictConfig, src,
                key: str) -> Generator[Tuple[Any, int], None, None]:
    """打开一个源端对象,yield (可读文件对象, botocore 重试次数)。

    local 给的是真实文件句柄(可 seek,s3transfer 走更省事的分片路径);
    s3 给的是 get_object 的 StreamingBody(不可 seek)。两边都必须关闭 ——
    StreamingBody 不关会把连接池占住,几十万个对象之后就挂死。
    """
    kind = src_kind(cfg)
    if kind in ("local", "stac"):
        path = (Path(cfg.src.dir).expanduser() / key if kind == "local"
                else stac_path(cfg.src.catalog, key))
        fh = open(path, "rb")
        try:
            yield fh, 0
        finally:
            fh.close()
    else:
        got = src.get_object(Bucket=cfg.src.bucket, Key=key)
        body = got["Body"]
        try:
            yield body, retry_count(got)
        finally:
            body.close()


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
    "norm_patterns", "list_local", "list_source", "list_source_items", "list_stac", "load_lines",
    "load_tile_filter",
    "map_key", "normalize_cfg_prefixes", "normalize_prefix", "open_source",
    "print_source_roots", "retry_count", "source_label", "source_root_summary",
    "src_kind", "src_required", "stac_path", "transfer_config",
]
