from __future__ import annotations
import hashlib
import json
import uuid as uuidlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence


CHUNK_SIZE = 1024 * 1024  # 1 MiB


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mtime_utc_iso(p: Path) -> str:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path, include_hidden: bool = False) -> Iterable[Path]:
    # Deterministic traversal by sorting paths
    all_paths = sorted([p for p in root.rglob("*") if p.is_file()])
    for p in all_paths:
        if not include_hidden:
            # skip any file inside a hidden directory or hidden file
            parts = p.relative_to(root).parts
            if any(part.startswith(".") for part in parts):
                continue
        yield p


def to_posix_relpath(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def canonical_json_bytes(obj: object) -> bytes:
    # Canonical-ish JSON: sorted keys, no whitespace.
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def compute_content_hash(files: Sequence[dict]) -> str:
    """
    Content hash is based on:
      - relative path (posix)
      - file sha256
    in deterministic order.
    """
    h = hashlib.sha256()
    for rec in sorted(files, key=lambda r: r["path"]):
        # Avoid ambiguity: include lengths + separators
        path_b = rec["path"].encode("utf-8")
        sha_b = rec["sha256"].encode("ascii")
        h.update(len(path_b).to_bytes(8, "big"))
        h.update(path_b)
        h.update(len(sha_b).to_bytes(8, "big"))
        h.update(sha_b)
    return h.hexdigest()


def load_or_create_uuid(manifest_path: Path) -> str:
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            u = data.get("dataset_uuid")
            if isinstance(u, str) and u:
                return u
        except Exception:
            pass
    return str(uuidlib.uuid4())


def make_manifest(
    data_dir: str,
    dataset_name: str,
    include_hidden: bool = False,
    root_note: Optional[str] = None,
    **kwargs: Any
) -> dict:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset root directory not found or not a directory: {data_dir}")

    manifest_path = data_dir.parent / f'manifest_{data_dir.stem}.json'
    dataset_uuid = load_or_create_uuid(manifest_path)

    files = []
    total_size = 0

    for f in iter_files(data_dir, include_hidden=include_hidden):
        rel = to_posix_relpath(data_dir, f)
        size = f.stat().st_size
        total_size += size
        rec = {
            "path": rel,
            "size_bytes": size,
            "mtime_utc": mtime_utc_iso(f),
            "sha256": sha256_file(f),
        }
        files.append(rec)

    content_hash = compute_content_hash(files)

    manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_uuid": dataset_uuid,
        "created_utc": utc_now_iso(),
        "root_note": root_note or "",
        "files": sorted(files, key=lambda r: r["path"]),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "content_hash_sha256": content_hash,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


