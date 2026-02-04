# lumio_io.py
import os
import re
import time
import subprocess
from pathlib import Path


DIFF_PATTERN = r": (\d+) differences found"

def parse_rclone_check_differences(log_output: str) -> int:
    m = re.search(DIFF_PATTERN, log_output)
    if not m:
        raise ValueError("Could not find the summary line in rclone output.")
    return int(m.group(1))


def rclone_lumio(remote: str, local: Path, mode: str = 'from', *, transfers=16, checkers=16, multi_thread_streams=4) -> None:
    if mode == 'from':
        local.mkdir(parents=True, exist_ok=True)
        os.system(
            f"rclone copy {remote} {local} "
            f"--transfers={transfers} --checkers={checkers} --multi-thread-streams={multi_thread_streams}"
        )
    elif mode == 'to':
        os.system(
            f"rclone sync {local} {remote} "
            f"--transfers={transfers} --checkers={checkers} --multi-thread-streams={multi_thread_streams}"
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # check if the local files are exactly the same as the remote files
    difference_count = rclone_check(remote, local)
    if difference_count > 0:
        print(f"🚨 FAILURE: Found {difference_count} differences/corruptions.")
        rclone_lumio(remote, local, mode=mode, transfers=transfers, checkers=checkers, multi_thread_streams=multi_thread_streams)
    else:
        print("✨ SUCCESS: All files verified as identical and uncorrupted.")
    


def rclone_check(remote: str, local: Path, *, checkers=16) -> int:
    cmd = ["rclone", "check", "--checkers=16", "--checksum", "--stats-one-line", remote, str(local)]
    task = subprocess.run(cmd, capture_output=True, text=True)
    if task.returncode != 0:
        raise ValueError(f"rclone check failed for remote={remote} local={local}")
    log_output = task.stdout + task.stderr
    return parse_rclone_check_differences(log_output)


def make_lumio_remote_in(tile_id: str, year: int, *, project_id=465001846) -> str:
    zone_name = tile_id[:3].lower()
    bucket_name = f"{zone_name}-{year}"
    return f"lumi-{project_id}-private:{bucket_name}/predictions_GTiff_{year}/{tile_id}"


def make_lumio_remote_out(tile_id: str, year: int, *, project_id=465001846) -> str:
    zone_name = tile_id[:3].lower()
    bucket_name = f"{zone_name}-{year}"
    return f"lumi-{project_id}-private:{bucket_name}/{tile_id}"