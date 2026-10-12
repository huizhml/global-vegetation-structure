"""Per-key task state ("done flags") without one empty file per key.

Replaces the `{tile}_done` pattern. A state lives in one directory, kept
under the folder of the task it tracks, e.g. ~/data/gvs/sync/copy_from_lumio/2024_q1:

    <path>/log/<host>_<pid>_<job>.tsv   append-only, one file per writer process
    <path>/snapshot.tsv                 merged state, rewritten by compact()
    <path>/offsets.tsv                  how far compact() has read each log
    <path>/expected.tsv                 optional: every key the task should end
                                        up with (set_expected), so state_summary
                                        can count keys that were never started

Every writer process appends only to its own log file, so any number of SLURM
tasks on any number of nodes can write at the same time with no locking. That
is what makes it safe on NFS / Lustre, where SQLite's file locks are not
reliable across nodes. The file count grows with the number of processes that
ever wrote, not with the number of tiles.

A line is `key \t status \t time \t info`. The newest line for a key wins,
so `reset` undoes a `done` and a retry that succeeds overrides a `failed`.
time is UTC in fixed-width ISO 8601 (2026-10-09T12:31:47.570972Z): readable,
and string order is time order, which local time is not across a DST change.
Lines from before that format carry Unix seconds; they still read, and
compact() rewrites them.

compact() never moves or truncates a log a writer may still append to. It
records how far it has read each log (by inode and byte offset) in
offsets.tsv, so readers only scan what was written since, and it deletes a log
only after it has been idle for `idle_s` and is fully merged.

    state = TaskState('~/data/gvs/blended/2020/state')
    todo = [t for t in tiles if t not in state.done()]
    for tile in todo:
        try:
            process(tile)
            state.mark_done(tile)
        except Exception as e:
            state.mark_failed(tile, repr(e))

Bash can append in the same format (one line per write, to its own file):
    printf '%s\tdone\t%s\t\n' "$tile" "$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)" \
        >> "$STATE/log/$(hostname -s)_$$_${SLURM_JOB_ID:-local}.tsv"
"""
import os
import re
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DONE, FAILED, RESET = 'done', 'failed', 'reset'
_STATUSES = {DONE, FAILED, RESET}
_OFFSET = '#offset'     # offsets line: #offset \t log name \t inode \t bytes read
_HEADER = '#key\tstatus\ttime (UTC)\tinfo\n'
# Must not start with _OFFSET; _parse skips it (second field is not a status).
_OFFSET_HEADER = '#type\tlog file\tinode\tbytes merged into snapshot\n'
_ISO = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z')


def _iso(t: float = None) -> str:
    """Unix seconds (default: now) -> '2026-10-09T12:31:47.570972Z'."""
    dt = datetime.now(timezone.utc) if t is None else datetime.fromtimestamp(t, timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _clean(text) -> str:
    # tabs/newlines would split the line into extra fields
    return ' '.join(str(text).split())[:500]


def _parse(line: str, state: dict):
    parts = line.rstrip('\n').split('\t')
    # skips a line cut short by a crash mid-write
    if len(parts) != 4 or parts[1] not in _STATUSES:
        return
    t = parts[2]
    if not _ISO.fullmatch(t):
        try:                        # older lines: Unix seconds
            t = _iso(float(t))
        except ValueError:
            return
    key = parts[0]
    if key not in state or t >= state[key][1]:
        state[key] = (parts[1], t, parts[3])


class TaskState:

    def __init__(self, path, create: bool = True):
        """create=False for read-only use: a missing state is an error (most
        likely a wrong path), not an empty state to be made on the spot."""
        self.path = Path(path).expanduser()
        self.log_dir = self.path / 'log'
        if create:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_dir():
            raise FileNotFoundError(f'no task state at {self.path}')
        self.snapshot = self.path / 'snapshot.tsv'
        self.expected_file = self.path / 'expected.tsv'
        self._lock = threading.Lock()
        self._cache = None

    # Picklable, so it can ride along to dask / multiprocessing workers. A lock
    # cannot be pickled, and a copied cache would go stale in the worker.
    def __getstate__(self):
        return {'path': self.path}

    def __setstate__(self, d):
        self.__init__(d['path'])

    # ------------------------------------------------------------------ write
    def _log_file(self) -> Path:
        # Resolved per write, so a forked worker gets its own file instead of
        # sharing the parent's.
        job = os.environ.get('SLURM_JOB_ID', 'local')
        task = os.environ.get('SLURM_PROCID')
        if task is not None:
            job = f'{job}.{task}'
        return self.log_dir / f'{socket.gethostname().split(".")[0]}_{os.getpid()}_{job}.tsv'

    def _write(self, rows):
        lines = []
        for key, status, t, info in rows:
            if not key or key.startswith('#') or any(c in key for c in '\t\n\r'):
                raise ValueError(f'invalid key: {key!r}')
            lines.append(f'{key}\t{status}\t{t}\t{_clean(info)}\n')
        # Open-append-close per call, so a log is never held open and an idle
        # one can be deleted by compact(); the next write recreates it.
        with self._lock, open(self._log_file(), 'a') as fh:
            fh.write(''.join(lines))
        if self._cache is not None:
            for key, status, t, info in rows:
                self._cache[key] = (status, t, _clean(info))

    def mark_done(self, key: str, info: str = ''):
        self._write([(key, DONE, _iso(), info)])

    def mark_failed(self, key: str, reason: str = ''):
        self._write([(key, FAILED, _iso(), reason)])

    def reset(self, key: str):
        """Forget a key, i.e. what `rm {tile}_done` used to do."""
        self._write([(key, RESET, _iso(), '')])

    def mark_many(self, keys, status: str = DONE, info: str = '', times=None):
        """Write many keys in one append (used by import_flags).

        times: Unix seconds per key, default now.
        """
        assert status in _STATUSES, status
        now = _iso()
        stamps = [_iso(t) for t in times] if times else [now] * len(keys)
        self._write([(k, status, t, info) for k, t in zip(keys, stamps)])

    def set_expected(self, keys, source: str = ''):
        """Record every key the task should end up with (rewritten each call).

        A key that is in neither done() nor failed() leaves no trace, e.g. when
        the job was killed while it ran; only this list shows it is missing.
        """
        keys = sorted(set(keys))
        header = f'# {len(keys)} expected keys, from {source or "?"}, written {_iso()}\n'
        self._replace(self.expected_file, [header] + [f'{k}\n' for k in keys])

    def expected(self):
        """Keys from set_expected, or None if it was never called."""
        try:
            with open(self.expected_file) as fh:
                return {line.rstrip('\n') for line in fh if not line.startswith('#')}
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------- read
    def _read_snapshot(self):
        # Offsets first: compact() replaces snapshot.tsv before offsets.tsv,
        # so offsets read here are never ahead of the snapshot read after
        # them, at worst behind it, which only re-reads a few lines.
        state, offsets = {}, {}
        # snapshot.tsv held the offsets before offsets.tsv existed
        for f in (self.path / 'offsets.tsv', self.snapshot):
            try:
                with open(f) as fh:
                    for line in fh:
                        if line.startswith(_OFFSET):
                            _, name, ino, off = line.rstrip('\n').split('\t')
                            offsets[name] = (int(ino), int(off))
                        else:
                            _parse(line, state)
            except FileNotFoundError:
                pass
        return state, offsets

    @staticmethod
    def _read_log(path: Path, offsets: dict, complete_only: bool):
        """Bytes of `path` after its recorded offset -> (text, inode, new offset).

        complete_only keeps a trailing partial line (a write in flight) for
        the next pass instead of consuming it.
        """
        with open(path, 'rb') as fh:
            ino = os.fstat(fh.fileno()).st_ino
            seen_ino, start = offsets.get(path.name, (None, 0))
            if seen_ino != ino:         # deleted and recreated since: from 0
                start = 0
            fh.seek(start)
            data = fh.read()
        if complete_only:
            data = data[:data.rfind(b'\n') + 1]
        return data.decode(errors='replace'), ino, start + len(data)

    def load(self) -> dict:
        """{key: (status, time, info)} for every key ever written, newest wins."""
        for _ in range(20):
            state, offsets = self._read_snapshot()
            vanished = False
            for f in sorted(self.log_dir.glob('*.tsv')):
                try:
                    text, _, _ = self._read_log(f, offsets, complete_only=False)
                except FileNotFoundError:
                    # deleted by compact() after we read the snapshot
                    vanished = True
                    continue
                for line in text.splitlines():
                    _parse(line, state)
            if not vanished:
                self._cache = state
                return state
            time.sleep(0.5)
        raise RuntimeError(f'{self.path}: logs kept disappearing during read')

    def _state(self, refresh: bool) -> dict:
        if refresh or self._cache is None:
            return self.load()
        return self._cache

    def done(self, refresh: bool = True) -> set:
        return {k for k, (s, _, _) in self._state(refresh).items() if s == DONE}

    def failed(self, refresh: bool = True) -> dict:
        """{key: reason} for keys whose latest status is failed."""
        return {k: info for k, (s, _, info) in self._state(refresh).items() if s == FAILED}

    def is_done(self, key: str, refresh: bool = False) -> bool:
        """Cached after the first call; refresh=True to see other jobs' writes."""
        entry = self._state(refresh).get(key)
        return entry is not None and entry[0] == DONE

    # ---------------------------------------------------------------- compact
    def compact(self, idle_s: float = 24 * 3600) -> int:
        """Merge new log lines into snapshot.tsv. Safe while writers run.

        Logs idle for longer than `idle_s` are deleted once fully merged.
        Returns the number of log files deleted.
        """
        # mkdir is atomic on NFS too, so it serves as the compaction lock.
        lock = self.path / 'compact.lock'
        try:
            lock.mkdir()
        except FileExistsError:
            raise RuntimeError(f'{lock} exists: another compact() is running, '
                               'or one crashed (then remove it by hand)')
        try:
            state, offsets = self._read_snapshot()
            new_offsets, idle = {}, []
            now = time.time()
            for f in sorted(self.log_dir.glob('*.tsv')):
                try:
                    st = f.stat()
                    text, ino, off = self._read_log(f, offsets, complete_only=True)
                except FileNotFoundError:
                    continue
                for line in text.splitlines():
                    _parse(line, state)
                if now - st.st_mtime > idle_s:
                    # whatever is left after `off` has no newline: a write cut
                    # off by a crash, nothing to keep
                    idle.append((f, ino))
                else:
                    new_offsets[f.name] = (ino, off)

            rows = [_HEADER]
            for key in sorted(state):
                status, t, info = state[key]
                if status != RESET:     # nothing left to remember
                    rows.append(f'{key}\t{status}\t{t}\t{info}\n')
            self._replace(self.snapshot, rows)
            # After the snapshot: offsets never claim lines it does not hold.
            self._replace(self.path / 'offsets.tsv',
                          [_OFFSET_HEADER] +
                          [f'{_OFFSET}\t{name}\t{ino}\t{off}\n'
                           for name, (ino, off) in sorted(new_offsets.items())])

            # Only after the snapshot holds their lines. Re-check right before
            # unlinking in case a writer woke up after the idle check.
            n_deleted = 0
            for f, ino in idle:
                try:
                    st = f.stat()
                except FileNotFoundError:
                    continue
                if st.st_ino == ino and now - st.st_mtime > idle_s:
                    f.unlink(missing_ok=True)
                    n_deleted += 1
            self._cache = None
            return n_deleted
        finally:
            lock.rmdir()

    def _replace(self, path: Path, lines):
        """Write a file atomically: readers see the old or the new one, whole."""
        tmp = self.path / f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp'
        with open(tmp, 'w') as fh:
            fh.write(''.join(lines))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)


# ======================================================== tools.run ops
def state_summary(path: str, **kwargs):
    st = TaskState(path, create=False)
    counts = {}
    for status, _, _ in st.load().values():
        counts[status] = counts.get(status, 0) + 1
    n_logs = len(list(st.log_dir.glob('*.tsv')))
    print(f'{st.path}: {counts}  ({n_logs} log files)')
    failed = st.failed(refresh=False)
    for key in sorted(failed)[:20]:
        print(f'  failed  {key}  {failed[key]}')
    if len(failed) > 20:
        print(f'  ... {len(failed) - 20} more failed')

    expected = st.expected()
    if expected is None:
        print('  no expected.tsv: total unknown, cannot tell whether keys are missing')
        return
    done = st.done(refresh=False)
    not_started = sorted(expected - done - failed.keys())
    print(f'  expected {len(expected)}: {len(done & expected)} done, '
          f'{len(failed.keys() & expected)} failed, {len(not_started)} never started'
          f'  -> {"COMPLETE" if len(done & expected) == len(expected) else "INCOMPLETE"}')
    for key in not_started[:20]:
        print(f'  never started  {key}')
    if len(not_started) > 20:
        print(f'  ... {len(not_started) - 20} more never started')
    extra = done - expected
    if extra:
        print(f'  {len(extra)} done keys not in expected.tsv, e.g. {sorted(extra)[:5]}')


def compact_state(path: str, idle_hours: float = 24, **kwargs):
    st = TaskState(path, create=False)
    n = st.compact(idle_s=idle_hours * 3600)
    print(f'compacted {st.snapshot}, deleted {n} idle log files')


def import_flags(flag_dir: str, path: str, suffix: str = '_done', delete: bool = False,
                 **kwargs):
    """Move `{key}{suffix}` flag files into a TaskState.

    The flag's mtime becomes the done time. With delete=True the flag files are
    removed, but only after every key reads back as done.
    """
    flag_dir = Path(flag_dir).expanduser()
    keys, times, files = [], [], []
    with os.scandir(flag_dir) as it:
        for e in it:
            if e.is_file() and e.name.endswith(suffix):
                keys.append(e.name[:-len(suffix)])
                times.append(e.stat().st_mtime)
                files.append(e.path)
    state = TaskState(path)
    if keys:
        state.mark_many(keys, DONE, info='imported', times=times)
        state.compact()
    print(f'imported {len(keys)} flags from {flag_dir} into {state.path}')
    if delete:
        missing = set(keys) - state.done()
        if missing:
            raise RuntimeError(f'{len(missing)} keys not in state after import, '
                               f'nothing deleted, e.g. {sorted(missing)[:5]}')
        for f in files:
            os.remove(f)
        print(f'deleted {len(files)} flag files')
