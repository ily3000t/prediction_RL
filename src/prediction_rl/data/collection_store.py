"""Immutable local receipts, verified reuse and a single-writer run lock."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, payload):
    """Publish complete JSON without replacing an existing artifact (NTFS/POSIX)."""
    path = Path(path)
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.t', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-filesystem atomic publication; unlike replace(), never overwrites.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(directory):
    path = Path(directory) / 'writer.lock'
    token = uuid.uuid4().hex
    write_once(path, {'pid': os.getpid(), 'token': token})
    try:
        yield
    finally:
        # A crashed process leaves a visible lock; never infer that it is stale.
        if read_json(path).get('token') != token:
            raise RuntimeError('Run lock ownership changed; refusing cleanup')
        path.unlink()


def inventory(directory):
    directory = Path(directory).resolve()
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError('Artifact symlink/path escape')
        if path.is_file() and path != directory / 'complete.json':
            result[path.relative_to(directory).as_posix()] = file_hash(path)
    return result


def seal_episode(directory, binding, metadata):
    files = inventory(directory)
    if not files or 'failure.json' in files or 'started.json' not in files:
        raise ValueError('Cannot seal empty, unstarted or failed episode')
    receipt = {'schema_version': 1, 'status': 'complete', 'binding': binding,
               'files': files, 'metadata': metadata}
    write_once(Path(directory) / 'complete.json', receipt)
    return receipt


def verify_episode(directory, binding):
    directory = Path(directory)
    if not (directory / 'complete.json').is_file():
        raise ValueError('Incomplete/failed episode preserved; automatic retry is disabled')
    receipt = read_json(directory / 'complete.json')
    if (set(receipt) != {'schema_version', 'status', 'binding', 'files', 'metadata'}
            or receipt['schema_version'] != 1 or receipt['status'] != 'complete'
            or receipt['binding'] != binding):
        raise ValueError('Episode receipt/request mismatch')
    if not receipt['files'] or receipt['files'] != inventory(directory):
        raise ValueError('Episode artifacts changed, missing or unexpectedly added')
    if 'failure.json' in receipt['files'] or 'started.json' not in receipt['files']:
        raise ValueError('Invalid completed episode')
    return receipt
