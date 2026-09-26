"""Audit the supplied source against a fixed GitHub tree; never load models."""
import hashlib
import importlib.metadata
import json
import platform
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'RL-MPC-LaneMerging-master'
REVISION = 'd60087f511e7a551e27c4e9fdd87f87f8a84b406'


def main():
    url = f'https://api.github.com/repos/jlubars/RL-MPC-LaneMerging/git/trees/{REVISION}?recursive=1'
    request = urllib.request.Request(url, headers={'User-Agent': 'Prediction-RL-Provenance'})
    with urllib.request.urlopen(request, timeout=30) as response:
        tree = json.load(response)
    if tree.get('truncated') or tree['sha'] != REVISION:
        raise RuntimeError('Unverified or truncated source tree')
    rows = []
    for item in tree['tree']:
        if item['type'] != 'blob':
            continue
        path = SOURCE / item['path']
        raw = path.read_bytes() if path.is_file() else None
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() if raw is not None else None
        rows.append({'path': item['path'], 'upstream_git_blob': item['sha'],
                     'local_git_blob': blob, 'matches': blob == item['sha'],
                     'sha256': hashlib.sha256(raw).hexdigest() if raw is not None else None})
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    report = {'upstream_url': 'https://github.com/jlubars/RL-MPC-LaneMerging',
              'upstream_commit': REVISION, 'upstream_branch': 'master',
              'audited_at': datetime.now(timezone.utc).isoformat(),
              'source_kind': 'user supplied ZIP, no upstream local git worktree',
              'license_status': 'unresolved; no LICENSE/NOTICE in pinned tree; no redistribution',
              'all_upstream_blobs_match': all(r['matches'] for r in rows),
              'python': sys.version, 'python_executable': sys.executable,
              'os': platform.platform(), 'packages': packages, 'files': rows,
              'git_snapshot_excludes': ['weights', 'experiment_data', 'generated outputs']}
    target = ROOT / 'provenance' / 'upstream_snapshot.json'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError('Refusing to overwrite source provenance')
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(target), 'blobs': len(rows), 'matched': sum(r['matches'] for r in rows),
                      'mismatches': [r['path'] for r in rows if not r['matches']]}))
    if not report['all_upstream_blobs_match']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
