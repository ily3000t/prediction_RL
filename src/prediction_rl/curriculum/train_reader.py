"""Explicit read-only train receipt traversal. No simulator/model imports."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from prediction_rl.data.dataset_contract import digest, episode_id
from prediction_rl.data.merge_geometry import MergeGeometry

BASELINE_COMMIT = '75051acc06f6ea2ab6dda9be678f191b63d04569'


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON field: '+key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON literal: '+value)
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=number)


def inside(root, relative):
    relative = Path(str(relative).replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Absolute/parent traversal in artifact reference')
    path = (Path(root).resolve()/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact reference escaped read-only root')
    return path


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, encoding='utf-8').strip()


def validate_config(c):
    keys = {'schema_version', 'protocol_id', 'scope', 'baseline_references', 'critical_roles',
            'sensor_radius_m', 'conflict_horizon_s', 'tick_s', 'horizon_steps', 'probe_jerks',
            'minimum_common_steps', 'minimum_supported_pair_fraction', 'lower_quantile',
            'upper_quantile', 'scale_quantile', 'response_epsilon', 'C1_maximum_critical_actors',
            'minimum_layer_episodes', 'classification', 'training_ready'}
    if not isinstance(c, dict) or set(c) != keys:
        raise ValueError('Unknown/missing IC01 configuration fields')
    if c['scope'] != 'train':
        raise ValueError('IC01 cannot open non-train split')
    if (c['schema_version'] != 1 or c['protocol_id'] != 'ic01-train-neighbor-interaction-v1'
            or c['training_ready'] is not False
            or c['classification'] != 'provisional_root_only_not_training_release'):
        raise ValueError('IC01 is an offline design audit, never a training release')
    numeric = ('sensor_radius_m', 'conflict_horizon_s', 'tick_s', 'minimum_supported_pair_fraction',
               'lower_quantile', 'upper_quantile', 'scale_quantile', 'response_epsilon')
    import math
    if any(isinstance(c[k], bool) or not isinstance(c[k], (int, float)) or not math.isfinite(c[k]) for k in numeric):
        raise ValueError('Need finite numeric IC01 fields')
    if (c['tick_s'] != .2 or c['horizon_steps'] != 25 or c['probe_jerks'] != [-5, -2.5, 0, 2.5, 5]
            or any(isinstance(x, bool) for x in c['probe_jerks'])
            or c['conflict_horizon_s'] != 5 or c['sensor_radius_m'] != 125):
        raise ValueError('Keep original probes, horizon and sensor contract')
    if c['critical_roles'] != ['target_front', 'target_rear', 'nearest_cross_path_arrival', 'lowest_same_path_ttc']:
        raise ValueError('Critical roles must remain result-independent root roles')
    for key, maximum in (('minimum_common_steps', 25), ('C1_maximum_critical_actors', 4), ('minimum_layer_episodes', 256)):
        if type(c[key]) is not int or not 1 <= c[key] <= maximum:
            raise ValueError('Invalid positive integer '+key)
    if (not 0 < c['lower_quantile'] < c['upper_quantile'] < 1
            or not 0 < c['scale_quantile'] < 1 or not 0 < c['minimum_supported_pair_fraction'] <= 1
            or not 0 < c['response_epsilon'] < 1):
        raise ValueError('Invalid quantile/support rule')
    return c


class TrainReader:
    def __init__(self, worktree, baseline, config):
        self.worktree = Path(worktree).resolve(); self.baseline = Path(baseline).resolve()
        self.config = validate_config(config); self.hashes = {}; self.files_read = 0
        refs_path = inside(self.worktree, config['baseline_references'])
        self.references = strict_json(refs_path.read_bytes())
        if (self.references['baseline_commit'] != BASELINE_COMMIT
                or Path(self.references['baseline_repository']).resolve() != self.baseline
                or Path(self.references['worktree']).resolve() != self.worktree
                or self.worktree == self.baseline):
            raise ValueError('Need the isolated pinned curriculum worktree and baseline')
        if git(self.baseline, 'rev-parse', 'HEAD') != BASELINE_COMMIT or git(self.baseline, 'status', '--porcelain'):
            raise ValueError('Validated baseline HEAD/clean state changed')
        self.verify_references()
        review_ref = next(r for r in self.references['references'] if r['role'] == 'predictor_dataset_review')
        self.review = self.read(inside(self.baseline, review_ref['relative_path']), review_ref['sha256'])
        if self.review['status'] != 'complete' or self.review['test_locked'] is not True:
            raise ValueError('Need complete immutable dataset review with locked test')
        self.review_hashes = {}
        for name, value in self.review['input_hashes'].items():
            key = str(name).replace('\\', '/')
            if key in self.review_hashes and self.review_hashes[key] != value:
                raise ValueError('Conflicting normalized review paths')
            self.review_hashes[key] = value
        request_name = 'artifacts/data/response_v1/request.json'
        self.request = self.read_reviewed(request_name)
        self.request_hash = digest(self.request)
        if (self.review['collection_request_hash'] != self.request_hash
                or self.request['test_locked'] is not True
                or self.request['protocol'] != self.review['protocol']):
            raise ValueError('Collection/review protocol binding changed')
        report_names = [name for name in self.review_hashes if name.startswith('artifacts/data/response_v1/invocations/')
                        and name.endswith('/report.json')]
        if len(report_names) != 1:
            raise ValueError('Ambiguous collection report')
        report = self.read_reviewed(report_names[0])
        preparation = self.read_reviewed('artifacts/data/response_v1/preparation.json')
        if (report['status'] != 'complete' or report['working_tree_dirty'] is not False
                or report['git_commit'] != self.review['collection_commit']
                or report['request_hash'] != self.request_hash or report['test_locked'] is not True
                or preparation['request_hash'] != self.request_hash):
            raise ValueError('Collection provenance invalid')
        collection = self.request['protocol']['collection']
        if (collection['branch_repeats'] != 2 or collection['future_metadata'] != 'lane_position_length_width_v1'
                or collection['probe_jerks'] != config['probe_jerks'] or collection['tick_s'] != config['tick_s']
                or collection['horizon_steps'] != config['horizon_steps'] or collection['continuation_jerk'] != 0):
            raise ValueError('Unsupported collection branch contract')
        self.jobs = [j for j in self.request['plan']['jobs'] if j['split'] == 'train']
        manifest = self.request['plan']['split_manifest']
        self.episodes = [j['episode_id'] for j in self.jobs]
        if (self.review['manifest'] != manifest or self.episodes != manifest['train']
                or len(set(self.episodes)) != len(self.episodes)):
            raise ValueError('Train episode roster changed')
        network = self.request['network']
        key = network.replace('\\', '/')
        expected = self.request['input_hashes'][network]
        path = inside(self.baseline, key)
        self.bytes(path, expected)
        self.geometry = MergeGeometry(path)
        self.unavailable = []; self.roster = []

    def verify_references(self):
        for ref in self.references['references']:
            self.bytes(inside(self.baseline, ref['relative_path']), ref['sha256'])

    def bytes(self, path, expected):
        path = Path(path).resolve()
        if not path.is_relative_to(self.baseline):
            raise ValueError('Input escaped baseline; no external asset search')
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError('Immutable input SHA256 mismatch: '+str(path))
        self.hashes[path.relative_to(self.baseline).as_posix()] = actual
        self.files_read += 1
        return raw

    def read(self, path, expected):
        return strict_json(self.bytes(path, expected))

    def read_reviewed(self, name):
        if name not in self.review_hashes:
            raise ValueError('Input not bound by dataset review: '+name)
        return self.read(inside(self.baseline, name), self.review_hashes[name])

    def root_families(self):
        """Visit only train directories; never glob across collection splits."""
        namespace = self.request['protocol']['baseline']['episode_group_namespace_sha256']
        targets = [t['id'] for t in self.request['protocol']['collection']['root_targets']]
        for number, job in enumerate(self.jobs):
            if job['split'] != 'train' or job['episode_id'] != episode_id(namespace, job['simulator_seed'], 0):
                raise ValueError('Invalid train task seed/identity')
            directory = inside(self.baseline, 'artifacts/data/response_v1/'+job['job_id'])
            receipt = self.read_reviewed((directory/'complete.json').relative_to(self.baseline).as_posix())
            if (receipt['status'] != 'complete' or receipt['schema_version'] != 1
                    or receipt['binding'] != {'request_hash': self.request_hash, 'job': job}
                    or receipt['metadata']['episode_id'] != job['episode_id']
                    or receipt['metadata']['split'] != 'train' or receipt['metadata']['status'] != 'complete'
                    or receipt['metadata']['seed'] != job['simulator_seed']
                    or [r['target'] for r in receipt['metadata']['roots']] != targets):
                raise ValueError('Train receipt/request mismatch')
            actual_files = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                            if p.is_file() and p != directory/'complete.json'}
            if actual_files != set(receipt['files']) or 'failure.json' in actual_files:
                raise ValueError('Train receipt inventory changed')
            verified = set()
            def read_file(name, parse=True):
                if name not in receipt['files']:
                    raise ValueError('File missing from sealed train receipt: '+name)
                raw = self.bytes(inside(directory, name), receipt['files'][name]); verified.add(name)
                return strict_json(raw) if parse else raw
            discovery = read_file('discovery.json')
            for index, meta in enumerate(receipt['metadata']['roots']):
                prefix = f'r{index}/'
                accounting = read_file(prefix+'accounting.json')
                if accounting != meta:
                    raise ValueError('Root accounting changed')
                if meta['status'] == 'unavailable':
                    self.unavailable.append({'episode_id': job['episode_id'], 'simulator_seed': job['simulator_seed'],
                        'target': meta['target'], 'reason': meta['reason']})
                    continue
                if (meta['status'] != 'complete' or meta['candidate_count'] != 5
                        or meta['branch_executions'] != 10 or meta['future_repeat_exact'] is not True):
                    raise ValueError('Missing complete root/repeat evidence')
                history = read_file(prefix+'history.json'); pack = read_file(prefix+'labels.json')
                if (pack['episode_id'] != job['episode_id'] or pack['root_id'] != meta['root_id']
                        or pack['simulator_seed'] != job['simulator_seed'] or pack['episode_index'] != 0
                        or history['label_pack_sha256'] != receipt['files'][prefix+'labels.json']
                        or inside(self.baseline, history['label_pack']) != directory/prefix/'labels.json'
                        or len(pack['prefix_actions']) != meta['prefix_steps']
                        or pack['prefix_actions'] != discovery['targets'][index]['prefix_actions']):
                    raise ValueError('Train history/labels/prefix join changed')
                branches, repeats = {}, {}
                for cid in range(5):
                    name = prefix+f'c{cid}.json'; repeated_name = prefix+f'c{cid}_repeat.json'
                    branches[cid] = read_file(name)
                    raw_repeat = read_file(repeated_name, parse=False)
                    if receipt['files'][name] != receipt['files'][repeated_name]:
                        raise ValueError('Collected repeat not byte-exact despite exact-repeat receipt')
                    del raw_repeat
                    repeats[cid] = branches[cid]  # Verified equal actual bytes, not substituted simulation.
                self.roster.append((meta['root_id'], job['episode_id']))
                yield history, pack, branches, repeats, {'target': meta['target'],
                    'simulator_seed': job['simulator_seed'], 'job_id': job['job_id'],
                    'source_root_directory': (directory/prefix).relative_to(self.baseline).as_posix()}
            for name in sorted(set(receipt['files'])-verified):
                read_file(name, parse=False)
            if (number+1) % 32 == 0:
                print(f'[ic01] verified_train_episodes={number+1}/{len(self.jobs)} roots={len(self.roster)}', flush=True)
        expected = [(r['root_id'], r['episode_id']) for r in self.review['roots']['train']]
        if self.roster != expected:
            raise ValueError('Reached train root roster differs from frozen dataset review')
