"""The DQ study ("same policy, different learning"): its Study, namespaces, gates, write guard and provenance.

the DQ protocol section 2. Everything a DQ tool needs to know about WHERE it may read, write and draw seeds
lives here, so no other DQ module re-derives a path, a namespace or a seed rule.

    Study('dq')      salt DQ_SALT = 20260923, prefixed to every SeedSequence; namespaces 10-18 of P3's own rule
                     (start = 20260921 * 1000 + index * 100000), after E1's 7-9
    open now         dq_development only (engineering checks D-1..D-6 of the protocol)
    closed           every other namespace, until a reviewer-owned approval names a freeze that verifies against the
                     live idea_dq tree (the verify_freeze pattern of p4_train); today no such approval exists
    write guard      claim_output(): a DQ tool writes only strictly under output/study/gnss_qrl_v1/dq (resolved, so a
                     symlink into idea / idea_e1 / idea_p4 cannot be written through) or under a caller-named test
                     root, and never overwrites
    tree watch       the live tree hashes of the four other checkouts, re-computed and appended at every step; a
                     change is a refusal (TreeWatchError), never a warning
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import p4_train as p4                                                                            # noqa: E402
from aerial_qrl.training.checkpoint_identity import canonical_sha256                             # noqa: E402

TOOL = 'gnss-dq/1'
DQ_SALT = 20260923
NAMESPACE_BASE = 20260921
NAMESPACE_WIDTH = 100_000
#: index -> name, after the last E1 namespace (e1_evaluation = 9). The order is part of the design: never reorder.
NAMESPACE_INDEX = {10: 'dq_development', 11: 'dq_fit', 12: 'dq_oracle', 13: 'dq_calibration', 14: 'dq_heldout',
                   15: 'dq_fresh_eval', 16: 'dq_stress_eval', 17: 'dq_e2_training', 18: 'dq_e2_evaluation'}
DQ_NAMESPACES = {name: {'start': NAMESPACE_BASE * 1000 + index * NAMESPACE_WIDTH,
                        'end': NAMESPACE_BASE * 1000 + (index + 1) * NAMESPACE_WIDTH}
                 for index, name in NAMESPACE_INDEX.items()}
OPEN_NAMESPACES = ('dq_development',)
#: namespace -> the reviewer-owned approval that would open it (none exists; DQ protocol section 2.5)
GATED = {'dq_fit': 'b1', 'dq_oracle': 'b1', 'dq_calibration': 'b1', 'dq_heldout': 'b1', 'dq_fresh_eval': 'b1',
         'dq_stress_eval': 'a', 'dq_e2_training': 'b2', 'dq_e2_evaluation': 'b2'}
APPROVAL_DIR = ROOT / 'docs' / 'study' / 'gnss_qrl_v1' / 'approvals'
APPROVALS = {stage: dict(path=APPROVAL_DIR / f'DQ_{stage.upper()}_PROTOCOL_APPROVAL.json',
                         version=f'gnss-dq-{stage}-protocol-approval/1', key=f'dq_{stage}_protocol_approval')
             for stage in ('a', 'b1', 'b2')}
CALENDAR_CUTOFF_UTC = '2026-10-31T00:00:00+00:00'
OUTPUT_ROOT = ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'dq'
DEVELOPMENT_ROOT = OUTPUT_ROOT / 'development'
DEVELOPMENT_STATEMENT = 'development material; not a confirmatory outcome'
#: The approved trees of the four other checkouts (DQ protocol section 7). Any change stops the work.
WATCHED_TREES = {
    'idea': (str(ROOT.parent / 'idea'),
             '7bc87360556220883debd7cf537f3ff99b2a0f0a83c6fe3ba93786ad752c895a'),
    'idea_e1': (str(ROOT.parent / 'idea_e1'),
                '4a1f2e8e6a76cbc3ba286397139da9bb888c28f2e8a4bf84adb9f2494e46ddc5'),
    'idea_p4': (str(ROOT.parent / 'idea_p4'),
                '03bfd0fe10bc6aa3c2fb503de19d91a9091af8ce5fa319016bf3f434438ad222'),
    'idea_e1t': (str(ROOT.parent / 'idea_e1t'),
                 '41759b78fc20cf4cef73b86c195a7d23a7dbed235759f0996ca4c8631e1a09ef'),
}
E1_TREE = WATCHED_TREES['idea_e1'][1]
#: E1 modules DQ reuses unchanged; a test proves they are byte-identical to the approved E1 tree.
E1_REUSED_MODULES = ('src/aerial_qrl/training/actor_heads.py', 'src/aerial_qrl/training/smdp_ppo.py',
                     'src/aerial_qrl/training/rl_policy.py', 'src/aerial_qrl/training/episodes.py',
                     'scripts/gnss_qrl_runtime/p4_train.py', 'scripts/fast_mission.py',
                     'scripts/gnss_qrl_runtime/run_p3_cells.py')


class DQError(ValueError):
    """A DQ refusal: a closed namespace, a write outside the DQ root, a past cut-off, a moved tree."""


class TreeWatchError(DQError):
    """A watched checkout's live tree is not its approved tree: stop, report, never continue."""


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def _past_cutoff(now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now >= datetime.datetime.fromisoformat(CALENDAR_CUTOFF_UTC)


DQ_STUDY = p4.Study(
    name='dq', tool=TOOL, salt=(DQ_SALT,), training_namespace='dq_e2_training', extra_namespaces=DQ_NAMESPACES,
    approval_path=APPROVALS['b1']['path'], approval_version=APPROVALS['b1']['version'],
    approval_key=APPROVALS['b1']['key'],
    non_claim=('A DQ development output is engineering evidence about instruments (parity, correctness, cost). It is '
               'not a confirmatory outcome and supports no scientific claim.'))


def require_open(namespace, now=None):
    """The single gate every DQ collector passes before drawing a seed from `namespace` (DQ protocol section 2.5)."""
    if namespace not in DQ_NAMESPACES:
        raise DQError(f'{namespace!r} is not a DQ namespace; DQ tools draw only from {sorted(DQ_NAMESPACES)}')
    if _past_cutoff(now):
        raise DQError(f'the DQ calendar cut-off {CALENDAR_CUTOFF_UTC} has passed')
    if namespace in OPEN_NAMESPACES:
        return namespace
    stage = GATED[namespace]
    approval = APPROVALS[stage]
    if not approval['path'].exists():
        raise DQError(f'{namespace} is closed: it needs approval {approval["path"].name}, which does not exist '
                      f'(DQ protocol section 2.5)')
    # An approval exists: it must open a freeze that verifies against this live tree, exactly as E1 and P4 do.
    study = p4.Study(name=f'dq_{stage}', tool=TOOL, salt=(DQ_SALT,), training_namespace='dq_e2_training',
                     approval_path=approval['path'], approval_version=approval['version'],
                     approval_key=approval['key'])
    try:
        p4.require_approved_freeze(study=study)
    except p4.FreezeBindingError as exc:
        raise DQError(f'{namespace} is closed: {exc}') from exc
    return namespace


def cell_runtime():
    """The frozen P3 cells under the DQ study (CellRuntime refuses a DQ range that overlaps any frozen range)."""
    c = 'output/study/gnss_qrl_v1/frozen/'
    return p4.CellRuntime(ROOT / (c + 'p3_freeze_v1/p3_frozen_configuration.json'),
                          ROOT / (c + 'p3_execution_manifest_v1/p3_execution_manifest.json'),
                          ROOT / 'output/training/time_model_village_v1/20260917T074923Z/time_model.json',
                          study=DQ_STUDY)


def seed_sequence(*parts):
    return DQ_STUDY.seed_sequence(*parts)


# -- the write guard ------------------------------------------------------------------------------------------------

def _inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path != root and os.path.commonpath([path, root]) == root


def vet_output(path, test_root=None):
    """Refuse (DQError) unless `path` resolves strictly inside the DQ output root (or `test_root`) and does not exist.
    Creates nothing; claim_output uses it, and so does any writer that opens its own file with 'x' mode."""
    path = Path(path)
    roots = [OUTPUT_ROOT] + ([Path(test_root)] if test_root is not None else [])
    probe = os.path.join(os.path.realpath(path.parent), path.name)
    if not any(_inside(probe, root) for root in roots):
        raise DQError(f'refused: {path} resolves to {probe}, outside the DQ output root {OUTPUT_ROOT}')
    if os.path.lexists(path):
        raise DQError(f'refused: {path} exists; DQ never overwrites')
    return path


def claim_output(path, test_root=None, directory=False):
    """Create `path` (a new file opened 'xb', or a new directory) only strictly inside the DQ output root.

    `test_root` admits one extra root for tests (a pytest tmp_path). The check runs on the RESOLVED path, so a
    symlink leading into a frozen checkout is refused even when its link name sits under the DQ root. Nothing is ever
    overwritten: an existing target is a refusal.
    """
    path = vet_output(path, test_root=test_root)
    parent = path.parent
    roots = [OUTPUT_ROOT] + ([Path(test_root)] if test_root is not None else [])
    parent.mkdir(parents=True, exist_ok=True)
    if not any(_inside(os.path.realpath(parent), root) or os.path.realpath(parent) == os.path.realpath(root)
               for root in roots):
        raise DQError(f'refused: {parent} resolves outside the DQ output root')
    if directory:
        path.mkdir(exist_ok=False)
        return path
    return path.open('xb')


def write_json(path, payload, test_root=None):
    with claim_output(path, test_root=test_root) as handle:
        handle.write((json.dumps(payload, indent=1, sort_keys=False, allow_nan=False) + '\n').encode())
    return Path(path)


# -- provenance and the tree watch ------------------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_of(root):
    """The live code tree hash of a checkout, computed with THAT checkout's own code_version rules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f'code_version_{abs(hash(root))}', Path(root) / 'scripts' / 'code_version.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hashes = module.file_hashes(Path(root))
    return module.tree_hash(hashes), len(hashes)


def dq_tree():
    return tree_of(ROOT)


def watch_trees():
    """{name: {tree, files, approved, unchanged}}; raises TreeWatchError if any watched checkout moved."""
    record, moved = {}, []
    for name, (root, approved) in WATCHED_TREES.items():
        tree, files = tree_of(root)
        record[name] = {'root': root, 'tree': tree, 'files': files, 'approved': approved, 'unchanged': tree == approved}
        if tree != approved:
            moved.append(name)
    if moved:
        raise TreeWatchError(f'watched checkouts moved: {moved} {json.dumps(record)}')
    return record


def append_tree_watch(step, test_root=None):
    """Append one tree-watch line to development/tree_watch.jsonl (append-only; the file is created once)."""
    record = {'utc': utc(), 'step': step, 'dq_tree': dq_tree()[0], 'watched': watch_trees()}
    path = (Path(test_root) if test_root is not None else DEVELOPMENT_ROOT) / 'tree_watch.jsonl'
    if not path.exists():
        claim_output(path, test_root=test_root).close()
    if not (_inside(path, OUTPUT_ROOT) or (test_root is not None and _inside(path, test_root))):
        raise DQError(f'refused: tree watch {path} outside the DQ root')
    with path.open('a') as handle:
        handle.write(json.dumps(record) + '\n')
    return record


def provenance(namespace, sources=(), extra=None):
    """The provenance record of one development output (DQ protocol section 2.4)."""
    return {'tool': TOOL, 'statement': DEVELOPMENT_STATEMENT, 'utc': utc(), 'host': socket.gethostname(),
            'namespace': namespace, 'dq_tree': dq_tree()[0], 'watched_trees': watch_trees(),
            'sources': {str(p): sha256_file(p) for p in sources}, **(extra or {})}


__all__ = ['APPROVALS', 'CALENDAR_CUTOFF_UTC', 'DEVELOPMENT_ROOT', 'DEVELOPMENT_STATEMENT', 'DQError', 'DQ_NAMESPACES',
           'DQ_SALT', 'DQ_STUDY', 'E1_REUSED_MODULES', 'E1_TREE', 'OPEN_NAMESPACES', 'OUTPUT_ROOT', 'TreeWatchError',
           'append_tree_watch', 'canonical_sha256', 'cell_runtime', 'claim_output', 'vet_output', 'dq_tree', 'provenance',
           'require_open', 'seed_sequence', 'sha256_file', 'tree_of', 'watch_trees', 'write_json']
