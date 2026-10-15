"""The immutable tuning campaign: paired manifest, validated evidence, exclusive commit, recovery, completeness.

review 11 broke the first executable version with four probes, and each one is answered by a named
mechanism here.

    an edited result metric survived verification    -> `validate_raw_artifact` + recomputation on verify
    a poisoned raw artifact was reconciled into it   -> the same validator guards recovery
    two writers disagreed about the committed bytes  -> a campaign lock and a no-replace install
    a reduced campaign founded a selection           -> `p3_selection`, which owns that half

The rule that ties them together: **a result record is never evidence on its own.** It is a derivation from
the raw paired artifact, and every reader re-derives it. `completeness()` rebuilds each record from the
validated raw rows and requires exact agreement, and `aggregate()` reads the raw rows rather than the records,
so editing a record can only ever cause a refusal - never a different number.

Two levels of identity, because the experiment has two
    A **unit** is what runs: one candidate on one presence pair at one realisation, both members sharing one
    CRN group. 9 x 4 x 20 = **720 units**. A **job** is one cell of one unit; there are 1440, two per unit, and
    a unit is not complete until both are committed.

The transaction, in the only order that survives a power cut
    1. both episodes run, under a campaign-wide exclusive lock held from unit resolution to commit;
    2. the raw paired artifact is written with a unique temporary name, fsynced, and installed by a link that
       **fails** if the target exists - POSIX rename would silently replace it, which is how two writers can
       disagree about what was committed;
    3. both cell records are derived from that artifact after it has been read back and validated.
    A crash between 2 and 3 is repaired by `reconcile()`, which validates the artifact exactly as execution
    would before deriving anything from it. A crash before 2 leaves nothing to misinterpret.

Nothing here runs an episode, and nothing here deletes or overwrites a committed file. The only `os.unlink` in
this module removes a temporary file this process just created and never installed.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts/gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from aerial_qrl.training.checkpoint_identity import canonical_sha256              # noqa: E402
from aerial_qrl.mission.executive import MissionConfig                            # noqa: E402
from aerial_qrl.training.reward_gnss import timing_weight                         # noqa: E402
from aerial_qrl.mission.contracts import (ActionType, ContractError,               # noqa: E402
                                          action_from_dict)
import comparator_protocol as protocol_module                                     # noqa: E402
import gnss_policies                                                              # noqa: E402
import run_p3_cells as runner                                                     # noqa: E402

CAMPAIGN_VERSION = 'gnss-p3-tuning-campaign/14'
TUNING_NAMESPACE = 'tuning'
#: What a cell record must repeat back about itself.
IDENTITY_FIELDS = ('candidate_sha256', 'cell_id', 'namespace', 'realization', 'seed')
#: What a unit is: one candidate, one presence pair, one realisation, one seed, one CRN group.
UNIT_FIELDS = ('candidate_sha256', 'pair_id', 'namespace', 'realization', 'seed')
#: The selection quantities and the runner-row fields they are derived from. They are not the same names,
#: which is how the two halves drifted apart in the first place.
METRIC_SOURCES = {'score_J': ('score_J',), 'fp_spatial': ('components', 'fp_spatial'),
                  'fix_attempts': ('fixes_attempted',)}
METRIC_FIELDS = tuple(METRIC_SOURCES)
#: The frozen score specification a campaign binds, and every row must declare. review 13: the validator
#: compared `score_J` with the STORED `potential`, so a coherent edit of all three outputs survived. The score
#: is now recomputed from primitives - TP_timely, FP_spatial and these - and the outputs must match it.
SCORE_FIELDS = ('tolerance_m', 'metric', 'false_report_weight', 'timing_weight')
#: What the diagnostics must declare so the row's operational record is evidence rather than decoration.
DIAGNOSTICS_SCHEMA = 'gnss-episode-diagnostics/11'   # /11: exact numeric identities (revision 22)
#: The four identities a GATE manifest and every gate raw artifact must repeat, so the same parents are
#: enforced in loading, commit, recovery, completeness, aggregation and the verdict.
GATE_PARENT_FIELDS = ('selection_sha256', 'selection_campaign_manifest_sha256', 'reference_sha256',
                      'challenger_sha256')
#: Every field of a cell record that is a function of the raw artifact. Completeness recomputes all of them.
DERIVED_FIELDS = METRIC_FIELDS + ('job', 'job_sha256', 'unit_sha256', 'campaign_manifest_sha256',
                                  'code_tree_sha256', 'raw_artifact', 'raw_artifact_sha256', 'row_sha256',
                                  'episode_id', 'termination_reason', 'classification')

#: The frozen infrastructure-failure taxonomy. An episode that flies and ends badly is a mission OUTCOME and
#: belongs in the statistics; a campaign that cannot write a file is an infrastructure FAILURE and does not.
FAILURE_TAXONOMY = {
    'infrastructure/code_tree_drift': 'the live code tree is not the tree this campaign was frozen against',
    'infrastructure/data_binding': 'a protocol, bank, manifest or candidate hash on disk differs from the frozen one',
    'infrastructure/validation': 'an episode ran but its row failed validation against the frozen manifest',
    'infrastructure/pairing': 'the two members of a unit did not share a disturbance',
    'infrastructure/non_finite_metric': 'an episode produced a NaN or infinite selection quantity',
    'infrastructure/corrupt_evidence': 'a raw paired artifact on disk failed validation; nothing was derived from it',
    'infrastructure/lock': 'another process holds this campaign; a second writer is refused before it runs',
    'infrastructure/disk': 'a write, fsync or install failed; nothing was committed',
    'infrastructure/power_or_process_loss': 'a raw paired artifact exists whose cell records were never committed',
    'infrastructure/executor': 'the executor itself raised before or after an episode',
    'mission/outcome': 'the episodes ran and were recorded; a safety abort or a lost batch is a RESULT, not a failure',
}
EVENTS = ('unit/started', 'unit/committed', 'unit/reconciled', 'campaign/frozen', 'campaign/status')


class CampaignError(ValueError):
    """The campaign cannot be built, committed into, reconciled or aggregated."""


class CampaignLocked(CampaignError):
    """Another process holds this campaign."""


#: Only `exclusive()` holds this. A `Lease` constructed without it raises, which is what makes the lease a
#: proof rather than a claim (review 13: `campaign.Lease(directory, "forged")` let recovery write
#: outside the lock).
_LEASE_KEY = object()
#: The attestations of the leases this process is actually holding, by id. `exclusive()` records one on entry
#: and removes it on exit.
#:
#: review 15: the registry used to hold the lease object itself, and `require_lease` then read
#: `campaign_dir`, `pid` and `fileno` back OFF that object - all mutable. A genuine lease taken on a decoy
#: directory, with those three fields reassigned to the target, passed every check and drove recovery to write
#: two records into a campaign whose lock was never held. The attestation is immutable and the guard uses ITS
#: values; the supplied object is only ever used for identity.
_HELD_LEASES = {}

Attestation = collections.namedtuple(
    'Attestation', 'lease pid campaign_dir fileno st_dev st_ino')


class Lease:
    """Proof that the holder owns this campaign right now. It carries no authority of its own and no descriptor.

    review 12 asked for a lease because `reconcile()` took the live tree as an optional argument that a
    caller could omit. Review 13 found the lease itself forgeable: its constructor was public, and
    `require()` checked only the flag, the pid and the path, all of which a fabricated object supplies.

    review 16 took the last thing off it. The lease used to carry `fileno`, the descriptor the OS lock
    was taken on, and that descriptor is a **capability**: a validation callback - untrusted code, running
    inside `validate_row` - did `fcntl.flock(lease.fileno, fcntl.LOCK_UN)` and the commit then completed with
    no lock held at all. The handle now lives only in the module's own registry, where no caller can reach it,
    and `require_lease()` proves the lock is still held rather than assuming it.

    Nothing here is a field a caller can set to gain anything: the object was made by `exclusive()` (the
    private key), this process is still holding it (the registry), and the guard reads the registry.
    """

    __slots__ = ('campaign_dir', 'purpose', 'pid')

    def __init__(self, campaign_dir, purpose, key=None):
        if key is not _LEASE_KEY:
            raise CampaignError('a campaign lease can only be created by campaign.exclusive(); a constructed '
                                'one is a claim, not a proof')
        self.campaign_dir = Path(campaign_dir).resolve()
        self.purpose, self.pid = purpose, os.getpid()

    @property
    def active(self):
        attestation = _HELD_LEASES.get(id(self))
        return attestation is not None and attestation.lease is self

    def require(self, campaign_dir):
        """Kept for readability at call sites; it delegates to the module guard and adds no authority."""
        return require_lease(self, campaign_dir)


def require_lease(lease, campaign_dir):
    """THE authority check. Module code, never a method the supplied object gets to answer.

    review 14: every mutation asked the object to authenticate itself - `lease.require(...)` - so an
    unrelated class with a `require` method that returns `self` drove recovery to write two records with no
    OS lock held at all. A private constructor stops `Lease(path, 'forged')`; it does nothing about an object
    that was never a `Lease`.

    Four things are checked here, by this function, about an object it does not trust: it is a genuine `Lease`
    and not a subclass pretending to be one; it is the exact object this process registered when it took the
    lock; it belongs to this process and this campaign; and the descriptor it was taken on is still the
    campaign's own lock file.
    """
    if type(lease) is not Lease:
        raise CampaignError(f'{type(lease).__name__} is not a campaign lease; a mutation needs the object '
                            'campaign.exclusive() returned, and an object that answers require() is not one')
    attestation = _HELD_LEASES.get(id(lease))
    if attestation is None or attestation.lease is not lease:
        raise CampaignError('the campaign lease has been released; a mutation needs a live lease')
    # Everything below reads the ATTESTATION, never the lease object: its fields are mutable and a genuine
    # lease with retargeted fields is exactly the attack this guards against.
    if attestation.pid != os.getpid():
        raise CampaignError('the campaign lease belongs to another process')
    wanted = Path(campaign_dir).resolve()
    if wanted != attestation.campaign_dir:
        raise CampaignError(f'the lease is for {attestation.campaign_dir}, not {wanted}')
    lock_path = paths(attestation.campaign_dir)['lock']
    try:
        held = os.fstat(attestation.fileno)
        on_disk = os.stat(lock_path)
    except OSError as error:
        raise CampaignError(f'the campaign lock is no longer readable: {error}') from error
    if (held.st_dev, held.st_ino) != (attestation.st_dev, attestation.st_ino):
        raise CampaignError('the descriptor this lease was taken on is no longer the file it was taken on')
    if (on_disk.st_dev, on_disk.st_ino) != (attestation.st_dev, attestation.st_ino):
        raise CampaignError('the lease was taken on a different lock file than the campaign now has')
    _prove_lock_held(attestation, lock_path)
    return lease


def _prove_lock_held(attestation, lock_path):
    """Prove the advisory OS lock on this campaign is held, and held by US. The kernel answers, not the lease.

    review 16: every check above was about identity - the right type, the right registry entry, the
    right process, the right inode - and identity is not possession. A callback inside `validate_row()`
    released the `flock` on the attested descriptor and `commit_unit()` then installed a raw artifact and two
    result records with the campaign unlocked, so a second writer could have been running at the same time.

    Two questions, and they have to be asked in this order.

    **Is it held at all?** A SECOND descriptor on the same file is a different open file description, so
    `LOCK_EX | LOCK_NB` on it fails while anybody holds the lock and SUCCEEDS the moment nobody does. Success
    is therefore the failure case here, and the probe drops the lock it just took before refusing so that a
    refusal never leaves the campaign locked by a descriptor nobody owns.

    **Is it held by us?** Only then is the attested handle asked for `LOCK_EX | LOCK_NB`. Within this process
    that is a no-op on the description that already owns the lock, and it fails with `BlockingIOError` if some
    other process took it - which is the case review 16 asked for by name. Asking this FIRST would defeat the
    whole check: on a released lock the no-op would quietly re-take it and the probe would then see it held.
    """
    try:
        probe = os.open(lock_path, os.O_RDONLY)
    except OSError as error:
        raise CampaignError(f'the campaign lock cannot be probed: {error}') from error
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass                         # somebody holds it; the next question is whether that is us
        except OSError as error:
            raise CampaignError(f'the campaign lock cannot be probed: {error}') from error
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise CampaignError('the campaign lock is not held: the lease attests to a descriptor but the OS '
                                'lock on it is gone, so this campaign has no writer and a mutation would race')
    finally:
        os.close(probe)
    try:
        fcntl.flock(attestation.fileno, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise CampaignError('the campaign lock is held by another writer, not by this lease; a mutation would '
                            'race with whoever owns it') from error
    except OSError as error:
        raise CampaignError(f'the campaign lock cannot be reacquired: {error}') from error


# --- identity ---------------------------------------------------------------------------------------------

def code_tree_hash():
    """The LIVE tree, recomputed. Never a value a caller supplied."""
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/code_version.py')],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or None


def job_id(execution_manifest_sha256, identity):
    label = '|'.join([CAMPAIGN_VERSION, execution_manifest_sha256]
                     + [str(identity[field]) for field in IDENTITY_FIELDS])
    return hashlib.sha256(label.encode()).hexdigest()


def unit_id(execution_manifest_sha256, identity):
    label = '|'.join([CAMPAIGN_VERSION, 'unit', execution_manifest_sha256]
                     + [str(identity[field]) for field in UNIT_FIELDS])
    return hashlib.sha256(label.encode()).hexdigest()


def _identity_of(job):
    return {field: job[field] for field in IDENTITY_FIELDS}


def paths(campaign_dir):
    campaign_dir = Path(campaign_dir)
    return {'raw': campaign_dir / 'raw', 'results': campaign_dir / 'results',
            'registry': campaign_dir / 'run_registry.jsonl', 'lock': campaign_dir / 'campaign.lock',
            'episodes': campaign_dir / 'episodes'}


# --- the frozen manifest ------------------------------------------------------------------------------------

def cell_identity(cell):
    """What a row repeats about its execution-manifest cell, exactly as the runner copies it into the row."""
    return {'budget_name': cell['budget_name'], 'd600_rms_m': cell['d600_rms_m'],
            'georeference_sha256': cell['georeference_sha256'],
            'protocol_path': cell['data']['protocol_path'], 'protocol_sha256': cell['data']['protocol_sha256'],
            'bank_manifest_sha256': [part['manifest_sha256'] for part in cell['data']['bank_parts']]}


def build(execution_manifest, comparator_protocol, frozen_configuration, namespace=TUNING_NAMESPACE,
          code_tree_sha256=None, candidates=None):
    """Enumerate every unit and every job once, from the frozen execution manifest and comparator protocol.

    `namespace` is `tuning` for the real campaign, `gate` for a manifest built from a verified selection, and
    `pilot` for a bounded smoke whose jobs carry a different identity by construction.
    """
    if namespace not in ('tuning', 'pilot', 'gate'):
        raise CampaignError(f'namespace {namespace!r}: a campaign is tuning, gate, or pilot for a smoke')
    if frozen_configuration['manifest_sha256'] != execution_manifest['frozen_manifest_sha256']:
        raise CampaignError('the frozen configuration is not the one the execution manifest was built for')
    score = frozen_configuration['score']
    missing = [key for key in SCORE_FIELDS if key not in score]
    if missing:
        raise CampaignError(f'the frozen score specification is missing {missing}')
    if namespace == 'gate' and not candidates:
        raise CampaignError('a gate manifest must name the selected reference and the fixed challenger; it is '
                            'never built from the whole eligible set')
    if namespace != 'gate' and candidates is not None:
        raise CampaignError(f'a {namespace} campaign runs every reference-eligible candidate and takes no '
                            'candidate list')
    eligible = candidates if candidates is not None else [
        c for c in comparator_protocol['candidates'] if c['role'] == protocol_module.ROLE_REFERENCE]
    if not eligible:
        raise CampaignError('the comparator protocol declares no reference-eligible candidate')
    if len({c['candidate_sha256'] for c in eligible}) != len(eligible):
        raise CampaignError('the candidate list repeats a candidate')
    cells = {cell['cell_id']: cell for cell in execution_manifest['cells']}
    if not cells:
        raise CampaignError('the execution manifest declares no cell')
    pairs = execution_manifest['pairs']
    execution_sha256 = execution_manifest['manifest_sha256']

    realizations = set()
    for members in pairs.values():
        if len(members) != 2:
            raise CampaignError(f'a presence pair must have exactly two members, got {members}')
        seeds = [tuple(cells[m]['seeds'][namespace]) for m in members]
        if seeds[0] != seeds[1]:
            raise CampaignError(f'the members of {members} do not share their {namespace} seeds')
        realizations.add(len(seeds[0]))
    if len(realizations) != 1:
        raise CampaignError(f'pairs disagree on the number of {namespace} realisations: {sorted(realizations)}')
    per_cell = realizations.pop()
    if not per_cell:
        raise CampaignError(f'namespace {namespace!r} has no seed in the execution manifest')

    units, jobs = [], []
    for candidate in eligible:
        for pair_id, members in sorted(pairs.items()):
            pair_cells = [cells[m] for m in members]
            for realization, seed in enumerate(pair_cells[0]['seeds'][namespace]):
                unit_identity = {'candidate_sha256': candidate['candidate_sha256'], 'pair_id': pair_id,
                                 'namespace': namespace, 'realization': realization, 'seed': seed}
                unit_sha256 = unit_id(execution_sha256, unit_identity)
                member_jobs = []
                for cell in sorted(pair_cells, key=lambda c: c['cell_index']):
                    identity = {'candidate_sha256': candidate['candidate_sha256'], 'cell_id': cell['cell_id'],
                                'namespace': namespace, 'realization': realization, 'seed': seed}
                    job_sha256 = job_id(execution_sha256, identity)
                    member_jobs.append(job_sha256)
                    jobs.append(dict(identity, job_sha256=job_sha256, unit_sha256=unit_sha256,
                                     candidate_index=candidate['candidate_index'],
                                     candidate_name=candidate['name'],
                                     candidate_parameters=candidate['parameters'],
                                     # review 20: the record of what the policy ran with was checked
                                     # only against the tuned subset, so `behaviour_version` accepted any
                                     # string and a field could simply be deleted. The COMPLETE effective
                                     # record is frozen here, from the same classmethod the policy fills
                                     # `self.parameters` with, and the row must repeat it exactly.
                                     candidate_policy_parameters=effective_policy_parameters(
                                         candidate['name'], candidate['parameters']),
                                     candidate_role=candidate['role'],
                                     cell_index=cell['cell_index'], pair_id=pair_id,
                                     presence=cell['presence'], mission_budget_s=cell['mission_budget_s'],
                                     cell_weight=cell['weight'], n_ref=cell['n_ref'],
                                     # revision 21: the row repeats these about its cell, and nothing
                                     # compared them - the mutation sweep rewrote each one without a
                                     # refusal. Frozen here from the execution manifest's own cell.
                                     cell_identity=cell_identity(cell)))
                units.append(dict(unit_identity, unit_sha256=unit_sha256, jobs=member_jobs,
                                  candidate_index=candidate['candidate_index'],
                                  candidate_name=candidate['name'],
                                  candidate_parameters=candidate['parameters'],
                                  crn_sha256=runner.crn_group(pair_id, namespace, seed)['crn_sha256']))

    if len({unit['unit_sha256'] for unit in units}) != len(units):
        raise CampaignError('duplicate unit identity; the enumeration is not one-to-one')
    if len({job['job_sha256'] for job in jobs}) != len(jobs):
        raise CampaignError('duplicate job identity; the enumeration is not one-to-one')

    manifest = {
        'campaign_version': CAMPAIGN_VERSION,
        'status': (f'job manifest only; NO {namespace} job has been executed'
                   + ('' if namespace == 'gate' else ' and NO comparator has been selected')),
        'namespace': namespace,
        'namespace_rule': ('a pilot campaign is infrastructure evidence and can never select a comparator or '
                           'enter P3 statistics; a tuning campaign selects the reference; a gate campaign '
                           'scores the selected reference against the fixed challenger, once'),
        'frozen_manifest_sha256': execution_manifest['frozen_manifest_sha256'],
        'execution_manifest_sha256': execution_sha256,
        'comparator_protocol_sha256': comparator_protocol['protocol_sha256'],
        'score_version': frozen_configuration['score_version'],
        'reward_version': frozen_configuration['reward_version'],
        'score': {key: score[key] for key in SCORE_FIELDS},
        'report_thresholds': {'min_detections': MissionConfig.report_min_detections,
                              'min_views': MissionConfig.report_min_views,
                              'merge_radius_m': MissionConfig.report_merge_radius_m,
                              'view_separation_m': MissionConfig.view_separation_m,
                              'source': ('aerial_qrl.mission.executive.MissionConfig defaults, which the '
                                         'bound code tree already fixes; recorded here so a row\'s own '
                                         'operational record is checked against them rather than assumed')},
        'diagnostics_schema': DIAGNOSTICS_SCHEMA,
        'score_rule': ('potential = (TP_timely - false_report_weight * FP_spatial) / N_ref, and '
                       'J = (I(RETURN_OK) * TP_timely - false_report_weight * FP_spatial) / N_ref. Both are '
                       'recomputed from these primitives for every row; no stored output is trusted.'),
        'policy_behaviour_version': comparator_protocol['policy_behaviour_version'],
        'policy_source_sha256': comparator_protocol['policy_source_sha256'],
        'code_tree_sha256': code_tree_sha256,
        'crn_version': runner.CRN_VERSION,
        'reference_eligible_candidates': [c['candidate_sha256'] for c in eligible
                                          if c['role'] == protocol_module.ROLE_REFERENCE],
        'challenger_candidates': [c['candidate_sha256'] for c in comparator_protocol['candidates']
                                  if c['role'] == protocol_module.ROLE_CHALLENGER],
        'challenger_rule': ('the challenger takes no tuning job: it has no parameter to select, and it is '
                            'evaluated once on gate seeds against the selected reference'),
        'candidates': len(eligible), 'pairs': len(pairs), 'cells': len(cells),
        'realizations_per_cell': per_cell,
        'expected_units': len(units), 'expected_jobs': len(jobs),
        'unit_fields': list(UNIT_FIELDS), 'identity_fields': list(IDENTITY_FIELDS),
        'metric_sources': {k: list(v) for k, v in METRIC_SOURCES.items()},
        'derived_fields': list(DERIVED_FIELDS),
        'failure_taxonomy': dict(FAILURE_TAXONOMY),
        'commit_order': ['take the campaign lease', 'run both episodes',
                         'recompute the live code tree immediately before writing',
                         'install the raw paired artifact without replacing anything',
                         'read it back, validate it, and derive both cell records from it'],
        'verification_rule': ('a cell record is a DERIVATION, never evidence on its own: completeness '
                              'recomputes every derived field from the validated raw rows and requires exact '
                              'agreement, and aggregation reads the raw rows rather than the records'),
        'refusal_rules': [
            'a result file outside the frozen job list is foreign, not extra data',
            'a result whose recomputation from its raw artifact disagrees with it is tampered evidence',
            'a raw artifact that fails validation is corrupt evidence; nothing is ever derived from it',
            'a job executed under a different code tree is drift; a mixture of trees is not one campaign',
            'a second concurrent writer is refused by the campaign lock before it runs an episode',
            'every mutation requires a live campaign lease and recomputes the live tree for itself',
            'every redundant identity a row carries is recomputed; a stored assertion flag proves nothing',
            'aggregation and selection refuse any campaign that is not complete',
            'nothing under output/ is ever deleted or overwritten, including by recovery',
        ],
        'units': units, 'jobs': jobs,
    }
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    return manifest


def load(campaign_dir):
    """Read a frozen campaign back and verify it against its own hash before anything uses it."""
    campaign_dir = Path(campaign_dir)
    payload = strict_loads((campaign_dir / 'p3_tuning_campaign.json').read_text())
    manifest = payload['job_manifest']
    recomputed = canonical_sha256({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    if recomputed != manifest['manifest_sha256']:
        raise CampaignError(f'the campaign manifest does not match its own hash: {recomputed} != '
                            f'{manifest["manifest_sha256"]}')
    # review 13: the four gate parents were enforced by the executor and the verdict but not by the
    # shared contract, so a self-hashed gate manifest with none of them loaded and reported `complete`.
    present = [field for field in GATE_PARENT_FIELDS if manifest.get(field)]
    if manifest.get('namespace') == 'gate':
        absent = [field for field in GATE_PARENT_FIELDS if not manifest.get(field)]
        if absent:
            raise CampaignError(f'a gate campaign must name its parents; missing {absent}')
    elif present:
        raise CampaignError(f'a {manifest.get("namespace")!r} campaign must not carry gate parents {present}: '
                            'they would be read as evidence of a selection it never had')
    return manifest


# --- exclusive access and durable writing ---------------------------------------------------------------------

@contextlib.contextmanager
def exclusive(campaign_dir, purpose='execute'):
    """One writer per campaign, from unit resolution through commit.

    Without this, two executors can resolve the same pending unit, run it twice and race on the install. The
    lock is advisory but every writer here takes it, and it is non-blocking on purpose: a second executor is
    told to stop rather than left to wait and then duplicate work.
    """
    lock_path = paths(campaign_dir)['lock']
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, 'a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise CampaignLocked(f'another process holds {lock_path}; a campaign has exactly one writer '
                             f'(attempted: {purpose})')
    lease = Lease(campaign_dir, purpose, key=_LEASE_KEY)
    held = os.fstat(handle.fileno())
    _HELD_LEASES[id(lease)] = Attestation(lease, os.getpid(), Path(campaign_dir).resolve(),
                                          handle.fileno(), held.st_dev, held.st_ino)
    try:
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps({'pid': os.getpid(), 'purpose': purpose,
                                 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')})
                     + '\n')
        handle.flush()
        yield lease
    finally:
        _HELD_LEASES.pop(id(lease), None)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _refuse_constant(token):
    raise CampaignError(f'{token} is not a number; this campaign reads strict JSON only, because a NaN that '
                        'survives a read reaches geometry and canonical hashing as a silent answer')


def strict_loads(text):
    """`json.loads` with NaN, Infinity and -Infinity refused at the parse (review 16/17).

    Python accepts those three non-standard constants by default. Every validator below proves the values it
    reads are finite, but "every validator" is a claim about coverage, and the parse is one place.
    """
    return json.loads(text, parse_constant=_refuse_constant)


def strict_dumps(payload, **kwargs):
    """`json.dumps` that refuses to WRITE a non-finite value, so no artifact of ours can carry one."""
    try:
        return json.dumps(payload, allow_nan=False, **kwargs)
    except ValueError as error:
        raise CampaignError(f'the payload carries a value JSON cannot represent: {error}') from error


def _prepared(path, payload):
    """Serialise and fsync the bytes. Everything expensive happens HERE, before any authority is proved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial')
    text = strict_dumps(payload, indent=1, sort_keys=True) + '\n'
    with open(temporary, 'x') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return path, temporary


def _link_install(path, temporary):
    """The no-replace install itself. Nothing between entering this and `os.link` but the link.

    `os.rename` is atomic but REPLACES, so two writers can both believe they committed while only one set of
    bytes survives. `os.link` fails with FileExistsError instead, which makes the losing writer's refusal
    explicit. The only `os.unlink` in this module removes the temporary file this process just created.
    """
    try:
        os.link(temporary, path)                   # no-replace install: the loser of a race is refused
    except FileExistsError:
        os.unlink(temporary)                       # our own temporary file, never a committed artifact
        raise CampaignError(f'{path} already exists; this campaign never overwrites a committed file')
    finally:
        if temporary.exists() and path.exists():
            os.unlink(temporary)                   # our own temporary file, never a committed artifact
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def install_evidence(path, payload, manifest, campaign_dir, lease, where):
    """THE way this campaign installs evidence. One primitive, one order, no caller-supplied proof.

    review 16 moved the proof inside the write. Review 17 found what was still wrong with it: the proof
    was `require_lease(...)` followed by `require_frozen_tree(...)`, and the tree recomputation is *work* -
    it runs `scripts/code_version.py` over 466 files in a subprocess. An `flock` released during that
    subprocess left the lock check already behind, and all three production paths installed evidence with the
    campaign unlocked. No partial file and no success event were needed: the artifact was simply there.

    So the order is inverted and fixed here, where no caller can reorder it:

        1. serialise and fsync the bytes          (all the expensive work, before anything is proved)
        2. recompute and verify the live tree     (more work, but now it is BEFORE the last proof)
        3. prove the lease and the OS lock        (the final operation, and it is cheap and constant)
        4. os.link                                (the install)

    Between 3 and 4 there is no callback, no serialisation and no caller code - only the return from
    `require_lease`, whose own last act is to ask the kernel whether this process still owns the lock.
    """
    path, temporary = _prepared(path, payload)
    try:
        require_frozen_tree(manifest, where)       # step 2: the work, deliberately before the last proof
        require_lease(lease, campaign_dir)         # step 3: the LAST thing before the link
    except BaseException:
        os.unlink(temporary)                       # our own temporary file, never a committed artifact
        raise
    return _link_install(path, temporary)


def atomic_write_json(path, payload):
    """Install a file that either exists whole or not at all, and never replaces an existing one.

    This is the UNPROVED write, for artifacts that are not campaign evidence: the selection, the gate verdict
    and test fixtures. Campaign evidence goes through `install_evidence`, which proves the tree and the lock.
    """
    path, temporary = _prepared(path, payload)
    return _link_install(path, temporary)


def register(campaign_dir, event, **fields):
    """Append one line to the run registry. Append-only, flushed and fsynced, never rewritten."""
    if event not in FAILURE_TAXONOMY and event not in EVENTS:
        raise CampaignError(f'{event!r} is not in the frozen taxonomy or the event list')
    line = json.dumps(dict(fields, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                           event=event), sort_keys=True)
    registry = paths(campaign_dir)['registry']
    registry.parent.mkdir(parents=True, exist_ok=True)
    with open(registry, 'a') as handle:
        handle.write(line + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return line


# --- the evidence adapter and the one shared validator ----------------------------------------------------------

def _dig(row, path):
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise CampaignError(f'the runner row has no {".".join(path)}')
        value = value[key]
    return value


def derive_metrics(row):
    """The three selection quantities, taken from a validated runner row and from nowhere else."""
    metrics = {}
    for name, path in METRIC_SOURCES.items():
        value = _dig(row, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise CampaignError(f'{name} is not a finite number ({value!r})')
        metrics[name] = float(value)
    return metrics


def validate_row(manifest, job, row):
    """Everything a row must agree with before it can become evidence. Returns the disagreements."""
    expected_crn = runner.crn_group(job['pair_id'], job['namespace'], job['seed'])['crn_sha256']
    checks = [
        ('frozen_manifest_sha256', row.get('frozen_manifest_sha256'), manifest['frozen_manifest_sha256']),
        ('execution_manifest_sha256', row.get('execution_manifest_sha256'), manifest['execution_manifest_sha256']),
        ('comparator_protocol_sha256', row.get('comparator_protocol_sha256'), manifest['comparator_protocol_sha256']),
        ('candidate_sha256', row.get('candidate_sha256'), job['candidate_sha256']),
        ('candidate_index', row.get('candidate_index'), job['candidate_index']),
        ('candidate_role', row.get('candidate_role'), job['candidate_role']),
        ('comparator', row.get('comparator'), job['candidate_name']),
        ('comparator_parameters', row.get('comparator_parameters'), job['candidate_parameters']),
        # review 20: the COMPLETE effective record, frozen at build time from the policy's own
        # classmethod, so a deleted field, an extra one or an invented behaviour version are all refusals.
        ('policy_parameters', (row.get('diagnostics') or {}).get('policy_parameters'),
         job['candidate_policy_parameters']),
        ('seed_namespace', row.get('seed_namespace'), job['namespace']),
        ('cell_id', row.get('cell_id'), job['cell_id']),
        # review 19: `cell_index` was typed but never compared with the frozen job, so 999999 passed.
        ('cell_index', row.get('cell_index'), job['cell_index']),
        # ... and `reobservation_seed` was neither typed nor bound, although the runner sets it FROM the seed.
        ('reobservation_seed', row.get('reobservation_seed'), job['seed']),
        ('presence', row.get('presence'), job['presence']),
        ('pair_id', row.get('pair_id'), job['pair_id']),
        ('realization', row.get('realization'), job['realization']),
        ('seed', row.get('seed'), job['seed']),
        ('n_ref', row.get('n_ref'), job['n_ref']),
        ('mission_budget_s_applied', row.get('mission_budget_s_applied'), job['mission_budget_s']),
        ('crn_sha256', row.get('crn_sha256'), expected_crn),
        ('crn_version', row.get('crn_version'), manifest['crn_version']),
        ('reward_identity_holds', row.get('reward_identity_holds'), True),
    ]
    # revision 21: everything a row repeats about itself that the frozen job, the CRN group or the runner's own
    # constants determine. The mutation sweep rewrote every one of these on genuine rows without a refusal.
    group = runner.crn_group(job['pair_id'], job['namespace'], job['seed'])
    path_key = runner.crn_path_key(group)
    # `GeoreferenceService.__init__` keys the Brownian path `<key>.bias` and the anchor noise
    # `<key>.anchor_noise`, and `disturbance_identity` hashes each key; a test pins this to a real service.
    bias_key, noise_key = f'{path_key}.bias', f'{path_key}.anchor_noise'
    frozen_score = manifest['score']
    checks += [
        ('crn_label', row.get('crn_label'), group['crn_label']),
        ('disturbance_path_key', row.get('disturbance_path_key'), bias_key),
        ('disturbance_path_sha256', row.get('disturbance_path_sha256'), hashlib.sha256(bias_key.encode()).hexdigest()),
        ('measurement_noise_key', row.get('measurement_noise_key'), noise_key),
        ('measurement_noise_sha256', row.get('measurement_noise_sha256'),
         hashlib.sha256(noise_key.encode()).hexdigest()),
        ('episode_id', row.get('episode_id'),
         runner.episode_identity(job['candidate_index'], job['candidate_sha256'], job['candidate_name'],
                                 job['cell_id'], job['namespace'], job['realization'])),
        ('runner', row.get('runner'), runner.RUNNER_VERSION),
        ('classification', row.get('classification'), runner.CLASSIFICATION[job['namespace']]),
        ('mission_budget_s_requested', row.get('mission_budget_s_requested'), job['mission_budget_s']),
        ('cell_weight', row.get('cell_weight'), job['cell_weight']),
        ('score_spec', row.get('score_spec'),
         {'score_version': manifest['score_version'], 'reward_version': manifest['reward_version'],
          'n_ref': job['n_ref'], 'mission_budget_s': job['mission_budget_s'],
          'tolerance_m': frozen_score['tolerance_m'], 'metric': frozen_score['metric'],
          'false_report_weight': frozen_score['false_report_weight'],
          'timing_weight': frozen_score['timing_weight'], 'provenance': 'frozen'}),
    ] + [(field, row.get(field), value) for field, value in sorted(job['cell_identity'].items())]
    # Canonical bytes, not Python equality (review 20): `1881 == 1881.0` and `True == 1`, so every
    # frozen value used to accept a retyped copy of itself.
    problems = [f'{name}: {actual!r} != {expected!r}' for name, actual, expected in checks
                if not _same(actual, expected)]
    # review 18: every check above is an equality, and `3 == 3.0 == True` in Python. The identities
    # that are counts or indices have to be counts or indices as well as equal.
    for name in ('candidate_index', 'cell_index', 'realization', 'n_ref', 'seed', 'reobservation_seed'):
        if not _is_count(row.get(name)):
            problems.append(f'{name} is not a non-negative integer ({row.get(name)!r})')
    # review 19: an integer is not necessarily the right index. The submitted reports are numbered in
    # the order they were submitted, so the sequence is 0..N-1 - which is also unique and in range by
    # construction, and 999999 or a repeat is neither.
    for index, submission in enumerate(row.get('submissions') or []):
        if not isinstance(submission, dict) or submission.get('report_index') != index \
                or not _is_count(submission.get('report_index')):
            problems.append(f'submissions[{index}].report_index is {(submission or {}).get("report_index")!r} '
                            'and the reports are numbered 0..N-1 in the order they were submitted')
    for field in ('disturbance_path_sha256', 'measurement_noise_sha256', 'georeference_sha256'):
        if not isinstance(row.get(field), str) or len(row[field]) != 64:
            problems.append(f'{field} is not a stochastic identity ({row.get(field)!r})')
    try:
        derive_metrics(row)
    except CampaignError as error:
        problems.append(str(error))
    problems += semantic_problems(row, manifest)
    return problems


#: THE ONE RETAINED NUMERIC TOLERANCE (review 21, revision 22). Every other numeric identity in the row is
#: checked by canonical typed equality: the validator performs the producer's own arithmetic, in the producer's
#: order, and on the bounded smoke evidence every such identity is bit-exact. Review 21 showed why a global
#: `1e-9` was wrong: `score_J` 0.0 -> 9e-10 passed every layer and recovery wrote it into a derived record.
#:
#: `reward_sum` is the exception, and it is declared rather than hidden. It is `RewardBreakdown.total()`: the
#: telescoped sum of per-report potential increments plus the terminal correction. Its terms are prefix
#: potentials re-matched against evaluator-private targets that the row does not carry, so the validator
#: cannot redo that sum, and floating-point telescoping does not guarantee the final potential bit for bit.
#: The bound is the runner's own invariant-10 check (`reward_gnss.check_reward_identity`, 1e-9), and no
#: statistic reads `reward_sum`: selection and the gate read `score_J`, which IS recomputed exactly. So
#: `reward_sum` is TOLERANCE-EQUIVALENT to J, never "exact", and a change inside the bound is accepted - the
#: permanent sweep probes both sides of it.
REWARD_IDENTITY_TOLERANCE = 1e-9
TOLERANCE_PATHS = {'reward_sum': REWARD_IDENTITY_TOLERANCE}


def semantic_problems(row, manifest=None):
    """Recompute every redundant identity the row contains, instead of trusting a stored flag.

    review 12's decisive probe: change `score_J` from 0.0 to 999999.0 in a genuine raw artifact and leave
    `reward_identity_holds` set to true. The old validator read the Boolean and accepted it, and recovery
    turned the poison into a derived result. A stored assertion is the attacker's field, not the evidence.

    Review 13 found the next layer of the same mistake: the identity compared `score_J` with the STORED
    `potential`, so editing `score_J`, `reward_sum` and `potential` together - a coherent edit of all three
    outputs - survived while TP_timely and FP said the score was zero. With a `manifest`, the score is now
    recomputed from the FROZEN specification and the primitive components:

        potential = (TP_timely - false_report_weight * FP_spatial) / N_ref
        J         = (I(RETURN_OK) * TP_timely - false_report_weight * FP_spatial) / N_ref

    and all three stored outputs must equal it. No output is checked against another output. Every remaining
    identity is a second copy of a quantity the row records twice, in places an editor has to find all of.
    """
    found = []
    # revision 21: the closed schema runs FIRST. The generated retype attacks showed why: with `tp_timely` a
    # string or a fix event a list, the score arithmetic and the event identities below raised TypeError and
    # AttributeError - inside recovery, too - instead of refusing. Every type is proved before any is used.
    if not isinstance(row, dict):
        return ['the row is not an object']
    if not isinstance(row.get('diagnostics'), dict):
        return ['diagnostics is missing: an episode row without its operational record is not evidence']
    # A row of another diagnostics schema is refused by its VERSION, before its shape is compared with this one.
    if row['diagnostics'].get('schema') != DIAGNOSTICS_SCHEMA:
        return [f'recomputed identity fails: the diagnostics schema is {row["diagnostics"].get("schema")!r}, '
                f'not {DIAGNOSTICS_SCHEMA!r}']
    schema = _schema_problems(row)
    if schema:
        return schema
    components = row.get('components')
    diagnostics = row.get('diagnostics')
    if not isinstance(components, dict):
        return ['components is not an object']
    if not isinstance(diagnostics, dict):
        return ['diagnostics is missing: an episode row without its operational record is not evidence']
    for field in ('score_J', 'reward_sum', 'n_ref', 'elapsed_s', 'decisions'):
        if field not in row:
            return [f'the row has no {field}']
    for field in ('tp_timely', 'tp_spatial', 'fp_spatial', 'duplicates', 'potential', 'matched_report_indices'):
        if field not in components:
            return [f'components has no {field}']

    n_ref = row['n_ref']
    if not _is_count(n_ref) or n_ref <= 0:
        return [f'n_ref is not a positive count ({n_ref!r})']
    # review 18: `components.unresolved = false` and a `matched_report_indices` of Booleans both passed,
    # because `False == 0` and `True == 1`. An index into the submitted reports is an int, in range, once.
    submitted = row.get('submissions')
    indices = components.get('matched_report_indices')
    if not _is_count(components.get('unresolved')):
        return [f'components.unresolved is not a count ({components.get("unresolved")!r})']
    if not isinstance(indices, (list, tuple)) or not all(_is_count(i) for i in indices):
        return [f'components.matched_report_indices is not a list of indices ({indices!r})']
    if len(set(indices)) != len(indices):
        return ['components.matched_report_indices names a submitted report twice']

    # The frozen score, recomputed from primitives with the producer's exact expressions
    # (`reward_gnss.components` and `terminal_score`), and compared as canonical typed values - no tolerance.
    # Without a manifest the weight is read from the row's own `score_spec`; `validate_row` always passes the
    # manifest, which binds that `score_spec` to the frozen one.
    specification = row.get('score_spec')
    if manifest is not None:
        frozen_score = manifest.get('score') or {}
        if not isinstance(specification, dict):
            return ['score_spec is missing: a row must declare the score it was scored under']
        declared = [f'score_spec.{key}: {specification.get(key)!r} != {frozen_score.get(key)!r}'
                    for key in SCORE_FIELDS if specification.get(key) != frozen_score.get(key)]
        for key, expected in (('score_version', manifest.get('score_version')),
                              ('reward_version', manifest.get('reward_version')),
                              ('n_ref', n_ref),
                              ('mission_budget_s', row.get('mission_budget_s_applied'))):
            if specification.get(key) != expected:
                declared.append(f'score_spec.{key}: {specification.get(key)!r} != {expected!r}')
        # review 19: `score_spec.n_ref = 18.0` compared equal to 18 and passed. It is a count.
        if not _is_count(specification.get('n_ref')):
            declared.append(f'score_spec.n_ref is not a count ({specification.get("n_ref")!r})')
        if declared:
            return declared
        weight = frozen_score['false_report_weight']
    else:
        weight = specification['false_report_weight']
    expected_potential = (components['tp_timely'] - weight * components['fp_spatial']) / n_ref
    credited = components['tp_timely'] if row.get('return_ok') else 0.
    expected_score = (credited - weight * components['fp_spatial']) / n_ref
    found += [f'recomputed identity fails: {name}' for name, holds in (
        ('the frozen potential against TP_timely, FP and N_ref, exactly',
         _same(components['potential'], expected_potential)),
        ('the frozen score against TP_timely, FP and N_ref, exactly', _same(row['score_J'], expected_score)),
        # The one declared tolerance: see `REWARD_IDENTITY_TOLERANCE`.
        ('the reward sum within the declared invariant-10 bound of the frozen score',
         abs(row['reward_sum'] - expected_score) <= REWARD_IDENTITY_TOLERANCE),
    ) if not holds]

    submissions = row.get('submissions')
    events = diagnostics.get('fix_events')
    identities = [
        ('the diagnostics schema', diagnostics.get('schema') == DIAGNOSTICS_SCHEMA),
        # A row is validated twice: in memory, where these are tuples, and after JSON, where they are lists.
        # The identity is about the counts, so the container type must not decide it.
        ('matched report indices against tp_spatial',
         isinstance(components['matched_report_indices'], (list, tuple))
         and len(components['matched_report_indices']) == components['tp_spatial']),
        ('tp + fp + duplicates against the submitted reports',
         isinstance(submissions, (list, tuple))
         and components['tp_spatial'] + components['fp_spatial'] + components['duplicates'] == len(submissions)),
        # Each timing weight is in [0, 1] and IEEE rounding is monotonic, so these hold with no slack at all.
        ('tp_timely never exceeds tp_spatial', components['tp_timely'] <= components['tp_spatial']),
        ('tp_timely is non-negative', components['tp_timely'] >= 0.),
        ('return_ok against the termination',
         row.get('return_ok') == (row.get('termination_reason') == 'mission_complete'
                                  and bool(row.get('returned')) and bool(row.get('landed')))),
        ('fix attempts against the fix events',
         isinstance(events, (list, tuple))
         and row.get('fixes_attempted') == len(events) == diagnostics.get('fix_attempts')),
        ('accepted fixes against the fix events',
         isinstance(events, (list, tuple))
         and row.get('fixes_accepted') == diagnostics.get('fix_accepted')
         == sum(1 for event in events if event.get('accepted'))),
        ('submitted reports against the diagnostics',
         isinstance(submissions, (list, tuple)) and len(submissions) == diagnostics.get('reports_submitted')
         == len(diagnostics.get('submission_times_s') or [])),
        ('elapsed time against the diagnostics, exactly', _same(row['elapsed_s'], diagnostics.get('elapsed_s'))),
        ('decisions against the diagnostics', row['decisions'] == diagnostics.get('decisions')),
    ]
    for name, holds in identities:
        if not holds:
            found.append(f'recomputed identity fails: {name}')
    if isinstance(submitted, (list, tuple)) and any(i >= len(submitted) for i in indices):
        found.append(f'components.matched_report_indices points past the {len(submitted)} reports submitted')
    found += _declared_integer_problems(row)
    if found:
        return found
    found += _bound_count_problems(row)
    return found + diagnostic_problems(row, manifest)


def _bound_count_problems(row):
    """review 20: a correctly typed count can still be false. Bind each one to what produced it.

    Every identity here is one the implementation already guarantees, checked against the 22 genuine pilot
    rows before it was written down - not a rule invented to make a mutation fail.
    """
    found = []
    bank, components, diagnostics = row['bank'], row['components'], row['diagnostics']
    if bank['frames'] != bank['bank_frames'] + bank['off_node_frames'] + bank['repeated_capture_frames']:
        found.append(f'bank.frames is {bank["frames"]} but its three parts sum to '
                     f'{bank["bank_frames"] + bank["off_node_frames"] + bank["repeated_capture_frames"]}')
    # The reference set is the cell's: every target is either matched spatially or unresolved. A no-human
    # cell has no target at all, and the runner refuses a cell whose evaluator disagrees with its presence.
    if row['presence'] == 'no_human':
        if components['unresolved'] or components['tp_spatial']:
            found.append(f'a no_human cell resolved {components["tp_spatial"]} and left '
                         f'{components["unresolved"]} unresolved; it has no target')
    elif components['unresolved'] + components['tp_spatial'] != row['n_ref']:
        found.append(f'unresolved + tp_spatial is {components["unresolved"] + components["tp_spatial"]} '
                     f'but the cell declares {row["n_ref"]} references')
    expected_unsubmitted = max(diagnostics['tracks_seen'] - diagnostics['reports_submitted'], 0)
    if diagnostics['unsubmitted_detections'] != expected_unsubmitted:
        found.append(f'unsubmitted_detections is {diagnostics["unsubmitted_detections"]} but the tracks seen '
                     f'and the reports submitted give {expected_unsubmitted}')
    return found


#: The exact record `GeoreferenceService.attempt_fix` leaves for one measurement opportunity.
FIX_EVENT_FIELDS = ('accepted', 'anchor_id', 'available', 'sim_time_s')


def _fix_event_problems(row):
    """Every fix event is the one measurement of one RELOCALIZE the action evidence says was flown.

    The executive's RELOCALIZE macro flies to the anchor, pays the dwell, and only then calls `attempt_fix`
    exactly once (`executive._macro`), which accepts a measurement exactly when the anchor was available
    (`GeoreferenceService.attempt_fix`). So the ordered fix events are the ordered executed RELOCALIZE actions,
    anchor for anchor, except that an action which TERMINATED the episode may have been cut off before its
    measurement. `anchor_visits` is the set of anchors those events name.
    """
    diagnostics = row['diagnostics']
    events = diagnostics['fix_events']
    found = []
    if not isinstance(events, list):
        return [f'fix_events is not a list ({events!r})']
    for position, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != set(FIX_EVENT_FIELDS):
            found.append(f'fix_events[{position}] is not exactly {list(FIX_EVENT_FIELDS)} ({event!r})')
        elif not (isinstance(event['anchor_id'], str) and isinstance(event['available'], bool)
                  and isinstance(event['accepted'], bool) and _is_finite(event['sim_time_s'])):
            found.append(f'fix_events[{position}] is not a typed measurement record ({event!r})')
        elif event['accepted'] != event['available']:
            found.append(f'fix_events[{position}]: a measurement is accepted exactly when its anchor is '
                         f'available, and this one says accepted={event["accepted"]}, '
                         f'available={event["available"]}')
    if found:
        return found
    times = [event['sim_time_s'] for event in events]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        found.append('fix_events are not in strictly increasing time: each one ends its own dwell')
    if times and not times[-1] <= row['elapsed_s']:
        found.append(f'a fix event at {times[-1]} s is later than the episode, which ended at {row["elapsed_s"]} s')
    flown = [entry for entry in diagnostics['action_evidence']
             if entry['action_type'] == ActionType.RELOCALIZE.value and entry['status'] == 'executed']
    completed = [entry['action']['anchor_id'] for entry in flown if not entry['terminated']]
    cut_off = [entry['action']['anchor_id'] for entry in flown if entry['terminated']]
    anchors = [event['anchor_id'] for event in events]
    if not (anchors == completed or (cut_off and anchors == completed + cut_off)):
        found.append(f'the fix events name anchors {anchors} but the executed RELOCALIZE actions flew to '
                     f'{completed + cut_off}; every completed relocalisation records exactly one measurement')
    if not _same(diagnostics['anchor_visits'], sorted(set(anchors))):
        found.append(f'anchor_visits is {diagnostics["anchor_visits"]!r} but the fix events visit '
                     f'{sorted(set(anchors))}')
    return found


#: The fallbacks the end-batch phase machine can record (`gnss_policies.EndBatchAtHome`), and the phases.
BATCH_FALLBACKS = (None, 'nothing_held', 'home_anchor_unreachable', 'fix_would_buy_nothing',
                   'single_fix_attempt_failed')
BATCH_PHASES = (gnss_policies.BATCH_SEARCHING, gnss_policies.BATCH_FIXING, gnss_policies.BATCH_RELEASING)
#: The keys a fix attempt has when it is issued; `fixes_accepted` joins them only once the outcome was read.
FIX_ATTEMPT_ISSUED_FIELDS = ('accepted', 'anchor_id', 'elapsed_s', 'executed', 'issued', 'reason')


def _batch_state_problems(row):
    """The end-batch phase machine's record, bound to the action evidence and the fix events.

    revision 21's first version matched the attempt's `elapsed_s` to the fix event's time. The bounded smoke
    refused its own genuine episode with it, because they are different instants: `elapsed_s` is the decision
    that ISSUED the RELOCALIZE, and the fix event is stamped at the end of the flight and the dwell, about 30 s
    later. The binding is to the action instead - the policy issues at most one RELOCALIZE in the whole
    episode, so that action, its status and its one measurement are the attempt.

    What remains ordered rather than derived: `elapsed_s` itself. The action evidence carries no decision time,
    so it is bounded by the episode (before the measurement it caused, and within the episode) and no
    statistic reads it.
    """
    diagnostics = row['diagnostics']
    state = diagnostics['final_batch_state']
    if row['comparator'] != gnss_policies.EndBatchAtHome.name:
        return [] if state is None else [f'final_batch_state is written only by end_batch_at_home, not by '
                                         f'{row["comparator"]!r}']
    if not isinstance(state, dict):
        return [f'an end_batch_at_home row has no final_batch_state ({state!r})']
    found = []
    for key, expected in (('behaviour_version', gnss_policies.POLICY_BEHAVIOUR_VERSION),
                          ('fallback_rule', gnss_policies.BATCH_FALLBACK_RULE)):
        if state[key] != expected:
            found.append(f'final_batch_state.{key} is {state[key]!r}, not the frozen {expected!r}')
    if state['phase'] not in BATCH_PHASES:
        found.append(f'final_batch_state.phase {state["phase"]!r} is not one of {BATCH_PHASES}')
    if state['fallback'] not in BATCH_FALLBACKS:
        found.append(f'final_batch_state.fallback {state["fallback"]!r} is not one of {BATCH_FALLBACKS}')
    issued = [entry for entry in diagnostics['action_evidence']
              if entry['action_type'] == ActionType.RELOCALIZE.value and not entry['forced']]
    events = diagnostics['fix_events']
    attempt = state['fix_attempt']
    if attempt is None:
        if issued or events:
            found.append(f'no fix attempt is recorded, but the policy issued {len(issued)} RELOCALIZE and the '
                         f'service recorded {len(events)} measurements')
        # Without an attempt the machine is still searching with no fallback, or it released on one of the
        # three fallbacks that skip the fix; `fixing` is entered only by issuing it.
        releasing = state['phase'] == gnss_policies.BATCH_RELEASING
        if state['phase'] == gnss_policies.BATCH_FIXING or releasing != (state['fallback'] in (
                'nothing_held', 'home_anchor_unreachable', 'fix_would_buy_nothing')):
            found.append(f'with no fix attempt, phase {state["phase"]!r} and fallback {state["fallback"]!r} '
                         'are not a state the phase machine can reach')
        return found
    if found:
        return found
    if len(issued) != 1:
        return [f'the end-batch rule issues exactly one RELOCALIZE per attempt; the action evidence has '
                f'{len(issued)}']
    action = issued[0]
    keys = set(attempt)
    outcome_read = 'fixes_accepted' in keys
    if keys - {'fixes_accepted'} != set(FIX_ATTEMPT_ISSUED_FIELDS):
        return [f'final_batch_state.fix_attempt is not exactly {list(FIX_ATTEMPT_ISSUED_FIELDS)} (+ '
                f'fixes_accepted once read): {sorted(keys)}']
    if attempt['issued'] is not True:
        found.append(f'final_batch_state.fix_attempt.issued is {attempt["issued"]!r}; an attempt is recorded '
                     'when it is issued')
    if attempt['anchor_id'] != action['action']['anchor_id']:
        found.append(f'the fix attempt names {attempt["anchor_id"]!r} and the RELOCALIZE flew to '
                     f'{action["action"]["anchor_id"]!r}')
    if attempt['executed'] is not (action['status'] == 'executed'):
        found.append(f'the fix attempt says executed={attempt["executed"]!r} and its RELOCALIZE was '
                     f'{action["status"]}')
    if not _is_finite(attempt['elapsed_s']) or not 0. <= attempt['elapsed_s'] <= row['elapsed_s']:
        found.append(f'the fix attempt was issued at {attempt["elapsed_s"]!r} s, outside the episode')
    if found:
        return found
    if action['status'] != 'executed':
        # Rejected before it ever flew: nothing was measured, and the batch is released on the fallback.
        if events:
            found.append('a rejected RELOCALIZE recorded a measurement')
        if attempt['accepted'] is not False or outcome_read:
            found.append('a rejected fix attempt accepts nothing and reads no outcome')
        if state['fallback'] != 'single_fix_attempt_failed':
            found.append(f'a rejected fix attempt releases on single_fix_attempt_failed, not '
                         f'{state["fallback"]!r}')
        return found
    if attempt['reason'] != action['reason']:
        found.append(f'the fix attempt records reason {attempt["reason"]!r} and its transition {action["reason"]!r}')
    if len(events) != 1:
        return found + [f'one flown attempt makes one measurement; the service recorded {len(events)}']
    event = events[0]
    if not event['sim_time_s'] > attempt['elapsed_s']:
        found.append(f'the measurement at {event["sim_time_s"]} s precedes the decision at '
                     f'{attempt["elapsed_s"]} s that issued it')
    if not outcome_read:
        # The episode ended before the phase machine decided again, so the outcome was never read back.
        if attempt['accepted'] is not None or state['phase'] != gnss_policies.BATCH_FIXING \
                or state['fallback'] is not None:
            found.append('an unread fix attempt has accepted=None, no fallback, and leaves the machine in fixing')
        return found
    if attempt['accepted'] is not event['accepted']:
        found.append(f'the fix attempt says accepted={attempt["accepted"]!r} and its measurement says '
                     f'{event["accepted"]!r}')
    if not _is_count(attempt['fixes_accepted']) or attempt['fixes_accepted'] != int(event['accepted']):
        found.append(f'the fix attempt claims {attempt["fixes_accepted"]!r} accepted fixes and the '
                     f'measurements record {int(event["accepted"])}')
    if state['phase'] != gnss_policies.BATCH_RELEASING:
        found.append(f'once the attempt is read the machine releases; it is in {state["phase"]!r}')
    expected_fallback = None if event['accepted'] else 'single_fix_attempt_failed'
    if state['fallback'] != expected_fallback:
        found.append(f'the fallback is {state["fallback"]!r} and the one fix was '
                     f'{"accepted" if event["accepted"] else "refused"}, which releases on '
                     f'{expected_fallback!r}')
    return found


#: EVERY integer leaf a persisted row is allowed to have, as a path. `[]` is every element of a list and `*`
#: is every value of a mapping. review 18 asked for the named examples and I closed those; review 19
#: showed the rest were still open - `bank.frames`, `bank.repeated_capture_frames = false`,
#: `policy_parameters.max_fix_attempts_per_report = true` - and asked for an audit rather than more examples.
#: This IS the audit: `_declared_integer_problems` proves every path here is a strict integer, and a test
#: walks a genuine row and refuses any integer leaf that is not listed, so a new count cannot arrive unnoticed.
INTEGER_PATHS = (
    'candidate_index', 'cell_index', 'realization', 'seed', 'reobservation_seed', 'n_ref',
    'decisions', 'rejections', 'fixes_attempted', 'fixes_accepted',
    'score_spec.n_ref',
    'components.tp_spatial', 'components.fp_spatial', 'components.duplicates', 'components.unresolved',
    'components.matched_report_indices[]',
    'submissions[].report_index',
    'bank.frames', 'bank.bank_frames', 'bank.off_node_frames', 'bank.repeated_capture_frames',
    'diagnostics.decisions', 'diagnostics.rejections', 'diagnostics.valid_decisions',
    'diagnostics.invalid_decisions', 'diagnostics.policy_decisions', 'diagnostics.forced_return_count',
    'diagnostics.fix_attempts', 'diagnostics.fix_accepted', 'diagnostics.reports_submitted',
    'diagnostics.tracks_seen', 'diagnostics.threshold_ready_tracks', 'diagnostics.open_reportable_tracks',
    'diagnostics.unsubmitted_detections',
    'diagnostics.action_counts.*', 'diagnostics.action_evidence[].index',
    'diagnostics.report_thresholds.min_detections', 'diagnostics.report_thresholds.min_views',
    'diagnostics.evidence_witness.tracks[].detections',
    'diagnostics.policy_parameters.max_fix_attempts_per_report',
    # Only an `end_batch_at_home` row that fixed at home has this one, which is precisely why the audit walks
    # every genuine row rather than one: I declared the list from a single row and this leaf was missing.
    'diagnostics.final_batch_state.fix_attempt.fixes_accepted',
)


#: The COMPLETE parameter record each comparator reports, as a function of the frozen candidate parameters.
#: review 20: `diagnostics.policy_parameters` was checked only against the tuned subset, so
#: `behaviour_version` accepted any string and a field could simply be deleted. The campaign freezes this
#: record per candidate at build time and the row must repeat it exactly.
#:
#: Why it lives HERE and not in `gnss_policies`: that file's SHA-256 is the comparator protocol's policy
#: binding (`comparator_protocol.verify_policy_binding`), so editing it - even to refactor - invalidates the
#: frozen protocol and every campaign bound to it. The anti-drift mechanism is therefore a test rather than a
#: shared call: `test_the_frozen_policy_record_is_what_a_real_policy_reports` constructs every frozen
#: candidate for real and requires its `.parameters` to equal what this returns.
POLICY_PARAMETER_DEFAULTS = {'max_fix_attempts_per_report': 1}
POLICY_PARAMETER_EXTRAS = {
    'periodic_fix': {'stale_after_s': 300.},
    'uncertainty_fix': {'sigma_threshold_m': 1.5},
    'short_horizon_planner': {'measurement_sigma_m': 0.5, 'report_dwell_s': 6., 'tolerance_m': 3.},
    'end_batch_at_home': {'fallback_rule': gnss_policies.BATCH_FALLBACK_RULE,
                          'behaviour_version': gnss_policies.POLICY_BEHAVIOUR_VERSION},
}


def effective_policy_parameters(name, parameters):
    """What the named comparator will report as `parameters`, given its frozen candidate parameters."""
    if name not in gnss_policies.POLICY_CLASSES:
        raise CampaignError(f'unknown comparator {name!r}')
    record = dict(POLICY_PARAMETER_DEFAULTS)
    for key, default in POLICY_PARAMETER_EXTRAS.get(name, {}).items():
        record[key] = default
    for key, value in (parameters or {}).items():
        if key not in record:
            raise CampaignError(f'{name} is not parameterised by {key!r}')
        record[key] = value
    record['max_fix_attempts_per_report'] = int(record['max_fix_attempts_per_report'])
    return record


#: THE ROW SCHEMA: every node a persisted row may contain, by path, with its exact type. `[]` is every element
#: of a list. review 20: `_at()` yields nothing for a missing path, so a declared integer check was silent
#: when the field was deleted, and the whole `bank` object could go while the artifact still validated.
#: revision 21's first answer was a list of required paths; the mutation sweep over 72 genuine episodes then
#: showed a list is still a habit - 83 paths accepted a deletion, an addition, a retype or a new value. So the
#: schema is CLOSED instead: `_schema_problems` walks the whole row, every node must be declared here and have
#: exactly this type, every object must have exactly the keys `OBJECT_KEYS` gives it, and a node that is not
#: declared is itself a refusal. A test walks every genuine smoke row and fails on any node not declared.
#:
#: Types: `int` is a strict non-negative integer (never a bool, never a float); `float` is a finite float
#: (never an int, so the canonical bytes are the runner's); `bool`, `str`, `null`, `list`, `object`;
#: `sha256` is 64 lowercase hex characters; `a|b` admits either. Three nodes are not walked, because a
#: stronger check owns them: `contract` (the action payload, rebuilt as a genuine `contracts.Action` and
#: compared by canonical hash), `frozen` (a parameter record compared by canonical hash with the frozen job),
#: and `counts` (a table of strict counts compared exactly with the one recomputed from the action evidence).
ROW_SCHEMA = {
    '': 'object',
    'bank': 'object', 'bank.bank_frames': 'int', 'bank.frames': 'int', 'bank.off_node_frames': 'int',
    'bank.repeated_capture_frames': 'int',
    'bank_manifest_sha256': 'list', 'bank_manifest_sha256[]': 'sha256',
    'budget_name': 'str', 'candidate_index': 'int', 'candidate_role': 'str', 'candidate_sha256': 'sha256',
    'cell_id': 'str', 'cell_index': 'int', 'cell_weight': 'float', 'classification': 'str', 'comparator': 'str',
    'comparator_parameters': 'frozen', 'comparator_protocol_sha256': 'sha256',
    'components': 'object', 'components.duplicates': 'int', 'components.fp_spatial': 'int',
    'components.matched_report_indices': 'list', 'components.matched_report_indices[]': 'int',
    'components.potential': 'float', 'components.tp_spatial': 'int', 'components.tp_timely': 'float',
    'components.unresolved': 'int',
    'crn_label': 'str', 'crn_sha256': 'sha256', 'crn_version': 'str', 'd600_rms_m': 'float', 'decisions': 'int',
    'diagnostics': 'object',
    'diagnostics.action_counts': 'counts',
    'diagnostics.action_evidence': 'list', 'diagnostics.action_evidence[]': 'object',
    'diagnostics.action_evidence[].action': 'contract', 'diagnostics.action_evidence[].action_type': 'str',
    'diagnostics.action_evidence[].forced': 'null|str', 'diagnostics.action_evidence[].index': 'int',
    'diagnostics.action_evidence[].reason': 'null|str', 'diagnostics.action_evidence[].receipts': 'list',
    'diagnostics.action_evidence[].receipts[]': 'object',
    'diagnostics.action_evidence[].receipts[].local_position_enu': 'list',
    'diagnostics.action_evidence[].receipts[].local_position_enu[]': 'float',
    'diagnostics.action_evidence[].receipts[].sim_time_s': 'float',
    'diagnostics.action_evidence[].receipts[].track_id': 'str',
    'diagnostics.action_evidence[].status': 'str',
    # A rejected action has no transition, so it has no validity and no termination.
    'diagnostics.action_evidence[].terminated': 'bool|null', 'diagnostics.action_evidence[].valid': 'bool|null',
    'diagnostics.anchor_visits': 'list', 'diagnostics.anchor_visits[]': 'str',
    'diagnostics.collision_or_contact': 'object', 'diagnostics.collision_or_contact.reason': 'str',
    'diagnostics.collision_or_contact.value': 'null',
    'diagnostics.decisions': 'int', 'diagnostics.elapsed_s': 'float', 'diagnostics.estimated_distance_m': 'float',
    'diagnostics.evidence_witness': 'object',
    'diagnostics.evidence_witness.final_public_receipts': 'list',
    'diagnostics.evidence_witness.final_public_receipts[]': 'object',
    'diagnostics.evidence_witness.final_public_receipts[].dedup_position_xy': 'list',
    'diagnostics.evidence_witness.final_public_receipts[].dedup_position_xy[]': 'float',
    'diagnostics.evidence_witness.final_public_receipts[].track_id': 'str',
    'diagnostics.evidence_witness.rule': 'str',
    'diagnostics.evidence_witness.tracks': 'list', 'diagnostics.evidence_witness.tracks[]': 'object',
    'diagnostics.evidence_witness.tracks[].credited_views_xy': 'list',
    'diagnostics.evidence_witness.tracks[].credited_views_xy[]': 'list',
    'diagnostics.evidence_witness.tracks[].credited_views_xy[][]': 'float',
    'diagnostics.evidence_witness.tracks[].detections': 'int',
    # A track the last observation did not show has no position there (`evidence_witness`).
    'diagnostics.evidence_witness.tracks[].position_enu': 'list|null',
    'diagnostics.evidence_witness.tracks[].position_enu[]': 'float',
    'diagnostics.evidence_witness.tracks[].track_id': 'str',
    'diagnostics.final_batch_state': 'null|object',
    'diagnostics.final_batch_state.behaviour_version': 'str', 'diagnostics.final_batch_state.fallback': 'null|str',
    'diagnostics.final_batch_state.fallback_rule': 'str', 'diagnostics.final_batch_state.fix_attempt': 'null|object',
    # Unread (the episode ended first) leaves `accepted` null; a rejection records the executive's reason.
    'diagnostics.final_batch_state.fix_attempt.accepted': 'bool|null',
    'diagnostics.final_batch_state.fix_attempt.anchor_id': 'str',
    'diagnostics.final_batch_state.fix_attempt.elapsed_s': 'float',
    'diagnostics.final_batch_state.fix_attempt.executed': 'bool',
    'diagnostics.final_batch_state.fix_attempt.fixes_accepted': 'int',
    'diagnostics.final_batch_state.fix_attempt.issued': 'bool',
    'diagnostics.final_batch_state.fix_attempt.reason': 'null|str',
    'diagnostics.final_batch_state.phase': 'str',
    'diagnostics.first_submission_s': 'float|null', 'diagnostics.fix_accepted': 'int',
    'diagnostics.fix_attempts': 'int',
    'diagnostics.fix_events': 'list', 'diagnostics.fix_events[]': 'object',
    'diagnostics.fix_events[].accepted': 'bool', 'diagnostics.fix_events[].anchor_id': 'str',
    'diagnostics.fix_events[].available': 'bool', 'diagnostics.fix_events[].sim_time_s': 'float',
    'diagnostics.forced_return_count': 'int', 'diagnostics.forced_return_reason': 'null|str',
    'diagnostics.invalid_decision_rate': 'float|null', 'diagnostics.invalid_decisions': 'int',
    'diagnostics.last_submission_s': 'float|null',
    'diagnostics.open_reportable_track_ids': 'list', 'diagnostics.open_reportable_track_ids[]': 'str',
    'diagnostics.open_reportable_tracks': 'int', 'diagnostics.policy_decisions': 'int',
    'diagnostics.policy_parameters': 'frozen', 'diagnostics.rejections': 'int',
    'diagnostics.report_thresholds': 'object', 'diagnostics.report_thresholds.merge_radius_m': 'float',
    'diagnostics.report_thresholds.min_detections': 'int', 'diagnostics.report_thresholds.min_views': 'int',
    'diagnostics.report_thresholds.rule': 'str', 'diagnostics.report_thresholds.view_separation_m': 'float',
    'diagnostics.reports_submitted': 'int', 'diagnostics.schema': 'str',
    'diagnostics.submission_times_s': 'list', 'diagnostics.submission_times_s[]': 'float',
    'diagnostics.submitted_track_ids': 'list', 'diagnostics.submitted_track_ids[]': 'str',
    'diagnostics.threshold_ready_track_ids': 'list', 'diagnostics.threshold_ready_track_ids[]': 'str',
    'diagnostics.threshold_ready_tracks': 'int', 'diagnostics.tracks_seen': 'int',
    'diagnostics.unsubmitted_detections': 'int', 'diagnostics.valid_decisions': 'int',
    'disturbance_path_key': 'str', 'disturbance_path_sha256': 'sha256', 'elapsed_s': 'float', 'episode_id': 'str',
    'execution_manifest_sha256': 'sha256', 'fixes_accepted': 'int', 'fixes_attempted': 'int',
    'frozen_manifest_sha256': 'sha256', 'georeference_sha256': 'sha256', 'landed': 'bool',
    'measurement_noise_key': 'str', 'measurement_noise_sha256': 'sha256', 'mission_budget_s_applied': 'float',
    'mission_budget_s_requested': 'float', 'n_ref': 'int', 'pair_id': 'str', 'presence': 'str',
    'protocol_path': 'str', 'protocol_sha256': 'sha256',
    'public_receipt_types': 'list', 'public_receipt_types[]': 'str',
    'public_reports': 'list', 'public_reports[]': 'object', 'public_reports[].local_position_enu': 'list',
    'public_reports[].local_position_enu[]': 'float', 'public_reports[].track_id': 'str',
    'realization': 'int', 'rejections': 'int', 'reobservation_seed': 'int', 'return_ok': 'bool', 'returned': 'bool',
    'reward_identity_holds': 'bool', 'reward_sum': 'float', 'runner': 'str', 'score_J': 'float',
    'score_spec': 'object', 'score_spec.false_report_weight': 'float', 'score_spec.metric': 'str',
    'score_spec.mission_budget_s': 'float', 'score_spec.n_ref': 'int', 'score_spec.provenance': 'str',
    'score_spec.reward_version': 'str', 'score_spec.score_version': 'str', 'score_spec.timing_weight': 'str',
    'score_spec.tolerance_m': 'float',
    'seed': 'int', 'seed_namespace': 'str',
    'submissions': 'list', 'submissions[]': 'object', 'submissions[].bias_enu': 'list',
    'submissions[].bias_enu[]': 'float', 'submissions[].correction_enu': 'list',
    'submissions[].correction_enu[]': 'float', 'submissions[].global_position_enu': 'list',
    'submissions[].global_position_enu[]': 'float', 'submissions[].local_position_enu': 'list',
    'submissions[].local_position_enu[]': 'float', 'submissions[].report_index': 'int',
    'submissions[].residual_enu': 'list', 'submissions[].residual_enu[]': 'float',
    'submissions[].residual_norm_m': 'float', 'submissions[].sim_time_s': 'float', 'submissions[].track_id': 'str',
    'termination_reason': 'str', 'wall_s': 'float',
    # `--log` in the pilot namespace only (`run_p3_cells.run_cell_episode`); refused anywhere else.
    'streams': 'str',
}
#: The exact key set of every object the schema walks, as (required, optional). An absent required key and an
#: undeclared key are both refusals, so absence can no longer silence a check further down.
OBJECT_KEYS = {
    path: (tuple(sorted(child.rsplit('.', 1)[-1] for child in ROW_SCHEMA
                        if child.rsplit('.', 1)[0] == path and child.count('.') == path.count('.') + 1
                        and not child.endswith('[]') and child != path)), ())
    for path, kind in ROW_SCHEMA.items() if 'object' in kind.split('|') and path
}
OBJECT_KEYS[''] = (tuple(sorted(key for key in ROW_SCHEMA if key and '.' not in key and '[' not in key
                                and key != 'streams')), ('streams',))
#: Only a fix attempt whose outcome was read back carries `fixes_accepted` (`EndBatchAtHome._record_fix_outcome`).
OBJECT_KEYS['diagnostics.final_batch_state.fix_attempt'] = (
    tuple(k for k in OBJECT_KEYS['diagnostics.final_batch_state.fix_attempt'][0] if k != 'fixes_accepted'),
    ('fixes_accepted',))
_SHA256 = re.compile(r'[0-9a-f]{64}')
#: The nodes a row records FIRST-HAND - the primitives every other field is derived from - and why no check
#: inside the row can decide them. revision 21 swept every node of 72 genuine episodes with a deletion, a
#: retype, a +1, a changed string, a reordering and an addition: every mutation that still validates is on a
#: path listed here, and `test_every_accepted_mutation_of_a_genuine_row_is_a_declared_primitive` keeps it so.
#: Everything NOT listed is either frozen (compared by canonical bytes with the job, the CRN group or a runner
#: constant) or derived (recomputed from these). A coherent edit of several primitives together cannot be
#: refused from inside the row at all; what binds them is that an episode is a deterministic function of the
#: code tree, the candidate, the cell and the seed, so re-running the unit reproduces the row.
PRIMITIVE_LEAVES = {
    # The three decision identities. revision 22's per-form sweep found the INSPECT_VIEW pair: H21's sweep read
    # only the first occurrence of `action.track_id`, which was a REPORT's - and a REPORT's track and coordinate
    # ARE bound to its receipt and submission (every probe of them is refused). An inspection's track and view,
    # and a search's sector, are what the policy chose; the payload is rebuilt as a genuine contracts.Action,
    # but a choice that changed no derived quantity is simply the decision.
    'diagnostics.action_evidence[].action.sector_id': 'the sector a SEARCH_SECTOR action chose',
    'diagnostics.action_evidence[].action.track_id':
        'the track an INSPECT_VIEW action chose (a REPORT track_id is bound to its receipt and is refused)',
    'diagnostics.action_evidence[].action.view_id': 'the view an INSPECT_VIEW action chose',
    'diagnostics.estimated_distance_m': "the executive's odometry of the flown path; recorded, bounded >= 0",
    'diagnostics.evidence_witness.tracks[]':
        "the tracker's state at the last decision (detections, credited view positions, position, id). Both "
        'the ready and the open set are recomputed from it, so an edit that moves either is refused; an edit '
        'that moves neither changes no other persisted field',
    'diagnostics.fix_events[].sim_time_s':
        'when a measurement completed: strictly increasing, within the episode, one per completed RELOCALIZE',
    'diagnostics.final_batch_state.fix_attempt.elapsed_s':
        'the decision time that issued the one end-batch attempt: within the episode and before the '
        'measurement it caused. The action evidence carries no decision time to derive it from',
    'wall_s': 'host wall-clock time of the episode; not evidence of anything, bounded >= 0',
    # revision 22: the adjacent-float sweep found this one. dhat(t) is the private filter's mean at the
    # submission; the row binds it EXACTLY through residual = bias - correction, but a one-ULP change of a
    # correction that is zero (no fix yet) or much smaller than the bias is absorbed by the rounding of that
    # subtraction, so it moves no other stored value. Re-running the unit reproduces it; nothing inside the row
    # can. (The bias is first-hand too, but every probe of it moved the residual and was refused.)
    'submissions[].correction_enu':
        'the private filter estimate dhat(t) at a submission, bound exactly by residual = bias - correction; '
        'a change that the subtraction rounds away changes no other stored field',
}
#: The raw wrapper has one additional first-hand measurement. Its remaining fields are frozen campaign/unit
#: identities or canonical projections of the two rows and are all recomputed by ``validate_raw_artifact``.
RAW_PRIMITIVE_LEAVES = {
    'wall_s': 'host wall-clock time of the paired unit; not evidence of an outcome, bounded >= 0',
}


def primitive_path(path):
    """The declared primitive a generalized node path belongs to, or None."""
    return next((root for root in PRIMITIVE_LEAVES if path == root or path.startswith(root + '.')
                 or path.startswith(root + '[')), None)




def _is_kind(value, kind):
    """One node against one declared type.

    A row is validated twice - in memory at commit, where the runner's sequences can be tuples and a float can
    be a numpy float64, and after JSON, where they are lists and floats - and both must give the same answer
    (`test_a_row_is_validated_the_same_in_memory_and_after_json`). So a sequence is a list OR a tuple and a
    float is any float subclass; both serialise to the same canonical bytes. What stays strict is what JSON
    does NOT normalise: an int is never a float or a bool, a float is never an int, a bool is never an int.
    """
    if kind == 'int':
        return _is_count(value)
    if kind == 'float':
        return isinstance(value, float) and math.isfinite(value)
    if kind == 'bool':
        return type(value) is bool
    if kind == 'str':
        return isinstance(value, str)
    if kind == 'sha256':
        return isinstance(value, str) and _SHA256.fullmatch(value) is not None
    if kind == 'null':
        return value is None
    if kind == 'list':
        return isinstance(value, (list, tuple))
    if kind in ('object', 'contract', 'frozen'):
        return isinstance(value, dict)
    if kind == 'counts':
        return isinstance(value, dict) and all(isinstance(k, str) and _is_count(v) for k, v in value.items())
    raise CampaignError(f'the schema names an unknown type {kind!r}')


def _schema_problems(row):
    """Walk the whole row against `ROW_SCHEMA` and `OBJECT_KEYS`. Nothing undeclared, nothing missing."""
    found = []

    def visit(node, path):
        kind = ROW_SCHEMA.get(path)
        where = path or 'the row'
        if kind is None:
            found.append(f'{where} is not a field of this schema')
            return
        if not any(_is_kind(node, option) for option in kind.split('|')):
            if kind.split('|')[0] == 'int':
                found.append(f'{where} is not a non-negative integer ({node!r})')
            elif kind == 'counts' and isinstance(node, dict):
                found.extend([f'{where}.* is not a non-negative integer ({value!r})'
                              for value in node.values() if not _is_count(value)]
                             or [f'{where} has a non-text key'])
            else:
                found.append(f'{where} is {type(node).__name__} {str(node)[:60]!r}, not {kind}')
            return
        if kind in ('contract', 'frozen', 'counts'):
            return
        if isinstance(node, dict):
            required, optional = OBJECT_KEYS[path]
            missing = sorted(set(required) - set(node))
            extra = sorted(set(node) - set(required) - set(optional))
            if missing:
                found.append(f'{where} is missing {missing}, so nothing about them could be checked')
            if extra:
                found.append(f'{where} carries {extra}, which this schema does not declare')
            for key in sorted(set(node) & (set(required) | set(optional))):
                visit(node[key], f'{path}.{key}' if path else key)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item, f'{path}[]')

    visit(row, '')
    if 'streams' in (row if isinstance(row, dict) else {}) and row.get('seed_namespace') != 'pilot':
        found.append('streams is written only by a pilot run with --log')
    return found


def _at(node, parts):
    """Every value a path reaches. A missing key yields nothing: presence is somebody else's check."""
    if not parts:
        yield node
        return
    head, rest = parts[0], parts[1:]
    if head == '[]':
        if isinstance(node, (list, tuple)):
            for item in node:
                yield from _at(item, rest)
    elif head == '*':
        if isinstance(node, dict):
            for item in node.values():
                yield from _at(item, rest)
    elif isinstance(node, dict) and head in node:
        yield from _at(node[head], rest)


def _split(path):
    parts = []
    for chunk in path.split('.'):
        while chunk.endswith('[]'):
            chunk = chunk[:-2]
            parts.append(chunk)
            parts.append('[]')
            chunk = ''
        if chunk:
            parts.append(chunk)
    return parts


def _declared_integer_problems(row):
    """Every declared integer leaf is a strict integer. One place, so the audit is a list and not a habit."""
    found = []
    for path in INTEGER_PATHS:
        for value in _at(row, _split(path)):
            if not _is_count(value):
                found.append(f'{path} is not a non-negative integer ({value!r})')
    return found


def _is_count(value):
    """A count is a non-negative int, never a bool and never a float.

    review 16 named the bool: `True` IS an `int`, so `isinstance(value, int)` alone is not a count check.
    Review 17 named the other half: Python equality cannot tell `122` from `122.0` or `0` from `False`, so
    `policy_decisions == 122.0`, `reports_submitted = False` and `min_detections = 4.0` all passed every
    identity below. A count that arrived as a float did not come from counting.
    """
    return type(value) is int and value >= 0


def _is_finite(value):
    """A coordinate is a real, finite number. review 16: `credited_views_xy=[[NaN, 0], [NaN, 0]]` passed
    validation because `math.dist` returns NaN and `NaN <= separation` is false, so the geometric rule silently
    had no opinion; canonical hashing then raised `ValueError` deep inside recovery. Non-finite values are
    rejected BEFORE any geometry or canonicalisation, which is the only place the rejection is cheap."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _position(value, length):
    """EXACTLY `length` finite numbers.

    review 17: this took `len(value) >= length` and looked only at the first two, so the Z of a
    `local_position_enu` could be NaN - through submissions, public receipts and the witness alike - and pass
    raw validation, after which recovery raised an uncaught `ValueError` in canonical hashing. A local ENU
    position is three numbers; a de-duplication XY is two; neither is "at least".
    """
    return isinstance(value, (list, tuple)) and len(value) == length and all(_is_finite(v) for v in value)


#: The three overrides the executive is allowed to force, from `executive._loop`. A forced action is a RETURN,
#: it happens at most once, and it ends the episode.
FORCED_REASONS = ('return_reserve', 'rejection_cap', 'max_decisions')
#: EXACTLY what one entry of the ordered public action evidence says about itself. review 18: `track_id`
#: was not among them, so a genuine executed REPORT could drop its identity and keep its receipt. Review 19:
#: the flat `track_id`, `sector_id`, `view_id` and `anchor_id` projections were redundant copies of the
#: payload and only one of them was ever compared with it, so `sector_id = "X"` beside a REPORT payload that
#: has no sector passed. They are gone: the action IS the payload, and the payload is rebuilt as a genuine
#: `contracts.Action` before anything else is asked of it. The key set is exact, so a new field is a refusal
#: rather than an invisible one.
ACTION_EVIDENCE_FIELDS = ('index', 'action', 'action_type', 'status', 'forced', 'valid', 'terminated',
                          'reason', 'receipts')


def _action_evidence_problems(row):
    """The ordered public action record, and every operational summary recomputed from it.

    review 16 asked for a decision primitive and got one. Review 17 showed it was not enough: the list
    said what each action WAS but nothing about what the executive was allowed to do, so relabelling all 123
    records `forced: return_reserve` - with the summaries moved coherently - turned 122 policy decisions into
    zero, erased ten open reports, and passed raw validation, recovery and completeness. And the whole list
    could simply be deleted while `valid_decisions: 123` stayed behind, because nothing required it to exist.

    The executive's loop is the contract and it is now enforced, not merely transcribed:

      * one entry per `public/actions.jsonl` record, in order, indexed 1..N with no gaps;
      * `policy.act` runs on exactly the entries that were NOT forced;
      * a forced action is a RETURN, uses one of the three frozen reasons, happens **at most once**, and is
        the **last** record, because forcing one is how the episode ends;
      * an executed entry has a transition - valid, terminated and reason come from it - and a rejected one
        has none;
      * receipts are appended only by an executed REPORT, and the concatenation of those appends IS the
        episode's submission list, in order.

    Every count in the row and the diagnostics is then a function of this list, including the ones an editor
    would otherwise leave behind.
    """
    diagnostics = row.get('diagnostics') or {}
    evidence = diagnostics.get('action_evidence')
    if not isinstance(evidence, (list, tuple)) or not evidence:
        return ['action_evidence is missing: the operational record cannot be derived, and a count nobody can '
                'recompute is an assertion']
    found = []
    for position, entry in enumerate(evidence, start=1):
        if not isinstance(entry, dict) or set(entry) != set(ACTION_EVIDENCE_FIELDS):
            found.append(f'action_evidence[{position}] is not exactly an action record '
                         f'({sorted(entry) if isinstance(entry, dict) else entry!r})')
            continue
        where = f'action_evidence[{position}]'
        # review 18: `index = 1.0` passed, because Python equality cannot tell it from 1.
        if not _is_count(entry['index']) or entry['index'] != position:
            found.append(f'{where}: index is {entry["index"]!r}, so the record is out of order or has a gap')
        found += _action_payload_problems(entry, where)
        if entry['status'] not in ('executed', 'rejected'):
            found.append(f'{where}: status is neither executed nor rejected ({entry["status"]!r})')
        if entry['forced'] is not None and entry['forced'] not in FORCED_REASONS:
            found.append(f'{where}: {entry["forced"]!r} is not one of the executive\'s overrides '
                         f'{FORCED_REASONS}')
        executed = entry['status'] == 'executed'
        if executed:
            if not isinstance(entry['valid'], bool) or not isinstance(entry['terminated'], bool):
                found.append(f'{where}: an executed action has a transition, with valid and terminated')
        elif entry['valid'] is not None or entry['terminated'] is not None or entry['receipts']:
            found.append(f'{where}: a rejected action has no transition, no receipts and no outcome')
        if not isinstance(entry['receipts'], (list, tuple)):
            found.append(f'{where}: receipts is not a list ({entry["receipts"]!r})')
        elif executed and entry['valid'] is True and entry['action_type'] == ActionType.REPORT.value:
            # review 18: a submitted report is one executed REPORT macro, and it appends exactly one
            # receipt - not zero, and not a list somebody lengthened.
            if len(entry['receipts']) != 1:
                found.append(f'{where}: a valid executed REPORT appends exactly one public receipt, not '
                             f'{len(entry["receipts"])}')
        elif entry['receipts']:
            found.append(f'{where}: only a valid executed REPORT appends a public receipt')
    if found:
        return found

    forced = [entry for entry in evidence if entry['forced']]
    if len(forced) > 1:
        found.append(f'the executive forces a return once and the episode ends; {len(forced)} are recorded')
    elif forced:
        if forced[0]['action_type'] != 'RETURN':
            found.append(f'a forced action is a RETURN, not {forced[0]["action_type"]!r}')
        if forced[0]['index'] != len(evidence):
            found.append('a forced return is the last action of the episode, and this one is not')
    terminal = [entry for entry in evidence if entry['terminated']]
    if len(terminal) > 1:
        found.append('more than one action terminated the episode')
    elif terminal and terminal[0]['index'] != len(evidence):
        found.append('the terminating action is not the last action of the episode')
    if terminal and terminal[0]['reason'] == 'mission_complete' and not (
            terminal[0]['status'] == 'executed' and terminal[0]['valid'] is True
            and terminal[0]['action_type'] == ActionType.RETURN.value):
        # review 18: a mission completes by flying home, so the action that ends it that way is a valid
        # executed RETURN and nothing else.
        found.append('the episode ended as mission_complete on an action that is not a valid executed RETURN')
    if row.get('return_ok') and not (terminal and terminal[0]['reason'] == 'mission_complete'):
        found.append('return_ok is claimed but no action terminated the episode as mission_complete')
    if terminal and terminal[0]['reason'] != row.get('termination_reason'):
        found.append(f'the episode ended as {terminal[0]["reason"]!r} but the row says '
                     f'{row.get("termination_reason")!r}')
    if found:
        return found

    executed = [entry for entry in evidence if entry['status'] == 'executed']
    counts = {}
    for entry in evidence:
        key = f'{entry["action_type"]}/{entry["status"]}'
        counts[key] = counts.get(key, 0) + 1
    invalid = sum(1 for entry in executed if not entry['valid'])
    derived = {
        'decisions': len(executed),
        'rejections': len(evidence) - len(executed),
        'valid_decisions': len(executed) - invalid,
        'invalid_decisions': invalid,
        'policy_decisions': len(evidence) - len(forced),
        'forced_return_count': len(forced),
    }
    for field in ('decisions', 'rejections'):
        for owner, value in (('the row', row.get(field)), ('the diagnostics', diagnostics.get(field))):
            if not _is_count(value) or value != derived[field]:
                found.append(f'{field} in {owner} is {value!r} but the action evidence says {derived[field]}')
    for field in ('valid_decisions', 'invalid_decisions', 'policy_decisions', 'forced_return_count'):
        value = diagnostics.get(field)
        if not _is_count(value) or value != derived[field]:
            found.append(f'{field} is {value!r} but the action evidence says {derived[field]}')
    rate = (invalid / len(executed)) if executed else None
    stored_rate = diagnostics.get('invalid_decision_rate')
    if rate is None:
        if stored_rate is not None:
            found.append(f'invalid_decision_rate is {stored_rate!r} with no executed decision')
    elif not _same(stored_rate, rate):
        found.append(f'invalid_decision_rate is {stored_rate!r} but the action evidence says {rate}')
    stored_counts = diagnostics.get('action_counts')
    if not isinstance(stored_counts, dict) or any(not _is_count(v) for v in stored_counts.values()):
        found.append(f'action_counts is not a table of counts ({stored_counts!r})')
    elif stored_counts != counts:
        found.append('action_counts disagrees with the action evidence')
    reason = forced[0]['forced'] if forced else None
    if diagnostics.get('forced_return_reason') != reason:
        found.append(f'forced_return_reason is {diagnostics.get("forced_return_reason")!r} but the action '
                     f'evidence says {reason!r}')
    if 'action_sequence' in diagnostics:
        # review 20: it was a projection of the executed actions that only its `action_type` was ever
        # compared on, so its track, sector and anchor identities were free. It is removed rather than
        # derived: `action_evidence` already carries every executed action, in order, with its full payload.
        found.append('action_sequence is not a field of this schema: the executed actions are in '
                     'action_evidence, with their complete payloads')
    return found


def _action_payload_problems(entry, where):
    """Rebuild the persisted action as a genuine `contracts.Action`. The contract is the validator.

    review 18: the comparison of a REPORT's `track_id` with its receipt ran only when the `track_id` was
    not None, and `track_id` was not a required field. Review 19: replacing that with a hand-rolled version of
    the contract was the same mistake one level up - the present keys were computed from a fixed list, so
    `action.bogus_identity = "X"` was invisible; any non-empty string passed where the contract applies an
    identifier grammar, so a track named `" "` passed; and only one of the four flat projections was compared
    with the payload, so `sector_id = "X"` beside a REPORT passed.

    So this does not re-implement anything. It builds the object - unknown key, bad identifier, wrong
    required/optional set, non-finite vector, unknown type, all refused by `Action.__post_init__` - and then
    requires the stored bytes to be that object's own serialisation, so there is nothing left over.
    """
    try:
        action = action_from_dict(entry.get('action'))
    except ContractError as error:
        return [f'{where}: {error}']
    # review 20: `==` on dictionaries is Python numeric equality, so a stored `-3` passed where the
    # reconstruction holds `-3.0`. The claim is exact serialisation, so the comparison is canonical bytes.
    if canonical_sha256(action.to_dict()) != canonical_sha256(entry['action']):
        return [f'{where}: the stored action is not the exact serialisation of the action it describes']
    if entry['action_type'] != entry['action']['action_type']:
        return [f'{where}: action_type {entry["action_type"]!r} is not the payload\'s '
                f'{entry["action"]["action_type"]!r}']
    return []


def _receipt_problems(row):
    """Every report is one executed REPORT transition, and the receipts are its append-only delta.

    review 17: revision 17 claimed `submissions` was anchored because the frozen score constrains
    `tp_spatial + fp_spatial + duplicates == len(submissions)`. That was wrong - **duplicates do not enter the
    score** - so ten fabricated reports could be inserted into the submissions, the public receipts and the
    witness together, `duplicates` moved to 10, the open set emptied, and J stayed at zero. Counting is not
    evidence that a report action happened.

    The evidence that a report action happened is the action record that did it, so the receipt list is now
    rebuilt from the executed REPORT entries: the appended receipts, in order, with their track identities,
    local positions and submission times, and each REPORT's own `track_id` payload must be the track it
    appended. The row's `submissions` must BE that list, and both serialized report lists project from it.
    """
    diagnostics = row.get('diagnostics') or {}
    evidence = diagnostics.get('action_evidence') or []
    submissions = row.get('submissions')
    public = row.get('public_reports')
    witness = (diagnostics.get('evidence_witness') or {})
    witnessed = witness.get('final_public_receipts')
    found = []
    if 'last_observation_reports' in witness:
        # review 17 accepted the end-of-episode reading and asked for the name to say so: track presence
        # and position come from the last observation the policy decided on, report CLOSURE comes from the
        # episode's final receipts. A field named for the observation cannot hold the second thing.
        found.append('last_observation_reports is not a field of this schema: report closure is judged on the '
                     'final public receipts, and the field is named final_public_receipts')

    appended = []
    for entry in evidence:
        if not isinstance(entry, dict):
            continue
        for receipt in entry.get('receipts') or []:
            if not isinstance(receipt, dict) or not isinstance(receipt.get('track_id'), str) \
                    or not _position(receipt.get('local_position_enu'), 3) \
                    or not _is_finite(receipt.get('sim_time_s')):
                found.append(f'action_evidence[{entry.get("index")!r}] appended a receipt that is not one '
                             f'({receipt!r})')
                continue
            payload = entry.get('action') or {}
            if payload.get('track_id') != receipt['track_id']:
                found.append(f'action_evidence[{entry["index"]}] reported {payload.get("track_id")!r} but '
                             f'appended a receipt for {receipt["track_id"]!r}')
            # review 20: the coordinate the REPORT chose and the coordinate the receipt carries were
            # two unbound numbers, so a genuine action could claim [999, 998, 997] for a receipt it did not
            # move. A report submits the coordinate it names - the executive builds the receipt FROM
            # `action.report_position_enu` - so the two are compared as canonical bytes, not to 9 decimals:
            # all 138 genuine receipts of smokes v15 and v16 are bit-identical to their actions.
            elif canonical_sha256(payload['report_position_enu']) != \
                    canonical_sha256(receipt['local_position_enu']):
                found.append(f'action_evidence[{entry["index"]}] reported {payload["report_position_enu"]!r} '
                             f'and appended a receipt at {receipt["local_position_enu"]!r}')
            appended.append(receipt)
    for name, entries, key, length in (('submissions', submissions, 'local_position_enu', 3),
                                       ('public_reports', public, 'local_position_enu', 3),
                                       ('final_public_receipts', witnessed, 'dedup_position_xy', 2)):
        if not isinstance(entries, (list, tuple)):
            found.append(f'{name} is not a list of reports ({entries!r})')
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get('track_id'), str):
                found.append(f'{name}: a report has no track identity ({entry!r})')
            elif not _position(entry.get(key), length):
                found.append(f'{name}: {entry.get("track_id")!r} has no finite {key} ({entry.get(key)!r})')
    if found:
        return found

    # review 21: these were compared after rounding to nine decimals, so a receipt moved by 1e-10 or a
    # submission moved by 1e-12 described "the same" report and passed every layer. The executive builds the
    # receipt FROM the action and the service freezes the submission FROM the same coordinate and time, so all
    # 265 genuine v18 submissions are bit-identical to their receipts. Canonical typed equality, in order - no
    # rounding before an identity; rounding belongs in presentation only.
    expected = [[r['track_id'], r['local_position_enu'], r['sim_time_s']] for r in appended]
    submitted = [[s['track_id'], s['local_position_enu'], s['sim_time_s']] for s in submissions]
    if not _same(submitted, expected):
        found.append('submissions is not, exactly, the receipt sequence the executed REPORT actions appended')
    if not _same([[r['track_id'], r['local_position_enu']] for r in public],
                 [[entry[0], entry[1]] for entry in expected]):
        found.append('public_reports is not, exactly and in order, the receipts the episode appended')
    if not _same([[r['track_id'], r['dedup_position_xy']] for r in witnessed],
                 [[entry[0], entry[1][:2]] for entry in expected]):
        found.append('final_public_receipts is not, exactly and in order, the XY of the receipts appended')
    return found


def _same(actual, expected):
    """Canonical-byte equality. review 20: Python `==` says `-3 == -3.0` and `1 == True`, so a frozen
    value compared with `!=` accepted a retyped copy. A value that cannot even be canonicalised is not equal."""
    try:
        return canonical_sha256(actual) == canonical_sha256(expected)
    except (TypeError, ValueError):
        return False


#: The one receipt type the fast tier appends (`executive._macro`), named from the contract itself.
LOCAL_RECEIPT_TYPE = 'LocalReportReceipt'


def _derived_field_problems(row, manifest=None):
    """revision 21: every field the row records a second time, derived again from the first time.

    The mutation sweep over 72 genuine episodes found each of these rewritable without a refusal. None of the
    identities below is new arithmetic - each is how the runner or the service builds the field, and each held
    exactly - bit for bit, with the producer's own arithmetic (revision 22) - on all 72 rows:

      * a submission's private record: residual = bias - correction, the global coordinate is the local one
        plus that residual (the runtime gauge is zero, ruling 2), and the norm is the residual's norm;
      * TP_timely is the frozen timing weight summed over the matched reports' own submission times, so the
        matched indices and the submission times are bound to the score, not only counted;
      * the submission times, the first and last of them, the public reports and the witness's final receipts
        are the submissions again, in submission order; the id lists are sorted sets;
      * the witness is one entry per track, sorted by track id, as `evidence_witness` writes it;
      * a valid transition that did not end the episode carries no reason;
      * the rule texts and the no-contact record are the runner's constants.
    """
    diagnostics, submissions = row['diagnostics'], row['submissions']
    components, witness = row['components'], diagnostics['evidence_witness']
    found = []
    for index, submission in enumerate(submissions):
        where = f'submissions[{index}]'
        lengths = {name: len(submission[name]) for name in ('bias_enu', 'correction_enu', 'residual_enu',
                                                            'global_position_enu', 'local_position_enu')}
        if lengths != {'bias_enu': 2, 'correction_enu': 2, 'residual_enu': 2, 'global_position_enu': 3,
                       'local_position_enu': 3}:
            found.append(f'{where} does not have the vector lengths of a submission ({lengths})')
            continue
        # revision 22: the service's own arithmetic, in its own order, compared exactly (review 21).
        # `GeoreferenceService.submit_report`: residual = d(t) - dhat(t); global = (local + gauge) + [r, 0]
        # with the zero runtime gauge of ruling 2; `GlobalSubmission.residual_norm_m` = math.hypot(residual).
        residual = [submission['bias_enu'][k] - submission['correction_enu'][k] for k in range(2)]
        if not _same(submission['residual_enu'], residual):
            found.append(f'{where}.residual_enu is not exactly bias_enu - correction_enu')
        if not _same(submission['residual_norm_m'], math.hypot(*submission['residual_enu'])):
            found.append(f'{where}.residual_norm_m is not exactly math.hypot(residual_enu)')
        local = submission['local_position_enu']
        expected = [(local[0] + 0.) + submission['residual_enu'][0], (local[1] + 0.) + submission['residual_enu'][1],
                    (local[2] + 0.) + 0.]
        if not _same(submission['global_position_enu'], expected):
            found.append(f'{where}.global_position_enu is not exactly (local + zero gauge) + residual')
    times = [submission['sim_time_s'] for submission in submissions]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        found.append('the submissions are not in strictly increasing time: each one pays its own dwell')
    if times and times[-1] > row['elapsed_s']:
        found.append('a report was submitted after the episode ended')
    for name, expected in (('submission_times_s', times),
                           ('first_submission_s', times[0] if times else None),
                           ('last_submission_s', times[-1] if times else None),
                           ('submitted_track_ids', sorted({s['track_id'] for s in submissions})),
                           ('threshold_ready_track_ids', sorted(diagnostics['threshold_ready_track_ids'])),
                           ('open_reportable_track_ids', sorted(diagnostics['open_reportable_track_ids']))):
        if not _same(diagnostics[name], expected):
            found.append(f'{name} is not {"the sorted set" if name.endswith("_ids") else "the submission times"} '
                         f'it is derived from')
    if not _same(row['public_reports'], [{'track_id': s['track_id'], 'local_position_enu': s['local_position_enu']}
                                        for s in submissions]):
        found.append('public_reports is not the submissions, in submission order')
    if not _same(witness['final_public_receipts'],
                 [{'track_id': s['track_id'], 'dedup_position_xy': s['local_position_enu'][:2]}
                  for s in submissions]):
        found.append('final_public_receipts is not the submissions\' local XY, in submission order')
    if not _same(row['public_receipt_types'], [LOCAL_RECEIPT_TYPE] if submissions else []):
        found.append(f'public_receipt_types is {row["public_receipt_types"]!r} for {len(submissions)} receipts')
    matched = components['matched_report_indices']
    if any(later <= earlier for earlier, later in zip(matched, matched[1:])):
        found.append('components.matched_report_indices is not strictly increasing')
    elif all(i < len(submissions) for i in matched):
        # `reward_gnss.components`: the frozen timing weight summed over the matches in match order, which is
        # report order (`evaluation.match_reports` sorts its pairs), then `float()`. Exactly that, compared exactly.
        budget = row['mission_budget_s_applied']
        timely = float(sum(timing_weight(submissions[i]['sim_time_s'], budget) for i in matched))
        if not _same(components['tp_timely'], timely):
            found.append(f'components.tp_timely is {components["tp_timely"]} but the frozen timing weight over '
                         f'the matched reports\' submission times gives {timely}')
    track_ids = [track['track_id'] for track in witness['tracks']]
    if not _same(track_ids, sorted(set(track_ids))):
        found.append('the witness is not one entry per track, sorted by track id')
    for entry in diagnostics['action_evidence']:
        if entry['status'] == 'executed' and entry['valid'] and not entry['terminated'] \
                and entry['reason'] is not None:
            found.append(f'action_evidence[{entry["index"]}] is a valid transition that did not end the episode, '
                         f'and carries reason {entry["reason"]!r}')
    for name, actual, expected in (
            ('evidence_witness.rule', witness['rule'], runner.WITNESS_RULE),
            ('report_thresholds.rule', diagnostics['report_thresholds']['rule'], runner.OPEN_REPORTABLE_RULE),
            ('collision_or_contact', diagnostics['collision_or_contact'], runner.NO_CONTACT_MODEL)):
        if not _same(actual, expected):
            found.append(f'{name} is not the runner\'s {name}')
    if manifest is not None:
        for key in ('min_detections', 'min_views', 'merge_radius_m', 'view_separation_m'):
            if not _same(diagnostics['report_thresholds'][key], manifest['report_thresholds'][key]):
                found.append(f'report_thresholds.{key} is not the frozen {manifest["report_thresholds"][key]!r}')
    for name, value in (('diagnostics.estimated_distance_m', diagnostics['estimated_distance_m']),
                        ('wall_s', row['wall_s'])):
        if value < 0:
            found.append(f'{name} is negative ({value})')
    return found


def diagnostic_problems(row, manifest=None):
    """Bind the operational record to the episode it describes.

    review 13: schema `/2` was emitted and never validated, so the thresholds, the ready and open lists
    and the counts could all be rewritten - turning "ten ready reports lost" into "nothing ready" - without a
    single check failing. A record nobody validates is decoration.
    """
    diagnostics = row.get('diagnostics') or {}
    submissions = row.get('submissions') or []
    found = []
    thresholds = diagnostics.get('report_thresholds') or {}
    if manifest is not None:
        for key in ('min_detections', 'min_views', 'merge_radius_m', 'view_separation_m'):
            expected = (manifest.get('report_thresholds') or {}).get(key)
            if expected is not None and thresholds.get(key) != expected:
                found.append(f'report_thresholds.{key}: {thresholds.get(key)!r} != {expected!r}')
    for key in ('min_detections', 'min_views'):
        # review 17: `4.0` and `2.0` compared equal to the frozen `4` and `2`, so a float passed a
        # threshold that counts detections and views. A threshold on a count is a count.
        if not _is_count(thresholds.get(key)) or thresholds.get(key) <= 0:
            found.append(f'report_thresholds.{key} is not a positive integer threshold '
                         f'({thresholds.get(key)!r})')
    for key in ('merge_radius_m', 'view_separation_m'):
        value = thresholds.get(key)
        if not _is_finite(value) or value <= 0:
            found.append(f'report_thresholds.{key} is not a positive finite distance ({value!r})')

    pairs = (('threshold_ready_track_ids', 'threshold_ready_tracks'),
             ('open_reportable_track_ids', 'open_reportable_tracks'),
             ('submitted_track_ids', None))
    sets = {}
    for ids_key, count_key in pairs:
        ids = diagnostics.get(ids_key)
        if not isinstance(ids, (list, tuple)) or any(not isinstance(i, str) for i in ids):
            found.append(f'{ids_key} is not a list of track identities ({ids!r})')
            continue
        if len(set(ids)) != len(ids):
            found.append(f'{ids_key} repeats a track identity')
        if count_key is not None and diagnostics.get(count_key) != len(ids):
            found.append(f'{count_key} is {diagnostics.get(count_key)!r} but {ids_key} has {len(ids)}')
        sets[ids_key] = set(ids)
    if len(sets) != len(pairs):
        return found

    submitted = {s.get('track_id') for s in submissions if isinstance(s, dict)}
    if sets['submitted_track_ids'] != submitted:
        found.append('submitted_track_ids does not match the submitted reports')
    if not sets['open_reportable_track_ids'] <= sets['threshold_ready_track_ids']:
        found.append('open_reportable_track_ids is not a subset of threshold_ready_track_ids')
    if sets['open_reportable_track_ids'] & sets['submitted_track_ids']:
        found.append('a track cannot be open and submitted at once')

    # review 14: every check above compares one reported summary with another, so rewriting ten ready
    # and ten open tracks to zero passed all of them. Both sets are now RECOMPUTED from the primitive witness.
    witness = diagnostics.get('evidence_witness')
    if witness is None:
        return found + ['evidence_witness is missing: the ready and open sets cannot be recomputed, and a '
                        'number nobody can recompute is an assertion']
    if 'last_observation_present' in witness:
        found.append('last_observation_present is derived from the decision count, not supplied')
    # review 15: the witness could report transmissions the episode never made. Review 16: so could the
    # row. Review 17: so could the action list itself, because nothing tied it to what the executive is
    # allowed to do, and nothing required it to exist at all. Both report lists and every operational count
    # are now rebuilt from the ordered public action evidence, and that evidence is checked against the
    # executive's own invariants. Everything below is DERIVED; nothing here is taken on trust.
    found += _action_evidence_problems(row)
    if found:
        return found
    found += _receipt_problems(row)
    if found:
        return found
    # revision 21: the fix record and the end-batch record are bound to the action evidence, so they are read
    # only once that evidence has been proved - a deleted payload key used to raise here instead of refusing.
    found += _fix_event_problems(row)
    if found:
        return found
    found += _batch_state_problems(row)
    found += _derived_field_problems(row, manifest)
    if found:
        return found
    evidence = diagnostics['action_evidence']
    decisions = len(evidence) - sum(1 for entry in evidence if entry['forced'])

    # The credited views are positions, the frozen separation rule is checked against them, and every value
    # that reaches geometry or canonical hashing is proved finite first (review 16).
    separation = thresholds.get('view_separation_m')
    for entry in witness.get('tracks') or []:
        if not isinstance(entry, dict) or not isinstance(entry.get('track_id'), str):
            found.append(f'the witness carries an entry that is not a track ({entry!r})')
            continue
        track_id = entry['track_id']
        if not _is_count(entry.get('detections')):
            found.append(f'{track_id!r}: detections is not a count ({entry.get("detections")!r})')
        if not (entry.get('position_enu') is None or _position(entry.get('position_enu'), 3)):
            found.append(f'{track_id!r}: position_enu is not a finite ENU position '
                         f'({entry.get("position_enu")!r})')
        views = entry.get('credited_views_xy')
        if not isinstance(views, list) or any(not _position(v, 2) for v in views):
            found.append(f'{track_id!r}: credited_views_xy is not a list of finite XY positions ({views!r})')
            continue
        if separation:
            for i, first in enumerate(views):
                for second in views[i + 1:]:
                    if math.dist(first, second) <= separation:
                        found.append(f'{track_id!r}: two credited views are within '
                                     f'{separation} m of each other, which the evidence rule forbids')
                        break
    if found:
        return found

    try:
        ready, open_reportable = runner.sets_from_witness(witness, thresholds, decisions)
    except (TypeError, KeyError, ValueError) as error:
        return found + [f'the evidence witness is unusable: {error}']
    if sorted(ready) != sorted(diagnostics.get('threshold_ready_track_ids') or []):
        found.append('threshold_ready_track_ids disagrees with the set recomputed from the witness')
    if sorted(open_reportable) != sorted(diagnostics.get('open_reportable_track_ids') or []):
        found.append('open_reportable_track_ids disagrees with the set recomputed from the witness')

    # And the witness is tied to the rest of the row, so a coherent forgery has to move those too.
    witness_ids = {entry.get('track_id') for entry in witness.get('tracks') or []}
    for field in ('tracks_seen', 'reports_submitted', 'threshold_ready_tracks', 'open_reportable_tracks',
                  'fix_attempts', 'fix_accepted', 'unsubmitted_detections'):
        if not _is_count(diagnostics.get(field)):
            found.append(f'{field} is not a count ({diagnostics.get(field)!r})')
    for field in ('fixes_attempted', 'fixes_accepted'):
        if not _is_count(row.get(field)):
            found.append(f'{field} is not a count ({row.get(field)!r})')
    for field in ('tp_spatial', 'fp_spatial', 'duplicates'):
        if not _is_count((row.get('components') or {}).get(field)):
            found.append(f'components.{field} is not a count ({(row.get("components") or {}).get(field)!r})')
    if found:
        return found
    if diagnostics.get('tracks_seen') != len(witness_ids):
        found.append(f'tracks_seen is {diagnostics.get("tracks_seen")!r} but the witness carries '
                     f'{len(witness_ids)} tracks')
    if not sets['submitted_track_ids'] <= witness_ids:
        found.append('a report was submitted for a track the witness never saw')
    return found


#: The executor version a raw artifact records. It lives here, beside the validator that checks it;
#: `p3_execute.EXECUTOR_VERSION` is this constant, so the writer and the checker cannot name two versions.
EXECUTOR_VERSION = 'gnss-p3-executor/6'
#: EXACTLY the keys of a raw paired artifact (a gate artifact adds `GATE_PARENT_FIELDS`). revision 21: the
#: wrapper's `executor`, `namespace`, `classification` and `pairing` were written and never compared, and an
#: added key was invisible.
RAW_FIELDS = ('campaign_manifest_sha256', 'campaign_version', 'classification', 'code_tree_sha256', 'executor',
              'jobs', 'namespace', 'pairing', 'rows', 'unit', 'unit_sha256', 'wall_s')


def raw_header(manifest, extra):
    """The executor's part of a raw artifact: its version, the namespace, the classification and the wall time.

    Only `wall_s` is a measurement; the rest are fixed by this campaign, so they are written here rather than
    taken from the caller, and a caller that names a different value is refused.
    """
    extra = dict(extra or {})
    header = {'executor': EXECUTOR_VERSION, 'namespace': manifest['namespace'],
              'classification': FAILURE_TAXONOMY['mission/outcome']}
    unknown = sorted(set(extra) - set(header) - {'wall_s'})
    if unknown:
        raise CampaignError(f'a raw artifact has no field {unknown}')
    for key, value in header.items():
        if key in extra and extra[key] != value:
            raise CampaignError(f'{key} is {value!r} in this campaign, not {extra[key]!r}')
    wall = extra.get('wall_s')
    if type(wall) is not float or not math.isfinite(wall) or wall < 0:
        raise CampaignError(f'a raw artifact records the wall time its two episodes took, as a finite '
                            f'non-negative float ({wall!r})')
    return dict(header, wall_s=wall)


def validate_raw_artifact(manifest, raw_path):
    """The ONE validator. Commit, recovery, completeness and aggregation all go through it.

    It proves that a file on disk is this campaign's evidence for one of its own units: the name matches the
    payload, the unit fields are exactly the frozen ones, the campaign and code tree are this campaign's, the
    two jobs are exactly the unit's two jobs, both rows pass `validate_row`, and the paired state is
    **recomputed** rather than read from the file.
    """
    raw_path = Path(raw_path)
    name = raw_path.name
    if not (name.startswith('unit_') and name.endswith('.json')):
        raise CampaignError(f'{name}: not a raw paired artifact name')
    unit_sha256 = name[len('unit_'):-len('.json')]
    units = {unit['unit_sha256']: unit for unit in manifest['units']}
    if unit_sha256 not in units:
        raise CampaignError(f'{unit_sha256[:12]}: not a unit of this campaign')
    unit = units[unit_sha256]
    jobs = {job['job_sha256']: job for job in manifest['jobs']}
    try:
        raw = strict_loads(raw_path.read_text())
    except (OSError, ValueError) as error:
        raise CampaignError(f'{unit_sha256[:12]}: unreadable raw artifact: {error}') from error
    if not isinstance(raw, dict):
        raise CampaignError(f'{unit_sha256[:12]}: the raw artifact is not an object')

    problems = []
    header = [
        ('unit_sha256', raw.get('unit_sha256'), unit_sha256),
        ('campaign_version', raw.get('campaign_version'), manifest['campaign_version']),
        ('campaign_manifest_sha256', raw.get('campaign_manifest_sha256'), manifest['manifest_sha256']),
        ('code_tree_sha256', raw.get('code_tree_sha256'), manifest['code_tree_sha256']),
        ('unit', raw.get('unit'), {field: unit[field] for field in UNIT_FIELDS}),
    ]
    # review 20: the header comparison is Python equality, so `unit.realization = 0.0` and
    # `unit.seed = float(seed)` compared equal to the frozen integers. The wrapper's identities are typed
    # here, before they are compared.
    wrapper = raw.get('unit') if isinstance(raw.get('unit'), dict) else {}
    problems += [f'unit.{field} is not a non-negative integer ({wrapper.get(field)!r})'
                 for field in ('realization', 'seed') if not _is_count(wrapper.get(field))]
    # A gate raw artifact repeats the parents, so the evidence itself names the selection it belongs to.
    # review 14: `load()` refused them in a pilot or tuning MANIFEST, but the raw artifact accepted them,
    # where they would read as evidence of a selection the campaign never had.
    if manifest['namespace'] == 'gate':
        header += [(field, raw.get(field), manifest[field]) for field in GATE_PARENT_FIELDS]
    else:
        dressed = [field for field in GATE_PARENT_FIELDS if field in raw]
        if dressed:
            problems.append(f'a {manifest["namespace"]!r} raw artifact carries gate-only parents {dressed}')
    # revision 21: exact keys, and the executor's fixed fields compared - they were written and never read.
    expected_keys = set(RAW_FIELDS) | (set(GATE_PARENT_FIELDS) if manifest['namespace'] == 'gate' else set())
    if set(raw) != expected_keys:
        problems.append(f'the raw artifact is not exactly {sorted(expected_keys)}: missing '
                        f'{sorted(expected_keys - set(raw))}, undeclared {sorted(set(raw) - expected_keys)}')
    header += [('executor', raw.get('executor'), EXECUTOR_VERSION),
               ('namespace', raw.get('namespace'), manifest['namespace']),
               ('classification', raw.get('classification'), FAILURE_TAXONOMY['mission/outcome'])]
    wall = raw.get('wall_s')
    if type(wall) is not float or not math.isfinite(wall) or wall < 0:
        problems.append(f'wall_s is not a finite non-negative float ({wall!r})')
    problems += [f'{key}: {actual!r} != {expected!r}' for key, actual, expected in header
                 if not _same(actual, expected)]
    expected_jobs = {jobs[j]['cell_id']: j for j in unit['jobs']}
    if not _same(raw.get('jobs'), expected_jobs):
        problems.append(f'jobs: {raw.get("jobs")!r} is not the unit\'s exact two-job mapping')
    rows = raw.get('rows')
    if not isinstance(rows, dict) or set(rows) != set(expected_jobs):
        problems.append(f'rows: {sorted(rows) if isinstance(rows, dict) else rows!r} is not '
                        f'{sorted(expected_jobs)}')
    if problems:
        raise CampaignError(f'{unit_sha256[:12]}: ' + '; '.join(problems))

    ordered = [(jobs[expected_jobs[cell_id]], rows[cell_id]) for cell_id in sorted(expected_jobs)]
    for job, row in ordered:
        found = validate_row(manifest, job, row)
        if found:
            raise CampaignError(f'{unit_sha256[:12]} / {job["cell_id"]}: ' + '; '.join(found))
    pairing = runner.paired_state([row for _, row in ordered])      # recomputed, never read from the file
    if pairing['paired'] is not True:
        raise CampaignError(f'{unit_sha256[:12]}: the two members are not a pair -> {pairing}')
    # ... and the stored copy must BE that recomputation (revision 21): it was written and never compared.
    if not _same(raw.get('pairing'), pairing):
        raise CampaignError(f'{unit_sha256[:12]}: the stored pairing is not the one its rows recompute to')

    return {'unit': unit, 'unit_sha256': unit_sha256, 'raw': raw, 'rows': rows, 'jobs': expected_jobs,
            'pairing': pairing, 'path': raw_path,
            'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest()}


def derived_record(manifest, validated, job):
    """The canonical content of one cell record. The single source of truth for writing AND for checking."""
    row = validated['rows'][job['cell_id']]
    return dict(derive_metrics(row),
                job=_identity_of(job), job_sha256=job['job_sha256'], unit_sha256=validated['unit_sha256'],
                campaign_manifest_sha256=manifest['manifest_sha256'],
                code_tree_sha256=manifest['code_tree_sha256'],
                raw_artifact=f'raw/{validated["path"].name}',
                raw_artifact_sha256=validated['raw_sha256'],
                row_sha256=canonical_sha256(row),
                episode_id=row.get('episode_id'),
                termination_reason=row.get('termination_reason'),
                classification='mission/outcome')


# --- the transaction --------------------------------------------------------------------------------------------

def require_frozen_tree(manifest, where):
    """Recompute the live tree HERE and refuse unless it is the frozen one.

    review 12: the executor proved the tree once, before the episode pair, and a change during the first
    episode was still committed under the old hash. Every mutation now proves it for itself, immediately
    before it writes, so the proof cannot be stale by the width of an episode.
    """
    live = code_tree_hash()
    if manifest.get('code_tree_sha256') and live != manifest['code_tree_sha256']:
        raise CampaignError(f'{where}: the live code tree is {str(live)[:12]} but the campaign was frozen '
                            f'against {manifest["code_tree_sha256"][:12]}; a mixture of code trees is not one '
                            f'campaign, so freeze a new campaign directory instead')
    return live


def commit_unit(manifest, campaign_dir, unit_sha256, rows, lease, extra=None):
    """Install the raw paired artifact, then derive both cell records from the artifact as written.

    `lease` is the live campaign lease from `exclusive()`. The live code tree is recomputed here rather than
    supplied, because a caller-supplied tree is a caller's assertion.
    """
    require_lease(lease, campaign_dir)
    units = {unit['unit_sha256']: unit for unit in manifest['units']}
    jobs = {job['job_sha256']: job for job in manifest['jobs']}
    if unit_sha256 not in units:
        raise CampaignError(f'{unit_sha256[:12]} is not a unit in this frozen campaign')
    unit = units[unit_sha256]
    code_tree_sha256 = require_frozen_tree(manifest, unit_sha256[:12])
    if len(rows) != 2:
        raise CampaignError(f'{unit_sha256[:12]}: a unit is two episodes, got {len(rows)}')

    by_cell = {row.get('cell_id'): row for row in rows}
    ordered, problems = [], []
    for job_sha256 in unit['jobs']:
        job = jobs[job_sha256]
        row = by_cell.get(job['cell_id'])
        if row is None:
            problems.append(f'no row for {job["cell_id"]}')
            continue
        ordered.append((job, row))
        problems += [f'{job["cell_id"]}: {p}' for p in validate_row(manifest, job, row)]
    if problems:
        raise CampaignError(f'{unit_sha256[:12]}: ' + '; '.join(problems))
    pairing = runner.paired_state([row for _, row in ordered])
    if pairing['paired'] is not True:
        raise CampaignError(f'{unit_sha256[:12]}: the two members are not a pair -> {pairing}')

    parents = ({field: manifest[field] for field in GATE_PARENT_FIELDS}
               if manifest['namespace'] == 'gate' else {})
    raw_path = paths(campaign_dir)['raw'] / f'unit_{unit_sha256}.json'
    if not raw_path.exists():
        # review 15: the lease and the tree were proved BEFORE row validation and then the raw artifact
        # was installed without proving either again. A lease invalidated or a tree moved from inside
        # validation left a raw artifact behind, which ordinary recovery then accepted. Both are proved here,
        # with nothing between this and the write, and the tree written into the header is THIS proof's value.
        # Review 16: "nothing between this and the write" was still false - serialization, file creation and
        # fsync all happened after it. The same proof now runs again INSIDE the install, as the instruction
        # before `os.link`, so the header's tree is the tree at the moment the artifact became visible.
        require_lease(lease, campaign_dir)
        code_tree_sha256 = require_frozen_tree(manifest, unit_sha256[:12])
        install_evidence(raw_path, dict(
            raw_header(manifest, extra), **parents, campaign_version=CAMPAIGN_VERSION,
            unit={f: unit[f] for f in UNIT_FIELDS},
            unit_sha256=unit_sha256, campaign_manifest_sha256=manifest['manifest_sha256'],
            code_tree_sha256=code_tree_sha256, pairing=pairing,
            rows={job['cell_id']: row for job, row in ordered},
            jobs={job['cell_id']: job['job_sha256'] for job, _ in ordered}),
            manifest, campaign_dir, lease, unit_sha256[:12])
    return _commit_cells_from_raw(manifest, campaign_dir, raw_path, lease)


def _commit_cells_from_raw(manifest, campaign_dir, raw_path, lease):
    """Derive whatever cell records a VALIDATED raw artifact still owes. Never rewrites an existing one.

    review 13: this is a write-capable function, and it had neither a lease nor a tree check of its own.
    A tree that moved between recovery's entry check and this call was still written under the frozen hash.
    Both are proved HERE, immediately before each write.
    """
    require_lease(lease, campaign_dir)
    validated = validate_raw_artifact(manifest, raw_path)
    jobs = {job['job_sha256']: job for job in manifest['jobs']}
    written = []
    for cell_id, job_sha256 in sorted(validated['jobs'].items()):
        path = paths(campaign_dir)['results'] / f'job_{job_sha256}.json'
        if path.exists():
            continue
        require_lease(lease, campaign_dir)
        require_frozen_tree(manifest, job_sha256[:12])
        install_evidence(path, derived_record(manifest, validated, jobs[job_sha256]),
                         manifest, campaign_dir, lease, job_sha256[:12])
        written.append(path)
    return written


def reconcile(manifest, campaign_dir, lease):
    """Repair units whose raw artifact survived but whose cell records did not. Reruns and trusts nothing.

    review 11 made recovery validate what it reads. Review 12 found the remaining hole: the live tree
    was an OPTIONAL argument, so a direct call could simply omit it and mutate anyway. The tree is now
    recomputed inside, and a live campaign lease is required, so there is no argument a caller can leave out.
    """
    require_lease(lease, campaign_dir)
    try:
        require_frozen_tree(manifest, 'recovery')
    except CampaignError as error:
        register(campaign_dir, 'infrastructure/code_tree_drift', message=f'recovery refused: {error}')
        raise CampaignError(f'recovery refused: {error}') from error
    repaired, refused = [], []
    raw_dir = paths(campaign_dir)['raw']
    known = {unit['unit_sha256'] for unit in manifest['units']}
    for raw_path in sorted(raw_dir.glob('unit_*.json')) if raw_dir.exists() else []:
        unit_sha256 = raw_path.stem[len('unit_'):]
        if unit_sha256 not in known:
            continue
        try:
            written = _commit_cells_from_raw(manifest, campaign_dir, raw_path, lease)
        except CampaignError as error:
            refused.append({'unit_sha256': unit_sha256, 'reason': str(error)})
            register(campaign_dir, 'infrastructure/corrupt_evidence', unit_sha256=unit_sha256,
                     message=str(error))
            continue
        if written:
            repaired.append({'unit_sha256': unit_sha256, 'records': [p.name for p in written]})
            register(campaign_dir, 'unit/reconciled', unit_sha256=unit_sha256,
                     records=[p.name for p in written])
    return {'repaired': repaired, 'refused': refused}


# --- completeness ----------------------------------------------------------------------------------------------

def completeness(manifest, campaign_dir):
    """Judge the directory by recomputing it, not by reading what it says about itself."""
    campaign_dir = Path(campaign_dir)
    where = paths(campaign_dir)
    expected = {job['job_sha256']: job for job in manifest['jobs']}
    expected_units = {unit['unit_sha256'] for unit in manifest['units']}
    found, unreadable, mislabeled, tampered, drift, corrupt = {}, [], [], [], [], []

    validated_units, raw_names = {}, set()
    if where['raw'].exists():
        for raw_path in sorted(where['raw'].glob('unit_*.json')):
            raw_names.add(raw_path.name)
            unit_sha256 = raw_path.stem[len('unit_'):]
            if unit_sha256 not in expected_units:
                continue
            try:
                validated_units[unit_sha256] = validate_raw_artifact(manifest, raw_path)
            except CampaignError as error:
                corrupt.append(f'{unit_sha256[:12]}: {error}')

    results_dir = where['results']
    for path in sorted(results_dir.glob('job_*.json')) if results_dir.exists() else []:
        digest = path.stem[len('job_'):]
        found[digest] = path
        if digest not in expected:
            continue
        try:
            payload = strict_loads(path.read_text())
        except (OSError, ValueError) as error:
            unreadable.append(f'{digest[:12]}: {error}')
            continue
        job = expected[digest]
        validated = validated_units.get(job['unit_sha256'])
        if validated is None:
            mislabeled.append(f'{digest[:12]}: its raw paired artifact is missing or corrupt')
            continue
        try:
            expected_record = derived_record(manifest, validated, job)
        except CampaignError as error:
            corrupt.append(f'{digest[:12]}: {error}')
            continue
        differing = sorted(field for field in DERIVED_FIELDS if payload.get(field) != expected_record[field])
        if differing:
            tampered.append(f'{digest[:12]}: {differing} disagree with the value recomputed from the raw row')
            continue
        if job_id(manifest['execution_manifest_sha256'], payload['job']) != digest:
            mislabeled.append(f'{digest[:12]}: identity does not re-derive the file name')
            continue
        if payload.get('classification') not in FAILURE_TAXONOMY:
            mislabeled.append(f'{digest[:12]}: classification {payload.get("classification")!r} is not in the '
                              'frozen taxonomy')
            continue
        if manifest.get('code_tree_sha256') and payload.get('code_tree_sha256') != manifest['code_tree_sha256']:
            drift.append(f'{digest[:12]}: {str(payload.get("code_tree_sha256"))[:12]}')

    missing = sorted(set(expected) - set(found))
    foreign = sorted(set(found) - set(expected))
    orphan_raw = sorted(name for name in raw_names if name[len('unit_'):-len('.json')] not in expected_units)
    committed_units = {expected[d]['unit_sha256'] for d in found if d in expected}
    half_units = sorted(unit['unit_sha256'][:12] for unit in manifest['units']
                        if f'unit_{unit["unit_sha256"]}.json' in raw_names
                        and any(not (results_dir / f'job_{j}.json').exists() for j in unit['jobs']))

    complete = not (missing or foreign or mislabeled or tampered or drift or unreadable or corrupt
                    or orphan_raw or half_units)
    return {'expected_jobs': len(expected), 'expected_units': len(manifest['units']),
            'found_jobs': len(found), 'found_units': len(committed_units), 'raw_artifacts': len(raw_names),
            'validated_units': len(validated_units),
            'complete': complete, 'missing': missing, 'foreign': foreign, 'mislabeled': mislabeled,
            'tampered': tampered, 'corrupt_raw_artifacts': corrupt, 'code_tree_drift': drift,
            'unreadable': unreadable, 'orphan_raw_artifacts': orphan_raw,
            'units_awaiting_reconciliation': half_units,
            'rule': ('every frozen job present exactly once, every record equal to its recomputation from a '
                     'validated raw artifact, none foreign, all under the frozen code tree, and no unit left '
                     'half-committed')}


def pending_units(manifest, campaign_dir):
    """Units with no raw artifact yet, in manifest order, so a resume is deterministic."""
    raw_dir = paths(campaign_dir)['raw']
    return [unit for unit in manifest['units']
            if not (raw_dir / f'unit_{unit["unit_sha256"]}.json').exists()]


def pending(manifest, campaign_dir):
    """Jobs with no committed record yet, in manifest order."""
    results_dir = paths(campaign_dir)['results']
    return [job for job in manifest['jobs']
            if not (results_dir / f'job_{job["job_sha256"]}.json').exists()]


# --- aggregation -------------------------------------------------------------------------------------------------

def aggregate(manifest, campaign_dir):
    """One summary row per candidate, computed from the VALIDATED raw rows, never from the result records."""
    state = completeness(manifest, campaign_dir)
    if not state['complete']:
        raise CampaignError('refusing to aggregate an incomplete campaign: '
                            + json.dumps({k: v for k, v in state.items() if k != 'rule'}))
    raw_dir = paths(campaign_dir)['raw']
    rows = {}
    for unit in manifest['units']:
        validated = validate_raw_artifact(manifest, raw_dir / f'unit_{unit["unit_sha256"]}.json')
        for job in (j for j in manifest['jobs'] if j['unit_sha256'] == unit['unit_sha256']):
            metrics = derive_metrics(validated['rows'][job['cell_id']])
            bucket = rows.setdefault(job['candidate_sha256'],
                                     {'name': job['candidate_name'], 'parameters': job['candidate_parameters'],
                                      'candidate_index': job['candidate_index'], 'by_cell': {}})
            cell = bucket['by_cell'].setdefault(job['cell_id'],
                                                {'weight': job['cell_weight'], 'J': [], 'fp': [], 'fixes': []})
            cell['J'].append(metrics['score_J'])
            cell['fp'].append(metrics['fp_spatial'])
            cell['fixes'].append(metrics['fix_attempts'])

    summaries = []
    for candidate_sha256, bucket in rows.items():
        per_cell = bucket['by_cell']
        if len(per_cell) != manifest['cells']:
            raise CampaignError(f'{bucket["name"]}: covers {len(per_cell)} cells, expected {manifest["cells"]}')
        for cell_id, cell in per_cell.items():
            if len(cell['J']) != manifest['realizations_per_cell']:
                raise CampaignError(f'{bucket["name"]} / {cell_id}: {len(cell["J"])} realisations, expected '
                                    f'{manifest["realizations_per_cell"]}')
        summaries.append({
            'name': bucket['name'], 'parameters': bucket['parameters'],
            'candidate_sha256': candidate_sha256, 'candidate_index': bucket['candidate_index'],
            'weighted_mean_J': sum(c['weight'] * statistics.fmean(c['J']) for c in per_cell.values()),
            'weighted_mean_fp_spatial': sum(c['weight'] * statistics.fmean(c['fp']) for c in per_cell.values()),
            'mean_fix_attempts': statistics.fmean([f for c in per_cell.values() for f in c['fixes']]),
            'cells_covered': len(per_cell),
            'realizations_per_cell': manifest['realizations_per_cell'],
        })
    if len(summaries) != manifest['candidates']:
        raise CampaignError(f'{len(summaries)} candidate summaries, expected {manifest["candidates"]}')
    return sorted(summaries, key=lambda row: row['candidate_index'])


# --- CLI ------------------------------------------------------------------------------------------------------

def _verified(path, key, hash_field):
    payload = strict_loads(Path(path).read_text())[key]
    recomputed = canonical_sha256({k: v for k, v in payload.items() if k != hash_field})
    if recomputed != payload[hash_field]:
        raise SystemExit(f'{path} does not match its own {hash_field}')
    return payload


def summary(manifest, campaign_dir):
    state = completeness(manifest, campaign_dir)
    return {'campaign_dir': str(campaign_dir), 'manifest_sha256': manifest['manifest_sha256'],
            'namespace': manifest['namespace'],
            'candidates': manifest['candidates'], 'pairs': manifest['pairs'], 'cells': manifest['cells'],
            'realizations_per_cell': manifest['realizations_per_cell'],
            'expected_units': manifest['expected_units'], 'expected_jobs': manifest['expected_jobs'],
            'estimated_hours_at_17s': round(manifest['expected_jobs'] * 17 / 3600, 2),
            'executed_jobs': state['found_jobs'], 'raw_artifacts': state['raw_artifacts'],
            'pending_units': len(pending_units(manifest, campaign_dir)),
            'complete': state['complete'], 'missing': len(state['missing']), 'foreign': state['foreign'],
            'mislabeled': state['mislabeled'], 'tampered': state['tampered'],
            'corrupt_raw_artifacts': state['corrupt_raw_artifacts'],
            'code_tree_drift': state['code_tree_drift'],
            'units_awaiting_reconciliation': state['units_awaiting_reconciliation']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execution-manifest', required=True, type=Path)
    parser.add_argument('--freeze', required=True, type=Path,
                        help='the frozen configuration; the campaign binds its score specification')
    parser.add_argument('--comparator-protocol', required=True, type=Path)
    parser.add_argument('--out', type=Path, help='freeze a new campaign directory')
    parser.add_argument('--status', type=Path, help='report on an existing campaign directory (read only)')
    parser.add_argument('--namespace', default=TUNING_NAMESPACE, choices=('tuning', 'pilot'),
                        help='a gate manifest is emitted by p3_selection.py, from a verified selection')
    args = parser.parse_args(argv)
    if bool(args.out) == bool(args.status):
        raise SystemExit('choose exactly one of --out (freeze a campaign) or --status (report on one)')

    execution = _verified(args.execution_manifest, 'execution_manifest', 'manifest_sha256')
    frozen = _verified(args.freeze, 'configuration', 'manifest_sha256')
    comparator = _verified(args.comparator_protocol, 'comparator_protocol', 'protocol_sha256')

    if args.status:
        # Read only. Recovery is a MUTATION and lives in exactly one place, p3_execute.py, where it takes the
        # campaign lock, requires the live code tree and registers what it did (review 11).
        manifest = load(args.status)
        if manifest['execution_manifest_sha256'] != execution['manifest_sha256']:
            raise SystemExit('this campaign was frozen against a different execution manifest')
        if manifest['comparator_protocol_sha256'] != comparator['protocol_sha256']:
            raise SystemExit('this campaign was frozen against a different comparator protocol')
        campaign_dir = args.status
    else:
        if args.out.exists():
            raise SystemExit(f'{args.out} exists; this tool never overwrites a campaign directory')
        try:
            protocol_module.verify_policy_binding(comparator)
        except ValueError as error:
            raise SystemExit(str(error))
        manifest = build(execution, comparator, frozen, namespace=args.namespace,
                         code_tree_sha256=code_tree_hash())
        payload = {'campaign': CAMPAIGN_VERSION,
                   'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                   'code_tree_sha256': manifest['code_tree_sha256'], 'job_manifest': manifest}
        args.out.mkdir(parents=True)
        (args.out / 'p3_tuning_campaign.json').write_text(json.dumps(payload, indent=1) + '\n')
        campaign_dir = args.out
        paths(campaign_dir)['raw'].mkdir()
        paths(campaign_dir)['results'].mkdir()
        register(campaign_dir, 'campaign/frozen', manifest_sha256=manifest['manifest_sha256'],
                 expected_units=manifest['expected_units'], expected_jobs=manifest['expected_jobs'],
                 code_tree_sha256=manifest['code_tree_sha256'])

    print(json.dumps(summary(manifest, campaign_dir), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
