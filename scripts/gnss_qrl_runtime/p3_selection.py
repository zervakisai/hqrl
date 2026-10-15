"""The reference selection, and the gate manifest that may only be built from a verified one.

review 10 found that `require_gate_binding` trusted a file. Review 11 found that the fix trusted the
*campaign the file named*: the verifier re-derived whatever campaign it was pointed at, so a one-pair,
two-cell, one-realisation campaign with 18 jobs selected a reference, and so did a complete **pilot** campaign.
A chain that checks its own links but not where the chain is anchored proves nothing.

The anchor, stated independently of any artifact
    `REQUIRED_SCOPE` is the study's own description of what a tuning campaign must be - the tuning namespace,
    the approved execution and frozen manifest hashes, 8 cells, 4 pairs, 20 realisations, 720 units and 1440
    jobs - and `REQUIRED_REFERENCE_CANDIDATES` names the nine eligible candidates **by hash**, because a count
    of nine is not an identity: review 12 passed the old check with nine wrong ones. Both are checked
    against the campaign rather than read from it. The one campaign the reviewer has approved for execution is named in `CAMPAIGN_APPROVAL.json` under the reviewer's
    own directory, which is outside the hashed code tree so that granting the approval does not invalidate the
    campaign it approves. While no approval exists - and none does - **no selection can be made or verified at
    all**: a selection whose source nobody approved is not a selection.

What the verifier re-derives
    the campaign on disk and its own hash; its completeness, recomputed from validated raw artifacts; its
    aggregate table, recomputed from the raw rows; the frozen selector run again on that table; the stored
    table and ranking payloads against their declared hashes; the reference and challenger names and
    parameters re-derived from their candidate hashes; the live policy behaviour, policy source and selector
    implementation; and the campaign's frozen code tree against the tree the selection was made under.

The gate manifest
    320 episodes = 2 strategies x 8 cells x 20 gate realisations, as 160 paired units. It is built from a
    verified selection FILE, not from a dictionary a caller assembled, and it records the selection hash and
    the selection's campaign hash so the executor can refuse a gate unit that belongs to neither.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts/gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from aerial_qrl.training.checkpoint_identity import canonical_sha256              # noqa: E402
import comparator_protocol as protocol_module                                     # noqa: E402
import p3_campaign as campaign                                                    # noqa: E402

SELECTION_VERSION = 'gnss-p3-reference-selection/4'
GATE_NAMESPACE = 'gate'

#: The study's own description of a tuning campaign, checked against the campaign rather than inherited from
#: it. A campaign that does not match this cannot found a selection however internally consistent it is.
REQUIRED_SCOPE = {
    'namespace': 'tuning',
    'execution_manifest_sha256': 'c064546513b077f08abee37e7912e6bf2353dd4b80d3625335bb7f80c5e46ee4',
    'frozen_manifest_sha256': 'a869e737dce2512ef43e1d941d2d43f0a055ae68a71cb68ee5c6622a19d80003',
    'candidates': 9, 'pairs': 4, 'cells': 8, 'realizations_per_cell': 20,
    'expected_units': 720, 'expected_jobs': 1440,
}
#: The nine eligible candidates, by hash, in the frozen order. review 12: the scope check counted
#: candidates, so a manifest declaring nine WRONG hashes passed it. A count is not an identity.
REQUIRED_REFERENCE_CANDIDATES = (
    '27a3a50455a6bf16046f4fc48844f0e6e5865a648634a8b028875f2fa8c2f56c',   # never_fix
    '9827676594a09db933fb6e5f72e33808eab589fa6706db8243aa0b6247a474f2',   # always_fix_before_report
    'f6d6beb2649d718f7463891f3169eaca66afcdceedc4b939bd4d479d475bd5f5',   # end_batch_at_home
    'd89ac3628d2ac98d06140aba399c0543850cfec10466a6d4fba721b182f0081d',   # periodic_fix 150 s
    '891affc4206e1d3a28da2ca845ceaa6bcceda25c02931089324b2a9d6edec177',   # periodic_fix 300 s
    'c1b4e045af5d28940356555dcf32b41e233c3e09cddbfa4f31a8a2c199bd97f7',   # periodic_fix 600 s
    '88db4eedbad17b0792bb1c1852d20cdf5680f3db3b0e72445d953b5400fd4845',   # uncertainty_fix 0.75 m
    '09e437b9c3db35ebb7bd07b065484d4258f3f102fb7e2eeae88b361604835b4d',   # uncertainty_fix 1.50 m
    'e9bc8afd288cb21b2e36d25819c7cd224218547bfcf46a173dac91c68a49b488',   # uncertainty_fix 2.25 m
)
#: The one fixed challenger, by hash.
REQUIRED_CHALLENGER = '19dc833e78a024a5facb52f17c7d97d9f4418e33bcad95646280819db87ce754'
#: the reviewer's approval of ONE campaign manifest for execution, read from the reviewer's own directory rather than held
#: as a constant here.
#:
#: The reason is a deadlock, not a preference. review 11 requires the approved campaign to be executed
#: "without changing any source, test or configuration file", and the campaign binds the code tree. An
#: approval written into this file would change the tree and kill the campaign it approves, every time. So the
#: authority lives in `docs/study/gnss_qrl_v1/approvals/CAMPAIGN_APPROVAL.json`, which `code_version.py` does not
#: hash, is owned by the reviewer, and can therefore be granted without moving anything.
#:
#: While that file is absent - which it is - **no selection can be made or verified at all**.
APPROVAL_PATH = ROOT / 'docs/study/gnss_qrl_v1/approvals/CAMPAIGN_APPROVAL.json'
APPROVAL_VERSION = 'gnss-campaign-approval/1'
APPROVAL_FIELDS = ('approval_version', 'approved_campaign_sha256', 'approved_by', 'review', 'utc')
#: The selector implementation itself: the tie rule and the refusals are code, so they are part of identity.
SELECTOR_SOURCE = ROOT / 'scripts/gnss_qrl_runtime/comparator_protocol.py'

REQUIRED_FIELDS = ('selection_version', 'campaign_dir', 'campaign_manifest_sha256', 'approved_campaign_sha256',
                   'execution_manifest_sha256', 'frozen_manifest_sha256', 'comparator_protocol_sha256',
                   'policy_behaviour_version', 'policy_source_sha256', 'selector_source_sha256',
                   'code_tree_sha256', 'campaign_complete', 'scope',
                   'aggregate_table', 'aggregate_table_sha256', 'ranking', 'ranking_sha256',
                   'reference_sha256', 'reference_name', 'reference_parameters',
                   'challenger_sha256', 'challenger_name', 'challenger_parameters', 'selection_sha256')


class SelectionError(ValueError):
    """A selection cannot be made, or a selection artifact cannot be trusted."""


def approved_campaign(path=None):
    """The campaign manifest hash the reviewer has approved, or None when no approval exists.

    A malformed or unsigned approval is a refusal, not a fallback to "none approved": a file that is there but
    wrong means somebody tried to approve something, and guessing what they meant is exactly what this whole
    chain exists to prevent.
    """
    path = Path(path if path is not None else APPROVAL_PATH)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())['campaign_approval']
    except (OSError, ValueError, KeyError) as error:
        raise SelectionError(f'the campaign approval cannot be read: {error}') from error
    absent = [field for field in APPROVAL_FIELDS if field not in payload]
    if absent:
        raise SelectionError(f'the campaign approval is missing {absent}')
    if payload['approval_version'] != APPROVAL_VERSION:
        raise SelectionError(f"the campaign approval is {payload['approval_version']!r}, not {APPROVAL_VERSION!r}")
    if payload['approved_by'] != 'reviewer':
        raise SelectionError(f"a campaign is approved by the reviewer, not by {payload['approved_by']!r}")
    digest = payload['approved_campaign_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or not all(c in '0123456789abcdef' for c in digest):
        raise SelectionError(f'the approved campaign hash is not a sha256: {digest!r}')
    return digest


def selector_source_sha256():
    return hashlib.sha256(SELECTOR_SOURCE.read_bytes()).hexdigest()


def check_scope(manifest, approved_campaign_sha256):
    """Require the campaign to be the approved one AND to match the study's own description of a campaign."""
    if not approved_campaign_sha256:
        raise SelectionError('no campaign manifest has been approved for execution yet; until the reviewer approves '
                             'one, a selection cannot be made or verified')
    if manifest['manifest_sha256'] != approved_campaign_sha256:
        raise SelectionError(f'campaign {manifest["manifest_sha256"][:12]} is not the approved campaign '
                             f'{approved_campaign_sha256[:12]}')
    problems = [f'{key}: {manifest.get(key)!r} != {value!r}'
                for key, value in REQUIRED_SCOPE.items() if manifest.get(key) != value]
    eligible = tuple(manifest.get('reference_eligible_candidates') or ())
    if tuple(sorted(eligible)) != tuple(sorted(REQUIRED_REFERENCE_CANDIDATES)):
        problems.append('reference_eligible_candidates are not the nine frozen candidate hashes: '
                        f'missing {sorted(set(REQUIRED_REFERENCE_CANDIDATES) - set(eligible))[:3]}, '
                        f'unexpected {sorted(set(eligible) - set(REQUIRED_REFERENCE_CANDIDATES))[:3]}')
    if tuple(manifest.get('challenger_candidates') or ()) != (REQUIRED_CHALLENGER,):
        problems.append(f'challenger_candidates is {manifest.get("challenger_candidates")!r}, not the one '
                        'frozen challenger')
    if problems:
        raise SelectionError('the campaign is not a tuning campaign of the approved scope: ' + '; '.join(problems))
    return dict(REQUIRED_SCOPE, reference_eligible_candidates=list(REQUIRED_REFERENCE_CANDIDATES),
                challenger_candidates=[REQUIRED_CHALLENGER])


def _candidate_by_hash(comparator_protocol, digest, role, what):
    matches = [c for c in comparator_protocol['candidates'] if c['candidate_sha256'] == digest]
    if len(matches) != 1:
        raise SelectionError(f'the {what} {digest[:12]} is not a candidate in the frozen protocol')
    candidate = matches[0]
    if candidate['role'] != role:
        raise SelectionError(f'the {what} {candidate["name"]!r} is a {candidate["role"]}, not a {role}')
    recomputed = canonical_sha256({'name': candidate['name'], 'parameters': candidate['parameters']})
    if recomputed != digest:
        raise SelectionError(f'the {what} hash does not follow from its own name and parameters')
    return candidate


def select_from_campaign(campaign_dir, comparator_protocol, root=None):
    """Aggregate an APPROVED, complete tuning campaign and run the frozen selector on it."""
    root = Path(root or ROOT)
    campaign_dir = Path(campaign_dir)
    manifest = campaign.load(campaign_dir)
    approved = approved_campaign()          # the reviewer-owned artifact, never a caller-supplied digest
    scope = check_scope(manifest, approved)
    if manifest['comparator_protocol_sha256'] != comparator_protocol['protocol_sha256']:
        raise SelectionError('the campaign was frozen against a different comparator protocol')
    if manifest['code_tree_sha256'] != campaign.code_tree_hash():
        raise SelectionError('the live tree is not the tree this campaign was frozen against; a selection made '
                             'by different code from the code that produced the numbers is not reproducible')
    state = campaign.completeness(manifest, campaign_dir)
    if not state['complete']:
        raise SelectionError('refusing to select from an incomplete campaign: '
                             + json.dumps({k: v for k, v in state.items() if k != 'rule'}))
    table = campaign.aggregate(manifest, campaign_dir)
    decision = protocol_module.select(table, expected_cells=manifest['cells'],
                                      expected_realizations=manifest['realizations_per_cell'])

    winner = decision['winner']
    reference = _candidate_by_hash(comparator_protocol, winner['candidate_sha256'],
                                   protocol_module.ROLE_REFERENCE, 'selected reference')
    challengers = protocol_module.challenger_candidates()
    if len(challengers) != 1:
        raise SelectionError(f'the protocol must fix exactly one challenger, it fixes {len(challengers)}')
    challenger = _candidate_by_hash(comparator_protocol, protocol_module.candidate_key(challengers[0]),
                                    protocol_module.ROLE_CHALLENGER, 'fixed challenger')
    if challenger['candidate_sha256'] == reference['candidate_sha256']:
        raise SelectionError('the challenger cannot be the reference')

    ranking = [row['candidate_sha256'] for row in decision['ranking']]
    selection = {
        'selection_version': SELECTION_VERSION,
        'campaign_dir': str(campaign_dir.resolve().relative_to(root.resolve())),
        'campaign_manifest_sha256': manifest['manifest_sha256'],
        'approved_campaign_sha256': approved,
        'execution_manifest_sha256': manifest['execution_manifest_sha256'],
        'frozen_manifest_sha256': manifest['frozen_manifest_sha256'],
        'comparator_protocol_sha256': comparator_protocol['protocol_sha256'],
        'policy_behaviour_version': comparator_protocol['policy_behaviour_version'],
        'policy_source_sha256': comparator_protocol['policy_source_sha256'],
        'selector_source_sha256': selector_source_sha256(),
        'code_tree_sha256': manifest['code_tree_sha256'],
        'campaign_complete': True,
        'scope': scope,
        'campaign_units': manifest['expected_units'], 'campaign_jobs': manifest['expected_jobs'],
        'aggregate_table': table, 'aggregate_table_sha256': canonical_sha256(table),
        'ranking': ranking, 'ranking_sha256': canonical_sha256(ranking),
        'decided_at_step': decision['decided_at_step'],
        'reference_sha256': reference['candidate_sha256'], 'reference_name': reference['name'],
        'reference_parameters': reference['parameters'],
        'challenger_sha256': challenger['candidate_sha256'], 'challenger_name': challenger['name'],
        'challenger_parameters': challenger['parameters'],
        'selection_rule': ('weighted mean J over the tuning realisations with the frozen cell weights, then '
                           'fewer false reports, then fewer fix attempts, then the frozen candidate order'),
        'gate_rule': ('the gate scores exactly these two candidates, once, on gate seeds that no selection '
                      'ever saw; a candidate that is neither is refused by identity'),
    }
    selection['selection_sha256'] = canonical_sha256(selection)
    return selection


def verify_selection(selection_path, comparator_protocol, root=None):
    """Re-derive the whole chain from disk, and require the campaign to be the approved one."""
    root = Path(root or ROOT)
    try:
        payload = json.loads(Path(selection_path).read_text())['reference_selection']
    except (OSError, ValueError, KeyError) as error:
        raise SelectionError(f'the selection artifact cannot be read: {error}') from error

    absent = [f for f in REQUIRED_FIELDS if f not in payload]
    if absent:
        raise SelectionError(f'the selection artifact is missing {absent}')
    if payload['selection_version'] != SELECTION_VERSION:
        raise SelectionError(f"the selection artifact is {payload['selection_version']!r}, not {SELECTION_VERSION!r}")
    if canonical_sha256({k: v for k, v in payload.items() if k != 'selection_sha256'}) != payload['selection_sha256']:
        raise SelectionError('the selection artifact does not match its own hash')
    if payload['campaign_complete'] is not True:
        raise SelectionError('the selection artifact does not declare a complete tuning campaign')

    # identity of the code that produced and will consume it
    if payload['comparator_protocol_sha256'] != comparator_protocol['protocol_sha256']:
        raise SelectionError('the selection artifact was made against a different comparator protocol')
    if (payload['policy_behaviour_version'] != comparator_protocol['policy_behaviour_version']
            or payload['policy_source_sha256'] != comparator_protocol['policy_source_sha256']):
        raise SelectionError('the selection artifact was made against different comparator behaviour')
    if payload['selector_source_sha256'] != selector_source_sha256():
        raise SelectionError('the selection artifact was made by a different selector implementation')

    # the two roles, re-derived from their candidate hashes rather than read from the artifact
    reference = _candidate_by_hash(comparator_protocol, payload['reference_sha256'],
                                   protocol_module.ROLE_REFERENCE, 'selected reference')
    challenger = _candidate_by_hash(comparator_protocol, payload['challenger_sha256'],
                                    protocol_module.ROLE_CHALLENGER, 'fixed challenger')
    if (payload['reference_name'], payload['reference_parameters']) != (reference['name'], reference['parameters']):
        raise SelectionError('the stored reference name or parameters are not the ones its hash denotes')
    if (payload['challenger_name'], payload['challenger_parameters']) != (challenger['name'], challenger['parameters']):
        raise SelectionError('the stored challenger name or parameters are not the ones its hash denotes')
    if payload['challenger_sha256'] != protocol_module.candidate_key(protocol_module.challenger_candidates()[0]):
        raise SelectionError('the artifact names a challenger that is not the single frozen challenger')
    if payload['reference_sha256'] == payload['challenger_sha256']:
        raise SelectionError('the selection artifact names one candidate as both reference and challenger')

    # the stored payloads against their own declared hashes, before anything is recomputed from them
    if canonical_sha256(payload['aggregate_table']) != payload['aggregate_table_sha256']:
        raise SelectionError('the stored aggregate table does not match its own declared hash')
    if canonical_sha256(payload['ranking']) != payload['ranking_sha256']:
        raise SelectionError('the stored ranking does not match its own declared hash')

    # the campaign on disk: approved, in scope, complete, and producing this very table and ranking
    campaign_dir = root / payload['campaign_dir']
    if not (campaign_dir / 'p3_tuning_campaign.json').exists():
        raise SelectionError(f'the campaign this selection names does not exist: {payload["campaign_dir"]}')
    manifest = campaign.load(campaign_dir)
    if manifest['manifest_sha256'] != payload['campaign_manifest_sha256']:
        raise SelectionError('the campaign on disk is not the campaign this selection was made from')
    approved = approved_campaign()
    if payload['approved_campaign_sha256'] != approved:
        raise SelectionError(f'the selection was made against approved campaign '
                             f'{str(payload["approved_campaign_sha256"])[:12]}, not {str(approved)[:12]}')
    check_scope(manifest, approved)
    for field in ('execution_manifest_sha256', 'frozen_manifest_sha256', 'code_tree_sha256'):
        if payload[field] != manifest[field]:
            raise SelectionError(f'{field} in the artifact is not the campaign\'s')
    # review 12: `select_from_campaign` checked the live tree and `verify_selection` did not, so a
    # selection stayed acceptable at gate time after the tree had moved. The gate reads this function.
    live = campaign.code_tree_hash()
    if live != manifest['code_tree_sha256']:
        raise SelectionError(f'the live code tree is {str(live)[:12]} but this selection and its campaign were '
                             f'made under {str(manifest["code_tree_sha256"])[:12]}')
    state = campaign.completeness(manifest, campaign_dir)
    if not state['complete']:
        raise SelectionError('the campaign this selection names is not complete: '
                             + json.dumps({k: v for k, v in state.items() if k != 'rule'}))
    table = campaign.aggregate(manifest, campaign_dir)
    if canonical_sha256(table) != payload['aggregate_table_sha256']:
        raise SelectionError('the aggregate table re-derived from the campaign is not the one selected from')
    decision = protocol_module.select(table, expected_cells=manifest['cells'],
                                      expected_realizations=manifest['realizations_per_cell'])
    ranking = [row['candidate_sha256'] for row in decision['ranking']]
    if canonical_sha256(ranking) != payload['ranking_sha256']:
        raise SelectionError('the ranking re-derived from the campaign is not the one recorded')
    if decision['winner']['candidate_sha256'] != payload['reference_sha256']:
        raise SelectionError(f"the frozen selector chooses {decision['winner']['name']!r}, not "
                             f"{payload['reference_name']!r}")
    return dict(payload, verified={'campaign_manifest_sha256': manifest['manifest_sha256'],
                                   'completeness': {k: v for k, v in state.items() if k != 'rule'},
                                   'aggregate_table_sha256': canonical_sha256(table),
                                   'ranking_sha256': canonical_sha256(ranking)})


def gate_manifest(selection_path, execution_manifest, comparator_protocol, frozen_configuration,
                  code_tree_sha256=None, root=None):
    """The 320-episode gate manifest, built ONLY from a selection file that survives verification."""
    selection = verify_selection(selection_path, comparator_protocol, root=root)
    if execution_manifest['manifest_sha256'] != selection['execution_manifest_sha256']:
        raise SelectionError('the gate must run on the same execution manifest the tuning campaign used')
    if execution_manifest['frozen_manifest_sha256'] != selection['frozen_manifest_sha256']:
        raise SelectionError('the gate must run on the same frozen configuration the tuning campaign used')

    tuning_seeds = {s for cell in execution_manifest['cells'] for s in cell['seeds']['tuning']}
    gate_seeds = {s for cell in execution_manifest['cells'] for s in cell['seeds'][GATE_NAMESPACE]}
    shared = sorted(tuning_seeds & gate_seeds)
    if shared:
        raise SelectionError(f'the gate seeds are not disjoint from the tuning seeds: {shared[:5]}')

    by_hash = {c['candidate_sha256']: c for c in comparator_protocol['candidates']}
    candidates = [by_hash[selection['reference_sha256']], by_hash[selection['challenger_sha256']]]
    manifest = campaign.build(execution_manifest, comparator_protocol, frozen_configuration,
                              namespace=GATE_NAMESPACE, code_tree_sha256=code_tree_sha256,
                              candidates=candidates)
    manifest = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    manifest['selection_sha256'] = selection['selection_sha256']
    manifest['selection_campaign_manifest_sha256'] = selection['campaign_manifest_sha256']
    manifest['reference_sha256'] = selection['reference_sha256']
    manifest['challenger_sha256'] = selection['challenger_sha256']
    manifest['seed_disjointness'] = ('verified: no seed in the gate namespace appears in the tuning namespace '
                                     'of any cell of this execution manifest')
    manifest['gate_rule'] = ('scored once; the seeds here were never used by any selection, and the two '
                             'candidates are bound to the verified selection by hash')
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    return manifest


def _verified(path, key, hash_field):
    payload = json.loads(Path(path).read_text())[key]
    if canonical_sha256({k: v for k, v in payload.items() if k != hash_field}) != payload[hash_field]:
        raise SystemExit(f'{path} does not match its own {hash_field}')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', required=True, type=Path, help='a COMPLETE, APPROVED tuning campaign')
    parser.add_argument('--comparator-protocol', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--execution-manifest', type=Path, help='with --gate-manifest')
    parser.add_argument('--freeze', type=Path, help='with --gate-manifest: the frozen configuration')
    parser.add_argument('--gate-manifest', action='store_true',
                        help='also emit the 320-episode gate manifest bound to this selection')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this tool never overwrites a selection directory')
    if args.gate_manifest and not (args.execution_manifest and args.freeze):
        raise SystemExit('--gate-manifest needs --execution-manifest and --freeze')

    comparator = _verified(args.comparator_protocol, 'comparator_protocol', 'protocol_sha256')
    protocol_module.verify_policy_binding(comparator)
    selection = select_from_campaign(args.campaign, comparator)

    args.out.mkdir(parents=True)
    payload = {'selection': SELECTION_VERSION,
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               'code_tree_sha256': campaign.code_tree_hash(), 'reference_selection': selection}
    selection_path = campaign.atomic_write_json(args.out / 'reference_selection.json', payload)
    verify_selection(selection_path, comparator)                       # it must survive its own gate

    emitted = {'out': str(args.out), 'selection_sha256': selection['selection_sha256'],
               'reference': selection['reference_name'], 'reference_parameters': selection['reference_parameters'],
               'reference_sha256': selection['reference_sha256'],
               'challenger': selection['challenger_name'], 'decided_at_step': selection['decided_at_step']}
    if args.gate_manifest:
        execution = _verified(args.execution_manifest, 'execution_manifest', 'manifest_sha256')
        frozen = _verified(args.freeze, 'configuration', 'manifest_sha256')
        gate = gate_manifest(selection_path, execution, comparator, frozen,
                             code_tree_sha256=campaign.code_tree_hash())
        campaign.atomic_write_json(args.out / 'p3_gate_manifest.json',
                                   {'campaign': campaign.CAMPAIGN_VERSION, 'utc': payload['utc'],
                                    'code_tree_sha256': gate['code_tree_sha256'], 'job_manifest': gate})
        emitted.update(gate_manifest_sha256=gate['manifest_sha256'], gate_units=gate['expected_units'],
                       gate_jobs=gate['expected_jobs'])
    print(json.dumps(emitted, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
