"""Fail-closed P3 cell runner: the frozen configuration and execution manifest decide everything (review 7).

Why a separate runner
    `run_gnss_episodes.py` is the P2 engineering runner. It accepts an independent candidate config, derives
    its own seeds, and - the defect the review found - its budget reached only the SCORE. The executive context, the
    `ReturnReserve`, the action mask and the episode deadline all still came from the protocol briefing at
    2700 s, so a "tight" episode would have been labelled and scored at 1881 s while flying 2700 s dynamics.
    That runner is not extended; this one replaces it for P3, and it takes no scientific parameter at all.

Everything comes from two hashed objects
    the frozen configuration (`p3_freeze_v1`)  - encoder, action order, reward/score versions, anchors, dwell,
                                                 noise, availability, budgets, cells and weights
    the execution manifest (`p3_execution_manifest_v1`) - per cell: protocol, bank parts, presence, N_ref,
                                                 drift configuration, budget, weight and the exact seed lists
    Both are re-hashed on load and refused if they do not match their recorded hash or each other. There is no
    command-line option that can change a scientific value.

The budget reaches the dynamics, not only the score
    `fast_mission.load_context(..., budget_overrides={'mission_time_s': cell budget})` rebuilds the briefing
    BEFORE anything else exists, so the same number reaches the `ReturnReserve`, the executive's deadline, the
    observation's `remaining_time_s`, the anchor reachability mask and the score together. Every episode row
    records the budget actually applied, read back from the live briefing rather than from the request.

The comparator comes from the frozen grid, by hash
    `--candidate-sha256` names ONE frozen candidate; the runner resolves it to that candidate's name and
    parameters and refuses anything else. A policy NAME is not accepted for scientific execution, because
    `uncertainty_fix` is three candidates in the grid and a name cannot say which. The candidate identity goes
    into the episode ID, the output path and every row.

Namespaces, closed by evidence rather than by a source edit
    All three namespaces exist in code, and none is reachable without the artifact that authorises it: a
    tuning episode needs a frozen campaign, and a gate episode needs a reference selection that re-derives
    from disk. That way the tree a campaign is frozen against never has to change to run the next stage.

The gate, bound to evidence
    `require_gate_binding` refuses a gate episode that does not name a reference selection which survives
    re-derivation - the campaign it names is loaded, re-hashed, rechecked for completeness, re-aggregated and
    re-selected - and refuses any candidate that is neither the selected reference nor the fixed challenger.
    `--selection` is refused outside the gate, so a pilot or tuning episode can never be dressed up as one
    that scores a selected rule.

Per-episode diagnostics
    Every row carries the operational record section 4.4 asks for: distance, the anchor and timestamp of each
    fix attempt with its availability and acceptance, submission times and latencies, held-but-unsubmitted
    candidates, the executed action sequence and counts, the forced-return reason, the comparator's final
    internal state, and an explicit null for collision/contact evidence, which the fast tier does not model.

Reproduce
    .venv/bin/python scripts/gnss_qrl_runtime/run_p3_cells.py \
        --freeze output/study/gnss_qrl_v1/frozen/p3_freeze_v1/p3_frozen_configuration.json \
        --execution-manifest output/study/gnss_qrl_v1/frozen/p3_execution_manifest_v1/p3_execution_manifest.json \
        --time-model output/training/time_model_village_v1/20260917T074923Z/time_model.json \
        --comparator-protocol output/study/gnss_qrl_v1/frozen/comparator_protocol_v2/comparator_protocol.json \
        --namespace pilot --candidate-sha256 <exact frozen candidate hash> --pair d2__tight_derived --out <dir>
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts/gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from aerial_qrl.mission.contracts import dedup_position_xy                                    # noqa: E402
from aerial_qrl.mission.cached_view_backend import CachedViewConfig                          # noqa: E402
from aerial_qrl.mission.georeference import GeoreferenceService                              # noqa: E402
from aerial_qrl.mission.georeference_contracts import GeoreferenceConfig                     # noqa: E402
from aerial_qrl.mission.observation_bank import open_banks                                   # noqa: E402
from aerial_qrl.mission.relocalization import AnchorSet                                      # noqa: E402
from aerial_qrl.mission.time_model import TimeModel                                          # noqa: E402
from aerial_qrl.training.checkpoint_identity import canonical_sha256                         # noqa: E402
from aerial_qrl.training.reward_gnss import (REWARD_VERSION, SCORE_VERSION, TimelyScoreSpec,  # noqa: E402
                                             check_reward_identity, episode_rewards)
import fast_mission                                                                          # noqa: E402
from gnss_config import check_anchor_safety                                                  # noqa: E402
from gnss_policies import COMPARATORS, build_policy                                          # noqa: E402

#: /16 (revision 22): the rows are the same bytes as /15 except `runner` and `diagnostics.schema`; the version moves
#: because the evidence contract they are validated under moved to exact numeric identity (review 21).
RUNNER_VERSION = 'gnss-p3-runner/16'
#: Version of the common-random-number derivation. It is part of the identity, so a future change to the
#: rule cannot silently make old and new runs look paired.
CRN_VERSION = 'gnss-crn/1'
#: The per-episode operational record, versioned so a later field change is visible in every row.
DIAGNOSTICS_SCHEMA = 'gnss-episode-diagnostics/11'
#: Every namespace is reachable in CODE, and none of them is reachable without its evidence. review 11:
#: "Gate must be enabled by this evidence binding rather than a later source edit, so the v6 tree can remain
#: fixed." A `tuning` episode needs a frozen campaign; a `gate` episode additionally needs a reference
#: selection that survives re-derivation from disk (`require_gate_binding`, `p3_selection.verify_selection`),
#: and the executor refuses a gate unit whose candidate is not one of the two that selection bound. Opening a
#: namespace in code is not running it: this revision reports zero tuning and zero gate jobs.
ALLOWED_NAMESPACES = ('pilot', 'tuning', 'gate')
#: review 9 requirement 6, hardened by review 10. The gate may never score a comparator that was chosen
#: informally, and a selection artifact that is merely self-consistent is a signature on a blank page: a review
#: built one naming a campaign that does not exist, with the planner as reference and `never_fix` as
#: challenger, and it was accepted. The verification now lives in `p3_selection.verify_selection`, which
#: re-derives the campaign, its completeness, its aggregate table and the selector's own ranking from disk.
#: What a run of each namespace is, said once. review 10: the runner hard-coded a PILOT classification
#: and a `p3_pilot_episodes.json` filename into every payload, so a tuning or gate run would have described
#: itself as a technical smoke. The namespace decides both.
CLASSIFICATION = {
    'pilot': ('pilot technical smoke: infrastructure evidence only. It cannot select a comparator and cannot '
              'enter P3 statistics.'),
    'tuning': ('tuning realisation: used ONLY to select the reference rule from the frozen reference-eligible '
               'set. It is never a gate number and never appears as a result.'),
    'gate': ('gate realisation: the published comparison of the selected reference against the fixed '
             'challenger. Scored once, on seeds no selection ever saw.'),
}
BACKEND = {'reobservation_model': 'visit_frame_permutation/3', 'repeated_capture': 'invalid',
           'evidence_serving': 'all', 'provenance': 'frozen'}
SOURCES = ('scripts/gnss_qrl_runtime/run_p3_cells.py', 'scripts/gnss_qrl_runtime/gnss_policies.py',
           'scripts/fast_mission.py', 'src/aerial_qrl/mission/executive.py',
           'src/aerial_qrl/mission/georeference.py', 'src/aerial_qrl/training/reward_gnss.py')


class P3RunnerError(ValueError):
    """A refusal. Every one of these happens before a policy is constructed or a macro is executed."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_hash():
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/code_version.py')], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or None


def crn_group(pair_id, namespace, seed):
    """The common-random-number group of a disturbance realisation.

    Built from the frozen `pair_id`, the namespace and the ACTUAL manifest seed - and from nothing else.
    Presence, comparator, action history and output path are deliberately excluded, so the two members of a
    pair and every comparator run on that pair meet the SAME Brownian path and the same anchor-measurement
    noise.

    This is the defect review 8 found: the first runner built the key from `cell_id`, which contains
    `positive` or `no_human`, so the two members of a declared pair walked different paths. Their true bias at
    600 s differed by 5.02 m - larger than the 3 m report tolerance - while the artifact still claimed
    `paired_identically = true`, because that flag compared a descriptive seed field instead of the actual
    stochastic identity. Both halves of that mistake are fixed: the key below, and the check in `paired_state`.
    """
    label = f'{CRN_VERSION}|{pair_id}|{namespace}|{int(seed)}'
    return {'crn_version': CRN_VERSION, 'crn_label': label,
            'crn_sha256': hashlib.sha256(label.encode()).hexdigest()}


def crn_path_key(group):
    """The georeference path key of a CRN group: a valid identifier derived from the group hash alone."""
    return f'crn-{group["crn_sha256"][:40]}'


#: What two rows must share for a contrast between them to be paired. Named here, once, so that the gate
#: verdict compares the same tuple `paired_state` does rather than a shorter one of its own (review 14).
PAIRED_IDENTITY_FIELDS = ('crn_sha256', 'disturbance_path_sha256', 'measurement_noise_sha256',
                          'georeference_sha256', 'reobservation_seed', 'n_ref', 'mission_budget_s_applied',
                          'pair_id')


def paired_state(rows):
    """Whether the rows really share a disturbance, judged on the PERSISTED stochastic identities.

    Equality of a descriptive seed field is not equality of a process, so this compares the actual path and
    noise hashes the services used, the georeference configuration hash, the reobservation seed, N_ref and the
    budget. Anything less would be the flag that lied.
    """
    if len(rows) < 2:
        return {'paired': None, 'reason': 'fewer than two members to compare'}
    keys = PAIRED_IDENTITY_FIELDS          # one definition of "paired", used here and by the gate verdict
    differing = [key for key in keys if len({json.dumps(row[key], sort_keys=True) for row in rows}) != 1]
    presences = {row['presence'] for row in rows}
    return {'paired': bool(not differing and len(presences) == len(rows)),
            'compared': list(keys), 'differing': differing, 'presences': sorted(presences),
            'rule': ('the members must agree on every stochastic identity and configuration hash, and must be '
                     'distinct presence conditions')}


def load_comparator_protocol(path):
    """Load and canonically verify the hashed comparator protocol.

    review 9: the previous runner took only `--comparator <name>` and passed empty parameters, so
    enabling tuning would have executed the policy DEFAULTS instead of the frozen grid - the 150/300/600 s
    staleness periods and the 0.75/1.5/2.25 m thresholds would never have been run at all.
    """
    payload = json.loads(Path(path).read_text())['comparator_protocol']
    recomputed = canonical_sha256({k: v for k, v in payload.items() if k != 'protocol_sha256'})
    if recomputed != payload['protocol_sha256']:
        raise P3RunnerError('the comparator protocol does not match its own hash')
    # review 10: a candidate hash names a name and its parameters, so a comparator's BEHAVIOUR can change
    # without any hash moving. The protocol carries the policy behaviour version and source hash, and they are
    # checked against the live tree here, before a candidate is ever resolved.
    import comparator_protocol                                                   # late: it imports this module
    try:
        comparator_protocol.verify_policy_binding(payload)
    except ValueError as error:
        raise P3RunnerError(str(error)) from error
    return payload


def resolve_candidate(protocol, candidate_sha256):
    """Resolve an exact candidate hash to its frozen name, parameters, index and role.

    A name alone is refused for scientific execution: `uncertainty_fix` is three different candidates in the
    frozen grid, and a name cannot say which.
    """
    matches = [c for c in protocol['candidates'] if c['candidate_sha256'] == candidate_sha256]
    if len(matches) != 1:
        known = [(c['candidate_sha256'][:12], c['name'], c['parameters']) for c in protocol['candidates']]
        raise P3RunnerError(f'candidate {candidate_sha256[:12]!r} is not in the frozen protocol; known: {known}')
    candidate = matches[0]
    recomputed = canonical_sha256({'name': candidate['name'], 'parameters': candidate['parameters']})
    if recomputed != candidate_sha256:
        raise P3RunnerError('the candidate hash does not follow from its own name and parameters')
    return candidate


def require_gate_binding(namespace, candidate, protocol, selection_path):
    """Gate episodes must name a selection artifact that survives re-derivation, and run only what it binds.

    Everything the artifact claims is recomputed from disk by `p3_selection.verify_selection`: the campaign it
    names is loaded and re-hashed, its completeness is recomputed from its raw artifacts and cell records, its
    aggregate table is rebuilt, and the frozen selector is run again on that table. On top of the chain, the
    reference must carry `reference_eligible` in the frozen protocol and the challenger must be the single
    frozen challenger, so a role swap is refused by identity and not by ranking.

    All of it happens before a policy is constructed.
    """
    if namespace != 'gate':
        if selection_path is not None:
            raise P3RunnerError(f'--selection binds a gate reference and namespace {namespace!r} is not the '
                                'gate; a pilot or tuning episode must not carry one')
        return None
    if selection_path is None:
        raise P3RunnerError('gate execution requires --selection: the reference must come from a complete '
                            'tuning campaign, by hash, not from a name on the command line')
    import p3_selection                                                    # late: it imports p3_campaign
    try:
        selection = p3_selection.verify_selection(selection_path, protocol)
    except p3_selection.SelectionError as error:
        raise P3RunnerError(str(error)) from error
    bound = {selection['reference_sha256']: 'reference', selection['challenger_sha256']: 'challenger'}
    if candidate['candidate_sha256'] not in bound:
        raise P3RunnerError(f"candidate {candidate['name']!r} is neither the selected reference nor the fixed "
                            'challenger; the gate scores those two and nothing else')
    return dict(selection, gate_role=bound[candidate['candidate_sha256']])


def load_frozen(freeze_path, manifest_path):
    """Load both objects, re-hash them, and refuse any mismatch. Nothing else is trusted."""
    frozen = json.loads(Path(freeze_path).read_text())['configuration']
    recomputed = canonical_sha256({k: v for k, v in frozen.items() if k != 'manifest_sha256'})
    if recomputed != frozen['manifest_sha256']:
        raise P3RunnerError('the frozen configuration does not match its own manifest hash')
    manifest = json.loads(Path(manifest_path).read_text())['execution_manifest']
    recomputed = canonical_sha256({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    if recomputed != manifest['manifest_sha256']:
        raise P3RunnerError('the execution manifest does not match its own manifest hash')
    if manifest['frozen_manifest_sha256'] != frozen['manifest_sha256']:
        raise P3RunnerError(
            f"the execution manifest was built for frozen configuration {manifest['frozen_manifest_sha256'][:12]}, "
            f"not {frozen['manifest_sha256'][:12]}")
    return frozen, manifest


def verify_cell_data(cell):
    """Re-hash the protocol and every bank part named by the cell, before anything is opened."""
    protocol = ROOT / cell['data']['protocol_path']
    if sha256_file(protocol) != cell['data']['protocol_sha256']:
        raise P3RunnerError(f"{cell['cell_id']}: the protocol on disk differs from the manifest hash")
    for part in cell['data']['bank_parts']:
        manifest_path = ROOT / part['path'] / 'manifest.json'
        if not manifest_path.exists():
            raise P3RunnerError(f"{cell['cell_id']}: bank part {part['path']} is missing")
        if sha256_file(manifest_path) != part['manifest_sha256']:
            raise P3RunnerError(f"{cell['cell_id']}: bank part {part['path']} differs from the manifest hash")
    return protocol, [ROOT / part['path'] for part in cell['data']['bank_parts']]


#: The two rule statements a row carries beside its witness and its thresholds. Named once, here, so the
#: validator compares the stored text with the text the runner writes rather than leaving it free (revision 21:
#: the mutation sweep found both could be rewritten without a refusal).
WITNESS_RULE = ('threshold_ready = detections >= min_detections AND len(credited_views_xy) >= min_views; '
                'open_reportable = threshold_ready AND present in the last observation the policy decided '
                'on AND not closed by a report - by track id or within merge_radius_m of a receipt dedup '
                'position. PRESENCE and POSITION come from that last observation; report CLOSURE comes from '
                'the episode\'s FINAL public receipts, which are the receipts its executed REPORT actions '
                'appended. Presence itself is derived from the ordered action evidence, never supplied.')
OPEN_REPORTABLE_RULE = ('open_reportable = threshold-ready AND not already reported by id or '
                        'within merge_radius_m, which is the executive\'s own is_reported')


def episode_identity(candidate_index, candidate_sha256, name, cell_id, namespace, realization):
    """The episode ID. The candidate identity is in it, so two parameter variants of one policy cannot collide
    (review 9); the validator derives it with this same function."""
    return f'c{candidate_index:02d}-{candidate_sha256[:12]}__{name}__{cell_id}__{namespace}{realization}'


#: Collision/contact evidence is not modelled by the fast tier, so it is recorded as an explicit null with
#: its reason rather than omitted. review 11: "including null-valued collision or contact evidence when
#: unavailable".
NO_CONTACT_MODEL = {'value': None,
                    'reason': 'the cached-view fast tier replays recorded frames and models no contact or '
                              'collision dynamics; a collision outcome exists only in a Gazebo/PX4 flight'}


def evidence_witness(policy, observation, receipts=()):
    """The PRIMITIVE evidence the ready and open sets are computed from, persisted so a reader can redo it.

    review 14: the sets were correct and unverifiable. Rewriting ten ready and ten open tracks to zero
    passed every check, because the validator could only compare one reported summary against another. A
    number nobody can recompute is an assertion.

    The witness is the state the executive's own rule reads: per track, how many detections the evidence
    policy counted and how many distinct credited view positions it held; whether the track was present in
    the last observation the policy decided on and where; and the local report identities and de-duplication
    positions that `is_reported` compares against. From this and the frozen thresholds, both sets follow.
    """
    detections = getattr(policy, '_detections', None)
    credited = getattr(policy, '_credited', None)
    if detections is None or credited is None:
        return None
    present = {track.track_id: [float(v) for v in track.position_enu]
               for track in (observation.tracks if observation is not None else ())}
    # review 16: the report list used to be read off the last observation, and it therefore agreed with
    # the episode's submissions only because no episode happened to end between a REPORT and the next
    # observation. `receipts` is the episode's OWN public receipt list, so the validator can require the two
    # to be the same thing by construction instead of by luck. Review 17 accepted that reading and named it:
    # track presence and position come from the last observation the policy decided on, report CLOSURE comes
    # from the final receipts, and the field says which.
    reports = [{'track_id': report.track_id,
                'dedup_position_xy': [float(v) for v in dedup_position_xy(report)]}
               for report in receipts]
    return {
        # review 15: the witness said it preserved distinct credited views and stored only their count,
        # which is a second summary rather than the evidence. The positions are here; the count is derived
        # from them and the frozen separation rule is checked against them.
        'tracks': sorted(({'track_id': track_id, 'detections': int(count),
                           'credited_views_xy': [[float(x), float(y)]
                                                 for x, y in credited.get(track_id, ())],
                           'position_enu': present.get(track_id)}
                          for track_id, count in detections.items()),
                         key=lambda entry: entry['track_id']),
        'final_public_receipts': reports,
        'rule': WITNESS_RULE,
    }


def sets_from_witness(witness, thresholds, decisions=None):
    """Recompute both sets from the witness alone. The validator and the runner use this same function.

    `decisions` is the number of decisions the POLICY made. Presence of a last observation is derived from it
    rather than read as a Boolean the witness supplies (review 15): an episode with recorded decisions
    had a last observation, and one with none did not.
    """
    if not witness or not thresholds:
        return [], []
    minimum_detections = thresholds.get('min_detections')
    minimum_views = thresholds.get('min_views')
    radius = thresholds.get('merge_radius_m')
    if minimum_detections is None or minimum_views is None or radius is None:
        return [], []
    if decisions is None:
        decisions = witness.get('policy_decisions')
    observed = bool(decisions)
    ready, open_reportable = [], []
    reports = witness.get('final_public_receipts') or []
    for entry in witness.get('tracks') or []:
        if entry['detections'] < minimum_detections or len(entry.get('credited_views_xy') or []) < minimum_views:
            continue
        ready.append(entry['track_id'])
        position = entry.get('position_enu')
        if position is None or not observed:
            continue                       # not in the last observation: not an open candidate there
        reported = any(report['track_id'] == entry['track_id']
                       or math.dist(report['dedup_position_xy'], position[:2]) <= radius
                       for report in reports)
        if not reported:
            open_reportable.append(entry['track_id'])
    return sorted(ready), sorted(open_reportable)


def report_ready(policy, observation, decisions=None, receipts=()):
    """Threshold-ready tracks, and the ones that were still OPEN by the executive's own rule.

    review 12 replaced `tracks_seen - reports_submitted`, which counted every low-evidence detection.
    Review 13 found the replacement still wrong in the other direction: it applied only the detection and view
    thresholds, while the evidence policy treats a track as open only when it is **also not already reported**
    - by track id or within `report_merge_radius_m` of an existing report. On the nominal EndBatch rerun all
    four tracks it called "held" had already been submitted under another id, 0.64 to 1.39 m away, so the
    policy had deliberately de-duplicated them. They were not held candidates.

    Two different quantities, named precisely:
      threshold_ready   met `report_min_detections` detections and `report_min_views` distinct credited views
      open_reportable   threshold-ready AND `is_reported` is false on the last observation the policy saw
    """
    config = getattr(policy, 'config', None)
    witness = evidence_witness(policy, observation, receipts)
    if config is None or witness is None:
        return [], [], None
    thresholds = {'min_detections': config.report_min_detections, 'min_views': config.report_min_views,
                  'merge_radius_m': config.report_merge_radius_m,
                  'view_separation_m': config.view_separation_m}
    ready, open_reportable = sets_from_witness(witness, thresholds, decisions=decisions)
    return ready, open_reportable, witness


def action_evidence(log):
    """The ordered public action record: one entry per loop iteration, with what the transition did.

    review 17: the decision list said what each action WAS and nothing about what the executive was
    allowed to do, so relabelling every record `forced` erased the policy's decisions, and deleting the list
    entirely left `valid_decisions` behind unchallenged. Two public streams already hold the facts, and this
    joins them without adding an opinion:

      `public/actions.jsonl`      one record per iteration - the action, its status, and any forced override
      `public/transitions.jsonl`  one record per EXECUTED action, in the same order - validity, termination,
                                  the terminal reason, and the observation pair whose report lists differ by
                                  exactly the receipts that action appended

    The receipt delta is the point: a report exists because a REPORT action ran and the public observation
    afterwards carried one more receipt than the one before it. `DecisionRecord.__post_init__` already refuses
    a transition whose report history is not append-only, so the delta is a suffix by construction.
    """
    records = log.records.get('public/actions.jsonl', []) if log is not None else []
    transitions = log.records.get('public/transitions.jsonl', []) if log is not None else []
    executed = [r for r in records if r['status'] == 'executed']
    if len(transitions) != len(executed):
        raise P3RunnerError(f'{len(executed)} actions were executed but {len(transitions)} transitions were '
                            'recorded; the episode did not record what it did')
    by_action, evidence = iter(transitions), []
    for index, record in enumerate(records, start=1):
        action = record['action']
        # review 18: the flattened identities were a summary of the payload, and `track_id` was not
        # even required - so a genuine REPORT could drop it and keep its receipt. Review 19: the flat copies
        # that replaced it were themselves unchecked, so a REPORT could carry `sector_id = "X"` beside a
        # payload with no sector. There is one action now, the payload, and the validator rebuilds it as a
        # genuine `contracts.Action` rather than re-implementing its rules.
        entry = {'index': index, 'action': dict(action), 'action_type': action['action_type'],
                 'status': record['status'], 'forced': record.get('forced'),
                 'valid': None, 'terminated': None, 'reason': None, 'receipts': []}
        if record['status'] == 'executed':
            transition = next(by_action)
            if transition['action']['action_type'] != entry['action_type']:
                raise P3RunnerError(f'action {index} was {entry["action_type"]} but its transition was '
                                    f'{transition["action"]["action_type"]}')
            before = transition['observation']['reports']
            after = (transition['next_observation'] or {}).get('reports', before)
            entry.update(valid=bool(transition['valid']), terminated=bool(transition['terminated']),
                         reason=transition['reason'],
                         receipts=[{'track_id': r['track_id'],
                                    'local_position_enu': [float(v) for v in r['local_position_enu']],
                                    'sim_time_s': float(r['sim_time_s'])} for r in after[len(before):]])
        evidence.append(entry)
    return evidence


def episode_diagnostics(log, outcome, service, policy, last_observation=None, policy_decisions=0):
    """The operational record `STATISTICAL_PROTOCOL_EN.md` section 4.4 asks for, persisted with every episode.

    Why it is in the row rather than in an optional stream: a paper record that depends on a flag somebody may
    forget is not a record. In particular the tight-budget end-batch case has to be readable from persisted
    fields alone - whether an OPEN reportable batch was lost to a forced return, or no track ever became
    reportable. That distinction needs `open_reportable_tracks`, which is threshold-readiness AND the
    executive's own de-duplication rule; see `report_ready`.
    """
    evidence = action_evidence(log)
    records = log.records.get('public/actions.jsonl', []) if log is not None else []
    executed = [r for r in records if r['status'] == 'executed']
    forced = [entry for entry in evidence if entry['forced']]
    counts = {}
    for entry in evidence:
        key = f"{entry['action_type']}/{entry['status']}"
        counts[key] = counts.get(key, 0) + 1

    fixes = [{'sim_time_s': m.sim_time_s, 'anchor_id': m.anchor_id, 'available': m.available,
              'accepted': m.accepted} for m in (service.measurements if service is not None else [])]
    submissions = [s.sim_time_s for s in (service.submissions if service is not None else [])]
    submitted_ids = sorted({s.track_id for s in (service.submissions if service is not None else [])})
    detections = dict(getattr(policy, '_detections', {}) or {})
    # review 16: the persisted decision count must BE the derived one, not a second opinion that
    # happens to agree. The executive writes one action record per loop iteration and calls the policy on
    # exactly the iterations it did not force, so this is the count, and the wrapper's own tally is only a
    # cross-check. If they ever disagree the episode is an executor failure, not a row to be argued about.
    derived_decisions = len(evidence) - len(forced)
    if policy_decisions is not None and policy_decisions != derived_decisions:
        raise P3RunnerError(f'the policy was called {policy_decisions} times but the action records say '
                            f'{derived_decisions}; the episode did not record what it did')
    ready_ids, open_ids, witness = report_ready(policy, last_observation, derived_decisions,
                                                receipts=outcome.reports)
    batch_state = getattr(policy, 'batch_state', None)
    return {
        'schema': DIAGNOSTICS_SCHEMA,
        'estimated_distance_m': outcome.estimated_distance_m,
        'elapsed_s': outcome.elapsed_s,
        'decisions': outcome.decisions, 'rejections': outcome.rejections,
        'valid_decisions': outcome.valid_decisions, 'invalid_decisions': outcome.invalid_decisions,
        'invalid_decision_rate': (outcome.invalid_decisions / outcome.decisions) if outcome.decisions else None,
        'action_counts': dict(sorted(counts.items())),
        # `action_sequence` was here and is gone (review 20): a projection of the executed actions
        # whose identities nothing compared with the actions themselves. `action_evidence` carries every
        # executed action in order, with its complete payload, so the projection was a second place to lie.
        # review 16 asked for the primitive every operational summary is derived from; review 17 asked
        # for enough of it to enforce the executive's own rules and to tie each report to the action that
        # made it. This is that record: ordered, one entry per loop iteration, carrying the transition's
        # validity and termination and the receipts the action appended. See `action_evidence`.
        'action_evidence': evidence,
        'forced_return_reason': forced[0]['forced'] if forced else None,
        'forced_return_count': len(forced),
        'fix_events': fixes,
        'fix_attempts': len(fixes), 'fix_accepted': sum(1 for f in fixes if f['accepted']),
        'anchor_visits': sorted({f['anchor_id'] for f in fixes}),
        'submission_times_s': submissions,
        'first_submission_s': submissions[0] if submissions else None,
        'last_submission_s': submissions[-1] if submissions else None,
        'tracks_seen': len(detections),
        'reports_submitted': len(submissions),
        'submitted_track_ids': submitted_ids,
        'threshold_ready_track_ids': ready_ids, 'threshold_ready_tracks': len(ready_ids),
        'open_reportable_track_ids': open_ids, 'open_reportable_tracks': len(open_ids),
        'evidence_witness': witness,
        'policy_decisions': derived_decisions,
        'unsubmitted_detections': max(len(detections) - len(submissions), 0),
        'report_thresholds': {'min_detections': getattr(getattr(policy, 'config', None),
                                                        'report_min_detections', None),
                              'min_views': getattr(getattr(policy, 'config', None), 'report_min_views', None),
                              'merge_radius_m': getattr(getattr(policy, 'config', None),
                                                        'report_merge_radius_m', None),
                              'view_separation_m': getattr(getattr(policy, 'config', None),
                                                           'view_separation_m', None),
                              'rule': OPEN_REPORTABLE_RULE},
        'final_batch_state': batch_state,
        'policy_parameters': dict(getattr(policy, 'parameters', {}) or {}),
        'collision_or_contact': NO_CONTACT_MODEL,
    }


def prepare_cell(frozen, cell):
    """Everything about a cell that does not depend on the episode: verified data, banks, the mission context with
    the cell budget in the BRIEFING, the frozen anchors, the drift configuration and the frozen score.

    P4 (learned arms) runs its episodes through this same function, so a trained policy meets exactly the cell a
    rule comparator met in P3 - one construction, not two that could drift. Refusals are unchanged.
    """
    protocol_path, bank_paths = verify_cell_data(cell)
    banks = open_banks([str(p) for p in bank_paths])
    # The budget reaches the BRIEFING, so the reserve, deadline, mask and score all share one number.
    context = fast_mission.load_context(protocol_path, banks,
                                        budget_overrides={'mission_time_s': cell['mission_budget_s']})
    applied = float(context['briefing'].budgets.mission_time_s)
    if abs(applied - cell['mission_budget_s']) > 1e-9:
        raise P3RunnerError(f"{cell['cell_id']}: the briefing budget is {applied}, not {cell['mission_budget_s']}")
    if context['scenario'].scenario_id != cell['data']['scenario_id']:
        raise P3RunnerError(f"{cell['cell_id']}: scenario {context['scenario'].scenario_id!r} is not "
                            f"{cell['data']['scenario_id']!r}")
    targets = len(context['evaluator'].targets)
    expected_targets = cell['n_ref'] if cell['presence'] == 'positive' else 0
    if targets != expected_targets:
        raise P3RunnerError(f"{cell['cell_id']}: the evaluator declares {targets} targets, "
                            f'{cell["presence"]!r} requires {expected_targets}')

    anchors = AnchorSet.from_dict(frozen['anchors'])
    georeference_config = GeoreferenceConfig.from_dict(cell['georeference'])
    if canonical_sha256(georeference_config.to_dict()) != cell['georeference_sha256']:
        raise P3RunnerError(f"{cell['cell_id']}: the drift configuration differs from its manifest hash")
    score = TimelyScoreSpec(SCORE_VERSION, mission_budget_s=applied, n_ref=cell['n_ref'],
                            tolerance_m=frozen['score']['tolerance_m'], metric=frozen['score']['metric'],
                            false_report_weight=frozen['score']['false_report_weight'], provenance='frozen')
    return {'banks': banks, 'context': context, 'applied_budget_s': applied, 'anchors': anchors,
            'georeference_config': georeference_config, 'score': score}


def cell_episode_parts(static, cell, namespace, seed):
    """The per-episode stochastic parts of a prepared cell: backend, CRN group and a fresh georeference service.

    The disturbance identity comes from the PAIR, never from the cell (see `crn_group`), and the re-observation
    seed is the episode seed, exactly as every P3 episode has it.
    """
    backend_config = CachedViewConfig(**dict(BACKEND, reobservation_seed=seed))
    check_anchor_safety(static['anchors'], static['banks'], backend_config)
    group = crn_group(cell['pair_id'], namespace, seed)
    service = GeoreferenceService(static['georeference_config'], crn_path_key(group))
    if service.disturbance_identity['config_sha256'] != canonical_sha256(static['georeference_config'].to_dict()):
        raise P3RunnerError(f"{cell['cell_id']}: the service reports another configuration than the cell's")
    return backend_config, group, service


def run_cell_episode(frozen, manifest, cell, comparator, namespace, realization, time_model, out_dir, log_streams):
    """One P3 episode. Every refusal below happens before a policy exists or a macro runs."""
    if namespace not in ALLOWED_NAMESPACES:
        raise P3RunnerError(f'namespace {namespace!r} is not open; allowed: {ALLOWED_NAMESPACES}. '
                            'Tuning and gate seeds require a further review.')
    seeds = cell['seeds'][namespace]
    if realization >= len(seeds):
        raise P3RunnerError(f"{cell['cell_id']}: namespace {namespace!r} has {len(seeds)} seeds, "
                            f'realisation {realization} requested')
    seed = seeds[realization]

    static = prepare_cell(frozen, cell)
    banks, context, anchors, score = static['banks'], static['context'], static['anchors'], static['score']
    applied = static['applied_budget_s']
    backend_config, group, service = cell_episode_parts(static, cell, namespace, seed)
    georeference_config = static['georeference_config']
    identity = service.disturbance_identity
    # The candidate identity is in the episode ID and therefore in the output path, so two parameter variants
    # of one policy cannot collide (review 9).
    episode_id = episode_identity(comparator['candidate_index'], comparator['candidate_sha256'],
                                  comparator['name'], cell['cell_id'], namespace, realization)

    from aerial_qrl.mission.executive import MemoryLog
    # The log is ALWAYS kept in memory: the per-episode diagnostics below are read from it. `log_streams` only
    # decides whether it is also written to disk, and that is refused outside the pilot namespace.
    log = MemoryLog()
    built_policy = {}

    def policy_factory(briefing, obstacle_map, config, reserve):
        instance = build_policy(comparator['name'], briefing, obstacle_map, config, reserve,
                                **comparator['parameters'])
        # Record the last observation the policy decided on, without changing what it decides. Readiness has
        # to use the executive's own `is_reported` rule, and that rule needs the reports the policy could see.
        decide = instance.act

        def act(observation):
            built_policy['last_observation'] = observation
            built_policy['policy_decisions'] = built_policy.get('policy_decisions', 0) + 1
            return decide(observation)

        instance.act = act
        built_policy['instance'] = instance
        return instance
    started = time.perf_counter()
    result = fast_mission.run_fast_episode(
        context, banks, time_model, episode_id, log=log, backend_config=backend_config,
        executive_kwargs={'georeference': service, 'anchor_set': anchors},
        policy_factory=policy_factory,
        reports_for_matching=service.global_report_positions)
    wall = time.perf_counter() - started

    outcome = result['outcome']
    return_ok = outcome.termination_reason == 'mission_complete' and outcome.returned and outcome.landed
    breakdown = episode_rewards(service.global_report_positions(), [s.sim_time_s for s in service.submissions],
                                context['evaluator'].targets, score, return_ok)
    check_reward_identity(breakdown)

    row = {
        'episode_id': episode_id, 'runner': RUNNER_VERSION,
        'frozen_manifest_sha256': frozen['manifest_sha256'],
        'execution_manifest_sha256': manifest['manifest_sha256'],
        'cell_id': cell['cell_id'], 'cell_index': cell['cell_index'], 'pair_id': cell['pair_id'],
        'presence': cell['presence'], 'd600_rms_m': cell['d600_rms_m'],
        'georeference_sha256': cell['georeference_sha256'],
        'budget_name': cell['budget_name'],
        'mission_budget_s_requested': cell['mission_budget_s'],
        'mission_budget_s_applied': applied,
        'cell_weight': cell['weight'], 'n_ref': cell['n_ref'],
        'protocol_path': cell['data']['protocol_path'], 'protocol_sha256': cell['data']['protocol_sha256'],
        'bank_manifest_sha256': [p['manifest_sha256'] for p in cell['data']['bank_parts']],
        'score_spec': {'score_version': SCORE_VERSION, 'reward_version': REWARD_VERSION,
                       'n_ref': cell['n_ref'], 'mission_budget_s': applied,
                       'tolerance_m': frozen['score']['tolerance_m'], 'metric': frozen['score']['metric'],
                       'false_report_weight': frozen['score']['false_report_weight'],
                       'timing_weight': frozen['score']['timing_weight'], 'provenance': 'frozen'},
        'seed_namespace': namespace, 'realization': realization, 'seed': seed,
        'reobservation_seed': backend_config.reobservation_seed,
        'crn_version': group['crn_version'], 'crn_label': group['crn_label'], 'crn_sha256': group['crn_sha256'],
        'disturbance_path_key': identity['path_key'], 'disturbance_path_sha256': identity['path_sha256'],
        'measurement_noise_key': identity['noise_key'], 'measurement_noise_sha256': identity['noise_sha256'],
        'comparator': comparator['name'], 'comparator_parameters': comparator['parameters'],
        'comparator_protocol_sha256': comparator['protocol_sha256'],
        'candidate_sha256': comparator['candidate_sha256'], 'candidate_index': comparator['candidate_index'],
        'candidate_role': comparator['role'],
        'termination_reason': outcome.termination_reason, 'returned': outcome.returned,
        'landed': outcome.landed, 'return_ok': return_ok, 'decisions': outcome.decisions,
        'rejections': outcome.rejections, 'elapsed_s': outcome.elapsed_s,
        'score_J': breakdown.score, 'reward_sum': breakdown.total(),
        'reward_identity_holds': abs(breakdown.total() - breakdown.score) <= 1e-9,
        'components': breakdown.final.to_dict(),
        'fixes_attempted': len(service.measurements),
        'fixes_accepted': sum(m.accepted for m in service.measurements),
        'submissions': [s.to_dict() for s in service.submissions],
        'public_receipt_types': sorted({type(r).__name__ for r in outcome.reports}),
        'bank': {k: result['stats'][k] for k in ('frames', 'bank_frames', 'off_node_frames',
                                                 'repeated_capture_frames')},
        'wall_s': wall,
        'classification': CLASSIFICATION[namespace],
        'diagnostics': episode_diagnostics(log, outcome, service, built_policy.get('instance'),
                                           built_policy.get('last_observation'),
                                           built_policy.get('policy_decisions', 0)),
        'public_reports': [{'track_id': r.track_id,
                            'local_position_enu': [float(v) for v in r.local_position_enu]}
                           for r in outcome.reports],
    }
    if log_streams:
        # Exclusive create, never `exist_ok` and never mode 'w': a second run of the same episode id must fail
        # loudly rather than truncate a stream somebody is citing (review 11).
        stream_dir = out_dir / 'streams' / episode_id
        stream_dir.mkdir(parents=True, exist_ok=False)
        for name, prefix in (('public.jsonl', False), ('evaluator.jsonl', True)):
            with (stream_dir / name).open('x') as handle:
                for stream, records in sorted(log.records.items()):
                    if stream.startswith('evaluator/') != prefix:
                        continue
                    for record in records:
                        handle.write(json.dumps({'stream': stream, 'record': record}, sort_keys=True) + '\n')
        row['streams'] = str(stream_dir.relative_to(out_dir))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--execution-manifest', required=True, type=Path)
    parser.add_argument('--time-model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--namespace', required=True, choices=('pilot', 'tuning', 'gate'))
    parser.add_argument('--comparator-protocol', required=True, type=Path)
    parser.add_argument('--candidate-sha256', required=True,
                        help='the exact frozen candidate hash; a policy name alone is refused, because one '
                             'name can be several candidates in the grid')
    parser.add_argument('--selection', type=Path,
                        help='the frozen reference-selection artifact; REQUIRED for the gate namespace and '
                             'refused for any other, so a gate number can only ever score the selected '
                             'reference or the fixed challenger')
    parser.add_argument('--pair', required=True, help='pair_id, e.g. d2__nominal_2700s; both members are run')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this runner never overwrites a run directory')
    if args.namespace not in ALLOWED_NAMESPACES:
        raise SystemExit(f'namespace {args.namespace!r} is not open; allowed: {ALLOWED_NAMESPACES}. '
                         'Tuning and gate seeds require a further review.')

    frozen, manifest = load_frozen(args.freeze, args.execution_manifest)
    protocol = load_comparator_protocol(args.comparator_protocol)
    candidate = resolve_candidate(protocol, args.candidate_sha256)
    selection = require_gate_binding(args.namespace, candidate, protocol, args.selection)
    if sha256_file(args.time_model) != manifest['time_model_sha256']:
        raise SystemExit('the time model differs from the execution manifest hash')
    members = manifest['pairs'].get(args.pair)
    if not members:
        raise SystemExit(f'unknown pair {args.pair!r}; known: {sorted(manifest["pairs"])}')
    cells = [c for c in manifest['cells'] if c['cell_id'] in members]
    comparator = dict(candidate, protocol_sha256=protocol['protocol_sha256'])
    time_model = TimeModel.from_file(args.time_model)

    args.out.mkdir(parents=True)
    rows = [run_cell_episode(frozen, manifest, cell, comparator, args.namespace, args.realization,
                             time_model, args.out, args.log) for cell in cells]
    payload = {
        'runner': RUNNER_VERSION,
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'classification': CLASSIFICATION[args.namespace],
        'code_tree_sha256': code_tree_hash(),
        'source_sha256': {path: sha256_file(ROOT / path) for path in SOURCES},
        'frozen_manifest_sha256': frozen['manifest_sha256'],
        'execution_manifest_sha256': manifest['manifest_sha256'],
        'namespace': args.namespace, 'pair': args.pair, 'realization': args.realization,
        'comparator_protocol_sha256': protocol['protocol_sha256'],
        'candidate': {k: candidate[k] for k in ('candidate_index', 'candidate_sha256', 'name', 'parameters',
                                                'role')},
        'reference_selection': selection,
        'backend': BACKEND,
        'episodes': rows,
        'pairing': paired_state(rows),
    }
    if payload['pairing']['paired'] is not True:
        raise SystemExit(f"refusing to write: the pair members do not share a disturbance -> {payload['pairing']}")
    (args.out / f'p3_{args.namespace}_episodes.json').write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 'namespace': args.namespace, 'pair': args.pair,
                      'pairing': payload['pairing'],
                      'episodes': [{'cell_id': r['cell_id'], 'presence': r['presence'],
                                    'budget_applied': r['mission_budget_s_applied'], 'seed': r['seed'],
                                    'termination': r['termination_reason'], 'J': round(r['score_J'], 4),
                                    'reward_identity': r['reward_identity_holds'],
                                    'crn_sha256': r['crn_sha256'][:16],
                                    'tp': r['components']['tp_spatial'], 'fp': r['components']['fp_spatial']}
                                   for r in rows]}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
