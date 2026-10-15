"""Freeze the comparator candidate set, its parameters, the selection score and the tie rule (review 8).

Why this is frozen BEFORE any tuning run
    `STATISTICAL_PROTOCOL_EN.md` section 5.3 requires the reference rule to be the strongest tuned eligible
    simple rule, selected on tuning realisations and then evaluated ONCE on gate realisations. If the candidate
    list, the parameter grid, the scalar used to choose or the tie rule were settled after seeing tuning
    numbers, the selection would be a choice made with the data, and the gate would no longer mean what the
    protocol says it means. Freezing them in a hashed artifact makes the order auditable rather than asserted.

    Nothing here runs anything. No tuning outcome exists and no comparator has been selected.

The selection scalar, stated once
    `mean J over the tuning realisations, weighted by the frozen cell weights` - the same score and the same
    weights the gate uses, so the selection cannot be optimising a different quantity from the one reported.

The tie rule, deterministic and declared in advance
    1. higher weighted mean J;
    2. then lower weighted mean FP_spatial (a rule that ties on J but reports fewer false coordinates is
       preferred, which is the operational direction the score's own lambda already expresses);
    3. then fewer mean fix attempts (a rule that buys the same outcome with less time is preferred);
    4. then the candidate's position in the frozen `CANDIDATES` order, which is fixed here and never sorted by
       any observed quantity.
    Step 4 alone makes the rule total: there is always exactly one winner, and it never depends on dictionary
    iteration order or on floating-point noise beyond steps 1 to 3.

Common random numbers
    Every candidate is evaluated on the SAME CRN groups, because `run_p3_cells.crn_group` excludes the
    comparator from the disturbance identity. That is recorded here so a reader does not have to infer it.

Eligibility
    A candidate is eligible only if it uses public information alone. `gnss_policies` is asserted by test to
    import nothing from the evaluator side, which is what makes the whole set eligible; a future candidate that
    needed truth would have to be rejected rather than granted an exception.
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

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts/gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from aerial_qrl.training.checkpoint_identity import canonical_sha256              # noqa: E402
import gnss_policies                                                              # noqa: E402
from gnss_policies import COMPARATORS                                             # noqa: E402
import run_p3_cells as p3_runner                                                  # noqa: E402

PROTOCOL_VERSION = 'gnss-comparator-protocol/3'
#: The policy source this protocol is bound to. review 10: a candidate hash names a name and its
#: parameters and says nothing about what the code does with them, so `end_batch_at_home` could change from
#: "one home fix for the batch" to "a fix before nearly every report" without a single hash moving. The
#: protocol therefore carries both the behaviour version and the exact source hash, and `verify_policy_binding`
#: refuses a live tree that does not match. Candidate identity is never used without protocol identity.
POLICY_SOURCE = ROOT / 'scripts/gnss_qrl_runtime/gnss_policies.py'
#: Roles, frozen. review 9: the planner must never be eligible to become the simple-rule reference, or
#: the gate would compare the planner with itself and prove nothing. Section 5.3 asks for a nontrivial strategy
#: against the strongest tuned eligible SIMPLE rule.
ROLE_REFERENCE = 'reference_eligible'
ROLE_CHALLENGER = 'challenger'
#: The protocol's own report hold. The planner budgets a report dwell; using anything else would make its
#: lookahead disagree with the mission it is planning for.
PROTOCOL_REPORT_HOLD_S = 1.0
#: The planner's remaining approximations, disclosed rather than hidden. A gate failure by this planner cannot
#: establish that the task has no decision value, and that sentence belongs in the paper.
PLANNER_APPROXIMATIONS = (
    'the posterior RMS after a fix omits the covariance growth during the travel to the anchor, so the planner '
    'is slightly optimistic about what a fix will buy',
    'the utility maximises timely true-positive credit and does not subtract the false-report penalty, so it '
    'does not price a report that lands outside the tolerance',
    'submission happens at the START of the report macro, and the planner budgets a dwell rather than modelling '
    'that ordering',
    'one-step lookahead only: it never considers a sequence of fixes',
)

#: The candidate set and its parameter grid, in a FIXED order that the tie rule's last step depends on.
#: Every value is a declared rule parameter, not a threshold tuned against results. A grid entry is one
#: candidate: `uncertainty_fix` at two thresholds is two candidates, not one candidate with a free parameter.
CANDIDATES = (
    {'name': 'never_fix', 'parameters': {}, 'role': ROLE_REFERENCE,
     'why': 'the floor: it shows what drift alone costs and cannot be beaten by doing less'},
    {'name': 'always_fix_before_report', 'parameters': {}, 'role': ROLE_REFERENCE,
     'why': 'the trivial rule the 18 Sep pricing found hard to beat; the reference to beat, not a strawman'},
    {'name': 'end_batch_at_home', 'parameters': {}, 'role': ROLE_REFERENCE,
     'why': 'delayed batch submission, which the model contract admits and section 5.3 names explicitly'},
    {'name': 'periodic_fix', 'parameters': {'stale_after_s': 150.0}, 'role': ROLE_REFERENCE,
     'why': 'staleness rule, short period'},
    {'name': 'periodic_fix', 'parameters': {'stale_after_s': 300.0}, 'role': ROLE_REFERENCE,
     'why': 'staleness rule, medium period'},
    {'name': 'periodic_fix', 'parameters': {'stale_after_s': 600.0}, 'role': ROLE_REFERENCE,
     'why': 'staleness rule, long period'},
    {'name': 'uncertainty_fix', 'parameters': {'sigma_threshold_m': 0.75}, 'role': ROLE_REFERENCE,
     'why': 'belief rule at a quarter of the 3.0 m tolerance'},
    {'name': 'uncertainty_fix', 'parameters': {'sigma_threshold_m': 1.5}, 'role': ROLE_REFERENCE,
     'why': 'belief rule at half the tolerance'},
    {'name': 'uncertainty_fix', 'parameters': {'sigma_threshold_m': 2.25}, 'role': ROLE_REFERENCE,
     'why': 'belief rule at three quarters of the tolerance'},
    {'name': 'short_horizon_planner',
     'parameters': {'measurement_sigma_m': 0.5, 'report_dwell_s': PROTOCOL_REPORT_HOLD_S, 'tolerance_m': 3.0},
     'role': ROLE_CHALLENGER,
     'why': ('the nontrivial strategy the gate evaluates against the selected reference. It is a heuristic, '
             'not an optimal planner, and it is NEVER reference-eligible: selecting it as the reference would '
             'compare it with itself. Its three parameters are supplied here explicitly so no hidden default '
             'can reach a scientific run; see PLANNER_APPROXIMATIONS for what it still omits.')},
)

SELECTION = {
    'scalar': 'weighted_mean_J',
    'definition': ('mean J over the tuning realisations of each core cell, combined with the frozen cell '
                   'weights; the same score and weights the gate reports'),
    'namespace': 'tuning',
    'evaluated_once_on': 'gate',
    'selection_pool': 'reference_eligible candidates only; the challenger never enters the selection',
    'completeness_required': ('exactly one finite summary per reference-eligible candidate, each covering all '
                              'eight cells and twenty tuning realisations; anything else is a refusal'),
    'tie_rule': [
        {'step': 1, 'key': 'weighted_mean_J', 'direction': 'higher is better'},
        {'step': 2, 'key': 'weighted_mean_fp_spatial', 'direction': 'lower is better'},
        {'step': 3, 'key': 'mean_fix_attempts', 'direction': 'lower is better'},
        {'step': 4, 'key': 'candidate_index', 'direction': 'lower is better',
         'note': 'the frozen CANDIDATES order; it makes the rule total and depends on no observed quantity'},
    ],
    'common_random_numbers': ('every candidate meets the same CRN groups: run_p3_cells.crn_group excludes the '
                              'comparator from the disturbance identity, so the comparison is paired across '
                              'candidates as well as across presence members'),
    'forbidden': [
        'adding, removing or re-parameterising a candidate after seeing any tuning number',
        'choosing a different scalar or tie rule after seeing any tuning number',
        'selecting on gate realisations, or reporting a gate verdict for a rule selected on the same seeds',
        'selecting the challenger as the reference, or ranking it inside the selection pool',
        'selecting from an incomplete table, or from one with a missing, duplicated or non-finite summary',
    ],
}


def policy_source_sha256():
    return hashlib.sha256(POLICY_SOURCE.read_bytes()).hexdigest()


def verify_policy_binding(protocol):
    """Refuse a protocol whose policies are not the ones on disk. Called before any scientific episode."""
    live_version = gnss_policies.POLICY_BEHAVIOUR_VERSION
    live_source = policy_source_sha256()
    if protocol.get('policy_behaviour_version') != live_version:
        raise ValueError(f"the protocol was frozen against policy behaviour "
                         f"{protocol.get('policy_behaviour_version')!r}, the live policies are {live_version!r}")
    if protocol.get('policy_source_sha256') != live_source:
        raise ValueError(f"the protocol was frozen against gnss_policies.py "
                         f"{str(protocol.get('policy_source_sha256'))[:12]}, the live file is "
                         f"{live_source[:12]}; a comparator's behaviour is part of its identity")
    return {'policy_behaviour_version': live_version, 'policy_source_sha256': live_source}


def candidate_key(candidate):
    return canonical_sha256({'name': candidate['name'], 'parameters': candidate['parameters']})


def reference_candidates():
    """The candidates that may become the simple-rule reference, in frozen order."""
    return tuple(c for c in CANDIDATES if c['role'] == ROLE_REFERENCE)


def challenger_candidates():
    return tuple(c for c in CANDIDATES if c['role'] == ROLE_CHALLENGER)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def select(rows, expected_cells=8, expected_realizations=20):
    """Pick the reference rule from a COMPLETE table of reference-eligible candidates only.

    review 9 found two holes in the first version and both are closed here. It pooled all ten candidates,
    so a high-scoring `short_horizon_planner` could be returned as the reference - which would compare the
    challenger with itself and destroy the gate. And it accepted a single row as a winner, so an incomplete or
    half-failed campaign could still name one.

    This refuses, rather than ranks, when:
      * a row names a candidate outside the frozen set, or a challenger;
      * any reference-eligible candidate is missing, or appears more than once;
      * any selection quantity is missing, non-numeric, NaN or infinite;
      * a row does not cover the expected cells and realisations.
    """
    eligible = reference_candidates()
    index = {candidate_key(c): i for i, c in enumerate(CANDIDATES)}
    challengers = {candidate_key(c) for c in challenger_candidates()}
    problems = []

    seen = {}
    for row in rows:
        key = candidate_key(row)
        if key not in index:
            problems.append(f'row names a candidate outside the frozen set: {row.get("name")!r} {row.get("parameters")!r}')
            continue
        if key in challengers:
            problems.append(f'{row["name"]!r} is a {ROLE_CHALLENGER} and can never be the reference')
            continue
        if key in seen:
            problems.append(f'duplicate summary for {row["name"]!r} {row.get("parameters")!r}')
            continue
        seen[key] = row
        for field in ('weighted_mean_J', 'weighted_mean_fp_spatial', 'mean_fix_attempts'):
            if not _finite(row.get(field)):
                problems.append(f'{row["name"]!r}: {field} is not a finite number ({row.get(field)!r})')
        if row.get('cells_covered') != expected_cells:
            problems.append(f'{row["name"]!r}: covers {row.get("cells_covered")!r} cells, expected {expected_cells}')
        if row.get('realizations_per_cell') != expected_realizations:
            problems.append(f'{row["name"]!r}: {row.get("realizations_per_cell")!r} realisations per cell, '
                            f'expected {expected_realizations}')

    missing = [f'{c["name"]} {c["parameters"]}' for c in eligible if candidate_key(c) not in seen]
    if missing:
        problems.append(f'missing summaries for reference-eligible candidates: {missing}')
    if problems:
        raise ValueError('refusing to select: ' + '; '.join(problems))

    ordered = sorted(seen.values(), key=lambda r: (-r['weighted_mean_J'], r['weighted_mean_fp_spatial'],
                                                   r['mean_fix_attempts'], index[candidate_key(r)]))
    return {'winner': ordered[0], 'winner_sha256': candidate_key(ordered[0]), 'ranking': ordered,
            'decided_at_step': _decided_at_step(ordered) if len(ordered) > 1 else 4,
            'challenger': challenger_candidates()[0],
            'challenger_sha256': candidate_key(challenger_candidates()[0])}


def _decided_at_step(ordered):
    first, second = ordered[0], ordered[1]
    if first['weighted_mean_J'] != second['weighted_mean_J']:
        return 1
    if first['weighted_mean_fp_spatial'] != second['weighted_mean_fp_spatial']:
        return 2
    if first['mean_fix_attempts'] != second['mean_fix_attempts']:
        return 3
    return 4


def build():
    protocol = {
        'protocol_version': PROTOCOL_VERSION,
        'task_id': 'flood-sar-village-gnss/1',
        'status': ('frozen selection protocol only; NO tuning run has been executed and NO comparator has been '
                   'selected'),
        'candidates': [dict(c, candidate_index=i, candidate_sha256=candidate_key(c))
                       for i, c in enumerate(CANDIDATES)],
        'candidate_count': len(CANDIDATES),
        'reference_eligible_count': len(reference_candidates()),
        'challenger_count': len(challenger_candidates()),
        'roles': {'reference_eligible': [candidate_key(c) for c in reference_candidates()],
                  'challenger': [candidate_key(c) for c in challenger_candidates()]},
        'role_rule': ('section 5.3 evaluates a nontrivial strategy against the strongest tuned eligible SIMPLE '
                      'rule. The planner is the nontrivial strategy and is never reference-eligible: selecting '
                      'it as the reference would compare it with itself and the gate would prove nothing.'),
        'planner_scope': {
            'is_optimal': False,
            'description': 'a one-step-lookahead heuristic on public quantities only',
            'report_dwell_s': PROTOCOL_REPORT_HOLD_S,
            'report_dwell_source': "the protocol's own mission.config.report_hold_s",
            'approximations': list(PLANNER_APPROXIMATIONS),
            'consequence': ('a gate failure by this planner cannot establish that the task has no decision '
                            'value; it establishes that THIS heuristic did not beat the reference'),
        },
        'policy_behaviour_version': gnss_policies.POLICY_BEHAVIOUR_VERSION,
        'policy_source_sha256': policy_source_sha256(),
        'policy_binding_rule': ('a candidate hash names a name and its parameters only. The behaviour version '
                                'and the source hash above are what bind it to a behaviour, and every '
                                'scientific run verifies them against the live tree before an episode exists. '
                                'Candidate identity is never used without this protocol identity.'),
        'end_batch_strategy': {
            'phases': [gnss_policies.BATCH_SEARCHING, gnss_policies.BATCH_FIXING, gnss_policies.BATCH_RELEASING],
            'rule': ('search and finish the evidence work holding every reportable candidate; make at most ONE '
                     'paid home-fix attempt for the whole batch; release the prepared reports consecutively '
                     'with no further search, inspection or fix; return'),
            'fallback_rule': gnss_policies.BATCH_FALLBACK_RULE,
            'fallback': ('if the home anchor is unreachable, or the one attempt fails, the batch is released '
                         'unfixed and the vehicle returns; the anchor is never retried'),
            'known_risk': ('holding the batch means the return reserve can end the mission before any report '
                           'is released, and the batch is then lost; that is the strategy\'s own cost and no '
                           'early-release threshold is added to hide it'),
            'corrected_in': 'review 10 blocker 1; the previous behaviour paid a fix before nearly every report',
        },
        'distinct_policies': sorted({c['name'] for c in CANDIDATES}),
        'all_policies_are_implemented': sorted({c['name'] for c in CANDIDATES}) == sorted(
            set(COMPARATORS) & {c['name'] for c in CANDIDATES}),
        'eligibility': ('public information only; gnss_policies imports nothing from the evaluator side, which '
                        'a syntax-tree test asserts. A candidate needing truth would be rejected, not excused.'),
        'selection': SELECTION,
        'crn_version': p3_runner.CRN_VERSION,
        'criterion_reference': ('the GATE criterion is STATISTICAL_PROTOCOL_EN.md section 5.3 and is not '
                                'restated here, so it cannot drift; this document fixes only how the reference '
                                'rule is chosen before that criterion is applied'),
    }
    protocol['protocol_sha256'] = canonical_sha256(protocol)
    return protocol


def code_tree_hash():
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/code_version.py')], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this tool never overwrites a protocol directory')
    protocol = build()
    if not protocol['all_policies_are_implemented']:
        raise SystemExit('a frozen candidate names a policy that gnss_policies does not implement')
    payload = {'protocol': PROTOCOL_VERSION,
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               'code_tree_sha256': code_tree_hash(),
               'comparator_protocol': protocol}
    args.out.mkdir(parents=True)
    (args.out / 'comparator_protocol.json').write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 'protocol_sha256': protocol['protocol_sha256'],
                      'policy_behaviour_version': protocol['policy_behaviour_version'],
                      'policy_source_sha256': protocol['policy_source_sha256'],
                      'candidates': protocol['candidate_count'],
                      'distinct_policies': protocol['distinct_policies'],
                      'selection_scalar': SELECTION['scalar'],
                      'tie_rule_steps': len(SELECTION['tie_rule'])}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
