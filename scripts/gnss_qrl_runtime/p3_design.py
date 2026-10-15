"""The P3 task-relevance suite: the eight core cells, the derived tight budget and the seed namespaces.

This file only DESIGNS and records. It runs no gate and decides nothing; `run_gnss_episodes.py` executes cells
and a separate verdict tool will apply the criterion once, after the reference rule has been selected on
disjoint tuning realisations.

The design is `STATISTICAL_PROTOCOL_EN.md` section 3.2 taken literally, after review 4 ruled out running
P3 on the D600 = 8 m stress condition or on an ad hoc tighter budget chosen from those results:

    drift      D600 = 2 m and 4 m                    (8 m and 0 m are FINAL-ONLY stress, not core)
    budget     2700 s, and B_tight from the rule below
    presence   the positive scenario and its paired no-human scenario
    -> 2 x 2 x 2 = 8 core cells; weights 0.8 over the four positive cells, 0.2 over the four negative ones,
       uniform inside each group. They are a benchmark convention, not a population estimate.

The tight budget, computed once and never swept
    B_tight = ceil(1.10 * T_route_no_fix) seconds, where T_route_no_fix is the simulator time of ONE fixed,
    untuned, target-independent route: the protocol's own coverage plan flown to completion with no fix, no
    report and no inspection, plus the return and the landing. The route and the multiplier are recorded BEFORE
    anyone knows which method wins. `derive_tight_budget` measures it by running exactly that episode.

    Two conditions are checked and recorded rather than assumed: B_tight must be shorter than 2700 s, and a
    safe return must still be affordable from the allowed states. If either fails, the geometry/budget design
    has to be resolved and versioned before the lock; the function reports the failure instead of adjusting the
    multiplier.

Seed namespaces, disjoint by construction
    A seed that selects a reference rule must never score it. The namespaces below are disjoint integer ranges
    derived from one base, and `seed_namespaces` records them so the separation is checkable rather than
    promised. Reuse ACROSS arms inside a namespace is intentional (common random numbers); reuse BETWEEN
    selection and evaluation is prohibited.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts/gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from aerial_qrl.mission.contracts import Action, ActionType                          # noqa: E402
from aerial_qrl.mission.executive import CoverageBaselinePolicy, MissionPolicy       # noqa: E402

P3_DESIGN_VERSION = 'gnss-p3-design/2'
SOURCES = ('scripts/gnss_qrl_runtime/p3_design.py', 'src/aerial_qrl/mission/executive.py',
           'src/aerial_qrl/mission/planning.py', 'src/aerial_qrl/mission/cached_view_backend.py',
           'src/aerial_qrl/mission/time_model.py', 'scripts/fast_mission.py')


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_hash():
    import subprocess
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/code_version.py')], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or None
#: Core factor levels (statistical protocol section 3.2). 0 m and 8 m are FINAL-ONLY stress conditions.
CORE_D600_M = (2.0, 4.0)
CORE_PRESENCE = ('positive', 'no_human')
TIGHT_BUDGET_MULTIPLIER = 1.10
NOMINAL_BUDGET_S = 2700.0
POSITIVE_TOTAL_WEIGHT, NEGATIVE_TOTAL_WEIGHT = 0.8, 0.2
#: Section 5.3: at least 20 paired disturbance realisations per development core cell.
GATE_REALIZATIONS_PER_CELL = 20
TUNING_REALIZATIONS_PER_CELL = 20

#: Disjoint namespaces. Width 100000 is far above any realisation count this study can afford, so the ranges
#: cannot overlap by accident.
NAMESPACE_WIDTH = 100000
NAMESPACES = ('tuning', 'gate', 'pilot', 'main_training', 'final_evaluation', 'stress', 'demonstration')
#: One block of seeds per cell inside a namespace. `seed_for` REFUSES a realisation outside its block rather
#: than letting it alias into the neighbouring cell: (cell 0, realisation 1000) must not become (cell 1,
#: realisation 0), and a negative realisation must not reach into the previous cell. The planned 0..19 never
#: come close, but a reusable scheduler has to reject out-of-domain input instead of relying on that.
CELL_BLOCK_STRIDE = 1000
CORE_CELL_COUNT = 8


def seed_namespaces(base=20260921):
    """Disjoint [start, end) integer ranges, one per purpose, recorded so the separation can be audited."""
    spaces = {}
    for index, name in enumerate(NAMESPACES):
        start = base * 1000 + index * NAMESPACE_WIDTH
        spaces[name] = {'start': start, 'end': start + NAMESPACE_WIDTH}
    return spaces


def seed_for(namespaces, namespace, cell_index, realization, cell_count=CORE_CELL_COUNT,
             stride=CELL_BLOCK_STRIDE):
    """A seed inside its namespace and inside its cell block. Every out-of-domain input is refused.

    The domain is checked BEFORE the arithmetic, so an alias is impossible rather than merely unlikely:
    without this, (cell 0, realisation `stride`) would land on (cell 1, realisation 0) and a negative
    realisation would reach into the previous cell's block.
    """
    if namespace not in namespaces:
        raise ValueError(f'namespace: expected one of {sorted(namespaces)}, got {namespace!r}')
    if not isinstance(cell_index, int) or isinstance(cell_index, bool) or not 0 <= cell_index < cell_count:
        raise ValueError(f'cell_index: expected an integer in [0, {cell_count}), got {cell_index!r}')
    if not isinstance(realization, int) or isinstance(realization, bool) or not 0 <= realization < stride:
        raise ValueError(f'realization: expected an integer in [0, {stride}) so it cannot alias into the next '
                         f'cell block, got {realization!r}')
    space = namespaces[namespace]
    seed = space['start'] + cell_index * stride + realization
    if not space['start'] <= seed < space['end']:
        raise ValueError(f'{namespace}: seed {seed} falls outside {space}; widen the namespace instead of wrapping')
    return seed


def core_cells(tight_budget_s):
    """The eight core cells with their weights, in a fixed order."""
    cells = []
    for d600 in CORE_D600_M:
        for budget, budget_name in ((NOMINAL_BUDGET_S, 'nominal_2700s'), (tight_budget_s, 'tight_derived')):
            for presence in CORE_PRESENCE:
                weight = (POSITIVE_TOTAL_WEIGHT if presence == 'positive' else NEGATIVE_TOTAL_WEIGHT) / 4.
                cells.append({'name': f'd{d600:g}__{budget_name}__{presence}', 'd600_rms_m': d600,
                              'budget_s': budget, 'budget_name': budget_name, 'presence': presence,
                              'weight': weight})
    total = sum(c['weight'] for c in cells)
    assert abs(total - 1.0) < 1e-12 and len(cells) == 8
    return cells


class FixedRoutePolicy(MissionPolicy):
    """The untuned, target-independent route: fly the coverage plan to completion, then return.

    It never reports, never inspects and never relocalises, so its duration depends on the scene geometry and
    the time model alone. That is exactly what `B_tight` must be derived from.
    """
    name = 'fixed_route_no_fix'

    def __init__(self, briefing, obstacle_map, config, reserve):
        super().__init__(briefing, obstacle_map, config, reserve)
        self.coverage = CoverageBaselinePolicy(briefing, obstacle_map, config, reserve)

    def act(self, observation):
        waypoint = self.coverage._pending()
        return Action(ActionType.RETURN) if waypoint is None else Action(ActionType.SEARCH_SECTOR, sector_id=waypoint.sector_id)

    def search_goal(self, sector_id):
        return self.coverage.search_goal(sector_id)

    def feedback(self, action, executed, reason=None, goal=None):
        self.coverage.feedback(action, executed, reason, goal)


def derive_tight_budget(context, banks, time_model, backend_config, nominal_budget_s=NOMINAL_BUDGET_S,
                        multiplier=TIGHT_BUDGET_MULTIPLIER):
    """Measure T_route_no_fix once and report B_tight with its validity checks. Decides nothing else.

    The record carries a digest of the executed route so `T_route_no_fix` is reconstructable: the ordered
    (action type, macro duration) sequence, hashed, plus its length. With the protocol, bank, time-model and
    backend-config hashes recorded beside it, an auditor can rerun the same route and compare.
    """
    import fast_mission
    import hashlib
    result = fast_mission.run_fast_episode(context, banks, time_model, 'p3_tight_budget_route',
                                           backend_config=backend_config,
                                           policy_factory=FixedRoutePolicy)
    outcome = result['outcome']
    trace = [(record.action.action_type.value, None if record.duration_s is None else round(record.duration_s, 9))
             for record in result['records']]
    route_trace = {'decisions': len(trace),
                   'digest': hashlib.sha256(repr(trace).encode()).hexdigest(),
                   'action_counts': {kind: sum(1 for a, _ in trace if a == kind) for kind in sorted({a for a, _ in trace})},
                   'first_five': trace[:5], 'last_five': trace[-5:]}
    route_s = outcome.elapsed_s
    tight = float(math.ceil(multiplier * route_s))
    shorter_than_nominal = tight < nominal_budget_s
    return {'design_version': P3_DESIGN_VERSION,
            'route_policy': FixedRoutePolicy.name,
            'route_trace': route_trace,
            'backend_config': {'reobservation_model': backend_config.reobservation_model,
                               'repeated_capture': backend_config.repeated_capture,
                               'evidence_serving': backend_config.evidence_serving,
                               'reobservation_seed': backend_config.reobservation_seed},
            'route_description': ('the protocol coverage plan flown to completion with no fix, no report and no '
                                  'inspection, plus the return and the landing'),
            't_route_no_fix_s': route_s, 'multiplier': multiplier, 'b_tight_s': tight,
            'route_decisions': outcome.decisions, 'route_terminated': outcome.termination_reason,
            'route_returned_and_landed': bool(outcome.returned and outcome.landed),
            'checks': {'shorter_than_nominal_2700s': shorter_than_nominal,
                       'route_completed_and_returned': outcome.termination_reason == 'mission_complete'
                       and bool(outcome.returned and outcome.landed)},
            'valid': shorter_than_nominal and outcome.termination_reason == 'mission_complete'
            and bool(outcome.returned and outcome.landed),
            'if_invalid': ('resolve the geometry/budget design and version it before the lock; do not adjust the '
                           'multiplier or sweep budgets until a method wins')}


def design(tight_budget_record, base_seed=20260921):
    """The full recorded P3 design. `tight_budget_record` comes from `derive_tight_budget`."""
    namespaces = seed_namespaces(base_seed)
    cells = core_cells(tight_budget_record['b_tight_s'])
    return {
        'design_version': P3_DESIGN_VERSION,
        'source': 'STATISTICAL_PROTOCOL_EN.md sections 3.2 and 5.3; review 4 P3 ruling',
        'status': 'design_only_nothing_executed',
        'core_cells': cells,
        'weights_note': ('0.8 over the four positive cells and 0.2 over the four no-human cells, uniform inside '
                         'each group; a benchmark convention, not a population estimate of rescue missions'),
        'excluded_from_core': {'d600_0m': 'final-only stress', 'd600_8m': 'final-only stress',
                               'one_anchor': 'final-only stress', 'intermittent_90_30': 'final-only stress',
                               'note': ('the P2 bundle ran at D600 = 8 m, which is a stress setting. Its near-tie '
                                        'between the fixing rules is a warning and must not move the criterion '
                                        'or serve as a P3 verdict.')},
        'tight_budget': tight_budget_record,
        'realizations': {'tuning_per_cell': TUNING_REALIZATIONS_PER_CELL,
                         'gate_per_cell': GATE_REALIZATIONS_PER_CELL,
                         'episodes_tuning': TUNING_REALIZATIONS_PER_CELL * len(cells),
                         'episodes_gate_per_strategy': GATE_REALIZATIONS_PER_CELL * len(cells)},
        'seed_namespaces': namespaces,
        'seed_rule': ('seed_for(namespaces, namespace, cell_index, realization) = start + 1000*cell_index + '
                      'realization; ranges are disjoint by construction and a seed outside its range raises'),
        'separation_rule': ('the reference rule is SELECTED on the tuning namespace and EVALUATED once on the '
                            'gate namespace. Reuse across arms inside a namespace is intentional (common random '
                            'numbers); reuse between selection and evaluation is prohibited.'),
        'criterion': ('statistical protocol section 5.3, unchanged and not restated here so it cannot drift: at '
                      'least 1.0 additional timely report-equivalent without increased mean FP_spatial, OR at '
                      'least 0.5 fewer FP_spatial without reduced timely report-equivalents, against the '
                      'strongest tuned eligible simple rule including end-batch-at-home, with raw TP_credit and '
                      'return behaviour not worse'),
        'not_decided_here': ['which reference rule wins the tuning selection', 'the gate verdict',
                             'anchor coordinates, dwell, measurement noise and availability, which must be '
                             'versioned before any tuning or gate seed is generated'],
    }


def check_disjoint(namespaces):
    """Every namespace range is disjoint from every other. Returns the offending pairs, empty when clean."""
    overlaps = []
    items = sorted(namespaces.items(), key=lambda kv: kv[1]['start'])
    for (name_a, a), (name_b, b) in zip(items, items[1:]):
        if a['end'] > b['start']:
            overlaps.append((name_a, name_b))
    return overlaps


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--bank', required=True, nargs='+', type=Path)
    parser.add_argument('--time-model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--base-seed', type=int, default=20260921)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this tool never overwrites a design directory')

    from aerial_qrl.mission.cached_view_backend import CachedViewConfig
    from aerial_qrl.mission.observation_bank import open_banks
    from aerial_qrl.mission.time_model import TimeModel
    import fast_mission

    banks = open_banks([str(p) for p in args.bank])
    context = fast_mission.load_context(args.protocol, banks)
    backend_config = CachedViewConfig(reobservation_model='visit_frame_permutation/3', repeated_capture='invalid',
                                      evidence_serving='all', reobservation_seed=0,
                                      provenance='engineering_default_unfrozen')
    record = derive_tight_budget(context, banks, TimeModel.from_file(args.time_model), backend_config)
    record['provenance'] = {
        'code_tree_sha256': code_tree_hash(),
        'source_sha256': {path: sha256_file(ROOT / path) for path in SOURCES},
        'protocol_path': context['protocol_path'], 'protocol_sha256': context['protocol_sha256'],
        'bank_manifest_sha256': [part.manifest_sha256 for part in getattr(banks, 'banks', [banks])],
        'bank_identity': context['bank_identity'],
        'time_model_sha256': sha256_file(args.time_model),
        'reconstruct': ('rerun FixedRoutePolicy with this protocol, bank, time model and backend config and '
                        'compare route_trace.digest; T_route_no_fix and therefore B_tight follow from it'),
    }
    payload = design(record, args.base_seed)
    payload['overlapping_namespaces'] = check_disjoint(payload['seed_namespaces'])
    assert not payload['overlapping_namespaces']
    args.out.mkdir(parents=True)
    (args.out / 'p3_design.json').write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 't_route_no_fix_s': record['t_route_no_fix_s'],
                      'b_tight_s': record['b_tight_s'], 'tight_budget_valid': record['valid'],
                      'core_cells': [c['name'] for c in payload['core_cells']],
                      'namespaces_disjoint': not payload['overlapping_namespaces']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
