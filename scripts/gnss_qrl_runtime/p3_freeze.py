"""Emit the immutable primary P3 configuration and its one manifest hash (review 6 versioning ruling).

What this freezes, and on whose authority
    review 6 ruled the primary benchmark values after the route audit passed. This tool writes them into
    one object, hashes it, and refuses to emit if any prerequisite artifact is missing or failing. Nothing here
    is the author's choice: the anchors, dwell, noise, availability and the eight cells are the reviewer's ruling,
    and the provenance fields come from artifacts that already exist.

    anchors        A_home (20,20,20), A_field0 (160,50,20), A_field1 (50,160,20), local ENU
    dwell          20 s - a SIMULATION BENCHMARK CONVENTION, not an operational or sensor measurement
    noise          isotropic sigma = 0.5 m - likewise a benchmark convention
    availability   at_anchor/1, continuously available at an eligible anchor
    cells          exactly the eight of p3_design v2

    Temporal intermittent 90/30, D600 0 and 8 m, and altered anchor or dwell cases are FINAL-ONLY sensitivity
    or stress conditions. They are listed here so they are visible, and marked as unable to influence tuning or
    the gate criterion.

One hash for everything
    `manifest_sha256` is the canonical hash of the whole frozen object: action order, observation encoder,
    reward and score versions, bank/protocol/time-model hashes, anchor set, dwell, noise, availability,
    budgets, cell weights, seed namespaces and the checkpoint identity inputs. Every comparator and every later
    learned arm must carry it, so two runs that disagree about any of those cannot be pooled by accident.

Reproduce
    .venv/bin/python scripts/gnss_qrl_runtime/p3_freeze.py \
        --design output/study/gnss_qrl_v1/frozen/p3_design_v2/p3_design.json \
        --route-audit output/study/gnss_qrl_v1/frozen/route_feasibility_v2/route_feasibility.json \
        --probe output/study/gnss_qrl_v1/frozen/bank_integrity_probe_v2/bank_integrity_probe.json \
        --protocol output/mission/village_r6_s0/protocol_all_victims.json \
        --bank <bank part dirs...> --time-model <time_model.json> \
        --out output/study/gnss_qrl_v1/frozen/p3_freeze_v1
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts/gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from aerial_qrl.mission.georeference_contracts import GEOREFERENCE_VERSION, GeoreferenceConfig   # noqa: E402
from aerial_qrl.mission.relocalization import FIX_TIMESTAMP_CONVENTION, AnchorSet                # noqa: E402
from aerial_qrl.training.checkpoint_identity import IDENTITY_VERSION, canonical_sha256           # noqa: E402
from aerial_qrl.training.encoding import ANCHOR_FEATURES, FIX_COUNT_CAP, GEOREFERENCE_FEATURES   # noqa: E402
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS, TOLERANCE_SCALE_M, _action_names        # noqa: E402
from aerial_qrl.training.reward_gnss import REWARD_VERSION, SCORE_VERSION                        # noqa: E402
import p3_design as p3                                                                           # noqa: E402

FREEZE_VERSION = 'gnss-p3-freeze/1'
#: review 6 versioning ruling. These are the reviewer's values, recorded here, not re-derived.
RULED_ANCHORS = (('A_home', (20.0, 20.0, 20.0)), ('A_field0', (160.0, 50.0, 20.0)),
                 ('A_field1', (50.0, 160.0, 20.0)))
RULED_DWELL_S = 20.0
RULED_MEASUREMENT_SIGMA_M = 0.5
RULED_AVAILABILITY = 'at_anchor/1'
FINAL_ONLY = {
    'temporal_intermittent_90_30': 'final-only sensitivity; cannot influence tuning or the gate criterion',
    'd600_0m': 'final-only stress; cannot influence tuning or the gate criterion',
    'd600_8m': 'final-only stress; cannot influence tuning or the gate criterion',
    'altered_anchor_or_dwell': 'final-only sensitivity; cannot influence tuning or the gate criterion',
}
CONVENTIONS = {
    'dwell_s': ('a simulation benchmark convention, not an operational or sensor measurement; no rescue '
                'service or hardware was consulted for it'),
    'measurement_sigma_m': ('a simulation benchmark convention; it is not a measured GNSS, VIO or survey '
                            'accuracy'),
    'timing_weight': ('max(0, 1 - t/B) is a research scoring convention and does not claim to be a survival '
                      'probability'),
    'false_report_weight': 'a research scoring convention; 0.5 and 2.0 are the declared descriptive sensitivities',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_hash():
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/code_version.py')], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or None


def ruled_anchor_set(dwell_s=RULED_DWELL_S, availability=RULED_AVAILABILITY):
    anchors = tuple({'anchor_id': anchor_id, 'position_enu': position, 'yaw_rad': 0., 'dwell_s': dwell_s}
                    for anchor_id, position in RULED_ANCHORS)
    return AnchorSet(schema_version=GEOREFERENCE_VERSION, anchors=anchors, availability_rule=availability,
                     selection_rule=('review 6 versioning ruling; the geometry originated from '
                                     'gnss_config.derive_anchors (pad centre plus world-bound extent fractions '
                                     '(0.80, 0.25) and (0.25, 0.80)), briefing geometry only, no victim, '
                                     'evaluator or detector input'))


def georeference_for(d600_rms_m):
    return GeoreferenceConfig(version=GEOREFERENCE_VERSION, d600_rms_m=float(d600_rms_m),
                              initial_bias_enu=(0., 0.), initial_correction_enu=(0., 0.),
                              initial_covariance_m2=((0., 0.), (0., 0.)),
                              measurement_sigma_m=RULED_MEASUREMENT_SIGMA_M, path_horizon_s=4096.,
                              path_resolution_s=0.05, provenance='frozen')


#: The route audit must answer exactly these questions. An empty or partial set is a refusal: a loop over
#: whatever keys happen to be present would accept an artifact that checked nothing (review 7).
REQUIRED_ROUTE_HARD_CHECKS = ('every_anchor_return_path_free', 'every_anchor_reachable_from_some_origin',
                              'every_anchor_to_anchor_leg_free')


def prerequisites(design_path, route_path, probe_path):
    """Refuse to freeze on a missing or failing prerequisite, rather than freezing over it.

    The probe is re-verified with the HARDENED `recompute_from_payload` rather than by reading its stored
    `self_audit`: an artifact produced before that checker was hardened carries a flag whose meaning has since
    changed, and trusting it would reintroduce exactly the weakness the review found.
    """
    import bank_integrity_probe as probe_module
    problems = []
    design = json.loads(Path(design_path).read_text())
    route = json.loads(Path(route_path).read_text())
    probe = json.loads(Path(probe_path).read_text())
    if not design['tight_budget']['valid']:
        problems.append('the P3 design tight budget is not valid')
    if design['overlapping_namespaces']:
        problems.append('the P3 seed namespaces overlap')
    if not route.get('passed'):
        problems.append('the route feasibility audit did not pass')
    hard_checks = route.get('hard_checks') or {}
    missing_checks = sorted(set(REQUIRED_ROUTE_HARD_CHECKS) - set(hard_checks))
    if missing_checks:
        problems.append(f'route audit is missing hard checks: {missing_checks}')
    for check in REQUIRED_ROUTE_HARD_CHECKS:
        if check in hard_checks and not hard_checks[check]:
            problems.append(f'route audit hard check failed: {check}')
    if sorted(route.get('budgets_audited_s', {})) != ['nominal_2700s', 'tight_derived']:
        problems.append('the route audit does not cover both core budgets')
    if route.get('origin_kinds', {}).get('anchor', 0) != len(RULED_ANCHORS):
        problems.append('the route audit does not include the anchors as origins')
    recomputed = probe_module.recompute_from_payload(probe)
    if probe.get('verdict') != 'PASS':
        problems.append(f'the bank integrity probe verdict is {probe.get("verdict")!r}')
    if recomputed['recomputed_verdict'] != 'PASS' or not recomputed['agrees_with_recorded']:
        problems.append(f'the bank integrity probe fails independent recomputation: {recomputed["disagreements"][:3]}')
    return design, route, probe, problems


def build(design, route, probe, protocol_path, protocol_sha256, bank_sha256, bank_identity, time_model_sha256):
    anchors = ruled_anchor_set()
    cells = design['core_cells']
    namespaces = design['seed_namespaces']
    frozen = {
        'freeze_version': FREEZE_VERSION,
        'task_id': 'flood-sar-village-gnss/1',
        'authority': 'review 6 configuration versioning ruling',
        'encoder': {'obs_encoding': LAYOUT_V4_GNSS.obs_version, 'dim': LAYOUT_V4_GNSS.dim,
                    'georeference_features': GEOREFERENCE_FEATURES, 'anchor_features': ANCHOR_FEATURES,
                    'anchor_slots': LAYOUT_V4_GNSS.anchor_slots,
                    'tolerance_scale_m': TOLERANCE_SCALE_M, 'fix_count_cap': FIX_COUNT_CAP},
        'action_space': {'version': LAYOUT_V4_GNSS.action_version, 'n_actions': LAYOUT_V4_GNSS.n_actions,
                         'return_mask_rule': 'gnss_exhausted/1',
                         'action_order': list(_action_names(LAYOUT_V4_GNSS)),
                         'relocalize_actions': list(LAYOUT_V4_GNSS.relocalize_actions),
                         'return_action': LAYOUT_V4_GNSS.return_action,
                         'anchor_order': [anchor_id for anchor_id, _ in RULED_ANCHORS]},
        'reward_version': REWARD_VERSION, 'score_version': SCORE_VERSION,
        'score': {'tolerance_m': 3.0, 'metric': 'xyz', 'false_report_weight': 1.0,
                  'timing_weight': 'max(0, 1 - submit_time/mission_budget)'},
        'anchors': anchors.to_dict(),
        'dwell_s': RULED_DWELL_S,
        'measurement_sigma_m': RULED_MEASUREMENT_SIGMA_M,
        'availability_rule': RULED_AVAILABILITY,
        'fix_timestamp_convention': FIX_TIMESTAMP_CONVENTION,
        'georeference_by_drift': {f'd{d:g}': georeference_for(d).to_dict() for d in p3.CORE_D600_M},
        'budgets_s': {'nominal': p3.NOMINAL_BUDGET_S, 'tight': design['tight_budget']['b_tight_s']},
        'core_cells': cells,
        'cell_weights': {c['name']: c['weight'] for c in cells},
        'realizations': design['realizations'],
        'seed_namespaces': namespaces,
        'seed_rule': design['seed_rule'], 'separation_rule': design['separation_rule'],
        'criterion': design['criterion'],
        'data': {'protocol_path': protocol_path, 'protocol_sha256': protocol_sha256,
                 'bank_manifest_sha256': bank_sha256, 'bank_identity': bank_identity,
                 'time_model_sha256': time_model_sha256},
        'checkpoint_identity_inputs': {
            'identity_version': IDENTITY_VERSION, 'task_id': 'flood-sar-village-gnss/1',
            'obs_encoding_version': LAYOUT_V4_GNSS.obs_version,
            'action_space_version': f'{LAYOUT_V4_GNSS.action_version}+return/gnss_exhausted/1',
            'action_order': list(_action_names(LAYOUT_V4_GNSS)),
            'reward_version': REWARD_VERSION, 'score_version': SCORE_VERSION,
            'disturbance_model': 'brownian_translation_2d/1',
            'disturbance_config_sha256_by_drift': {f'd{d:g}': canonical_sha256(georeference_for(d).to_dict())
                                                   for d in p3.CORE_D600_M},
            'anchor_set_sha256': canonical_sha256(anchors.to_dict()),
            'manifest_sha256_note': ('the per-run manifest hash is this frozen object\'s manifest_sha256; the '
                                     'critic and actor_head are filled per arm at training time'),
        },
        'final_only_conditions': FINAL_ONLY,
        'conventions_not_measurements': CONVENTIONS,
        'prerequisite_artifacts': {
            'p3_design': {'version': design['design_version'],
                          't_route_no_fix_s': design['tight_budget']['t_route_no_fix_s'],
                          'route_trace_digest': design['tight_budget']['route_trace']['digest']},
            'route_feasibility': {'version': route['audit'], 'passed': route['passed'],
                                  'origins_audited': route['origins_audited'],
                                  'hard_checks': route['hard_checks']},
            'bank_integrity_probe': {'version': probe['probe'], 'verdict': probe['verdict'],
                                     'anchors_checked': probe['anchors_checked']},
        },
    }
    frozen['manifest_sha256'] = canonical_sha256(frozen)
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--design', required=True, type=Path)
    parser.add_argument('--route-audit', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--bank', required=True, nargs='+', type=Path)
    parser.add_argument('--time-model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this tool never overwrites a freeze directory')

    design, route, probe, problems = prerequisites(args.design, args.route_audit, args.probe)
    if problems:
        raise SystemExit('refusing to freeze:\n  ' + '\n  '.join(problems))

    from aerial_qrl.mission.observation_bank import open_banks
    import fast_mission
    banks = open_banks([str(p) for p in args.bank])
    context = fast_mission.load_context(args.protocol, banks)
    frozen = build(design, route, probe, context['protocol_path'], context['protocol_sha256'],
                   [part.manifest_sha256 for part in getattr(banks, 'banks', [banks])],
                   context['bank_identity'], sha256_file(args.time_model))

    payload = {'freeze': FREEZE_VERSION,
               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               'code_tree_sha256': code_tree_hash(),
               'status': ('frozen configuration only; NO tuning or gate outcome has been generated and P3 '
                          'execution is not open'),
               'configuration': frozen}
    args.out.mkdir(parents=True)
    (args.out / 'p3_frozen_configuration.json').write_text(json.dumps(payload, indent=1, sort_keys=False) + '\n')
    print(json.dumps({'out': str(args.out), 'manifest_sha256': frozen['manifest_sha256'],
                      'anchors': [a['anchor_id'] for a in frozen['anchors']['anchors']],
                      'dwell_s': frozen['dwell_s'], 'measurement_sigma_m': frozen['measurement_sigma_m'],
                      'availability_rule': frozen['availability_rule'],
                      'budgets_s': frozen['budgets_s'], 'cells': len(frozen['core_cells']),
                      'action_space': frozen['action_space']['version'],
                      'n_actions': frozen['action_space']['n_actions']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
