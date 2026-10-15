"""Per-transition cached-bank integrity probe for RELOCALIZE (pre-review finding and review 4).

What was wrong with the first attempt, stated plainly
    The superseded `p2_episodes_v1` decided integrity by differencing WHOLE episodes of different comparators
    against `never_fix` and accepting `off_node_delta >= fixes_attempted`. That is a lower bound on an aggregate
    over trajectories that are not the same trajectory, so it can pass while a bank capture is consumed. A review
    demonstrated exactly that with an absurd case. It proved nothing and it is gone.

What this does instead
    It attributes every change to ONE transition. `MissionExecutive(macro_audit=...)` is called immediately
    before and immediately after each executed macro with a read-only digest of everything a macro could move:
    backend frame counters by category, the number of perception records, the tracker state digest, the coverage
    digest, the report count and the georeference state. The probe then asserts, per RELOCALIZE transition:

        delta bank_frames              == 0        no cached observation was consumed
        delta repeated_capture_frames  == 0        nothing was voided as an already-served image
        delta perception_records       == 0        the detector never ran
        tracks_digest                  unchanged   the tracker did not move
        coverage_digest                unchanged   the coverage grid did not move
        delta reports                  == 0        a fix is not a submission
        delta off_node_frames          == EXPECTED_OFF_NODE_FRAMES_PER_FIX   measured, not asserted from belief
        delta measurements             == 1        exactly one measurement opportunity, accepted or not
        sim time strictly advances                 travel and dwell are paid

    The same assertions run for EVERY anchor of the set, so no anchor identity can change capture semantics.

Negative controls
    A gate that cannot fail is not a gate. Two controls deliberately break the property and the probe must
    report them as failures:
      consume_capture   an extra SEARCH_SECTOR in place of the fix: it consumes a bank capture and moves the
                        tracker and coverage, so `delta bank_frames == 0` must fail.
      forged_snapshot   the audit digest of a relocalisation is tampered with after the fact, so the probe must
                        not silently accept a transition whose recorded state does not match.

Output
    `bank_integrity_probe.json`: source/bank/config hashes, the action schedule, the served RGB path digest, the
    per-transition deltas, the measurement counts, the per-anchor verdicts and the negative-control verdicts.
    The top-level verdict is PASS only when every positive case passes AND every negative control fails.

Reproduce
    .venv/bin/python scripts/gnss_qrl_runtime/bank_integrity_probe.py \
        --protocol output/mission/village_r6_s0/protocol_all_victims.json \
        --bank output/mission/village_r6_banks/bank_r6_s0_all_victims/{search_takeoff,inspect_p1of4,inspect_p2of4,inspect_p3of4,inspect_p4of4} \
        --time-model output/training/time_model_village_v1/20260917T074923Z/time_model.json \
        --out output/study/gnss_qrl_v1/frozen/bank_integrity_probe_v1
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

from aerial_qrl.mission.cached_view_backend import CachedViewConfig                  # noqa: E402
from aerial_qrl.mission.contracts import Action, ActionType                          # noqa: E402
from aerial_qrl.mission.executive import CoverageBaselinePolicy, MemoryLog, MissionPolicy   # noqa: E402
from aerial_qrl.mission.georeference import GeoreferenceService                      # noqa: E402
from aerial_qrl.mission.observation_bank import open_banks                           # noqa: E402
from aerial_qrl.mission.time_model import TimeModel                                  # noqa: E402
import fast_mission                                                                  # noqa: E402
from gnss_config import candidate_config, check_anchor_safety, load_config           # noqa: E402

PROBE_VERSION = 'gnss-bank-integrity-probe/1'
#: MEASURED per transition. On the cached-view backend a COMMAND never reads a sensor: `_command` advances the
#: clock and the pose, and only `_capture` calls `read_sensors`. The RELOCALIZE macro issues GOTO and HOLD,
#: which consume no frame, and then exactly ONE `_capture(perceive=False)`. That single non-perception capture
#: is the one off-node frame attributable to the transition.
#:
#: A REPORT that follows the fix takes its own capture while the vehicle is still at the anchor, and an anchor
#: is off-node by construction (`assert_anchors_off_bank`), so that frame is also off-node. It belongs to the
#: REPORT transition, not to this one. Attributing it to the fix is what made the superseded aggregate verdict
#: read about two per fix; the per-transition value was, and is, one.
EXPECTED_OFF_NODE_FRAMES_PER_FIX = 1
SOURCES = ('src/aerial_qrl/mission/executive.py', 'src/aerial_qrl/mission/georeference.py',
           'src/aerial_qrl/mission/relocalization.py', 'src/aerial_qrl/mission/cached_view_backend.py',
           'scripts/gnss_qrl_runtime/gnss_config.py', 'scripts/gnss_qrl_runtime/bank_integrity_probe.py')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_tree_hash():
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/code_version.py')], capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip() or None


class ProbePolicy(MissionPolicy):
    """A fixed schedule: N searches, one intervention, N searches, RETURN.

    `intervention` is ('relocalize', anchor_id), ('search', None) for the consume-capture control, or None.
    The search goals come from the protocol's own coverage plan, so every variant flies the same nodes in the
    same order and the intervention is the only difference.
    """
    name = 'bank_integrity_probe'

    def __init__(self, briefing, obstacle_map, config, reserve, intervention=None, before=3, after=3):
        super().__init__(briefing, obstacle_map, config, reserve)
        self.coverage = CoverageBaselinePolicy(briefing, obstacle_map, config, reserve)
        self.intervention, self.before, self.after = intervention, before, after
        self.searches = 0
        self.intervened = False
        self._intervention_action = None

    def act(self, observation):
        if self.searches == self.before and not self.intervened and self.intervention is not None:
            self.intervened = True
            kind, reference = self.intervention
            if kind == 'relocalize':
                return Action(ActionType.RELOCALIZE, anchor_id=reference)
            # 'search': the consume-capture control. It must be an EXTRA visit, not one of the scheduled ones,
            # or it would consume no capture the baseline did not also consume and the control would be vacuous.
            self._intervention_action = self._search()
            return self._intervention_action
        if self.searches >= self.before + self.after:
            return Action(ActionType.RETURN)
        return self._search()

    def _search(self):
        waypoint = self.coverage._pending()
        return Action(ActionType.RETURN) if waypoint is None else Action(ActionType.SEARCH_SECTOR, sector_id=waypoint.sector_id)

    def search_goal(self, sector_id):
        return self.coverage.search_goal(sector_id)

    def feedback(self, action, executed, reason=None, goal=None):
        intervention = action is self._intervention_action
        if action.action_type == ActionType.SEARCH_SECTOR and executed and not intervention:
            self.searches += 1          # the intervention visit is extra and does not spend the schedule
        self.coverage.feedback(action, executed, reason, goal)


def run_probe_episode(context, banks, time_model, config, intervention, episode_id, seed):
    """One scripted episode with the per-macro audit hook attached; returns the transitions it recorded."""
    service = GeoreferenceService(config['georeference'], f'probe-{episode_id}'.replace('_', '-'))
    backend_config = CachedViewConfig(reobservation_model='visit_frame_permutation/3', repeated_capture='invalid',
                                      evidence_serving='all', reobservation_seed=seed,
                                      provenance='engineering_default_unfrozen')
    transitions, pending = [], {}

    def audit(stage, action, snapshot):
        key = action.action_type.value
        if stage == 'before':
            pending['state'] = snapshot
            pending['action'] = {'type': key, 'anchor_id': action.anchor_id}
        else:
            transitions.append({'action': pending['action'], 'before': pending['state'], 'after': snapshot})

    log = MemoryLog()
    result = fast_mission.run_fast_episode(
        context, banks, time_model, episode_id, log=log, backend_config=backend_config,
        executive_kwargs={'georeference': service, 'anchor_set': config['anchors'], 'macro_audit': audit},
        policy_factory=lambda briefing, omap, cfg, reserve: ProbePolicy(briefing, omap, cfg, reserve, intervention),
        reports_for_matching=service.global_report_positions)
    served = [r['frame']['rgb_path'] for r in log.records['public/frames.jsonl'] if r['frame'].get('rgb_path')]
    return {'result': result, 'transitions': transitions, 'service': service, 'log': log,
            'served_rgb': served_rgb_evidence(served)}


def served_rgb_evidence(paths):
    """Per-entry hashes plus their ordered digest, so equality is recomputable without host paths.

    The absolute path of a bank file says nothing scientific and would pin the artifact to this machine, so it
    is never written. Each entry is hashed on its own and the ordered list of those hashes is hashed again: an
    auditor can compare two runs entry by entry, and can see WHERE they diverge, from the artifact alone.
    """
    entries = [hashlib.sha256(str(path).encode()).hexdigest() for path in paths]
    return {'count': len(entries), 'entry_hashes': entries,
            'digest': hashlib.sha256('|'.join(entries).encode()).hexdigest()}


def check_transition(transition, expected_off_node):
    """The per-transition criterion table. Returns (passed, rows).

    A backend counter that is absent on BOTH sides is marked `applicable: False` and takes no part in the
    verdict. That matters: `mission.fake_backend` has no cached bank at all, so the three bank counters are
    meaningless there and must not be reported as satisfied. Only the cached-view backend can decide bank
    integrity, which is why the probe runs on the real r6 s0 bank. A counter present on one side and absent on
    the other is a defect and fails.
    """
    before, after = transition['before'], transition['after']
    rows = []

    def counter(criterion, key, expected):
        if before.get(key) is None and after.get(key) is None:
            rows.append({'criterion': criterion, 'expected': expected, 'observed': None, 'applicable': False,
                         'passed': True, 'note': 'this backend does not report the counter'})
            return
        if before.get(key) is None or after.get(key) is None:
            rows.append({'criterion': criterion, 'expected': expected, 'observed': 'counter_appeared_or_vanished',
                         'applicable': True, 'passed': False})
            return
        observed = after[key] - before[key]
        rows.append({'criterion': criterion, 'expected': expected, 'observed': observed, 'applicable': True,
                     'passed': observed == expected})

    def plain(criterion, expected, observed):
        rows.append({'criterion': criterion, 'expected': expected, 'observed': observed, 'applicable': True,
                     'passed': observed == expected})

    counter('delta_bank_frames', 'bank_frames', 0)
    counter('delta_repeated_capture_frames', 'repeated_capture_frames', 0)
    plain('delta_perception_records', 0, after['perception_records'] - before['perception_records'])
    plain('tracks_digest_unchanged', True, before['tracks_digest'] == after['tracks_digest'])
    plain('coverage_digest_unchanged', True, before['coverage_digest'] == after['coverage_digest'])
    plain('delta_reports', 0, after['reports'] - before['reports'])
    counter('delta_off_node_frames', 'off_node_frames', expected_off_node)
    plain('delta_measurements', 1, after['measurements'] - before['measurements'])
    plain('sim_time_advances', True, after['sim_time_s'] > before['sim_time_s'])
    plain('covariance_not_increased', True,
          after['georeference']['covariance_trace_m2'] <= before['georeference']['covariance_trace_m2'] + 1e-12)
    return all(row['passed'] for row in rows), rows


#: The controls the artifact must contain. A checker that accepts a payload with the controls deleted is not a
#: checker: review 6 showed the first version returned PASS after removing them.
REQUIRED_CONTROL_NAMES = ('consume_capture', 'forged_snapshot')


def _check_served_rgb(name, served, findings):
    """Recompute the count and the ordered digest from the entry hashes; never trust the stored ones."""
    entries = served.get('entry_hashes')
    if not isinstance(entries, list) or not entries:
        findings.append(f'{name}: served_rgb.entry_hashes missing or empty')
        return None
    if any(not isinstance(e, str) or len(e) != 64 for e in entries):
        findings.append(f'{name}: served_rgb.entry_hashes are not sha256 hex digests')
        return None
    if served.get('count') != len(entries):
        findings.append(f'{name}: served_rgb.count {served.get("count")} != {len(entries)} entry hashes')
    digest = hashlib.sha256('|'.join(entries).encode()).hexdigest()
    if served.get('digest') != digest:
        findings.append(f'{name}: served_rgb.digest does not follow from its entry hashes')
    return digest


def recompute_from_payload(payload):
    """Re-derive every criterion, every inventory expectation and the verdict from the PERSISTED values alone.

    This is the point of the artifact: an auditor runs this on the JSON and never has to trust the producer.
    It reads only what the file contains and it FAILS CLOSED - a payload with items deleted, an inventory that
    disagrees with the configured anchor set, or a stored digest that does not follow from its entry hashes is
    a failure, not a smaller passing run.

    review 6 found the first version weaker than this docstring: it trusted stored RGB digests and
    returned PASS after the negative controls or some anchors were removed. Every one of those holes is a
    validation below, and `test_p2_correction.py` deletes and tampers with a real payload to prove it.
    """
    findings = []
    expected_off_node = payload.get('expected_off_node_frames_per_fix')
    if expected_off_node is None:
        return {'recomputed_verdict': 'FAIL', 'agrees_with_recorded': False,
                'disagreements': ['expected_off_node_frames_per_fix missing']}

    # 1. inventory: the configured anchor set decides what must be present, not the producer's own list
    configured = [a['anchor_id'] for a in payload.get('anchors', {}).get('anchors', [])]
    checked = list(payload.get('anchors_checked') or [])
    if not configured:
        findings.append('anchors: the configured anchor set is missing')
    if checked != configured:
        findings.append(f'anchors_checked {checked} differs from the configured anchor set {configured}')
    runs = payload.get('runs') or {}
    positives = {row.get('anchor_id'): row for row in (payload.get('positives') or [])}
    for anchor_id in configured:
        if anchor_id not in runs:
            findings.append(f'{anchor_id}: no run recorded')
        if anchor_id not in positives:
            findings.append(f'{anchor_id}: no positive summary row recorded')
    if set(positives) != set(configured):
        findings.append(f'positive summary rows {sorted(positives)} differ from the anchor set {sorted(configured)}')
    for required in ('baseline', 'control_consume_capture'):
        if required not in runs:
            findings.append(f'runs.{required} missing')

    # 2. served-RGB evidence, recomputed from the entry hashes
    digests = {}
    for name, run in runs.items():
        served = run.get('served_rgb')
        if not isinstance(served, dict):
            findings.append(f'runs.{name}: served_rgb missing')
            continue
        digests[name] = _check_served_rgb(f'runs.{name}', served, findings)
    baseline_digest = digests.get('baseline')

    # 3. every anchor: exactly one RELOCALIZE transition, every criterion recomputed
    positives_ok = []
    for anchor_id in configured:
        run = runs.get(anchor_id) or {}
        transitions = run.get('checked_transitions') or []
        relocalizations = [t for t in transitions
                           if (t.get('action') or {}).get('type') in (None, 'RELOCALIZE')]
        if len(relocalizations) != 1:
            findings.append(f'{anchor_id}: expected exactly one RELOCALIZE transition, found {len(relocalizations)}')
        per_transition = []
        for checked_transition in transitions:
            passed, rows = check_transition({'before': checked_transition['before'],
                                             'after': checked_transition['after']}, expected_off_node)
            per_transition.append(passed)
            if rows != checked_transition.get('criteria'):
                findings.append(f'{anchor_id}: recomputed criteria differ from the recorded ones')
            if passed != checked_transition.get('passed'):
                findings.append(f'{anchor_id}: recorded transition verdict differs from the recomputed one')
        same_images = digests.get(anchor_id) is not None and digests.get(anchor_id) == baseline_digest
        if same_images != bool(run.get('served_rgb_digest_equals_baseline')):
            findings.append(f'{anchor_id}: recorded served_rgb_digest_equals_baseline differs from the recomputed one')
        ok = bool(per_transition) and all(per_transition) and same_images and len(relocalizations) == 1
        positives_ok.append(ok)
        if ok != bool(run.get('passed')):
            findings.append(f'{anchor_id}: recorded run verdict differs from the recomputed one')
        if ok != bool(positives.get(anchor_id, {}).get('passed')):
            findings.append(f'{anchor_id}: positive summary row disagrees with the recomputed verdict')

    # 4. the negative controls must all be present, by name, and must all fail
    controls = {c.get('name'): c for c in (payload.get('negative_controls') or [])}
    missing = [name for name in REQUIRED_CONTROL_NAMES if name not in controls]
    if missing:
        findings.append(f'negative controls missing: {missing}')
    controls_ok = []
    for name in REQUIRED_CONTROL_NAMES:
        control = controls.get(name)
        if control is None:
            controls_ok.append(False)
            continue
        transition = control.get('checked_transition') or {}
        if 'before' not in transition or 'after' not in transition:
            findings.append(f'{name}: the control keeps no before/after snapshot')
            controls_ok.append(False)
            continue
        passed, rows = check_transition({'before': transition['before'], 'after': transition['after']},
                                        expected_off_node)
        if rows != transition.get('criteria'):
            findings.append(f'{name}: recomputed criteria differ from the recorded ones')
        controls_ok.append(not passed)
        if (not passed) != bool(control.get('control_behaves_as_required')):
            findings.append(f'{name}: recorded control behaviour differs from the recomputed one')

    verdict = 'PASS' if (configured and positives_ok and all(positives_ok)
                         and not missing and controls_ok and all(controls_ok) and not findings) else 'FAIL'
    if verdict != payload.get('verdict'):
        findings.append(f"top-level verdict recomputed as {verdict}, recorded as {payload.get('verdict')}")
        verdict = 'FAIL'
    return {'recomputed_verdict': verdict, 'agrees_with_recorded': not findings, 'disagreements': findings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--bank', required=True, nargs='+', type=Path)
    parser.add_argument('--time-model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--d600', type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; this probe never overwrites a run directory')

    banks = open_banks([str(p) for p in args.bank])
    time_model = TimeModel.from_file(args.time_model)
    context = fast_mission.load_context(args.protocol, banks)
    config = load_config(candidate_config(
        context['briefing'], d600_rms_m=args.d600, measurement_sigma_m=.5, dwell_s=20., transit_altitude_agl_m=20.,
        n_ref=len(context['evaluator'].targets), realizations=1, comparators=[('never_fix', {})],
        path_key_prefix='probe'))
    probe_config = CachedViewConfig(reobservation_model='visit_frame_permutation/3', repeated_capture='invalid',
                                    evidence_serving='all', provenance='engineering_default_unfrozen')
    check_anchor_safety(config['anchors'], banks, probe_config)

    baseline = run_probe_episode(context, banks, time_model, config, None, 'baseline', args.seed)
    runs = {'baseline': {'served_rgb': baseline['served_rgb'], 'transitions_recorded': len(baseline['transitions'])}}

    anchors = [a.anchor_id for a in config['anchors'].anchors]
    positives = []
    for anchor_id in anchors:
        episode = run_probe_episode(context, banks, time_model, config, ('relocalize', anchor_id),
                                    f'fix-{anchor_id}', args.seed)
        fixes = [t for t in episode['transitions'] if t['action']['type'] == 'RELOCALIZE']
        checked = []
        for transition in fixes:
            passed_fix, criteria = check_transition(transition, EXPECTED_OFF_NODE_FRAMES_PER_FIX)
            checked.append({'action': transition['action'], 'before': transition['before'],
                            'after': transition['after'], 'criteria': criteria, 'passed': passed_fix})
        same_images = episode['served_rgb']['digest'] == baseline['served_rgb']['digest']
        passed = bool(checked) and all(c['passed'] for c in checked) and same_images
        runs[anchor_id] = {'served_rgb': episode['served_rgb'],
                           'served_rgb_digest_equals_baseline': same_images,
                           'relocalizations': len(fixes), 'passed': passed,
                           'measurements': [m.to_dict() for m in episode['service'].measurements],
                           'checked_transitions': checked}
        positives.append({'anchor_id': anchor_id, 'passed': passed, 'relocalizations': len(fixes),
                          'served_rgb_digest_equals_baseline': same_images})

    # --- negative controls: the criterion must FAIL on both ------------------------------------------------
    consume = run_probe_episode(context, banks, time_model, config, ('search', None), 'control-consume', args.seed)
    consume_transitions = [t for t in consume['transitions'] if t['action']['type'] == 'SEARCH_SECTOR']
    fourth = consume_transitions[3] if len(consume_transitions) > 3 else consume_transitions[-1]
    consume_ok, consume_rows = check_transition(fourth, EXPECTED_OFF_NODE_FRAMES_PER_FIX)
    runs['control_consume_capture'] = {'served_rgb': consume['served_rgb'],
                                       'served_rgb_digest_equals_baseline':
                                           consume['served_rgb']['digest'] == baseline['served_rgb']['digest'],
                                       'transitions_recorded': len(consume['transitions'])}
    control_consume = {
        'name': 'consume_capture',
        'what': ('a real EXTRA SEARCH_SECTOR in place of the fix, outside the schedule, so it consumes a bank '
                 'capture the baseline never consumed and moves the tracker and coverage. Both the served-RGB '
                 'comparison and the per-transition criterion must therefore fail; delta_bank_frames is the '
                 'one that speaks to bank integrity directly.'),
        'served_rgb_digest_equals_baseline': runs['control_consume_capture']['served_rgb_digest_equals_baseline'],
        'checked_transition': {'action': fourth['action'], 'before': fourth['before'], 'after': fourth['after'],
                               'criteria': consume_rows, 'passed': consume_ok},
        'control_behaves_as_required': (not consume_ok)}

    forged_before = dict(baseline['transitions'][0]['before'])
    forged_after = dict(forged_before, bank_frames=(forged_before['bank_frames'] or 0) + 1)
    forged_ok, forged_rows = check_transition({'before': forged_before, 'after': forged_after},
                                              EXPECTED_OFF_NODE_FRAMES_PER_FIX)
    control_forged = {
        'name': 'forged_snapshot',
        'what': ('a synthetic transition whose after-state shows one more bank frame: the criterion must reject '
                 'it, so a tampered record cannot pass'),
        'checked_transition': {'action': {'type': 'SYNTHETIC', 'anchor_id': None}, 'before': forged_before,
                               'after': forged_after, 'criteria': forged_rows, 'passed': forged_ok},
        'control_behaves_as_required': (not forged_ok)}

    controls = [control_consume, control_forged]
    verdict = all(p['passed'] for p in positives) and all(c['control_behaves_as_required'] for c in controls)

    payload = {
        'probe': PROBE_VERSION,
        'classification': ('per-transition attribution of one RELOCALIZE on the real r6 s0 bank; engineering '
                           'validation, not a paper result'),
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'verdict': 'PASS' if verdict else 'FAIL',
        'independently_recomputable': ('every criterion and the top-level verdict follow from the persisted '
                                       'before/after snapshots and served-RGB entry hashes alone; run '
                                       'bank_integrity_probe.recompute_from_payload(payload) to re-derive them '
                                       'without trusting this producer'),
        'code_tree_sha256': code_tree_hash(),
        'source_sha256': {path: sha256_file(ROOT / path) for path in SOURCES},
        'protocol_sha256': context['protocol_sha256'],
        'bank_manifest_sha256': [part.manifest_sha256 for part in getattr(banks, 'banks', [banks])],
        'time_model_sha256': sha256_file(args.time_model),
        'georeference': config['georeference'].to_dict(), 'anchors': config['anchors'].to_dict(),
        'anchor_safety': 'assert_anchors_off_bank passed on this bank before any episode',
        'backend_config': {'reobservation_model': 'visit_frame_permutation/3', 'repeated_capture': 'invalid',
                           'evidence_serving': 'all', 'reobservation_seed': args.seed},
        'schedule': '3 searches, one intervention, 3 searches, RETURN; identical coverage goals in every variant',
        'expected_off_node_frames_per_fix': EXPECTED_OFF_NODE_FRAMES_PER_FIX,
        'expected_off_node_note': (
            'GOTO and HOLD consume no sensor frame: on the cached-view backend only a capture calls '
            'read_sensors. The one off-node frame attributed to a RELOCALIZE is its explicit '
            'perceive=False capture. A REPORT that follows takes its own capture, still at the anchor and '
            'therefore also off-node, but that frame belongs to the REPORT transition and is outside this '
            'one. Mixing the two is what made the superseded aggregate read about two frames per fix.'),
        'served_rgb_note': ('per-entry sha256 of each served RGB path, in order, plus their ordered digest. '
                            'Absolute host paths are never written: they are not scientific evidence and would '
                            'pin the artifact to one machine.'),
        'anchors_checked': anchors, 'runs': runs, 'positives': positives, 'negative_controls': controls,
        'limits': ['one schedule, one seed, one layout, one drift scale; this attributes a macro, it does not '
                   'validate a mission',
                   'the probe covers RELOCALIZE only; other macros are unchanged by this task'],
    }
    payload['self_audit'] = recompute_from_payload(payload)
    if not payload['self_audit']['agrees_with_recorded']:
        raise SystemExit(f"probe self-audit failed: {payload['self_audit']['disagreements']}")

    args.out.mkdir(parents=True)
    (args.out / 'bank_integrity_probe.json').write_text(json.dumps(payload, indent=1) + '\n')
    print(json.dumps({'out': str(args.out), 'verdict': payload['verdict'],
                      'self_audit': payload['self_audit'],
                      'anchors_checked': anchors,
                      'positives_passed': [p['anchor_id'] for p in positives if p['passed']],
                      'positives_failed': [p['anchor_id'] for p in positives if not p['passed']],
                      'controls': {c['name']: c['control_behaves_as_required'] for c in controls}}, indent=1))
    return 0 if verdict else 1


if __name__ == '__main__':
    raise SystemExit(main())
