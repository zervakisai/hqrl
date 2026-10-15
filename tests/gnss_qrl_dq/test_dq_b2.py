"""Freeze B2: studies, gates, config validation, launch binding, the twin at deployment, precision, and the integrity-
proving analysis with mutation tests (review 40, B40-1..B40-5)."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_b2 as b2                                                                                # noqa: E402
import dq_b2_analysis as an                                                                       # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import p4_train as p4                                                                             # noqa: E402

E1_CONFIG = (ROOT.parent / 'idea_e1' / 'output/study/gnss_qrl_v1/e1/e1_campaign_v2/evaluation/'
                 'quantum__lr3e-3_ent0__b10/config.json')
FS = 'f' * 64                      # a stand-in freeze hash for configuration tests


# -- studies, gates, configuration -----------------------------------------------------------------------------------

def test_studies_are_registered_with_their_own_salts_namespaces_and_candidates():
    assert p4.STUDIES['dq_b2'] is b2.STUDY_B2 and p4.STUDIES['dq_b2_smoke'] is b2.STUDY_SMOKE
    assert b2.STUDY_B2.salt == (20260923, 2) and b2.STUDY_B2.training_namespace == 'dq_e2_training'
    assert b2.STUDY_B2.suites['dq_b2_evaluation'] == ('dq_e2_evaluation', tuple(range(10)))
    assert b2.STUDY_B2.candidates == b2.CANDIDATES
    rt = b2.runtime(b2.STUDY_B2)
    for b in (0, 35):
        assert rt.schedule(b, 0)[1] == dq.DQ_NAMESPACES['dq_e2_training']['start'] + b * p4.BLOCK_STRIDE


def test_b2_namespaces_are_closed_and_launch_refuses(tmp_path):
    for ns in ('dq_e2_training', 'dq_e2_evaluation'):
        with pytest.raises(dq.DQError, match='closed'):
            dq.require_open(ns)
    with pytest.raises((p4.FreezeBindingError, dq.DQError)):
        b2.launch(b2.CAMPAIGN, 1, 'a' * 64)
    record = b2.run_one(('dq_b2', 'quantum', 0, str(tmp_path / 'runs'), 'dq_b2', 'a' * 64, tmp_path))
    assert record['status'] == 'refused' and not (tmp_path / 'runs').exists()


def test_training_configuration_equals_e1s_except_study_labels_and_validates():
    e1 = json.loads(E1_CONFIG.read_text())
    ours = b2.config_for('quantum', 0, b2.STUDY_B2, freeze_sha256=FS)
    for key in ('hidden', 'memory', 'critic_gradient', 'dtype', 'torch_threads', 'critic', 'transitions_per_update',
                'gamma_ref', 'reference_duration_s', 'lambda_ref', 'lambda_mode', 'total_policy_decisions',
                'checkpoint_decisions', 'ppo', 'freeze', 'execution_manifest', 'time_model', 'config_version'):
        assert ours[key] == e1[key], key
    for arm in b2.ARMS:
        b2.validate(arm, b2.config_for(arm, 5, b2.STUDY_B2, freeze_sha256=FS), b2.STUDY_B2, FS)


@pytest.mark.parametrize('mutation', ['learning_rate', 'entropy', 'candidate_pair', 'budget', 'grid', 'block', 'suite',
                                      'study_role', 'freeze', 'namespace'])
def test_the_validator_refuses_every_changed_binding(mutation):
    arm, study = 'quantum', b2.STUDY_B2
    cfg = b2.config_for(arm, 3, study, freeze_sha256=FS)
    approved = FS
    if mutation == 'learning_rate':
        cfg['ppo']['learning_rate'] = 1e-3
    elif mutation == 'entropy':
        cfg['ppo']['entropy_coef'] = 0.01
    elif mutation == 'candidate_pair':
        cfg['candidate'] = 'lr1e-3_ent0'
        cfg['ppo']['learning_rate'] = 1e-3
    elif mutation == 'budget':
        cfg['total_policy_decisions'] = 20000
    elif mutation == 'grid':
        cfg['checkpoint_decisions'] = cfg['checkpoint_decisions'][:-1]
    elif mutation == 'block':
        cfg['block'] = 36
    elif mutation == 'suite':
        cfg['evaluation']['suite'] = 'e1_evaluation'
    elif mutation == 'study_role':
        cfg['role'] = 'e1_evaluation'
    elif mutation == 'freeze':
        approved = 'e' * 64
    elif mutation == 'namespace':
        study = dataclasses.replace(b2.STUDY_B2, training_namespace='dq_development')
    with pytest.raises((p4.P4Error, b2.B2Error)):
        b2.validate(arm, cfg, study, approved)


def test_launch_binds_the_approved_freeze_and_the_pristine_frozen_campaign(tmp_path, monkeypatch):
    frozen = {'files': {b2.CAMPAIGN_README: 'x'}}
    monkeypatch.setattr(p4, 'require_approved_freeze', lambda sha, study=None: (Path('/f'), 'a' * 64, frozen))
    monkeypatch.setattr(dq, 'require_open', lambda ns: ns)
    with pytest.raises(b2.B2Error, match='approved campaign'):
        b2.launch(tmp_path, 1, 'a' * 64)
    other = tmp_path / 'b2_campaign_v3'
    other.mkdir()
    monkeypatch.setattr(b2, 'CAMPAIGN', other)
    (other / 'README.md').write_text('readme')
    (other / 'runs').mkdir()
    with pytest.raises(b2.B2Error, match='pristine'):
        b2.launch(other, 1, 'a' * 64)
    (other / 'runs').rmdir()
    with pytest.raises(b2.B2Error, match='README'):
        b2.launch(other, 1, 'a' * 64)


def test_arms_share_the_trunk_and_the_twin_is_the_same_policy_as_the_blocks_hq():
    cfg = b2.config_for('quantum', 7, b2.STUDY_B2, freeze_sha256=FS)
    hq, cm, twin = (b2.dq_build_model(a, cfg, 7) for a in ('quantum', 'bottleneck', 'surrogate625'))
    for part in ('encoder', 'gru', 'critic'):
        for a, b, c in zip(getattr(hq, part).state_dict().values(), getattr(cm, part).state_dict().values(),
                           getattr(twin, part).state_dict().values()):
            assert torch.equal(a, b) and torch.equal(a, c)
    assert b2.TWIN_RECORDS[('dq_b2', 7)]['static']['static_failures'] == [] and twin.training
    x = torch.randn(3, 5, hq.config['obs_dim'], dtype=torch.float64)
    with torch.no_grad():
        assert float((hq.eval().policy(x)[0] - twin.eval().policy(x)[0]).abs().max()) <= 1e-9


def test_other_studies_keep_e1s_build():
    model = b2.dq_build_model('surrogate625', dict(b2.config_for('surrogate625', 0, b2.STUDY_B2), study='e1'), 10)
    assert model.config['actor_head'] == 'surrogate625' and ('e1', 10) not in b2.TWIN_RECORDS


def test_precision_is_computed_from_the_e1_report_of_record():
    d = b2.e1_paired()
    assert len(d['quantum_minus_bottleneck']) == 12
    assert abs(np.std(d['quantum_minus_bottleneck'], ddof=1) - 0.01366569487701738) < 1e-12
    assert 'not a twin' in b2.precision()['source']


def test_manifest_lists_108_runs_in_triplet_order_and_binds_the_campaign():
    m = b2.manifest()
    assert len(m['runs']) == 108 and m['runs'][:3] == ['quantum__b0', 'bottleneck__b0', 'surrogate625__b0']
    assert m['campaign'] == 'output/study/gnss_qrl_v1/dq/b2_campaign_v3'
    assert b2.CAMPAIGN_README in b2.FROZEN_FILES


def test_development_runs_cannot_carry_a_freeze_hash(tmp_path):
    rec = b2.run_one(('dq_b2_smoke', 'quantum', b2.SMOKE_BLOCK, str(tmp_path / 'r'), 'dq_b2_test', 'a' * 64, tmp_path))
    assert rec['status'] == 'refused' and 'development label' in rec['error']


# -- a real (tiny) development run of the twin through E1's trainer ------------------------------------------------------

def test_twin_arm_trains_through_e1s_trainer_on_the_development_namespace(tmp_path):
    record = b2.run_one(('dq_b2_smoke', 'surrogate625', b2.SMOKE_BLOCK, str(tmp_path / 'runs'), 'dq_b2_test',
                         b2.DEV_FREEZE_LABEL, tmp_path))
    assert record['status'] == 'completed', record.get('error', '') + record.get('traceback', '')
    assert record['twin_parity']['P4']['max_abs_dlogprob'] <= 1e-9
    assert record['twin_parity']['static']['static_failures'] == []
    assert record['summary']['updates'] == 1 and record['config_sha256'] == dq.canonical_sha256(record['config'])


# -- the integrity-proving analysis on a fabricated, internally consistent campaign ---------------------------------------

def fabricate(root, effect_twin=-0.01, fail=(), duplicate_manifest_identity=False):
    """A complete, consistent synthetic campaign + frozen manifest + freeze that p4_train.verify_freeze accepts against
    the LIVE tree; returns (freeze_path, freeze_sha)."""
    rng = np.random.default_rng(0)
    tree = p4.live_copy_tree()
    runs = [f'{a}__b{b}' for b in an.BLOCKS for a in an.ARMS]
    if duplicate_manifest_identity:
        runs = runs + [runs[0]]
    manifest = {'runs': runs, 'grid': b2.GRID, 'budget': b2.BUDGET, 'campaign': str(root)}
    manifest['manifest_sha256'] = dq.canonical_sha256(manifest)
    (root / 'manifest.json').write_text(json.dumps({'seed_manifest': manifest}))
    (root / 'member.txt').write_text('a frozen member')
    freeze = {'files': {str(root / 'member.txt'): hashlib.sha256(b'a frozen member').hexdigest()},
              'seed_manifest': str(root / 'manifest.json'), 'seed_manifest_sha256': manifest['manifest_sha256'],
              'copy_code_tree_sha256': tree}
    (root / 'freeze.json').write_text(json.dumps(freeze))
    fsha = hashlib.sha256((root / 'freeze.json').read_bytes()).hexdigest()
    (root / 'launch.json').write_text(json.dumps({'freeze': str(root / 'freeze.json'), 'freeze_sha256': fsha,
                                                 'code_tree_sha256': tree, 'workers': 12}))
    (root / 'records').mkdir()
    for blk in an.BLOCKS:
        base = 0.02 + 0.005 * rng.standard_normal()
        for arm, shift in (('quantum', 0.), ('bottleneck', 0.), ('surrogate625', -effect_twin)):
            cfg = b2.config_for(arm, blk, b2.STUDY_B2, freeze_sha256=fsha)
            rec = {'study': 'dq_b2', 'arm': arm, 'role': b2.ARMS[arm]['role'], 'block': blk, 'config': cfg,
                   'config_sha256': dq.canonical_sha256(cfg), 'protocol_freeze_sha256': fsha, 'code_tree_sha256': tree}
            d = root / 'runs' / f'{arm}__b{blk}' / 'run' / arm
            d.mkdir(parents=True)
            if (arm, blk) in fail:
                summary = {'status': 'algorithmic_failure', 'auc': None, 'updates': 3, 'block': blk, 'arm': arm}
                events = [{'kind': 'arm_start'}, {'kind': 'algorithmic_failure'}, dict(summary, kind='arm_end')]
                rec.update(status='algorithmic_failure', summary=summary)
            else:
                level = base + shift + 0.004 * rng.standard_normal()
                values = [level * p / b2.BUDGET for p in b2.GRID]
                events = [{'kind': 'arm_start'}]
                for i, (p, v) in enumerate(zip(b2.GRID, values)):
                    if i:
                        events.append({'kind': 'episode', 'J': 0.0, 'decisions': 10, 'termination_reason': 'mission_complete'})
                        events.append({'kind': 'update', 'update_index': i, 'metrics': {'approx_kl': 0.01}})
                    events.append({'kind': 'checkpoint', 'final': False, 'points': [p]})
                    eps = [{'tp_timely': 0.5, 'fp_spatial': 0.3, 'return_ok': True, 'fixes_attempted': 3}]
                    events.append({'kind': 'evaluation', 'points': [p], 'results': {'sample': {
                        'weighted_J': v, 'per_cell_J': {c: v for c in an.TARGET_CELLS}, 'episodes': eps}}})
                events.append({'kind': 'checkpoint', 'final': True, 'points': []})
                auc = p4.learning_curve_auc(b2.GRID, values, b2.BUDGET)
                summary = {'status': 'completed', 'auc': {'sample': auc}, 'updates': 10, 'policy_decisions': 30000,
                           'episodes': 10, 'block': blk, 'arm': arm}
                events.append(dict(summary, kind='arm_end'))
                (d / 'checkpoints').mkdir()
                (d / 'checkpoints' / 'final_update00010_decisions00030000.pt').write_text('x')
                rec.update(status='completed', summary=summary)
                if arm == 'surrogate625':
                    rec['twin_parity'] = {'static': {'static_failures': []}, 'P4': {'max_abs_dlogprob': 1e-15}}
            (d / 'metrics.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n')
            (root / 'records' / f'{arm}__b{blk}.json').write_text(json.dumps(rec))
    return root / 'freeze.json', fsha


def _rewrite(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def _rewrite_events(root, arm, blk, fn):
    p = root / 'runs' / f'{arm}__b{blk}' / 'run' / arm / 'metrics.jsonl'
    events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    events = fn(events)
    p.write_text('\n'.join(json.dumps(e) for e in events) + '\n')


def test_a_consistent_campaign_is_proven_and_analysed(tmp_path):
    freeze, fsha = fabricate(tmp_path)
    r = an.analyze(tmp_path, freeze, fsha)
    assert r['status_counts'] == {'completed': 108}
    assert r['primary']['B-H2']['rejected'] and r['primary']['B-H2']['mean'] < 0
    assert 'preselected training configurations' in r['reading']


def test_failed_runs_are_outcomes_under_the_missing_block_rule(tmp_path):
    freeze, fsha = fabricate(tmp_path, fail={('surrogate625', b) for b in range(4)})
    r = an.analyze(tmp_path, freeze, fsha)
    assert r['primary']['B-H2']['estimable'] is False and len(r['primary']['B-H2']['dropped_blocks']) == 4
    assert r['primary']['B-H1']['estimable']


MUTATIONS = {
    'altered_stored_auc': lambda root: _rewrite(root / 'records' / 'quantum__b3.json',
                                                lambda d: d['summary']['auc'].__setitem__('sample', 0.5)),
    'missing_grid_point': lambda root: _rewrite_events(root, 'quantum', 3, lambda ev: [
        e for e in ev if not (e['kind'] == 'evaluation' and e['points'] == [3000])]),
    'duplicate_grid_point': lambda root: _rewrite_events(root, 'quantum', 3, lambda ev: ev[:3] + [ev[2]] + ev[3:]),
    'reordered_grid': lambda root: _rewrite_events(root, 'quantum', 3, lambda ev: [
        dict(e, points=[6000]) if e['kind'] == 'evaluation' and e['points'] == [3000] else
        dict(e, points=[3000]) if e['kind'] == 'evaluation' and e['points'] == [6000] else e for e in ev]),
    'extra_record': lambda root: (root / 'records' / 'quantum__b36.json').write_text('{}'),
    'misnamed_record': lambda root: _rewrite(root / 'records' / 'quantum__b3.json', lambda d: d.__setitem__('block', 4)),
    'wrong_config': lambda root: _rewrite(root / 'records' / 'bottleneck__b5.json',
                                          lambda d: d['config']['ppo'].__setitem__('learning_rate', 1e-3)),
    'wrong_tree': lambda root: _rewrite(root / 'records' / 'bottleneck__b5.json',
                                        lambda d: d.__setitem__('code_tree_sha256', 'x' * 64)),
    'wrong_freeze': lambda root: _rewrite(root / 'records' / 'bottleneck__b5.json',
                                          lambda d: d.__setitem__('protocol_freeze_sha256', 'x' * 64)),
    'missing_twin_certificate': lambda root: _rewrite(root / 'records' / 'surrogate625__b2.json',
                                                      lambda d: d.pop('twin_parity')),
    'failed_twin_certificate': lambda root: _rewrite(root / 'records' / 'surrogate625__b2.json',
                                                     lambda d: d['twin_parity']['static'].__setitem__('static_failures', ['P2'])),
    'record_contradicts_arm_end': lambda root: _rewrite(root / 'records' / 'quantum__b7.json',
                                                        lambda d: d['summary'].__setitem__('episodes', 11)),
    'non_finite_value': lambda root: _rewrite_events(root, 'quantum', 3, lambda ev: [
        dict(e, results={'sample': dict(e['results']['sample'], weighted_J=float('nan'))})
        if e['kind'] == 'evaluation' and e['points'] == [9000] else e for e in ev]),
    'missing_metrics': lambda root: (root / 'runs' / 'quantum__b8' / 'run' / 'quantum' / 'metrics.jsonl').unlink(),
    'unknown_status': lambda root: _rewrite(root / 'records' / 'quantum__b9.json', lambda d: d.__setitem__('status', 'done')),
    # R41-1: a completed run relabelled as each accepted outcome, and with its configuration removed
    **{f'completed_relabelled_{status}': (lambda status: lambda root: _rewrite(
        root / 'records' / 'quantum__b3.json', lambda d: d.update(status=status, error='relabelled')))(status)
       for status in ('algorithmic_failure', 'infrastructure_failure', 'parity_refused', 'refused')},
    'completed_relabelled_refused_without_config': lambda root: _rewrite(
        root / 'records' / 'quantum__b3.json', lambda d: (d.update(status='refused', error='relabelled'),
                                                         d.pop('config'), d.pop('summary'))),
    'relabelled_with_summary_removed': lambda root: _rewrite(
        root / 'records' / 'quantum__b3.json', lambda d: (d.update(status='infrastructure_failure', error='x'),
                                                         d.pop('summary'))),
    # R41-2: freeze members, campaign identity, launch record
    'changed_freeze_member': lambda root: (root / 'member.txt').write_text('changed'),
    'missing_launch_json': lambda root: (root / 'launch.json').unlink(),
    'altered_launch_json': lambda root: _rewrite(root / 'launch.json', lambda d: d.__setitem__('freeze_sha256', '0' * 64)),
    'launch_json_foreign_tree': lambda root: _rewrite(root / 'launch.json', lambda d: d.__setitem__('code_tree_sha256', 'x' * 64)),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_every_integrity_mutation_is_refused_not_dropped(tmp_path, name):
    freeze, fsha = fabricate(tmp_path)
    MUTATIONS[name](tmp_path)
    with pytest.raises(an.IntegrityError):
        an.analyze(tmp_path, freeze, fsha)


def test_a_genuine_algorithmic_failure_remains_an_outcome(tmp_path):
    freeze, fsha = fabricate(tmp_path, fail={('surrogate625', 5)})
    r = an.analyze(tmp_path, freeze, fsha)
    assert r['status_counts'] == {'completed': 107, 'algorithmic_failure': 1}
    assert r['primary']['B-H2']['dropped_blocks'] == [5]


def test_a_foreign_campaign_path_is_refused(tmp_path):
    (tmp_path / 'c').mkdir()
    freeze, fsha = fabricate(tmp_path / 'c')
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises(an.IntegrityError, match='frozen campaign'):
        an.analyze(other, freeze, fsha)


def test_a_duplicated_manifest_identity_is_refused(tmp_path):
    freeze, fsha = fabricate(tmp_path, duplicate_manifest_identity=True)
    with pytest.raises(an.IntegrityError, match='108 unique'):
        an.analyze(tmp_path, freeze, fsha)


def test_a_wrong_freeze_hash_is_refused(tmp_path):
    freeze, _ = fabricate(tmp_path)
    with pytest.raises(an.IntegrityError, match='freeze'):
        an.analyze(tmp_path, freeze, '0' * 64)


def test_zero_variance_behaviour_is_predeclared():
    zero = an.t_test({b: 0.0 for b in range(36)}, [])
    assert zero['p'] == 1.0 and zero['t'] == 0.0 and zero['degenerate']
    const = an.t_test({b: -0.002 for b in range(36)}, [])
    assert const['p'] == 0.0 and const['t'] == float('-inf') and const['degenerate']
    normal = an.t_test({b: 0.001 * (b % 3 - 1) for b in range(36)}, [])
    assert not normal['degenerate'] and 0 < normal['p'] <= 1


def test_tost_requires_the_90_percent_interval_inside_delta():
    assert an.tost({'estimable': True, 'ci90': [-0.003, 0.004]})['equivalent_within_delta'] is True
    assert an.tost({'estimable': True, 'ci90': [-0.007, 0.001]})['equivalent_within_delta'] is False
