"""Unit tests of the E3 runner and analysis (no training, no new run; synthetic data and stored KL/AUC only).

    /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -B -m pytest -q -p no:cacheprovider test_b2k.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import b2k_analysis as an
import runner


# -- selection rule ------------------------------------------------------------------------------------------------------

def _synthetic(level, slope, candidates, noise=None):
    """{lr: [KL of each block]} from ln KL = level + slope ln lr (+ per-run multiplicative noise)."""
    noise = noise or {}
    return {lr: [math.exp(level + slope * math.log(lr)) * f for f in noise.get(lr, [1.0, 1.0, 1.0])]
            for lr in candidates}


@pytest.mark.parametrize('arm', ['quantum', 'surrogate625'])
def test_select_recovers_exact_lr_on_a_noiseless_line(arm):
    s, cands, T = runner.SLOPES[arm], runner.CANDIDATES[arm], runner.TARGETS[arm]['value']
    true_lr = cands[1] * 1.07
    level = math.log(T) - s * math.log(true_lr)
    sel = runner.select(_synthetic(level, s, cands), T, s, cands)
    assert abs(sel['lr_selected_raw'] / true_lr - 1) < 1e-12
    assert sel['lr_selected'] == runner.round3(true_lr) and not sel['stage_1b_needed']
    assert abs(sel['descriptive_not_used']['ols_slope_pilot'] - s) < 1e-12


def test_select_uses_fixed_slope_not_the_pilot_slope():
    arm = 'quantum'
    s, cands, T = runner.SLOPES[arm], runner.CANDIDATES[arm], runner.TARGETS[arm]['value']
    # pilot points on a line of slope 0.3 through the target at the middle candidate: the fixed-slope rule must still
    # return exp(xbar + (ln T - ybar)/s), computed by hand here
    kl = {lr: [T * (lr / cands[1]) ** 0.3] * 3 for lr in cands}
    x, y = np.log(cands), np.log([np.mean(kl[lr]) for lr in cands])
    expected = math.exp(x.mean() + (math.log(T) - y.mean()) / s)
    sel = runner.select(kl, T, s, cands)
    assert abs(sel['lr_selected_raw'] - expected) < 1e-15
    assert abs(sel['descriptive_not_used']['ols_slope_pilot'] - 0.3) < 1e-12


def test_select_arithmetic_mean_within_candidate():
    arm = 'surrogate625'
    s, cands, T = runner.SLOPES[arm], runner.CANDIDATES[arm], runner.TARGETS[arm]['value']
    kl = {cands[0]: [0.004, 0.006, 0.008], cands[1]: [0.006, 0.010], cands[2]: [0.009, 0.011, 0.013]}
    sel = runner.select(kl, T, s, cands)
    assert sel['mean_kl'] == {runner.lr_str(cands[0]): pytest.approx(0.006, abs=1e-15),
                              runner.lr_str(cands[1]): pytest.approx(0.008, abs=1e-15),
                              runner.lr_str(cands[2]): pytest.approx(0.011, abs=1e-15)}


def test_range_guard_flags_stage_1b():
    arm = 'surrogate625'
    s, cands, T = runner.SLOPES[arm], runner.CANDIDATES[arm], runner.TARGETS[arm]['value']
    level = math.log(T) - s * math.log(cands[-1] * 2.0)              # true lr 2e-3 > 1e-3 * 1.5
    sel = runner.select(_synthetic(level, s, cands), T, s, cands)
    assert sel['stage_1b_needed'] and not sel['range_guard']['inside']
    level = math.log(T) - s * math.log(cands[-1] * 1.45)             # inside the guard
    assert not runner.select(_synthetic(level, s, cands), T, s, cands)['stage_1b_needed']


def test_select_needs_two_candidates():
    with pytest.raises(ValueError):
        runner.select({1e-3: [0.01], 7e-4: []}, 0.0078, 0.94, (5e-4, 7e-4, 1e-3))


def test_round3_and_spelling():
    assert runner.lr_str(7.7593e-4) == '7.76e-04' and runner.round3(7.7593e-4) == 7.76e-4
    assert runner.lr_str(1.4333e-3) == '1.43e-03'


def test_pilot_blocks_are_disjoint_and_development():
    blocks = [runner.pilot_blocks(i) for i in range(3)]
    flat = [b for bl in blocks for b in bl]
    assert len(set(flat)) == len(flat) == 9 and min(flat) == 90 and max(flat) == 98
    assert not set(flat) & set(range(36))


# -- configurations (no run) ---------------------------------------------------------------------------------------------

def test_main_config_differs_from_b2_only_in_declared_keys():
    import dq_b2
    for arm in runner.SETS.values():
        new = runner.config_main(arm, 7, 1.23e-3, plan='a' * 64)
        old = dq_b2.config_for(arm, 7, dq_b2.STUDY_B2, freeze_sha256='a' * 64)
        assert {k for k in new if new[k] != old[k]} == {'ppo', 'candidate', 'note'}
        assert {k for k in new['ppo'] if new['ppo'][k] != old['ppo'][k]} == {'learning_rate'}
        assert new['ppo']['learning_rate'] == 1.23e-3 and new['ppo']['entropy_coef'] == 0.0
        runner.p4.validate_config(new)                              # E1's validator, as B2's runs passed it


def test_pilot_config_is_development_without_evaluation():
    import dq_b2
    c = runner.config_pilot('surrogate625', 93, 7e-4)
    assert c['study'] == 'dq_b2_smoke' and c['evaluation'] is None and c['total_policy_decisions'] == 30000
    assert dq_b2.STUDY_SMOKE.training_namespace == 'dq_development'
    assert c['ppo'] == dict(dq_b2.PPO_BASE, learning_rate=7e-4, entropy_coef=0.0)


def test_static_checks_all_pass():
    r = runner.static_checks()
    assert all(v['ok'] for v in r.values()), {k: v for k, v in r.items() if not v['ok']}


# -- KL statistic and analysis -------------------------------------------------------------------------------------------

def test_kl_of_matches_independent_reading_of_a_stored_run():
    f = runner.CMLR_RUNS / 'C1_bottleneck__b0/run/bottleneck/metrics.jsonl'
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    kls = [r['metrics']['approx_kl'] for r in rows if r['kind'] == 'update']
    assert runner.kl_of(f)['mean'] == float(np.mean(kls)) and runner.kl_of(f)['updates'] == len(kls)


def test_t_test_against_scipy_and_holm_family_of_two():
    rng = np.random.default_rng(0)
    a = {b: {'auc': float(x)} for b, x in enumerate(rng.normal(0.02, 0.01, 36))}
    b = {k: {'auc': v['auc'] + float(rng.normal(0.004, 0.01))} for k, v in a.items()}
    r = an.t_test(*an.differences(a, b))
    t = stats.ttest_rel([a[k]['auc'] for k in a], [b[k]['auc'] for k in a])
    assert r['estimable'] and abs(r['p'] - t.pvalue) < 1e-12 and r['n'] == 36 and r['dropped_blocks'] == []
    est = lambda p: {'estimable': True, 'p': p}                               # noqa: E731
    rej = lambda tests: {k: v['rejected'] for k, v in an.holm(tests).items()}  # noqa: E731
    assert rej({'K-H1': est(0.01), 'K-H2': est(0.04)}) == {'K-H1': True, 'K-H2': True}
    assert rej({'K-H1': est(0.03), 'K-H2': est(0.04)}) == {'K-H1': False, 'K-H2': False}
    assert rej({'K-H1': est(0.04), 'K-H2': est(0.02)}) == {'K-H1': True, 'K-H2': True}
    assert rej({'K-H1': est(0.2), 'K-H2': est(0.02)}) == {'K-H1': False, 'K-H2': True}


# -- review B1: B2's frozen rules (dropped-block ceiling, not estimable, degenerate) ------------------------------------

def _diff_sets(values, dropped=()):
    a = {b: {'auc': float(v)} for b, v in zip([b for b in an.BLOCKS if b not in dropped], values)}
    return a, {b: {'auc': 0.0} for b in an.BLOCKS}


def test_more_than_three_dropped_blocks_is_not_estimable():
    rng = np.random.default_rng(1)
    a, b = _diff_sets(rng.normal(-0.01, 0.01, 33), dropped=(3, 9, 30))
    r3 = an.t_test(*an.differences(a, b))
    assert r3['estimable'] and r3['n'] == 33 and r3['dropped_blocks'] == [3, 9, 30]
    a, b = _diff_sets(rng.normal(-0.01, 0.01, 32), dropped=(3, 9, 30, 31))
    r4 = an.t_test(*an.differences(a, b))
    assert r4['estimable'] is False and r4['p'] is None and r4['dropped_blocks'] == [3, 9, 30, 31]
    assert an.tost(r4) == {'equivalent_within_delta': False, 'delta': an.TOST_DELTA, 'reason': 'not estimable'}
    d, text = an.reading('K-H2', dict(r4, rejected=False, tost=an.tost(r4)), True, 1.0)
    assert d == 'not estimable' and 'NOT ESTIMABLE' in text


def test_not_estimable_is_left_out_of_holm_but_the_denominator_stays_two():
    ne = {'estimable': False, 'p': None}
    for p, expected in ((0.03, False), (0.025, True), (0.02, True)):
        h = an.holm({'K-H1': ne, 'K-H2': {'estimable': True, 'p': p}})
        assert h['K-H2']['rejected'] is expected and h['K-H2']['holm_threshold'] == 0.025
        assert h['K-H1'] == {'rejected': False, 'holm_threshold': None}


def test_degenerate_sd_zero():
    a, b = _diff_sets([0.0] * 36)
    r = an.t_test(*an.differences(a, b))
    assert r['estimable'] and r['degenerate'] and r['p'] == 1.0 and r['t'] == 0.0
    a, b = _diff_sets([-0.004] * 36)
    r = an.t_test(*an.differences(a, b))
    assert r['degenerate'] and r['p'] == 0.0 and r['t'] == -math.inf and r['ci95'] == [-0.004, -0.004]
    assert an.holm({'K-H1': {'estimable': True, 'p': 0.5}, 'K-H2': r})['K-H2']['rejected']


def test_descriptive_interval_has_no_p_value():
    rng = np.random.default_rng(2)
    a, b = _diff_sets(rng.normal(0.0, 0.01, 36))
    iv = an.interval(a, b)
    assert 'p' not in iv and set(iv) == {'n', 'dropped_blocks', 'mean', 'lo', 'hi', 'dz', 'share_negative'}


# -- reading rules (incl. review M2) ---------------------------------------------------------------------------------

def _res(mean, rejected, lo, hi, tost_ok=False):
    return {'estimable': True, 'mean': mean, 'rejected': rejected, 'ci95': [lo, hi],
            'tost': {'equivalent_within_delta': tost_ok}}


def test_reading_rules():
    b = an.B_H2_RECORD
    assert an.reading('K-H2', _res(-0.010, True, -0.014, -0.006), True, 1.0)[0] == 'below zero'
    d, text = an.reading('K-H2', _res(-0.004, True, b + 0.0005, -0.001), True, 1.0)
    assert d == 'partial reduction' and 'part of B-H2' in text
    d, text = an.reading('K-H2', _res(-0.003, False, b - 0.0005, 0.002), True, 1.0)
    assert d == 'inconclusive' and 'neither removes nor confirms' in text
    d, text = an.reading('K-H2', _res(-0.001, False, b + 0.002, 0.004, tost_ok=True), True, 1.0)
    assert d == 'not resolved' and 'update-size account' in text and 'equivalence margin' in text
    d, text = an.reading('K-H2', _res(-0.001, False, b + 0.002, 0.004), True, 1.0)
    assert d == 'not resolved' and 'equivalence margin' not in text
    assert an.reading('K-H2', _res(-0.006, False, -0.011, -0.0005), True, 1.0)[0] == 'not resolved after Holm'
    assert an.reading('K-H2', _res(0.006, True, 0.002, 0.010), True, 1.0)[0] == 'above zero'
    assert an.reading('K-H1', _res(-0.01, True, -0.015, -0.005), True, 1.1)[0] == 'below zero'
    assert an.reading('K-H1', _res(0.01, True, 0.005, 0.015), True, 1.1)[0] == 'above zero'
    d, text = an.reading('K-H1', _res(-0.002, False, -0.006, 0.002, tost_ok=True), True, 0.9)
    assert d == 'not resolved' and 'equivalence margin' in text
    # round 2: K-H1 not rejected although its unadjusted interval excludes 0 -> 'not resolved after Holm' (no
    # update-size statement, no TOST add-on), as for K-H2
    for name in ('K-H1', 'K-H2'):
        for lo, hi in ((-0.0076, -0.0004), (0.0004, 0.0076)):
            d, text = an.reading(name, _res((lo + hi) / 2, False, lo, hi, tost_ok=True), True, 1.0)
            assert d == 'not resolved after Holm' and 'no update-size statement' in text
            assert 'consistent with' not in text and 'equivalence margin' not in text and 'update-size account' not in text
    # not matched: only the pre-declared one-sided statements
    assert 'smaller' in an.reading('K-H2', _res(-0.01, True, -0.014, -0.006), False, 0.6)[1]
    assert 'smaller' in an.reading('K-H2', _res(-0.004, True, b + 0.0005, -0.001), False, 0.6)[1]
    assert 'no update-size statement' in an.reading('K-H2', _res(-0.01, True, -0.014, -0.006), False, 1.4)[1]
    assert 'no update-size statement' in an.reading('K-H2', _res(-0.003, False, -0.009, 0.002), False, 0.6)[1]
    assert 'larger' in an.reading('K-H1', _res(-0.01, True, -0.015, -0.005), False, 1.4)[1]
    assert 'no update-size statement' in an.reading('K-H1', _res(-0.01, True, -0.015, -0.005), False, 0.6)[1]
    d, text = an.reading('K-H1', _res(-0.004, False, -0.0076, -0.0004), False, 1.4)
    assert d == 'not resolved after Holm' and 'no update-size statement' in text


def test_not_resolved_after_holm_through_holm_both_orders():
    """Round 2 (verifier's two reproductions): Holm itself produces the 'not resolved after Holm' cases.
    (1) K-H1 p = 0.030 first in the order, tested at 0.025 and not rejected, interval [-0.0076, -0.0004];
    (2) K-H2 p = 0.040 behind an unrejected K-H1: Holm stops, K-H2 is not tested (threshold None), interval
    [-0.0059, -0.0001]. Neither gets an update-size statement or a TOST add-on, matched or not."""
    tests = {'K-H1': dict(_res(-0.0040, None, -0.0076, -0.0004, tost_ok=True), p=0.030),
             'K-H2': dict(_res(-0.0030, None, -0.0059, -0.0001, tost_ok=True), p=0.040)}
    h = an.holm(tests)
    assert h == {'K-H1': {'rejected': False, 'holm_threshold': 0.025},
                 'K-H2': {'rejected': False, 'holm_threshold': None}}
    for name, t in tests.items():
        t.update(h[name])
        for matched in (True, False):
            d, text = an.reading(name, t, matched, 1.0 if matched else 1.4)
            assert d == 'not resolved after Holm' and 'no update-size statement' in text
            assert 'consistent with' not in text and 'equivalence margin' not in text
    # K-H2 tested at 0.025 (smaller p, first) and not rejected: the same direction for K-H2
    tests = {'K-H1': dict(_res(-0.004, None, -0.009, 0.001), p=0.20),
             'K-H2': dict(_res(-0.003, None, -0.0059, -0.0001), p=0.040)}
    h = an.holm(tests)
    assert h['K-H2'] == {'rejected': False, 'holm_threshold': 0.025}
    tests['K-H2'].update(h['K-H2'])
    tests['K-H1'].update(h['K-H1'])
    assert an.direction('K-H2', tests['K-H2']) == 'not resolved after Holm'
    assert an.direction('K-H1', tests['K-H1']) == 'not resolved'


def test_beside_b_h2():
    b = an.B_H2_RECORD
    r = an.beside_b_h2(_res(-0.003, False, b - 0.001, 0.001))
    assert r['contains_0'] and r['contains_B-H2_of_record'] and not r['entirely_above_B-H2_of_record']
    assert an.beside_b_h2({'estimable': False}) == {'B-H2_of_record': b, 'K-H2_estimable': False}


def test_b_h2_record_constant_equals_the_stored_runs():
    st = an.load_stored()
    assert an.t_test(*an.differences(st['HQ_3e-3'], st['twin_1e-3']))['mean'] == an.B_H2_RECORD


def test_analysis_refuses_partial_data(tmp_path):
    (tmp_path / 'records').mkdir()
    (tmp_path / 'records' / 'T1_quantum__b0.json').write_text('{}')
    with pytest.raises(SystemExit):
        an.main(provisional=False, root=tmp_path)


def test_analysis_targets_equal_runner_targets():
    assert an.T_H2 == runner.TARGETS['surrogate625']['value'] and an.T_H1 == runner.TARGETS['quantum']['value']
    assert an.TOST_DELTA == runner.dq_b2.TOST_DELTA and an.TOLERANCE == runner.TOLERANCE


def test_run_path_never_passes_the_calendar_gate(monkeypatch):
    """Build, twin transplant and P4 parity (the per-run steps before train_arm) with dq.require_open made to refuse:
    the post-hoc path must not depend on the 2026-10-31 cut-off (train_arm itself has no gate, see README_BUILD)."""
    import dq_b2
    import dq_runtime

    def refuse(*a, **k):
        raise dq_runtime.DQError('cut-off (test)')
    monkeypatch.setattr(dq_runtime, 'require_open', refuse)
    monkeypatch.setattr(dq_runtime, '_past_cutoff', lambda now=None: True)
    runner.torch.set_num_threads(1)
    cfg = runner.config_main('surrogate625', 0, 7.76e-4)
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    twin = dq_b2.dq_build_model('surrogate625', cfg, 0)
    assert type(twin.actor).__name__ == 'SurrogateActorHead'
    parity = dq_b2.twin_trajectory_parity(dq_b2.STUDY_B2, rt, cfg, 0)
    assert parity['max_abs_dlogprob'] <= 1e-9


# -- review fixes M7 / M8: single-instance lock, atomic records, unparsable record = no record ---------------------------

def test_single_instance_lock_refuses_a_second_holder(tmp_path):
    path = tmp_path / 'runner.lock'
    first = runner.single_instance(path)
    with pytest.raises(SystemExit):
        runner.single_instance(path)                  # another open file description: refused (flock, non-blocking)
    first.close()                                     # released: the next holder gets it
    runner.single_instance(path).close()


def test_lock_survives_in_a_forked_child_after_the_parent_side_closes(tmp_path):
    """A fork-pool worker inherits the lock: while it lives, a new launch is refused even if the launcher is gone."""
    import os
    import time as _time
    path = tmp_path / 'runner.lock'
    handle = runner.single_instance(path)
    pid = os.fork()
    if pid == 0:                                      # child: keeps the inherited descriptor open for a moment
        _time.sleep(1.5)
        os._exit(0)
    handle.close()                                    # the 'launcher' lets go; the child still holds the description
    with pytest.raises(SystemExit):
        runner.single_instance(path)
    os.waitpid(pid, 0)
    runner.single_instance(path).close()              # child gone: free again


def test_write_json_atomic_leaves_no_temporary_file(tmp_path):
    target = tmp_path / 'T1_quantum__b0.json'
    runner.write_json_atomic(target, {'status': 'completed', 'x': 1.5})
    assert json.loads(target.read_text()) == {'status': 'completed', 'x': 1.5}
    assert [p.name for p in tmp_path.iterdir()] == ['T1_quantum__b0.json']


def test_prepare_resume_rule(tmp_path):
    runs, records, intr = tmp_path / 'runs', tmp_path / 'records', tmp_path / 'interrupted' / 'U'
    runs.mkdir(); records.mkdir()
    names = ['T1_quantum__b0', 'T2_surrogate625__b0', 'T1_quantum__b1', 'T2_surrogate625__b1', 'T1_quantum__b2']
    (records / 'T1_quantum__b0.json').write_text(json.dumps({'status': 'completed'}))       # done: skipped
    (runs / 'T1_quantum__b0').mkdir()
    (records / 'T2_surrogate625__b0.json').write_text(json.dumps({'status': 'failure'}))    # a failure stays a failure
    (records / 'T1_quantum__b1.json').write_text('{"status": "compl')                        # truncated: no record
    (runs / 'T1_quantum__b1').mkdir()
    (runs / 'T2_surrogate625__b1').mkdir()                                                    # crash-killed, no record
    (records / '.T1_quantum__b2.json.99.tmp').write_text('{')                                 # stale temporary record
    todo, moved = runner.prepare(names, runs, records, intr)
    assert todo == ['T1_quantum__b1', 'T2_surrogate625__b1', 'T1_quantum__b2']
    assert sorted(moved) == sorted(['.T1_quantum__b2.json.99.tmp', 'T1_quantum__b1.json', 'T1_quantum__b1',
                                    'T2_surrogate625__b1'])
    assert sorted(p.name for p in records.iterdir()) == ['T1_quantum__b0.json', 'T2_surrogate625__b0.json']
    assert sorted(p.name for p in runs.iterdir()) == ['T1_quantum__b0']
    assert (intr / 'T1_quantum__b1.json').read_text() == '{"status": "compl'                  # moved unchanged
    assert runner.read_record(intr / 'T1_quantum__b1.json') is None


# -- review B2: integrity of the new runs (synthetic campaign, doctored copies, the real precheck run) -----------------

import hashlib                                                                                    # noqa: E402

HERE = Path(__file__).resolve().parent
SEL = json.loads((HERE / 'selection.json').read_text())
LR_SEL = {arm: SEL['arms'][arm]['lr'] for arm in runner.SETS.values()}


def _stored_synthetic():
    rng = np.random.default_rng(5)
    mk = lambda mu, kl: {b: {'auc': float(mu + rng.normal(0, 0.004)), 'J0': 0.1, 'Jfinal': 0.2, 'kl': kl,  # noqa: E731
                             'kl_updates': [kl] * 3, 'updates': 3} for b in an.BLOCKS}
    return {'HQ_1e-3': mk(0.02, 0.003), 'HQ_3e-3': mk(0.025, 0.0078), 'HQ_1e-2': mk(0.028, 0.007),
            'Cmatched_3e-3': mk(0.034, 0.0023), 'Cmatched_1e-2': mk(0.037, 0.0042),
            'twin_1e-3': mk(0.033, 0.0099), 'twin_3e-3': mk(0.035, 0.028)}


def _events(arm, block, auc_target_values, kls, grid, budget, duplicate_update=False, duplicate_eval=False,
            two_starts=False, auc_offset=0.0):
    ev = [{'kind': 'arm_start', 'arm': arm, 'block': block}]
    if two_starts:
        ev.append({'kind': 'arm_start', 'arm': arm, 'block': block})
    for i, k in enumerate(kls, 1):
        ev.append({'kind': 'update', 'update_index': i, 'metrics': {'approx_kl': k}})
        if duplicate_update and i == 2:
            ev.append({'kind': 'update', 'update_index': i, 'metrics': {'approx_kl': k}})
    for p, v in zip(grid, auc_target_values):
        ev.append({'kind': 'evaluation', 'points': [p], 'results': {'sample': {'weighted_J': v}}})
        if duplicate_eval and p == grid[3]:
            ev.append({'kind': 'evaluation', 'points': [p], 'results': {'sample': {'weighted_J': v}}})
    ev.append({'kind': 'checkpoint', 'final': True})
    auc = runner.p4.learning_curve_auc(list(grid), list(auc_target_values), budget)
    end = {'kind': 'arm_end', 'status': 'completed', 'auc': {'sample': auc + auc_offset}, 'updates': len(kls),
           'policy_decisions': budget, 'episodes': 600, 'block': block, 'arm': arm}
    return ev + [end], auc


def build_campaign(root, doctor=None, failed=()):
    """A complete, consistent synthetic E3 campaign under `root` (72 records + raw events); `doctor` names one defect."""
    root = Path(root)
    (root / 'records').mkdir(parents=True)
    (root / 'runs').mkdir()
    plan_text = ('synthetic PLAN for tests\nrunner.py ' + runner.RUNNER_SHA256 + '\nb2k_analysis.py ' +
                 hashlib.sha256((HERE / 'b2k_analysis.py').read_bytes()).hexdigest() + '\n' +
                 ' '.join(SEL['arms'][a]['lr_str'] for a in runner.SETS.values()) + '\n')
    if doctor == 'plan_lacks_analysis_hash':
        plan_text = plan_text.replace(hashlib.sha256((HERE / 'b2k_analysis.py').read_bytes()).hexdigest(), 'x')
    (root / 'PLAN.md').write_text(plan_text)
    plan_sha = hashlib.sha256(plan_text.encode()).hexdigest()
    (root / 'PLAN.sha256').write_text(f'{plan_sha}  PLAN.md\n')
    if doctor == 'plan_edited_after_hash':
        (root / 'PLAN.md').write_text(plan_text + 'edited\n')
    (root / 'precheck.json').write_text(json.dumps({'passed': True, 'runner_sha256': runner.RUNNER_SHA256}))
    (root / 'selection.json').write_text(json.dumps(SEL))
    rng = np.random.default_rng(11)
    for b in an.BLOCKS:
        for tag, arm in runner.SETS.items():
            name = f'{tag}_{arm}__b{b}'
            lr = LR_SEL[arm]
            cfg = runner.config_main(arm, b, lr, plan=plan_sha)
            grid, budget = cfg['checkpoint_decisions'], cfg['total_policy_decisions']
            first = b == 5 and (tag == 'T2' if doctor and doctor.startswith('twin') else tag == 'T1')
            if (tag, b) in failed:
                rec = {'kind': 'main', 'set': tag, 'arm': arm, 'block': b, 'learning_rate': lr, 'status': 'failure',
                       'plan_sha256': plan_sha, 'runner_sha256': runner.RUNNER_SHA256,
                       'error': 'RuntimeError: synthetic', 'config': cfg, 'code_tree_sha256': an.B2_TREE}
                if doctor == 'failure_with_completed_run' and first:
                    pass
                else:
                    (root / 'records' / f'{name}.json').write_text(json.dumps(rec, default=float))
                    continue
            js = [0.1 + 0.01 * i + float(rng.normal(0, 0.002)) for i in range(len(grid))]
            kls = [float(x) for x in rng.uniform(0.002, 0.012, 57)]
            ev, auc = _events(arm, b, js, kls, grid, budget,
                              duplicate_update=first and doctor == 'duplicate_update',
                              duplicate_eval=first and doctor == 'duplicate_eval',
                              two_starts=first and doctor == 'two_arm_starts',
                              auc_offset=1e-12 if first and doctor == 'auc_mismatch' else 0.0)
            d = root / 'runs' / name / 'run' / arm
            d.mkdir(parents=True)
            (d / 'metrics.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in ev))
            summ = {k: v for k, v in ev[-1].items() if k != 'kind'}
            summ['auc'] = {'sample': auc}
            rec = {'kind': 'main', 'set': tag, 'arm': arm, 'block': b, 'learning_rate': lr, 'status': 'completed',
                   'plan_sha256': plan_sha, 'runner_sha256': runner.RUNNER_SHA256, 'config': cfg,
                   'code_tree_sha256': an.B2_TREE, 'summary': summ,
                   'kl': {'mean': float(np.mean(kls)), 'median': float(np.median(kls)), 'updates': len(kls)}}
            if (tag, b) in failed:                                   # doctor 'failure_with_completed_run'
                rec.update(status='failure', error='RuntimeError: after training')
            if arm == 'surrogate625':
                rec['twin_parity'] = {'P4': {'max_abs_dlogprob': 2e-9 if first and doctor == 'twin_P4' else 1e-15},
                                      'static': {'static_failures': ['P1'] if first and doctor == 'twin_static' else []}}
            if first and doctor == 'config_lr':
                rec['config'] = runner.config_main(arm, b, 1e-3, plan=plan_sha)
            if first and doctor == 'record_plan':
                rec['plan_sha256'] = 'f' * 64
            if first and doctor == 'record_runner':
                rec['runner_sha256'] = '0' * 64
            if first and doctor == 'record_tree':
                rec['code_tree_sha256'] = '1' * 64
            if first and doctor == 'record_kl':
                rec['kl']['mean'] *= 1.0000001
            (root / 'records' / f'{name}.json').write_text(json.dumps(rec, default=float))
    if doctor == 'foreign_record':
        (root / 'records' / 'T3_quantum__b0.json').write_text('{}')
    if doctor == 'unparsable_record':
        (root / 'records' / 'T1_quantum__b7.json').write_text('{"kind": "ma')
    return root


def _run(root, provisional=False):
    return an.main(provisional=provisional, root=root, stored=_stored_synthetic(),
                   b2f={b: {'auc': 0.03} for b in an.BLOCKS})


def _no_p(obj):
    if isinstance(obj, dict):
        return 'p' not in obj and all(_no_p(v) for v in obj.values())
    return True


def test_clean_synthetic_campaign_passes_and_matches_hand_computation(tmp_path):
    root = build_campaign(tmp_path / 'c')
    out = _run(root)
    assert out['provisional'] is False and out['completed'] == {'T1_HQ': 36, 'T2_twin': 36} and not out['failed_runs']
    st = _stored_synthetic()
    twin = {b: json.loads((root / 'records' / f'T2_surrogate625__b{b}.json').read_text())['summary']['auc']['sample']
            for b in an.BLOCKS}
    kh2 = np.mean([st['HQ_3e-3'][b]['auc'] - twin[b] for b in an.BLOCKS])
    assert abs(out['K-H2_HQ_B2_minus_twin_new']['mean'] - kh2) < 1e-15
    assert out['K-H2_HQ_B2_minus_twin_new']['estimable'] and out['K-H1_HQ_new_minus_Cmatched_CMLR']['estimable']
    assert _no_p(out['descriptive'])                                                          # review M1
    assert out['descriptive']['twin_new_minus_HQ_frozen_B2F']['n'] == 36
    assert out['descriptive']['K-H2_beside_B-H2_of_record']['B-H2_of_record'] == an.B_H2_RECORD
    assert (root / 'b2k_results.json').exists()


@pytest.mark.parametrize('doctor', ['duplicate_update', 'duplicate_eval', 'two_arm_starts', 'auc_mismatch',
                                    'config_lr', 'record_plan', 'record_runner', 'record_tree', 'record_kl',
                                    'twin_P4', 'twin_static', 'plan_edited_after_hash', 'plan_lacks_analysis_hash',
                                    'foreign_record'])
def test_doctored_campaign_is_refused(tmp_path, doctor):
    root = build_campaign(tmp_path / 'c', doctor=doctor)
    with pytest.raises(an.IntegrityError):
        _run(root)
    assert not (root / 'b2k_results.json').exists()                                           # writes nothing


def test_failure_with_a_completed_run_on_disk_is_refused(tmp_path):
    root = build_campaign(tmp_path / 'c', doctor='failure_with_completed_run', failed={('T1', 5)})
    with pytest.raises(an.IntegrityError):
        _run(root)


def test_unparsable_record_counts_as_no_record(tmp_path):
    root = build_campaign(tmp_path / 'c', doctor='unparsable_record')
    with pytest.raises(SystemExit):
        _run(root)                                                                            # 71 of 72: waits
    out = _run(root, provisional=True)
    assert out['unparsable_records_counted_as_none'] == ['T1_quantum__b7.json'] and out['provisional']
    assert out['K-H1_HQ_new_minus_Cmatched_CMLR']['dropped_blocks'] == [7]
    assert (root / 'b2k_results_PROVISIONAL.json').exists() and not (root / 'b2k_results.json').exists()


def test_failed_runs_drop_blocks_and_four_make_k_h2_not_estimable(tmp_path):
    root = build_campaign(tmp_path / 'c3', failed={('T2', 1), ('T2', 2), ('T2', 3)})
    out = _run(root)
    assert out['K-H2_HQ_B2_minus_twin_new']['estimable'] and out['K-H2_HQ_B2_minus_twin_new']['n'] == 33
    assert len(out['failed_runs']) == 3
    root = build_campaign(tmp_path / 'c4', failed={('T2', 1), ('T2', 2), ('T2', 3), ('T2', 4)})
    out = _run(root)
    kh2 = out['K-H2_HQ_B2_minus_twin_new']
    assert kh2['estimable'] is False and kh2['rejected'] is False and kh2['tost']['reason'] == 'not estimable'
    assert out['readings']['K-H2']['direction'] == 'not estimable'
    assert out['K-H1_HQ_new_minus_Cmatched_CMLR']['holm_threshold'] == 0.025                 # denominator stays 2


def test_raw_event_check_on_the_real_precheck_runs(tmp_path):
    """The checker accepts real trainer output: config equality and exact AUC recomputation on the precheck runs."""
    (tmp_path / 'runs').symlink_to(HERE / 'precheck')
    for arm in ('quantum', 'surrogate625'):
        rec = json.loads((HERE / 'precheck' / f'{arm}.json').read_text())
        cfg = runner.config_main(arm, 0, runner.STORED_LR[arm], 3000, [0, 3000], None)
        assert an.canonical(rec['config']) == an.canonical(cfg)
        run = an.check_completed_run(tmp_path, f'PC_{arm}__b0', rec, json.loads(an.canonical(cfg)))
        assert run['updates'] == rec['summary']['updates'] == 6
        if arm == 'surrogate625':
            an.check_twin('PC_surrogate625__b0', rec)


def test_b2f_reader_uses_only_the_summary_line(tmp_path):
    d = tmp_path / 'F_quantum__b0' / 'run' / 'quantum'
    d.mkdir(parents=True)
    (d / 'metrics.jsonl').write_text('not json at all\n' + json.dumps({'status': 'completed', 'auc': {'sample': 0.5}}))
    d1 = tmp_path / 'F_quantum__b1' / 'run' / 'quantum'
    d1.mkdir(parents=True)
    (d1 / 'metrics.jsonl').write_text(json.dumps({'status': 'algorithmic_failure', 'auc': None}))
    assert an.load_b2f(tmp_path, blocks=(0, 1, 2)) == {0: {'auc': 0.5}}


def test_launch_refuses_a_plan_that_does_not_pin_the_code(tmp_path, monkeypatch):
    """Review B3: the launch needs the sha256 of runner.py and of b2k_analysis.py in PLAN.md (no run is started)."""
    def no_run(*a, **k):
        raise AssertionError('pool_run must not be reached in this test')
    started = []
    monkeypatch.setattr(runner, 'pool_run', lambda tasks, workers, write: started.extend(tasks))
    (tmp_path / 'precheck.json').write_text(json.dumps({'passed': True, 'runner_sha256': runner.RUNNER_SHA256}))
    (tmp_path / 'selection.json').write_text(json.dumps(SEL))
    lrs = ' '.join(SEL['arms'][a]['lr_str'] for a in runner.SETS.values())
    for text, ok in ((f'plan {lrs} {runner.RUNNER_SHA256}', False),
                     (f'plan {lrs} {runner.RUNNER_SHA256} {runner.sha256_file(runner.ANALYSIS)}', True)):
        (tmp_path / 'PLAN.md').write_text(text)
        sha = hashlib.sha256(text.encode()).hexdigest()
        (tmp_path / 'PLAN.sha256').write_text(sha + '  PLAN.md\n')
        monkeypatch.setattr(runner, 'HERE', tmp_path)
        monkeypatch.setattr(runner, 'PLAN', tmp_path / 'PLAN.md')
        monkeypatch.setattr(runner, 'PLAN_SHA256', sha)
        if ok:
            runner.launch(8)
            assert len(started) == 72 and [t['tag'] for t in started[:2]] == ['T1', 'T2']
            assert all(t['lr'] == LR_SEL[t['arm']] for t in started)
        else:
            monkeypatch.setattr(runner, 'pool_run', no_run)
            with pytest.raises(SystemExit, match='sha256'):
                runner.launch(8)
            monkeypatch.setattr(runner, 'pool_run', lambda tasks, workers, write: started.extend(tasks))
