"""Unit tests of B1R's statistics, analysis logic and reading rules, seed rule, comparison helper, readiness checks,
failure rules and resumable launch. No episode is run and no fresh-unit outcome exists or is computed here.

    PYTHONDONTWRITEBYTECODE=1 /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -m pytest -q \
        -p no:cacheprovider test_b1r.py
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import stats

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b1r_analysis as an                                                                         # noqa: E402
import runner                                                                                     # noqa: E402

RUN_SHA = 'f' * 64                                # the runner hash carried by the synthetic records


# -- statistics -------------------------------------------------------------------------------------------------------

def test_t_test_matches_scipy():
    d = np.random.default_rng(1).normal(-1e-5, 3e-5, 36)
    for alt in ('less', 'greater', 'two-sided'):
        assert abs(an.t_test(d, alt)['p'] - stats.ttest_1samp(d, 0, alternative=alt).pvalue) < 1e-12


def test_margin_test_is_the_shifted_one_sided_test():
    d = np.random.default_rng(2).normal(-0.6e-5, 4.8e-5, 36)
    ref = stats.ttest_1samp(d + an.DELTA_R3, 0, alternative='greater').pvalue
    assert abs(an.t_test(d, 'greater', -an.DELTA_R3)['p'] - ref) < 1e-12


def test_tost_is_max_of_two_one_sided():
    d = np.random.default_rng(3).normal(0., 2e-5, 36)
    r = an.tost(d, an.DELTA_R3)
    lo = stats.ttest_1samp(d + an.DELTA_R3, 0, alternative='greater').pvalue
    hi = stats.ttest_1samp(d - an.DELTA_R3, 0, alternative='less').pvalue
    assert abs(r['p'] - max(lo, hi)) < 1e-12


def test_nan_dropped_and_small_n():
    assert an.t_test([1e-5, np.nan, 2e-5, 3e-5, np.nan])['n'] == 3
    assert an.t_test([1e-5, np.nan])['p'] is None


def test_holm():
    assert an.holm([0.01, 0.04, 0.03]) == [True, False, False]
    assert an.holm([0.001, 0.02, 0.04]) == [True, True, True]
    assert an.holm([0.001, 0.06, 0.02]) == [True, False, True]
    assert an.holm([0.001, 0.03, 0.02]) == [True, True, True]
    assert an.holm([None, 0.001, 0.01]) == [False, True, True]
    assert an.holm([0.2, 0.001, 0.01]) == [False, True, True]
    assert an.holm([0.02, 0.03]) == [True, True]                      # family R: Holm over two
    assert an.holm([0.03, 0.02]) == [True, True]
    assert an.holm([0.026, 0.06]) == [False, False]


def test_fresh_units_agree_between_runner_and_analysis():
    assert runner.FRESH == an.FRESH == tuple(range(100, 136)) and runner.CANARY == 100


# -- the analysis end to end on synthetic records ---------------------------------------------------------------------

def _entry(v, status='ok'):
    return {'status': status, 'G_H': float(v), 'alpha': 1.0, 'kl_K': 0.01, 'L_F': float(v) * 2}


def _record(i, tan, hqf, ctrig, oc, ot, plan_sha, repro=0.0, runner_sha=RUN_SHA, e_tan_status='ok'):
    arms, opt = {}, {}
    for k in an.KAPPAS:
        s = 0.1 if k == '0.001' else 1.0
        arms.setdefault('HQ-Fisher', {})[k] = _entry(hqf * s)
        arms.setdefault('TAN', {})[k] = _entry(tan * s)
        arms.setdefault('QTAY', {})[k] = _entry(hqf * s)
        opt[k] = {'HQ-Fisher': _entry(hqf * s), 'TAN': _entry(tan * s, e_tan_status), 'HQ-plain': _entry(0.),
                  'TAN-c': _entry(tan * s),
                  'OPT-CIRC': {'feasible': True, 'G_H': oc * s, 'L_F': oc * s * 3, 'start': 'HQ-Fisher',
                               'iterations': 100, 'max_abs_angle_step': 3.0},
                  'OPT-TAN': {'feasible': True, 'G_H': ot * s, 'L_F': ot * s * 3, 'start': 'TAN', 'iterations': 90},
                  'reproduction_max_rel_diff': repro,
                  'first_crossing_HQ-Fisher': {'monotone_on_grid': True, 'earlier_crossing': False}}
        opt[k]['HQ-Fisher']['max_abs_angle_step'] = 40.0
    draws = []
    for j in range(20):
        d = {}
        for name in ('CTRIG', 'CTRIGx0.1', 'CTRIGx10', 'CQR'):
            d[name] = {k: _entry((ctrig + (j - 9.5) * 1e-7) * (0.1 if k == '0.001' else 1.0)) for k in an.KAPPAS}
        draws.append(d)
    arms['CTRIG_draws'] = draws
    return {'circuit': 'shipped', 'index': i, 'status': 'complete', 'plan_sha256': plan_sha, 'runner_sha256': runner_sha,
            'arms': arms, 'opt': {'arms': opt}}


def _setup(tmp_path, monkeypatch, gen, plan_text=None):
    (tmp_path / 'records').mkdir()
    (tmp_path / 'PLAN.md').write_text(plan_text if plan_text is not None else
                                      f'synthetic plan\nb1r_analysis.py {an.ANALYSIS_SHA256}\nrunner.py {RUN_SHA}\n')
    sha = hashlib.sha256((tmp_path / 'PLAN.md').read_bytes()).hexdigest()
    for i in an.FRESH:
        (tmp_path / 'records' / f'shipped_{i}.json').write_text(json.dumps(gen(i, sha)))
    monkeypatch.setattr(an, 'HERE', tmp_path)
    return sha


def test_all_primary_rejected(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        return _record(i, tan, tan - 6e-5 + rng.normal(0, 3e-5), tan - 4e-5 + rng.normal(0, 2e-5),
                       3e-5 + rng.normal(0, 2e-5), 3e-5 + rng.normal(0, 2e-5), sha)
    _setup(tmp_path, monkeypatch, gen)
    out = an.main(False)
    assert out['units_analysed'] == 36 and out['incomplete'] is False
    assert out['family_R']['holm_rejected'] == {'R1': True, 'R2': True}
    assert out['family_S']['holm_rejected']['R3'] is True
    rd = out['readings']
    assert rd['R1'].startswith('replicated') and rd['R2'].startswith('replicated')
    assert 'S1 not rejected is not equivalence' in rd['R2']
    assert rd['R3'].startswith('margin passed (secondary)')
    assert rd['joint_R1_R3'].startswith('the B1 shortfall replicates (R1); secondary')
    assert '71%' in rd['R3_scale'] and '29%' in rd['R3_scale'] and '0.0038' in rd['R3_scale']
    assert 'OPT-CIRC_minus_OPT-TAN_L_F' in out['0.01'] and out['0.01']['OPT-CIRC_minus_OPT-TAN_L_F']['n'] == 36
    assert (tmp_path / 'b1r_results.json').exists()


def test_no_headline_wording_for_r3():
    """R3 is never worded 'non-inferior' or 'replicated' (B1(c), m12), except the disclosed rule text itself."""
    p = {'R1_HQ-Fisher_minus_TAN': {'lo': -2, 'p': 0.001}, 'R2_CTRIG_minus_TAN': {'lo': -2, 'p': 0.001},
         'S1_CTRIG_minus_HQ-Fisher': {'mean': 1., 'p': 0.5}}
    for sign_p, mean in ((0.01, -1.), (0.01, 1.), (0.5, -1.)):
        p['R3_two_sided_difference'] = {'p': sign_p, 'mean': mean}
        for r3 in (True, False):
            r = an.readings(p, (True, True), (False, False, r3))
            assert 'non-inferior' not in r['R3'] and 'replicat' not in r['R3'] and 'non-inferior' not in r['joint_R1_R3']
            joint = r['joint_R1_R3'].replace('shortfall replicates (R1)', '').replace('did not replicate', '')
            assert 'replicat' not in joint                                       # 'replicate' refers to R1 only


def test_none_rejected(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        return _record(i, tan, tan + rng.normal(0, 3e-5), tan + rng.normal(0, 3e-5),
                       -2e-5 + rng.normal(0, 2e-5), 3e-5 + rng.normal(0, 2e-5), sha)
    _setup(tmp_path, monkeypatch, gen)
    out = an.main(False)
    assert out['family_R']['holm_rejected'] == {'R1': False, 'R2': False}
    assert out['family_S']['holm_rejected']['R3'] is False
    rd = out['readings']
    assert rd['R1'].startswith('not replicated') and rd['R2'].startswith('not replicated')
    assert rd['R3'].startswith('margin not passed') and 'negative at 0.05' in rd['R3']


def test_opposite_direction_contradicts_and_removes_the_path_claim(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        return _record(i, tan, tan + 3e-5 + rng.normal(0, 1e-5), tan + 3e-5 + rng.normal(0, 1e-5),
                       3e-5 + rng.normal(0, 2e-5), 3e-5 + rng.normal(0, 2e-5), sha)
    _setup(tmp_path, monkeypatch, gen)
    out = an.main(False)
    assert out['family_R']['holm_rejected'] == {'R1': False, 'R2': False}
    assert out['family_R']['opposite_direction'] == {'R1': True, 'R2': True}
    rd = out['readings']
    assert rd['R1'].startswith('contradicts B1') and 'removes the path claim' in rd['R1'] and 'not qualified' in rd['R1']
    assert rd['R2'].startswith('contradicts B1C') and 'removes the matched-chart path claim' in rd['R2']


def test_within_margin_yet_worse(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        ot = 3e-5 + rng.normal(0, 2e-5)
        return _record(i, tan, tan - 7e-5 + rng.normal(0, 3e-5), tan - 4e-5 + rng.normal(0, 2e-5),
                       ot - 1e-5 + rng.normal(0, 0.5e-5), ot, sha)
    _setup(tmp_path, monkeypatch, gen)
    out = an.main(False)
    assert out['family_S']['holm_rejected']['R3'] is True and out['family_S']['R3_two_sided_sign_at_0.05'] == -1
    assert out['readings']['R3'].startswith('within the margin, yet worse (secondary)')
    # rounds 2-3: with R1 rejected, the joint reading never places the shortfall in the path 'rather than' reachability
    assert out['family_R']['holm_rejected']['R1'] is True
    rule = out['joint_R1_R3_rule']
    assert rule['limit_minus_2_delta_x1e-5'] == pytest.approx(-5.70)
    assert rule['R1_mean_at_or_below_limit'] is True and rule['R1_mean_x1e-5'] <= -5.70   # fresh mean about -7e-5
    joint = out['readings']['joint_R1_R3']
    assert 'rather than' not in joint and 'reachability gap' not in joint
    assert 'gains significantly less' in joint and 'held-out gap smaller than delta is shown' in joint
    assert 'mainly, not wholly,' in joint and an.lf_clause(out['0.01']) in joint


def _joint_p(r1_mean, sign_p, sign_mean, lf=(-7.98, -9.34, -6.62)):
    """Synthetic family-R/S contrasts at kappa 0.01 (x 1e-5), as contrasts() lays them out."""
    return {'R1_HQ-Fisher_minus_TAN': {'lo': r1_mean - 2, 'p': 0.001, 'mean': r1_mean},
            'R2_CTRIG_minus_TAN': {'lo': -2, 'p': 0.001}, 'S1_CTRIG_minus_HQ-Fisher': {'mean': 1., 'p': 0.5},
            'R3_two_sided_difference': {'p': sign_p, 'mean': sign_mean},
            'OPT-CIRC_minus_OPT-TAN_L_F': {'mean': lf[0], 'lo': lf[1], 'hi': lf[2], 'n': 36}}


def test_joint_r1_r3_branches():
    """Round 3: with R1 and R3 rejected, 'mainly' is read only when B1R's own R1 mean is at or below -2 delta
    (-5.70e-5), 'not wholly' only when R3's two-sided difference is also negative at 0.05; otherwise the split is not
    settled. Never 'rather than'; the L_F difference and delta's anchor are in the same sentence."""
    for r1_mean in (-8.0, -5.70, -5.69, -3.6):
        half = r1_mean <= -5.70
        for sign_p, sign_mean in ((0.01, -1.), (0.01, 1.), (0.5, -1.), (0.5, 1.)):
            p = _joint_p(r1_mean, sign_p, sign_mean)
            worse = sign_p <= 0.05 and sign_mean < 0
            r = an.readings(p, (True, True), (False, False, True))
            joint = r['joint_R1_R3']
            assert an.at_most_half(p['R1_HQ-Fisher_minus_TAN']) is half
            assert joint.startswith('the B1 shortfall replicates (R1); secondary: on held-out gain')
            assert 'rather than' not in joint and 'reachability gap' not in joint
            assert an.lf_clause(p) in joint and '-7.98 [-9.34, -6.62]' in joint
            assert 'half of B1\'s natural-step shortfall on the original units (5.70e-5)' in joint
            assert '. ' not in joint                                         # one sentence: L_F is in it
            assert f'R1 mean {r1_mean:+.2f} x 1e-5' in joint
            if half:
                assert 'at or below -2 delta' in joint and 'not settled' not in joint
                assert ('mainly, not wholly, outside' in joint) is worse
                assert ('mainly outside' in joint) is (not worse)
                assert joint.endswith('reachability concerns L_F and is not tested')
            else:
                assert 'mainly' not in joint and 'above -2 delta' in joint
                assert joint.endswith('what circuit policies can reach is not settled')
            assert ('but gains significantly less, so a held-out gap smaller than delta is shown' in joint) is worse
            assert r['R3'].startswith('within the margin, yet worse (secondary)') is worse
    # the other joint branches do not depend on the sign or on the R1 mean
    for r1_mean in (-8.0, -3.6):
        p = _joint_p(r1_mean, 0.01, -1.)
        assert an.readings(p, (True, True), (False, False, False))['joint_R1_R3'] == \
            'the B1 shortfall replicates (R1); path against reachability is not settled'
        r3_only = an.readings(p, (False, False), (False, False, True))['joint_R1_R3']
        assert r3_only.startswith('the natural-step shortfall did not replicate; the secondary held-out margin test')
        assert an.lf_clause(p) in r3_only and 'mainly' not in r3_only
    # a missing R1 mean never takes the 'mainly' branch
    p = _joint_p(-8.0, 0.01, -1.)
    del p['R1_HQ-Fisher_minus_TAN']['mean']
    assert 'mainly' not in an.readings(p, (True, True), (False, False, True))['joint_R1_R3']


def test_r3_readings_are_held_out_and_carry_l_f():
    """Round 3: every R3 reading speaks of held-out gain, anchors delta to B1's original shortfall, carries the L_F
    difference in the same sentence, and never calls the G_H difference a reachability gap or limit."""
    for r3 in (True, False):
        for sign_p, sign_mean in ((0.01, -1.), (0.01, 1.), (0.5, -1.)):
            p = _joint_p(-6.0, sign_p, sign_mean, lf=(-1.11, -2.22, -0.01))
            text = an.readings(p, (True, True), (False, False, r3))['R3']
            assert 'held-out' in text and '(5.70e-5), not rescaled' in text
            assert an.lf_clause(p) in text and '-1.11 [-2.22, -0.01]' in text and '. ' not in text
            assert 'reachability gap' not in text and 'evidence of a reachability limit' not in text
            assert 'rather than' not in text and 'mainly' not in text
    p = _joint_p(-6.0, 0.5, -1.)
    p['OPT-CIRC_minus_OPT-TAN_L_F'] = {'n': 2, 'mean': None, 'p': None}       # too few units: stated, not omitted
    assert 'OPT-CIRC - OPT-TAN is not available (descriptive, n = 2)' in an.readings(p, (True, True),
                                                                                     (False, False, True))['R3']


def test_disclosed_original_unit_values():
    """The values PLAN.md and R3_SCALE disclose for the 36 original units (stored B1, B1C and B1E records; no fresh
    unit): B1's shortfall is 2 delta, and OPT-CIRC - OPT-TAN on L_F is -7.98 [-9.34, -6.62] x 1e-5, about 2.8 delta."""
    if not (an.B1_UNITS.exists() and an.B1C_RECORDS.exists() and an.B1E_RECORDS.exists()):
        pytest.skip('stored original-unit records not available')
    U = an.original_units('0.01')
    c = lambda a: an.col(U, (a, '0.01'))                                              # noqa: E731
    r1 = an.t_test(c('HQ-Fisher') - c('TAN'))
    lf = an.t_test(c('OPT-CIRC_L_F') - c('OPT-TAN_L_F'))
    assert round(r1['mean'], 2) == -5.70 and round(r1['mean'] / (an.DELTA_R3 * an.SCALE), 2) == -2.00
    assert lf['n'] == 36 and [round(lf[k], 2) for k in ('mean', 'lo', 'hi')] == [-7.98, -9.34, -6.62]
    assert round(lf['mean'] / (an.DELTA_R3 * an.SCALE), 1) == -2.8
    assert '-7.98e-5' in an.R3_SCALE and '[-9.34, -6.62]' in an.R3_SCALE and '2.8 delta' in an.R3_SCALE


def test_r2_without_r1_is_conditional(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        return _record(i, tan, tan + rng.normal(0, 3e-5), tan - 4e-5 + rng.normal(0, 2e-5),
                       3e-5 + rng.normal(0, 2e-5), 3e-5 + rng.normal(0, 2e-5), sha)
    _setup(tmp_path, monkeypatch, gen)
    out = an.main(False)
    assert out['family_R']['holm_rejected'] == {'R1': False, 'R2': True}
    r2 = out['readings']['R2']
    assert r2.startswith('shown for the matched chart only (conditional on R1)')
    assert 'did not replicate' in r2 and 'generic reading is not confirmed' in r2 and 'not equivalence' in r2
    assert out['readings']['generic_vs_specific'].startswith('conditional')


def test_incomplete_flag_below_30_units(tmp_path, monkeypatch):
    rng = np.random.default_rng(10)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        r = _record(i, tan, tan - 6e-5, tan - 4e-5 + rng.normal(0, 1e-5), 3e-5, 3e-5 + rng.normal(0, 1e-5), sha)
        if i >= 129:                                                       # 7 failed units: 29 left
            r.update(status='failure', failure_class='episode_failure', error='BatchError: infrastructure')
        return r
    _setup(tmp_path, monkeypatch, gen)
    out = an.main(False)
    assert out['units_analysed'] == 29 and out['incomplete'] is True
    assert 'incomplete' in out['incomplete_statement'] and '29 units' in out['incomplete_statement']
    assert out['readings']['incomplete'] == out['incomplete_statement']
    assert len(out['failed']) == 7


def test_refuses_partial_and_excludes_inconsistent(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)

    def gen(i, sha):
        tan = 4e-5 + rng.normal(0, 2e-5)
        return _record(i, tan, tan - 6e-5, tan - 4e-5, 3e-5, 3e-5, sha, repro=(1e-3 if i == 100 else 0.0),
                       e_tan_status=('no_bracket' if i == 101 else 'ok'))
    _setup(tmp_path, monkeypatch, gen)
    (tmp_path / 'records' / 'shipped_135.json').unlink()
    with pytest.raises(SystemExit):
        an.main(False)
    out = an.main(True)
    assert out['missing'] == [135] and out['excluded_internal_consistency'] == [100, 101] and out['units_analysed'] == 33
    assert 'status' in out['excluded_internal_consistency_reasons'][101][0]          # m7: E's status differs from C's
    assert (tmp_path / 'b1r_results_provisional.json').exists() and not (tmp_path / 'b1r_results.json').exists()


def test_refuses_records_of_another_plan(tmp_path, monkeypatch):
    def gen(i, sha):
        return _record(i, 4e-5, -2e-5, 0., 3e-5, 3e-5, 'another plan')
    _setup(tmp_path, monkeypatch, gen)
    with pytest.raises(SystemExit):
        an.main(False)


def test_refuses_mixed_or_unrecorded_runner_hashes(tmp_path, monkeypatch):
    def gen(i, sha):
        return _record(i, 4e-5, -2e-5, 0., 3e-5, 3e-5, sha, runner_sha=(RUN_SHA if i != 120 else 'e' * 64))
    _setup(tmp_path, monkeypatch, gen)
    with pytest.raises(SystemExit, match='one runner'):
        an.main(False)


def test_refuses_unrecorded_analysis(tmp_path, monkeypatch):
    def gen(i, sha):
        return _record(i, 4e-5, -2e-5, 0., 3e-5, 3e-5, sha)
    _setup(tmp_path, monkeypatch, gen, plan_text=f'plan without the analysis hash\nrunner.py {RUN_SHA}\n')
    with pytest.raises(SystemExit, match='b1r_analysis.py'):
        an.main(False)
    (tmp_path / 'PLAN_LOG.md').write_text(f'- deviation: b1r_analysis.py {an.ANALYSIS_SHA256}\n')
    assert an.main(False)['analysis_recorded'] is True


# -- the runner's seed rule and helpers (no episode) -------------------------------------------------------------------

def test_seed_rule_equals_b1_for_b1_units_and_fresh_is_disjoint():
    b1 = runner.b1
    for s in runner.SETS:
        ns = b1.SET_NAMESPACE[s]
        for i in (0, 17, 35):
            for c in (0, 7):
                for o in (0, 31):
                    assert runner.episode_seed(ns, i, c, o, s) == b1.episode_seed(ns, 'shipped', i, c, o, s)
                    assert runner.action_seed(s, i, c, o) == runner.dq.DQ_STUDY.stream(
                        runner.db.SET_TAG[s], b1.stream_unit('shipped', i), c, o)
        start = runner.dq.DQ_NAMESPACES[ns]['start']
        fresh = [runner.episode_seed(ns, i, c, o, s) - start for i in runner.FRESH for c in range(8) for o in range(32)]
        assert min(fresh) == 25_600 and max(fresh) == 34_815 and len(set(fresh)) == 36 * 256
        assert min(fresh) >= 72 * 8 * 32                                   # above every B1 seed (both circuits)
        assert max(fresh) < 50_000                                         # below extension N's extra fit seeds
    for bad in (36, 71, 99, 136):
        with pytest.raises(ValueError):
            runner.check_index(bad)


def test_compare_helper():
    old = {'a': {'G_H': 1.0, 'status': 'ok', 'l': [1, 2.0]}, 'b': 2.0}
    assert runner.ok(runner.compare({'a': {'G_H': 1.0, 'status': 'ok', 'l': [1, 2.0]}, 'b': 2.0, 'c': 9}, old))
    c = runner.compare({'a': {'G_H': 1.0 + 1e-7, 'status': 'ok', 'l': [1, 2.0]}}, old)
    assert c['mismatch'] and c['missing'] == ['/b']
    assert runner.ok(runner.compare({'a': {'G_H': 1.0 + 1e-7, 'status': 'ok', 'l': [1, 2.0]}, 'b': 2.0}, old, tol=1e-6))
    assert runner.compare({'a': {'G_H': 1.0, 'status': 'bad', 'l': [1, 2.0]}, 'b': 2.0}, old)['mismatch']


def test_compare_nan_against_a_number_is_a_mismatch():                             # m1
    nan = float('nan')
    assert runner.compare({'G_H': 1.0}, {'G_H': nan})['mismatch']
    assert runner.compare({'G_H': nan}, {'G_H': 1.0})['mismatch']
    assert runner.ok(runner.compare({'G_H': nan}, {'G_H': nan}))
    assert runner.compare({'G_H': math.inf}, {'G_H': 1.0})['mismatch']


def test_failure_classes():
    assert runner.failure_class(MemoryError()) == 'environmental'
    assert runner.failure_class(OSError(28, 'No space left on device')) == 'environmental'
    assert runner.failure_class(runner.db.BatchError('x')) == 'episode_failure'
    assert runner.failure_class(runner.tp.TransplantParityError('x')) == 'parity_refused'
    assert runner.failure_class(runner.dq.DQError('x')) == 'refused'
    assert runner.failure_class(RuntimeError('x')) == 'algorithmic_failure'


def _rows_not_saved(reason):
    """The exception run_unit raises when extension C's save_rows returns {'saved': False, 'reason': ...}."""
    return RuntimeError(f'rows not saved: {{\'saved\': False, \'reason\': {reason!r}}}')


def test_environmental_by_error_text():                                               # round 3, failure rule 3
    """Memory exhaustion or a failed write is environmental wherever it was raised, also when torch reports it as a
    RuntimeError or extension C / E pass on only its text (the messages are torch 2.14's, verified)."""
    env = [RuntimeError('[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can\'t allocate memory: you '
                        'tried to allocate 1152921504606846976 bytes. Error code 12 (Cannot allocate memory)'),
           _rows_not_saved('RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 704 vs 598'),
           _rows_not_saved('RuntimeError: basic_ios::clear: iostream error'),
           _rows_not_saved('RuntimeError: PytorchStreamWriter failed writing file data/0: file write failed'),
           _rows_not_saved('MemoryError: '),
           RuntimeError('extension E arms failed: MemoryError: '),
           RuntimeError('extension E arms failed: OSError: [Errno 28] No space left on device'),
           RuntimeError('extension E arms failed: OSError: [Errno 122] Disk quota exceeded'),
           ValueError('out of memory in a solver')]
    for exc in env:
        assert runner.failure_class(exc) == 'environmental', exc
        assert runner.environmental_match(exc).startswith('text ')
    assert runner.environmental_match(MemoryError()) == 'type MemoryError'
    assert runner.environmental_match(FileNotFoundError('x')) == 'type FileNotFoundError'
    not_env = [_rows_not_saved('fit: rebuilt features differ'), RuntimeError('extension E arms failed: ValueError: x'),
               RuntimeError('extension E arms failed: KeyError: \'OPT-CIRC\''), RuntimeError('unexpected position'),
               RuntimeError('MemoryErrors'), RuntimeError('PytorchStreamReader failed reading zip archive')]
    for exc in not_env:
        assert runner.failure_class(exc) == 'algorithmic_failure', exc
        assert runner.environmental_match(exc) is None
    assert runner.failure_class(runner.db.BatchError('infrastructure/invalid episode fit c0 o0: '
                                                     'infrastructure_failure')) == 'episode_failure'
    assert runner.failure_class(runner.tp.TransplantParityError('P4: decision 3 differs')) == 'parity_refused'
    assert set(runner.ENVIRONMENTAL_NAMES) >= {'MemoryError', 'OSError', 'FileNotFoundError', 'PermissionError'}


def test_environmental_real_torch_failures():
    """The real torch failures behind the verifier's cases (i) and (ii): a torch.save on a full disk (/dev/full) and
    a failed CPU allocation are RuntimeErrors, classed environmental raw and as run_unit re-raises them."""
    torch = runner.torch
    cases = [lambda: torch.empty(2 ** 60, dtype=torch.uint8)]
    if Path('/dev/full').exists():
        cases += [lambda: torch.save({'a': torch.zeros(10)}, '/dev/full'),
                  lambda: torch.save({'a': torch.zeros(10 ** 6, dtype=torch.float64)}, '/dev/full')]
    for make in cases:
        with pytest.raises(RuntimeError) as info:
            make()
        assert runner.failure_class(info.value) == 'environmental'
        assert runner.failure_class(_rows_not_saved(f'{type(info.value).__name__}: {info.value}')) == 'environmental'


# -- readiness, canary, failure rules and resumable launch (fake unit tasks; no episode) --------------------------------

@pytest.fixture(autouse=True)
def far_cutoff(monkeypatch):
    monkeypatch.setattr(runner.dq, 'CALENDAR_CUTOFF_UTC', '2099-01-01T00:00:00+00:00')


def _ok_task(args):
    index, stage_root = args
    return {'index': index, 'status': 'complete', 'wall_s': 0.0, 'stage_root': stage_root,
            'runner_sha256': runner.RUNNER_SHA256}


def _fail_task(args):
    index, stage_root = args
    return {'index': index, 'status': 'failure', 'wall_s': 0.0, 'failure_class': 'algorithmic_failure',
            'error_type': 'RuntimeError', 'error': 'RuntimeError: synthetic'}


def _env_fail_task(args):
    index, stage_root = args
    return {'index': index, 'status': 'failure', 'wall_s': 0.0, 'failure_class': 'environmental',
            'error_type': 'MemoryError', 'error': 'MemoryError: synthetic'}


def _env_runtime_fail_task(args):
    """An out-of-memory failure as torch reports it (RuntimeError), classed by its text (round 3)."""
    index, stage_root = args
    exc = RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 8 bytes")
    return {'index': index, 'status': 'failure', 'wall_s': 0.0, 'failure_class': runner.failure_class(exc),
            'environmental_match': runner.environmental_match(exc), 'error_type': 'RuntimeError',
            'error': f'RuntimeError: {exc}'}


def _batch_fail_task(args):
    index, stage_root = args
    return {'index': index, 'status': 'failure', 'wall_s': 0.0, 'failure_class': 'episode_failure',
            'error_type': 'BatchError', 'error': 'BatchError: infrastructure/invalid episode'}


def _unserialisable_task(args):
    index, stage_root = args
    return {'index': index, 'status': 'complete', 'wall_s': 0.0, 'bad': object()}


def _precheck(tmp_path, name='precheck', kind='precheck of record', passed=True, runner_sha=None, tree=None,
              plan_sha=None):
    plan_sha = plan_sha or hashlib.sha256((tmp_path / 'PLAN.md').read_bytes()).hexdigest()
    (tmp_path / f'{name}.json').write_text(json.dumps({
        'kind': kind, 'passed': passed, 'plan_sha256': plan_sha, 'runner_sha256': runner_sha or runner.RUNNER_SHA256,
        'reused_sha256': {str(p): h for p, h in runner.REUSED.items()}, 'b1_tree_live_copy': tree or runner.B1_TREE}))


def _ready_dir(tmp_path, kind='precheck of record', passed=True, plan_text=None):
    (tmp_path / 'PLAN.md').write_text(plan_text if plan_text is not None else f'plan\nrunner.py {runner.RUNNER_SHA256}\n')
    sha = hashlib.sha256((tmp_path / 'PLAN.md').read_bytes()).hexdigest()
    (tmp_path / 'PLAN.sha256').write_text(f'{sha}  PLAN.md\n')
    _precheck(tmp_path, kind=kind, passed=passed)
    return sha


def _write(tmp_path, index, rec):
    (tmp_path / 'records').mkdir(exist_ok=True)
    (tmp_path / 'records' / f'shipped_{index:02d}.json').write_text(json.dumps({'index': index, **rec}))


def _last_launch(tmp_path):
    return json.loads((tmp_path / 'launches.jsonl').read_text().splitlines()[-1])


def test_launch_skips_done_moves_interrupted_and_never_overwrites(tmp_path):
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete', 'kept': True})
    (tmp_path / 'stage' / 'rows').mkdir(parents=True)
    (tmp_path / 'stage' / 'rows' / 'shipped_101.pt').write_bytes(b'partial')
    (tmp_path / 'stage' / 'started').mkdir(parents=True)
    (tmp_path / 'stage' / 'started' / 'shipped_101.json').write_text('{"index": 101, "pid": 1}')
    runner.launch(1, root=tmp_path, task=_ok_task, indices=(100, 101, 102))
    assert json.loads((tmp_path / 'records' / 'shipped_100.json').read_text())['kept'] is True
    assert json.loads((tmp_path / 'records' / 'shipped_101.json').read_text())['status'] == 'complete'
    assert (tmp_path / 'records' / 'shipped_102.json').exists()
    moved = list((tmp_path / 'interrupted').glob('*/rows_shipped_101.pt'))
    assert len(moved) == 1 and moved[0].read_bytes() == b'partial'
    assert list((tmp_path / 'interrupted').glob('*/started_shipped_101.json'))
    assert _last_launch(tmp_path)['tasks'] == [101, 102]
    assert list((tmp_path / 'launch_snapshots').glob('*_runner.py.txt'))
    assert not (tmp_path / 'launch.lock').exists()
    runner.launch(1, root=tmp_path, task=_ok_task, indices=(100, 101, 102))      # nothing left to do
    assert _last_launch(tmp_path)['tasks'] == []


@pytest.mark.parametrize('kind,passed', [('development check (not the precheck of record)', True),
                                         ('precheck of record', False)])
def test_launch_refuses_without_precheck_of_record(tmp_path, kind, passed):
    _ready_dir(tmp_path, kind, passed)
    with pytest.raises(SystemExit):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=(100,))


def test_launch_refuses_unhashed_or_changed_plan(tmp_path):
    _ready_dir(tmp_path)
    (tmp_path / 'PLAN.md').write_text('plan, edited after hashing\n')
    with pytest.raises(SystemExit):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=(100,))
    (tmp_path / 'PLAN.sha256').unlink()
    with pytest.raises(SystemExit):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=(100,))


def test_require_ready_checks_runner_tree_and_the_latest_precheck(tmp_path, monkeypatch):     # m3, rule 4
    _ready_dir(tmp_path)
    assert runner.require_ready(tmp_path)['precheck'] == 'precheck.json'
    _precheck(tmp_path, runner_sha='0' * 64)                                    # precheck run by another runner
    with pytest.raises(SystemExit, match='not by this runner'):
        runner.require_ready(tmp_path)
    _precheck(tmp_path, tree='1' * 64)                                          # another B1 tree
    with pytest.raises(SystemExit, match='B1 tree'):
        runner.require_ready(tmp_path)
    _precheck(tmp_path, passed=False)                                           # failed precheck of record: no launch
    with pytest.raises(SystemExit, match='did not pass'):
        runner.require_ready(tmp_path)
    (tmp_path / 'precheck_2').mkdir()                                           # its rerun is still running
    with pytest.raises(SystemExit, match='running or was interrupted'):
        runner.require_ready(tmp_path)
    _precheck(tmp_path, name='precheck_2')                                      # the rerun passed
    assert runner.require_ready(tmp_path)['precheck'] == 'precheck_2.json'
    assert runner.next_precheck_name(tmp_path) == 'precheck_3'
    _precheck(tmp_path, name='precheck', passed=True)
    _precheck(tmp_path, name='precheck_3', passed=False)                        # the latest one counts
    with pytest.raises(SystemExit, match='precheck_3.json'):
        runner.require_ready(tmp_path)
    monkeypatch.setattr(runner.dq, 'CALENDAR_CUTOFF_UTC', '2000-01-01T00:00:00+00:00')
    _precheck(tmp_path, name='precheck_4')
    with pytest.raises(SystemExit, match='cut-off'):
        runner.require_ready(tmp_path)


def test_require_ready_needs_the_runner_hash_in_plan_or_log(tmp_path):
    _ready_dir(tmp_path, plan_text='plan without the runner hash\n')
    with pytest.raises(SystemExit, match='recorded neither'):
        runner.require_ready(tmp_path)
    (tmp_path / 'PLAN_LOG.md').write_text(f'- deviation: runner.py now {runner.RUNNER_SHA256}\n')
    assert runner.require_ready(tmp_path)


def test_canary_runs_first_and_alone(tmp_path):                                    # rule 1
    _ready_dir(tmp_path)
    with pytest.raises(SystemExit, match='canary'):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:3])
    runner.launch(1, root=tmp_path, task=_ok_task, canary=True)
    assert sorted(p.name for p in (tmp_path / 'records').glob('*.json')) == ['shipped_100.json']
    assert _last_launch(tmp_path)['canary'] is True and _last_launch(tmp_path)['tasks'] == [100]
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:3])
    assert _last_launch(tmp_path)['tasks'] == [101, 102]


def test_failed_canary_blocks_the_rest_until_logged(tmp_path):
    _ready_dir(tmp_path)
    runner.launch(1, root=tmp_path, task=_fail_task, canary=True)
    with pytest.raises(SystemExit, match='canary failed'):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:3])
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:3], canary_failure_logged=True)
    assert json.loads((tmp_path / 'records' / 'shipped_100.json').read_text())['status'] == 'failure'
    assert json.loads((tmp_path / 'records' / 'shipped_102.json').read_text())['status'] == 'complete'


def test_two_failures_of_one_type_stop_the_launch(tmp_path):                        # rule 2
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    with pytest.raises(SystemExit, match='failure rule 2'):
        runner.launch(1, root=tmp_path, task=_fail_task, indices=runner.FRESH[:6])
    recs = runner.read_records(tmp_path)
    assert sorted(i for i, r in recs.items() if r['status'] == 'failure') == [101, 102]
    assert 'stopped' in _last_launch(tmp_path)
    with pytest.raises(SystemExit, match='share a failure type'):             # no relaunch before the diagnosis
        runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:6])
    dest = runner.supersede(tmp_path)                                         # full rerun from scratch
    assert (dest / 'records' / 'shipped_101.json').exists() and not (tmp_path / 'records').exists()
    with pytest.raises(SystemExit, match='canary'):                           # the canary runs first again
        runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:6])


def test_diagnosed_unit_level_failures_may_continue(tmp_path):
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    for i in (101, 102):
        _write(tmp_path, i, {'status': 'failure', 'error_type': 'BatchError', 'failure_class': 'episode_failure'})
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:5], failures_diagnosed=True)
    assert _last_launch(tmp_path)['failures_acknowledged'] == ['BatchError']
    assert runner.read_records(tmp_path)[104]['status'] == 'complete'


def test_environmental_failure_rerun_once(tmp_path):                                # rule 3
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    _write(tmp_path, 101, {'status': 'failure', 'error_type': 'MemoryError', 'failure_class': 'environmental'})
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:2])          # not rerun without the option
    assert runner.read_records(tmp_path)[101]['status'] == 'failure'
    runner.launch(1, root=tmp_path, task=_env_fail_task, indices=runner.FRESH[:2], rerun_environmental=True)
    assert _last_launch(tmp_path)['environmental_rerun'] == [101]
    assert len(list((tmp_path / 'failed_environmental').glob('*/shipped_101.json'))) == 1
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:2], rerun_environmental=True)
    assert _last_launch(tmp_path)['environmental_kept_failed'] == [101]               # once only
    assert runner.read_records(tmp_path)[101]['status'] == 'failure'


def test_environmental_runtime_errors_stop_then_rerun_once(tmp_path):               # round 3, rules 2 and 3
    """Two out-of-memory RuntimeErrors stop the launch (rule 2); after the diagnosis is logged, the launch continues
    with --failures-diagnosed --rerun-environmental and reruns both from scratch, once."""
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    with pytest.raises(SystemExit, match='failure rule 2'):
        runner.launch(1, root=tmp_path, task=_env_runtime_fail_task, indices=runner.FRESH[:4])
    recs = runner.read_records(tmp_path)
    assert {i: r['failure_class'] for i, r in recs.items() if r['status'] != 'complete'} == \
        {101: 'environmental', 102: 'environmental'}
    assert recs[101]['environmental_match'] == "text \"can't allocate memory\""
    with pytest.raises(SystemExit, match='share a failure type'):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:4], rerun_environmental=True)
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:4], failures_diagnosed=True,
                  rerun_environmental=True)
    last = _last_launch(tmp_path)
    assert last['environmental_rerun'] == [101, 102] and last['tasks'] == [101, 102, 103]
    assert last['failures_acknowledged_units'] == {'RuntimeError': [101, 102]}
    assert all(r['status'] == 'complete' for r in runner.read_records(tmp_path).values())
    assert len(list((tmp_path / 'failed_environmental').glob('*/shipped_10[12].json'))) == 2


def test_a_diagnosis_covers_only_the_units_it_read(tmp_path):                       # round 3, rule 2
    """--failures-diagnosed acknowledges the failed units it was given; a later failure of the same type is not
    diagnosed and stops the launch again."""
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    for i in (101, 102):
        _write(tmp_path, i, {'status': 'failure', 'error_type': 'BatchError', 'failure_class': 'episode_failure'})
    with pytest.raises(SystemExit, match='failure rule 2: units \\[101, 102, 103\\]'):
        runner.launch(1, root=tmp_path, task=_batch_fail_task, indices=runner.FRESH[:6], failures_diagnosed=True)
    assert sorted(runner.read_records(tmp_path)) == [100, 101, 102, 103]


def test_record_is_serialised_before_writing(tmp_path):                             # m4
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    runner.launch(1, root=tmp_path, task=_unserialisable_task, indices=runner.FRESH[:2])
    r = json.loads((tmp_path / 'records' / 'shipped_101.json').read_text())
    assert r['status'] == 'failure' and 'not serialisable' in r['error'] and r['original_status'] == 'complete'
    assert not list((tmp_path / 'records').glob('.*.tmp'))


def test_a_second_launch_is_refused_while_one_runs(tmp_path):
    _ready_dir(tmp_path)
    _write(tmp_path, 100, {'status': 'complete'})
    (tmp_path / 'launch.lock').write_text(json.dumps({'pid': os.getpid(), 'utc': 'now'}))
    with pytest.raises(SystemExit, match='another launch'):
        runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:2])
    (tmp_path / 'launch.lock').write_text(json.dumps({'pid': 999_999_999, 'utc': 'then'}))     # stale: moved aside
    runner.launch(1, root=tmp_path, task=_ok_task, indices=runner.FRESH[:2])
    assert list((tmp_path / 'interrupted').glob('*/launch.lock')) and not (tmp_path / 'launch.lock').exists()


def test_status_shows_no_arm_value(tmp_path):
    rec = _record(100, 4e-5, -2e-5, 0., 3e-5, 3e-5, 'plan')
    rec.update(certificate={'P1_solve_residual': 1e-14, 'static_failures': []}, timing={'episodes_fit': 256})
    _write(tmp_path, 100, rec)
    (tmp_path / 'stage' / 'started').mkdir(parents=True)
    (tmp_path / 'stage' / 'started' / 'shipped_101.json').write_text(json.dumps(
        {'index': 101, 'pid': os.getpid(), 'utc_start': runner.utc()}))
    s = runner.status(tmp_path)
    assert s['complete'] == 1 and s['started_without_record'][0]['index'] == 101
    assert s['started_without_record'][0]['pid_alive'] is True
    text = json.dumps(s, default=float)
    assert 'G_H' not in text and 'L_F' not in text and 'CTRIG' not in text
