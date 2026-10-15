"""Freeze B1 analysis: the two-test primary family, the interpretation gate, stratified and secondary reports.

    .venv/bin/python scripts/gnss_qrl_runtime/dq_b1_analysis.py --campaign <dir> --out <new dir>

Frozen by review 36. At kappa = 1e-2, units are fresh initialisations; every contrast is paired within unit.

PRIMARY (shipped circuit, 36 units; Holm over the two, familywise alpha 0.05)
    M-H1  Y_u = G_H(SPAN)_u - G_H(HQ-Fisher)_u;              one-sided paired t, H1: mean > 0
          (the direction was selected on development material, D-6, before any validation unit existed)
    A-H1  Delta_u = G_H(HQ-Fisher)_u - mean_r G_H(RND-k#r)_u; two-sided paired t; the 20 draws are averaged inside
          the unit and never counted as units; rank_ambiguous units are excluded
INTERPRETATION GATE: the one-sided 95% t lower bound of mean_u G_H(SPAN)_u (shipped) must exceed 0. If it does not,
    the primary estimates and tests are still reported, and the only permitted conclusion is "insensitive at the
    tested KL and batch scale".
HELD-OUT CIRCUIT (36 units): the same estimates, intervals and tests, reported SEPARATELY (never pooled into the
    shipped tests). A cross-circuit statement is permitted only if the held-out M-H1 mean is > 0 AND its one-sided
    95% lower bound is > 0.
EXCLUSIONS: a unit counts for a test only if its status is complete and every calibration the test reads is 'ok'
    (A-H1: HQ-Fisher and all 20 RND-k draws; M-H1: SPAN and HQ-Fisher; gate: SPAN); A-H1 also drops rank_ambiguous
    units. If more than 10% of a stratum's 36 units are excluded from a test, that test is reported NOT ESTIMABLE (and
    counts as not rejected in Holm).
SECONDARY (per stratum; never in the confirmatory family; a null is not evidence against the geometric account)
    P-H1  Spearman(D_u, Y_u), 95% percentile bootstrap over units (10,000, frozen stream); stratified statistic = mean
          of the two within-circuit Spearmans, one-sided within-stratum permutation test (10,000).
    P-H1b Spearman(X_u, Y_u) for X in {grad_norm, fisher_condition_tangent, entropy}; paired bootstrap of
          rho(D) - rho(X).
    Reliability: split-half Fisher-metric cosine distribution; jackknife SEs of the held-out gains; the distribution
          of Y_snr = Y / SE(Y); the uncorrected Spearman with its interval. No attenuation correction.
    E (fresh episodes, 12 + 12 units): Delta J per E arm with a 95% t interval; secondary, never changes a primary
          decision.
DESCRIPTIVE: every arm at every kappa (5e-2 labelled stress); TAN's percentile among the RND-k draws; ORTH, FIT-1,
    FIT-k, RND-48/192, RND-spec-plain, HQ-plain, HQ-Adam-dir gains; failures by status.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_runtime as dq                                                                           # noqa: E402

TOOL = 'gnss-dq-b1-analysis/1'
ALPHA = 0.05
UNITS_PER_STRATUM = 36
MAX_EXCLUDED_FRACTION = 0.10
KAPPA = '0.01'
RND_DRAWS = 20
BOOTSTRAP, PERMUTATIONS = 10_000, 10_000
BOOT_TAG, PERM_TAG = 60, 61
PREDICTORS = ('grad_norm', 'fisher_condition_tangent', 'entropy')
E_ARMS = ('HQ-Fisher', 'RND-k#0', 'FIT-k', 'SPAN')
PERMITTED = {
    'insensitive': 'insensitive at the tested KL and batch scale',
}


def _gain(unit, arm, kappa=KAPPA):
    entry = unit['instruments']['arms'].get(arm, {}).get(kappa)
    if entry is None or entry['status'] != 'ok':
        return None
    return entry['G_H']


def complete(units):
    return [u for u in units if u.get('status') == 'complete']


def m_values(units):
    out = {}
    for u in complete(units):
        a, b = _gain(u, 'SPAN'), _gain(u, 'HQ-Fisher')
        if a is not None and b is not None:
            out[u['index']] = a - b
    return out


def a_values(units):
    out = {}
    for u in complete(units):
        if u['instruments']['rank_ambiguous']:
            continue
        hq = _gain(u, 'HQ-Fisher')
        rnd = [_gain(u, f'RND-k#{r}') for r in range(RND_DRAWS)]
        if hq is not None and all(x is not None for x in rnd):
            out[u['index']] = hq - float(np.mean(rnd))
    return out


def gate_values(units):
    return {u['index']: _gain(u, 'SPAN') for u in complete(units) if _gain(u, 'SPAN') is not None}


def paired_test(values, alternative):
    """t test on unit values: mean, sd, 95% two-sided CI, one-sided 95% lower bound, standardised effect, p."""
    x = np.asarray(list(values.values()), dtype=float)
    n = x.size
    excluded = UNITS_PER_STRATUM - n
    record = {'n': int(n), 'excluded': int(excluded), 'unit_values': {str(k): v for k, v in sorted(values.items())}}
    if excluded > MAX_EXCLUDED_FRACTION * UNITS_PER_STRATUM or n < 3:
        return dict(record, estimable=False, p=None)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else math.copysign(math.inf, mean) if mean else 0.
    q975, q95 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    p = {'greater': stats.t.sf(t, n - 1), 'two-sided': 2 * stats.t.sf(abs(t), n - 1)}[alternative]
    return dict(record, estimable=True, alternative=alternative, mean=mean, sd=sd, se=se, t=float(t), p=float(p),
                ci95=[mean - q975 * se, mean + q975 * se], lower95_one_sided=mean - q95 * se,
                standardised_effect=mean / sd if sd > 0 else None)


def holm(tests):
    """tests: {name: record}; not-estimable tests are not rejected. Returns {name: {'rejected', 'holm_threshold'}}."""
    ordered = sorted(((r['p'], name) for name, r in tests.items() if r.get('estimable')), key=lambda x: x[0])
    m = len(tests)
    decisions = {name: {'rejected': False, 'holm_threshold': None} for name in tests}
    for rank, (p, name) in enumerate(ordered):
        threshold = ALPHA / (m - rank)
        decisions[name]['holm_threshold'] = threshold
        if p <= threshold:
            decisions[name]['rejected'] = True
        else:
            break
    return decisions


def stratum_primary(units):
    m = paired_test(m_values(units), 'greater')
    a = paired_test(a_values(units), 'two-sided')
    g = paired_test(gate_values(units), 'greater')
    return m, a, g


def spearman(x, y):
    if len(x) < 3:
        return None
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_ci(x, y, seed, fn=spearman):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    stats_ = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        v = fn(x[idx], y[idx])
        if v is not None and not math.isnan(v):
            stats_.append(v)
    return [float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))] if stats_ else None


def secondary(units, stratum_tag):
    rows = [(u['instruments']['diagnostics'], u['instruments']['reliability']) for u in complete(units)
            if u['instruments']['reliability'].get('Y') is not None]
    D = [d['D'] for d, r in rows]
    Y = [r['Y'] for d, r in rows]
    out = {'n': len(rows), 'P_H1': {'spearman': spearman(D, Y),
                                    'ci95': bootstrap_ci(D, Y, dq.DQ_STUDY.stream(BOOT_TAG, stratum_tag, 0))}}
    out['P_H1b'] = {}
    for j, name in enumerate(PREDICTORS, start=1):
        X = [d[name] for d, r in rows]
        rng = np.random.default_rng(dq.DQ_STUDY.stream(BOOT_TAG, stratum_tag, j))
        diffs = []
        for _ in range(BOOTSTRAP):
            idx = rng.integers(0, len(rows), len(rows))
            a, b = spearman(np.asarray(D)[idx], np.asarray(Y)[idx]), spearman(np.asarray(X)[idx], np.asarray(Y)[idx])
            if a is not None and b is not None and not (math.isnan(a) or math.isnan(b)):
                diffs.append(a - b)
        out['P_H1b'][name] = {'spearman': spearman(X, Y),
                              'rho_D_minus_rho_X_ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
                              if diffs else None}
    rel = [u['instruments']['reliability'] for u in complete(units)]
    split = [r['split_half_cosine'] for r in rel]
    snr = [r['Y_snr'] for r in rel if r.get('Y_snr') is not None]
    ses = {arm: [u['instruments']['arms'][arm][KAPPA].get('G_H_jackknife_se') for u in complete(units)
                 if u['instruments']['arms'].get(arm, {}).get(KAPPA, {}).get('status') == 'ok']
           for arm in ('SPAN', 'HQ-Fisher', 'RND-k#0', 'FIT-k')}
    q = lambda v: {p: float(np.percentile(v, p)) for p in (0, 25, 50, 75, 100)} if v else None     # noqa: E731
    out['reliability'] = {'split_half_cosine': q(split), 'Y_snr': q(snr), 'jackknife_se': {a: q(v) for a, v in ses.items()},
                          'uncorrected_spearman': out['P_H1'], 'attenuation_correction': 'none (review 36)'}
    return out, D, Y


def stratified_permutation(pairs, seed):
    """pairs: [(D, Y) per stratum]; statistic = mean within-stratum Spearman; one-sided (greater) p by within-stratum
    permutation of Y."""
    observed = np.mean([spearman(d, y) for d, y in pairs])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(PERMUTATIONS):
        s = np.mean([spearman(d, rng.permutation(y)) for d, y in pairs])
        count += s >= observed
    return {'statistic': float(observed), 'p_one_sided': (count + 1) / (PERMUTATIONS + 1)}


def fresh_eval_summary(units):
    out = {}
    for arm in E_ARMS:
        vals = {u['index']: u['fresh_eval']['delta_J'][arm] for u in complete(units)
                if 'fresh_eval' in u and arm in u['fresh_eval']['delta_J']}
        x = np.asarray(list(vals.values()), float)
        if x.size >= 3:
            se = x.std(ddof=1) / math.sqrt(x.size)
            q = stats.t.ppf(0.975, x.size - 1)
            out[arm] = {'n': int(x.size), 'mean_delta_J': float(x.mean()), 'ci95': [float(x.mean() - q * se), float(x.mean() + q * se)],
                        'unit_values': {str(k): v for k, v in sorted(vals.items())}}
        else:
            out[arm] = {'n': int(x.size)}
    return out


def descriptive(units):
    out = {}
    arms = sorted({a for u in complete(units) for a in u['instruments']['arms']})
    for arm in arms:
        out[arm] = {}
        for kappa in ('0.001', '0.01', '0.05'):
            v = [_gain(u, arm, kappa) for u in complete(units)]
            v = [x for x in v if x is not None]
            out[arm][kappa + (' (stress)' if kappa == '0.05' else '')] = {
                'n_ok': len(v), 'mean_G_H': float(np.mean(v)) if v else None}
    tan_pct = []
    for u in complete(units):
        t = _gain(u, 'TAN')
        rnd = [_gain(u, f'RND-k#{r}') for r in range(RND_DRAWS)]
        if t is not None and all(x is not None for x in rnd):
            tan_pct.append(float(np.mean([x <= t for x in rnd]) * 100))
    out['TAN_percentile_among_RND_k'] = tan_pct
    return out


def analyze(units_by_stratum):
    """units_by_stratum: {'shipped': [unit records], 'heldout': [...]} -> the full report."""
    report = {'tool': TOOL, 'kappa_primary': 1e-2, 'alpha': ALPHA,
              'status_counts': {s: _counts(u) for s, u in units_by_stratum.items()}}
    m, a, g = stratum_primary(units_by_stratum['shipped'])
    decisions = holm({'M-H1': m, 'A-H1': a})
    gate_passed = bool(g.get('estimable') and g['lower95_one_sided'] > 0)
    report['primary'] = {'M-H1': dict(m, **decisions['M-H1']), 'A-H1': dict(a, **decisions['A-H1']),
                         'gate_SPAN': dict(g, passed=gate_passed),
                         'permitted_conclusion': None if gate_passed else PERMITTED['insensitive']}
    hm, ha, hg = stratum_primary(units_by_stratum['heldout'])
    cross = bool(hm.get('estimable') and hm['mean'] > 0 and hm['lower95_one_sided'] > 0)
    report['heldout_circuit'] = {'M-H1': hm, 'A-H1': ha, 'gate_SPAN': hg, 'cross_circuit_statement_permitted': cross,
                                 'note': 'reported separately; never pooled into the shipped tests'}
    sec, pairs = {}, []
    for tag, stratum in enumerate(('shipped', 'heldout')):
        sec[stratum], D, Y = secondary(units_by_stratum[stratum], tag)
        pairs.append((D, Y))
        sec[stratum]['fresh_eval_delta_J'] = fresh_eval_summary(units_by_stratum[stratum])
        sec[stratum]['descriptive'] = descriptive(units_by_stratum[stratum])
    if all(len(d) >= 3 for d, _ in pairs):
        sec['P_H1_stratified'] = stratified_permutation(pairs, dq.DQ_STUDY.stream(PERM_TAG, 0))
    report['secondary'] = sec
    report['secondary_note'] = ('P-H1, P-H1b, reliability and fresh-episode results are secondary; a null P-H1 is '
                                'inconclusive at this measurement reliability and is not evidence against the '
                                'geometric account (review 36)')
    return report


def _counts(units):
    out = {}
    for u in units:
        out[u.get('status', 'missing')] = out.get(u.get('status', 'missing'), 0) + 1
    return out


def load_units(campaign):
    units = {'shipped': [], 'heldout': []}
    for path in sorted((Path(campaign) / 'units').glob('*.json')):
        u = json.loads(path.read_text())
        units[u['circuit']].append(u)
    for stratum, us in units.items():
        present = {u['index'] for u in us}
        for i in range(UNITS_PER_STRATUM):
            if i not in present:
                us.append({'circuit': stratum, 'index': i, 'status': 'missing'})
    return units


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(load_units(args.campaign))
    dq.claim_output(args.out, directory=True)
    dq.write_json(args.out / 'b1_report.json', json.loads(json.dumps(report, default=float)))
    print(json.dumps(report['primary'], default=float)[:2000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
