"""Analysis of extension C, exactly as fixed in PLAN.md (written before any run).

    python b1c_analysis.py [--provisional]      # writes b1c_results.json; refuses incomplete data unless --provisional
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
B1X = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1x/records')
RE = 18.
H_EPISODES = 256                                  # episodes per B1 set (8 cells x 32)


def t_sum(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = d.size
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.8, n - 1)) * se
    return {'n': int(n), 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': float(2 * stats.t.sf(abs(m / se), n - 1)),
            'dz': m / sd, 'share_negative': float(np.mean(d < 0)), 'mde80': mde}


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    out = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            out[i] = True
        else:
            break
    return out


def g(entry):
    return entry['G_H'] if entry.get('status') == 'ok' else np.nan


def unit_values(r):
    kk = ['0.001', '0.01']
    v = {}
    for kappa in kk:
        for arm in ('HQ-Fisher', 'TAN', 'QTAY'):
            v[(arm, kappa)] = g(r['arms'][arm][kappa])
        for arm in ('CTRIG', 'CTRIGx0.1', 'CTRIGx10', 'CQR'):
            v[(arm, kappa)] = np.nanmean([g(d[arm][kappa]) for d in r['arms']['CTRIG_draws']])
        v[('CTRIG#0', kappa)] = g(r['arms']['CTRIG_draws'][0]['CTRIG'][kappa])
    e = r['arms']['CTRIG_draws'][0].get('CTRIG-16', {}).get('0.01', {})
    v[('CTRIG-16', '0.01')] = e['G_H'] if e.get('status') == 'ok' else np.nan
    x = B1X / f"{r['circuit']}_{r['index']:02d}.json"
    if x.exists():
        b = json.loads(x.read_text())
        ok = b.get('status') == 'complete' and b['reproduction']['passed']
        e = b['arms'].get('HQ-Fisher-m16', {}).get('0.01', {})
        v[('HQ-Fisher-16', '0.01')] = e['G_H'] if ok and e.get('status') == 'ok' else np.nan
    m = r['mission']
    for arm in ('BASE', 'HQ-Fisher', 'TAN', 'CTRIG#0'):
        v[('J', arm)] = m[arm]['weighted_J'] if m[arm]['status'] == 'ok' else np.nan
    v['decisions_per_episode_H'] = r['rows']['heldout'] / H_EPISODES
    return v


def components(records, arm):
    eps = [e for r in records for e in r['mission'][arm].get('episodes', [])]
    return {k: float(np.mean([e[k] for e in eps])) for k in ('tp_timely', 'fp_spatial', 'return_ok', 'fixes_attempted')}


def stratum(records):
    U = [unit_values(r) for r in records]
    col = lambda key: np.array([u.get(key, np.nan) for u in U])                       # noqa: E731
    k1 = '0.01'
    out = {'n_units': len(U)}
    out['Y1_CTRIG_minus_HQ-Fisher'] = t_sum(col(('CTRIG', k1)) - col(('HQ-Fisher', k1)))
    out['Y2_CTRIG_minus_TAN'] = t_sum(col(('CTRIG', k1)) - col(('TAN', k1)))
    out['W1_J_TAN_minus_HQ-Fisher_RE'] = t_sum(RE * (col(('J', 'TAN')) - col(('J', 'HQ-Fisher'))))
    out['W2_J_TAN_minus_BASE_RE'] = t_sum(RE * (col(('J', 'TAN')) - col(('J', 'BASE'))))
    desc = {}
    for kappa in ('0.001', '0.01'):
        for arm in ('HQ-Fisher', 'TAN', 'QTAY', 'CTRIG', 'CTRIGx0.1', 'CTRIGx10', 'CQR'):
            desc[f'{arm}@{kappa}'] = t_sum(col((arm, kappa)) * 1e5)
        for a, b in (('QTAY', 'HQ-Fisher'), ('CQR', 'CTRIG'), ('CTRIG', 'HQ-Fisher'), ('CTRIG', 'TAN'),
                     ('CTRIGx0.1', 'TAN'), ('CTRIGx10', 'TAN'), ('HQ-Fisher', 'TAN')):
            desc[f'{a}-{b}@{kappa}'] = t_sum((col((a, kappa)) - col((b, kappa))) * 1e5)
    for a, b in (('CTRIG-16', 'CTRIG#0'), ('CTRIG-16', 'TAN'), ('HQ-Fisher-16', 'HQ-Fisher'), ('HQ-Fisher-16', 'TAN')):
        desc[f'{a}-{b}@0.01'] = t_sum((col((a, '0.01')) - col((b, '0.01'))) * 1e5)
    # mission against BASE, with first-order surrogate predictions
    dec = col('decisions_per_episode_H')
    mission = {}
    for arm in ('HQ-Fisher', 'TAN', 'CTRIG#0'):
        meas = RE * (col(('J', arm)) - col(('J', 'BASE')))
        pred = RE * dec * col((arm, '0.01'))
        ok = ~np.isnan(meas) & ~np.isnan(pred)
        mission[arm] = {'measured_RE': t_sum(meas), 'predicted_first_order_RE_mean': float(np.nanmean(pred)),
                        'pearson_r_units': float(stats.pearsonr(pred[ok], meas[ok]).statistic) if ok.sum() > 2 else None}
    mission['components'] = {arm: components(records, arm) for arm in ('BASE', 'HQ-Fisher', 'TAN', 'CTRIG#0')}
    # diagnostics
    dg = [r['arms']['diagnostics'] for r in records]
    draws = [d for r in records for d in r['arms']['CTRIG_draws']]
    alpha = lambda arm: float(np.median([r['arms'][arm]['0.01']['alpha'] for r in records                 # noqa: E731
                                         if r['arms'][arm]['0.01'].get('status') == 'ok']))
    qerr = lambda arm: float(np.median([r['arms'][arm]['0.01']['quadratic_rel_error'] for r in records      # noqa: E731
                                        if r['arms'][arm]['0.01'].get('status') == 'ok']))
    out['diagnostics'] = {
        'circuit_h_dd_tangential_share_median': float(np.median([x['circuit_h_dd_tangential_share'] for x in dg])),
        'circuit_along_d_ratio_median': float(np.median([x['circuit_along_d_ratio'] for x in dg])),
        'ctrig_along_d_ratio_median': float(np.median([x['along_d_ratio'] for x in draws])),
        'circuit_h_dd_fisher_norm2_median': float(np.median([x['circuit_h_dd_fisher_norm2'] for x in dg])),
        'ctrig_h_dd_fisher_norm2_median': float(np.median([x['h_dd_fisher_norm2'] for x in draws])),
        'median_alpha_HQ-Fisher': alpha('HQ-Fisher'), 'median_alpha_TAN': alpha('TAN'),
        'median_angle_step_HQ-Fisher': float(np.median([r['arms']['HQ-Fisher']['0.01']['alpha'] * x['d_norm']
                                                         for r, x in zip(records, dg)
                                                         if r['arms']['HQ-Fisher']['0.01'].get('status') == 'ok'])),
        'median_quadratic_rel_error': {a: qerr(a) for a in ('HQ-Fisher', 'TAN', 'QTAY')},
        'median_quadratic_rel_error_CTRIG': float(np.median([d['CTRIG']['0.01']['quadratic_rel_error'] for d in draws
                                                             if d['CTRIG']['0.01'].get('status') == 'ok'])),
        'calibration_failures': {a: int(sum(r['arms'][a][k].get('status') != 'ok' for r in records for k in ('0.001', '0.01')))
                                 for a in ('HQ-Fisher', 'TAN', 'QTAY')},
        'ctrig_calibration_failures': int(sum(d[n][k].get('status') != 'ok' for d in draws
                                              for n in ('CTRIG', 'CTRIGx0.1', 'CTRIGx10', 'CQR') for k in ('0.001', '0.01')))}
    out['descriptive_e-5'] = desc
    out['mission'] = mission
    return out


def reading_A(y1, y2, r1, r2):
    s1 = 'pos' if r1 and y1['mean'] > 0 else 'neg' if r1 else 'nr'
    s2 = 'pos' if r2 and y2['mean'] > 0 else 'neg' if r2 else 'nr'
    if s2 == 'neg' and s1 == 'nr':
        return 'generic: the matched classical chart shows a path shortfall not distinguishable from the circuit\'s'
    if s2 == 'neg' and s1 == 'pos':
        return 'generic component and circuit-specific excess'
    if s2 == 'nr' and s1 == 'pos':
        return 'circuit-specific relative to this control'
    if s1 == 'neg':
        return 'the matched classical chart gains less than the circuit'
    if s2 == 'pos':
        return 'the matched classical chart gains more than the linear path'
    return 'inconclusive'


def main(provisional):
    plan = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / 'records' / '*.json')))]
    failed = [(r['circuit'], r['index'], r.get('error')) for r in recs if r['status'] != 'complete']
    excluded = [(r['circuit'], r['index']) for r in recs if r['status'] == 'complete' and not r['reproduction']['passed']]
    good = [r for r in recs if r['status'] == 'complete' and r['reproduction']['passed']]
    if len(recs) < 72 and not provisional:
        raise SystemExit(f'{len(recs)} of 72 unit records; use --provisional for a partial look')
    out = {'plan_sha256': plan, 'provisional': len(recs) < 72, 'records': len(recs), 'failed': failed,
           'excluded_reproduction': excluded,
           'max_reproduction_rel_diff': max(r['reproduction']['max_rel_diff_G_H_B1_arms'] for r in recs
                                            if r['status'] == 'complete')}
    for circuit in ('shipped', 'heldout'):
        out[circuit] = stratum([r for r in good if r['circuit'] == circuit])
    s = out['shipped']
    rA = holm([s['Y1_CTRIG_minus_HQ-Fisher']['p'], s['Y2_CTRIG_minus_TAN']['p']])
    rC = holm([s['W1_J_TAN_minus_HQ-Fisher_RE']['p'], s['W2_J_TAN_minus_BASE_RE']['p']])
    out['family_A'] = {'holm_rejected': {'Y1': rA[0], 'Y2': rA[1]},
                       'reading': reading_A(s['Y1_CTRIG_minus_HQ-Fisher'], s['Y2_CTRIG_minus_TAN'], *rA)}
    out['family_C'] = {'holm_rejected': {'W1': rC[0], 'W2': rC[1]}}
    for key, rej in (('W1', rC[0]), ('W2', rC[1])):
        x = s['W1_J_TAN_minus_HQ-Fisher_RE' if key == 'W1' else 'W2_J_TAN_minus_BASE_RE']
        out['family_C'][f'{key}_reading'] = ('resolved positive' if rej and x['mean'] > 0 else
                                             'resolved negative' if rej else
                                             f"not resolved (minimum detectable effect at 80% power {x['mde80']:.3f} RE)")
    (HERE / 'b1c_results.json').write_text(json.dumps(out, indent=1, default=float))
    brief = {k: out[k] for k in ('records', 'failed', 'excluded_reproduction', 'max_reproduction_rel_diff', 'family_A',
                                 'family_C')}
    print(json.dumps(brief, indent=1, default=float))
    for c in ('shipped', 'heldout'):
        x = out[c]
        print(c, 'n', x['n_units'])
        for k in ('Y1_CTRIG_minus_HQ-Fisher', 'Y2_CTRIG_minus_TAN'):
            v = x[k]
            print(f"  {k}: {v['mean'] * 1e5:+.2f} [{v['lo'] * 1e5:+.2f}, {v['hi'] * 1e5:+.2f}] e-5  p={v['p']:.2g}")
        for k in ('W1_J_TAN_minus_HQ-Fisher_RE', 'W2_J_TAN_minus_BASE_RE'):
            v = x[k]
            print(f"  {k}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] RE  p={v['p']:.2g}  mde80={v['mde80']:.3f}")


if __name__ == '__main__':
    main('--provisional' in sys.argv)
