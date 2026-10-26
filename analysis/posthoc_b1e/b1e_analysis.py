"""Analysis of extension E, exactly as fixed in PLAN.md (written before any run).

    python b1e_analysis.py [--provisional]      # writes b1e_results.json
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent


def t_sum(d, scale=1e5):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)] * scale
    n = d.size
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return {'n': int(n), 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': float(2 * stats.t.sf(abs(m / se), n - 1)),
            'dz': m / sd, 'share_negative': float(np.mean(d < 0))}


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    out = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            out[i] = True
        else:
            break
    return out


def gh(res, arm):
    e = res.get(arm, {})
    return e.get('G_H', np.nan) if (e.get('status', 'ok') == 'ok' and 'G_H' in e) else np.nan


def stratum(recs):
    out = {'n_units': len(recs)}
    for kappa in ('0.001', '0.01'):
        R = [r['arms'][kappa] for r in recs]
        col = lambda arm: np.array([gh(x, arm) for x in R])                              # noqa: E731
        s = {'E1_OPT-CIRC_minus_OPT-TAN': t_sum(col('OPT-CIRC') - col('OPT-TAN')),
             'E2_OPT-CIRC_minus_HQ-Fisher': t_sum(col('OPT-CIRC') - col('HQ-Fisher')),
             'OPT-TAN_minus_TAN': t_sum(col('OPT-TAN') - col('TAN')),
             'TAN-c_minus_TAN': t_sum(col('TAN-c') - col('TAN')),
             'OPT-CIRC_minus_TAN': t_sum(col('OPT-CIRC') - col('TAN'))}
        s['arm_means_e-5'] = {a: t_sum(col(a)) for a in ('HQ-Fisher', 'HQ-plain', 'TAN', 'TAN-c', 'OPT-CIRC', 'OPT-TAN')}
        s['fit_objective_median'] = {a: float(np.nanmedian([x.get(a, {}).get('L_F', np.nan) for x in R]))
                                     for a in ('HQ-Fisher', 'TAN', 'OPT-CIRC', 'OPT-TAN')}
        fc = [x.get('first_crossing_HQ-Fisher', {}) for x in R]
        s['first_crossing'] = {'units_checked': sum(1 for f in fc if f),
                               'not_monotone_on_grid': sum(1 for f in fc if f and not f['monotone_on_grid']),
                               'earlier_crossing': sum(1 for f in fc if f.get('earlier_crossing')),
                               'G_H_first_minus_B1_e-5': t_sum([f['G_H_first'] - x['HQ-Fisher']['G_H']
                                                                for f, x in zip(fc, R) if f.get('earlier_crossing')])
                               if sum(1 for f in fc if f.get('earlier_crossing')) > 1 else None}
        disp = {}
        for arm in ('HQ-Fisher', 'HQ-plain', 'OPT-CIRC'):
            mx = [x.get(arm, {}).get('max_abs_angle_step') for x in R]
            mx = [v for v in mx if v is not None]
            disp[arm] = {'median_max_abs_angle_step': float(np.median(mx)) if mx else None,
                         'share_beyond_pi': float(np.mean(np.array(mx) > np.pi)) if mx else None}
        for arm in ('HQ-Fisher', 'HQ-plain'):
            mm = [x.get(arm, {}).get('mean_abs_angle_step') for x in R if x.get(arm, {}).get('mean_abs_angle_step')]
            disp[arm]['median_mean_abs_angle_step'] = float(np.median(mm)) if mm else None
        s['angle_displacement'] = disp
        s['reproduction_max_rel_diff'] = float(max(x['reproduction_max_rel_diff'] for x in R))
        s['optimiser'] = {a: {'feasible_units': sum(1 for x in R if x[a].get('feasible')),
                              'best_start_counts': {k: sum(1 for x in R if x[a].get('start') == k)
                                                    for k in sorted({x[a].get('start') for x in R if x[a].get('start')})}}
                          for a in ('OPT-CIRC', 'OPT-TAN')}
        out[kappa] = s
    return out


def main(provisional):
    plan = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / 'records' / '*.json')))]
    good = [r for r in recs if r['status'] == 'complete']
    if len(recs) < 60 and not provisional:
        raise SystemExit(f'{len(recs)} unit records; use --provisional for a partial look')
    out = {'plan_sha256': plan, 'provisional': provisional, 'records': len(recs),
           'failed': [(r['circuit'], r['index'], r.get('error')) for r in recs if r['status'] != 'complete']}
    for c in ('shipped', 'heldout'):
        out[c] = stratum([r for r in good if r['circuit'] == c])
    s = out['shipped']['0.01']
    rej = holm([s['E1_OPT-CIRC_minus_OPT-TAN']['p'], s['E2_OPT-CIRC_minus_HQ-Fisher']['p']])
    e1, e2 = s['E1_OPT-CIRC_minus_OPT-TAN'], s['E2_OPT-CIRC_minus_HQ-Fisher']
    st = lambda x, r: ('pos' if r and x['mean'] > 0 else 'neg' if r else 'nr')                 # noqa: E731
    s1, s2 = st(e1, rej[0]), st(e2, rej[1])
    reading = ('optimised circuit steps reach gains not distinguishable from the best linear tangent step found; the '
               'shortfall lies in the natural step\'s path' if s2 == 'pos' and s1 == 'nr' else
               'even optimised circuit steps gain less than the best linear tangent step (reachable set limits, local '
               'optimiser)' if s1 == 'neg' else
               'optimisation did not improve on the natural circuit step' if s2 == 'nr' else
               'optimised circuit steps exceed the best linear tangent step found' if s1 == 'pos' else 'other')
    out['family_E'] = {'holm_rejected': {'E1': rej[0], 'E2': rej[1]}, 'reading': reading}
    (HERE / 'b1e_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({'records': out['records'], 'failed': out['failed'], 'family_E': out['family_E']}, indent=1))
    for c in ('shipped', 'heldout'):
        for kappa in ('0.001', '0.01'):
            x = out[c][kappa]
            for key in ('E1_OPT-CIRC_minus_OPT-TAN', 'E2_OPT-CIRC_minus_HQ-Fisher', 'OPT-TAN_minus_TAN', 'TAN-c_minus_TAN'):
                v = x[key]
                print(f"{c} {kappa} {key}: {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] p={v['p']:.2g} n={v['n']}")


if __name__ == '__main__':
    main('--provisional' in sys.argv)
