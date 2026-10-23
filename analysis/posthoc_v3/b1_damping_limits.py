"""Post-hoc, descriptive (2026-10-22): the two limits of a damping sweep are already prespecified B1 arms.

Eq. (4) with relative damping delta -> 0 is HQ-Fisher; with delta -> infinity, (F + lambda I)^-1 g -> g / lambda, so
after KL calibration the step is the plain-gradient theta step realised through the circuit, i.e. the HQ-plain arm.
This script reads the stored B1 per-unit records (unchanged) and reports paired contrasts of these circuit-realised
arms against the linear arms (TAN, RND-k, SPAN), with individual 95% paired t intervals, plus median quadratic error.
Usage: python b1_damping_limits.py <b1_campaign_v1/units dir> <out.json>
"""
import glob
import json
import sys

import numpy as np
from scipy import stats

KAPPAS = ('0.001', '0.01', '0.05')
CONTRASTS = [('SPAN', 'HQ-Fisher'), ('HQ-plain', 'TAN'), ('HQ-plain', 'RND-k'), ('HQ-plain', 'SPAN'),
             ('HQ-Adam-dir', 'TAN'), ('HQ-plain', 'HQ-Fisher'), ('HQ-Fisher', 'TAN')]


def gain(u, arm, k):
    if arm == 'RND-k':
        v = [u['instruments']['arms'][f'RND-k#{i}'][k] for i in range(20)]
        return np.nan if any(x['status'] != 'ok' for x in v) else float(np.mean([x['G_H'] for x in v]))
    x = u['instruments']['arms'][arm][k]
    return x['G_H'] if x['status'] == 'ok' else np.nan


def paired(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    q = stats.t.ppf(0.975, n - 1)
    return {'n': n, 'mean': m, 'lo': m - q * se, 'hi': m + q * se}


def main(units, out):
    res = {'statement': 'post hoc, descriptive; individual 95% paired t intervals; no p-values; units J per decision'}
    for stratum, label in (('shipped', 'two-layer'), ('heldout', 'three-layer')):
        U = [json.load(open(f)) for f in sorted(glob.glob(f'{units}/{stratum}_*.json'))]
        r = {'units': len(U), 'contrasts': {}, 'median_quadratic_rel_error': {}}
        for k in KAPPAS:
            r['contrasts'][k] = {f'{a} - {b}': paired([gain(u, a, k) - gain(u, b, k) for u in U]) for a, b in CONTRASTS}
            r['median_quadratic_rel_error'][k] = {
                a: float(np.nanmedian([u['instruments']['arms'][a][k].get('quadratic_rel_error', np.nan) for u in U]))
                for a in ('HQ-Fisher', 'HQ-plain', 'HQ-Adam-dir', 'TAN')}
        res[label] = r
    json.dump(res, open(out, 'w'), indent=1)
    for label in ('two-layer', 'three-layer'):
        for k in KAPPAS:
            print(label, k, {c: '%+.2f [%+.2f, %+.2f] n=%d' % (v['mean'] * 1e5, v['lo'] * 1e5, v['hi'] * 1e5, v['n'])
                             for c, v in res[label]['contrasts'][k].items()})
        print(label, 'quadratic error', res[label]['median_quadratic_rel_error'])


if __name__ == '__main__':
    main(*sys.argv[1:])
