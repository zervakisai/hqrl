"""Analysis of extension N, exactly as fixed in PLAN.md (written before any run).

    python b1n_analysis.py [--provisional]      # writes b1n_results.json
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
B1_UNITS = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units')
SIZES = ('64', '128', '256', '1024')


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


def g(arms, arm, k):
    e = arms[arm][k]
    return e['G_H'] if e.get('status') == 'ok' else np.nan


def rnd(arms, k):
    return np.nanmean([g(arms, f'RND-k#{j}', k) for j in range(20)])


def main(provisional):
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / 'records' / '*.json')))]
    good = [r for r in recs if r['status'] == 'complete' and r['reproduction']['passed']]
    if len(recs) < 36 and not provisional:
        raise SystemExit(f'{len(recs)} unit records; use --provisional for a partial look')
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': len(recs) < 36,
           'records': len(recs), 'excluded': [(r['index'], r['status'], r.get('error'), r.get('reproduction'))
                                              for r in recs if r not in good]}
    ladder = {}
    for k in ('0.001', '0.01'):
        for s in SIZES:
            A = [r['arms'][s] for r in good]
            ladder[f'{k}|{s}'] = {
                'HQ-Fisher': t_sum([g(a, 'HQ-Fisher', k) for a in A]), 'TAN': t_sum([g(a, 'TAN', k) for a in A]),
                'SPAN': t_sum([g(a, 'SPAN', k) for a in A]), 'RND-k': t_sum([rnd(a, k) for a in A]),
                'HQ-Fisher-TAN': t_sum([g(a, 'HQ-Fisher', k) - g(a, 'TAN', k) for a in A]),
                'SPAN-RND-k': t_sum([g(a, 'SPAN', k) - rnd(a, k) for a in A]),
                'SPAN-HQ-Fisher': t_sum([g(a, 'SPAN', k) - g(a, 'HQ-Fisher', k) for a in A]),
                'HQ-Fisher-RND-k': t_sum([g(a, 'HQ-Fisher', k) - rnd(a, k) for a in A])}
    out['ladder_e-5'] = ladder
    k = '0.01'
    N1 = t_sum([(g(r['arms']['1024'], 'HQ-Fisher', k) - g(r['arms']['1024'], 'TAN', k))
                - (g(r['arms']['256'], 'HQ-Fisher', k) - g(r['arms']['256'], 'TAN', k)) for r in good])
    N2 = t_sum([(g(r['arms']['1024'], 'SPAN', k) - rnd(r['arms']['1024'], k))
                - (g(r['arms']['256'], 'SPAN', k) - rnd(r['arms']['256'], k)) for r in good])
    rej = holm([N1['p'], N2['p']])
    rd = lambda x, r_: ('resolved positive' if r_ and x['mean'] > 0 else 'resolved negative' if r_ else 'not resolved')  # noqa: E731
    out['family_N'] = {'N1_path_gap_change_1024_minus_256': N1, 'N2_SPAN_RNDk_change_1024_minus_256': N2,
                       'holm_rejected': {'N1': rej[0], 'N2': rej[1]},
                       'N1_reading': rd(N1, rej[0]), 'N2_reading': rd(N2, rej[1]),
                       'path_gap_at_1024': ladder['0.01|1024']['HQ-Fisher-TAN']}
    sh = [r.get('split_half_cosine_1024') for r in good]
    b1 = [json.loads((B1_UNITS / f'shipped_{r["index"]:02d}.json').read_text())['instruments']['reliability']
          ['split_half_cosine'] for r in good]
    out['split_half_cosine'] = {'F1024_median': float(np.median([x for x in sh if x is not None])) if any(sh) else None,
                                'F256_B1_median': float(np.median(b1))}
    (HERE / 'b1n_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('records', 'excluded', 'family_N', 'split_half_cosine')}, indent=1, default=float))
    for s in SIZES:
        x = ladder[f'0.01|{s}']
        print(f"F{s}: HQ-Fisher−TAN {x['HQ-Fisher-TAN']['mean']:+.2f} [{x['HQ-Fisher-TAN']['lo']:+.2f}, {x['HQ-Fisher-TAN']['hi']:+.2f}]; "
              f"SPAN−RND-k {x['SPAN-RND-k']['mean']:+.2f}; TAN {x['TAN']['mean']:+.2f}; HQ-Fisher {x['HQ-Fisher']['mean']:+.2f}")


if __name__ == '__main__':
    main('--provisional' in sys.argv)
