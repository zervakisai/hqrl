"""Analysis of extension T, exactly as fixed in PLAN.md (written before any run).

    python b1t_analysis.py [--provisional]      # writes b1t_results.json
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
B1G = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1g/records')
RNG = np.random.default_rng(20261025)


def t_sum(d, scale=1e5, alternative='two-sided'):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)] * scale
    n = d.size
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    t = m / se
    p = 2 * stats.t.sf(abs(t), n - 1) if alternative == 'two-sided' else stats.t.sf(t, n - 1)
    return {'n': int(n), 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': float(p), 'dz': m / sd,
            'share_negative': float(np.mean(d < 0))}


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


def share(nat, lin, n=20_000):
    nat, lin = np.asarray(nat, float), np.asarray(lin, float)
    ok = ~np.isnan(nat) & ~np.isnan(lin)
    nat, lin = nat[ok], lin[ok]
    idx = RNG.integers(0, nat.size, size=(n, nat.size))
    b = nat[idx].mean(1) / lin[idx].mean(1)
    return {'n': int(nat.size), 'share': float(nat.mean() / lin.mean()), 'lo': float(np.percentile(b, 2.5)),
            'hi': float(np.percentile(b, 97.5))}


def main(provisional):
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / 'records' / '*.json')))]
    if len(recs) < 72 and not provisional:
        raise SystemExit(f'{len(recs)} task records; use --provisional for a partial look')
    T = [r for r in recs if r['part'] == 'T' and r['status'] == 'complete']
    D = [r for r in recs if r['part'] == 'TD' and r['status'] == 'complete' and r['self_check']['passed']]
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': len(recs) < 72,
           'records': len(recs), 'failed_or_excluded': [(r['part'], r['block'], r['status'], r.get('error'))
                                                        for r in recs if r not in T and r not in D]}
    TA = [r['instruments']['arms'] for r in T]
    DA = [r['arms'] for r in D]
    k = '0.01'
    T1 = t_sum([g(a, 'HQ-Fisher', k) - g(a, 'TAN', k) for a in TA])
    T2 = t_sum([g(a, 'SPAN', k) - g(a, 'HQ-Fisher', k) for a in TA], alternative='greater')
    T3 = t_sum([g(a, 'HQ-Fisher', k) - rnd(a, k) for a in TA])
    TD1 = t_sum([g(a, 'C-Fisher', k) - g(a, 'C-TAN', k) for a in DA])
    rej = holm([T1['p'], T2['p'], T3['p'], TD1['p']])
    out['family_T'] = {'T1_HQ-Fisher_minus_TAN': T1, 'T2_SPAN_minus_HQ-Fisher_one_sided': T2,
                       'T3_HQ-Fisher_minus_RND-k': T3, 'TD1_C-Fisher_minus_C-TAN': TD1,
                       'holm_rejected': dict(zip(('T1', 'T2', 'T3', 'TD1'), rej))}
    desc = {}
    for kk in ('0.001', '0.01'):
        desc[kk] = {'arms_trained_HQ': {a: t_sum([g(x, a, kk) for x in TA]) for a in ('HQ-Fisher', 'TAN', 'SPAN', 'HQ-plain')},
                    'RND-k_trained_HQ': t_sum([rnd(x, kk) for x in TA]),
                    'arms_trained_Cmatched': {a: t_sum([g(x, a, kk) for x in DA]) for a in ('C-Fisher', 'C-TAN', 'C-plain')},
                    'T1': t_sum([g(a, 'HQ-Fisher', kk) - g(a, 'TAN', kk) for a in TA]),
                    'TD1': t_sum([g(a, 'C-Fisher', kk) - g(a, 'C-TAN', kk) for a in DA])}
        b1 = [json.loads(Path(f).read_text())['instruments']['arms'] for f in sorted(glob.glob(str(B1_UNITS / 'shipped_*.json')))]
        gd = [json.loads(Path(f).read_text())['arms'] for f in sorted(glob.glob(str(B1G / 'D_*.json')))
              if json.loads(Path(f).read_text()).get('status') == 'complete']
        desc[kk]['retained_share'] = {
            'HQ trained': share([g(a, 'HQ-Fisher', kk) for a in TA], [g(a, 'TAN', kk) for a in TA]),
            'C-matched trained': share([g(a, 'C-Fisher', kk) for a in DA], [g(a, 'C-TAN', kk) for a in DA]),
            'HQ untrained (B1)': share([g(a, 'HQ-Fisher', kk) for a in b1], [g(a, 'TAN', kk) for a in b1]),
            'C-matched untrained (extension G)': share([g(a, 'C-Fisher', kk) for a in gd], [g(a, 'C-TAN', kk) for a in gd])
            if gd else None}
    med = lambda xs: float(np.median([x for x in xs if x is not None and np.isfinite(x)]))      # noqa: E731
    desc['conditioning_trained_HQ'] = {
        'jacobian_cond2_median': med([(r['instruments']['singular_values'][0] / r['instruments']['singular_values'][r['instruments']['k'] - 1]) ** 2 for r in T]),
        'fisher_condition_tangent_median': med([r['instruments']['diagnostics']['fisher_condition_tangent'] for r in T]),
        'rank_k': sorted({r['instruments']['k'] for r in T}),
        'HQ-Fisher_alpha_median@0.01': med([r['instruments']['arms']['HQ-Fisher']['0.01'].get('alpha') for r in T]),
        'mean_abs_angle_median': med([float(np.mean(np.abs(r['circuit_weights']))) for r in T])}
    desc['conditioning_trained_Cmatched'] = {
        'jacobian_cond2_median': med([a['diagnostics']['jacobian_cond2'] for a in DA]),
        'fisher_condition_median': med([a['diagnostics']['fisher_condition'] for a in DA])}
    out['descriptive_e-5'] = desc
    (HERE / 'b1t_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('records', 'failed_or_excluded', 'family_T')}, indent=1, default=float))
    print(json.dumps(desc['0.01']['retained_share'], indent=1), json.dumps(desc['conditioning_trained_HQ'], indent=1))


if __name__ == '__main__':
    main('--provisional' in sys.argv)
