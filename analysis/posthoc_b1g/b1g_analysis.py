"""Analysis of extension G, exactly as fixed in PLAN.md (written before any run).

    python b1g_analysis.py [--provisional]      # writes b1g_results.json; refuses incomplete data unless --provisional
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
RNG = np.random.default_rng(20261024)


def t_sum(d, scale=1.):
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


def gain(arms, arm, kappa):
    e = arms[arm][kappa]
    return e['G_H'] if e.get('status') == 'ok' else np.nan


def rnd(arms, kappa):
    return np.nanmean([gain(arms, f'RND-k#{j}', kappa) for j in range(20)])


def share(nat, lin, n_boot=20_000):
    nat, lin = np.asarray(nat, float), np.asarray(lin, float)
    ok = ~np.isnan(nat) & ~np.isnan(lin)
    nat, lin = nat[ok], lin[ok]
    point = nat.mean() / lin.mean()
    idx = RNG.integers(0, nat.size, size=(n_boot, nat.size))
    boot = nat[idx].mean(1) / lin[idx].mean(1)
    return {'n': int(nat.size), 'share': float(point), 'lo': float(np.percentile(boot, 2.5)),
            'hi': float(np.percentile(boot, 97.5))}


def b1_like(arms_by_unit, nat, lin):
    out = {}
    for kappa in ('0.001', '0.01', '0.05'):
        if kappa not in arms_by_unit[0][nat]:
            continue
        a = np.array([gain(x, nat, kappa) for x in arms_by_unit])
        b = np.array([gain(x, lin, kappa) for x in arms_by_unit])
        out[kappa] = {'natural': t_sum(a, 1e5), 'linear': t_sum(b, 1e5), 'natural_minus_linear': t_sum(a - b, 1e5),
                      'retained_share': share(a, b)}
    return out


def main(provisional):
    plan = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / 'records' / '*.json')))]
    if len(recs) < 72 and not provisional:
        raise SystemExit(f'{len(recs)} of 72 task records; use --provisional for a partial look')
    G = [r for r in recs if r['part'] == 'G' and r['status'] == 'complete']
    D = [r for r in recs if r['part'] == 'D' and r['status'] == 'complete' and r['self_check']['passed']
         and r['trunk_identical_to_B1_unit']]
    out = {'plan_sha256': plan, 'provisional': len(recs) < 72, 'records': len(recs),
           'failed_or_excluded': [(r['part'], r['index'], r.get('status'), r.get('error')) for r in recs
                                  if r not in G and r not in D]}
    gA = [r['instruments']['arms'] for r in G]
    dA = [r['arms'] for r in D]
    k = '0.01'
    G1 = t_sum([gain(a, 'HQ-Fisher', k) - gain(a, 'TAN', k) for a in gA])
    D1 = t_sum([gain(a, 'C-Fisher', k) - gain(a, 'C-TAN', k) for a in dA])
    rej = holm([G1['p'], D1['p']])
    out['family_G'] = {'G1_HQ-Fisher_minus_TAN_generic_angles': G1, 'D1_C-Fisher_minus_C-TAN': D1,
                       'holm_rejected': {'G1': rej[0], 'D1': rej[1]}}
    for key, x, r_ in (('G1', G1, rej[0]), ('D1', D1, rej[1])):
        out['family_G'][f'{key}_reading'] = ('resolved negative' if r_ and x['mean'] < 0 else
                                             'resolved positive' if r_ else 'not resolved')
    # descriptive
    desc = {'HQ generic angles': b1_like(gA, 'HQ-Fisher', 'TAN'), 'C-matched': b1_like(dA, 'C-Fisher', 'C-TAN')}
    b1 = [json.loads(Path(f).read_text())['instruments']['arms'] for f in sorted(glob.glob(str(B1_UNITS / 'shipped_*.json')))]
    desc['HQ B1 angles (stored B1)'] = b1_like(b1, 'HQ-Fisher', 'TAN')
    other = {}
    for kappa in ('0.001', '0.01'):
        other[f'SPAN-HQ-Fisher@{kappa}'] = t_sum([gain(a, 'SPAN', kappa) - gain(a, 'HQ-Fisher', kappa) for a in gA], 1e5)
        other[f'HQ-Fisher-RND-k@{kappa}'] = t_sum([gain(a, 'HQ-Fisher', kappa) - rnd(a, kappa) for a in gA], 1e5)
        other[f'TAN-RND-k@{kappa}'] = t_sum([gain(a, 'TAN', kappa) - rnd(a, kappa) for a in gA], 1e5)
        other[f'HQ-plain-TAN@{kappa}'] = t_sum([gain(a, 'HQ-plain', kappa) - gain(a, 'TAN', kappa) for a in gA], 1e5)
        other[f'C-plain-C-TAN@{kappa}'] = t_sum([gain(a, 'C-plain', kappa) - gain(a, 'C-TAN', kappa) for a in dA], 1e5)
    desc['other_contrasts_e-5'] = other

    def med(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.median(xs)) if xs else None
    cond = {}
    for name, recs_, getter in (('HQ generic angles', G, lambda r: r['instruments']),
                                ('HQ B1 angles', [json.loads(Path(f).read_text()) for f in
                                                  sorted(glob.glob(str(B1_UNITS / 'shipped_*.json')))],
                                 lambda r: r['instruments'])):
        ins_ = [getter(r) for r in recs_]
        cond[name] = {'jacobian_cond2_median': med([(i['singular_values'][0] / i['singular_values'][i['k'] - 1]) ** 2
                                                     for i in ins_]),
                      'rank_k': sorted({i['k'] for i in ins_}),
                      'fisher_condition_tangent_median': med([i['diagnostics']['fisher_condition_tangent'] for i in ins_]),
                      'HQ-Fisher_alpha_median@0.01': med([i['arms']['HQ-Fisher']['0.01'].get('alpha') for i in ins_]),
                      'HQ-Fisher_quadratic_rel_error_median@0.01': med(
                          [i['arms']['HQ-Fisher']['0.01'].get('quadratic_rel_error') for i in ins_])}
    cond['C-matched'] = {'jacobian_cond2_median': med([a['diagnostics']['jacobian_cond2'] for a in dA]),
                         'fisher_condition_median': med([a['diagnostics']['fisher_condition'] for a in dA]),
                         'C-Fisher_alpha_median@0.01': med([a['C-Fisher']['0.01'].get('alpha') for a in dA]),
                         'C-Fisher_quadratic_rel_error_median@0.01': med(
                             [a['C-Fisher']['0.01'].get('quadratic_rel_error') for a in dA]),
                         'C-Fisher_parameter_displacement_median@0.01': med(
                             [a['C-Fisher']['0.01'].get('parameter_displacement') for a in dA]),
                         'theta0_norm_median': med([a['diagnostics']['theta0_norm'] for a in dA])}
    desc['conditioning'] = cond
    out['descriptive'] = desc
    (HERE / 'b1g_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('records', 'failed_or_excluded', 'family_G')}, indent=1, default=float))
    for name in ('HQ B1 angles (stored B1)', 'HQ generic angles', 'C-matched'):
        x = desc[name]['0.01']
        print(f"{name}: natural {x['natural']['mean']:+.2f}, linear {x['linear']['mean']:+.2f}, diff "
              f"{x['natural_minus_linear']['mean']:+.2f} [{x['natural_minus_linear']['lo']:+.2f}, "
              f"{x['natural_minus_linear']['hi']:+.2f}] e-5; retained share {x['retained_share']['share']:.2f} "
              f"[{x['retained_share']['lo']:.2f}, {x['retained_share']['hi']:.2f}]")
    print(json.dumps(cond, indent=1, default=float))


if __name__ == '__main__':
    main('--provisional' in sys.argv)
