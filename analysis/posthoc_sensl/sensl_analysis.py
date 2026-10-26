"""Descriptive re-scoring at the declared false-report weights 0.5 and 2.0 (PLAN.md). No episode is run.

    python sensl_analysis.py          # writes sensl_results.json
"""
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq')
LAMBDAS = (0.5, 1.0, 2.0)
N_REF = 18.
GRID = np.arange(0, 30001, 3000)
FAMILIES = {'quantum': 'HQ', 'bottleneck': 'C-matched', 'mlp': 'C-strong', 'surrogate625': 'Surrogate625'}


def j_lam(e, lam):
    return ((e['tp_timely'] if e['return_ok'] else 0.) - lam * e['fp_spatial']) / N_REF


def check(e):
    if abs(j_lam(e, 1.0) - e['J']) > 1e-12:
        raise SystemExit(f'J_1 does not reproduce the stored J: {e}')


def t_sum(x):
    x = np.asarray(x, float)
    n = x.size
    m, sd = x.mean(), x.std(ddof=1)
    q = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'mean': m, 'lo': m - q, 'hi': m + q, 'p': float(2 * stats.t.sf(abs(m / (sd / np.sqrt(n))), n - 1))}


def stress():
    out = {}
    jobs = {}
    for f in glob.glob(str(DQ / 'freeze_a_campaign_v1/jobs/*__stress.json')):
        j = json.loads(Path(f).read_text())
        name = Path(f).name.replace('__stress.json', '')
        for e in j['episodes']:
            check(e)
        jobs[name] = j['episodes']

    def weighted(eps, lam, d600):
        cells = {}
        for e in eps:
            if e['d600'] == d600:
                cells.setdefault((e['budget'], e['presence']), []).append(j_lam(e, lam))
        assert len(cells) == 4, (d600, cells.keys())
        return sum((0.4 if p == 'positive' else 0.1) * np.mean(v) for (b, p), v in cells.items())
    levels = sorted({e['d600'] for e in jobs['p3_rule']})
    for lam in LAMBDAS:
        res = {}
        for d600 in levels:
            row = {'rule': weighted(jobs['p3_rule'], lam, d600), 'random': weighted(jobs['random'], lam, d600)}
            for fam, role in FAMILIES.items():
                vals = [weighted(eps, lam, d600) for name, eps in jobs.items() if name.startswith(fam + '__b')]
                row[role] = t_sum(vals)
            best = max(row[r]['mean'] for r in FAMILIES.values())
            row['rule_minus_best_learned_RE'] = N_REF * (row['rule'] - best)
            row['ordering_rule_above_all_learned'] = bool(row['rule'] > max(row[r]['hi'] for r in FAMILIES.values()))
            res[f'{d600:g}'] = row
        out[repr(lam)] = res
    return out


def auc_lam(path, lam):
    curve = []
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e.get('kind') != 'evaluation':
            continue
        cells = {}
        for x in e['results']['sample']['episodes']:
            check(x)
            cells.setdefault(x['cell_id'], []).append(j_lam(x, lam))
        J = sum((0.2 if c.endswith('positive') else 0.05) * np.mean(v) for c, v in cells.items())
        for p in e['points']:
            curve.append((p, J))
    curve.sort()
    pts = np.array([c[0] for c in curve], float)
    assert np.array_equal(pts, GRID)
    Js = np.array([c[1] for c in curve])
    return float(np.sum((Js[1:] + Js[:-1]) / 2 * np.diff(pts)) / 30000.)


def b2():
    out = {}
    runs = {a: {int(f.split('__b')[1].split('/')[0]): f for f in
                glob.glob(str(DQ / f'b2_campaign_v3/runs/{a}__b*/run/{a}/metrics.jsonl'))}
            for a in ('quantum', 'bottleneck', 'surrogate625')}
    for lam in LAMBDAS:
        auc = {a: {k: auc_lam(f, lam) for k, f in v.items()} for a, v in runs.items()}
        b = sorted(auc['quantum'])
        out[repr(lam)] = {'B-H1': t_sum([auc['quantum'][k] - auc['bottleneck'][k] for k in b]),
                          'B-H2': t_sum([auc['quantum'][k] - auc['surrogate625'][k] for k in b])}
    return out


def main():
    res = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'stress': stress(), 'b2': b2()}
    (HERE / 'sensl_results.json').write_text(json.dumps(res, indent=1, default=float))
    for lam in ('0.5', '1.0', '2.0'):
        s = res['stress'][lam]
        print(f'lambda {lam}:', {d: (round(v['rule'], 3), {r: round(v[r]['mean'], 3) for r in FAMILIES.values()},
                                     round(v['random'], 3), 'gap RE', round(v['rule_minus_best_learned_RE'], 2),
                                     v['ordering_rule_above_all_learned']) for d, v in s.items()})
        x = res['b2'][lam]
        print('   B2', {k: (round(v['mean'], 4), round(v['lo'], 4), round(v['hi'], 4), round(v['p'], 4)) for k, v in x.items()})


if __name__ == '__main__':
    main()
