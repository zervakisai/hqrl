"""Analysis of the post-hoc runs of HQ with frozen circuit angles, exactly as fixed in PLAN.md (written before any run).

    python b2f_analysis.py [--provisional]     # writes b2f_results.json; refuses incomplete data unless --provisional
    python b2f_analysis.py --selftest          # B-H1 of record recomputed from the stored B2 runs (no new run read)
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
B2 = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs')
TOST_DELTA = 0.0056          # dq_b2.TOST_DELTA (0.1 RE for Nref = 18)


def summary(path):
    last = json.loads(Path(path).read_text().splitlines()[-1])
    if last.get('status') != 'completed':
        return None
    lc = last['learning_curve']['sample']
    return {'auc': last['auc']['sample'], 'J0': lc[0][2], 'Jfinal': lc[-1][2]}


def runs(root, pattern, arm):
    out = {}
    for f in glob.glob(str(root / pattern / f'run/{arm}/metrics.jsonl')):
        s = summary(f)
        if s is not None:
            out[int(f.split('__b')[1].split('/')[0])] = s
    return out


def paired(a, b, key='auc'):
    blocks = sorted(set(a) & set(b))
    d = np.array([a[k][key] - b[k][key] for k in blocks])
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    q90 = stats.t.ppf(0.95, n - 1)
    return {'n': n, 'mean': float(m), 'lo': float(m - q * se), 'hi': float(m + q * se),
            'lo90': float(m - q90 * se), 'hi90': float(m + q90 * se),
            'p': float(2 * stats.t.sf(abs(m / se), n - 1)), 'dz': float(m / sd), 'share_negative': float(np.mean(d < 0))}


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, still = [False] * len(ps), True
    for rank, i in enumerate(order):
        still = still and ps[i] <= alpha / (len(ps) - rank)
        rej[i] = still
    return rej


def selftest():
    hq, cm = runs(B2, 'quantum__b*', 'quantum'), runs(B2, 'bottleneck__b*', 'bottleneck')
    bh1 = paired(hq, cm)
    print('B-H1 recomputed', json.dumps(bh1))
    assert bh1['n'] == 36 and abs(bh1['mean'] + 0.0087) < 5e-5 and abs(bh1['p'] - 0.0017) < 2e-4, bh1


def main(provisional=False):
    new = runs(HERE / 'runs', 'F_quantum__b*', 'quantum')
    hq, cm = runs(B2, 'quantum__b*', 'quantum'), runs(B2, 'bottleneck__b*', 'bottleneck')
    recs = [json.loads(p.read_text()) for p in (HERE / 'records').glob('*.json')]
    complete = len(new) == 36 and len(recs) == 36
    if not complete and not provisional:
        raise SystemExit(f'incomplete: {len(new)} completed runs; the read-out waits for all 36')
    F1, F2 = paired(new, hq), paired(new, cm)
    F1['holm_rejected'], F2['holm_rejected'] = holm([F1['p'], F2['p']])
    F1['tost_equivalent'] = bool(-TOST_DELTA < F1['lo90'] and F1['hi90'] < TOST_DELTA)
    r1 = ('equivalent within ±0.1 RE: angle training made no difference to early learning' if F1['tost_equivalent'] else
          'below zero: angle training helped' if F1['holm_rejected'] and F1['hi'] < 0 else
          'above zero: angle training hurt (frozen learned more)' if F1['holm_rejected'] and F1['lo'] > 0 else
          'not resolved (not evidence of equality)')
    r2 = ('below zero: the deficit to C-matched remains without angle training' if F2['holm_rejected'] and F2['hi'] < 0 else
          'above zero: with frozen angles HQ learned more than C-matched' if F2['holm_rejected'] and F2['lo'] > 0 else
          'not resolved (not evidence of equality)')
    mean = lambda v, key: float(np.mean([v[b][key] for b in sorted(new)]))          # noqa: E731
    desc = {k: {'J0': mean(v, 'J0'), 'Jfinal': mean(v, 'Jfinal'), 'AUC': mean(v, 'auc')}
            for k, v in (('HQ frozen', new), ('HQ trained', hq), ('C-matched', cm))}
    desc['B-H1_same_blocks'] = paired(hq, cm)
    ang = [r.get('angles', {}) for r in recs]
    desc['angles'] = {'runs_checked': sum(1 for a in ang if a), 'all_equal_initial': all(a.get('all_equal_initial') for a in ang if a),
                      'max_abs_change': max((a.get('max_abs_change', 0.0) for a in ang if a), default=None)}
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': not complete,
           'F1_HQfrozen_minus_HQtrained': F1, 'F2_HQfrozen_minus_Cmatched': F2, 'F1_reading': r1, 'F2_reading': r2,
           'tost_delta': TOST_DELTA, 'descriptive': desc}
    (HERE / 'b2f_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('provisional', 'F1_reading', 'F2_reading')}, indent=1))
    print('F1', json.dumps(F1)); print('F2', json.dumps(F2))


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main('--provisional' in sys.argv)
