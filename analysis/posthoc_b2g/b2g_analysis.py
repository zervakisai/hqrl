"""Analysis of the post-hoc runs of HQ from generic circuit angles, exactly as fixed in PLAN.md (written before any run).

    python b2g_analysis.py [--provisional]     # writes b2g_results.json; refuses incomplete data unless --provisional
    python b2g_analysis.py --selftest          # B-H1 of record recomputed from the stored B2 runs (no new run read)
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
BLOCKS = range(36)


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
    return {'n': n, 'mean': float(m), 'lo': float(m - q * se), 'hi': float(m + q * se),
            'p': float(2 * stats.t.sf(abs(m / se), n - 1)), 'dz': float(m / sd), 'share_negative': float(np.mean(d < 0))}


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, still = [False] * len(ps), True
    for rank, i in enumerate(order):
        still = still and ps[i] <= alpha / (len(ps) - rank)
        rej[i] = still
    return rej


def angles(run_dir):
    import torch
    ck = sorted(glob.glob(str(Path(run_dir) / 'checkpoints' / '*.pt')))
    first = [c for c in ck if Path(c).name.startswith('update00000_')][0]
    final = [c for c in ck if Path(c).name.startswith('final_')][0]
    a = torch.load(first, map_location='cpu', weights_only=False)['model_state_dict']['actor.weights']
    b = torch.load(final, map_location='cpu', weights_only=False)['model_state_dict']['actor.weights']
    return {'init_mean_abs_rad': float(a.abs().mean()), 'final_mean_abs_rad': float(b.abs().mean()),
            'mean_abs_change_rad': float((b - a).abs().mean())}


def selftest():
    hq, cm = runs(B2, 'quantum__b*', 'quantum'), runs(B2, 'bottleneck__b*', 'bottleneck')
    bh1 = paired(hq, cm)
    print('B-H1 recomputed', json.dumps(bh1))
    assert bh1['n'] == 36 and abs(bh1['mean'] + 0.0087) < 5e-5 and abs(bh1['p'] - 0.0017) < 2e-4, bh1


def main(provisional=False):
    new = runs(HERE / 'runs', 'G_quantum__b*', 'quantum')
    hq, cm = runs(B2, 'quantum__b*', 'quantum'), runs(B2, 'bottleneck__b*', 'bottleneck')
    complete = len(new) == 36 and len(list((HERE / 'records').glob('*.json'))) == 36
    if not complete and not provisional:
        raise SystemExit(f'incomplete: {len(new)} completed runs; the read-out waits for all 36')
    G1, G2 = paired(new, cm), paired(new, hq)
    r1, r2 = holm([G1['p'], G2['p']])
    G1['holm_rejected'], G2['holm_rejected'] = r1, r2
    read = ('below zero: the deficit persists from generic angles' if r1 and G1['hi'] < 0 else
            'above zero: from generic angles HQ exceeds C-matched' if r1 and G1['lo'] > 0 else
            'not resolved from generic angles (not evidence of equality)')
    desc = {'J0_mean': {k: float(np.mean([v[b]['J0'] for b in sorted(new)])) for k, v in
                        (('HQ generic', new), ('HQ small-angle', hq), ('C-matched', cm))},
            'Jfinal_mean': {k: float(np.mean([v[b]['Jfinal'] for b in sorted(new)])) for k, v in
                            (('HQ generic', new), ('HQ small-angle', hq), ('C-matched', cm))},
            'AUC_mean': {k: float(np.mean([v[b]['auc'] for b in sorted(new)])) for k, v in
                         (('HQ generic', new), ('HQ small-angle', hq), ('C-matched', cm))},
            'B-H1_same_blocks': paired(hq, cm)}
    if complete:
        ang = {'HQ generic': [angles(HERE / f'runs/G_quantum__b{b}/run/quantum') for b in BLOCKS],
               'HQ small-angle': [angles(B2 / f'quantum__b{b}/run/quantum') for b in BLOCKS]}
        desc['angles_median'] = {k: {f: float(np.median([x[f] for x in v])) for f in v[0]} for k, v in ang.items()}
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': not complete,
           'G1_HQgeneric_minus_Cmatched': G1, 'G2_HQgeneric_minus_HQsmallangle': G2, 'G1_reading': read,
           'descriptive': desc}
    (HERE / 'b2g_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('provisional', 'G1_reading')}, indent=1))
    print('G1', json.dumps(G1)); print('G2', json.dumps(G2))


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main('--provisional' in sys.argv)
