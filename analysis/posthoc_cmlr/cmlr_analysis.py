"""Analysis of the post-hoc C-matched-at-1e-2 runs, exactly as fixed in PLAN.md (written before the runs).

    python cmlr_analysis.py [--provisional]     # writes cmlr_results.json; refuses incomplete data unless --provisional
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
HQLR = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_hqlr/runs')


def summary(path):
    lines = Path(path).read_text().splitlines()
    last = json.loads(lines[-1])
    if last.get('status') != 'completed':
        return None
    kls = [json.loads(x)['metrics']['approx_kl'] for x in lines if '"kind": "update"' in x]
    return {'auc': last['auc']['sample'], 'Jfinal': last['learning_curve']['sample'][-1][2],
            'mean_update_kl': float(np.mean(kls)), 'updates': len(kls)}


def runs(root, pattern, arm):
    out = {}
    for f in glob.glob(str(root / pattern / f'run/{arm}/metrics.jsonl')):
        s = summary(f)
        if s is not None:
            out[int(f.split('__b')[1].split('/')[0])] = s
    return out


def paired(a, b):
    blocks = sorted(set(a) & set(b))
    d = np.array([a[k]['auc'] - b[k]['auc'] for k in blocks])
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return {'n': n, 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': 2 * stats.t.sf(abs(m / se), n - 1),
            'dz': m / sd, 'share_negative': float(np.mean(d < 0))}


def num(x, nd=4):
    return f'{x:+.{nd}f}'.replace('-', '−')


def main(provisional):
    plan = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
    cm_new = runs(HERE / 'runs', 'C1_bottleneck__b*', 'bottleneck')
    hq_1e2 = runs(HQLR, 'H1_quantum__b*', 'quantum')
    cm_3e3 = runs(B2, 'bottleneck__b*', 'bottleneck')
    if len(cm_new) < 36 and not provisional:
        raise SystemExit(f'{len(cm_new)} of 36 runs complete; use --provisional for a partial look')
    z1 = paired(hq_1e2, cm_new)
    desc = paired(cm_new, cm_3e3)
    out = {'plan_sha256': plan, 'provisional': len(cm_new) < 36, 'n_new_runs': len(cm_new),
           'Z1_HQ_1e-2_minus_Cmatched_1e-2': z1, 'Z1_rejected': bool(z1['p'] < 0.05),
           'Z1_reading': ('below zero' if z1['hi'] < 0 else 'above zero' if z1['lo'] > 0 else 'not resolved'),
           'Cmatched_1e-2_minus_3e-3': desc,
           'final_J': {'Cmatched_1e-2': float(np.mean([v['Jfinal'] for v in cm_new.values()])),
                       'Cmatched_3e-3': float(np.mean([v['Jfinal'] for v in cm_3e3.values()])),
                       'HQ_1e-2': float(np.mean([v['Jfinal'] for v in hq_1e2.values()]))},
           'mean_update_kl': {'Cmatched_1e-2': float(np.mean([v['mean_update_kl'] for v in cm_new.values()])),
                              'Cmatched_3e-3': float(np.mean([v['mean_update_kl'] for v in cm_3e3.values()])),
                              'HQ_1e-2': float(np.mean([v['mean_update_kl'] for v in hq_1e2.values()]))},
           'updates_range': [min(v['updates'] for v in cm_new.values()), max(v['updates'] for v in cm_new.values())]}
    reading = {'below zero': 'HQ learned less than C-matched also when both used 1e-2',
               'not resolved': 'at 1e-2 the HQ - C-matched difference is not resolved (not evidence of equality)',
               'above zero': 'HQ at 1e-2 exceeded C-matched at 1e-2'}[out['Z1_reading']]
    out['text'] = (
        f"C-matched beyond its grid edge (post hoc). Under an analysis plan fixed by hash before the runs, C-matched was "
        f"also re-trained at 10⁻² on the same 36 blocks, so that both arms were run at 3 × 10⁻³ and 10⁻². HQ (10⁻²) − "
        f"C-matched (10⁻²) AUC was {num(z1['mean'])} [{num(z1['lo'])}, {num(z1['hi'])}] (two-sided p = {z1['p']:.2g}; HQ "
        f"lower in {100 * z1['share_negative']:.0f}% of blocks): {reading}. C-matched's own AUC changed by "
        f"{num(desc['mean'])} [{num(desc['lo'])}, {num(desc['hi'])}] from 3 × 10⁻³ to 10⁻², and its mean final-checkpoint "
        f"weighted J was {out['final_J']['Cmatched_1e-2']:.3f} at 10⁻².")
    (HERE / 'cmlr_results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != 'text'}, indent=1, default=float))
    print(out['text'])


if __name__ == '__main__':
    main('--provisional' in sys.argv)
