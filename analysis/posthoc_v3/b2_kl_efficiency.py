"""Post-hoc, descriptive (2026-10-22): B2 learning gain per unit of cumulative PPO update KL, from stored records.

gain    = AUC - J(point 0)  (the paper's 'learning gain', subsection 4.6)
cum_kl  = sum over PPO updates of PPO's approx_kl (the per-update diagnostic of subsection 4.6)
Reports per-arm medians over 36 blocks. No test; the KL is PPO's sample estimate, not B1's calibration-set KL.
Usage: python b2_kl_efficiency.py <b2_campaign_v3/runs> <out.json>
"""
import glob
import json
import sys

import numpy as np


def main(runs, out):
    res = {}
    for arm in ('quantum', 'bottleneck', 'surrogate625'):
        rows = []
        for f in sorted(glob.glob(f'{runs}/{arm}__b*/run/*/metrics.jsonl')):
            ev = [json.loads(x) for x in open(f)]
            curve = ev[-1]['learning_curve']['sample']
            kl = sum(e['metrics']['approx_kl'] for e in ev if e['kind'] == 'update')
            rows.append((ev[-1]['auc']['sample'] - curve[0][2], kl))
        g, k = np.array(rows).T
        res[arm] = {'blocks': len(rows), 'gain_mean': g.mean(), 'cum_kl_median': float(np.median(k)),
                    'gain_per_cum_kl_median': float(np.median(g / k)),
                    'gain_per_cum_kl_iqr': [float(np.percentile(g / k, 25)), float(np.percentile(g / k, 75))]}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
