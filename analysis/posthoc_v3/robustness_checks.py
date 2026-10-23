"""Post-hoc robustness checks (2026-10-22). Reads stored records only; descriptive.

1. B1: realised mean forward KL on the held-out set H for each arm at kappa = 0.01 (does the K-set match transfer?)
2. B1/B2 primary contrasts: share of units/blocks in the hypothesised direction; sign-flip permutation p-values
   (2e5 random sign assignments, fixed seed), as a distribution-free companion of the paired t tests
3. Stress evaluation: block-level 95% t intervals of each learned family's target-present cell mean J at D600 = 8 m
Usage: python robustness_checks.py <idea_dq dq output dir> <out.json>
"""
import glob
import json
import sys

import numpy as np
from scipy import stats

RNG = np.random.default_rng(20261022)
NPERM = 200_000


def signflip(d, alternative):
    d = np.asarray(d, float)
    obs = d.mean()
    s = RNG.choice([-1., 1.], size=(NPERM, len(d)))
    null = (s * d).mean(axis=1)
    if alternative == 'greater':
        return float((np.sum(null >= obs) + 1) / (NPERM + 1))
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (NPERM + 1))


def gain(u, arm, k='0.01', key='G_H'):
    if arm == 'RND-k':
        return float(np.mean([u['instruments']['arms'][f'RND-k#{i}'][k][key] for i in range(20)]))
    return u['instruments']['arms'][arm][k][key]


def main(root, out):
    res = {}
    for stratum, label in (('shipped', 'two-layer'), ('heldout', 'three-layer')):
        U = [json.load(open(f)) for f in sorted(glob.glob(f'{root}/b1_campaign_v1/units/{stratum}_*.json'))]
        klh = {a: [gain(u, a, key='kl_H') for u in U] for a in ('HQ-Fisher', 'SPAN', 'RND-k', 'TAN')}
        mh1 = np.array([gain(u, 'SPAN') - gain(u, 'HQ-Fisher') for u in U])
        ah1 = np.array([gain(u, 'HQ-Fisher') - gain(u, 'RND-k') for u in U])
        res[label] = {
            'kl_H_at_kappa_0.01': {a: {'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)),
                                       'p90': float(np.percentile(v, 90))} for a, v in klh.items()},
            'M-H1': {'share_SPAN_gt_HQFisher': float(np.mean(mh1 > 0)), 'units': len(mh1),
                     'signflip_p_one_sided': signflip(mh1, 'greater'), 'signflip_p_two_sided': signflip(mh1, 'two'),
                     'standardised_effect': float(mh1.mean() / mh1.std(ddof=1))},
            'A-H1': {'share_HQFisher_lt_RNDk': float(np.mean(ah1 < 0)), 'signflip_p_two_sided': signflip(ah1, 'two'),
                     'standardised_effect': float(ah1.mean() / ah1.std(ddof=1))}}
    auc = {}
    for arm in ('quantum', 'bottleneck', 'surrogate625'):
        for f in glob.glob(f'{root}/b2_campaign_v3/runs/{arm}__b*/run/*/metrics.jsonl'):
            b = int(f.split('__b')[1].split('/')[0])
            auc[(arm, b)] = json.loads(open(f).read().splitlines()[-1])['auc']['sample']
    for name, other in (('B-H1', 'bottleneck'), ('B-H2', 'surrogate625')):
        d = np.array([auc[('quantum', b)] - auc[(other, b)] for b in range(36)])
        res[name] = {'share_HQ_lower': float(np.mean(d < 0)), 'signflip_p_two_sided': signflip(d, 'two'),
                     'blocks': len(d)}
    stress = {}
    for f in sorted(glob.glob(f'{root}/freeze_a_campaign_v1/jobs/*__stress.json')):
        fam, blk = f.split('/')[-1].split('__')[:2]
        for e in json.load(open(f))['episodes']:
            stress.setdefault((fam, e['budget'], e['d600'], e['presence']), {}).setdefault(blk, []).append(e['J'])
    res['stress_8m_target_present'] = {}
    for (fam, budget, d600, presence), blocks in sorted(stress.items()):
        if d600 != 8.0 or presence != 'positive':
            continue
        m = np.array([np.mean(v) for v in blocks.values()])
        se = m.std(ddof=1) / np.sqrt(len(m))
        q = stats.t.ppf(0.975, len(m) - 1)
        res['stress_8m_target_present'][f'{fam} {budget}'] = {'blocks': len(m), 'mean': float(m.mean()),
                                                               'ci95': [float(m.mean() - q * se), float(m.mean() + q * se)]}
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
