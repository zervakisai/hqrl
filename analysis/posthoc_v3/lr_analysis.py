"""Analysis of the post-hoc matched-learning-rate runs, exactly as fixed in PLAN.md (written before the runs).

    python lr_analysis.py [--provisional]     # writes lr_results.json (and refuses incomplete data unless --provisional)

L1: AUC(HQ, 3e-3; stored B2) - AUC(twin, 3e-3; new)      L2: AUC(HQ, 1e-3; new) - AUC(twin, 1e-3; stored B2)
Paired over blocks; mean, 95% paired t interval, two-sided p, standardised effect, Holm over the two (post hoc).
The manuscript wording for both outcomes is generated here from the numbers, so it cannot drift from them.
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


def summary(path):
    last = json.loads(Path(path).read_text().splitlines()[-1])
    if last.get('status') != 'completed':
        return None
    curve = last['learning_curve']['sample']
    return {'auc': last['auc']['sample'], 'J0': curve[0][2], 'Jfinal': curve[-1][2]}


def stored(arm):
    out = {}
    for f in glob.glob(str(B2 / f'{arm}__b*/run/{arm}/metrics.jsonl')):
        out[int(f.split('__b')[1].split('/')[0])] = summary(f)
    return out


def new(tag, arm):
    out = {}
    for f in glob.glob(str(HERE / f'runs/{tag}_{arm}__b*/run/{arm}/metrics.jsonl')):
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
    p = 2 * stats.t.sf(abs(m / se), n - 1)
    return {'n': n, 'blocks': blocks, 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': p, 'dz': m / sd,
            'share_hq_lower': float(np.mean(d < 0))}


def num(x, nd=4):
    return f'{x:+.{nd}f}'.replace('-', '−')


def ci(r):
    return f'{num(r["mean"])} [{num(r["lo"])}, {num(r["hi"])}]'


def pstr(p):
    if p >= 0.001:
        return f'{p:.2g}'
    m, e = f'{p:.1e}'.split('e')
    return f'{m} × 10⁻' + str(-int(e)).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def main(provisional=False):
    hq3, tw1 = stored('quantum'), stored('surrogate625')
    tw3, hq1 = new('L1', 'surrogate625'), new('L2', 'quantum')
    if not provisional and (len(tw3) < 36 or len(hq1) < 36):
        n_fail = len(list((HERE / 'records').glob('*.json')))
        if n_fail < 72:
            raise SystemExit(f'incomplete: {len(tw3)} L1 and {len(hq1)} L2 runs completed; {n_fail}/72 records')
    L1, L2 = paired(hq3, tw3), paired(hq1, tw1)
    ps = sorted([('L1', L1['p']), ('L2', L2['p'])], key=lambda x: x[1])
    holm = {ps[0][0]: ps[0][1] <= 0.025, ps[1][0]: ps[0][1] <= 0.025 and ps[1][1] <= 0.05}
    for k, r in (('L1', L1), ('L2', L2)):
        r['holm_rejects'] = holm[k]
        r['cell'] = ci(r)
    within = {'HQ 3e-3 - HQ 1e-3': paired(hq3, hq1), 'twin 3e-3 - twin 1e-3': paired(tw3, tw1)}

    # the reading follows the Holm decision of PLAN.md (familywise 0.05 over L1 and L2), not the unadjusted intervals
    neg1, neg2 = L1['holm_rejects'] and L1['mean'] < 0, L2['holm_rejects'] and L2['mean'] < 0
    pos1, pos2 = L1['holm_rejects'] and L1['mean'] > 0, L2['holm_rejects'] and L2['mean'] > 0
    res1 = f'{ci(L1)} at 3 × 10⁻³ (two-sided p = {pstr(L1["p"])}; HQ lower in {round(100 * L1["share_hq_lower"])}% of blocks)'
    res2 = f'{ci(L2)} at 10⁻³ (p = {pstr(L2["p"])}; {round(100 * L2["share_hq_lower"])}%)'
    if pos1 or pos2:
        which = ' and '.join(x for x, pos in (('3 × 10⁻³', pos1), ('10⁻³', pos2)) if pos)
        reading = (f'At a matched learning rate of {which} HQ learned more than its exact twin (Holm-adjusted), so the '
                   'B-H2 difference reflects the learning-rate difference of B2 rather than the hybrid head; the '
                   'early-training evidence for a hybrid deficit rests on the comparison with C-matched (B-H1).')
        concl = ('In paired early training, HQ has lower learning-curve area than C-matched; its lower area than the '
                 f'exact twin reverses at a matched learning rate of {which}.')
        abstract = 'and than its exact twin, a difference that reverses at matched learning rates'
    elif neg1 and neg2:
        reading = ('At both matched learning rates HQ learned less than its exact twin, so the B-H2 difference is not '
                   'explained by the learning-rate difference.')
        concl = ('In paired early training, HQ has lower learning-curve area than C-matched and than its exact twin; '
                 'a post-hoc check at matched learning rates leaves the HQ–twin difference in place.')
        abstract = 'and than its exact twin, also at matched learning rates'
    elif not neg1 and not neg2:
        reading = ('At matched learning rates the HQ–twin difference was not resolved, so B-H2 cannot be separated '
                   'from the learning-rate difference of B2; the early-training evidence for a hybrid deficit rests '
                   'on the comparison with C-matched (B-H1).')
        concl = ('In paired early training, HQ has lower learning-curve area than C-matched; its lower area than the '
                 'exact twin is not resolved once the two use the same learning rate.')
        abstract = 'and than its exact twin, a difference not resolved at matched learning rates'
    else:
        which, other = ('3 × 10⁻³', '10⁻³') if neg1 else ('10⁻³', '3 × 10⁻³')
        reading = (f'HQ learned less than its twin when both used {which}, but the difference was not resolved at '
                   f'{other}; the HQ–twin difference therefore depends on the learning rate and is weaker evidence '
                   'than B-H1.')
        concl = ('In paired early training, HQ has lower learning-curve area than C-matched and than its exact twin; '
                 f'at matched learning rates the HQ–twin difference holds at {which} but is not resolved at {other}.')
        abstract = f'and than its exact twin, at matched learning rates only at {which}'
    decision = {(True, True): 'both null hypotheses were rejected', (False, False): 'neither null hypothesis was rejected',
                (True, False): 'only the null hypothesis at 3 × 10⁻³ was rejected',
                (False, True): 'only the null hypothesis at 10⁻³ was rejected'}[(bool(L1['holm_rejects']), bool(L2['holm_rejects']))]
    jf = {k: float(np.mean([r['Jfinal'] for r in v.values()])) for k, v in
          (('HQ 3e-3', hq3), ('twin 3e-3', tw3), ('HQ 1e-3', hq1), ('twin 1e-3', tw1))}
    paragraph = (
        'Matched learning rates (post hoc). B-H2 compared HQ at 3 × 10⁻³ with its twin at 10⁻³. After the B2 results '
        'were known, and under an analysis plan written before the runs, each pairing was therefore repeated at a '
        "common learning rate: the twin was re-trained at HQ's 3 × 10⁻³ and HQ at the twin's 10⁻³, on the same 36 "
        'blocks, each new run paired with the stored B2 run of the other arm, which shares its initialisation, '
        f'training schedule and evaluation episodes. HQ − twin AUC was {res1}, and {res2} (Table 11); {decision} '
        'after Holm adjustment over the two contrasts at familywise 0.05. A machine crash '
        'interrupted the launch once; the 12 runs then in progress were restarted from the beginning, with the same '
        f'seeds, under a rule written before the restart. {reading} Descriptively, raising the learning rate from '
        f"10⁻³ to 3 × 10⁻³ changed HQ's own AUC by {ci(within['HQ 3e-3 - HQ 1e-3'])} and the twin's by "
        f"{ci(within['twin 3e-3 - twin 1e-3'])}, and mean final-checkpoint weighted J was {jf['HQ 3e-3']:.3f} (HQ) and "
        f"{jf['twin 3e-3']:.3f} (twin) at 3 × 10⁻³, and {jf['HQ 1e-3']:.3f} and {jf['twin 1e-3']:.3f} at 10⁻³.")
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': provisional,
           'L1': L1, 'L2': L2, 'within_arm_learning_rate_effects': within, 'final_J_mean': jf,
           'paragraph': paragraph, 'conclusion': concl, 'abstract_clause': abstract}
    (HERE / 'lr_results.json').write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ('L1', 'L2')}, indent=1, default=float))
    print(paragraph)


if __name__ == '__main__':
    main('--provisional' in sys.argv)
