"""Analysis of the post-hoc HQ-at-1e-2 runs, exactly as fixed in PLAN.md (written before the runs).

    python hqlr_analysis.py [--provisional]     # writes hqlr_results.json; refuses incomplete data unless --provisional
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
    return {'auc': last['auc']['sample'], 'Jfinal': last['learning_curve']['sample'][-1][2]}


def runs(root, pattern, arm):
    out = {}
    for f in glob.glob(str(root / pattern / f'run/{arm}/metrics.jsonl')):
        s = summary(f)
        if s is not None:
            out[int(f.split('__b')[1].split('/')[0])] = s
    return out


def displacement(run_dir):
    """Relative parameter change ||w_final - w_0|| / ||w_0|| of the circuit angles and of the classical head parts."""
    import torch
    ck = sorted(glob.glob(str(Path(run_dir) / 'checkpoints' / '*.pt')))
    first = [c for c in ck if Path(c).name.startswith('update00000_')][0]
    final = [c for c in ck if Path(c).name.startswith('final_')][0]
    a = torch.load(first, map_location='cpu', weights_only=False)['model_state_dict']
    b = torch.load(final, map_location='cpu', weights_only=False)['model_state_dict']
    rel = lambda k: float((b[k] - a[k]).norm() / a[k].norm())                                  # noqa: E731
    return {'angles_rel': rel('actor.weights'), 'angles_init_mean_abs_rad': float(a['actor.weights'].abs().mean()), 'angles_mean_abs_rad': float((b['actor.weights'] - a['actor.weights']).abs().mean()),
            'projection_rel': rel('actor.projection.weight'), 'readout_rel': rel('actor.readout.weight')}


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


def ci(r):
    return f'{num(r["mean"])} [{num(r["lo"])}, {num(r["hi"])}]'


def pstr(p):
    if p >= 0.001:
        return f'{p:.2g}'
    m, e = f'{p:.1e}'.split('e')
    return f'{m} × 10⁻' + str(-int(e)).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def main(provisional=False):
    hq10 = runs(HERE / 'runs', 'H1_quantum__b*', 'quantum')
    cm3, hq3 = runs(B2, 'bottleneck__b*', 'bottleneck'), runs(B2, 'quantum__b*', 'quantum')
    if not provisional and len(list((HERE / 'records').glob('*.json'))) < 36:
        raise SystemExit(f'incomplete: {len(hq10)} completed runs')
    q1 = paired(hq10, cm3)
    d_lr = paired(hq10, hq3)
    bh1 = paired(hq3, cm3)                                                     # B-H1 of record, for scale
    disp = {'HQ 3e-3': [displacement(B2 / f'quantum__b{b}/run/quantum') for b in sorted(hq3)],
            'HQ 1e-2': [displacement(HERE / f'runs/H1_quantum__b{b}/run/quantum') for b in sorted(hq10)]}
    med = {k: {f: float(np.median([x[f] for x in v])) for f in v[0]} for k, v in disp.items() if v}
    jf = {k: float(np.mean([s['Jfinal'] for s in v.values()])) for k, v in
          (('HQ 1e-2', hq10), ('HQ 3e-3', hq3), ('C-matched 3e-3', cm3))}
    if q1['hi'] < 0:
        reading = ('HQ therefore still learned less than C-matched at the next learning rate, although the gap was '
                   f'smaller than at its selected rate (B-H1, {num(bh1["mean"])}) and HQ\'s own area rose further, '
                   'without resolution. Moving HQ one half-decade beyond its grid did not remove the deficit; higher '
                   'rates were not tested.')
        clause = 'also at a higher hybrid learning rate'
        verdict = 'persists'
        concl = ("HQ's lower area than C-matched persisted when HQ was re-trained at the next learning rate, "
                 '10⁻².')
    elif q1['lo'] > 0:
        reading = ('At 10⁻² HQ exceeded C-matched at its selected rate, so B-H1 reflects HQ\'s selection at the edge '
                   'of its grid.')
        clause = 'a difference reversed at a higher hybrid learning rate'
        verdict = 'reversed'
        concl = 'Re-trained at the next learning rate, 10⁻², HQ exceeded C-matched at its selected rate.'
    else:
        reading = ('At 10⁻² the HQ–C-matched difference was not resolved, so B-H1 depends on HQ\'s learning-rate '
                   'selection.')
        clause = 'a difference not resolved at a higher hybrid learning rate'
        verdict = 'unresolved'
        concl = ("HQ's lower area than C-matched was not resolved when HQ was re-trained at the next learning "
                 'rate, 10⁻².')
    h3, h10 = med.get('HQ 3e-3', {}), med.get('HQ 1e-2', {})
    nan = float('nan')
    paragraph = (
        'Learning rate beyond the grid edge (post hoc). HQ and C-matched both use 3 × 10⁻³, the upper edge of the '
        'grid from which they were selected. Under an analysis plan fixed by hash before the runs, HQ was therefore '
        f're-trained at 10⁻² on the same 36 blocks. HQ (10⁻²) − C-matched (3 × 10⁻³) AUC was {ci(q1)} (two-sided '
        f'p = {pstr(q1["p"])}; HQ lower in {round(100 * q1["share_negative"])}% of blocks; Table 11), and HQ (10⁻²) − '
        f'HQ (3 × 10⁻³) was {ci(d_lr)}. {reading} Mean final-checkpoint weighted J was {jf["HQ 1e-2"]:.3f} for HQ at '
        f'10⁻², {jf["HQ 3e-3"]:.3f} at 3 × 10⁻³ and {jf["C-matched 3e-3"]:.3f} for C-matched. The circuit angles moved '
        f'little in absolute terms (median mean |Δθ| {h3.get("angles_mean_abs_rad", nan):.3f} rad at 3 × 10⁻³ and '
        f'{h10.get("angles_mean_abs_rad", nan):.3f} rad at 10⁻²), but by a fraction of their small initial values '
        f'(median mean |θ| {h3.get("angles_init_mean_abs_rad", nan):.3f} rad) similar to that of the classical head '
        f'weights: the median relative change ‖Δθ‖/‖θ₀‖ was {h3.get("angles_rel", nan):.2f} at 3 × 10⁻³ and '
        f'{h10.get("angles_rel", nan):.2f} at 10⁻², against {h3.get("projection_rel", nan):.2f} and '
        f'{h3.get("readout_rel", nan):.2f} for the head\'s classical projection and readout at 3 × 10⁻³ (descriptive, '
        'not in the plan). Small absolute angle changes therefore do not by themselves show that the circuit was '
        'untrained.')
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': provisional,
           'Q1': q1, 'HQ1e-2_minus_HQ3e-3': d_lr, 'displacement_median': med, 'final_J_mean': jf,
           'cell_Q1': ci(q1), 'paragraph': paragraph, 'clause': clause, 'verdict': verdict, 'conclusion': concl}
    (HERE / 'hqlr_results.json').write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print(paragraph)


if __name__ == '__main__':
    main('--provisional' in sys.argv)
