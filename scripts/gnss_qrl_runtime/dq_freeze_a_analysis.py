"""Freeze A analysis: the stress map (registered HQ - C-matched contrasts and two interaction tests) and reset swaps.

    .venv/bin/python scripts/gnss_qrl_runtime/dq_freeze_a_analysis.py --campaign <dir> --out <new dir>

Registered (reviews 34 and 36). Units are the 12 E1 evaluation blocks; HQ and C-matched are paired by block.
A block's cell value is its mean J over the cell's 20 realisations.

    per cell       d_b(cell) = J_HQ,b(cell) - J_Cmatched,b(cell); mean, 95% t interval over blocks, for all 16 cells
                   (descriptive; 0 m and 8 m are labelled out-of-training-range generalisation)
    TESTS (target-present cells only; Holm over the two, familywise alpha 0.05; two-sided paired t over blocks)
      I-drift      slope_b = least-squares slope of d_b over D600 in {0, 2, 4, 8} (both budgets pooled, 8 points)
      I-budget     beta_b  = mean over D600 of [d_b(D, tight) - d_b(D, nominal)]
    target-absent cells: false reports per family (mean over blocks) and the HQ - C-matched difference (descriptive)
    all four families, the P3 rule and the random policy: per-cell mean J (descriptive; the rule is a systems
                   comparison: it has a handcrafted executive the learned actors lack)
    RESET SWAPS (descriptive): per family and block, reset - final on the reset suite for the weighted J (P3 cell
                   weights over its 8 cells), timely true reports, false reports, return rate and fixes attempted;
                   mean and 95% t interval over blocks.
    MISSING DATA   a block whose job for either compared family is not complete is dropped from that contrast and
                   counted; with more than one of 12 blocks dropped (> 10%) the contrast is NOT ESTIMABLE.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_runtime as dq                                                                           # noqa: E402

TOOL = 'gnss-dq-freeze-a-analysis/1'
ALPHA = 0.05
BLOCKS = tuple(range(10, 22))
D600 = (0.0, 2.0, 4.0, 8.0)
BUDGETS = ('nominal_2700s', 'tight_derived')
FAMILIES = ('quantum', 'bottleneck', 'mlp', 'surrogate625')
MAX_DROPPED = 1
P3_WEIGHT = {'positive': 0.2, 'no_human': 0.05}


def cell_id(d, budget, presence):
    return f'd{d:g}__{budget}__{presence}'


def t_summary(values):
    x = np.asarray(values, float)
    n = x.size
    if n < 3:
        return {'n': int(n), 'estimable': False}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    t = mean / se if se > 0 else 0.
    return {'n': int(n), 'estimable': True, 'mean': mean, 'sd': sd, 'ci95': [mean - q * se, mean + q * se],
            't': float(t), 'p_two_sided': float(2 * stats.t.sf(abs(t), n - 1))}


def cell_means(record, key='J'):
    out = {}
    for e in record['episodes']:
        out.setdefault(e['cell_id'], []).append(e[key] if key != 'return_ok' else float(e[key]))
    return {c: float(np.mean(v)) for c, v in out.items()}


def jobs_by_name(jobs_dir):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(jobs_dir).glob('*.json'))}


def paired_blocks(jobs, fam_a, fam_b, kind='stress'):
    kept, dropped = [], []
    for b in BLOCKS:
        ja, jb = jobs.get(f'{fam_a}__b{b}__{kind}'), jobs.get(f'{fam_b}__b{b}__{kind}')
        if ja and jb and ja['status'] == 'complete' and jb['status'] == 'complete':
            kept.append((b, cell_means(ja), cell_means(jb)))
        else:
            dropped.append(b)
    return kept, dropped


def holm(tests):
    ordered = sorted(((r['p_two_sided'], n) for n, r in tests.items() if r.get('estimable')), key=lambda x: x[0])
    decisions = {n: {'rejected': False, 'holm_threshold': None} for n in tests}
    for rank, (p, n) in enumerate(ordered):
        thr = ALPHA / (len(tests) - rank)
        decisions[n]['holm_threshold'] = thr
        if p <= thr:
            decisions[n]['rejected'] = True
        else:
            break
    return decisions


def stress_map(jobs):
    kept, dropped = paired_blocks(jobs, 'quantum', 'bottleneck')
    estimable = len(dropped) <= MAX_DROPPED
    cells = [cell_id(d, b, p) for d in D600 for b in BUDGETS for p in ('positive', 'no_human')]
    per_cell = {}
    for c in cells:
        per_cell[c] = dict(t_summary([hq[c] - cm[c] for _, hq, cm in kept]),
                           out_of_training_range=c.split('__')[0] in ('d0', 'd8'))
    slopes, betas = [], []
    for _, hq, cm in kept:
        xs, ys = [], []
        for d in D600:
            for b in BUDGETS:
                c = cell_id(d, b, 'positive')
                xs.append(d)
                ys.append(hq[c] - cm[c])
        slopes.append(float(np.polyfit(xs, ys, 1)[0]))
        betas.append(float(np.mean([(hq[cell_id(d, 'tight_derived', 'positive')] - cm[cell_id(d, 'tight_derived', 'positive')])
                                    - (hq[cell_id(d, 'nominal_2700s', 'positive')] - cm[cell_id(d, 'nominal_2700s', 'positive')])
                                    for d in D600])))
    tests = {'I-drift': t_summary(slopes) if estimable else {'estimable': False},
             'I-budget': t_summary(betas) if estimable else {'estimable': False}}
    decisions = holm(tests)
    return {'blocks_kept': [b for b, _, _ in kept], 'blocks_dropped': dropped, 'estimable': estimable,
            'hq_minus_cmatched_per_cell': per_cell,
            'interactions': {n: dict(tests[n], **decisions[n]) for n in tests},
            'note': '0 m and 8 m are outside the training range (generalisation scenarios); E1 primary estimand unchanged'}


def descriptive(jobs):
    out = {}
    for fam in FAMILIES:
        cm = [cell_means(jobs[f'{fam}__b{b}__stress']) for b in BLOCKS
              if jobs.get(f'{fam}__b{b}__stress', {}).get('status') == 'complete']
        fp = [cell_means(jobs[f'{fam}__b{b}__stress'], 'fp_spatial') for b in BLOCKS
              if jobs.get(f'{fam}__b{b}__stress', {}).get('status') == 'complete']
        if cm:
            out[fam] = {'blocks': len(cm), 'mean_J': {c: float(np.mean([m[c] for m in cm])) for c in cm[0]},
                        'mean_fp': {c: float(np.mean([m[c] for m in fp])) for c in fp[0]}}
    for name in ('p3_rule__stress', 'random__stress'):
        if jobs.get(name, {}).get('status') == 'complete':
            out[name.split('__')[0]] = {'mean_J': cell_means(jobs[name]), 'mean_fp': cell_means(jobs[name], 'fp_spatial'),
                                        'label': 'systems comparison' if name.startswith('p3') else 'uniform random'}
    return out


def weighted(means, cells):
    w = {c: P3_WEIGHT[c.split('__')[-1]] for c in cells}
    return sum(w[c] * means[c] for c in cells) / sum(w.values())


def reset_swaps(jobs):
    out = {}
    for fam in FAMILIES:
        rows, dropped = [], []
        for b in BLOCKS:
            f, r = jobs.get(f'{fam}__b{b}__reset_suite_final'), jobs.get(f'{fam}__b{b}__reset_suite_reset')
            if not (f and r and f['status'] == 'complete' and r['status'] == 'complete'):
                dropped.append(b)
                continue
            delta = {}
            cells = sorted(cell_means(f))
            delta['weighted_J'] = weighted(cell_means(r), cells) - weighted(cell_means(f), cells)
            for key in ('tp_timely', 'fp_spatial', 'return_ok', 'fixes_attempted'):
                delta[key] = float(np.mean([e[key] for e in r['episodes']]) - np.mean([e[key] for e in f['episodes']]))
            rows.append(delta)
        out[fam] = {'blocks_dropped': dropped, 'estimable': len(dropped) <= MAX_DROPPED,
                    **{k: t_summary([r[k] for r in rows]) for k in ('weighted_J', 'tp_timely', 'fp_spatial',
                                                                    'return_ok', 'fixes_attempted')},
                    'label': 'descriptive mechanism check; replication of Chen 2309.07339'}
    return out


def analyze(jobs):
    return {'tool': TOOL, 'status_counts': _counts(jobs), 'stress_map': stress_map(jobs),
            'descriptive': descriptive(jobs), 'reset_swaps': reset_swaps(jobs)}


def _counts(jobs):
    out = {}
    for j in jobs.values():
        out[j['status']] = out.get(j['status'], 0) + 1
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(jobs_by_name(args.campaign / 'jobs'))
    dq.claim_output(args.out, directory=True)
    dq.write_json(args.out / 'freeze_a_report.json', json.loads(json.dumps(report, default=float)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
