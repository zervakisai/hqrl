"""POST-HOC DESCRIPTIVE analyses requested by the manuscript review (24 Sep 2026). Read-only; none of this is a frozen
test, and nothing here changes a frozen decision. Every quantity is computed from stored outputs of record:

    B1 units   idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units/*.json   (B1 report cc33c88b...)
    B2 runs    output/study/gnss_qrl_v1/dq/b2_campaign_v3/{records,runs}  (B2 report 0ae91cea...)

    python analysis/posthoc/posthoc_descriptives.py

Writes posthoc_descriptives.json next to this file (refuses to overwrite).
"""
from __future__ import annotations

import glob
import hashlib
import json
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DQ = REPO / 'output' / 'study' / 'gnss_qrl_v1' / 'dq'
REPORTS = {DQ / 'b1_analysis_v1/b1_report.json': 'cc33c88b135b0d14f622b9bc0396891e394152da53ae2f80e0135f37ca02000c',
           DQ / 'b2_analysis_v1/b2_report.json': '0ae91cea336c8a0409a7249b8d291d5355d9f6d47a1a44c362908609f18d40be'}
N_REF = 18
KAPPAS = ('0.001', '0.01', '0.05')
B1_ARMS = ('HQ-Fisher', 'HQ-plain', 'HQ-Adam-dir', 'TAN', 'RND-k', 'RND-48', 'RND-192', 'RND-spec-plain', 'SPAN', 'ORTH',
           'FIT-k', 'FIT-1')
rng = np.random.default_rng(20260924)


def t_ci(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = x.size
    if n < 3:
        return {'n': int(n)}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    q = stats.t.ppf(0.975, n - 1)
    se = sd / sqrt(n)
    return {'n': int(n), 'mean': m, 'ci95': [m - q * se, m + q * se], 'positive': int((x > 0).sum()),
            'p_two_sided': float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else None}


def gain(u, arm, k):
    if arm in ('RND-k', 'RND-48', 'RND-192'):
        vals = [u['instruments']['arms'][f'{arm}#{r}'][k].get('G_H') for r in range(20)]
        return np.nan if any(v is None for v in vals) else float(np.mean(vals))
    e = u['instruments']['arms'][arm][k]
    return e.get('G_H', np.nan) if e['status'] == 'ok' else np.nan


def b1():
    units = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(DQ / 'b1_campaign_v1/units/*.json')))]
    out = {}
    for circ in ('shipped', 'heldout'):
        us = sorted([u for u in units if u['circuit'] == circ], key=lambda u: u['index'])
        ladder = {k: {a: t_ci([gain(u, a, k) for u in us]) for a in B1_ARMS} for k in KAPPAS}
        contrasts = {}
        for k in KAPPAS:
            hq, tan, rnd, span = (np.array([gain(u, a, k) for u in us]) for a in ('HQ-Fisher', 'TAN', 'RND-k', 'SPAN'))
            contrasts[k] = {'HQ-Fisher_minus_TAN': t_ci(hq - tan), 'TAN_minus_RND-k': t_ci(tan - rnd),
                            'SPAN_minus_TAN': t_ci(span - tan), 'HQ-Fisher_minus_RND-k': t_ci(hq - rnd),
                            'SPAN_minus_HQ-Fisher': t_ci(span - hq)}
        quad = {k: {a: float(np.nanmedian([u['instruments']['arms'][a][k].get('quadratic_rel_error', np.nan)
                                            for u in us])) for a in ('HQ-Fisher', 'HQ-plain', 'HQ-Adam-dir', 'TAN', 'SPAN',
                                                                     'RND-k#0')} for k in KAPPAS}
        klH = {a: float(np.mean([u['instruments']['arms'][a]['0.01']['kl_H'] for u in us]))
               for a in ('HQ-Fisher', 'TAN', 'SPAN', 'RND-k#0')}
        dec = [np.mean([r['decisions'] for r in u['episode_records']['heldout']]) for u in us]
        e_units = [u for u in us if 'fresh_eval' in u]
        dj = {a: np.array([u['fresh_eval']['delta_J'][a] for u in e_units]) for a in ('HQ-Fisher', 'RND-k#0', 'FIT-k', 'SPAN')}
        e_pairs = {'SPAN_minus_HQ-Fisher': dj['SPAN'] - dj['HQ-Fisher'], 'HQ-Fisher_minus_RND-k#0': dj['HQ-Fisher'] - dj['RND-k#0'],
                   'SPAN_minus_RND-k#0': dj['SPAN'] - dj['RND-k#0']}
        e_contrasts = {}
        for name, v in e_pairs.items():
            s = t_ci(v)
            s['report_equivalents'] = {'mean': s['mean'] * N_REF, 'ci95': [c * N_REF for c in s['ci95']]}
            e_contrasts[name] = s
        out[circ] = {'arm_ladder': ladder, 'contrasts': contrasts, 'median_quadratic_rel_error': quad,
                     'mean_kl_on_H_at_1e-2': klH, 'mean_decisions_per_heldout_episode': float(np.mean(dec)),
                     'fisher_condition_tangent_median': float(np.median([u['instruments']['diagnostics']['fisher_condition_tangent'] for u in us])),
                     'hq_fisher_alpha_median_1e-2': float(np.median([u['instruments']['arms']['HQ-Fisher']['0.01']['alpha'] for u in us])),
                     'fresh_eval_paired_contrasts': e_contrasts}
    mh = {c: [u['instruments']['arms']['SPAN']['0.01']['G_H'] - u['instruments']['arms']['HQ-Fisher']['0.01']['G_H']
              for u in sorted([u for u in units if u['circuit'] == c], key=lambda u: u['index'])] for c in ('shipped', 'heldout')}
    out['cross_stratum_spearman_M-H1'] = float(stats.spearmanr(mh['shipped'], mh['heldout']).statistic)
    return out


def b2_curves():
    """{(arm, block): {point: {(cell, seed): J}}} from the raw evaluation events."""
    curves = {}
    for rec_path in sorted(glob.glob(str(DQ / 'b2_campaign_v3/records/*.json'))):
        rec = json.loads(Path(rec_path).read_text())
        arm, block = rec['arm'], rec['block']
        path = DQ / 'b2_campaign_v3/runs' / f'{arm}__b{block}' / 'run' / arm / 'metrics.jsonl'
        per_point = {}
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e.get('kind') == 'evaluation':
                eps = {(x['cell_id'], x['seed']): x['J'] for x in e['results']['sample']['episodes']}
                for p in e['points']:
                    per_point[p] = eps
        curves[(arm, block)] = per_point
    return curves


def b2(curves):
    import sys
    sys.path.insert(0, str(REPO / 'scripts' / 'gnss_qrl_runtime'))
    import dq_b2
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    weight = {c['cell_id']: p for c, p in zip(rt.cells, rt.probabilities)}
    grid = dq_b2.GRID
    keys = sorted(next(iter(curves.values()))[0])                      # (cell, seed) evaluation episodes
    cells = sorted({c for c, _ in keys})
    by_cell = {c: [k for k in keys if k[0] == c] for c in cells}

    def auc(arm, block, episode_index=None):
        vals = []
        for p in grid:
            eps = curves[(arm, block)][p]
            j = 0.
            for c in cells:
                ks = by_cell[c] if episode_index is None else [by_cell[c][i] for i in episode_index[c]]
                j += weight[c] * np.mean([eps[k] for k in ks])
            vals.append(j)
        return float(sum((b1 - b0) * (v1 + v0) / 2 for (b0, v0), (b1, v1) in zip(zip(grid, vals), zip(grid[1:], vals[1:])))) / grid[-1], vals

    blocks = sorted({b for _, b in curves})
    A = {a: {b: auc(a, b) for b in blocks} for a in ('quantum', 'bottleneck', 'surrogate625')}
    out = {'report_equivalent': '1/18 of J; AUC differences x 18 = report-equivalents per episode (time-averaged)'}
    for name, other in (('B-H1', 'bottleneck'), ('B-H2', 'surrogate625')):
        d = np.array([A['quantum'][b][0] - A[other][b][0] for b in blocks])
        gain_other = np.mean([A[other][b][0] - A[other][b][1][0] for b in blocks])
        gain_hq = np.mean([A['quantum'][b][0] - A['quantum'][b][1][0] for b in blocks])
        s = t_ci(d)
        s['report_equivalents'] = {'mean': s['mean'] * N_REF, 'ci95': [c * N_REF for c in s['ci95']]}
        s['fraction_of_comparator_learning_gain'] = s['mean'] / gain_other
        s['comparator_learning_gain_auc_minus_J0'] = gain_other
        s['hq_learning_gain_auc_minus_J0'] = gain_hq
        s['excluding_blocks_0_4'] = t_ci(d[5:])
        # crossed bootstrap: resample blocks AND evaluation realisations (within each cell), 2000 replicates
        boot = []
        for _ in range(2000):
            bs = rng.integers(0, len(blocks), len(blocks))
            idx = {c: rng.integers(0, len(by_cell[c]), len(by_cell[c])) for c in cells}
            boot.append(np.mean([auc('quantum', blocks[i], idx)[0] - auc(other, blocks[i], idx)[0] for i in bs]))
        s['crossed_bootstrap_block_x_episode_95'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out[name] = s
    kl = {}
    for a in ('quantum', 'bottleneck', 'surrogate625'):
        vals = []
        for rec_path in glob.glob(str(DQ / f'b2_campaign_v3/runs/{a}__b*/run/{a}/metrics.jsonl')):
            vals += [json.loads(l)['metrics']['approx_kl'] for l in open(rec_path) if '"kind": "update"' in l]
        kl[a] = {'median': float(np.median(vals)), 'iqr': [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]}
    out['approx_kl_per_update'] = kl
    out['mean_auc'] = {a: float(np.mean([A[a][b][0] for b in blocks])) for a in A}
    out['mean_J0'] = {a: float(np.mean([A[a][b][1][0] for b in blocks])) for a in A}
    out['mean_J_final'] = {a: float(np.mean([A[a][b][1][-1] for b in blocks])) for a in A}
    out['auc_reproduces_frozen'] = None
    return out


def main():
    for path, digest in REPORTS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f'{path} is not the report of record')
    target = HERE / 'posthoc_descriptives.json'
    if target.exists():
        raise SystemExit(f'{target} exists; never overwritten')
    result = {'label': 'POST-HOC DESCRIPTIVE; not a frozen test; no frozen decision changes', 'B1': b1()}
    curves = b2_curves()
    result['B2'] = b2(curves)
    frozen = json.loads((DQ / 'b2_analysis_v1/b2_report.json').read_text())
    recomputed = [result['B2']['B-H1']['mean'], result['B2']['B-H2']['mean']]
    result['B2']['auc_reproduces_frozen'] = bool(abs(recomputed[0] - frozen['primary']['B-H1']['mean']) < 1e-12 and
                                                 abs(recomputed[1] - frozen['primary']['B-H2']['mean']) < 1e-12)
    target.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print('written', target, 'reproduces frozen B2 means:', result['B2']['auc_reproduces_frozen'])


if __name__ == '__main__':
    main()
