"""POST-HOC DESCRIPTIVE analyses, version 2 (Review 45, R45-3/R45-4/R45-5). Read-only; no frozen test, no frozen
decision changes. posthoc_v1 is kept unchanged as history.

Changes against v1:
  * B2 bootstrap preserves the common-random-number pairing: one resampled list of realisation indices per
    drift/budget pair ID, shared by its positive and no-human cells and by all arms, blocks and checkpoints;
    training blocks are resampled separately (R45-3). Seed 20260924, generator reset per contrast.
  * B2 learning-gain summaries: difference and relative difference of the arms' own AUC-minus-J0 gains, and the
    v1 ratio renamed to what it is (R45-4). Actual PPO update counts per run (R45-4).
  * B1 arm ladder with per-entry n and 95 % t CI for every arm and kappa (R45-5).

    python analysis/posthoc/posthoc_descriptives_v2.py

Writes posthoc_descriptives_v2.json next to this file (refuses to overwrite).
"""
from __future__ import annotations

import collections
import glob
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V1 = HERE
V1_JSON_SHA = '9886768db6b0cde9504c3d120d73eab6f5229690bd61460a56dddde637c97b44'
spec = importlib.util.spec_from_file_location('ph1', V1 / 'posthoc_descriptives.py')
ph1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph1)
DQ = ph1.DQ
SEED = 20260924
N_BOOT = 2000


def b1_ladder():
    units = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(DQ / 'b1_campaign_v1/units/*.json')))]
    out = {}
    for circ in ('shipped', 'heldout'):
        us = sorted([u for u in units if u['circuit'] == circ], key=lambda u: u['index'])
        out[circ] = {k: {a: ph1.t_ci([ph1.gain(u, a, k) for u in us]) for a in ph1.B1_ARMS} for k in ph1.KAPPAS}
    return out


def b2():
    sys.path.insert(0, str(ph1.REPO / 'scripts' / 'gnss_qrl_runtime'))
    import dq_b2
    curves = ph1.b2_curves()
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    weights = {c['cell_id']: w for c, w in zip(rt.cells, rt.probabilities)}
    pairs = {c['cell_id']: c['pair_id'] for c in rt.cells}
    keys = sorted(next(iter(curves.values()))[0])
    cells = sorted({c for c, _ in keys})
    by = {c: [k for k in keys if k[0] == c] for c in cells}
    n_real = {len(v) for v in by.values()}
    assert len(n_real) == 1
    n_real = n_real.pop()
    arms = ['quantum', 'bottleneck', 'surrogate625']
    blocks = sorted({b for _, b in curves})
    grid = dq_b2.GRID
    X = np.empty((3, len(blocks), len(grid), len(cells), n_real))
    for ai, arm in enumerate(arms):
        for bi, block in enumerate(blocks):
            for pi, point in enumerate(grid):
                X[ai, bi, pi] = [[curves[arm, block][point][k] for k in by[c]] for c in cells]
    W = np.array([weights[c] for c in cells])
    E = np.trapezoid(X, x=grid, axis=2) / grid[-1]          # per arm, block, cell, realisation
    A = (E.mean(-1) * W).sum(-1)
    J0 = (X[:, :, 0].mean(-1) * W).sum(-1)
    frozen = json.loads((DQ / 'b2_analysis_v1/b2_report.json').read_text())
    auc_diff = max(float(np.max(np.abs(A[i] - frozen['per_arm_auc_recomputed'][a]))) for i, a in enumerate(arms))
    pair_ids = sorted(set(pairs.values()))
    gain = (A - J0).mean(1)
    out = {'pair_mapping': pairs, 'auc_max_abs_difference_vs_frozen': auc_diff,
           'mean_J0': dict(zip(arms, J0.mean(1).tolist())), 'mean_learning_gain_auc_minus_J0': dict(zip(arms, gain.tolist()))}
    for ai, name in ((1, 'B-H1'), (2, 'B-H2')):
        d, D = A[0] - A[ai], E[0] - E[ai]
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            bs = rng.integers(0, len(blocks), len(blocks))
            ix = {p: rng.integers(0, n_real, n_real) for p in pair_ids}
            boot.append(sum(W[ci] * D[bs][:, ci][:, ix[pairs[c]]].mean() for ci, c in enumerate(cells)))
        out[name] = {'mean': float(d.mean()), 'RE_per_episode_time_averaged': float(18 * d.mean()),
                     'pair_preserving_bootstrap_block_x_realisation_95': np.percentile(boot, [2.5, 97.5]).tolist(),
                     'difference_in_learning_gains': float(gain[0] - gain[ai]),
                     'relative_difference_in_learning_gains': float(1 - gain[0] / gain[ai]),
                     'auc_gap_over_comparator_auc_minus_J0': float(-d.mean() / gain[ai]),
                     'excluding_blocks_0_4': ph1.t_ci(d[5:])}
    recs = [json.loads(p.read_text()) for p in (DQ / 'b2_campaign_v3/records').glob('*.json')]
    out['update_counts'] = {str(k): v for k, v in sorted(collections.Counter(r['summary']['updates'] for r in recs).items())}
    out['update_counts_by_arm'] = {a: {str(k): v for k, v in sorted(collections.Counter(
        r['summary']['updates'] for r in recs if r['arm'] == a).items())} for a in arms}
    out['update_median'] = float(np.median([r['summary']['updates'] for r in recs]))
    return out


def main():
    for path, digest in ph1.REPORTS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f'{path} is not the report of record')
    if hashlib.sha256((V1 / 'posthoc_descriptives.json').read_bytes()).hexdigest() != V1_JSON_SHA:
        raise SystemExit('posthoc_v1 output changed')
    target = HERE / 'posthoc_descriptives_v2.json'
    if target.exists():
        raise SystemExit(f'{target} exists; never overwritten')
    result = {'label': 'POST-HOC DESCRIPTIVE v2 (Review 45); not a frozen test; no frozen decision changes',
              'B1_arm_ladder_with_n': b1_ladder(), 'B2': b2()}
    target.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print('written', target)
    print(json.dumps({k: v for k, v in result['B2'].items() if k != 'pair_mapping'}, indent=1, default=float))


if __name__ == '__main__':
    main()
