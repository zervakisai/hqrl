"""Post-hoc descriptive re-analysis of stored records, written AFTER the B2F result (2026-10-25, see PLAN_LOG.md); not part
of PLAN.md and not a test of it.

HQ with frozen circuit angles is the same model, with the same gradients (agreement 1.3e-15), as the S625-twin with its
625-to-4 surrogate layer frozen. At the common learning rate 3e-3 the per-block identity
    AUC(HQ) - AUC(twin) = [AUC(HQ) - AUC(HQ frozen)] - [AUC(twin) - AUC(HQ frozen)]
therefore splits the HQ-twin difference into what training the 16 circuit angles added and what training the twin's
coefficients added over the same fixed-feature model. Inputs: HQ stored B2 (3e-3), HQ frozen (this directory, 3e-3),
twin at 3e-3 (idea_dq_posthoc_lr, arm L1, new runs paired with the stored B2 blocks). Descriptive intervals only.

    python b2f_decomposition.py      # writes b2f_decomposition.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from b2f_analysis import B2, HERE, paired, runs

LR = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_lr')


def main():
    frozen = runs(HERE / 'runs', 'F_quantum__b*', 'quantum')
    hq = runs(B2, 'quantum__b*', 'quantum')
    twin3 = runs(LR / 'runs', 'L1_surrogate625__b*', 'surrogate625')
    assert len(frozen) == len(hq) == len(twin3) == 36, (len(frozen), len(hq), len(twin3))
    angle_gain = paired(hq, frozen)          # AUC(HQ) - AUC(HQ frozen): what training the angles added
    coef_gain = paired(twin3, frozen)        # AUC(twin, 3e-3) - AUC(HQ frozen): what training the coefficients added
    total = paired(hq, twin3)                # AUC(HQ) - AUC(twin) at 3e-3 = L1 of idea_dq_posthoc_lr
    resid = max(abs(hq[b]['auc'] - twin3[b]['auc'] - ((hq[b]['auc'] - frozen[b]['auc']) - (twin3[b]['auc'] - frozen[b]['auc'])))
                for b in hq)
    lr_l1 = json.loads((LR / 'lr_results.json').read_text())
    # robustness of F1 (requested by the independent re-derivation of 2026-10-25): sign balance, and F1 without its most
    # negative block
    d1 = {b: frozen[b]['auc'] - hq[b]['auc'] for b in frozen}
    worst = min(d1, key=d1.get)
    rest = {b: v for b, v in frozen.items() if b != worst}
    f1_wo = paired(rest, {b: hq[b] for b in rest})
    robust = {'n_positive': int(sum(v > 0 for v in d1.values())), 'median': float(np.median(list(d1.values()))),
              'most_negative_block': worst, 'most_negative_value': d1[worst], 'F1_without_it': f1_wo,
              'tost_without_it': bool(-0.0056 < f1_wo['lo90'] and f1_wo['hi90'] < 0.0056)}
    out = {'status': 'post hoc, descriptive; written after the B2F result; not in PLAN.md',
           'inputs_sha256': {'b2f_results.json': hashlib.sha256((HERE / 'b2f_results.json').read_bytes()).hexdigest(),
                             'lr_results.json': hashlib.sha256((LR / 'lr_results.json').read_bytes()).hexdigest()},
           'angle_training_gain_HQ_minus_HQfrozen': angle_gain,
           'coefficient_training_gain_twin3e3_minus_HQfrozen': coef_gain,
           'HQ_minus_twin_at_3e3': total,
           'identity_max_abs_residual': resid,
           'share_blocks_coef_gain_positive': float(np.mean([twin3[b]['auc'] > frozen[b]['auc'] for b in frozen])),
           'F1_robustness': robust,
           'lr_results_L1_for_cross_check': {k: v for k, v in lr_l1.items() if 'L1' in k and isinstance(v, dict)}}
    (HERE / 'b2f_decomposition.json').write_text(json.dumps(out, indent=1, default=float))
    for k in ('angle_training_gain_HQ_minus_HQfrozen', 'coefficient_training_gain_twin3e3_minus_HQfrozen', 'HQ_minus_twin_at_3e3'):
        print(k, {q: round(out[k][q], 5) for q in ('mean', 'lo', 'hi', 'p', 'dz')})
    print('identity residual', resid, '| share coef gain > 0', out['share_blocks_coef_gain_positive'])
    print('F1 robustness', {k: v for k, v in robust.items() if k != 'F1_without_it'}, {q: round(f1_wo[q], 5) for q in ('mean', 'lo90', 'hi90', 'p')})
    print('lr_results L1 keys', list(out['lr_results_L1_for_cross_check'])[:5])


if __name__ == '__main__':
    main()
