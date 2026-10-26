"""Post-hoc check, written after the B2F result (see PLAN_LOG.md): HQ with frozen circuit angles is the same model, with the
same gradients, as the S625-twin with its Fourier coefficients (actor.surrogate) frozen. Synthetic inputs only; no
environment episode is run.

For three blocks: both models are built from the block's B2 initialisation; forward logits and values, and gradients of a
scalar of both, are compared on random recurrent inputs. Writes b2f_identity_check.json.
"""
import json
import sys
from pathlib import Path

IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))
import torch  # noqa: E402

import dq_b2  # noqa: E402
import dq_transplant as tp  # noqa: E402
import p4_train as p4  # noqa: E402

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


def check(block):
    cfg = dq_b2.config_for('quantum', block, dq_b2.STUDY_B2, freeze_sha256='d' * 64)
    hq = p4.build_model('quantum', cfg, block)
    hq.actor.weights.requires_grad_(False)
    twin, _ = tp.make_twin(p4.build_model('quantum', cfg, block))
    twin.train()
    for p in twin.actor.surrogate.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 9, 254, dtype=torch.float64, generator=torch.Generator().manual_seed(100 + block))
    grads, fwd = [], []
    for model in (hq, twin):
        model.zero_grad()
        logits, v, _ = model(x)
        fwd.append((logits.detach(), v.detach()))
        (logits.log_softmax(-1).mean() + logits.log_softmax(-1)[..., 3].mean() + v.mean()).backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    shared = sorted(set(grads[0]) & set(grads[1]))
    return {'block': block, 'shared_trainable_tensors': len(shared), 'only_in_one': sorted(set(grads[0]) ^ set(grads[1])),
            'trainable_params_hq_frozen': sum(p.numel() for p in hq.parameters() if p.requires_grad),
            'trainable_params_twin_frozen': sum(p.numel() for p in twin.parameters() if p.requires_grad),
            'max_abs_logit_diff': float((fwd[0][0] - fwd[1][0]).abs().max()),
            'max_abs_value_diff': float((fwd[0][1] - fwd[1][1]).abs().max()),
            'max_abs_grad_diff': float(max((grads[0][k] - grads[1][k]).abs().max() for k in shared)),
            'max_abs_grad': float(max(grads[0][k].abs().max() for k in shared))}


if __name__ == '__main__':
    out = [check(b) for b in (0, 7, 21)]
    (HERE / 'b2f_identity_check.json').write_text(json.dumps(out, indent=1))
    for r in out:
        print(r)
