"""Part D instruments on synthetic rows (no episode data): self-check and arm calibration run end to end.

    python test_b1g_synthetic.py
"""
import torch
import runner as r

F64 = torch.float64
g = torch.Generator().manual_seed(7)


def synth(n):
    u = torch.tanh(torch.randn(n, 4, generator=g, dtype=F64))
    mask = torch.rand(n, 27, generator=g) < 0.6
    mask[:, 0] = True
    probs = mask.double() / mask.double().sum(1, keepdim=True)
    action = torch.multinomial(probs, 1, generator=g).squeeze(1)
    return r.CRows(u=u, Wr=0.5 * torch.randn(27, 4, generator=g, dtype=F64), br=0.1 * torch.randn(27, generator=g, dtype=F64),
                   mask=mask, action=action, advantage=torch.randn(n, generator=g, dtype=F64))


rows = {s: synth(3000) for s in ('fit', 'calibration', 'heldout')}
for s in ('calibration', 'heldout'):
    rows[s].Wr, rows[s].br = rows['fit'].Wr, rows['fit'].br
rows['theta0'] = 0.5 * torch.randn(20, generator=g, dtype=F64)
print('self-check', r.self_check(rows['fit'], rows['theta0']))
out = r.classical_arms(rows)
print('diagnostics', {k: v for k, v in out['diagnostics'].items() if k != 'jacobian_singular_values_rel'})
for arm in ('C-Fisher', 'C-TAN', 'C-plain'):
    print(arm, {k: (e['status'], round(e['kl_K'], 6)) for k, e in out[arm].items()})
