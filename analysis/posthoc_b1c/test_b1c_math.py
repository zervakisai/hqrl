"""Math checks of runner.py that need no episode data (no gain or mission value is computed).

    python test_b1c_math.py
"""
import math
import torch
import runner as r

F64 = torch.float64
torch.manual_seed(0)


def parameter_shift_hessian(spec, w, support):
    n = spec.n_weights
    H = torch.zeros(int(support.sum()), n, n, dtype=F64)
    s = math.pi / 2
    f = lambda v: r.coeffs(spec, v)[support]                                            # noqa: E731
    for i in range(n):
        for j in range(n):
            ei, ej = torch.zeros(n, dtype=F64), torch.zeros(n, dtype=F64)
            ei[i], ej[j] = s, s
            H[:, i, j] = (f(w + ei + ej) - f(w + ei - ej) - f(w - ei + ej) + f(w - ei - ej)) / 4
    return H


for circuit in ('shipped', 'heldout'):
    spec = r.b1.SPEC[circuit]
    model = r.b1.build_unit(circuit, 0)
    support = r.tp.support(spec)[0]
    w0 = r.b1.circuit_weights(model).reshape(-1).to(F64)
    Hc = r.circuit_hessian(spec, w0, support)
    Hps = parameter_shift_hessian(spec, w0, support)
    print(circuit, '|S| =', int(support.sum()), 'Hessian autograd vs parameter shift, max abs diff',
          f'{float((Hc - Hps).abs().max()):.2e}', 'max abs', f'{float(Hc.abs().max()):.2e}')
    assert float((Hc - Hps).abs().max()) < 1e-10
    inert = spec.inert_weights()
    assert float(Hc[:, list(inert), :].abs().max()) == 0. and float(Hc[:, :, list(inert)].abs().max()) == 0.
    # CTRIG chart: value, Jacobian and Hessian at the origin
    n_w, size = spec.n_weights, int(support.sum())
    A0 = r.qc.jacobian(spec, w0.reshape(spec.weight_shape))[support]
    c0 = r.coeffs(spec, w0)
    W, R = r.ctrig_draw(circuit, 0, 0, n_w, inert, size)
    assert set(W.unique().tolist()) <= {-1., 0., 1.} and float(W[:, list(inert)].abs().max()) == 0.
    s = 0.37
    chart, jac = r.ctrig_chart(c0, support, A0, W, R, s)
    f = lambda phi: chart(phi)[support]                                                 # noqa: E731
    z = torch.zeros(n_w, dtype=F64)
    assert torch.equal(chart(z), c0)
    J = torch.func.jacfwd(f)(z)
    print('  CTRIG Jacobian at 0 vs A, max abs diff', f'{float((J - A0).abs().max()):.2e}')
    assert float((J - A0).abs().max()) < 1e-12
    Hct = torch.func.jacfwd(torch.func.jacfwd(f))(z)
    Hformula = s * torch.einsum('sk,ki,kj->sij', R, W, W)
    print('  CTRIG Hessian vs formula, max abs diff', f'{float((Hct - Hformula).abs().max()):.2e}')
    assert float((Hct - Hformula).abs().max()) < 1e-10
    phi = 0.3 * torch.randn(n_w, dtype=F64)
    Jphi = torch.func.jacfwd(f)(phi)
    assert float((Jphi - jac(phi)).abs().max()) < 1e-10
    # norm formulas with a random SPD metric
    X = torch.randn(size, 2 * size, dtype=F64) / math.sqrt(2 * size)
    Fm = X @ X.T
    G_R = R.T @ Fm @ R
    explicit = float(torch.einsum('sij,st,tij->', Hformula / s, Fm, Hformula / s))
    print('  ctrig_norm formula rel err', f'{abs(r.ctrig_norm(W, G_R) - explicit) / explicit:.2e}')
    assert abs(r.ctrig_norm(W, G_R) - explicit) / explicit < 1e-10
    u = torch.randn(n_w, dtype=F64); u[list(inert)] = 0.; u /= u.norm()
    huu = torch.einsum('sij,i,j->s', Hformula / s, u, u)
    assert abs(r.ctrig_along(W, G_R, u) - float(huu @ Fm @ huu)) / float(huu @ Fm @ huu) < 1e-10
    # periodicity in each weight (degree <= 1 trigonometric)
    e = torch.zeros(n_w, dtype=F64); e[0] = 2 * math.pi
    assert float((chart(phi + e) - chart(phi)).abs().max()) < 1e-10
print('all math checks passed')
