"""Instrument correctness on synthetic rows (development check D-2 and D-4 of the DQ protocol)."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402

F64 = torch.float64
A_DIM = 9


@pytest.fixture(scope='module')
def setup():
    torch.manual_seed(0)
    spec = qc.SHIPPED
    support, _ = tp.support(spec)
    weights = 0.4 * torch.randn(*spec.weight_shape, dtype=F64)
    c = qc.coefficients(spec, weights).reshape(-1)
    rows = synthetic_rows(spec, n=300, seed=1)
    return spec, support, weights, c, rows


def synthetic_rows(spec, n, seed, episodes=12):
    g = torch.Generator().manual_seed(seed)
    angles = (torch.rand(n, spec.wires, generator=g, dtype=F64) * 2 - 1) * math.pi
    phi = qc.basis(spec, angles)
    W = torch.randn(A_DIM, spec.wires, generator=g, dtype=F64)
    b = 0.3 * torch.randn(A_DIM, generator=g, dtype=F64)
    mask = torch.rand(n, A_DIM, generator=g) > 0.35
    mask[:, 0] = True
    action = torch.stack([torch.multinomial(m.double(), 1, generator=g)[0] for m in mask])
    advantage = torch.randn(n, generator=g, dtype=F64)
    episode = torch.arange(n) % episodes
    return ins.Rows(phi=phi, W=W, b=b, mask=mask, action=action, advantage=advantage, episode=episode)


def test_gradient_equals_autograd(setup):
    spec, support, _, c, rows = setup
    c_s = c[support].clone().requires_grad_(True)
    full = ins.full(c_s, support, torch.zeros_like(c))
    lp = ins.log_policy(rows, full)[torch.arange(rows.n), rows.action]
    (lp @ rows.advantage / rows.n).backward()
    assert float((ins.gradient(rows, c, support) - c_s.grad).abs().max()) <= 1e-13


def test_fisher_is_the_hessian_of_the_mean_kl_and_matrix_free_agrees(setup):
    spec, support, _, c, rows = setup
    F = ins.fisher(rows, c, support)
    assert torch.allclose(F, F.T) and float(torch.linalg.eigvalsh(F).min()) >= -1e-12
    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        v = torch.randn(int(support.sum()), generator=g, dtype=F64)
        assert float((ins.fisher_vector(rows, c, support, v) - F @ v).abs().max()) <= 1e-12 * max(1., float(F.abs().max()))
        eps = 1e-4
        kl = ins.mean_kl(rows, c, ins.full(c[support] + eps * v, support, c))
        quad = 0.5 * eps ** 2 * float(v @ F @ v)
        assert abs(kl - quad) <= 1e-3 * quad


def test_fisher_is_invariant_to_chunking(setup, monkeypatch):
    spec, support, _, c, rows = setup
    F = ins.fisher(rows, c, support)
    monkeypatch.setattr(ins, 'FISHER_CHUNK', 7)
    assert float((ins.fisher(rows, c, support) - F).abs().max()) <= 1e-12


def test_theta_pullback_equals_autograd_through_the_circuit(setup):
    spec, support, weights, c, rows = setup
    A = qc.jacobian(spec, weights)[support]
    g_theta, F_theta = ins.theta_quantities(A, ins.gradient(rows, c, support), ins.fisher(rows, c, support))
    w = weights.clone().reshape(-1).requires_grad_(True)
    full = qc.coefficients(spec, w.reshape(spec.weight_shape)).reshape(-1)
    lp = ins.log_policy(rows, full)[torch.arange(rows.n), rows.action]
    (lp @ rows.advantage / rows.n).backward()
    assert float((g_theta - w.grad).abs().max()) <= 1e-12
    for inert in spec.inert_weights():
        assert abs(float(g_theta[inert])) <= 1e-13 and float(F_theta[inert].abs().max()) <= 1e-12


def test_natural_step_depends_only_on_the_span(setup):
    spec, support, _, c, rows = setup
    F, g = ins.fisher(rows, c, support), ins.gradient(rows, c, support)
    gen = torch.Generator().manual_seed(4)
    B = ins.haar_subspace(int(support.sum()), 12, gen)
    R = torch.randn(12, 12, generator=gen, dtype=F64) + 3 * torch.eye(12, dtype=F64)
    d1, _ = ins.natural_step(B, g, F, 0.)
    d2, _ = ins.natural_step(B @ R, g, F, 0.)
    assert float((d1 - d2).abs().max()) <= 1e-8 * float(d1.abs().max())
    # q_B is at most q_S (no damping), and the span step attains q_S
    q_B = ins.accessible(B, g, F, 1e-10)
    d_span, _ = ins.span_step(g, F, 1e-10)
    assert q_B <= float(g @ d_span) * (1 + 1e-8)


def test_orthogonal_complement_and_tangent_basis(setup):
    spec, support, weights, c, rows = setup
    A = qc.jacobian(spec, weights)[support]
    k, s, ambiguous = ins.chart_rank(A, 1e-6)
    assert k == 12 and not ambiguous
    T = ins.tangent_basis(A, k)
    O = ins.orthogonal_complement(T)
    n = int(support.sum())
    assert O.shape == (n, n - 12)
    assert float((T.T @ O).abs().max()) <= 1e-12
    assert float((O.T @ O - torch.eye(n - 12, dtype=F64)).abs().max()) <= 1e-12


def test_rank_ambiguity_band_flags_a_singular_value_near_the_threshold():
    U, _ = torch.linalg.qr(torch.randn(50, 5, dtype=F64))
    s = torch.tensor([1., 0.5, 0.1, 3e-6, 1e-15], dtype=F64)
    k, _, ambiguous = ins.chart_rank(U * s, 1e-6)
    assert k == 4 and ambiguous
    k, _, ambiguous = ins.chart_rank(U * torch.tensor([1., 0.5, 0.1, 0.04, 1e-15], dtype=F64), 1e-6)
    assert k == 4 and not ambiguous


def test_haar_subspace_is_orthonormal_and_reproducible():
    a = ins.haar_subspace(40, 6, torch.Generator().manual_seed(9))
    b = ins.haar_subspace(40, 6, torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    assert float((a.T @ a - torch.eye(6, dtype=F64)).abs().max()) <= 1e-13


def test_fit_k_recovers_a_planted_subspace():
    n, k, E = 60, 4, 200
    gen = torch.Generator().manual_seed(11)
    planted = ins.haar_subspace(n, k, gen)
    U = (torch.randn(E, k, generator=gen, dtype=F64) * torch.tensor([5., 4., 3., 2.], dtype=F64)) @ planted.T
    U += 1e-3 * torch.randn(E, n, generator=gen, dtype=F64)
    Q, mu, record = ins.fit_subspace(U, torch.eye(n, dtype=F64), k, 1e-12)
    # principal angles between span(Q) and the planted span are ~0
    cosines = torch.linalg.svdvals(Q.T @ planted)
    assert float(cosines.min()) >= 1 - 1e-5
    assert record['estimable'] and not record['fit_tie']
    B1, _ = ins.fit_one(U, torch.eye(n, dtype=F64), 1e-12)
    assert B1.shape == (n, 1) and abs(float(B1.norm()) - 1) <= 1e-12


def test_fit_k_needs_enough_episodes():
    U = torch.randn(3, 20, dtype=F64)
    with pytest.raises(ins.InstrumentError, match='episodes'):
        ins.fit_subspace(U, torch.eye(20, dtype=F64), 5, 1e-6)
    _, _, record = ins.fit_subspace(torch.randn(6, 20, dtype=F64), torch.eye(20, dtype=F64), 4, 1e-6)
    assert record['estimable'] is False


def test_calibration_hits_the_target_within_tolerance_on_linear_and_manifold_rays(setup):
    spec, support, weights, c, rows = setup
    F, g = ins.fisher(rows, c, support), ins.gradient(rows, c, support)
    d, _ = ins.span_step(g, F, 1e-4)
    for target in (1e-3, 1e-2, 5e-2):
        cal = ins.calibrate_kl(lambda a: ins.full(c[support] + a * d, support, c), c, rows, float(d @ F @ d), target)
        assert cal.status == 'ok' and abs(cal.kl - target) <= 1e-3 * target
    A = qc.jacobian(spec, weights)[support]
    g_t, F_t = ins.theta_quantities(A, g, F)
    dt, _ = ins.theta_natural(g_t, F_t, 1e-4)
    ray = lambda a: qc.coefficients(spec, weights + a * dt.reshape(spec.weight_shape)).reshape(-1)  # noqa: E731
    cal = ins.calibrate_kl(ray, c, rows, float(dt @ F_t @ dt), 1e-2)
    assert cal.status == 'ok' and abs(cal.kl - 1e-2) <= 1e-5


def test_calibration_status_paths_are_explicit(setup):
    spec, support, _, c, rows = setup
    F, g = ins.fisher(rows, c, support), ins.gradient(rows, c, support)
    d, _ = ins.span_step(g, F, 1e-4)
    ray = lambda a: ins.full(c[support] + a * d, support, c)             # noqa: E731
    curv = float(d @ F @ d)
    assert ins.calibrate_kl(ray, c, rows, 0., 1e-2).status == 'null_direction'
    assert ins.calibrate_kl(ray, c, rows, curv * 1e12, 1e-2, max_doublings=3).status == 'unattained'
    assert ins.calibrate_kl(ray, c, rows, curv * 1e6, 1e-2, max_bisections=1).status == 'no_convergence'


def test_heldout_gain_is_zero_at_the_old_policy_and_first_order_matches_the_gradient(setup):
    spec, support, _, c, rows = setup
    assert ins.heldout_gain(rows, c, c) == 0.
    g = ins.gradient(rows, c, support)
    eps = 1e-6
    gain = ins.heldout_gain(rows, c, ins.full(c[support] + eps * g, support, c))
    assert abs(gain - eps * float(g @ g)) <= 1e-4 * eps * float(g @ g)


def test_mc_advantages_leave_one_episode_out_per_cell():
    rewards = [1., 0., 2., 0., 3., 1.]
    episode = [0, 0, 1, 1, 2, 2]
    cell = ['a', 'a', 'a', 'a', 'b', 'b']
    adv, fallback = ins.mc_advantages(rewards, episode, cell)
    # G: ep0 [1, 0], ep1 [2, 0], ep2 [4, 1]
    assert np.allclose(adv.numpy()[:2], [1 - 1., 0 - 1.])        # baseline = mean G of ep1 = 1
    assert np.allclose(adv.numpy()[2:4], [2 - .5, 0 - .5])       # baseline = mean G of ep0 = 0.5
    assert fallback == 2                                          # cell b has one episode: set-wide fallback
    assert np.allclose(adv.numpy()[4:], [4 - .75, 1 - .75])


def test_rows_refuse_bad_inputs(setup):
    spec, support, _, c, rows = setup
    with pytest.raises(ins.InstrumentError, match='float64'):
        ins.Rows(phi=rows.phi.float(), W=rows.W, b=rows.b, mask=rows.mask, action=rows.action,
                 advantage=rows.advantage, episode=rows.episode)
    bad = rows.mask.clone()
    bad[0, rows.action[0]] = False
    with pytest.raises(ins.InstrumentError, match='infeasible'):
        ins.Rows(phi=rows.phi, W=rows.W, b=rows.b, mask=bad, action=rows.action, advantage=rows.advantage,
                 episode=rows.episode)
