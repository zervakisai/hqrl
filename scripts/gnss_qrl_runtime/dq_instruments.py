"""DQ instruments: the exact local quantities of the DQ protocol section 4, as pure float64 functions.

Nothing here collects data, reads a checkpoint or writes a file. Every function takes a `Rows` set (section 4.1: one
row per executed valid decision, with the head input already replayed) and coefficient vectors, and returns numbers.

Notation (protocol section 4). Row i has Fourier features phi_i (F,) of its angles, readout W (A, 4), b (A,), feasible
mask M_i, action a_i and advantage Ahat_i. The head coefficients are c = vec(C), C (F, 4), row-major over
(feature, wire); logits l_i(c) = W C^T phi_i + b are AFFINE in c; infeasible logits are set to the dtype minimum exactly
as smdp_ppo.apply_action_mask does. The support S (a boolean mask over vec(C)) restricts every vector and matrix; the
instruments act on c_S with the off-support coordinates held at their (zero) values.

    gradient          g = (1/N) sum_i Ahat_i J_i^T (e_{a_i} - pi_i),        J_i = d l_i / d c = W (x) phi_i^T
    fisher            F = (1/N) sum_i J_i^T (diag pi_i - pi_i pi_i^T) J_i   (the exact Hessian of the row-mean forward KL)
    natural_step      d(B) = B (B^T F B + lambda_B I)^{-1} B^T g,  lambda_B = delta tr(B^T F B) / k
    fit_subspace      top-k generalised eigenvectors of Sigma_O v = mu (F_O + lambda I) v  (FIT-k, section 4.6)
    chart_rank        k_u = #{sigma_j > tau sigma_1} of A(theta) with the ambiguity band [tau/100, 100 tau]
    calibrate_kl      forward KL on K to a target, bracket by doubling then bisection, first crossing (section 4.7)
    heldout_gain      (1/|H|) sum_i (pi'(a_i)/pi(a_i) - 1) Ahat_i                                     (section 4.8)
    accessible        q_B = g^T B (B^T F B + lambda_B I)^{-1} B^T g                                   (section 4.8)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np
import torch

F64 = torch.float64
KL_REL_TOL = 1e-3
MAX_DOUBLINGS = 60
MAX_BISECTIONS = 100
NULL_CURVATURE = 1e-30
RANK_BAND = 100.


class InstrumentError(ValueError):
    """An instrument refused its input (non-float64, non-finite, inconsistent shapes)."""


@dataclass
class Rows:
    phi: torch.Tensor          # (N, F) Fourier features of the replayed angles
    W: torch.Tensor            # (A, wires) readout weight
    b: torch.Tensor            # (A,) readout bias
    mask: torch.Tensor         # (N, A) bool, True = feasible (the stored final mask)
    action: torch.Tensor       # (N,) long
    advantage: torch.Tensor    # (N,) float64, Ahat of section 4.1 (built inside this set)
    episode: torch.Tensor      # (N,) long, episode index inside this set
    surrogate_bias: Optional[torch.Tensor] = None    # (wires,), zero for a transplant

    def __post_init__(self):
        for name in ('phi', 'W', 'b', 'advantage'):
            value = getattr(self, name)
            if value.dtype != F64:
                raise InstrumentError(f'{name}: float64 required, got {value.dtype}')
            if not bool(torch.isfinite(value).all()):
                raise InstrumentError(f'{name}: non-finite values')
        n = self.phi.shape[0]
        if self.mask.dtype != torch.bool or tuple(self.mask.shape) != (n, self.W.shape[0]):
            raise InstrumentError('mask: expected (N, A) bool')
        if not bool(self.mask.any(dim=1).all()):
            raise InstrumentError('mask: a row has no feasible action')
        if not bool(self.mask[torch.arange(n), self.action].all()):
            raise InstrumentError('action: infeasible under its stored mask')
        if self.surrogate_bias is None:
            self.surrogate_bias = torch.zeros(self.W.shape[1], dtype=F64)

    @property
    def n(self):
        return int(self.phi.shape[0])

    @property
    def wires(self):
        return int(self.W.shape[1])

    @property
    def features(self):
        return int(self.phi.shape[1])


# -- policy on rows ---------------------------------------------------------------------------------------------------

def logits(rows, c):
    C = c.reshape(rows.features, rows.wires)
    z = rows.phi @ C + rows.surrogate_bias
    raw = z @ rows.W.T + rows.b
    return raw.masked_fill(~rows.mask, torch.finfo(F64).min)


def log_policy(rows, c):
    return torch.log_softmax(logits(rows, c), dim=-1)


def policy(rows, c):
    return torch.softmax(logits(rows, c), dim=-1)


def full(c_s, support, base):
    """vec(C) with the support coordinates replaced by c_s (off-support coordinates keep `base`)."""
    out = base.clone()
    out[support] = c_s
    return out


# -- gradient and Fisher (section 4.4) ----------------------------------------------------------------------------------

def gradient(rows, c, support):
    """g restricted to S: (1/N) sum_i Ahat_i J_i^T (e_{a_i} - pi_i)."""
    pi = policy(rows, c)
    residual = -pi
    residual[torch.arange(rows.n), rows.action] += 1.
    residual = residual * rows.advantage[:, None]                       # (N, A)
    g = rows.phi.T @ (residual @ rows.W) / rows.n                        # (F, wires) = (1/N) sum phi_i (W^T R_i)^T
    return g.reshape(-1)[support]


FISHER_CHUNK = 512


def _fisher_factor(rows, c, support, start=0, stop=None):
    """Z (n*wires, |S|) of rows [start, stop) with F_S contribution Z^T Z: per row, phi_i (x) L_i where
    L_i L_i^T = W^T (diag pi_i - pi_i pi_i^T) W."""
    stop = rows.n if stop is None else stop
    pi = policy(rows, c)[start:stop]
    phi = rows.phi[start:stop]
    cov = torch.diag_embed(pi) - pi[:, :, None] * pi[:, None, :]          # (n, A, A)
    M = rows.W.T @ cov @ rows.W                                          # (n, w, w)
    M = 0.5 * (M + M.transpose(1, 2))
    evals, evecs = torch.linalg.eigh(M)
    L = evecs * evals.clamp_min(0.).sqrt()[:, None, :]                   # columns r: L_i[:, r]
    # z_{i,r}[(f, j)] = phi_if * L_i[j, r], flattened row-major over (f, j) like vec(C)
    Z = phi[:, None, :, None] * L.transpose(1, 2)[:, :, None, :]         # (n, r, F, w)
    return Z.reshape((stop - start) * rows.wires, -1)[:, support]


def fisher(rows, c, support):
    """Explicit F_S (|S|, |S|), accumulated over row chunks so the full (N*wires, F*wires) factor never exists."""
    size = int(support.sum())
    F = torch.zeros(size, size, dtype=F64)
    for start in range(0, rows.n, FISHER_CHUNK):
        Z = _fisher_factor(rows, c, support, start, min(start + FISHER_CHUNK, rows.n))
        F += Z.T @ Z
    F /= rows.n
    return 0.5 * (F + F.T)


def fisher_vector(rows, c, support, v):
    """Matrix-free F_S v (never forms F): (1/N) sum_i J_i^T Sigma_i (J_i v)."""
    pi = policy(rows, c)
    V = torch.zeros(rows.features * rows.wires, dtype=F64)
    V[support] = v
    Jv = rows.phi @ V.reshape(rows.features, rows.wires) @ rows.W.T     # (N, A) = J_i v
    Jv = Jv.masked_fill(~rows.mask, 0.)
    SJv = pi * Jv - pi * (pi * Jv).sum(-1, keepdim=True)                  # Sigma_i J_i v
    out = rows.phi.T @ (SJv @ rows.W) / rows.n
    return out.reshape(-1)[support]


def mean_kl(rows, c_old, c_new):
    """Row-mean forward KL(pi_old || pi_new) over each row's feasible actions (section 4.7)."""
    lp_old, lp_new = log_policy(rows, c_old), log_policy(rows, c_new)
    p_old = lp_old.exp()
    terms = torch.where(rows.mask, p_old * (lp_old - lp_new), torch.zeros_like(p_old))
    return float(terms.sum(-1).mean())


# -- chart geometry -----------------------------------------------------------------------------------------------------

def chart_rank(A_s, tau):
    """(k_u, singular values, ambiguous) of the chart Jacobian restricted to S (section 4.3)."""
    s = torch.linalg.svdvals(A_s)
    relative = s / s[0]
    k = int((relative > tau).sum())
    ambiguous = bool(((relative >= tau / RANK_BAND) & (relative <= tau * RANK_BAND)).any())
    return k, s, ambiguous


def tangent_basis(A_s, k):
    """Orthonormal basis (|S|, k) of span(A_S): the top-k left singular vectors."""
    U, _, _ = torch.linalg.svd(A_s, full_matrices=False)
    return U[:, :k]


def orthogonal_complement(Q):
    """Orthonormal basis (n, n - k) of the Euclidean complement of span(Q) in R^n (Q orthonormal, (n, k))."""
    n, k = Q.shape
    full_q, _ = torch.linalg.qr(Q, mode='complete')
    return full_q[:, k:]


def haar_subspace(dim, k, generator):
    """Orthonormal basis of a Haar-random k-dimensional subspace of R^dim (QR of a Gaussian, sign-fixed)."""
    G = torch.randn(dim, k, generator=generator, dtype=F64)
    Q, R = torch.linalg.qr(G, mode='reduced')
    return Q * torch.sign(torch.diagonal(R))[None, :]


def _damping(M, delta):
    k = M.shape[0]
    return delta * float(torch.trace(M)) / k


def natural_step(B, g, F, delta):
    """d(B) = B (B^T F B + lambda_B I)^{-1} B^T g; returns (d, lambda_B). delta = 0 uses the pseudo-inverse."""
    M = B.T @ F @ B
    M = 0.5 * (M + M.T)
    lam = _damping(M, delta)
    rhs = B.T @ g
    if delta == 0:
        coef = torch.linalg.pinv(M, hermitian=True) @ rhs
    else:
        coef = torch.linalg.solve(M + lam * torch.eye(M.shape[0], dtype=F64), rhs)
    return B @ coef, lam


def accessible(B, g, F, delta):
    """q_B = g^T B (B^T F B + lambda_B I)^{-1} B^T g (section 4.8)."""
    d, _ = natural_step(B, g, F, delta)
    return float(g @ d)


def span_step(g, F, delta):
    """The SPAN arm, d(I_S) = (F + lambda I)^{-1} g, without forming an identity basis product."""
    lam = _damping(F, delta)
    return torch.linalg.solve(F + lam * torch.eye(F.shape[0], dtype=F64), g), lam


def theta_quantities(A_s, g, F):
    """(g_theta, F_theta) = (A^T g, A^T F A): the exact pull-back to circuit weights."""
    return A_s.T @ g, A_s.T @ F @ A_s


def theta_natural(g_theta, F_theta, delta):
    lam = _damping(F_theta, delta)
    return torch.linalg.solve(F_theta + lam * torch.eye(F_theta.shape[0], dtype=F64), g_theta), lam


# -- FIT-1 / FIT-k (section 4.6) ------------------------------------------------------------------------------------------

def episode_contributions(rows, c, support):
    """U (E, |S|): u_e = sum_{i in e} Ahat_i J_i^T (e_{a_i} - pi_i), restricted to S."""
    pi = policy(rows, c)
    residual = -pi
    residual[torch.arange(rows.n), rows.action] += 1.
    residual = residual * rows.advantage[:, None]
    per_row_w = residual @ rows.W                                        # (N, wires)
    episodes = torch.unique(rows.episode)
    U = []
    for e in episodes:
        idx = rows.episode == e
        U.append((rows.phi[idx].T @ per_row_w[idx]).reshape(-1)[support])
    return torch.stack(U), episodes


def fit_subspace(U, F_O, k, delta):
    """Top-k generalised eigenvectors of Sigma_O v = mu (F_O + lambda I) v, Sigma_O = U^T U / E.

    Returns (basis (|S|, k) orthonormalised, eigenvalues mu (all, descending), record). The Euclidean orthonormalisation
    of the span is harmless: natural steps depend on the span only (section 4.5)."""
    E = U.shape[0]
    lam = _damping(F_O, delta)
    K = F_O + lam * torch.eye(F_O.shape[0], dtype=F64)
    L = torch.linalg.cholesky(K)
    # Sigma_O = U^T U / E; whitened: L^{-1} U^T U L^{-T} / E = Y Y^T with Y = L^{-1} U^T / sqrt(E)
    Y = torch.linalg.solve_triangular(L, U.T, upper=False) / math.sqrt(E)
    Uy, sy, _ = torch.linalg.svd(Y, full_matrices=False)                 # eigenpairs of Y Y^T, descending
    mu = sy ** 2
    if k > Uy.shape[1]:
        raise InstrumentError(f'FIT-{k} needs at least {k} episodes with independent contributions; E = {E}')
    V = torch.linalg.solve_triangular(L.T, Uy[:, :k], upper=True)        # v = L^{-T} y
    Q, R = torch.linalg.qr(V, mode='reduced')
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    tie = bool(k < mu.numel() and mu[k - 1] > 0 and (mu[k - 1] - mu[k]) / mu[k - 1] < 1e-8)
    return Q, mu, {'episodes': E, 'k': k, 'mu_k': float(mu[k - 1]), 'mu_k_plus_1': float(mu[k]) if k < mu.numel() else None,
                   'fit_tie': tie, 'estimable': E >= 2 * k, 'lambda': lam}


def fit_one(U, F_O, delta):
    """FIT-1: the unit vector along (F_O + lambda I)^{-1} g_O, g_O = sum_e u_e / |O| (scale irrelevant)."""
    g = U.sum(0)
    d, lam = span_step(g, F_O, delta)
    return (d / d.norm())[:, None], lam


# -- KL calibration (section 4.7) ------------------------------------------------------------------------------------------

@dataclass
class Calibration:
    status: str               # ok | null_direction | unattained | no_convergence
    alpha: float
    kl: float
    target: float
    doublings: int
    bisections: int
    alpha0: Optional[float]


def calibrate_kl(ray: Callable[[float], torch.Tensor], c0, rows_K, curvature, target, rel_tol=KL_REL_TOL,
                 max_doublings=MAX_DOUBLINGS, max_bisections=MAX_BISECTIONS):
    """Find alpha with mean forward KL on K = target (to rel_tol), first crossing along `ray` (alpha -> full vec(C)).

    `curvature` = d^T F_K d for the ray's initial direction (the quadratic prediction sets alpha0)."""
    if not math.isfinite(curvature) or curvature <= NULL_CURVATURE:
        return Calibration('null_direction', 0., 0., target, 0, 0, None)
    alpha0 = math.sqrt(2. * target / curvature)
    lo, hi = 0., alpha0
    kl_hi = mean_kl(rows_K, c0, ray(hi))
    doublings = 0
    while kl_hi < target:
        if doublings >= max_doublings:
            return Calibration('unattained', hi, kl_hi, target, doublings, 0, alpha0)
        lo, hi = hi, 2. * hi
        kl_hi = mean_kl(rows_K, c0, ray(hi))
        doublings += 1
    if abs(kl_hi - target) <= rel_tol * target:
        return Calibration('ok', hi, kl_hi, target, doublings, 0, alpha0)
    for bisections in range(1, max_bisections + 1):
        mid = 0.5 * (lo + hi)
        kl_mid = mean_kl(rows_K, c0, ray(mid))
        if abs(kl_mid - target) <= rel_tol * target:
            return Calibration('ok', mid, kl_mid, target, doublings, bisections, alpha0)
        if kl_mid < target:
            lo = mid
        else:
            hi = mid
    return Calibration('no_convergence', mid, kl_mid, target, doublings, max_bisections, alpha0)


# -- gains and advantages (sections 4.1, 4.8) -------------------------------------------------------------------------------

def heldout_gain(rows_H, c_old, c_new):
    """(1/|H|) sum_i (pi_new(a_i)/pi_old(a_i) - 1) Ahat_i; zero at c_new = c_old by construction."""
    idx = torch.arange(rows_H.n)
    lp_old = log_policy(rows_H, c_old)[idx, rows_H.action]
    lp_new = log_policy(rows_H, c_new)[idx, rows_H.action]
    return float(((lp_new - lp_old).exp() - 1.) @ rows_H.advantage / rows_H.n)


def mc_advantages(rewards, episode, cell):
    """Ahat_i = G_i - b, G the undiscounted return-to-go (gamma_ref = 1 in E1), b the leave-one-episode-out mean of G
    over the same set's rows of the same cell; a cell with one episode falls back to the set-wide leave-one-episode-out
    mean. Returns (advantages, fallback_rows)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    episode = np.asarray(episode)
    cell = np.asarray(cell)
    G = np.zeros_like(rewards)
    for e in np.unique(episode):
        idx = np.flatnonzero(episode == e)
        G[idx] = np.cumsum(rewards[idx][::-1])[::-1]
    advantages = np.zeros_like(G)
    fallback = 0
    for i in range(G.size):
        same = (cell == cell[i]) & (episode != episode[i])
        if same.any():
            advantages[i] = G[i] - G[same].mean()
        else:
            other = episode != episode[i]
            if not other.any():
                raise InstrumentError('a set with a single episode has no leave-one-episode-out baseline')
            advantages[i] = G[i] - G[other].mean()
            fallback += 1
    return torch.as_tensor(advantages, dtype=F64), fallback


__all__ = ['Calibration', 'InstrumentError', 'Rows', 'accessible', 'calibrate_kl', 'chart_rank', 'episode_contributions',
           'fisher', 'fisher_vector', 'fit_one', 'fit_subspace', 'full', 'gradient', 'haar_subspace', 'heldout_gain',
           'log_policy', 'logits', 'mc_advantages', 'mean_kl', 'natural_step', 'orthogonal_complement', 'policy',
           'span_step', 'tangent_basis', 'theta_natural', 'theta_quantities']
