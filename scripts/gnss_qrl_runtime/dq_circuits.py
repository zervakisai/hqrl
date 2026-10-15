"""Exact circuit algebra for DQ: the shipped 2-layer circuit, the held-out 3-layer circuit, and their Fourier twins.

the DQ protocol sections 3 and 4.2-4.3. Two circuits share one structure (the one of
actor_heads.QuantumActorHead): per layer and wire RY(angle), trainable RY(w0), RZ(w1), then a CNOT ring
(wire -> wire+1 mod 4); readout <Z_j> per wire. Only the number of layers differs:

    SHIPPED   4 wires, 2 layers  -> spectrum {-2..2} per angle, basis 5 per wire, 5^4 = 625 features, 16 weights
    HELDOUT   4 wires, 3 layers  -> spectrum {-3..3} per angle, basis 7 per wire, 7^4 = 2401 features, 24 weights

The expectations z(t; w) are simulated here directly in torch (complex128 state vector, PennyLane's gate and wire
conventions: wire 0 the most significant bit). A test proves `expectations(SHIPPED, ...)` equals the PennyLane circuit of
actor_heads.QuantumActorHead to 1e-13, so DQ can differentiate and batch it freely while the PennyLane head remains the
reference the parity certificate compares against.

Fourier twin. z is a trigonometric polynomial of degree <= L in each angle, so z(t; w) = C(w)^T phi(t) with phi the
product basis (1, cos t, sin t, cos 2t, sin 2t, ...) over the wires, wire 0 the slowest index; for L = 2 this is exactly
actor_heads.SurrogateActorHead.features. C(w) is SOLVED on the (2L+1)^4 tensor grid, never fitted:
C(w) = Phi_G^{-1} Z_G(w). Its Jacobian A(w) = dC/dw is exact (autograd through the solve).
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import torch
from torch import nn

F64, C128 = torch.float64, torch.complex128


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    wires: int
    layers: int
    entangle: bool = True

    @property
    def basis_size(self):
        return 2 * self.layers + 1

    @property
    def features(self):
        return self.basis_size ** self.wires

    @property
    def weight_shape(self):
        return (self.layers, self.wires, 2)

    @property
    def n_weights(self):
        return self.layers * self.wires * 2

    def inert_weights(self):
        """Flat indices of weights with an exactly zero gradient: the LAST layer's RZ angles (RZ, the CNOT ring and the
        Z readout are all diagonal or permutations in the computational basis)."""
        last = self.layers - 1
        return tuple(last * self.wires * 2 + wire * 2 + 1 for wire in range(self.wires))


SHIPPED = CircuitSpec('shipped_2layer_ring', 4, 2)
HELDOUT = CircuitSpec('heldout_3layer_ring', 4, 3)
SPECS = {SHIPPED.name: SHIPPED, HELDOUT.name: HELDOUT}
#: stable small integers for seed streams; never renumber
SPEC_INDEX = {SHIPPED.name: 0, HELDOUT.name: 1}


# -- state-vector simulation ------------------------------------------------------------------------------------------

def _apply_single(state, gate, wire, wires):
    """state (N, 2, ..., 2) complex; gate (N, 2, 2) or (2, 2) applied on `wire` (axis wire + 1)."""
    axis = wire + 1
    state = state.movedim(axis, -1)
    if gate.ndim == 2:
        state = state @ gate.transpose(-1, -2)
    else:
        shape = state.shape
        flat = state.reshape(shape[0], -1, 2)
        flat = torch.einsum('nij,nkj->nki', gate, flat)
        state = flat.reshape(shape)
    return state.movedim(-1, axis)


def _ry(theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack((torch.stack((c, -s), -1), torch.stack((s, c), -1)), -2).to(C128)


def _rz(theta):
    e = torch.exp(-0.5j * theta.to(C128))
    zero = torch.zeros_like(e)
    return torch.stack((torch.stack((e, zero), -1), torch.stack((zero, e.conj()), -1)), -2)


def _cnot(state, control, target):
    """CNOT on axes control+1 / target+1: flip target where control is 1."""
    t_axis = target + 1 - (1 if target > control else 0)   # the target axis once the control axis is selected away
    return torch.stack((state.select(control + 1, 0), state.select(control + 1, 1).flip(t_axis)), dim=control + 1)


def expectations(spec, angles, weights):
    """<Z_j> (N, wires) of the circuit on angles (N, wires) with weights spec.weight_shape; float64, differentiable."""
    angles = angles.to(F64)
    weights = weights.to(F64)
    n = angles.shape[0]
    state = torch.zeros((n,) + (2,) * spec.wires, dtype=C128)
    state[(slice(None),) + (0,) * spec.wires] = 1.
    for layer in range(spec.layers):
        for wire in range(spec.wires):
            state = _apply_single(state, _ry(angles[:, wire]), wire, spec.wires)
            state = _apply_single(state, _ry(weights[layer, wire, 0]), wire, spec.wires)
            state = _apply_single(state, _rz(weights[layer, wire, 1]), wire, spec.wires)
        if spec.entangle:
            for wire in range(spec.wires):
                state = _cnot(state, wire, (wire + 1) % spec.wires)
    probabilities = (state.conj() * state).real.reshape(n, -1)
    bits = torch.tensor([[(s >> (spec.wires - 1 - w)) & 1 for w in range(spec.wires)] for s in range(2 ** spec.wires)],
                        dtype=F64)
    return probabilities @ (1. - 2. * bits)


# -- Fourier twin -----------------------------------------------------------------------------------------------------

def basis(spec, angles):
    """(N, wires) angles -> (N, (2L+1)^wires) products of (1, cos t, sin t, ..., cos Lt, sin Lt), wire 0 slowest."""
    angles = angles.to(F64)
    harmonics = [torch.ones_like(angles)]
    for order in range(1, spec.layers + 1):
        harmonics += [(order * angles).cos(), (order * angles).sin()]
    per_wire = torch.stack(harmonics, dim=-1)
    products = per_wire[:, 0]
    for wire in range(1, spec.wires):
        products = (products.unsqueeze(-1) * per_wire[:, wire].unsqueeze(-2)).reshape(angles.shape[0], -1)
    return products


def grid(spec):
    points = torch.linspace(0., 2. * math.pi, spec.basis_size + 1, dtype=F64)[:-1]
    return torch.cartesian_prod(*[points] * spec.wires)


@functools.lru_cache(maxsize=None)
def _design_lu(spec):
    design = basis(spec, grid(spec))
    lu, pivots = torch.linalg.lu_factor(design)
    return design, lu, pivots


def design_matrix(spec):
    return _design_lu(spec)[0]


def coefficients(spec, weights):
    """C(w) (features, wires): solved on the tensor grid, never fitted; differentiable in w."""
    _, lu, pivots = _design_lu(spec)
    return torch.linalg.lu_solve(lu, pivots, expectations(spec, grid(spec), weights))


def solve_residual(spec, weights, C=None):
    C = coefficients(spec, weights) if C is None else C
    return float((design_matrix(spec) @ C - expectations(spec, grid(spec), weights)).abs().max())


def jacobian(spec, weights):
    """A(w) = d vec(C)/d vec(w): (features * wires, n_weights), vec row-major over (feature, wire)."""
    w = weights.detach().to(F64).reshape(-1)

    def flat(v):
        return coefficients(spec, v.reshape(spec.weight_shape)).reshape(-1)
    return torch.func.jacfwd(flat)(w)


# -- heads (for the held-out circuit; the shipped circuit uses actor_heads unchanged) -----------------------------------

class CircuitHead(nn.Module):
    """angles = pi*tanh(Linear(h,4)); the circuit of `spec` simulated exactly; Linear(4, A). The held-out-circuit
    analogue of actor_heads.QuantumActorHead (analytic only; no finite-shot mode)."""

    def __init__(self, hidden, action_dim, spec):
        super().__init__()
        self.spec = spec
        self.projection = nn.Linear(hidden, spec.wires)
        self.weights = nn.Parameter(0.1 * torch.randn(*spec.weight_shape))
        self.readout = nn.Linear(spec.wires, action_dim)

    def forward(self, features):
        shape = features.shape[:-1]
        angles = math.pi * torch.tanh(self.projection(features.reshape(-1, features.shape[-1])))
        z = expectations(self.spec, angles, self.weights).to(features.dtype)
        return self.readout(z).reshape(*shape, -1)


class TwinHead(nn.Module):
    """angles = pi*tanh(Linear(h,4)); Linear(basis, 4) on the Fourier basis of `spec`; Linear(4, A). For SHIPPED its
    computation and state_dict layout are those of actor_heads.SurrogateActorHead."""

    def __init__(self, hidden, action_dim, spec):
        super().__init__()
        self.spec = spec
        self.projection = nn.Linear(hidden, spec.wires)
        self.surrogate = nn.Linear(spec.features, spec.wires)
        self.readout = nn.Linear(spec.wires, action_dim)

    def forward(self, features):
        shape = features.shape[:-1]
        angles = math.pi * torch.tanh(self.projection(features.reshape(-1, features.shape[-1])))
        return self.readout(self.surrogate(basis(self.spec, angles).to(features.dtype))).reshape(*shape, -1)


__all__ = ['CircuitHead', 'CircuitSpec', 'HELDOUT', 'SHIPPED', 'SPECS', 'SPEC_INDEX', 'TwinHead', 'basis', 'coefficients',
           'design_matrix', 'expectations', 'grid', 'jacobian', 'solve_residual']
