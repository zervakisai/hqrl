"""The DQ circuit algebra against the PennyLane reference head (DQ protocol sections 3-4)."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
from aerial_qrl.training import actor_heads                                                       # noqa: E402

F64 = torch.float64


def pennylane_head(seed=0):
    torch.manual_seed(seed)
    return actor_heads.QuantumActorHead(64, 24).double()


def test_torch_simulation_equals_the_pennylane_circuit_of_the_shipped_head():
    head = pennylane_head(3)
    with torch.no_grad():
        head.weights.copy_(torch.randn(2, 4, 2, dtype=F64) * 1.3)
        angles = (torch.rand(257, 4, dtype=F64) * 2 - 1) * math.pi
        reference = torch.stack(head.circuit(angles, head.weights), dim=-1)
        ours = qc.expectations(qc.SHIPPED, angles, head.weights)
    assert float((ours - reference).abs().max()) <= 1e-13


def test_product_state_control_is_reproduced_without_the_ring():
    torch.manual_seed(1)
    head = actor_heads.QuantumActorHead(64, 24, entangle=False).double()
    spec = qc.CircuitSpec('noring', 4, 2, entangle=False)
    with torch.no_grad():
        angles = torch.randn(64, 4, dtype=F64)
        reference = torch.stack(head.circuit(angles, head.weights), dim=-1)
        assert float((qc.expectations(spec, angles, head.weights) - reference).abs().max()) <= 1e-13


def test_shipped_basis_is_the_surrogate625_feature_map():
    angles = torch.randn(33, 4, dtype=F64)
    assert torch.equal(qc.basis(qc.SHIPPED, angles), actor_heads.SurrogateActorHead.features(angles))
    assert qc.SHIPPED.features == actor_heads.SURROGATE_FEATURES == 625


@pytest.mark.parametrize('spec', [qc.SHIPPED, qc.HELDOUT])
def test_design_matrix_is_invertible_and_the_twin_reproduces_the_circuit_off_grid(spec):
    assert torch.linalg.matrix_rank(qc.design_matrix(spec)) == spec.features
    weights = torch.randn(*spec.weight_shape, dtype=F64)
    C = qc.coefficients(spec, weights)
    assert tuple(C.shape) == (spec.features, spec.wires)
    assert qc.solve_residual(spec, weights, C) <= 1e-12
    angles = (torch.rand(4096, 4, dtype=F64) * 2 - 1) * math.pi
    assert float((qc.basis(spec, angles) @ C - qc.expectations(spec, angles, weights)).abs().max()) <= 1e-12


@pytest.mark.parametrize('spec', [qc.SHIPPED, qc.HELDOUT])
def test_jacobian_matches_finite_differences_and_the_last_rz_layer_is_exactly_inert(spec):
    weights = torch.randn(*spec.weight_shape, dtype=F64)
    A = qc.jacobian(spec, weights)
    assert tuple(A.shape) == (spec.features * spec.wires, spec.n_weights)
    for inert in spec.inert_weights():
        assert float(A[:, inert].abs().max()) <= 1e-13
    flat = weights.reshape(-1)
    for k in (0, 1, 5, spec.n_weights - 2):
        e = torch.zeros_like(flat)
        e[k] = 1e-6
        fd = (qc.coefficients(spec, (flat + e).reshape(spec.weight_shape))
              - qc.coefficients(spec, (flat - e).reshape(spec.weight_shape))).reshape(-1) / 2e-6
        assert float((fd - A[:, k]).abs().max()) <= 1e-7


def test_shipped_chart_has_rank_twelve_at_generic_weights():
    torch.manual_seed(5)
    for _ in range(4):
        s = torch.linalg.svdvals(qc.jacobian(qc.SHIPPED, 0.1 * torch.randn(2, 4, 2, dtype=F64)))
        assert int((s > 1e-6 * s[0]).sum()) == 12, s


def test_circuit_and_twin_heads_agree_after_a_solve():
    torch.manual_seed(2)
    spec = qc.HELDOUT
    circuit = qc.CircuitHead(64, 24, spec).double()
    twin = qc.TwinHead(64, 24, spec).double()
    with torch.no_grad():
        twin.projection.load_state_dict(circuit.projection.state_dict())
        twin.readout.load_state_dict(circuit.readout.state_dict())
        twin.surrogate.weight.copy_(qc.coefficients(spec, circuit.weights).T)
        twin.surrogate.bias.zero_()
        x = torch.randn(5, 7, 64, dtype=F64)
        assert float((twin(x) - circuit(x)).abs().max()) <= 1e-12
