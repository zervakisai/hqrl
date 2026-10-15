"""The HQ -> twin transplant and its fail-closed certificate (DQ protocol section 3)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS                                           # noqa: E402
from aerial_qrl.training.smdp_ppo import RecurrentActorCritic, load_checkpoint                   # noqa: E402

F64 = torch.float64


@pytest.fixture(scope='module')
def shipped_support():
    return tp.support(qc.SHIPPED)


def hq_model(seed=0, dtype=F64, scale=None):
    torch.manual_seed(seed)
    model = RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, 'mlp', hidden=64,
                                 actor_head='quantum').to(dtype)
    if scale is not None:
        with torch.no_grad():
            model.actor.weights.mul_(scale)
    return model.eval()


def rows(model, n=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, model.config['obs_dim'], generator=g, dtype=F64)
    state = torch.randn(n, model.hidden, generator=g, dtype=F64) * 0.5
    mask = torch.rand(n, model.config['action_dim'], generator=g) > 0.3
    mask[:, 0] = True
    return dict(obs=obs, state=state, mask=mask)


def test_support_of_the_shipped_circuit_is_886_with_a_clean_gap(shipped_support):
    mask, record = shipped_support
    assert record['size'] == 886 == int(mask.sum())
    assert record['max_off_support'] <= 1e-14 and record['min_on_support'] >= 1e-3


@pytest.mark.parametrize('scale', [None, 10.0])      # initialiser-scale and large (trained-like) circuit weights
def test_twin_is_the_same_policy_and_every_check_passes(shipped_support, scale):
    model = hq_model(3, scale=scale)
    twin, C = tp.make_twin(model)
    record = tp.certify_static(model, twin, C, shipped_support[0])
    assert record['static_failures'] == [] and record['P5_unequal'] == []
    assert record['P2_offgrid_max_abs'] <= 1e-12
    row_record = tp.certify_rows(model, twin, rows(model))
    assert row_record['P3_max_abs_logprob'] <= 1e-9
    assert twin.config['actor_head'] == 'surrogate625'
    assert sum(p.numel() for p in twin.parameters()) > sum(p.numel() for p in model.parameters())


def test_float32_sources_are_refused():
    with pytest.raises(tp.TransplantParityError, match='float64'):
        tp.make_twin(hq_model(0, dtype=torch.float32))


def test_a_non_quantum_source_is_refused():
    torch.manual_seed(0)
    model = RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, 'mlp', actor_head='bottleneck').double()
    with pytest.raises(tp.TransplantParityError, match='not the shipped circuit'):
        tp.make_twin(model)


def test_a_tampered_shared_tensor_fails_p5(shipped_support):
    model = hq_model(4)
    twin, C = tp.make_twin(model)
    with torch.no_grad():
        twin.critic[0].weight[0, 0] += 1e-15 if twin.critic[0].weight[0, 0] != 0 else 1.
    with pytest.raises(tp.TransplantParityError, match='P5'):
        tp.certify_static(model, twin, C, shipped_support[0])


def test_wrong_coefficients_fail_p1_p2_and_rows(shipped_support):
    model = hq_model(5)
    twin, C = tp.make_twin(model)
    bad = C.clone()
    bad[0, 0] += 1e-6
    with torch.no_grad():
        twin.actor.surrogate.weight.copy_(bad.T)
    with pytest.raises(tp.TransplantParityError, match='P1'):
        tp.certify_static(model, twin, bad, shipped_support[0])
    with pytest.raises(tp.TransplantParityError, match='row parity'):
        tp.certify_rows(model, twin, rows(model))


def test_off_support_excursion_is_refused_not_clipped(shipped_support):
    mask = shipped_support[0].clone()
    first_on = int(torch.nonzero(mask)[0])
    mask[first_on] = False                     # pretend an on-support coordinate were off-support
    model = hq_model(6)
    twin, C = tp.make_twin(model)
    if abs(float(C.reshape(-1)[first_on])) > 1e-12:
        with pytest.raises(tp.TransplantParityError, match='P6'):
            tp.certify_static(model, twin, C, mask)


def test_save_twin_goes_through_the_write_guard_and_reloads_identically(tmp_path, shipped_support):
    model = hq_model(7)
    twin, C = tp.make_twin(model)
    record = tp.certify_static(model, twin, C, shipped_support[0])
    out = tmp_path / 'twin.pt'
    tp.save_twin(twin, record, out, test_root=tmp_path)
    loaded, optimizer, payload = load_checkpoint(out)
    assert optimizer is None and payload['optimizer_state_dict'] is None
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), twin.state_dict().values()))
    with pytest.raises(Exception):
        tp.save_twin(twin, record, out, test_root=tmp_path)
    with pytest.raises(tp.dq.DQError):
        tp.save_twin(twin, record, ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'e1' / 'x.pt')
