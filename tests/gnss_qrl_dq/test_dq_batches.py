"""Batch plumbing that needs no episode: replay refusal, closed sets, stream separation (DQ protocol section 4.1)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_batches as db                                                                           # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS                                           # noqa: E402
from aerial_qrl.training.smdp_ppo import RecurrentActorCritic                                    # noqa: E402


def model():
    torch.manual_seed(0)
    return RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, 'mlp', actor_head='quantum').double().eval()


def fake_raw(m, n=12, corrupt=False):
    g = torch.Generator().manual_seed(2)
    obs = torch.randn(n, m.config['obs_dim'], generator=g, dtype=torch.float64)
    state = 0.3 * torch.randn(n, m.hidden, generator=g, dtype=torch.float64)
    mask = torch.ones(n, m.config['action_dim'], dtype=torch.bool)
    action = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        logits = m.policy(obs.unsqueeze(1), state.unsqueeze(0), mask.unsqueeze(1))[0].squeeze(1)
    lp = torch.log_softmax(logits, -1)[:, 0]
    if corrupt:
        lp = lp.clone()
        lp[3] += 1e-8
    return {'obs': obs, 'state': state, 'mask': mask, 'action': action, 'log_prob': lp,
            'reward': np.zeros(n), 'episode': torch.arange(n) % 3, 'cell': np.array(['a'] * n)}


def test_replay_reproduces_stored_log_probs_and_refuses_a_mismatch():
    m = model()
    features, error = db.replay(m, fake_raw(m))
    assert features.shape == (12, m.hidden) and error <= 1e-12
    with pytest.raises(db.BatchError, match='replay'):
        db.replay(m, fake_raw(m, corrupt=True))


def test_head_rows_use_the_projection_and_readout_of_the_head():
    m = model()
    raw = fake_raw(m)
    features, _ = db.replay(m, raw)
    rows, info = db.head_rows(m, raw, features)
    assert rows.n == 12 and rows.features == 625 and info['baseline_fallback_rows'] == 0
    assert torch.equal(rows.W, m.actor.readout.weight.detach())


@pytest.mark.parametrize('set_name', ['fit', 'oracle', 'calibration', 'heldout', 'fresh_eval'])
def test_confirmatory_sets_are_closed(set_name):
    with pytest.raises(dq.DQError, match='closed'):
        db.run_episode(model(), dq.cell_runtime(), set_name, 0, dq.cell_runtime().cells[0], 0, 'x')


def test_every_set_has_its_own_namespace_and_stream():
    assert len(set(db.SET_NAMESPACE.values())) == len(db.SET_NAMESPACE) == len(db.SET_TAG)
    cell = {'cell_index': 3}
    seeds = {name: db.action_generator(name, 0, cell, 0).initial_seed() for name in db.SET_TAG}
    assert len(set(seeds.values())) == len(seeds)
