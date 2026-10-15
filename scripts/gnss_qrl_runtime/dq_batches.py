"""On-policy batch collection, exact trunk replay, instrument rows and trajectory parity (DQ protocol sections 3-4.1).

Every episode is run by the E1 episode runner itself (p4_train.run_learned_episode, byte-identical to the E1 tree) on
the frozen P3 cells, with the DQ study's namespaces and streams:

    disturbance seed   p3_design.seed_for(DQ namespace, cell, realisation)        (P3's rule; the gate refuses closed ones)
    action stream      SeedSequence([DQ_SALT, SET_TAG[set], unit, cell_index, realisation])
    rewards            placed by the E1 reward code; gamma_ref = 1.0 and tau = 60 s, exactly E1's configuration

Rows are the VALID transitions of each episode, exactly the rows PPO trains on (episodes.build_episode_transitions):
stored obs, GRU input state, final mask, action and log-prob. With the trunk frozen, a row's head input is a function
of (obs, state) alone, so `replay` recomputes the GRU features one step per row and refuses the batch unless every
stored log-prob is reproduced to 1e-9.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import p4_train as p4                                                                             # noqa: E402

GAMMA_REF, TAU = 1.0, 60.0                     # E1 configuration (config.json: gamma_ref 1.0, reference_duration_s 60)
#: one stream tag per episode set; never renumber
SET_TAG = {'development': 0, 'fit': 1, 'oracle': 2, 'calibration': 3, 'heldout': 4, 'fresh_eval': 5}
SET_NAMESPACE = {'development': 'dq_development', 'fit': 'dq_fit', 'oracle': 'dq_oracle',
                 'calibration': 'dq_calibration', 'heldout': 'dq_heldout', 'fresh_eval': 'dq_fresh_eval'}
REPLAY_TOL = 1e-9


class BatchError(dq.DQError):
    """A batch that cannot be used: replay mismatch, an infrastructure episode, a trajectory divergence."""


def action_generator(set_name, unit, cell, realization):
    return torch.Generator().manual_seed(dq.DQ_STUDY.stream(SET_TAG[set_name], unit, cell['cell_index'], realization))


def run_episode(model, runtime, set_name, unit, cell, realization, label):
    namespace = dq.require_open(SET_NAMESPACE[set_name])
    seed = p4.evaluation_seed(runtime, namespace, cell, realization)
    run = p4.run_learned_episode(runtime, cell, namespace, seed, model,
                                 action_generator(set_name, unit, cell, realization), False,
                                 f'dq-{set_name}-u{unit}-c{cell["cell_index"]}-r{realization}-{label}', 0, GAMMA_REF, TAU)
    return run


def collect(model, runtime, set_name, unit, realizations, cells=None, label='m'):
    """Episodes of `model` over cells x realisations; returns (episodes, timing). Infrastructure episodes refuse."""
    was_training = model.training
    model.eval()
    episodes, started = [], time.perf_counter()
    try:
        for cell in (runtime.cells if cells is None else cells):
            for r in realizations:
                run = run_episode(model, runtime, set_name, unit, cell, r, label)
                if run['record']['infrastructure'] or run['record']['invalid_rows']:
                    raise BatchError(f'infrastructure or invalid rows in {cell["cell_id"]} r{r}: {run["record"]}')
                episodes.append({'cell': cell, 'realization': r, 'record': run['record'],
                                 'rows': [row for row in run['transitions'].rows if row['valid']],
                                 'decisions': run['policy'].decisions})
    finally:
        model.train(was_training)
    return episodes, {'episodes': len(episodes), 'wall_s': time.perf_counter() - started,
                      'rows': sum(len(e['rows']) for e in episodes)}


def raw_rows(episodes):
    """Stack the valid rows of a set: obs, GRU input state, final mask, action, stored log-prob, reward, episode, cell."""
    obs, state, mask, action, logp, reward, episode, cell = [], [], [], [], [], [], [], []
    for e_index, e in enumerate(episodes):
        for row in e['rows']:
            obs.append(np.asarray(row['obs'], np.float64))
            state.append(np.asarray(row['state'], np.float64))
            mask.append(np.asarray(row['mask'], bool))
            action.append(int(row['action']))
            logp.append(float(row['log_prob']))
            reward.append(float(row['reward']))
            episode.append(e_index)
            cell.append(e['cell']['cell_id'])
    return {'obs': torch.as_tensor(np.stack(obs)), 'state': torch.as_tensor(np.stack(state)),
            'mask': torch.as_tensor(np.stack(mask)), 'action': torch.as_tensor(action),
            'log_prob': torch.as_tensor(logp, dtype=torch.float64), 'reward': np.asarray(reward),
            'episode': torch.as_tensor(episode), 'cell': np.asarray(cell)}


def replay(model, raw):
    """GRU features h (N, hidden) of every row, one trunk step from its stored state; refuses unless the model
    reproduces every stored log-prob to REPLAY_TOL."""
    with torch.no_grad():
        features, _ = model._trunk(raw['obs'].unsqueeze(1), raw['state'].unsqueeze(0))
        features = features.squeeze(1)
        logits = model.policy(raw['obs'].unsqueeze(1), raw['state'].unsqueeze(0), raw['mask'].unsqueeze(1))[0].squeeze(1)
    lp = torch.log_softmax(logits, -1)[torch.arange(len(raw['action'])), raw['action']]
    error = float((lp - raw['log_prob']).abs().max())
    if not error <= REPLAY_TOL:
        raise BatchError(f'replay does not reproduce the stored log-probs: max error {error:.3e}')
    return features, error


def head_rows(model, raw, features, spec=qc.SHIPPED):
    """ins.Rows for a shipped-circuit head (HQ or its twin: same projection and readout) on replayed features."""
    head = model.actor
    with torch.no_grad():
        angles = math.pi * torch.tanh(head.projection(features))
        phi = qc.basis(spec, angles)
    advantage, fallback = ins.mc_advantages(raw['reward'], raw['episode'].numpy(), raw['cell'])
    rows = ins.Rows(phi=phi, W=head.readout.weight.detach().clone(), b=head.readout.bias.detach().clone(),
                    mask=raw['mask'], action=raw['action'], advantage=advantage, episode=raw['episode'])
    return rows, {'baseline_fallback_rows': fallback}


def trajectory_parity(model, twin, runtime, unit, cell, realization):
    """P4: both models fly the same dq_development episode with the same action stream. Identical action sequences,
    byte-identical observation vectors and masks, the same termination, and |delta log-prob| <= 1e-9 at every
    decision are required; anything else raises BatchError."""
    runs = [run_episode(m, runtime, 'development', unit, cell, realization, tag)
            for m, tag in ((model, 'source'), (twin, 'twin'))]
    a, b = (r['policy'].decisions for r in runs)
    if len(a) != len(b):
        raise BatchError(f'trajectory parity: {len(a)} vs {len(b)} decisions')
    worst = 0.
    for x, y in zip(a, b):
        if x.action_index != y.action_index or not np.array_equal(x.obs, y.obs) or \
                not np.array_equal(x.public_mask, y.public_mask) or \
                (x.mask is None) != (y.mask is None) or (x.mask is not None and not np.array_equal(x.mask, y.mask)):
            raise BatchError(f'trajectory parity: decision {x.index} differs')
        if x.log_prob is not None:
            worst = max(worst, abs(x.log_prob - y.log_prob))
    ra, rb = runs[0]['record'], runs[1]['record']
    if ra['termination_reason'] != rb['termination_reason'] or ra['J'] != rb['J']:
        raise BatchError(f'trajectory parity: outcomes differ {ra["termination_reason"]} {rb["termination_reason"]}')
    if not worst <= 1e-9:
        raise BatchError(f'trajectory parity: max |delta log-prob| {worst:.3e}')
    return {'cell': cell['cell_id'], 'realization': realization, 'decisions': len(a), 'max_abs_dlogprob': worst,
            'termination': ra['termination_reason'], 'wall_s': [r['wall_s'] for r in runs]}


__all__ = ['BatchError', 'SET_NAMESPACE', 'SET_TAG', 'action_generator', 'collect', 'head_rows', 'raw_rows', 'replay',
           'run_episode', 'trajectory_parity']
