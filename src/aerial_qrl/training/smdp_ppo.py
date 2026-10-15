"""Semi-Markov recurrent PPO with interchangeable actor and critic heads (ADR 007).

Shapes and units
    Observations (B, T, obs_dim), boolean action masks (B, T, action_dim), GRU state (1, B, hidden), values (B, T).
    Durations are simulator seconds; rewards are already discounted to the start of their decision
    (smdp.discount_reward_events). Discounting and GAE conventions are those of smdp (see its docstring).

Model (RecurrentActorCritic)
    encoder Linear(obs_dim, hidden) + Tanh -> GRU(hidden, hidden) -> actor head -> action logits.
    Actor head on the GRU features, selected by name (actor_head, default 'linear'; actor_heads.ACTOR_HEADS, whose module
    docstring specifies the heads and their finite-shot execution): linear Linear(h, A), bottleneck, fourier, quantum,
    quantum_no_entanglement. The linear head is the actor of every earlier checkpoint: model_config always records
    actor_head, a model_config without it loads as 'linear' with the state_dict keys actor.weight and actor.bias, and the
    encoder, GRU and actor head are built before the critic head in that order.
    Paired initialisation across actor heads (same global torch generator state before construction): the encoder and GRU
    are identical for every head; a non-linear head is built inside torch.random.fork_rng from the state the linear actor
    would start from, and the global generator is then advanced by exactly the draws of the linear actor (a discarded
    Linear(h, A)), so the critic head, and the global generator state after construction, are identical to the linear-head
    model's. The linear head is built directly, as before (bit-identical). parameter_counts()['actor'] counts the actor
    head; describe()['actor_head'] records its description, name, parameter count and execution_shots (actor_shots():
    {shots, seed} of a quantum head in finite-shot mode, otherwise None).
    Critic head on the GRU features, selected by name:
      mlp                      Linear(h, h) Tanh Linear(h, h) Tanh Linear(h, 1)
      mlp_bottleneck           Linear(h, 4) Tanh Linear(4, h) Tanh Linear(h, 1)   (models.ActorCritic 'small')
      fourier                  models.FourierCritic(h)
      quantum                  models.QuantumCritic(h): 4 qubits, 2 layers, CNOT ring, analytic default.qubit, backprop
      quantum_no_entanglement  models.QuantumCritic(h, entangle=False)
      constant                 one learned scalar, independent of the input (assay-sensitivity control: a state-independent
                               baseline that a useful critic must beat; audit F7, 16 Sep 2026)
    critic_gradient='stop' (primary, ADR 007 section 5): the head reads features.detach(), so the encoder, GRU and
    actor receive gradients only from the actor terms (clipped surrogate and entropy) and the critic influences the
    actor only through the advantages. 'joint' (secondary ablation): value-loss gradients also reach the trunk.
    policy() evaluates the trunk and the actor head only (no critic head; a quantum actor head runs its circuit).
    Action masks: True = feasible. Infeasible logits are set to torch.finfo(dtype).min rather than -inf, so their
    probability is 0 to float precision while Categorical log-probabilities and entropy stay finite. A mask row with
    no feasible action raises ValueError.

Rollout buffer (RolloutBuffer.add, one entry per executed decision, in rollout order)
    obs, mask, action, log_prob (under that mask), value V(s_k), state (GRU input state of the decision; must be zero at
    the first decision of every episode), duration_s, reward, terminated, truncated (training_horizon), valid,
    episode_id, next_value (optional).
    Invalid (infrastructure) decisions carry reward=None (no synthetic reward) and are removed by
    smdp.clean_rollout; the preceding valid decision of the episode becomes a cut.
    Bootstrap V(s_{k+1}) of a non-terminal kept decision k: next_value if given; otherwise the stored value of the
    next decision of the same episode (computed from s_{k+1} with the GRU state after k, also when that decision then
    failed); otherwise ValueError, so training_horizon truncations and the last decision of a rollout must pass
    next_value computed with the GRU state after the decision. next_value is ignored at terminal decisions.
    The stored state of a segment's first decision is the initial state for its replay; recurrent segments are
    truncated-BPTT chunks from smdp.bptt_segments.

Update (ppo_update)
    Advantages are normalised once, with the mean and population standard deviation over all kept (valid)
    transitions: (A - mean) / (std + 1e-8) by default, or (A - mean) / max(std, advantage_std_floor) when the floor is
    given (train config /4). The floor, on the reward scale, keeps an update whose raw advantages are only critic error
    (e.g. an all-zero-reward rollout) from being blown up to unit variance (audit F1).
    Minibatches: minibatch_segments groups a fixed number of segments per optimizer step (the step count then depends on
    episode lengths, i.e. on the policy); minibatch_transitions (train config /4, audit F2) instead packs the shuffled
    segments greedily into minibatches of at most that many transitions (a longer segment is a minibatch of its own), so
    the number of optimizer steps per epoch is ceil(transitions / minibatch_transitions) up to packing slack.
    Every epoch shuffles the segments, right-pads each minibatch (padding is
    excluded from every loss term and statistic; padded mask rows are all-True placeholders) and replays each from its
    stored initial state. Losses are means over the minibatch's real transitions:
    policy_loss = -mean(min(ratio A, clip(ratio, 1 - c, 1 + c) A)), value_loss = mean((V - return)^2) (unclipped),
    entropy = mean(H), approx_kl = mean((ratio - 1) - log ratio), clip_fraction = mean(|ratio - 1| > c),
    loss = policy_loss + value_coef value_loss - entropy_coef entropy.
    Gradient clipping is per parameter group (RecurrentActorCritic.parameter_groups): the trunk+actor group (encoder, GRU,
    actor head, whatever its kind) and the critic-head group are each clipped to max_grad_norm with torch.nn.utils.clip_grad_norm_. A global clip
    would let the critic gradient rescale the actor/trunk step and break the stop-gradient isolation of ADR 007 section 5;
    one Adam optimizer is kept because its per-parameter moments do not couple parameters. grad_norm_trunk_actor and
    grad_norm_critic are the pre-clip norms of the two groups; grad_norm = sqrt(grad_norm_trunk_actor^2 +
    grad_norm_critic^2) is the total pre-clip norm (informational only).
    A non-finite input, loss or gradient norm raises FloatingPointError. Reported metrics are means over optimizer
    steps, plus: advantage_mean and advantage_std (raw), advantage_scale (the divisor used), advantage_below_floor,
    explained_variance of the stored values for the returns (1 - Var(return - V) / Var(return); None when Var(return)
    is 0), and the fractions of steps whose trunk/critic gradient norm exceeded max_grad_norm (clip_rate_trunk_actor,
    clip_rate_critic; audit F23).

Checkpoints
    save_checkpoint stores model_state_dict, optimizer_state_dict, model_config (including actor_head), parameter dtype,
    parameter counts, train_config and metadata (both JSON-serialisable), an optional training_state (tensors, str, int,
    float, bool, None, lists and str-keyed dicts only, so that weights_only loading accepts it; e.g. generator states and
    counters) and never overwrites a file. file_sha256 hashes the saved file for the caller's run record. load_checkpoint rebuilds the model
    with weights_only loading (a model_config without actor_head is a linear-head model). A quantum actor head's shot
    setting is not saved: a loaded model executes analytically until actor.set_shots is called.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..models import FourierCritic, QuantumCritic
from . import smdp
from .actor_heads import ACTOR_HEADS, QUANTUM_ACTOR_HEADS, actor_head_description, build_actor_head

CRITIC_HEADS = ('mlp', 'mlp_bottleneck', 'fourier', 'quantum', 'quantum_no_entanglement', 'constant')
MEMORY_MODES = ('gru', 'none')  # 'none': memoryless comparator (same trunk, no carried state)
CRITIC_GRADIENT_MODES = ('stop', 'joint')
CHECKPOINT_FORMAT = 'aerial-qrl/smdp-ppo-checkpoint/1'
METRIC_KEYS = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'grad_norm', 'grad_norm_trunk_actor',
               'grad_norm_critic')
_DTYPES = {'float32': torch.float32, 'float64': torch.float64}


class ConstantCritic(nn.Module):
    """One learned scalar value for every input (assay-sensitivity control)."""

    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        return self.value.to(features.dtype).expand(*features.shape[:-1], 1)


def build_critic_head(critic, hidden):
    if critic == 'constant':
        return ConstantCritic()
    if critic == 'mlp':
        return nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                             nn.Linear(hidden, 1))
    if critic == 'mlp_bottleneck':
        return nn.Sequential(nn.Linear(hidden, 4), nn.Tanh(), nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    if critic == 'fourier':
        return FourierCritic(hidden)
    if critic in ('quantum', 'quantum_no_entanglement'):
        return QuantumCritic(hidden, entangle=critic == 'quantum')
    raise ValueError(f'critic: expected one of {CRITIC_HEADS}, got {critic!r}')


def critic_head_description(critic):
    """Architecture record for the run config (ADR 007 section 5, 'Καταγραφή')."""
    if critic == 'constant':
        return dict(kind='constant', layers='one learned scalar, input ignored', role='assay-sensitivity control')
    if critic == 'mlp':
        return dict(kind='mlp', layers='Linear(h,h)-Tanh-Linear(h,h)-Tanh-Linear(h,1)')
    if critic == 'mlp_bottleneck':
        return dict(kind='mlp', layers='Linear(h,4)-Tanh-Linear(4,h)-Tanh-Linear(h,1)')
    if critic == 'fourier':
        return dict(kind='fourier', implementation='aerial_qrl.models.FourierCritic',
                    encoding='angles = pi*tanh(Linear(h,4)); sin and cos of angles*[1,2,3]',
                    readout='Linear(24,h)-Tanh-Linear(h,1)')
    if critic in ('quantum', 'quantum_no_entanglement'):
        return dict(kind='variational_circuit', implementation='aerial_qrl.models.QuantumCritic', qubits=4, layers=2,
                    encoding='angles = pi*tanh(Linear(h,4)); every layer RY(angle), trainable RY and RZ per wire',
                    entanglement='CNOT ring' if critic == 'quantum' else None, measurement='<Z> per wire',
                    readout='Linear(4,h)-Tanh-Linear(h,1)', backend='pennylane default.qubit', shots=None,
                    diff_method='backprop')
    raise ValueError(f'critic: expected one of {CRITIC_HEADS}, got {critic!r}')


def apply_action_mask(logits, mask):
    """Set infeasible logits to the dtype minimum. mask: bool tensor with the shape of logits, True = feasible."""
    if mask is None:
        return logits
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError('mask: expected a boolean tensor')
    if mask.shape != logits.shape:
        raise ValueError(f'mask: shape {tuple(mask.shape)} differs from logits {tuple(logits.shape)}')
    if not bool(mask.any(dim=-1).all()):
        raise ValueError('mask: a row has no feasible action')
    return logits.masked_fill(~mask.to(logits.device), torch.finfo(logits.dtype).min)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name}: expected a positive integer')
    return int(value)


class RecurrentActorCritic(nn.Module):
    """Recurrent actor with selectable actor and critic heads (module docstring)."""

    def __init__(self, obs_dim, action_dim, critic, hidden=64, critic_gradient='stop', memory='gru', actor_head='linear'):
        super().__init__()
        obs_dim, action_dim = _positive_int(obs_dim, 'obs_dim'), _positive_int(action_dim, 'action_dim')
        hidden = _positive_int(hidden, 'hidden')
        if critic not in CRITIC_HEADS:
            raise ValueError(f'critic: expected one of {CRITIC_HEADS}, got {critic!r}')
        if critic_gradient not in CRITIC_GRADIENT_MODES:
            raise ValueError(f'critic_gradient: expected one of {CRITIC_GRADIENT_MODES}, got {critic_gradient!r}')
        if memory not in MEMORY_MODES:
            raise ValueError(f'memory: expected one of {MEMORY_MODES}, got {memory!r}')
        if actor_head not in ACTOR_HEADS:
            raise ValueError(f'actor_head: expected one of {ACTOR_HEADS}, got {actor_head!r}')
        self.config = dict(obs_dim=obs_dim, action_dim=action_dim, critic=critic, hidden=hidden,
                           critic_gradient=critic_gradient, memory=memory, actor_head=actor_head)
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        if actor_head == 'linear':
            self.actor = build_actor_head(actor_head, hidden, action_dim)
        else:  # paired initialisation: the head starts where the linear actor would and the critic where it would after it
            with torch.random.fork_rng(devices=[]):
                self.actor = build_actor_head(actor_head, hidden, action_dim)
            nn.Linear(hidden, action_dim)  # discarded: advances the global generator by the linear actor's draws
        self.critic = build_critic_head(critic, hidden)

    def initial_state(self, batch):
        parameter = next(self.parameters())
        return torch.zeros(1, batch, self.hidden, dtype=parameter.dtype, device=parameter.device)

    def _trunk(self, obs, state):
        if obs.ndim != 3 or obs.shape[-1] != self.config['obs_dim']:
            raise ValueError(f'obs: expected (batch, time, {self.config["obs_dim"]}), got {tuple(obs.shape)}')
        if state is None:
            state = self.initial_state(obs.shape[0])
        elif tuple(state.shape) != (1, obs.shape[0], self.hidden):
            raise ValueError(f'state: expected (1, {obs.shape[0]}, {self.hidden}), got {tuple(state.shape)}')
        encoded = self.encoder(obs)
        if self.config['memory'] == 'none':
            # memoryless comparator: the same GRU cell applied to every decision from a zero state (no information carried)
            batch, steps = obs.shape[0], obs.shape[1]
            features, _ = self.gru(encoded.reshape(batch * steps, 1, self.hidden), self.initial_state(batch * steps))
            return features.reshape(batch, steps, self.hidden), self.initial_state(batch)
        return self.gru(encoded, state)

    def policy(self, obs, state=None, mask=None):
        """Actor only: (masked logits, next state). The critic head is not evaluated."""
        features, state = self._trunk(obs, state)
        return apply_action_mask(self.actor(features), mask), state

    def forward(self, obs, state=None, mask=None):
        """(masked logits (B, T, A), values (B, T), next state (1, B, H))."""
        features, state = self._trunk(obs, state)
        logits = apply_action_mask(self.actor(features), mask)
        critic_input = features.detach() if self.config['critic_gradient'] == 'stop' else features
        return logits, self.critic(critic_input).squeeze(-1), state

    def parameter_groups(self):
        """{'trunk_actor': encoder, GRU and actor-head parameters, 'critic': critic-head parameters}, clipped separately."""
        groups = dict(trunk_actor=[*self.encoder.parameters(), *self.gru.parameters(), *self.actor.parameters()],
                      critic=list(self.critic.parameters()))
        if sum(len(g) for g in groups.values()) != len(list(self.parameters())):
            raise RuntimeError('parameter groups do not cover the model parameters')
        return groups

    def parameter_counts(self):
        """Trainable parameters per part; 'actor' counts the actor head, whatever its kind."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        parts = dict(encoder=count(self.encoder), gru=count(self.gru), actor=count(self.actor))
        return dict(parts, encoder_gru_actor=sum(parts.values()), critic_head=count(self.critic), total=count(self))

    def actor_shots(self):
        """Finite-shot setting of a quantum actor head as {shots, seed}; None for analytic execution or a classical head."""
        if self.config['actor_head'] not in QUANTUM_ACTOR_HEADS or self.actor.shots is None:
            return None
        return dict(shots=self.actor.shots, seed=self.actor.seed)

    def describe(self):
        parameter = next(self.parameters())
        name = self.config['actor_head']
        actor = 'actor Linear(h,A)' if name == 'linear' else f'actor head {name}'
        return dict(config=dict(self.config), parameter_counts=self.parameter_counts(),
                    trunk=f'Linear(obs,h)-Tanh -> GRU(h,h) -> {actor}' + (' (memory none: zero GRU state at every decision)'
                                                                          if self.config['memory'] == 'none' else ''),
                    actor_head=dict(actor_head_description(name), name=name, parameters=self.parameter_counts()['actor'],
                                    execution_shots=self.actor_shots()),
                    critic_head=critic_head_description(self.config['critic']),
                    dtype=str(parameter.dtype).removeprefix('torch.'))


def _array(value, name, dtype=np.float64):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _finite(value, name):
    array = _array(value, name)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected one finite number')
    return float(array.reshape(-1)[0])


def _flag(value, name):
    if isinstance(value, torch.Tensor) and value.dtype == torch.bool and value.numel() == 1:
        return bool(value.item())
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name}: expected a boolean')
    return bool(value)


class RolloutBuffer:
    """Per-decision storage for one rollout (module docstring, 'Rollout buffer')."""

    def __init__(self, obs_dim, action_dim, hidden):
        self.obs_dim, self.action_dim = _positive_int(obs_dim, 'obs_dim'), _positive_int(action_dim, 'action_dim')
        self.hidden = _positive_int(hidden, 'hidden')
        self.rows = []

    def __len__(self):
        return len(self.rows)

    def clear(self):
        self.rows = []

    def add(self, *, obs, mask, action, log_prob, value, state, duration_s, reward, terminated, truncated, valid,
            episode_id, next_value=None):
        obs = _array(obs, 'obs').reshape(-1)
        if obs.shape != (self.obs_dim,) or not np.isfinite(obs).all():
            raise ValueError(f'obs: expected {self.obs_dim} finite values')
        mask = _array(mask, 'mask', dtype=None)
        if mask.dtype != np.bool_ or mask.size != self.action_dim:
            raise ValueError(f'mask: expected {self.action_dim} booleans')
        mask = mask.reshape(-1).copy()
        action_array = _array(action, 'action', dtype=None)
        if action_array.size != 1 or not np.issubdtype(action_array.dtype, np.integer):
            raise ValueError('action: expected one integer')
        action = int(action_array.reshape(-1)[0])
        if not 0 <= action < self.action_dim or not mask[action]:
            raise ValueError(f'action {action}: outside the action space or infeasible under its mask')
        state = _array(state, 'state').reshape(-1)
        if state.shape != (self.hidden,) or not np.isfinite(state).all():
            raise ValueError(f'state: expected {self.hidden} finite values')
        if isinstance(episode_id, bool) or not isinstance(episode_id, (int, np.integer)):
            raise ValueError('episode_id: expected an integer')
        valid, terminated, truncated = _flag(valid, 'valid'), _flag(terminated, 'terminated'), _flag(truncated, 'truncated')
        if self.rows and self.rows[-1]['episode_id'] != episode_id and np.any(state != 0):
            raise ValueError('state: the GRU state must be zero at the first decision of an episode')
        row = dict(obs=obs, mask=mask, action=action, log_prob=_finite(log_prob, 'log_prob'),
                   value=_finite(value, 'value'), state=state, episode_id=int(episode_id), valid=valid,
                   terminated=terminated, truncated=truncated)
        if valid:
            if terminated and truncated:
                raise ValueError('terminated and truncated are exclusive')
            if duration_s is None or reward is None:
                raise ValueError('a valid decision requires duration_s and reward')
            row.update(duration_s=float(smdp.validate_durations([_finite(duration_s, 'duration_s')])[0]),
                       reward=_finite(reward, 'reward'),
                       next_value=None if next_value is None else _finite(next_value, 'next_value'))
        else:
            if reward is not None:
                raise ValueError('an invalid (infrastructure) decision carries no reward: pass reward=None')
            if terminated or truncated or next_value is not None:
                raise ValueError('an invalid decision is neither terminated nor truncated and has no next_value')
            row.update(duration_s=math.nan, reward=None, next_value=None)
        self.rows.append(row)


def compute_advantages(buffer, gamma_ref, tau, lambda_ref, lambda_mode='time', max_segment_length=None):
    """Clean the rollout, compute semi-Markov GAE and split recurrent segments; returns a numpy batch dict."""
    rows = buffer.rows
    if not rows:
        raise ValueError('empty rollout')
    clean = smdp.clean_rollout([r['valid'] for r in rows], [r['terminated'] for r in rows],
                               [r['truncated'] for r in rows], [r['episode_id'] for r in rows])
    kept = clean.kept
    if not len(kept):
        raise ValueError('rollout contains no valid decision')
    next_values = np.zeros(len(kept))
    for j, i in enumerate(kept):
        row = rows[i]
        if row['terminated']:
            continue
        if row['next_value'] is not None:
            next_values[j] = row['next_value']
        elif i + 1 < len(rows) and rows[i + 1]['episode_id'] == row['episode_id']:
            next_values[j] = rows[i + 1]['value']
        else:
            raise ValueError(f'transition {i} ({clean.cut_reason[j]}) needs next_value = V(s_(k+1)) computed with '
                             'the GRU state after the decision')
    take = lambda key, dtype=np.float64: np.asarray([rows[i][key] for i in kept], dtype=dtype)  # noqa: E731
    values, durations, rewards = take('value'), take('duration_s'), take('reward')
    advantages, returns = smdp.compute_gae(rewards, values, next_values, durations, clean.terminated, clean.cut,
                                           gamma_ref, tau, lambda_ref, lambda_mode)
    segments = smdp.bptt_segments(clean.terminated, clean.cut, max_segment_length)
    return dict(obs=np.stack([rows[i]['obs'] for i in kept]), masks=np.stack([rows[i]['mask'] for i in kept]),
                actions=take('action', np.int64), old_log_probs=take('log_prob'), values=values,
                next_values=next_values, rewards=rewards, durations_s=durations, terminated=clean.terminated,
                cut=clean.cut, cut_reason=clean.cut_reason, advantages=advantages, returns=returns,
                segments=np.asarray(segments, dtype=np.int64).reshape(-1, 2),
                segment_states=np.stack([rows[kept[start]]['state'] for start, _ in segments]),
                kept_indices=kept, removed_indices=clean.removed,
                infrastructure_episode_ids=clean.infrastructure_episode_ids,
                advantage_config=dict(gamma_ref=smdp.check_gamma_ref(gamma_ref),
                                      reference_duration_s=smdp.check_tau(tau),
                                      lambda_ref=smdp.check_lambda_ref(lambda_ref), lambda_mode=lambda_mode,
                                      max_segment_length=max_segment_length,
                                      critic_target='lambda_return = advantage + value'))


def pad_segments(batch, segment_ids, dtype=torch.float64, device='cpu'):
    """Right-pad the selected segments of a batch into (B, T_max, ...) tensors with a `valid` position mask.

    state is (1, B, H) from the stored segment-start states. Padded observations are zero, padded masks all True,
    padded actions zero; none of them enter a loss because every loss indexes with `valid`.
    """
    segments = np.asarray(batch['segments'])[np.asarray(segment_ids, dtype=np.int64)]
    if not len(segments):
        raise ValueError('no segments selected')
    lengths = segments[:, 1] - segments[:, 0]
    count, longest = len(segments), int(lengths.max())
    obs_dim, action_dim = batch['obs'].shape[1], batch['masks'].shape[1]
    obs = np.zeros((count, longest, obs_dim))
    masks = np.ones((count, longest, action_dim), dtype=np.bool_)
    actions = np.zeros((count, longest), dtype=np.int64)
    floats = {key: np.zeros((count, longest)) for key in ('old_log_probs', 'advantages', 'returns', 'values')}
    valid = np.zeros((count, longest), dtype=np.bool_)
    for b, (start, stop) in enumerate(segments):
        length = stop - start
        obs[b, :length], masks[b, :length] = batch['obs'][start:stop], batch['masks'][start:stop]
        actions[b, :length] = batch['actions'][start:stop]
        for key, array in floats.items():
            array[b, :length] = batch[key][start:stop]
        valid[b, :length] = True
    as_float = lambda array: torch.as_tensor(array, dtype=dtype, device=device)  # noqa: E731
    return dict(obs=as_float(obs), masks=torch.as_tensor(masks, device=device),
                actions=torch.as_tensor(actions, device=device), valid=torch.as_tensor(valid, device=device),
                state=as_float(np.asarray(batch['segment_states'])[np.asarray(segment_ids, dtype=np.int64)]).unsqueeze(0),
                **{key: as_float(array) for key, array in floats.items()})


def ppo_losses(model, padded, clip_range):
    """PPO terms averaged over the real (non-padded) transitions of a padded minibatch."""
    logits, values, _ = model(padded['obs'], padded['state'], padded['masks'])
    valid = padded['valid']
    distribution = Categorical(logits=logits[valid])
    log_prob = distribution.log_prob(padded['actions'][valid])
    log_ratio = log_prob - padded['old_log_probs'][valid]
    ratio = log_ratio.exp()
    advantage = padded['advantages'][valid]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_range, 1 + clip_range) * advantage)
    return dict(policy_loss=-surrogate.mean(), value_loss=(values[valid] - padded['returns'][valid]).square().mean(),
                entropy=distribution.entropy().mean(), approx_kl=((ratio - 1) - log_ratio).mean(),
                clip_fraction=((ratio - 1).abs() > clip_range).to(ratio.dtype).mean())


def pack_minibatches(order, lengths, budget):
    """Greedy packing of segment ids (in the given order) into minibatches of at most `budget` transitions."""
    batches, current, size = [], [], 0
    for index in order:
        length = int(lengths[index])
        if current and size + length > budget:
            batches.append(current)
            current, size = [], 0
        current.append(int(index))
        size += length
    if current:
        batches.append(current)
    return batches


def ppo_update(model, optimizer, batch, clip_range, value_coef, entropy_coef, max_grad_norm, epochs,
               minibatch_segments=None, generator=None, advantage_std_floor=None, minibatch_transitions=None):
    """Run `epochs` passes of minibatch PPO over padded recurrent segments; returns mean metrics."""
    epochs = _positive_int(epochs, 'epochs')
    if (minibatch_segments is None) == (minibatch_transitions is None):
        raise ValueError('give exactly one of minibatch_segments and minibatch_transitions')
    if minibatch_segments is not None:
        minibatch_segments = _positive_int(minibatch_segments, 'minibatch_segments')
    else:
        minibatch_transitions = _positive_int(minibatch_transitions, 'minibatch_transitions')
    if advantage_std_floor is not None and (not math.isfinite(advantage_std_floor) or advantage_std_floor <= 0):
        raise ValueError('advantage_std_floor: expected a positive finite number')
    for name, value, low in (('clip_range', clip_range, 0.0), ('max_grad_norm', max_grad_norm, 0.0)):
        if not math.isfinite(value) or value <= low:
            raise ValueError(f'{name}: expected a positive finite number')
    for name, value in (('value_coef', value_coef), ('entropy_coef', entropy_coef)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name}: expected a non-negative finite number')
    for key in ('obs', 'old_log_probs', 'values', 'advantages', 'returns', 'segment_states'):
        if not np.isfinite(np.asarray(batch[key], dtype=np.float64)).all():
            raise FloatingPointError(f'non-finite {key} in the PPO batch')
    advantages = np.asarray(batch['advantages'], dtype=np.float64)
    mean, std = float(advantages.mean()), float(advantages.std())  # population std over kept transitions only
    scale = std + 1e-8 if advantage_std_floor is None else max(std, advantage_std_floor)
    work = dict(batch, advantages=(advantages - mean) / scale)
    returns, values = np.asarray(batch['returns'], dtype=np.float64), np.asarray(batch['values'], dtype=np.float64)
    explained = None if float(returns.var()) == 0. else 1. - float((returns - values).var()) / float(returns.var())
    lengths = np.asarray(batch['segments'])[:, 1] - np.asarray(batch['segments'])[:, 0]
    clipped = dict(trunk_actor=0, critic=0)
    parameter = next(model.parameters())
    groups = model.parameter_groups()
    model.train()
    count = len(batch['segments'])
    sums, steps = dict.fromkeys(METRIC_KEYS, 0.0), 0
    for _ in range(epochs):
        order = torch.randperm(count, generator=generator).numpy()
        if minibatch_segments is not None:
            minibatches = [order[start:start + minibatch_segments] for start in range(0, count, minibatch_segments)]
        else:
            minibatches = pack_minibatches(order, lengths, minibatch_transitions)
        for selected in minibatches:
            padded = pad_segments(work, selected, parameter.dtype, parameter.device)
            terms = ppo_losses(model, padded, clip_range)
            loss = terms['policy_loss'] + value_coef * terms['value_loss'] - entropy_coef * terms['entropy']
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trunk_norm = float(torch.nn.utils.clip_grad_norm_(groups['trunk_actor'], max_grad_norm))
            critic_norm = float(torch.nn.utils.clip_grad_norm_(groups['critic'], max_grad_norm))
            if not (math.isfinite(trunk_norm) and math.isfinite(critic_norm)):
                raise FloatingPointError('non-finite PPO gradient norm')
            clipped['trunk_actor'] += trunk_norm > max_grad_norm
            clipped['critic'] += critic_norm > max_grad_norm
            optimizer.step()
            for key, value in terms.items():
                sums[key] += float(value.detach())
            sums['grad_norm_trunk_actor'] += trunk_norm
            sums['grad_norm_critic'] += critic_norm
            sums['grad_norm'] += math.hypot(trunk_norm, critic_norm)
            steps += 1
    metrics = {key: value / steps for key, value in sums.items()}
    metrics.update(optimizer_steps=steps, transitions=len(advantages), segments=count, advantage_mean=mean,
                   advantage_std=std, advantage_scale=scale,
                   advantage_below_floor=None if advantage_std_floor is None else std < advantage_std_floor,
                   explained_variance=explained, clip_rate_trunk_actor=clipped['trunk_actor'] / steps,
                   clip_rate_critic=clipped['critic'] / steps,
                   minibatching='segments' if minibatch_segments is not None else 'transitions')
    return metrics


def _check_training_state(value, where='training_state'):
    if value is None or isinstance(value, (bool, int, str, torch.Tensor)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'{where}: non-finite float')
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_training_state(item, f'{where}[{i}]')
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{where}: keys must be strings')
            _check_training_state(item, f'{where}.{key}')
        return
    raise ValueError(f'{where}: unsupported type {type(value).__name__}')


def save_checkpoint(path, model, optimizer, train_config, metadata=None, training_state=None):
    """Write a checkpoint to a new file (FileExistsError if it exists) and return its path."""
    path = Path(path)
    try:
        train_config = json.loads(json.dumps(train_config, allow_nan=False))
        metadata = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('train_config and metadata must be JSON-serialisable without NaN/inf') from exc
    _check_training_state(training_state)
    parameter = next(model.parameters())
    payload = dict(format=CHECKPOINT_FORMAT, model_state_dict=model.state_dict(),
                   optimizer_state_dict=None if optimizer is None else optimizer.state_dict(),
                   model_config=dict(model.config), parameter_dtype=str(parameter.dtype).removeprefix('torch.'),
                   parameter_counts=model.parameter_counts(), train_config=train_config, metadata=metadata,
                   training_state=training_state)
    with path.open('xb') as handle:
        torch.save(payload, handle)
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, optimizer_factory=None, map_location='cpu', payload_check=None):
    """Return (model, optimizer or None, payload). optimizer_factory(parameters) builds the optimizer to restore.

    payload_check(payload): an optional gate run on the raw payload AFTER the format is validated and BEFORE the
    model is built, so a refusal happens before any weight is constructed or read. `checkpoint_identity`
    uses it to enforce the GNSS task identity (model contract invariant 11); None keeps the legacy behaviour.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'{path}: not a {CHECKPOINT_FORMAT} checkpoint')
    if payload['parameter_dtype'] not in _DTYPES:
        raise ValueError(f'{path}: unsupported parameter dtype {payload["parameter_dtype"]!r}')
    if payload_check is not None:
        payload_check(payload)
    model = RecurrentActorCritic(**payload['model_config']).to(dtype=_DTYPES[payload['parameter_dtype']])
    model.load_state_dict(payload['model_state_dict'])
    optimizer = None
    if optimizer_factory is not None:
        if payload['optimizer_state_dict'] is None:
            raise ValueError(f'{path}: checkpoint has no optimizer state')
        optimizer = optimizer_factory(model.parameters())
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    return model, optimizer, payload


__all__ = ['ACTOR_HEADS', 'CHECKPOINT_FORMAT', 'CRITIC_GRADIENT_MODES', 'CRITIC_HEADS', 'ConstantCritic', 'QUANTUM_ACTOR_HEADS',
           'RecurrentActorCritic', 'RolloutBuffer',
           'apply_action_mask', 'build_critic_head', 'compute_advantages', 'critic_head_description', 'file_sha256',
           'load_checkpoint', 'pack_minibatches', 'pad_segments', 'ppo_losses', 'ppo_update', 'save_checkpoint']
