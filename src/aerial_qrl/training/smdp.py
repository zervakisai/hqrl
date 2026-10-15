"""Semi-Markov discounting, reward-event discounting, rollout cleaning and GAE (ADR 007, sections 2-4).

Pure numpy; no torch. Every function validates its input and raises ValueError instead of repairing it.

Units and time base
    All times are simulator seconds (the mission.contracts time base). tau = reference_duration_s > 0.
    Decision k starts at t_k and lasts duration_k = t_{k+1} - t_k (measured simulator time). A duration must be
    finite and strictly greater than MIN_DURATION_S (= contracts.TIME_TOLERANCE_S = 1e-6 s, the threshold at
    which the executive rejects a macro that took no time); zero or negative durations raise ValueError.

Formulas (index k runs over the decisions kept after cleaning, in rollout order)
    gamma_k  = gamma_ref ** (duration_k / tau)
    r_k      = sum_e gamma_ref ** ((t_e - t_k) / tau) * r_e       events (t_e, r_e) with t_k <= t_e <= t_k + duration_k
    lambda_k = lambda_ref ** (duration_k / tau)                     lambda_mode='time' (default)
    lambda_k = lambda_ref                                           lambda_mode='decision'
    delta_k  = r_k + gamma_k * (1 - term_k) * V(s_{k+1}) - V(s_k)
    A_k      = delta_k + gamma_k * lambda_k * (1 - term_k) * (1 - cut_k) * A_{k+1}
    return_k = A_k + V(s_k)                                         lambda-return, the critic target used by PPO
    G_k      = r_k + gamma_k * (1 - term_k) * (cut_k * V(s_{k+1}) + (1 - cut_k) * G_{k+1})
               bootstrapped discounted return; it equals return_k when lambda_ref = 1 (either mode).
    V(s_{k+1}) is next_values[k]: the critic value of the observation after decision k, computed with the recurrent
    state after decision k. It is multiplied by zero at terminal decisions, so its value there is irrelevant (but
    must be finite).

Terminal, truncation and infrastructure failures
    terminated (mission_complete, mission_deadline, safety_abort, collision): zero bootstrap; the episode ends.
    truncated (training_horizon): cut_k = 1; bootstrap V(s_{k+1}); GAE does not propagate past k.
    End of rollout while the episode continues: cut_k = 1 with bootstrap, identical to a truncation.
    Invalid (infrastructure_failure) decision: removed. It must be the last decision of its episode; the preceding
    valid decision of the same episode (if any) becomes a truncation (cut_k = 1, bootstrap from V(s_{k+1})) and no
    reward is added for the failed decision. The episode is reported in infrastructure_episode_ids.
    An episode may not contain decisions after a terminal, truncated or invalid decision, and an episode boundary
    inside a rollout must be preceded by one of those three; otherwise ValueError.

Recurrent segments (truncated BPTT)
    bptt_segments splits the kept decisions at every terminal or cut decision and, optionally, every max_length
    decisions. A max_length split is only a BPTT boundary: GAE still propagates across it because the episode
    continues and V(s_{k+1}) is available.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from ..mission.contracts import INFRASTRUCTURE, TIME_TOLERANCE_S

MIN_DURATION_S = TIME_TOLERANCE_S
LAMBDA_MODES = ('time', 'decision')


def _scalar(value, name, low, high, open_low=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name}: expected a real number')
    value = float(value)
    if not math.isfinite(value) or value < low or (open_low and value == low) or value > high:
        raise ValueError(f'{name}: {value} outside {"(" if open_low else "["}{low}, {high}]')
    return value


def check_gamma_ref(gamma_ref):
    return _scalar(gamma_ref, 'gamma_ref', 0.0, 1.0, open_low=True)


def check_tau(tau):
    return _scalar(tau, 'tau', 0.0, math.inf, open_low=True)


def check_lambda_ref(lambda_ref):
    return _scalar(lambda_ref, 'lambda_ref', 0.0, 1.0)


def _reals(values, name, n=None):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f'{name}: expected a 1-D sequence')
    if n is not None and len(array) != n:
        raise ValueError(f'{name}: expected length {n}, got {len(array)}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name}: non-finite value')
    return array


def _flags(values, name, n):
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f'{name}: expected a 1-D sequence of length {n}')
    if array.dtype != np.bool_:
        if len(array) and not (np.issubdtype(array.dtype, np.integer) and np.all((array == 0) | (array == 1))):
            raise ValueError(f'{name}: expected booleans')
        array = array.astype(np.bool_)
    return array


def validate_durations(durations, min_duration_s=MIN_DURATION_S):
    """Return durations (seconds) as float64; raise ValueError for non-finite or <= min_duration_s values."""
    array = _reals(durations, 'durations')
    bad = np.flatnonzero(array <= min_duration_s)
    if len(bad):
        raise ValueError(f'durations: transition {int(bad[0])} has duration {array[bad[0]]!r} s <= {min_duration_s} s')
    return array


def semi_markov_discount(durations, gamma_ref, tau):
    """gamma_k = gamma_ref ** (duration_k / tau)."""
    return check_gamma_ref(gamma_ref) ** (validate_durations(durations) / check_tau(tau))


def decision_lambdas(durations, lambda_ref, tau, lambda_mode='time'):
    """lambda_k = lambda_ref ** (duration_k / tau) for 'time', lambda_ref for 'decision'."""
    durations, lambda_ref, tau = validate_durations(durations), check_lambda_ref(lambda_ref), check_tau(tau)
    if lambda_mode == 'time':
        return lambda_ref ** (durations / tau)
    if lambda_mode == 'decision':
        return np.full(len(durations), lambda_ref)
    raise ValueError(f'lambda_mode: expected one of {LAMBDA_MODES}')


def discount_reward_events(events, starts, gamma_ref, tau, durations=None):
    """Discount reward events to the start of their decision: r_k = sum_e gamma_ref ** ((t_e - t_k)/tau) * r_e.

    events[k] is a (possibly empty) sequence of (t_e, r_e) pairs with t_e in absolute simulator seconds, starts[k]
    is t_k. An event earlier than t_k - TIME_TOLERANCE_S raises; when durations are given, so does an event later
    than t_k + duration_k + TIME_TOLERANCE_S. Offsets within the tolerance are clipped into [0, duration_k].
    """
    gamma_ref, tau = check_gamma_ref(gamma_ref), check_tau(tau)
    starts = _reals(starts, 'starts')
    if isinstance(events, (str, bytes)) or len(events) != len(starts):
        raise ValueError('events: expected one sequence of (t_e, r_e) per transition')
    durations = None if durations is None else validate_durations(durations)
    if durations is not None and len(durations) != len(starts):
        raise ValueError('durations: length differs from starts')
    rewards = np.zeros(len(starts))
    for k, transition_events in enumerate(events):
        for pair in transition_events:
            if len(pair) != 2:
                raise ValueError(f'events[{k}]: expected (t_e, r_e) pairs')
            t_e = _scalar(pair[0], f'events[{k}] time', -math.inf, math.inf)
            r_e = _scalar(pair[1], f'events[{k}] reward', -math.inf, math.inf)
            offset = t_e - starts[k]
            if offset < -TIME_TOLERANCE_S:
                raise ValueError(f'events[{k}]: event at {t_e} s precedes the decision start {starts[k]} s')
            if durations is not None and offset > durations[k] + TIME_TOLERANCE_S:
                raise ValueError(f'events[{k}]: event at {t_e} s after the decision end {starts[k] + durations[k]} s')
            offset = min(max(offset, 0.0), durations[k]) if durations is not None else max(offset, 0.0)
            rewards[k] += gamma_ref ** (offset / tau) * r_e
    return rewards


@dataclass(frozen=True)
class CleanRollout:
    """Result of clean_rollout. Arrays over kept decisions are aligned with `kept` (original rollout indices)."""
    kept: np.ndarray
    terminated: np.ndarray
    cut: np.ndarray
    cut_reason: Tuple[Optional[str], ...]
    removed: np.ndarray
    infrastructure_episode_ids: Tuple[int, ...]


def clean_rollout(valid, terminated, truncated, episode_ids):
    """Remove invalid decisions and mark GAE cuts (module docstring, 'Terminal, truncation and infrastructure').

    cut_reason per kept decision: None (no cut, including terminal decisions), 'training_horizon',
    INFRASTRUCTURE (next decision of the episode was invalid) or 'rollout_end'.
    """
    valid_arr = np.asarray(valid)
    n = len(valid_arr) if valid_arr.ndim == 1 else -1
    valid = _flags(valid, 'valid', n)
    terminated, truncated = _flags(terminated, 'terminated', n), _flags(truncated, 'truncated', n)
    episodes = np.asarray(episode_ids)
    if episodes.ndim != 1 or len(episodes) != n or (n and not np.issubdtype(episodes.dtype, np.integer)):
        raise ValueError(f'episode_ids: expected {n} integers')
    if np.any(terminated & truncated):
        raise ValueError('terminated and truncated are exclusive')
    if np.any(~valid & (terminated | truncated)):
        raise ValueError('an invalid (infrastructure) transition cannot be terminated or truncated')
    seen = set()
    for i in range(n):
        if i and episodes[i] != episodes[i - 1]:
            if not (terminated[i - 1] or truncated[i - 1] or not valid[i - 1]):
                raise ValueError(f'episode {episodes[i - 1]} ends at transition {i - 1} without a terminal, '
                                 'truncation or infrastructure flag')
            seen.add(int(episodes[i - 1]))
            if int(episodes[i]) in seen:
                raise ValueError(f'episode {episodes[i]} is not contiguous')
        elif i and (terminated[i - 1] or truncated[i - 1] or not valid[i - 1]):
            raise ValueError(f'episode {episodes[i]} continues after transition {i - 1} ended it')
    kept = np.flatnonzero(valid)
    cut = np.zeros(len(kept), dtype=np.bool_)
    reasons = []
    for j, i in enumerate(kept):
        if terminated[i]:
            reason = None
        elif truncated[i]:
            reason = 'training_horizon'
        elif i + 1 < n and not valid[i + 1]:
            reason = INFRASTRUCTURE  # same episode, guaranteed by the continuity checks above
        elif i + 1 == n:
            reason = 'rollout_end'
        else:
            reason = None
        cut[j] = reason is not None
        reasons.append(reason)
    removed = np.flatnonzero(~valid)
    failed = tuple(dict.fromkeys(int(episodes[i]) for i in removed))
    return CleanRollout(kept=kept, terminated=terminated[kept].copy(), cut=cut, cut_reason=tuple(reasons),
                        removed=removed, infrastructure_episode_ids=failed)


def _gae_inputs(rewards, values, next_values, durations, terminated, cut):
    durations = validate_durations(durations)
    n = len(durations)
    rewards, values = _reals(rewards, 'rewards', n), _reals(values, 'values', n)
    next_values = _reals(next_values, 'next_values', n)
    terminated, cut = _flags(terminated, 'terminated', n), _flags(cut, 'cut', n)
    if n and not (terminated[-1] or cut[-1]):
        raise ValueError('the last transition must be terminated or cut; GAE would run past the data')
    return rewards, values, next_values, durations, terminated, cut


def compute_gae(rewards, values, next_values, durations, terminated, cut, gamma_ref, tau, lambda_ref,
                lambda_mode='time'):
    """Semi-Markov GAE; returns (advantages, returns) with returns = advantages + values (float64)."""
    rewards, values, next_values, durations, terminated, cut = _gae_inputs(
        rewards, values, next_values, durations, terminated, cut)
    gammas = semi_markov_discount(durations, gamma_ref, tau)
    lambdas = decision_lambdas(durations, lambda_ref, tau, lambda_mode)
    alive = 1.0 - terminated.astype(np.float64)
    carry = (1.0 - cut.astype(np.float64)) * alive
    advantages = np.zeros(len(rewards))
    following = 0.0
    for k in reversed(range(len(rewards))):
        delta = rewards[k] + gammas[k] * alive[k] * next_values[k] - values[k]
        following = delta + gammas[k] * lambdas[k] * carry[k] * following
        advantages[k] = following
    return advantages, advantages + values


def bootstrapped_returns(rewards, next_values, durations, terminated, cut, gamma_ref, tau):
    """G_k = r_k + gamma_k (1 - term_k) (cut_k V(s_{k+1}) + (1 - cut_k) G_{k+1})."""
    rewards, _, next_values, durations, terminated, cut = _gae_inputs(
        rewards, np.zeros(len(np.asarray(rewards))), next_values, durations, terminated, cut)
    gammas = semi_markov_discount(durations, gamma_ref, tau)
    returns = np.zeros(len(rewards))
    following = 0.0
    for k in reversed(range(len(rewards))):
        tail = next_values[k] if cut[k] else following
        following = rewards[k] + (0.0 if terminated[k] else gammas[k] * tail)
        returns[k] = following
    return returns


def classic_gae(rewards, values, next_values, terminated, cut, gamma, gae_lambda):
    """Fixed-step GAE obtained from compute_gae with every duration equal to tau (here tau = 1 s).

    With duration_k = tau both lambda modes give lambda_k = gae_lambda and gamma_k = gamma, so this is
    delta_k = r_k + gamma (1 - term_k) V(s_{k+1}) - V(s_k), A_k = delta_k + gamma gae_lambda (1 - term_k)(1 - cut_k) A_{k+1}.
    """
    n = len(np.asarray(rewards))
    return compute_gae(rewards, values, next_values, np.ones(n), terminated, cut, gamma, 1.0, gae_lambda, 'time')


def bptt_segments(terminated, cut, max_length=None):
    """Half-open (start, stop) index pairs over kept decisions for truncated BPTT.

    A segment ends after every terminal or cut decision and after max_length decisions (if given).
    """
    terminated_arr = np.asarray(terminated)
    n = len(terminated_arr) if terminated_arr.ndim == 1 else -1
    terminated, cut = _flags(terminated, 'terminated', n), _flags(cut, 'cut', n)
    if max_length is not None and (isinstance(max_length, bool) or not isinstance(max_length, (int, np.integer))
                                   or max_length <= 0):
        raise ValueError('max_length: expected a positive integer or None')
    if n and not (terminated[-1] or cut[-1]):
        raise ValueError('the last transition must be terminated or cut')
    segments, start = [], 0
    for k in range(n):
        if terminated[k] or cut[k] or (max_length is not None and k + 1 - start == max_length):
            segments.append((start, k + 1))
            start = k + 1
    return segments


__all__ = ['CleanRollout', 'LAMBDA_MODES', 'MIN_DURATION_S', 'bootstrapped_returns', 'bptt_segments',
           'check_gamma_ref', 'check_lambda_ref', 'check_tau', 'classic_gae', 'clean_rollout', 'compute_gae',
           'decision_lambdas', 'discount_reward_events', 'semi_markov_discount', 'validate_durations']
