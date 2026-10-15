"""reward-gnss-timely/1 as per-record reward events for `episodes.build_episode_transitions` (task
flood-sar-village-gnss/1, P4).

The score and its telescoping reward are `reward_gnss`'s; this module only places them on the executive's
decision records, so the learned arms are trained on exactly the quantity P3 scores.

Placement
    Report k (the k-th receipt the episode appended, which is the k-th submission of the private ledger) is placed
    on the decision record whose transition appended it: the valid record whose next observation carries receipt k
    and whose observation does not. Its event is (t_submit_k, Phi_k - Phi_{k-1}), where t_submit_k is the
    submission time the service froze - the start of that REPORT macro, so the event lies inside its transition.
    The terminal correction -I(not RETURN_OK) * TP_timely / N_ref is placed on the LAST valid record at that
    record's end time.

What is not placed, and is reported instead of hidden
    A report whose receipt no valid record appended (an executive exit between the submission and its record)
    keeps its increment out of the rollout, as `reward_gnss.assign_increments_to_records` specifies, and the lost
    mass is returned as `dropped`. The terminal correction of an episode with no valid record is dropped too.

The identity this preserves
    With gamma_ref = 1 (protocol section 4.2a) the rollout rewards of an episode sum to J minus `dropped`, and
    `dropped` is zero whenever every report has a valid record; `check_episode_reward_sum` asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .reward_gnss import RewardBreakdown

GNSS_REWARD_PLACEMENT_VERSION = 'gnss-reward-placement/1'


@dataclass(frozen=True)
class PlacedRewards:
    per_record: Tuple[Tuple[Tuple[float, float], ...], ...]   # one (t_e, r_e) tuple per record
    dropped: float                                            # reward mass no valid record could carry
    report_records: Tuple[object, ...]                        # record position of report k, or None
    terminal_record: object                                   # record position of the correction, or None
    placed_total: float


def _receipt_count(observation):
    return len(observation.reports) if observation is not None else 0


def report_record_positions(records, n_reports):
    """Record position of each appended receipt, in order; None for a receipt no valid record appended."""
    positions = [None] * n_reports
    for position, record in enumerate(records):
        if not record.valid or record.next_observation is None:
            continue
        before, after = _receipt_count(record.observation), _receipt_count(record.next_observation)
        for k in range(before, min(after, n_reports)):
            positions[k] = position
    return positions


def place_gnss_rewards(records, breakdown, submission_times_s):
    """PlacedRewards for one episode (module docstring)."""
    if not isinstance(breakdown, RewardBreakdown):
        raise TypeError('breakdown: expected reward_gnss.RewardBreakdown')
    records = list(records)
    increments = tuple(breakdown.increments)
    if len(submission_times_s) != len(increments):
        raise ValueError(f'{len(increments)} report increments but {len(submission_times_s)} submission times')
    per_record: List[list] = [[] for _ in records]
    dropped = 0.
    positions = report_record_positions(records, len(increments))
    for k, (increment, t_submit) in enumerate(zip(increments, submission_times_s)):
        if positions[k] is None:
            dropped += increment
            continue
        per_record[positions[k]].append((float(t_submit), float(increment)))
    valid = [position for position, record in enumerate(records) if record.valid]
    terminal = valid[-1] if valid else None
    correction = float(breakdown.terminal_correction)
    if correction:
        if terminal is None:
            dropped += correction
        else:
            record = records[terminal]
            per_record[terminal].append((float(record.observation.sim_time_s + record.duration_s), correction))
    placed = sum(r for events in per_record for _, r in events)
    return PlacedRewards(tuple(tuple(events) for events in per_record), float(dropped), tuple(positions),
                         terminal, float(placed))


def check_episode_reward_sum(transitions, breakdown, placed, tolerance=1e-9):
    """With gamma_ref = 1 the kept rewards sum to J minus the dropped mass. Raises ValueError otherwise.

    The tolerance is the runner's own invariant-10 bound: a sum of increments telescopes to the final potential
    only up to floating-point rounding.
    """
    kept = sum(row['reward'] for row in transitions.rows if row['valid'])
    expected = breakdown.score - placed.dropped
    if abs(kept - expected) > tolerance:
        raise ValueError(f'rollout rewards sum to {kept!r}, but J - dropped = {expected!r}')
    return True


__all__ = ['GNSS_REWARD_PLACEMENT_VERSION', 'PlacedRewards', 'check_episode_reward_sum', 'place_gnss_rewards',
           'report_record_positions']
