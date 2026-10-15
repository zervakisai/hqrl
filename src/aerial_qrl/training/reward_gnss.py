"""Timely-reporting score J and its telescoping reward for task flood-sar-village-gnss/1.

EVALUATOR SIDE. Everything here reads evaluator targets and the private global coordinates of the submitted
reports. Its outputs are the episode score and the PPO rollout rewards. No quantity computed here may become an
observation, an action mask or a planner input.

The score (STATISTICAL_PROTOCOL_EN.md section 4.2)
    w(t_r)     = max(0, 1 - t_r / B_mission)                         timing weight of a submission
    TP_timely  = sum of w(t_r) over the reports in the FINAL matching
    J          = (I(RETURN_OK) * TP_timely - lambda * FP_spatial) / N_ref

    N_ref is the target count of the frozen POSITIVE scenario manifest and is used for its paired no-human
    scenario too, so the denominator is never zero and a no-human episode is scored on the same scale. It is an
    evaluator-only normalisation and never a policy input. lambda is 1 in the main score; 0.5 and 2 are the
    declared descriptive sensitivities, and choosing among them after seeing results is forbidden.

The matching is the repository's existing one
    evaluation.match_reports: maximum cardinality first, then minimum summed distance, then earlier reports.
    Report times do not enter it. One consequence is load-bearing and is NOT a defect: a later, closer report
    can take a target away from an earlier, coarser one, which turns the earlier report into a duplicate and
    LOWERS TP_timely, because the target's credit now carries the later, smaller timing weight. The telescoping
    reward therefore has to produce a negative increment for that report. A fixed "+1 per new correct report"
    scheme cannot express this, which is why reward-v1 is not reused.

Telescoping identity (model contract invariant 10)
    Phi_k      = (TP_timely(reports[:k]) - lambda * FP(reports[:k])) / N_ref, recomputed with the same matching
    increment  = Phi_(k+1) - Phi_k, credited at the submission time of report k
    terminal   = 0 on a successful return, else -(TP_timely_final / N_ref): failure removes the timely
                 true-positive credit and KEEPS every false-report penalty.
    sum(increments) + terminal == J exactly, for rematching, duplicates, missed deadline, failed return, a
    no-human episode and a zero-report episode. `check_reward_identity` asserts it and the tests cover each case.

Not inherited from reward-v1
    no duplicate constant, no return bonus, no failure constant and no discount. Under gamma = 1 with a common
    deadline in the state, the undiscounted sum of these increments IS the objective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from ..mission.evaluation import match_reports

REWARD_VERSION = 'reward-gnss-timely/1'
SCORE_VERSION = 'score-gnss-timely/1'
#: Declared descriptive sensitivities of the false-report weight; the main score is 1.0.
SENSITIVITY_LAMBDAS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class TimelyScoreSpec:
    """Frozen conventions of the score. The weights are research conventions, not rescue-service utilities."""
    version: str
    mission_budget_s: float
    n_ref: int
    tolerance_m: float
    metric: str
    false_report_weight: float
    provenance: str

    def __post_init__(self):
        if self.version not in (SCORE_VERSION,):
            raise ValueError(f'version: expected {SCORE_VERSION}')
        if not self.mission_budget_s > 0:
            raise ValueError('mission_budget_s must be positive')
        if not (isinstance(self.n_ref, int) and self.n_ref > 0):
            raise ValueError('n_ref must be a positive integer: it is the paired positive manifest target count')
        if self.tolerance_m <= 0:
            raise ValueError('tolerance_m must be positive')
        if self.metric not in ('xyz', 'xy'):
            raise ValueError("metric: expected 'xyz' or 'xy'")
        if self.false_report_weight < 0:
            raise ValueError('false_report_weight must be non-negative')

    def to_dict(self):
        return asdict(self)


def timing_weight(submit_time_s, mission_budget_s):
    """w(t) = max(0, 1 - t/B). A submission after the deadline carries no credit."""
    return max(0., 1. - float(submit_time_s) / float(mission_budget_s))


@dataclass(frozen=True)
class ScoreComponents:
    tp_timely: float
    tp_spatial: int
    fp_spatial: int
    duplicates: int
    unresolved: int
    potential: float                  # (TP_timely - lambda*FP) / N_ref, before the return gate
    matched_report_indices: Tuple[int, ...]

    def to_dict(self):
        return asdict(self)


def components(global_positions, submit_times_s, targets, spec):
    """Score components of the reports submitted so far, under the frozen matching.

    `global_positions` must be the FROZEN GLOBAL coordinates of the private ledger, normally
    `GeoreferenceService.global_report_positions()`. A public local-frame receipt is refused here as well as in
    `evaluation.match_reports`, so neither entry point can score the local frame (review 1, ruling R1).
    """
    positions, times = tuple(global_positions), tuple(float(t) for t in submit_times_s)
    if len(positions) != len(times):
        raise ValueError('global_positions and submit_times_s must have the same length')
    for i, position in enumerate(positions):
        if hasattr(position, 'local_position_enu') or hasattr(position, 'position_enu'):
            raise ValueError(
                f'global_positions[{i}]: expected a frozen global coordinate, got {type(position).__name__}. '
                'Pass GeoreferenceService.global_report_positions(), not public report records.')
    matching = match_reports(positions, targets, spec.tolerance_m, spec.metric)
    tp_timely = sum(timing_weight(times[m.report_index], spec.mission_budget_s) for m in matching.matches)
    potential = (tp_timely - spec.false_report_weight * matching.false_reports) / spec.n_ref
    return ScoreComponents(float(tp_timely), matching.correct_distinct_reports, matching.false_reports,
                           matching.duplicate_reports, matching.unresolved_targets, float(potential),
                           tuple(m.report_index for m in matching.matches))


def terminal_score(global_positions, submit_times_s, targets, spec, return_ok):
    """J for a finished episode."""
    parts = components(global_positions, submit_times_s, targets, spec)
    credited = parts.tp_timely if return_ok else 0.
    return (credited - spec.false_report_weight * parts.fp_spatial) / spec.n_ref, parts


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-report increments, the terminal correction and the score they sum to."""
    spec_version: str
    increments: Tuple[float, ...]        # one per submitted report, at its submission time
    potentials: Tuple[float, ...]        # Phi_0 .. Phi_n, with Phi_0 = 0
    terminal_correction: float
    score: float
    final: ScoreComponents

    def total(self):
        return sum(self.increments) + self.terminal_correction

    def to_dict(self):
        return {'spec_version': self.spec_version, 'increments': list(self.increments), 'potentials': list(self.potentials),
                'terminal_correction': self.terminal_correction, 'score': self.score, 'final': self.final.to_dict()}


def episode_rewards(global_positions, submit_times_s, targets, spec, return_ok):
    """Telescoping increments and the terminal correction; their sum is exactly J (module docstring)."""
    positions, times = tuple(global_positions), tuple(float(t) for t in submit_times_s)
    potentials = [0.]
    for k in range(1, len(positions) + 1):
        potentials.append(components(positions[:k], times[:k], targets, spec).potential)
    increments = tuple(potentials[k + 1] - potentials[k] for k in range(len(positions)))
    score, final = terminal_score(positions, times, targets, spec, return_ok)
    correction = 0. if return_ok else -final.tp_timely / spec.n_ref
    return RewardBreakdown(REWARD_VERSION, increments, tuple(potentials), float(correction), float(score), final)


def check_reward_identity(breakdown, tolerance=1e-9):
    """Invariant 10. Raises ValueError with the two numbers when the telescoping sum misses the score."""
    total = breakdown.total()
    if abs(total - breakdown.score) > tolerance:
        raise ValueError(f'reward identity violated: sum(rewards) = {total!r} but J = {breakdown.score!r}')
    return True


def assign_increments_to_records(breakdown, report_record_index):
    """Map per-report increments onto decision records.

    report_record_index[k] is the index of the decision record that submitted report k, or None when no valid
    record covers it (an executive exit between the submission and its record). Reports without a record keep
    their increment out of the rollout, exactly as reward-v1 does, and the dropped total is returned so a caller
    can report it instead of silently losing reward mass.
    """
    per_record, dropped = {}, 0.
    for k, increment in enumerate(breakdown.increments):
        index = report_record_index[k] if k < len(report_record_index) else None
        if index is None:
            dropped += increment
            continue
        per_record[index] = per_record.get(index, 0.) + increment
    return per_record, dropped


__all__ = ['REWARD_VERSION', 'SCORE_VERSION', 'SENSITIVITY_LAMBDAS', 'RewardBreakdown', 'ScoreComponents',
           'TimelyScoreSpec', 'assign_increments_to_records', 'check_reward_identity', 'components',
           'episode_rewards', 'terminal_score', 'timing_weight']
