"""Reward reward-v1 (ADR 008 section 4): evaluator-side reward events aligned with the executive's DecisionRecords.

EVALUATOR SIDE (ADR 008 section 4, 'Χρήση ορθότητας')
    This module reads evaluator targets (contracts.EvaluatorSpec) and the evaluator's EpisodeAccount. Its outputs are
    rewards for the PPO rollout only; they never reach an observation, a mask or a policy.

Events (values in REWARD_V1, provenance engineering_default_unfrozen; to be frozen before any critic comparison and
identical for every arm)
    Report i (submission order) is classified with evaluation.match_reports over reports[:i+1] (the evaluator's matching
    rule, with the tolerance and metric of the episode account) compared with the matching over reports[:i]:
      new correct distinct target   the number of matched targets increases          +1      at report.sim_time_s
      false report                  report i is compatible with no target            -1      at report.sim_time_s
      duplicate report              otherwise (compatible, no new matched target)    -0.25   at report.sim_time_s
    One added report raises the maximum matching by at most one, so the three cases are exhaustive and exclusive.
    Terminal: decided by the episode's final transition itself (final_transition_end), not by the episode's
    termination reason. The final transition is terminal when the last record is valid and terminated (its own reason:
    mission_complete, mission_deadline, safety_abort, collision), or when the last record is valid and not terminated
    and the executive ended the episode with a terminal reason and no further record (an infeasible forced RETURN:
    'safety_abort'), which episodes.build_episode_transitions applies to that transition. A terminal final transition
    that did not return and land within budget, i.e. unless its reason is 'mission_complete' and account.returned and
    account.landed, gets -(correct distinct reports so far) - 1 at its end (last record's next_observation.sim_time_s).
    A decision that crosses the deadline is terminal ('mission_deadline'): it keeps the penalty even if the later
    post-deadline return fails with an infrastructure error (the episode's termination reason then is
    infrastructure_failure). No terminal event for an invalid last record or a non-terminal valid last record
    (infrastructure exits: ADR 007 section 3, no invented reward). No time reward, no detector-confidence reward.

Alignment with DecisionRecords
    The reports submitted during valid record j are next_observation.reports[len(observation.reports):] (the executive
    appends the report before the REPORT macro's HOLD). For an invalid record (next_observation None), the submitted
    reports beyond its observation's are attributed to it, but an invalid record carries no reward: their events are
    listed in report_events and dropped from per_record. A report no record covers (executive exit between the report and
    its record) gets no reward event either. Report events are at the report time, the start of its REPORT decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from ..mission.contracts import TERMINAL_REASONS
from ..mission.evaluation import match_reports

REWARD_VERSION = 'reward-v1'
NEW_CORRECT, FALSE, DUPLICATE = 'new_correct_distinct', 'false', 'duplicate'


@dataclass(frozen=True)
class RewardSpec:
    version: str
    new_correct_distinct_report: float
    false_report: float
    duplicate_report: float
    unsuccessful_end_constant: float
    unsuccessful_end_per_correct_report: float
    provenance: str

    def to_dict(self):
        return asdict(self)


REWARD_V1 = RewardSpec(REWARD_VERSION, new_correct_distinct_report=1., false_report=-1., duplicate_report=-.25,
                       unsuccessful_end_constant=-1., unsuccessful_end_per_correct_report=-1.,
                       provenance='engineering_default_unfrozen')


@dataclass(frozen=True)
class ReportEvent:
    report_index: int
    sim_time_s: float
    kind: str                        # NEW_CORRECT, FALSE or DUPLICATE
    reward: float
    correct_distinct_after: int      # matched targets over reports[:report_index + 1]


def classify_reports(reports, targets, tolerance_m, metric='xyz', spec=REWARD_V1):
    """ReportEvent per submitted report, by incremental evaluation.match_reports (module docstring)."""
    reports = tuple(reports)
    events, previous = [], 0
    for i, report in enumerate(reports):
        matching = match_reports(reports[:i + 1], targets, tolerance_m, metric)
        count = matching.correct_distinct_reports
        if count - previous not in (0, 1):
            raise RuntimeError(f'report {i}: matched targets changed by {count - previous}; expected 0 or 1')
        if count > previous:
            kind, reward = NEW_CORRECT, spec.new_correct_distinct_report
        elif i in matching.false_report_indices:
            kind, reward = FALSE, spec.false_report
        else:
            if i not in matching.duplicate_report_indices and all(m.report_index != i for m in matching.matches):
                raise RuntimeError(f'report {i}: neither matched, duplicate nor false')
            kind, reward = DUPLICATE, spec.duplicate_report
        events.append(ReportEvent(i, float(report.sim_time_s), kind, float(reward), count))
        previous = count
    return tuple(events)


def successful_end(termination_reason, returned, landed):
    """True when the episode returned and landed within budget (module docstring, 'Terminal')."""
    return termination_reason == 'mission_complete' and bool(returned) and bool(landed)


def final_transition_end(records, termination_reason):
    """(terminated, reason) of the episode's final transition (module docstring, 'Terminal'); shared with episodes.py."""
    if not records or not records[-1].valid:
        return False, None
    last = records[-1]
    if last.terminated:
        return True, last.reason
    if termination_reason in TERMINAL_REASONS:
        return True, termination_reason
    return False, None


def terminal_penalty(correct_distinct_reports, spec=REWARD_V1):
    """-(correct distinct reports so far) - 1 with REWARD_V1."""
    return float(spec.unsuccessful_end_constant + spec.unsuccessful_end_per_correct_report * int(correct_distinct_reports))


@dataclass(frozen=True)
class EpisodeRewardEvents:
    spec: RewardSpec
    per_record: Tuple[Tuple[Tuple[float, float], ...], ...]   # (t_e, r_e) per DecisionRecord, in record order
    report_events: Tuple[ReportEvent, ...]
    report_record_positions: Tuple[Optional[int], ...]        # record position covering each report, None if none
    dropped_report_indices: Tuple[int, ...]                    # reports without reward (invalid or no record)
    terminal_event: Optional[Tuple[int, float, float]]          # (record position, t_e, r_e)
    successful_end: bool
    correct_distinct_reports: int

    def undiscounted_total(self):
        return float(sum(r for events in self.per_record for _, r in events))

    def counts(self):
        kinds = {NEW_CORRECT: 0, FALSE: 0, DUPLICATE: 0}
        for event in self.report_events:
            kinds[event.kind] += 1
        return kinds


def episode_reward_events(records, reports, account, evaluator, spec=REWARD_V1):
    """reward-v1 events aligned with the executive's records.

    records: executive.DecisionRecord sequence of one episode; reports: MissionOutcome.reports (submission order);
    account: evaluation.EpisodeAccount of the episode (termination_reason, returned, landed, matching tolerance/metric);
    evaluator: contracts.EvaluatorSpec (targets). Raises ValueError when records, reports and account disagree.
    """
    records, reports = tuple(records), tuple(reports)
    matching = account.matching
    if matching.tolerance_m != evaluator.match_tolerance_m or matching.report_count != len(reports):
        raise ValueError('episode account matching differs from the evaluator tolerance or the submitted reports')
    events = classify_reports(reports, evaluator.targets, matching.tolerance_m, matching.metric, spec)
    final_count = events[-1].correct_distinct_after if events else 0
    if final_count != matching.correct_distinct_reports:
        raise ValueError(f'incremental matching gives {final_count} correct distinct reports, the account {matching.correct_distinct_reports}')
    positions = [None] * len(reports)
    per_record = [[] for _ in records]
    for position, record in enumerate(records):
        before = record.observation.reports
        after = record.next_observation.reports if record.next_observation is not None else reports
        if tuple(before) != reports[:len(before)] or tuple(after) != reports[:len(after)]:
            raise ValueError(f'record {record.index}: its reports are not a prefix of the submitted reports')
        for i in range(len(before), len(after)):
            if positions[i] is None:
                positions[i] = position
                if record.valid:
                    per_record[position].append((events[i].sim_time_s, events[i].reward))
    dropped = tuple(i for i, p in enumerate(positions) if p is None or not records[p].valid)
    terminated, reason = final_transition_end(records, account.termination_reason)
    success = terminated and successful_end(reason, account.returned, account.landed)
    terminal = None
    if terminated and not success:
        covered = [events[i].correct_distinct_after for i, p in enumerate(positions) if p is not None]
        correct = covered[-1] if covered else 0
        terminal = (len(records) - 1, float(records[-1].next_observation.sim_time_s), terminal_penalty(correct, spec))
        per_record[-1].append(terminal[1:])
    return EpisodeRewardEvents(spec, tuple(tuple(e) for e in per_record), events, tuple(positions), dropped, terminal, success,
                               final_count)


__all__ = ['DUPLICATE', 'EpisodeRewardEvents', 'FALSE', 'NEW_CORRECT', 'REWARD_V1', 'REWARD_VERSION', 'ReportEvent',
           'RewardSpec', 'classify_reports', 'episode_reward_events', 'final_transition_end', 'successful_end', 'terminal_penalty']


# ======================================================================================================================
# reward-v2 (ADR 009 v2 section 2.3; task flood-sar-civil/1). Appended below reward-v1, whose code, values and default
# (REWARD_V1, REWARD_VERSION, episode_reward_events(spec=REWARD_V1)) are unchanged: every name above keeps its behaviour.
#
# EVALUATOR SIDE, as reward-v1: urgency labels, like correctness, are evaluator-only. They enter this reward and the
# study outcomes (aerial_qrl.study.outcomes) and never an observation, a mask or a policy (ADR 008 section 4; ADR 009 v2
# section 2.2 S7). Terminology: a 'report' (SubmittedReport, REPORT action) is what the study calls a 'hand-off'.
#
# Events (values in REWARD_V2, provenance engineering_default_unfrozen; frozen before any critic comparison, identical for
# every arm). Report i is classified exactly as in reward-v1 (classify_reports: incremental evaluation.match_reports):
#   new correct distinct   +1.0 if the target it adds to the evaluator matching has urgency 'normal', +2.0 if 'high'
#   false                  -1.0
#   duplicate              -0.25
#   terminal (same final-transition rule as reward-v1, final_transition_end/successful_end): a terminal final transition
#   that did not return and land within budget gets  -(sum of the urgency weights collected so far) - 1.
#
# DESIGN CHOICE (terminal term). reward-v1 uses -(correct reports so far) - 1: with weight 1 per correct report the
# penalty cancels every positive report reward collected in the episode and adds -1 (ADR 008 section 4 intent: a
# find that is never brought back is worth nothing, and failing to return is worse than finding nothing). ADR 009 v2
# section 2.3 writes the v2 penalty as '-(σωστοί) - 1' without saying whether the correct ones are counted or weighted.
# reward-v2 keeps the v1 intent: it cancels the collected WEIGHTED positives, -(sum of collected urgency weights) - 1.
# Counting them unweighted would leave +1 per high-urgency find in an episode that never returned, i.e. a policy could
# gain by finding a high-urgency victim and not returning.
#
# Definition used for the weights (tie-robust). W_i = sum of the urgency weights of the targets in the evaluator matching
# over reports[:i+1] (W_-1 = 0). Report i gets r_i = (W_i - W_{i-1}) + (false_report if false) + (duplicate_report if
# duplicate). The evaluator matching (maximum cardinality, then minimum summed distance) only ever ADDS the one target a
# new correct distinct report brings, so W_i - W_{i-1} is that target's weight for a new correct report and 0 otherwise,
# which is the table above (property test with random continuous positions in tests/test_reward_v2.py). Only exactly
# equal report-target distances can make the optimal matching
# swap one target for another (seen with integer-grid positions); the difference form then still telescopes, so the
# collected positives equal the weight of the final evaluator matching and the terminal term still cancels them exactly.
# Such reports are flagged (ReportEventV2.matched_set_changed_otherwise).
# ======================================================================================================================

REWARD_V2_VERSION = 'reward-v2'
URGENCY_CLASSES = ('normal', 'high')


@dataclass(frozen=True)
class RewardSpecV2:
    version: str
    new_correct_distinct_normal: float
    new_correct_distinct_high: float
    false_report: float
    duplicate_report: float
    unsuccessful_end_constant: float
    unsuccessful_end_per_collected_weight: float
    provenance: str

    def to_dict(self):
        return asdict(self)

    def weight(self, urgency):
        """Reward of a new correct distinct report of a target with this urgency class."""
        if urgency == 'normal':
            return float(self.new_correct_distinct_normal)
        if urgency == 'high':
            return float(self.new_correct_distinct_high)
        raise ValueError(f'urgency: expected one of {URGENCY_CLASSES}, got {urgency!r}')


REWARD_V2 = RewardSpecV2(REWARD_V2_VERSION, new_correct_distinct_normal=1., new_correct_distinct_high=2., false_report=-1.,
                         duplicate_report=-.25, unsuccessful_end_constant=-1., unsuccessful_end_per_collected_weight=-1.,
                         provenance='engineering_default_unfrozen')


def urgency_labels(targets, urgency=None):
    """{target_id: urgency class} for every target (EVALUATOR SIDE).

    targets: TargetTruth objects or their dicts (as evaluation.match_reports accepts).
    urgency: mapping target_id -> 'normal' | 'high' supplied by the evaluator side; when None, each target's own
    `urgency` attribute (or key) is used (flood-sar-mission/1 TargetTruth has none, so labels must then be given). A missing,
    unknown or extra label raises ValueError: no class is ever invented.
    """
    targets = tuple(targets)
    as_dict = [t if isinstance(t, dict) else {'target_id': t.target_id, 'urgency': getattr(t, 'urgency', None)} for t in targets]
    ids = [t['target_id'] for t in as_dict]
    if urgency is None:
        labels = {t['target_id']: t.get('urgency') for t in as_dict}
    else:
        labels = dict(urgency)
        extra = set(labels) - set(ids)
        if extra:
            raise ValueError(f'urgency labels for unknown targets: {sorted(extra)}')
    missing = [i for i in ids if labels.get(i) is None]
    if missing:
        raise ValueError(f'no urgency label for targets {missing}; reward-v2 needs one per target')
    unknown = {i: labels[i] for i in ids if labels[i] not in URGENCY_CLASSES}
    if unknown:
        raise ValueError(f'unknown urgency classes {unknown}; expected one of {URGENCY_CLASSES}')
    return {i: labels[i] for i in ids}


@dataclass(frozen=True)
class MatchedSetStep:
    report_index: int
    matched_target_ids: Tuple[str, ...]     # evaluator matching over reports[:report_index + 1], sorted
    added_target_ids: Tuple[str, ...]       # not matched over reports[:report_index]
    removed_target_ids: Tuple[str, ...]     # matched before, not now (exact distance ties only)


def matched_target_steps(reports, targets, tolerance_m, metric='xyz'):
    """Evaluator matched-target set after each report (incremental evaluation.match_reports; EVALUATOR SIDE)."""
    reports, targets = tuple(reports), tuple(targets)
    steps, previous = [], frozenset()
    for i in range(len(reports)):
        matching = match_reports(reports[:i + 1], targets, tolerance_m, metric)
        matched = frozenset(m.target_id for m in matching.matches)
        steps.append(MatchedSetStep(i, tuple(sorted(matched)), tuple(sorted(matched - previous)), tuple(sorted(previous - matched))))
        previous = matched
    return tuple(steps)


@dataclass(frozen=True)
class ReportEventV2:
    report_index: int
    sim_time_s: float
    kind: str                               # NEW_CORRECT, FALSE or DUPLICATE (reward-v1 classification)
    reward: float
    correct_distinct_after: int
    new_target_id: Optional[str]            # the one target this report added to the matching (NEW_CORRECT), else None
    urgency: Optional[str]                  # urgency class of new_target_id
    collected_weight_after: float           # W_i: urgency weights of the matched targets over reports[:i+1]
    matched_set_changed_otherwise: bool     # the matching swapped targets (exact distance tie); reward is W_i - W_{i-1} + base


def classify_reports_v2(reports, targets, tolerance_m, metric='xyz', urgency=None, spec=REWARD_V2):
    """ReportEventV2 per submitted report (reward-v2 block above); urgency as in urgency_labels."""
    reports, targets = tuple(reports), tuple(targets)
    labels = urgency_labels(targets, urgency)
    base = classify_reports(reports, targets, tolerance_m, metric)
    steps = matched_target_steps(reports, targets, tolerance_m, metric)
    events, previous_weight = [], 0.
    for event, step in zip(base, steps):
        if len(step.matched_target_ids) != event.correct_distinct_after:
            raise RuntimeError(f'report {event.report_index}: matched set and reward-v1 count differ')
        weight = float(sum(spec.weight(labels[t]) for t in step.matched_target_ids))
        clean = not step.removed_target_ids and len(step.added_target_ids) == (1 if event.kind == NEW_CORRECT else 0)
        new_target = step.added_target_ids[0] if clean and event.kind == NEW_CORRECT else None
        kind_term = spec.false_report if event.kind == FALSE else spec.duplicate_report if event.kind == DUPLICATE else 0.
        events.append(ReportEventV2(event.report_index, event.sim_time_s, event.kind, float(weight - previous_weight + kind_term),
                                    event.correct_distinct_after, new_target, None if new_target is None else labels[new_target],
                                    weight, not clean))
        previous_weight = weight
    return tuple(events)


def terminal_penalty_v2(collected_weight, spec=REWARD_V2):
    """-(sum of collected urgency weights) - 1 with REWARD_V2 (design choice in the reward-v2 block above)."""
    return float(spec.unsuccessful_end_constant + spec.unsuccessful_end_per_collected_weight * float(collected_weight))


@dataclass(frozen=True)
class EpisodeRewardEventsV2(EpisodeRewardEvents):
    collected_weight: float                           # W over all submitted reports (evaluator's final matching)
    urgency_labels: Tuple[Tuple[str, str], ...]      # (target_id, class), evaluator order


def episode_reward_events_v2(records, reports, account, evaluator, urgency=None, spec=REWARD_V2):
    """reward-v2 events aligned with the executive's records; same inputs, alignment and terminal rule as
    episode_reward_events (reward-v1), plus urgency (urgency_labels). Raises ValueError when records, reports, account
    and evaluator disagree or a label is missing."""
    records, reports = tuple(records), tuple(reports)
    matching = account.matching
    if matching.tolerance_m != evaluator.match_tolerance_m or matching.report_count != len(reports):
        raise ValueError('episode account matching differs from the evaluator tolerance or the submitted reports')
    labels = urgency_labels(evaluator.targets, urgency)
    events = classify_reports_v2(reports, evaluator.targets, matching.tolerance_m, matching.metric, labels, spec)
    final_count = events[-1].correct_distinct_after if events else 0
    if final_count != matching.correct_distinct_reports:
        raise ValueError(f'incremental matching gives {final_count} correct distinct reports, the account {matching.correct_distinct_reports}')
    final_weight = float(sum(spec.weight(labels[m.target_id]) for m in matching.matches))
    if abs((events[-1].collected_weight_after if events else 0.) - final_weight) > 1e-9:
        raise ValueError('incremental matched-set weight differs from the account matching')
    positions = [None] * len(reports)
    per_record = [[] for _ in records]
    for position, record in enumerate(records):
        before = record.observation.reports
        after = record.next_observation.reports if record.next_observation is not None else reports
        if tuple(before) != reports[:len(before)] or tuple(after) != reports[:len(after)]:
            raise ValueError(f'record {record.index}: its reports are not a prefix of the submitted reports')
        for i in range(len(before), len(after)):
            if positions[i] is None:
                positions[i] = position
                if record.valid:
                    per_record[position].append((events[i].sim_time_s, events[i].reward))
    dropped = tuple(i for i, p in enumerate(positions) if p is None or not records[p].valid)
    terminated, reason = final_transition_end(records, account.termination_reason)
    success = terminated and successful_end(reason, account.returned, account.landed)
    terminal = None
    if terminated and not success:
        covered = [events[i].collected_weight_after for i, p in enumerate(positions) if p is not None]
        collected = covered[-1] if covered else 0.
        terminal = (len(records) - 1, float(records[-1].next_observation.sim_time_s), terminal_penalty_v2(collected, spec))
        per_record[-1].append(terminal[1:])
    return EpisodeRewardEventsV2(spec, tuple(tuple(e) for e in per_record), events, tuple(positions), dropped, terminal, success,
                                 final_count, final_weight, tuple((t.target_id, labels[t.target_id]) for t in evaluator.targets))


__all__ += ['EpisodeRewardEventsV2', 'MatchedSetStep', 'REWARD_V2', 'REWARD_V2_VERSION', 'ReportEventV2', 'RewardSpecV2',
            'URGENCY_CLASSES', 'classify_reports_v2', 'episode_reward_events_v2', 'matched_target_steps', 'terminal_penalty_v2',
            'urgency_labels']
