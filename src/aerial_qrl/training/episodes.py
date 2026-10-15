"""Per-episode transitions for smdp_ppo.RolloutBuffer from the RL policy's decisions, the executive's DecisionRecords and
reward-v1 events (ADR 008 section 5; ADR 007 sections 2-3).

Inputs
    decisions      rl_policy.PolicyDecision list of the episode (policy.decisions, in order).
    records        executive.DecisionRecord list of the episode (executive.records).
    record_events  per record, (t_e, r_e) events in absolute simulator seconds (reward.EpisodeRewardEvents.per_record).
    termination_reason  MissionOutcome.termination_reason.

Alignment rules (one transition per executed policy decision)
    A record with forced None belongs to the next policy decision whose observation time equals the record's observation
    time and whose current action equals the record's action. Earlier decisions that were never executed are skipped and
    counted (abandoned); a skipped executed decision raises EpisodeAlignmentError.
    Valid policy record -> valid transition: start t_k = observation.sim_time_s, duration_s, its events, terminated/reason.
    The matched decision must have been executed, with one exception: an incomplete command (executive._incomplete) writes
    a valid terminal 'safety_abort' record without calling policy feedback. Such a record, when it is the last record and
    not forced, belongs to the last sampled decision (matching action and observation time, no later decision with an
    action) and becomes a terminal transition with reason safety_abort (reward.py gives it the terminal penalty);
    info['unfed_terminal_record'] is set. Any other valid record of an unexecuted decision raises EpisodeAlignmentError.
    Invalid policy record (infrastructure failure) -> invalid transition (no duration, no reward); it ends the episode.
    Forced record (forced RETURN: 'return_reserve', 'rejection_cap', 'max_decisions'): merged into the previous policy
    transition (ADR 008 section 5). It must start where that transition ends (within contracts.TIME_TOLERANCE_S). If it is
    valid the durations are summed, its events are appended with their own times and its terminated flag and reason
    replace the policy transition's. If it is invalid the merged transition becomes invalid: no reward and no terminal
    flag, and the policy transition before it becomes the GAE cut, bootstrapped from the merged decision's stored value.
    Reason (ADR 007 section 3): a failed macro is removed and the episode is truncated at the last valid transition
    without invented reward; the merged transition's own policy part cannot be kept separately, because its end state is
    the forced RETURN's observation, which is not a policy decision and has no value from the policy's GRU, so no
    bootstrap for a cut there would exist; the report events of that policy part are lost with it. A forced record
    without a preceding policy transition (forced at the first observation) contributes nothing and is counted.
    Episode end: reward.final_transition_end (shared with reward-v1) decides; when the records end with a valid
    non-terminal transition,
      - a terminal termination_reason (e.g. 'safety_abort' after an infeasible forced RETURN, which leaves no record)
        makes that transition terminal with that reason;
      - otherwise (infrastructure exit without an invalid record: policy/preparation or record error) the first policy
        decision after the last matched one that has an action becomes an invalid transition; without such a decision
        the episode is dropped (no rows) and info['dropped'] says why.
Rewards
    r_k = smdp.discount_reward_events over the merged events relative to t_k with the merged duration:
    r_k = sum_e gamma_ref ** ((t_e - t_k) / tau) * r_e. Invalid transitions carry reward None.
Buffer rows
    Keyword arguments of RolloutBuffer.add with the decision's obs, final mask, action index, log_prob, value and GRU input
    state; truncated is always False and next_value None: whole episodes are collected, so the last kept transition of
    every emitted episode is terminal or followed by an invalid transition (whose stored value is the bootstrap).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..mission.contracts import INFRASTRUCTURE, TIME_TOLERANCE_S, ActionType
from . import smdp
from .reward import final_transition_end


class EpisodeAlignmentError(ValueError):
    pass


@dataclass
class _Piece:
    decision: object
    valid: bool
    start_s: float
    duration_s: Optional[float]
    events: list
    terminated: bool
    reason: Optional[str]
    record_positions: List[int] = field(default_factory=list)
    forced: List[str] = field(default_factory=list)


@dataclass
class EpisodeTransitions:
    episode_id: int
    rows: list      # RolloutBuffer.add keyword arguments, in order
    meta: list      # per row: record positions, start/end, merged forced, reason, events before discounting
    info: dict

    @property
    def valid_rows(self):
        return sum(1 for row in self.rows if row['valid'])

    def add_to(self, buffer):
        for row in self.rows:
            buffer.add(**row)


def _matches(decision, record):
    return (decision.action is not None and decision.action == record.action
            and abs(decision.sim_time_s - record.observation.sim_time_s) <= TIME_TOLERANCE_S)


def build_episode_transitions(decisions, records, record_events, termination_reason, episode_id, gamma_ref, tau):
    """EpisodeTransitions for one episode (module docstring)."""
    records, decisions = list(records), list(decisions)
    if len(record_events) != len(records):
        raise EpisodeAlignmentError('record_events: one event list per record required')
    info = {'records': len(records), 'policy_records': 0, 'forced_records': 0, 'merged_forced': 0,
            'forced_without_policy_transition': 0, 'invalid_records': 0, 'abandoned_decisions': 0,
            'terminal_reason_applied': None, 'invalid_from_unrecorded_decision': False, 'unfed_terminal_record': False,
            'unmatched_trailing_decisions': 0, 'dropped': None, 'termination_reason': termination_reason}
    pieces, d = [], 0
    for position, record in enumerate(records):
        if pieces and (pieces[-1].terminated or not pieces[-1].valid):
            raise EpisodeAlignmentError(f'record {record.index} follows the end of the episode')
        if record.forced is not None:
            info['forced_records'] += 1
            if record.action.action_type != ActionType.RETURN:
                raise EpisodeAlignmentError(f'forced record {record.index} is not a RETURN')
            if not pieces:
                info['forced_without_policy_transition'] += 1
                if record_events[position] and not record.valid:
                    raise EpisodeAlignmentError('an invalid record carries no reward events')
                continue
            previous = pieces[-1]
            end = previous.start_s + previous.duration_s
            if abs(record.observation.sim_time_s - end) > TIME_TOLERANCE_S:
                raise EpisodeAlignmentError(f'forced record {record.index} starts at {record.observation.sim_time_s} s, '
                                            f'the previous policy transition ends at {end} s')
            previous.record_positions.append(position)
            previous.forced.append(record.forced)
            if record.valid:
                previous.duration_s += record.duration_s
                previous.events.extend(record_events[position])
                previous.terminated, previous.reason = record.terminated, record.reason
            else:
                if record_events[position]:
                    raise EpisodeAlignmentError('an invalid record carries no reward events')
                previous.valid, previous.duration_s, previous.events = False, None, []
                previous.terminated, previous.reason = False, INFRASTRUCTURE
                info['invalid_records'] += 1
            info['merged_forced'] += 1
            continue
        while d < len(decisions) and not _matches(decisions[d], record):
            if decisions[d].executed:
                raise EpisodeAlignmentError(f'executed policy decision {decisions[d].index} has no record')
            info['abandoned_decisions'] += 1
            d += 1
        if d == len(decisions):
            raise EpisodeAlignmentError(f'record {record.index} ({record.action.action_type.value} at '
                                        f'{record.observation.sim_time_s} s) has no policy decision')
        decision = decisions[d]
        d += 1
        info['policy_records'] += 1
        if record.valid:
            if not decision.executed:
                unfed = (position == len(records) - 1 and record.terminated and record.reason == 'safety_abort'
                         and not any(x.action is not None for x in decisions[d:]))
                if not unfed:
                    raise EpisodeAlignmentError(f'valid record {record.index} matches a decision that was not executed')
                info['unfed_terminal_record'] = True
            pieces.append(_Piece(decision, True, float(record.observation.sim_time_s), float(record.duration_s),
                                 list(record_events[position]), bool(record.terminated), record.reason, [position]))
        else:
            if record_events[position]:
                raise EpisodeAlignmentError('an invalid record carries no reward events')
            pieces.append(_Piece(decision, False, float(record.observation.sim_time_s), None, [], False, INFRASTRUCTURE, [position]))
            info['invalid_records'] += 1
    trailing = decisions[d:]
    if pieces and pieces[-1].valid and not pieces[-1].terminated:
        terminated, reason = final_transition_end(records, termination_reason)
        if terminated:
            pieces[-1].terminated, pieces[-1].reason = True, reason
            info['terminal_reason_applied'] = termination_reason
        else:
            candidate = next((x for x in trailing if x.action is not None), None)
            if candidate is None:
                info['dropped'] = (f'episode ended ({termination_reason}) after a non-terminal transition and no later policy '
                                   'decision with an action exists to mark the failure')
                pieces = []
            else:
                trailing = trailing[trailing.index(candidate) + 1:]
                pieces.append(_Piece(candidate, False, float(candidate.sim_time_s), None, [], False, INFRASTRUCTURE))
                info['invalid_from_unrecorded_decision'] = True
    info['unmatched_trailing_decisions'] = len(trailing)
    valid = [p for p in pieces if p.valid]
    rewards = smdp.discount_reward_events([p.events for p in valid], [p.start_s for p in valid], gamma_ref, tau,
                                          durations=[p.duration_s for p in valid]) if valid else []
    reward_of = {id(p): float(r) for p, r in zip(valid, rewards)}
    rows, meta = [], []
    for piece in pieces:
        decision = piece.decision
        rows.append(dict(obs=decision.obs, mask=decision.mask, action=int(decision.action_index), log_prob=decision.log_prob,
                         value=decision.value, state=decision.state, duration_s=piece.duration_s if piece.valid else None,
                         reward=reward_of[id(piece)] if piece.valid else None, terminated=piece.terminated if piece.valid else False,
                         truncated=False, valid=piece.valid, episode_id=int(episode_id)))
        meta.append({'decision_index': decision.index, 'record_positions': list(piece.record_positions), 'forced': list(piece.forced),
                     'start_s': piece.start_s, 'end_s': None if not piece.valid else piece.start_s + piece.duration_s,
                     'reason': piece.reason, 'events': [list(e) for e in piece.events], 'action_index': decision.action_index})
    if rows and rows[-1]['valid'] and not rows[-1]['terminated']:
        raise EpisodeAlignmentError('internal: the last transition of an episode must be terminal or invalid')
    info['rows'], info['valid_rows'] = len(rows), sum(1 for r in rows if r['valid'])
    return EpisodeTransitions(int(episode_id), rows, meta, info)


def discounted_return(transitions, gamma_ref, tau):
    """sum_k (prod_{j<k} gamma_j) r_k over the leading valid transitions (evaluation logging; stops at an invalid row)."""
    total, weight = 0., 1.
    for row in transitions.rows:
        if not row['valid']:
            break
        total += weight * row['reward']
        weight *= float(smdp.semi_markov_discount([row['duration_s']], gamma_ref, tau)[0])
    return total


__all__ = ['EpisodeAlignmentError', 'EpisodeTransitions', 'build_episode_transitions', 'discounted_return']
