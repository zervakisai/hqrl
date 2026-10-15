"""RL mission policy for the executive (ADR 008 sections 1, 3, 5; ADR 007 section 4).

RLMissionPolicy is one more executive.MissionPolicy: it chooses where to spend time (which sector to search, which view to
inspect, which track to report, when to return). The shared planner builds goals and the shared executive flies them.
Inputs: PublicObservation, MissionBriefing, the public obstacle map and MissionConfig, the ReturnReserve object handed to
every policy (only used to build the coverage plan) and its own plan state. No evaluator object is accepted anywhere.

Decision cycle
    act(observation) for a new observation: encode it (encoding.ObservationEncoder, obs-encoding/2) with the current plan
    state, build the public mask (encoding.ActionSpace, action-space/1, under the caller's return_mask_rule: 'always' by
    default, and 'search_exhausted' masks a voluntary RETURN while a sector still has a pending search node, counted in
    stats['return_masked_by_rule']), run the RecurrentActorCritic once from the GRU
    state after the last executed policy decision (zero at the episode start), then sample from the masked categorical
    with the policy's torch.Generator (training) or take the argmax of the masked logits (deterministic evaluation; ties
    go to the lowest index). A PolicyDecision is appended to .decisions: observation vector, public mask, final mask,
    action index and contracts.Action, log_prob under the final mask, value V(s_k), GRU input state and overflow counts.
    In deterministic mode only the actor runs (RecurrentActorCritic.policy) unless compute_values=True, so value is None.
    The actor head is the model's (smdp_ppo actor_head); a quantum head executes with the finite-shot setting the caller gave
    it (actor_heads.QuantumActorHead.set_shots, e.g. run_mission.build_rl_policy with mission.rl.actor_shots), analytically
    otherwise. describe() records the actor head and that setting ('actor_shots': {shots, seed} or None).
    feedback(action, False, reason, goal): the executive rejected the pending decision's current action. Its index is
    masked for that observation only and the rejection is counted (by reason and action type). A rejected RETURN
    (index 15) is masked only while at least one other action stays feasible for that observation (public mask and not
    rejected); when nothing else is feasible RETURN is kept, so the mask is never empty (a RETURN rejected again then
    reaches the executive's max_consecutive_rejections cap, which forces RETURN). The same guard covers a RETURN the
    return mask rule made publicly infeasible: when every other action of that observation is rejected, RETURN is
    restored. A rejection creates no transition and costs no time.
    act(same observation) after a rejection: resample from the stored raw logits with the updated mask; no new forward
    pass (same observation, plan state and GRU state), so obs, value and GRU input state are unchanged and log_prob is
    recomputed under the final mask (ADR 008 section 3).
    feedback(action, True, None, goal) for the pending decision's current action: the decision is executed and the GRU
    state advances to its output state. Any executed SEARCH_SECTOR marks its goal waypoint flown. An executed action that
    act() did not return (the executive's forced RETURN) is counted in stats['forced_feedback'] and changes nothing else.
    feedback never raises (the executive calls it outside its failure handling); anomalies are counted in stats.
    An incomplete command (executive._incomplete) ends the episode with a valid terminal 'safety_abort' record and no
    feedback: that decision stays unexecuted and episodes.build_episode_transitions assigns the record to it.

Search goals (ADR 008 section 3)
    The plan is taken from an executive.CoverageBaselinePolicy constructed with the same public inputs, so the
    CoverageBaseline plan and the initial waypoint status (plan-issue waypoints excluded) are identical.
    search_goal(sector_id) returns the first pending waypoint of the sector in plan order as SearchGoal(f'wp{index}', ...),
    exactly as CoverageBaselinePolicy.search_goal. Unlike CoverageBaselinePolicy, a rejected waypoint stays pending: only
    executed feedback advances the plan state, and a rejection only masks the action for that observation.

Report semantics (ADR 008 section 3): the merge radius rule is the shared executive.EvidencePolicy.is_reported (encoding
mirrors it in track_is_reported); when to report is this policy's choice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

from ..mission.contracts import Action, ActionType, PublicObservation
from ..mission.executive import CoverageBaselinePolicy, EvidencePolicy, SearchGoal
from .encoding import LAYOUTS, ActionSlot, ActionSpace, ObservationEncoder, get_layout, plan_state_from_coverage
from .smdp_ppo import RecurrentActorCritic, apply_action_mask

POLICY_NAME = 'rl_smdp_ppo'


@dataclass(eq=False)
class PolicyDecision:
    """One policy decision (one observation); see the module docstring. Arrays are float64 / bool copies."""
    index: int
    sim_time_s: float
    observation: PublicObservation = field(repr=False)
    obs: np.ndarray = field(repr=False)
    public_mask: np.ndarray = field(repr=False)
    state: np.ndarray = field(repr=False)             # GRU input state (hidden,)
    value: Optional[float]
    overflow_tracks: int
    overflow_views: int
    views_without_track_slot: int
    logits: torch.Tensor = field(repr=False)          # raw (unmasked) actor logits (number of actions,)
    next_state: torch.Tensor = field(repr=False)      # GRU output state (1, 1, hidden)
    rejected: np.ndarray = field(default=None, repr=False)
    mask: Optional[np.ndarray] = field(default=None, repr=False)   # final mask = public_mask & ~rejected
    action_index: Optional[int] = None
    action: Optional[Action] = None
    slot: Optional[ActionSlot] = None
    log_prob: Optional[float] = None
    rejections: list = field(default_factory=list)
    samples: int = 0
    awaiting_resample: bool = False
    executed: bool = False
    abandoned: bool = False

    def __post_init__(self):
        if self.rejected is None:
            self.rejected = np.zeros(len(self.public_mask), dtype=np.bool_)

    def to_record(self):
        """JSON-serialisable summary (no observation, no arrays except the masks as index lists)."""
        return {'index': self.index, 'sim_time_s': self.sim_time_s, 'action_index': self.action_index,
                'action': None if self.action is None else self.action.to_dict(), 'log_prob': self.log_prob, 'value': self.value,
                'public_feasible': np.flatnonzero(self.public_mask).tolist(),
                'final_feasible': None if self.mask is None else np.flatnonzero(self.mask).tolist(),
                'rejections': list(self.rejections), 'samples': self.samples, 'executed': self.executed, 'abandoned': self.abandoned,
                'overflow_tracks': self.overflow_tracks, 'overflow_views': self.overflow_views}


def layout_for_model(model, obs_encoding=None):
    """EncodingLayout for a model: the named version (checked against the model's dimensions), or the unique layout whose
    dimensions match the model."""
    dims = (model.config['obs_dim'], model.config['action_dim'])
    if obs_encoding is not None:
        layout = get_layout(obs_encoding)
        if (layout.dim, layout.n_actions) != dims:
            raise ValueError(f'model: {layout.obs_version} expects obs_dim {layout.dim} and action_dim {layout.n_actions}, got {model.config}')
        return layout
    matches = [layout for layout in LAYOUTS.values() if (layout.dim, layout.n_actions) == dims]
    if len(matches) != 1:
        raise ValueError(f'model: no unique observation encoding with obs_dim {dims[0]} and action_dim {dims[1]}')
    return matches[0]


class RLMissionPolicy(EvidencePolicy):
    """RecurrentActorCritic policy over action-space/1 or /2 under a RETURN mask rule (module docstring; the layout follows
    the model or obs_encoding, the rule the caller: run_mission.build_rl_policy takes it from the checkpoint)."""
    name = POLICY_NAME

    def __init__(self, briefing, obstacle_map, config, reserve, model, generator=None, deterministic=False, compute_values=None,
                 obs_encoding=None, return_mask_rule=None, anchor_ids=None):
        super().__init__(briefing, obstacle_map, config, reserve)
        if not isinstance(model, RecurrentActorCritic):
            raise TypeError('model: expected smdp_ppo.RecurrentActorCritic')
        self.layout = layout_for_model(model, obs_encoding)
        if not deterministic and not isinstance(generator, torch.Generator):
            raise ValueError('sampling requires a torch.Generator')
        # flood-sar-village-gnss/1: the anchor slot mapping is part of the frozen schema, so a GNSS layout requires
        # the declared order and the encoder refuses an observation that offers the anchors in another one.
        if self.layout.anchor_slots and anchor_ids is None:
            raise ValueError(f'{self.layout.obs_version} requires the declared anchor order (anchor_ids)')
        if not self.layout.anchor_slots and anchor_ids is not None:
            raise ValueError(f'{self.layout.obs_version} has no anchor slots; anchor_ids must be None')
        self.anchor_ids = None if anchor_ids is None else tuple(anchor_ids)
        self.encoder = ObservationEncoder(briefing, config, self.layout, anchor_ids=self.anchor_ids)
        self.action_space = ActionSpace(briefing, config, self.layout, return_mask_rule, anchor_ids=self.anchor_ids)
        coverage = CoverageBaselinePolicy(briefing, obstacle_map, config, reserve)  # identical plan construction
        self.plan, self.status = coverage.plan, dict(coverage.status)
        self.sector_ids = self.encoder.sector_ids
        self.model, self.generator, self.deterministic = model, generator, bool(deterministic)
        self.compute_values = (not self.deterministic) if compute_values is None else bool(compute_values)
        parameter = next(model.parameters())
        self._dtype, self._device = parameter.dtype, parameter.device
        self._state = model.initial_state(1)
        self.decisions = []
        self.stats = {'decisions': 0, 'samples': 0, 'resamples': 0, 'rejections': 0, 'rejections_by_reason': {},
                      'rejections_by_action_type': {}, 'executed': 0, 'forced_feedback': 0, 'abandoned_decisions': 0,
                      'unmatched_rejection_feedback': 0, 'unknown_goal_feedback': 0, 'overflow_tracks_total': 0,
                      'overflow_views_total': 0, 'views_without_track_slot_total': 0, 'views_of_reported_tracks_total': 0,
                      'return_masked_after_rejection': 0, 'return_masked_by_rule': 0, 'decisions_with_track_overflow': 0,
                      'decisions_with_view_overflow': 0, 'max_tracks': 0, 'max_views': 0}

    def describe(self):
        return {'policy': self.name, 'obs_encoding': self.layout.obs_version, 'action_space': self.action_space.version,
                'anchor_order': None if self.anchor_ids is None else list(self.anchor_ids),
                'return_mask_rule': self.action_space.return_mask_rule,
                'deterministic': self.deterministic, 'compute_values': self.compute_values,
                'plan_waypoints': len(self.plan.waypoints), 'plan_issues': len(self.plan.issues),
                'actor_head': self.model.config['actor_head'], 'actor_shots': self.model.actor_shots()}

    # -- plan --
    def plan_state(self):
        return plan_state_from_coverage(self.plan, self.status, self.sector_ids)

    def search_goal(self, sector_id):
        w = next((w for w in self.plan.waypoints if self.status[w.index] is None and w.sector_id == sector_id), None)
        return None if w is None else SearchGoal(f'wp{w.index}', w.sector_id, w.position_enu, w.yaw_rad)

    # -- decisions --
    def _pending(self):
        pending = self.decisions[-1] if self.decisions else None
        return pending if pending is not None and not pending.executed and not pending.abandoned else None

    def act(self, observation):
        pending = self._pending()
        if pending is not None:
            if pending.awaiting_resample and (observation is pending.observation or observation == pending.observation):
                self.stats['resamples'] += 1
                self._choose(pending)
                return pending.action
            pending.abandoned = True
            self.stats['abandoned_decisions'] += 1
        plan_state = self.plan_state()
        obs, info = self.encoder.encode(observation, plan_state)
        public_mask = self.action_space.masks(observation, plan_state)
        x = torch.as_tensor(obs, dtype=self._dtype, device=self._device).reshape(1, 1, self.layout.dim)
        state_in = self._state
        with torch.no_grad():
            if self.compute_values:
                logits, value, next_state = self.model(x, state_in)
                value = float(value.reshape(-1)[0])
            else:
                logits, next_state = self.model.policy(x, state_in)
                value = None
        decision = PolicyDecision(len(self.decisions), float(observation.sim_time_s), observation, obs, public_mask,
                                  state_in.detach().cpu().numpy().reshape(-1).astype(np.float64), value, info.overflow_tracks,
                                  info.overflow_views, info.views_without_track_slot, logits.reshape(self.layout.n_actions), next_state)
        s = self.stats
        s['decisions'] += 1
        s['overflow_tracks_total'] += info.overflow_tracks
        s['overflow_views_total'] += info.overflow_views
        s['views_without_track_slot_total'] += info.views_without_track_slot
        s['views_of_reported_tracks_total'] += info.views_of_reported_tracks
        s['decisions_with_track_overflow'] += info.overflow_tracks > 0
        s['decisions_with_view_overflow'] += info.overflow_views > 0
        s['return_masked_by_rule'] += not public_mask[self.layout.return_action]
        s['max_tracks'], s['max_views'] = max(s['max_tracks'], info.tracks), max(s['max_views'], info.views)
        self.decisions.append(decision)
        self._choose(decision)
        return decision.action

    def _choose(self, decision):
        mask = decision.public_mask & ~decision.rejected
        return_action = self.layout.return_action
        if not mask[:return_action].any():
            mask[return_action] = True        # nothing else feasible: a rejected or rule-masked RETURN is kept
        elif decision.rejected[return_action]:
            self.stats['return_masked_after_rejection'] += 1
        masked = apply_action_mask(decision.logits, torch.as_tensor(mask, device=decision.logits.device))
        distribution = Categorical(logits=masked)
        if self.deterministic:
            index = int(torch.argmax(masked))
        else:
            index = int(torch.multinomial(distribution.probs, 1, generator=self.generator)[0])
        if not mask[index]:
            raise RuntimeError(f'chose action {index}, infeasible under its mask')
        action, slot = self.action_space.decode(index, decision.observation)
        decision.mask, decision.action_index, decision.action, decision.slot = mask.copy(), index, action, slot
        decision.log_prob = float(distribution.log_prob(torch.as_tensor(index, device=masked.device)))
        decision.awaiting_resample = False
        decision.samples += 1
        self.stats['samples'] += 1

    def feedback(self, action, executed, reason=None, goal=None):
        pending = self._pending()
        current = pending is not None and not pending.awaiting_resample and pending.action == action
        s = self.stats
        if not executed:
            if not current:
                s['unmatched_rejection_feedback'] += 1
                return
            pending.rejections.append({'action_index': pending.action_index, 'action_type': action.action_type.value, 'reason': reason})
            pending.rejected[pending.action_index] = True
            pending.awaiting_resample = True
            s['rejections'] += 1
            key = str(reason).split(':')[0]
            s['rejections_by_reason'][key] = s['rejections_by_reason'].get(key, 0) + 1
            s['rejections_by_action_type'][action.action_type.value] = s['rejections_by_action_type'].get(action.action_type.value, 0) + 1
            return
        if action.action_type == ActionType.SEARCH_SECTOR and goal is not None:
            goal_id = getattr(goal, 'goal_id', '')
            index = int(goal_id[2:]) if isinstance(goal_id, str) and goal_id.startswith('wp') and goal_id[2:].isdigit() else None
            if index in self.status:
                self.status[index] = 'flown'
            else:
                s['unknown_goal_feedback'] += 1
        if current:
            pending.executed = True
            self._state = pending.next_state
            s['executed'] += 1
        else:
            s['forced_feedback'] += 1


__all__ = ['POLICY_NAME', 'PolicyDecision', 'RLMissionPolicy']
