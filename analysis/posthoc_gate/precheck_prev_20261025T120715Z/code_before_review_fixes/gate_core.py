"""E1 learned correction trigger: the pieces shared by runner.py, the tests and the analysis (see PLAN_DRAFT.md).

Imports the idea_dq code tree unchanged. Nothing here edits a file of idea_dq; `gnss_policies.py` is only subclassed.

    GatePolicy        the tuned rule (UncertaintyFix, built with e1_baselines.p3_rule()['parameters']) in which ONLY
                      should_fix() is delegated to a gate. It is called exactly where the rule calls it (just before
                      a report, when the per-report fix allowance is left); everything else (search, inspection,
                      evidence, report, merge, return) is the rule's own code.
    RuleGate          should_fix = rms >= sigma (1.5 m): the rule itself, run through the gate code path (precheck).
    NetworkGate       a GateActorCritic, sampled (training) or argmax (evaluation).
    GateActorCritic   small classical MLP actor-critic (no recurrence), 2 actions: 0 = report now, 1 = fix first.
    build_gate_segments   gate decisions + executive records + placed reward events -> SMDP rows for
                      smdp_ppo.RolloutBuffer (one row per executed gate decision).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True                       # never write __pycache__ into the idea_dq tree

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq')
for _extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import gnss_policies                                                                              # noqa: E402
from aerial_qrl.mission.contracts import INFRASTRUCTURE, TIME_TOLERANCE_S, ActionType            # noqa: E402
from aerial_qrl.training import smdp                                                              # noqa: E402
from aerial_qrl.training.reward import final_transition_end                                       # noqa: E402
from aerial_qrl.training.smdp_ppo import apply_action_mask                                        # noqa: E402

GATE_VERSION = 'e1-learned-correction-gate/1'
REPORT_NOW, FIX_FIRST = 0, 1
N_GATE_ACTIONS = 2
#: Declared feature scales (not tuned). T = the cell's mission budget (briefing), R = its return reserve.
TOLERANCE_SCALE_M = 3.0          # the evaluator's matching tolerance (encoding.TOLERANCE_SCALE_M)
NOMINAL_BUDGET_S = 2700.0
ANCHOR_SCALE_S = 100.0
COUNT_CAP = {'fixes': 10, 'reports': 20, 'detections': 20, 'viewpoints': 8, 'ready': 5, 'open': 10}
TRACK_SPREAD_SCALE_M = 10.0      # encoding.TRACK_SPREAD_SCALE_M
FEATURES = (
    'rms_horizontal_error_m/3',             # the rule's only input
    'min(time_since_fix_s,T)/T',
    'min(fixes_attempted,10)/10',
    'min(fixes_accepted,10)/10',
    'last_fix_accepted',
    'no_fix_yet',
    'elapsed_s/T',
    'remaining_time_s/T',
    'return_estimate_s/T',
    '(remaining-return_estimate-R)/T',
    'T/2700',
    'cheapest_anchor.estimated_duration_s/100',
    'cheapest_anchor.available',
    'share_of_anchors_reachable_and_available',
    'min(reports,20)/20',
    'candidate.detections(cap20)/20',
    'candidate.credited_viewpoints(cap8)/8',
    'candidate.max_score',
    'candidate.sqrt(cov_xx+cov_yy)/10',
    'candidate.age_s/T',
    'candidate.inspections/max_inspections_per_track',
    'other_rule_ready_open_tracks(cap5)/5',
    'open_tracks(cap10)/10',
    'plan_pending_fraction',
)
N_FEATURES = len(FEATURES)


class GateAlignmentError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------- decisions --------

@dataclass
class GateDecision:
    index: int
    sim_time_s: float
    observation: object               # the PublicObservation object it was asked on (identity used for re-asks)
    obs: np.ndarray                    # gate features
    mask: np.ndarray                   # final mask (True = allowed), [report_now, fix_first]
    action_index: int
    log_prob: float
    value: float
    logits: object = None
    state: np.ndarray = field(default_factory=lambda: np.zeros(1))
    action: object = None              # the contracts.Action the policy returned for this decision
    executed: Optional[bool] = None
    abandoned: bool = False
    awaiting_resample: bool = False
    rejections: list = field(default_factory=list)
    samples: int = 1
    info: dict = field(default_factory=dict)


class GateBook:
    """Gate-decision bookkeeping, separated from the policy so the re-ask logic can be unit tested.

    A RELOCALIZE the executive rejects is answered (same observation, asked again) by RESAMPLING the same decision
    with FIX masked, exactly as RLMissionPolicy resamples a rejected decision in place: the decision keeps its time,
    its final mask is [True, False], its action is REPORT_NOW and its log-prob is 0. A decision that is never fed
    back (or whose REPORT was rejected) is abandoned and never becomes a training row.
    """

    def __init__(self, gate):
        self.gate = gate
        self.decisions = []
        self.stats = {'gate_calls': 0, 'gate_fix': 0, 'resamples': 0, 'abandoned': 0, 'no_anchor_skips': 0,
                      'unmatched_feedback': 0}
        self._touched = None

    def last_open(self):
        d = self.decisions[-1] if self.decisions else None
        return d if d is not None and d.executed is None and not d.abandoned else None

    def ask(self, observation, features, context):
        """The gate's answer (True = fix first) for this observation; records or resamples a decision."""
        pending = self.last_open()
        if pending is not None and pending.awaiting_resample:
            if pending.observation is observation:
                mask = np.array([True, False])
                a, lp, v, logits = self.gate.decide(observation, pending.obs, mask, context)
                pending.mask, pending.action_index, pending.log_prob = mask, int(a), float(lp)
                pending.awaiting_resample, pending.samples = False, pending.samples + 1
                pending.action, self._touched = None, pending
                self.stats['resamples'] += 1
                if a != REPORT_NOW:
                    raise RuntimeError('a masked FIX was chosen')
                return False
            pending.abandoned = True
            self.stats['abandoned'] += 1
        elif pending is not None:
            pending.abandoned = True                  # asked again without feedback: the old decision never ran
            self.stats['abandoned'] += 1
        mask = np.array([True, True])
        a, lp, v, logits = self.gate.decide(observation, features, mask, context)
        d = GateDecision(len(self.decisions), float(observation.sim_time_s), observation, np.asarray(features, float),
                         mask, int(a), float(lp), float(v), logits, info=dict(context.get('info', {})))
        self.decisions.append(d)
        self._touched = d
        self.stats['gate_calls'] += 1
        self.stats['gate_fix'] += int(a == FIX_FIRST)
        return a == FIX_FIRST

    def bind(self, action):
        """Called by the policy's act() with the action it returns: binds it to the decision touched in this act."""
        if self._touched is not None:
            self._touched.action = action
            self._touched = None

    def on_feedback(self, action, executed, reason=None):
        d = self.last_open()
        if d is None or d.awaiting_resample or d.action is None or not (d.action is action or d.action == action):
            return
        if executed:
            d.executed = True
        elif action.action_type == ActionType.RELOCALIZE:
            d.awaiting_resample = True
            d.rejections.append(str(reason))
        else:
            d.executed, d.abandoned = False, True
            d.rejections.append(str(reason))
            self.stats['abandoned'] += 1


# ------------------------------------------------------------------------------------------------- the policy -------

class GatePolicy(gnss_policies.UncertaintyFix):
    """UncertaintyFix(sigma 1.5 m) with should_fix delegated to a gate; nothing else changes.

    act() is the rule's act() (super().act) with one bookkeeping line after it; feedback() is the rule's feedback()
    with one bookkeeping line before it. should_fix() is asked by the rule's act() exactly where the rule asks
    (base action REPORT, per-report fix allowance left). When no anchor is reachable the rule would report anyway,
    so the gate is not asked and nothing is recorded (counted in stats['no_anchor_skips']).
    """
    name = 'uncertainty_fix_learned_gate'

    def __init__(self, briefing, obstacle_map, config, reserve, gate=None, **rule_parameters):
        super().__init__(briefing, obstacle_map, config, reserve, **rule_parameters)
        if gate is None:
            raise ValueError('a GatePolicy needs a gate')
        self.book = GateBook(gate)
        self.T = float(briefing.budgets.mission_time_s)
        self.R = float(briefing.budgets.return_reserve_s)

    @property
    def decisions(self):
        return self.book.decisions

    @property
    def stats(self):
        return self.book.stats

    # -- pure helpers (no side effects on the rule's bookkeeping) --
    def _ready(self, track):
        c = self.config
        return track.detections >= c.report_min_detections and \
            len(self._credited.get(track.track_id, ())) >= c.report_min_views

    def _open_tracks(self, observation):
        return [t for t in sorted(observation.tracks, key=lambda t: t.track_id) if not self.is_reported(t, observation)]

    def candidate(self, observation):
        """The track the rule's REPORT is about to submit (EvidencePolicy._evidence_action, report branch)."""
        return next((t for t in self._open_tracks(observation) if self._ready(t)), None)

    def features(self, observation, anchor, track):
        o, geo, T = observation, observation.georeference, self.T
        open_tracks = self._open_tracks(o)
        ready = [t for t in open_tracks if self._ready(t)]
        anchors = list(o.anchors)
        good = sum(1 for a in anchors if a.reachable and a.available)
        pending = sum(1 for v in self.status.values() if v is None)
        cov = track.covariance_enu
        cap = COUNT_CAP
        x = np.array([
            geo.rms_horizontal_error_m / TOLERANCE_SCALE_M,
            min(geo.time_since_fix_s, T) / T,
            min(geo.fixes_attempted, cap['fixes']) / cap['fixes'],
            min(geo.fixes_accepted, cap['fixes']) / cap['fixes'],
            1. if geo.last_fix_accepted else 0.,
            1. if geo.last_fix_accepted is None else 0.,
            o.elapsed_s / T,
            o.remaining_time_s / T,
            o.return_estimate_s / T,
            (o.remaining_time_s - o.return_estimate_s - self.R) / T,
            T / NOMINAL_BUDGET_S,
            anchor.estimated_duration_s / ANCHOR_SCALE_S,
            float(anchor.available),
            good / len(anchors) if anchors else 0.,
            min(len(o.reports), cap['reports']) / cap['reports'],
            min(track.detections, cap['detections']) / cap['detections'],
            min(len(self._credited.get(track.track_id, ())), cap['viewpoints']) / cap['viewpoints'],
            float(track.max_score or 0.),
            math.sqrt(max(cov[0][0] + cov[1][1], 0.)) / TRACK_SPREAD_SCALE_M,
            track.age_s / T,
            self._inspections.get(track.track_id, 0) / max(self.config.max_inspections_per_track, 1),
            min(len(ready) - 1, cap['ready']) / cap['ready'],
            min(len(open_tracks), cap['open']) / cap['open'],
            pending / len(self.status) if self.status else 0.,
        ], dtype=np.float64)
        if x.shape != (N_FEATURES,) or not np.isfinite(x).all():
            raise ValueError(f'gate features: expected {N_FEATURES} finite values, got {x}')
        return x

    # -- the one replaced hook --
    def should_fix(self, observation):
        anchor = self._cheapest_anchor(observation)
        pending = self.book.last_open()
        if pending is not None and pending.awaiting_resample and pending.observation is observation:
            return self.book.ask(observation, pending.obs, {'policy': self})
        if anchor is None:
            self.book.stats['no_anchor_skips'] += 1
            return False
        track = self.candidate(observation)
        if track is None:
            raise GateAlignmentError('should_fix was asked without a rule-ready candidate track')
        x = self.features(observation, anchor, track)
        info = {'rms_m': observation.georeference.rms_horizontal_error_m, 'reports': len(observation.reports),
                'track_id': track.track_id, 'anchor_id': anchor.anchor_id, 'elapsed_s': observation.elapsed_s}
        return self.book.ask(observation, x, {'policy': self, 'info': info})

    def act(self, observation):
        action = super().act(observation)
        self.book.bind(action)
        return action

    def feedback(self, action, executed, reason=None, goal=None):
        self.book.on_feedback(action, executed, reason)
        super().feedback(action, executed, reason, goal)


# ------------------------------------------------------------------------------------------------- gates ------------

class RuleGate:
    """The tuned rule as a gate: FIX iff the published RMS error >= sigma (reads the observation, not the features)."""
    kind = 'rule'

    def __init__(self, sigma_threshold_m):
        self.sigma = float(sigma_threshold_m)

    def decide(self, observation, features, mask, context):
        fix = observation.georeference.rms_horizontal_error_m >= self.sigma
        a = FIX_FIRST if (fix and mask[FIX_FIRST]) else REPORT_NOW
        return a, 0.0, 0.0, None


class NetworkGate:
    """A GateActorCritic as the gate: sampled from its own torch.Generator (training) or argmax (evaluation)."""
    kind = 'network'

    def __init__(self, model, generator=None, deterministic=False, compute_values=None):
        if not deterministic and not isinstance(generator, torch.Generator):
            raise ValueError('sampling requires a torch.Generator')
        self.model, self.generator, self.deterministic = model, generator, bool(deterministic)
        self.compute_values = (not deterministic) if compute_values is None else bool(compute_values)
        p = next(model.parameters())
        self._dtype = p.dtype

    def decide(self, observation, features, mask, context):
        x = torch.as_tensor(np.asarray(features, float), dtype=self._dtype).reshape(1, 1, -1)
        m = torch.as_tensor(np.asarray(mask, bool)).reshape(1, 1, -1)
        with torch.no_grad():
            logits, value, _ = self.model(x, None, m)
        logits = logits.reshape(-1)
        dist = torch.distributions.Categorical(logits=logits)
        if self.deterministic:
            a = int(torch.argmax(logits))
        else:
            a = int(torch.multinomial(dist.probs, 1, generator=self.generator)[0])
        if not mask[a]:
            raise RuntimeError(f'gate chose infeasible action {a}')
        return a, float(dist.log_prob(torch.as_tensor(a))), float(value.reshape(-1)[0]) if self.compute_values else 0.0, \
            logits.detach().clone()


# ------------------------------------------------------------------------------------------------- the model --------

class GateActorCritic(nn.Module):
    """Separate MLP actor and critic on the gate features; no recurrence.

    The interface is what smdp_ppo.ppo_update / pad_segments need: forward(obs (B,T,F), state (1,B,1), mask) ->
    (masked logits, values, state); parameter_groups() for the per-group gradient clipping; a 1-wide all-zero
    placeholder state (RolloutBuffer requires one; it is never read). The actor's last layer is scaled by 0.01 so
    the initial gate is close to a fair coin.
    """

    def __init__(self, obs_dim=N_FEATURES, hidden=64, action_dim=N_GATE_ACTIONS):
        super().__init__()
        self.config = dict(kind='gate-mlp/1', obs_dim=int(obs_dim), hidden=int(hidden), action_dim=int(action_dim))
        self.hidden = 1                               # placeholder state width (RolloutBuffer / pad_segments)
        h = int(hidden)
        self.actor = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()

    def initial_state(self, batch):
        p = next(self.parameters())
        return torch.zeros(1, batch, 1, dtype=p.dtype)

    def forward(self, obs, state=None, mask=None):
        if obs.ndim != 3 or obs.shape[-1] != self.config['obs_dim']:
            raise ValueError(f'obs: expected (batch, time, {self.config["obs_dim"]}), got {tuple(obs.shape)}')
        logits = apply_action_mask(self.actor(obs), mask)
        values = self.critic(obs).squeeze(-1)
        return logits, values, self.initial_state(obs.shape[0]) if state is None else state

    def parameter_groups(self):
        return dict(trunk_actor=list(self.actor.parameters()), critic=list(self.critic.parameters()))

    def parameter_counts(self):
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)   # noqa: E731
        return dict(actor=count(self.actor), critic=count(self.critic), total=count(self))

    def describe(self):
        return dict(config=dict(self.config), parameter_counts=self.parameter_counts(),
                    actor=f'Linear({self.config["obs_dim"]},h)-Tanh-Linear(h,h)-Tanh-Linear(h,2), last layer x0.01',
                    critic=f'Linear({self.config["obs_dim"]},h)-Tanh-Linear(h,h)-Tanh-Linear(h,1)',
                    dtype=str(next(self.parameters()).dtype).removeprefix('torch.'), features=list(FEATURES))


# ------------------------------------------------------------------------------------------------- segments ---------

@dataclass
class GateSegments:
    episode_id: int
    rows: list
    meta: list
    info: dict

    @property
    def valid_rows(self):
        return sum(1 for r in self.rows if r['valid'])

    @property
    def kept_reward(self):
        return float(sum(r['reward'] for r in self.rows if r['valid']))

    def add_to(self, buffer):
        for row in self.rows:
            buffer.add(**row)


def _matches(decision, record):
    return (decision.action is not None and record.forced is None and decision.action == record.action
            and abs(decision.sim_time_s - record.observation.sim_time_s) <= TIME_TOLERANCE_S)


def build_gate_segments(decisions, records, record_events, termination_reason, episode_id, gamma_ref, tau):
    """One SMDP row per executed gate decision (PLAN_DRAFT.md, 'Segments').

    Segment k runs from gate decision k (its record) to the record of gate decision k+1; the last segment runs to
    the end of the episode's last record. Its reward is the sum of the placed GNSS reward events of its records
    (discounted to its start with gamma_ref; gamma_ref = 1 here). Records before the first gate carry reward that no
    gate decision caused: it is counted in info['pre_gate_reward'] and kept out of the rollout. A segment
    containing an invalid (infrastructure) record, or the last segment of an episode that ended in an
    infrastructure exit without an invalid record, is an invalid row (no reward, not terminal): the valid row
    before it becomes a GAE cut bootstrapped from its stored value (smdp.clean_rollout). The last valid segment of
    a terminated episode is terminal.
    """
    records, decisions = list(records), list(decisions)
    if len(record_events) != len(records):
        raise GateAlignmentError('one event list per record required')
    info = {'records': len(records), 'gate_decisions': len(decisions), 'abandoned': 0, 'unexecuted': 0,
            'pre_gate_reward': 0.0, 'invalid_segment_reward': 0.0, 'termination_reason': termination_reason,
            'terminal_applied': None, 'invalid_last_segment': None}
    matched, start = [], 0
    for d in decisions:
        if d.abandoned or d.action is None:
            info['abandoned'] += 1
            continue
        pos = next((i for i in range(start, len(records)) if _matches(d, records[i])), None)
        if pos is None:
            if d.executed and termination_reason != INFRASTRUCTURE:
                raise GateAlignmentError(f'executed gate decision {d.index} at {d.sim_time_s} s has no record')
            info['unexecuted'] += 1            # never ran, or ran into an infrastructure exit before its record
            continue
        if d.executed is False:
            raise GateAlignmentError(f'gate decision {d.index} was rejected but has a record')
        matched.append((d, pos))
        start = pos + 1
    first = matched[0][1] if matched else len(records)
    info['pre_gate_reward'] = float(sum(r for i in range(first) for _, r in record_events[i]))
    pieces = []
    for k, (d, pos) in enumerate(matched):
        stop = matched[k + 1][1] if k + 1 < len(matched) else len(records)
        span = records[pos:stop]
        events = [e for i in range(pos, stop) for e in record_events[i]]
        invalid = [r for r in span if not r.valid]
        last = k + 1 == len(matched)
        if invalid and (not last or invalid[0] is not span[-1]):
            raise GateAlignmentError('an invalid record must be the last record of the episode')
        if not last:
            duration = float(matched[k + 1][0].sim_time_s - d.sim_time_s)
            pieces.append(dict(decision=d, valid=True, start=d.sim_time_s, duration=duration, events=events,
                               terminated=False, reason=None, records=(pos, stop)))
            continue
        if invalid:
            pieces.append(dict(decision=d, valid=False, start=d.sim_time_s, duration=None, events=events,
                               terminated=False, reason=INFRASTRUCTURE, records=(pos, stop)))
            info['invalid_last_segment'] = 'invalid_record'
            continue
        end_record = span[-1]
        duration = float(end_record.observation.sim_time_s + end_record.duration_s - d.sim_time_s)
        terminated, reason = final_transition_end(records, termination_reason)
        if not terminated:
            # the episode stopped without a terminal record (an infrastructure exit in a later decision): the
            # segment's own outcome is unknown, so it becomes an invalid row (no reward), as episodes.py does
            pieces.append(dict(decision=d, valid=False, start=d.sim_time_s, duration=None, events=events,
                               terminated=False, reason=INFRASTRUCTURE, records=(pos, stop)))
            info['invalid_last_segment'] = f'unterminated:{termination_reason}'
            continue
        info['terminal_applied'] = reason
        pieces.append(dict(decision=d, valid=True, start=d.sim_time_s, duration=duration, events=events,
                           terminated=True, reason=reason, records=(pos, stop)))
    valid = [p for p in pieces if p['valid']]
    rewards = smdp.discount_reward_events([p['events'] for p in valid], [p['start'] for p in valid], gamma_ref, tau,
                                          durations=[p['duration'] for p in valid]) if valid else []
    reward_of = {id(p): float(r) for p, r in zip(valid, rewards)}
    info['invalid_segment_reward'] = float(sum(r for p in pieces if not p['valid'] for _, r in p['events']))
    rows, meta = [], []
    for p in pieces:
        d = p['decision']
        rows.append(dict(obs=d.obs, mask=np.asarray(d.mask, bool), action=int(d.action_index), log_prob=d.log_prob,
                         value=d.value, state=d.state, duration_s=p['duration'] if p['valid'] else None,
                         reward=reward_of[id(p)] if p['valid'] else None,
                         terminated=p['terminated'] if p['valid'] else False, truncated=False, valid=p['valid'],
                         episode_id=int(episode_id)))
        meta.append({'decision_index': d.index, 'start_s': p['start'], 'duration_s': p['duration'],
                     'records': list(p['records']), 'reason': p['reason'], 'action_index': int(d.action_index),
                     'samples': d.samples, 'events': [list(e) for e in p['events']]})
    if rows and rows[-1]['valid'] and not rows[-1]['terminated']:
        raise GateAlignmentError('internal: the last row of an episode must be terminal or invalid')
    info['rows'], info['valid_rows'] = len(rows), sum(1 for r in rows if r['valid'])
    return GateSegments(int(episode_id), rows, meta, info)


def check_segment_identity(segments, J, dropped, tolerance=1e-9):
    """kept + pre-gate + invalid-segment reward = J - dropped (gamma_ref = 1). Returns the residual; raises if off."""
    total = segments.kept_reward + segments.info['pre_gate_reward'] + segments.info['invalid_segment_reward']
    residual = total - (J - dropped)
    if abs(residual) > tolerance:
        raise ValueError(f'gate segments sum to {total!r}, J - dropped = {J - dropped!r}')
    return residual


# ------------------------------------------------------------------------------------------------- checkpoints ------

def save_gate_checkpoint(path, model, optimizer, config, metadata):
    payload = {'format': GATE_VERSION, 'model_state_dict': model.state_dict(),
               'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
               'model_config': dict(model.config), 'config': config, 'metadata': metadata, 'features': list(FEATURES)}
    with open(path, 'xb') as handle:
        torch.save(payload, handle)
    return path


def load_gate_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != GATE_VERSION or payload.get('features') != list(FEATURES):
        raise ValueError(f'{path}: not a {GATE_VERSION} checkpoint with these features')
    c = payload['model_config']
    model = GateActorCritic(c['obs_dim'], c['hidden'], c['action_dim']).double()
    model.load_state_dict(payload['model_state_dict'])
    return model.eval(), payload
