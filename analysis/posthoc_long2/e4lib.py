"""E4 patch layer (see PLAN_DRAFT.md). Imports the B2 code tree of idea_dq UNCHANGED and, in its own process only,
swaps four module globals of p4_train, exactly as the LONG/B2G runners swap one:

    p4.LAYOUT_V4_GNSS   obs-encoding/4-gnss (254)  ->  WIDE = obs-encoding/4-gnss+ev16/1 (254 + 16)
    p4.RLMissionPolicy  RLMissionPolicy            ->  InfoRLMissionPolicy (the rule's own evidence bookkeeping)
    p4.build_model      dq_b2.dq_build_model       ->  build_model_e4 (LONG's model, widened by 16 zero columns)
    p4.ppo_update       smdp_ppo.ppo_update        ->  ppo_update_e4 (learning rate set before each update)

and sets p4.BLOCK_STRIDE = 10,000 (LONG's stride). `configure(info, lr0, horizon)` selects, per process, either the
pass-through path (info False, lr0 None: every wrapper calls the frozen code unchanged; this is the path precheck (a)
compares bit for bit with the stored LONG runs) or the E4 path.

Appended features (16), all computed from public quantities by the rule's own code (executive.EvidencePolicy):
    254 + i   (i = 0..7)   track slot i: min(credited distinct viewpoints, 4) / 4   (EvidencePolicy._update_evidence)
    262 + i   (i = 0..7)   track slot i: min(executed inspections, 3) / 3          (EvidencePolicy._evidence_feedback)
    (0 for an empty slot). Slots are the base encoder's own track slots (unreported tracks, its order).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
from pathlib import Path
import sys

IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq')
for _extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import numpy as np                                                                                # noqa: E402
import torch                                                                                      # noqa: E402

import dq_b2                                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import p4_train as p4                                                                             # noqa: E402
from aerial_qrl.mission.leakage import FORBIDDEN_TOKENS, key_tokens                               # noqa: E402
from aerial_qrl.training import encoding as enc                                                   # noqa: E402
from aerial_qrl.training.encoding import ObservationEncoder                                       # noqa: E402
from aerial_qrl.training.rl_policy import RLMissionPolicy                                         # noqa: E402
from aerial_qrl.training.smdp_ppo import RecurrentActorCritic                                     # noqa: E402

EXPECTED_TREE = 'ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952'
LONG_DIR = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_long')
B2_RUNS = IDEA_DQ / 'output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs'

# ------------------------------------------------------------------ LONG's design, reused ----------------------------
BLOCKS = (4, 5, 6, 7, 8, 9)
BUDGET = 300_000
GRID = list(range(0, BUDGET + 1, 30_000))
STRIDE = 10_000
ARMS = ('mlp', 'bottleneck')
ROLES = {'mlp': 'C-strong', 'bottleneck': 'C-matched'}
#: C-strong's E1 selection, added to the B2 arm table in this process only (identical to LONG runner.py:34)
dq_b2.ARMS.setdefault('mlp', {'learning_rate': 3e-3, 'entropy_coef': 0.0, 'role': 'C-strong', 'candidate': 'lr3e-3_ent0'})
p4.BLOCK_STRIDE = STRIDE
#: lr0 = each head's E1 selection (both 3e-3); decays linearly to 0 over the main budget
LR0 = {'mlp': 3e-3, 'bottleneck': 3e-3}
LR_HORIZON = BUDGET
ENTROPY_CANDIDATES = (0.003, 0.01)

# ------------------------------------------------------------------ the information-matched observation --------------
BASE = enc.LAYOUT_V4_GNSS
VIEWPOINT_CAP = 4
OBS_VERSION = 'obs-encoding/4-gnss+ev16/1'


@dataclass(frozen=True)
class WideLayout(enc.EncodingLayout):
    """obs-encoding/4-gnss with `extra_features` appended after its last feature; every offset and action unchanged."""
    extra_features: int = 0

    @property
    def evidence_offset(self):
        return enc.EncodingLayout.dim.fget(self)

    @property
    def dim(self):
        return enc.EncodingLayout.dim.fget(self) + self.extra_features


K = 2 * BASE.track_slots
WIDE = WideLayout(OBS_VERSION, BASE.action_version, BASE.track_slots, BASE.view_slots, BASE.report_cap,
                  BASE.exclude_reported_tracks, BASE.score_feature, anchor_slots=BASE.anchor_slots, extra_features=K)
enc.LAYOUTS.setdefault(WIDE.obs_version, WIDE)
if enc.LAYOUTS[WIDE.obs_version] is not WIDE:
    raise RuntimeError(f'another layout named {WIDE.obs_version} is registered in this process')
EXTRA_FEATURE_NAMES = tuple([f'evidence.track{i}.credited_viewpoints_capped{VIEWPOINT_CAP}_over_{VIEWPOINT_CAP}'
                             for i in range(BASE.track_slots)] +
                            [f'evidence.track{i}.inspections_capped_over_cap' for i in range(BASE.track_slots)])
assert len(EXTRA_FEATURE_NAMES) == K and WIDE.dim == BASE.dim + K == 270
assert not any(key_tokens(n) & FORBIDDEN_TOKENS for n in EXTRA_FEATURE_NAMES)
EV_STATS = ('ev_encodes', 'ev_viewpoint_feature_nonzero', 'ev_inspection_feature_nonzero', 'ev_slots_ge2_viewpoints',
            'ev_feedback_errors')


def fill_evidence(x, tracks, credited, inspections, layout, inspection_cap, stats=None):
    """Write the 16 evidence features of the slotted `tracks` into x[layout.evidence_offset:]; returns nothing."""
    off, n = layout.evidence_offset, layout.track_slots
    for i, track in enumerate(tracks):
        if track is None:
            continue
        x[off + i] = min(len(credited.get(track.track_id, ())), VIEWPOINT_CAP) / VIEWPOINT_CAP
        x[off + n + i] = min(inspections.get(track.track_id, 0), inspection_cap) / inspection_cap
    if stats is not None:
        stats['ev_encodes'] += 1
        stats['ev_viewpoint_feature_nonzero'] += bool((x[off:off + n] > 0).any())
        stats['ev_inspection_feature_nonzero'] += bool((x[off + n:off + 2 * n] > 0).any())
        stats['ev_slots_ge2_viewpoints'] += int((x[off:off + n] >= 2 / VIEWPOINT_CAP).sum())


class EvidenceEncoder(ObservationEncoder):
    """The frozen obs-encoding/4-gnss encoder for features 0..253 (byte for byte), then the owner's evidence state."""

    def __init__(self, briefing, config, layout, anchor_ids, owner, stats=None):
        if not isinstance(layout, WideLayout) or layout.extra_features != K:
            raise ValueError(f'EvidenceEncoder needs {OBS_VERSION}')
        super().__init__(briefing, config, layout, anchor_ids=anchor_ids)
        self.owner, self.stats = owner, stats
        self.inspection_cap = int(config.max_inspections_per_track)
        if self.inspection_cap <= 0:
            raise ValueError('max_inspections_per_track must be positive')

    def encode(self, observation, plan_state):
        x, info = super().encode(observation, plan_state)
        fill_evidence(x, info.slots.tracks, self.owner._credited, self.owner._inspections, self.layout,
                      self.inspection_cap, self.stats)
        return x, info


class InfoRLMissionPolicy(RLMissionPolicy):
    """RLMissionPolicy that keeps the rule's evidence state (it inherits EvidencePolicy but never updated it) and
    appends it to its observation. Action choice, masks, rejection handling and feedback are RLMissionPolicy's."""

    def __init__(self, briefing, obstacle_map, config, reserve, model, **kwargs):
        super().__init__(briefing, obstacle_map, config, reserve, model, **kwargs)
        if self.layout is not WIDE:
            raise ValueError(f'InfoRLMissionPolicy runs {OBS_VERSION} only, not {self.layout.obs_version}')
        self.stats.update(dict.fromkeys(EV_STATS, 0))
        self.encoder = EvidenceEncoder(briefing, config, self.layout, self.anchor_ids, owner=self, stats=self.stats)

    def act(self, observation):
        self._update_evidence(observation)       # as GnssRulePolicy.act; idempotent for a re-asked observation
        return super().act(observation)

    def feedback(self, action, executed, reason=None, goal=None):
        try:                                     # feedback never raises (the executive calls it outside its handling)
            self._evidence_feedback(action, executed, goal)
        except Exception:                        # noqa: BLE001 - counted, logged per episode in policy_stats
            self.stats['ev_feedback_errors'] += 1
        return super().feedback(action, executed, reason, goal)


# ------------------------------------------------------------------ model: LONG's, widened by zero columns ----------
_B2_BUILD = p4.build_model
if _B2_BUILD is not dq_b2.dq_build_model:
    raise RuntimeError('p4.build_model is not dq_b2.dq_build_model: import order changed')
_ORIGINAL = {'layout': p4.LAYOUT_V4_GNSS, 'policy': p4.RLMissionPolicy, 'build': p4.build_model,
             'ppo_update': p4.ppo_update}
if _ORIGINAL['layout'] is not BASE or _ORIGINAL['policy'] is not RLMissionPolicy:
    raise RuntimeError('p4_train globals are not the frozen ones at import')


@contextlib.contextmanager
def _base_layout():
    saved = p4.LAYOUT_V4_GNSS
    p4.LAYOUT_V4_GNSS = BASE
    try:
        yield
    finally:
        p4.LAYOUT_V4_GNSS = saved


def build_base(arm, config, block):
    """The frozen build (dq_b2.dq_build_model -> p4_train.build_model) under the 254-feature layout: LONG's model."""
    with _base_layout():
        return _B2_BUILD(arm, config, block)


def widen(base, k):
    """`base` with k zero columns appended to encoder.0.weight; every other tensor copied. The wide module is built
    inside fork_rng, so the global torch generator is left exactly where the base build left it."""
    cfg = dict(base.config)
    dtype = next(base.parameters()).dtype
    with torch.random.fork_rng(devices=[]):
        wide = RecurrentActorCritic(cfg['obs_dim'] + k, cfg['action_dim'], cfg['critic'], hidden=cfg['hidden'],
                                    critic_gradient=cfg['critic_gradient'], memory=cfg['memory'],
                                    actor_head=cfg['actor_head']).to(dtype=dtype)
    state = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    w = state['encoder.0.weight']
    state['encoder.0.weight'] = torch.cat([w, torch.zeros(w.shape[0], k, dtype=w.dtype)], dim=1)
    wide.load_state_dict(state, strict=True)
    wide.train(base.training)
    return wide


STATE = {'info': False}


def build_model_e4(arm, config, block):
    base = build_base(arm, config, block)
    return widen(base, K) if STATE['info'] else base


# ------------------------------------------------------------------ PPO: linear learning-rate decay ------------------
SCHED = {'lr0': None, 'horizon': None, 'done': 0}


def scheduled_lr(lr0, horizon, done):
    return lr0 * max(0.0, 1.0 - done / horizon)


def ppo_update_e4(model, optimizer, batch, clip_range, value_coef, entropy_coef, max_grad_norm, epochs, **kwargs):
    """smdp_ppo.ppo_update, unchanged; with a schedule, the Adam learning rate of every parameter group is first set to
    lr0 * (1 - D / horizon), D = policy decisions consumed by the earlier updates of this run."""
    if SCHED['lr0'] is None:
        return _ORIGINAL['ppo_update'](model, optimizer, batch, clip_range, value_coef, entropy_coef, max_grad_norm,
                                       epochs, **kwargs)
    done = SCHED['done']
    lr = scheduled_lr(SCHED['lr0'], SCHED['horizon'], done)
    for group in optimizer.param_groups:
        group['lr'] = lr
    metrics = _ORIGINAL['ppo_update'](model, optimizer, batch, clip_range, value_coef, entropy_coef, max_grad_norm,
                                      epochs, **kwargs)
    SCHED['done'] = done + int(metrics['transitions'])
    metrics.update(learning_rate=lr, lr_schedule=f'linear lr0={SCHED["lr0"]!r} horizon={SCHED["horizon"]}',
                   decisions_before_update=done, entropy_coef=float(entropy_coef))
    return metrics


def configure(info, lr0=None, horizon=None):
    """Per-process switch. info False and lr0 None: pass-through (LONG's code path); otherwise the E4 path."""
    STATE['info'] = bool(info)
    p4.LAYOUT_V4_GNSS = WIDE if info else BASE
    p4.RLMissionPolicy = InfoRLMissionPolicy if info else _ORIGINAL['policy']
    p4.build_model = build_model_e4
    p4.ppo_update = ppo_update_e4
    if (lr0 is None) != (horizon is None):
        raise ValueError('lr0 and horizon go together')
    SCHED.update(lr0=None if lr0 is None else float(lr0), horizon=None if horizon is None else int(horizon), done=0)
    return {'info': STATE['info'], 'obs_version': p4.LAYOUT_V4_GNSS.obs_version, 'obs_dim': p4.LAYOUT_V4_GNSS.dim,
            'lr0': SCHED['lr0'], 'lr_horizon': SCHED['horizon']}


def restore_originals():
    """Every p4_train global back to what it was at import (tests)."""
    p4.LAYOUT_V4_GNSS, p4.RLMissionPolicy = _ORIGINAL['layout'], _ORIGINAL['policy']
    p4.build_model, p4.ppo_update = _ORIGINAL['build'], _ORIGINAL['ppo_update']
    STATE['info'] = False
    SCHED.update(lr0=None, horizon=None, done=0)


# ------------------------------------------------------------------ the development study (pilot only) --------------
DEV_SALT = (dq.DQ_SALT, 41)
DEV_BLOCKS = (6, 7)
DEV_REALIZATIONS = tuple(range(600, 610))
PILOT_BUDGET = 100_000
PILOT_GRID = [0, 50_000, 80_000, 100_000]
SELECTION_POINTS = (80_000, 100_000)
SELECTION_TIE_J = dq_b2.TOST_DELTA            # 0.0056 J = 0.1 report-equivalent per episode
COLLAPSE_ENTROPY_NATS = 0.30
STUDY_DEV = p4.Study(
    name='e4_dev', tool='gnss-dq-posthoc-e4-pilot/1', salt=DEV_SALT, training_namespace='dq_development',
    extra_namespaces=dq.DQ_NAMESPACES, blocks={'e4_dev': DEV_BLOCKS},
    suites={'e4_dev_evaluation': ('dq_development', DEV_REALIZATIONS)},
    roles={'e4_dev': ('e4_dev', 'e4_dev_evaluation', PILOT_BUDGET, PILOT_GRID)}, candidates={},
    evaluation_suites=('e4_dev_evaluation',),
    non_claim=('Development pilot of post-hoc check E4 (dq_development seeds only); it selects the entropy '
               'coefficient by the rule fixed in PLAN_DRAFT.md and supports no claim.'))
p4.STUDIES.setdefault('e4_dev', STUDY_DEV)
if p4.STUDIES['e4_dev'] is not STUDY_DEV:
    raise RuntimeError('another e4_dev study is registered in this process')


def config(arm, block, study, budget, grid, entropy, lr0, freeze_sha256, note):
    """dq_b2.config_for (LONG's configuration) with only the entropy coefficient and the initial learning rate set."""
    c = dq_b2.config_for(arm, block, study, budget, grid, freeze_sha256=freeze_sha256)
    c['ppo'] = dict(c['ppo'], learning_rate=float(lr0), entropy_coef=float(entropy))
    c['note'] = note
    return c


def candidate_label(entropy, lr0, info, schedule):
    return f'lr{lr0:g}{"lin" if schedule else ""}_ent{entropy:g}' + ('_ev16' if info else '')


def seed_range(study, rt, block):
    space = rt.namespaces[study.training_namespace]
    return space['start'] + block * p4.BLOCK_STRIDE, space['start'] + (block + 1) * p4.BLOCK_STRIDE, space


__all__ = ['ARMS', 'BASE', 'BLOCKS', 'BUDGET', 'DEV_BLOCKS', 'ENTROPY_CANDIDATES', 'EXTRA_FEATURE_NAMES', 'GRID', 'K',
           'LR0', 'LR_HORIZON', 'OBS_VERSION', 'STUDY_DEV', 'WIDE', 'EvidenceEncoder', 'InfoRLMissionPolicy',
           'WideLayout', 'build_base', 'build_model_e4', 'configure', 'fill_evidence', 'ppo_update_e4',
           'restore_originals', 'scheduled_lr', 'widen']
