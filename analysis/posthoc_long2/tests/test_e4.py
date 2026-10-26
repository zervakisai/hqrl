"""Unit tests of the E4 patch layer (e4lib.py). The pilot selection rule and the fixes after review are tested in
tests/test_fixes.py.

    nice -n 10 /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/

Episodes run on dq_development seeds only (realisations 620-629 of the core cells: outside the pilot's development
evaluation suite 600-609 and the pilot's training seeds); nothing reads dq_e2_evaluation.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import e4lib as e4                                                                                # noqa: E402
import dq_b2                                                                                      # noqa: E402
import e1_baselines as e1b                                                                        # noqa: E402
import fast_mission                                                                               # noqa: E402
import gnss_policies                                                                              # noqa: E402
import p4_train as p4                                                                             # noqa: E402
import run_p3_cells as cells_runner                                                               # noqa: E402
import runner                                                                                     # noqa: E402
from aerial_qrl.mission.contracts import ActionType                                               # noqa: E402
from aerial_qrl.mission.leakage import FORBIDDEN_TOKENS, key_tokens                               # noqa: E402
from aerial_qrl.training import encoding as enc                                                   # noqa: E402
from aerial_qrl.training import smdp_ppo                                                          # noqa: E402

TEST_REALIZATIONS = range(620, 630)
torch.set_num_threads(1)


@pytest.fixture(scope='module')
def rt():
    return dq_b2.runtime(e4.STUDY_DEV)


@pytest.fixture(autouse=True)
def originals():
    e4.restore_originals()
    yield
    e4.restore_originals()


def cfg(arm, block=4):
    return dq_b2.config_for(arm, block, dq_b2.STUDY_B2, e4.BUDGET, e4.GRID, freeze_sha256=dq_b2.DEV_FREEZE_LABEL)


def positive_cell(rt, index=0):
    return [c for c in rt.cells if c['cell_id'].endswith('positive')][index]


# ------------------------------------------------------------------ layout, names, switches -----------------------
def test_wide_layout_matches_base_except_dim():
    b, w = e4.BASE, e4.WIDE
    for name in ('track_offset', 'view_offset', 'georeference_offset', 'anchor_offset', 'search_actions',
                 'inspect_actions', 'report_actions', 'relocalize_actions', 'return_action', 'n_actions',
                 'view_features', 'action_version', 'track_slots', 'view_slots', 'anchor_slots'):
        assert getattr(w, name) == getattr(b, name), name
    assert (b.dim, w.dim, w.evidence_offset, e4.K) == (254, 270, 254, 16)
    assert enc.get_layout(e4.OBS_VERSION) is w


def test_extra_feature_names_are_public():
    assert len(e4.EXTRA_FEATURE_NAMES) == e4.K
    assert not [n for n in e4.EXTRA_FEATURE_NAMES if key_tokens(n) & FORBIDDEN_TOKENS]


def test_configure_switches_and_restores():
    info = e4.configure(True, 3e-3, 300_000)
    assert info['obs_dim'] == 270 and p4.LAYOUT_V4_GNSS is e4.WIDE and p4.RLMissionPolicy is e4.InfoRLMissionPolicy
    assert p4.build_model is e4.build_model_e4 and p4.ppo_update is e4.ppo_update_e4
    e4.configure(False)
    assert p4.LAYOUT_V4_GNSS is enc.LAYOUT_V4_GNSS and p4.RLMissionPolicy is e4._ORIGINAL['policy']
    assert e4.SCHED['lr0'] is None
    e4.restore_originals()
    assert p4.build_model is dq_b2.dq_build_model and p4.ppo_update is smdp_ppo.ppo_update
    assert p4.BLOCK_STRIDE == 10_000


# ------------------------------------------------------------------ the model at initialisation -------------------
@pytest.mark.parametrize('arm', e4.ARMS)
def test_base_build_is_stored_long_initial_checkpoint_bitwise(arm):
    e4.configure(False)
    model = p4.build_model(arm, cfg(arm), 4)
    ck = next((e4.LONG_DIR / f'runs/{arm}__b4/run/{arm}/checkpoints').glob('update00000_*.pt'))
    stored = torch.load(ck, map_location='cpu', weights_only=False)['model_state_dict']
    built = model.state_dict()
    assert set(stored) == set(built) and all(torch.equal(stored[k], built[k]) for k in stored)


@pytest.mark.parametrize('arm', e4.ARMS)
def test_widened_model_equals_base_at_init_random_inputs(arm):
    e4.configure(False)
    base = p4.build_model(arm, cfg(arm), 4)
    rng_base = torch.get_rng_state()
    e4.configure(True, 3e-3, 300_000)
    wide = p4.build_model(arm, cfg(arm), 4)
    assert torch.equal(rng_base, torch.get_rng_state())
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 11, 254, generator=g, dtype=torch.float64)
    extra = torch.rand(3, 11, e4.K, generator=g, dtype=torch.float64)
    with torch.no_grad():
        l0, v0, s0 = base(x)
        l1, v1, s1 = wide(torch.cat([x, extra], -1))
    print(f'\n[{arm}] random inputs: max |dlogit| {float((l0 - l1).abs().max()):.3e}, '
          f'max |dvalue| {float((v0 - v1).abs().max()):.3e}')
    assert float((l0 - l1).abs().max()) < 1e-12 and float((v0 - v1).abs().max()) < 1e-12
    assert float((s0 - s1).abs().max()) < 1e-12


# ------------------------------------------------------------------ encode on real observations ------------------
class CheckedRule(gnss_policies.UncertaintyFix):
    """The tuned rule, encoding every observation with the E4 encoder bound to the rule's OWN evidence state."""
    rows = []

    def __init__(self, briefing, obstacle_map, config, reserve, anchor_ids=None, **params):
        super().__init__(briefing, obstacle_map, config, reserve, **params)
        self.ev = e4.EvidenceEncoder(briefing, config, e4.WIDE, anchor_ids, owner=self)
        self.plain = enc.ObservationEncoder(briefing, config, e4.BASE, anchor_ids=anchor_ids)
        self.cap = config.max_inspections_per_track

    def act(self, observation):
        self._update_evidence(observation)
        plan = enc.plan_state_from_coverage(self.plan, self.status, self.ev.sector_ids)
        x, info = self.ev.encode(observation, plan)
        xp, _ = self.plain.encode(observation, plan)
        action = super().act(observation)
        CheckedRule.rows.append({'x': x, 'xp': xp, 'tracks': info.slots.tracks, 'action': action, 'cap': self.cap,
                                 'credited': {k: len(v) for k, v in self._credited.items()},
                                 'inspections': dict(self._inspections),
                                 'detections': {t.track_id: t.detections for t in observation.tracks}})
        return action


def test_encode_on_rule_episode_reads_the_rules_own_evidence(rt):
    rule = e1b.p3_rule()
    assert rule['name'] == 'uncertainty_fix'
    CheckedRule.rows = []
    cell = positive_cell(rt)
    seed = p4.evaluation_seed(rt, 'dq_development', cell, 620)
    static = rt.static(cell)
    backend_config, group, service = cells_runner.cell_episode_parts(static, cell, 'dq_development', seed)

    def factory(briefing, obstacle_map, mission_config, reserve):
        return CheckedRule(briefing, obstacle_map, mission_config, reserve, anchor_ids=rt.anchor_ids(),
                           **rule['parameters'])
    fast_mission.run_fast_episode(static['context'], static['banks'], rt.time_model, 'e4-test-rule',
                                  backend_config=backend_config,
                                  executive_kwargs={'georeference': service, 'anchor_set': static['anchors']},
                                  policy_factory=factory, reports_for_matching=service.global_report_positions)
    rows = CheckedRule.rows
    assert len(rows) > 20
    off, n = e4.WIDE.evidence_offset, e4.WIDE.track_slots
    reports = inspects = 0
    for r in rows:
        assert np.array_equal(r['x'][:off], r['xp'])                         # 0..253 = frozen obs-encoding/4-gnss
        for i, t in enumerate(r['tracks']):
            if t is None:
                assert r['x'][off + i] == 0. and r['x'][off + n + i] == 0.
                continue
            assert r['x'][off + i] == min(r['credited'].get(t.track_id, 0), 4) / 4
            assert r['x'][off + n + i] == min(r['inspections'].get(t.track_id, 0), r['cap']) / r['cap']
        a = r['action']
        slot = {t.track_id: i for i, t in enumerate(r['tracks']) if t is not None}
        if a.action_type == ActionType.REPORT and a.track_id in slot:       # the rule's report gate is visible
            reports += 1
            assert r['x'][off + slot[a.track_id]] >= 0.5 and r['detections'][a.track_id] >= 4
        if a.action_type == ActionType.INSPECT_VIEW and a.track_id in slot:  # the inspection cap is visible
            inspects += 1
            assert r['x'][off + n + slot[a.track_id]] < 1.
    print(f'\n[rule episode] decisions {len(rows)}, reports checked {reports}, inspections checked {inspects}')
    assert reports > 0 and inspects > 0
    assert max(float(r['x'][off:off + n].max()) for r in rows) >= 0.5
    assert max(float(r['x'][off + n:off + 2 * n].max()) for r in rows) > 0.


class CheckedInfo(e4.InfoRLMissionPolicy):
    """InfoRLMissionPolicy recording every act, encode and feedback, for an independent replay of the bookkeeping."""
    log = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plain = enc.ObservationEncoder(self.briefing, self.config, e4.BASE, anchor_ids=self.anchor_ids)
        inner = self.encoder.encode

        def encode(observation, plan):
            x, info = inner(observation, plan)
            CheckedInfo.log.append(('encode', observation, x, info.slots.tracks, self.plain.encode(observation, plan)[0]))
            return x, info
        self.encoder.encode = encode
        CheckedInfo.log.append(('new', self.config.view_separation_m, self.config.max_inspections_per_track))

    def act(self, observation):
        CheckedInfo.log.append(('act', observation))
        return super().act(observation)

    def feedback(self, action, executed, reason=None, goal=None):
        CheckedInfo.log.append(('feedback', action, executed))
        return super().feedback(action, executed, reason, goal)


def replay(log):
    """Independent re-implementation of the rule's crediting (executive.EvidencePolicy docstring) and inspection
    count; returns (checked encodes, executed inspections, max viewpoints seen)."""
    sep, cap = log[0][1], log[0][2]
    detections, credited, inspections = {}, {}, {}
    checked = executed_inspections = 0
    max_vp = 0
    off, n = e4.WIDE.evidence_offset, e4.WIDE.track_slots
    for entry in log[1:]:
        if entry[0] == 'act':
            o = entry[1]
            xy = tuple(o.vehicle.position_enu[:2])
            for t in o.tracks:
                if t.detections > detections.get(t.track_id, 0):
                    pts = credited.setdefault(t.track_id, [])
                    if all(math.dist(xy, p) > sep for p in pts):
                        pts.append(xy)
                detections[t.track_id] = t.detections
        elif entry[0] == 'encode':
            _, o, x, tracks, plain = entry
            assert np.array_equal(x[:off], plain)
            for i, t in enumerate(tracks):
                want_v = 0. if t is None else min(len(credited.get(t.track_id, ())), 4) / 4
                want_i = 0. if t is None else min(inspections.get(t.track_id, 0), cap) / cap
                assert x[off + i] == want_v and x[off + n + i] == want_i
                if t is not None:
                    max_vp = max(max_vp, len(credited.get(t.track_id, ())))
            checked += 1
        elif entry[0] == 'feedback':
            _, action, executed = entry
            if executed and action.action_type == ActionType.INSPECT_VIEW:
                inspections[action.track_id] = inspections.get(action.track_id, 0) + 1
                executed_inspections += 1
    return checked, executed_inspections, max_vp


@pytest.mark.parametrize('arm', e4.ARMS)
def test_learned_policy_bookkeeping_and_logits_on_real_episodes(rt, arm):
    e4.configure(False)
    base = p4.build_model(arm, cfg(arm), 4)
    e4.configure(True, 3e-3, 300_000)
    wide = p4.build_model(arm, cfg(arm), 4)
    p4.RLMissionPolicy = CheckedInfo
    total_checked = total_inspections = 0
    worst = 0.
    for k, r in enumerate(TEST_REALIZATIONS):
        CheckedInfo.log = []
        cell = positive_cell(rt, k % 4)
        seed = p4.evaluation_seed(rt, 'dq_development', cell, r)
        run = p4.run_learned_episode(rt, cell, 'dq_development', seed, wide, torch.Generator().manual_seed(100 + k),
                                     False, f'e4-test-{arm}-{k}', 0, 1.0, 60.0)
        checked, inspections, _ = replay(CheckedInfo.log)
        assert checked == len(run['policy'].decisions) > 0
        assert run['policy'].stats['ev_feedback_errors'] == 0 and run['policy'].stats['ev_encodes'] == checked
        total_checked += checked
        total_inspections += inspections
        for d in run['policy'].decisions:                 # each decision from its own recorded GRU input state
            x = torch.as_tensor(d.obs, dtype=torch.float64).reshape(1, 1, -1)
            h = torch.as_tensor(d.state, dtype=torch.float64).reshape(1, 1, -1)
            assert x.shape[-1] == 270
            with torch.no_grad():
                l0, v0, _ = base(x[..., :254], h)
                l1, v1, _ = wide(x, h)
            worst = max(worst, float((l0 - l1).abs().max()), float((v0 - v1).abs().max()),
                        float((d.logits - l0.reshape(-1)).abs().max()))      # the logits the policy sampled from
        if total_inspections > 0 and k >= 1:
            break
    print(f'\n[{arm}] learned episodes {k + 1}, encodes checked {total_checked}, executed inspections '
          f'{total_inspections}, max |dlogit|,|dvalue| (wide vs base, real obs) {worst:.3e}')
    assert worst < 1e-12, worst
    assert total_inspections > 0, 'no executed inspection: the inspection bookkeeping path was not exercised'


# ------------------------------------------------------------------ the learning-rate schedule --------------------
def synthetic_batch(model, obs_dim, seed, segs=6, T=20, A=27):
    g = np.random.default_rng(seed)
    n = segs * T
    masks = g.random((n, A)) < 0.5
    masks[:, -1] = True
    acts = np.array([g.choice(np.flatnonzero(m)) for m in masks])
    batch = dict(obs=g.normal(size=(n, obs_dim)), masks=masks, actions=acts, old_log_probs=np.zeros(n),
                 values=np.zeros(n), advantages=g.normal(size=n), returns=g.normal(size=n) * 0.1,
                 segments=np.array([[i * T, (i + 1) * T] for i in range(segs)]), segment_states=np.zeros((segs, 64)))
    padded = smdp_ppo.pad_segments(batch, range(segs))
    with torch.no_grad():
        logits, _, _ = model(padded['obs'], padded['state'], padded['masks'])
        lp = torch.distributions.Categorical(logits=logits[padded['valid']]).log_prob(padded['actions'][padded['valid']])
    batch['old_log_probs'] = lp.numpy()
    return batch


ARGS = (0.2, 0.5, 0.01, 0.5, 4)
KW = dict(minibatch_transitions=40, advantage_std_floor=0.05)


def test_ppo_wrapper_pass_through_is_bitwise_the_frozen_update():
    e4.configure(False)
    a = p4.build_model('bottleneck', cfg('bottleneck'), 4)
    b = copy.deepcopy(a)
    oa, ob = torch.optim.Adam(a.parameters(), lr=3e-3), torch.optim.Adam(b.parameters(), lr=3e-3)
    batch = synthetic_batch(a, 254, 0)
    ma = e4.ppo_update_e4(a, oa, batch, *ARGS, generator=torch.Generator().manual_seed(7), **KW)
    mb = smdp_ppo.ppo_update(b, ob, batch, *ARGS, generator=torch.Generator().manual_seed(7), **KW)
    assert ma == mb
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_ppo_wrapper_linear_schedule_reaches_zero():
    e4.configure(True, 1e-3, 300)
    model = p4.build_model('mlp', cfg('mlp'), 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(3)
    seen = []
    for u in range(4):                                         # 120 transitions per update: D = 0, 120, 240, 360
        m = e4.ppo_update_e4(model, opt, synthetic_batch(model, 270, 10 + u), *ARGS, generator=gen, **KW)
        seen.append((m['decisions_before_update'], m['learning_rate'], opt.param_groups[0]['lr'], m['entropy_coef']))
    assert [s[0] for s in seen] == [0, 120, 240, 360]
    assert [s[1] for s in seen] == [1e-3 * (1 - d / 300) if d < 300 else 0. for d in (0, 120, 240, 360)]
    assert all(s[1] == s[2] for s in seen) and all(s[3] == 0.01 for s in seen)
    assert seen[0][1] == 1e-3 and seen[-1][1] == 0.


# ------------------------------------------------------------------ the pilot selection rule ----------------------
# The draft's rule (runner.select_entropy, entropy guard >= 0.30 nats, tie -> 0.003) was replaced on 2026-10-25 after
# review item B1, before any pilot result was read. The amended rule lives in pilot_select.py and is tested in
# tests/test_fixes.py.
