"""Unit tests of the gate segment builder and the gate bookkeeping (synthetic records; no simulator episode).

    python test_gate.py            # runs every test_* function, prints a JSON summary, exit code 1 on any failure
    python -m pytest test_gate.py  # the same tests under pytest
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
from pathlib import Path
from types import SimpleNamespace
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                                                # noqa: E402
import torch                                                                                      # noqa: E402

import gate_core as gc                                                                            # noqa: E402
from aerial_qrl.mission.contracts import INFRASTRUCTURE, Action, ActionType                       # noqa: E402
from aerial_qrl.training import smdp                                                              # noqa: E402
from aerial_qrl.training.smdp_ppo import RolloutBuffer, compute_advantages, ppo_update            # noqa: E402

GAMMA, TAU = 1.0, 60.0
REPORT = lambda tid='T1': Action(ActionType.REPORT, track_id=tid, report_position_enu=(1.0, 2.0, 0.0))  # noqa: E731
FIX = lambda: Action(ActionType.RELOCALIZE, anchor_id='A_home')                                  # noqa: E731
SEARCH = lambda s='S1': Action(ActionType.SEARCH_SECTOR, sector_id=s)                            # noqa: E731
RETURN = lambda: Action(ActionType.RETURN)                                                       # noqa: E731


def rec(t, action, dur, valid=True, terminated=False, reason=None, forced=None):
    return SimpleNamespace(observation=SimpleNamespace(sim_time_s=float(t)), action=action,
                           duration_s=None if not valid else float(dur), valid=valid, terminated=terminated,
                           reason=reason if valid else INFRASTRUCTURE, forced=forced)


def dec(index, t, action, a_index, executed=True, mask=(True, True), log_prob=-0.69, value=0.1):
    d = gc.GateDecision(index, float(t), object(), np.full(gc.N_FEATURES, 0.1 * (index + 1)), np.array(mask, bool),
                        a_index, log_prob, value)
    d.action, d.executed = action, executed
    return d


def episode(pre_event=0.0, last='terminal'):
    """8 records; gate 0 (FIX) at t=20, gate 1 (REPORT now) at t=71. Rewards: +0.05 (report 1), -0.02 (report 2),
    -0.01 terminal correction; optional pre-gate event at t=5."""
    records = [rec(0, SEARCH(), 10), rec(10, SEARCH('S2'), 10), rec(20, FIX(), 30), rec(50, REPORT('T1'), 1),
               rec(51, SEARCH('S3'), 20), rec(71, REPORT('T2'), 1), rec(72, SEARCH('S4'), 10)]
    events = [[(5.0, pre_event)] if pre_event else [], [], [], [(50.0, 0.05)], [], [(71.0, -0.02)], []]
    if last == 'terminal':
        records.append(rec(82, RETURN(), 18, terminated=True, reason='mission_complete'))
        events.append([(100.0, -0.01)])
        termination = 'mission_complete'
    elif last == 'invalid':
        records.append(rec(82, RETURN(), None, valid=False))
        events.append([])
        termination = INFRASTRUCTURE
    elif last == 'unterminated':
        termination = INFRASTRUCTURE                    # e.g. an exception in policy.act after record 6
    else:
        raise ValueError(last)
    decisions = [dec(0, 20, records[2].action, gc.FIX_FIRST), dec(1, 71, records[5].action, gc.REPORT_NOW)]
    J = pre_event + 0.05 - 0.02 + (-0.01 if last == 'terminal' else 0.0)
    return decisions, records, events, termination, J


def test_reward_sum_and_pre_gate_zero():
    decisions, records, events, term, J = episode()
    s = gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    assert s.info['pre_gate_reward'] == 0.0
    assert s.valid_rows == 2 and len(s.rows) == 2
    assert math.isclose(s.kept_reward, J, abs_tol=1e-12)
    assert gc.check_segment_identity(s, J, 0.0) == 0.0 or abs(gc.check_segment_identity(s, J, 0.0)) < 1e-12
    assert math.isclose(s.rows[0]['reward'], 0.05) and math.isclose(s.rows[1]['reward'], -0.03)


def test_pre_gate_reward_counted_and_excluded():
    decisions, records, events, term, J = episode(pre_event=0.011)
    s = gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    assert math.isclose(s.info['pre_gate_reward'], 0.011)
    assert math.isclose(s.kept_reward, J - 0.011, abs_tol=1e-12)
    gc.check_segment_identity(s, J, 0.0)
    # with a dropped mass the identity is kept + pre + invalid = J - dropped
    gc.check_segment_identity(s, J + 0.004, 0.004)


def test_durations_and_terminal_flag():
    decisions, records, events, term, J = episode()
    s = gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    assert [r['duration_s'] for r in s.rows] == [51.0, 29.0]       # 71-20; end of RETURN (82+18) - 71
    assert all(r['duration_s'] > smdp.MIN_DURATION_S for r in s.rows)
    assert [r['terminated'] for r in s.rows] == [False, True]
    assert s.info['terminal_applied'] == 'mission_complete'
    assert [r['action'] for r in s.rows] == [gc.FIX_FIRST, gc.REPORT_NOW]
    assert all(r['state'].shape == (1,) and not r['state'].any() for r in s.rows)


def test_invalid_record_makes_last_segment_invalid_and_cut():
    decisions, records, events, term, J = episode(last='invalid')
    s = gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    assert [r['valid'] for r in s.rows] == [True, False]
    assert s.rows[1]['reward'] is None and s.rows[1]['duration_s'] is None and not s.rows[1]['terminated']
    assert math.isclose(s.info['invalid_segment_reward'], -0.02)
    gc.check_segment_identity(s, J, 0.0)
    buf = RolloutBuffer(gc.N_FEATURES, 2, 1)
    s.add_to(buf)
    batch = compute_advantages(buf, GAMMA, TAU, 0.95, 'time', 1)
    assert list(batch['cut']) == [True] and batch['cut_reason'][0] == INFRASTRUCTURE
    assert math.isclose(batch['next_values'][0], s.rows[1]['value'])        # bootstrap from the invalid row's value


def test_unterminated_infrastructure_exit_is_invalid():
    decisions, records, events, term, J = episode(last='unterminated')
    s = gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    assert [r['valid'] for r in s.rows] == [True, False]
    assert s.info['invalid_last_segment'].startswith('unterminated')
    gc.check_segment_identity(s, J, 0.0)


def test_invalid_record_inside_episode_refused():
    decisions, records, events, term, J = episode()
    records[4] = rec(51, SEARCH('S3'), None, valid=False)
    try:
        gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    except gc.GateAlignmentError:
        return
    raise AssertionError('an invalid record in the middle of an episode was accepted')


def test_abandoned_and_unmatched_decisions_skipped():
    decisions, records, events, term, J = episode()
    ghost = dec(9, 30, REPORT('T9'), gc.REPORT_NOW, executed=None)
    ghost.abandoned = True
    unbound = dec(10, 40, None, gc.REPORT_NOW, executed=None)
    s = gc.build_gate_segments([decisions[0], ghost, unbound, decisions[1]], records, events, term, 7, GAMMA, TAU)
    assert s.valid_rows == 2 and s.info['abandoned'] == 2
    gc.check_segment_identity(s, J, 0.0)


def test_forced_return_merged_into_last_segment():
    decisions, records, events, term, J = episode()
    records[-1] = rec(82, RETURN(), 18, terminated=True, reason='mission_complete', forced='return_reserve')
    s = gc.build_gate_segments(decisions, records, events, term, 7, GAMMA, TAU)
    assert s.rows[-1]['terminated'] and s.rows[-1]['duration_s'] == 29.0
    gc.check_segment_identity(s, J, 0.0)


def test_no_gate_episode_has_no_rows():
    records = [rec(0, SEARCH(), 10), rec(10, RETURN(), 20, terminated=True, reason='mission_complete')]
    s = gc.build_gate_segments([], records, [[], []], 'mission_complete', 3, GAMMA, TAU)
    assert s.rows == [] and s.valid_rows == 0
    gc.check_segment_identity(s, 0.0, 0.0)


def _biased_model(fix_bias):
    torch.manual_seed(0)
    m = gc.GateActorCritic().double()
    with torch.no_grad():
        m.actor[-1].bias.copy_(torch.tensor([0.0, fix_bias], dtype=torch.float64))
    return m


def test_reasked_fix_is_resampled_with_fix_masked():
    """A rejected RELOCALIZE: the same observation is asked again; the decision is resampled with FIX masked."""
    gate = gc.NetworkGate(_biased_model(30.0), generator=torch.Generator().manual_seed(1))
    book = gc.GateBook(gate)
    obs = SimpleNamespace(sim_time_s=20.0)
    x = np.full(gc.N_FEATURES, 0.2)
    assert book.ask(obs, x, {}) is True                       # FIX chosen (bias 30)
    fix = FIX()
    book.bind(fix)
    book.on_feedback(fix, False, 'anchor_unreachable')
    assert book.decisions[0].awaiting_resample
    assert book.ask(obs, x, {}) is False                      # re-ask: FIX masked -> report now
    report = REPORT()
    book.bind(report)
    book.on_feedback(report, True)
    assert len(book.decisions) == 1 and book.stats['resamples'] == 1
    d = book.decisions[0]
    assert d.executed and list(d.mask) == [True, False] and d.action_index == gc.REPORT_NOW
    assert d.log_prob == 0.0 and d.rejections == ['anchor_unreachable'] and d.samples == 2
    records = [rec(20, report, 1), rec(21, RETURN(), 9, terminated=True, reason='mission_complete')]
    s = gc.build_gate_segments(book.decisions, records, [[(20.0, 0.05)], []], 'mission_complete', 1, GAMMA, TAU)
    assert len(s.rows) == 1 and list(s.rows[0]['mask']) == [True, False] and s.rows[0]['action'] == 0
    assert s.rows[0]['duration_s'] == 10.0 and s.rows[0]['terminated']
    buf = RolloutBuffer(gc.N_FEATURES, 2, 1)
    s.add_to(buf)                                             # accepted: action feasible under the final mask


def test_reask_on_a_new_observation_abandons():
    gate = gc.NetworkGate(_biased_model(30.0), generator=torch.Generator().manual_seed(1))
    book = gc.GateBook(gate)
    o1, o2 = SimpleNamespace(sim_time_s=20.0), SimpleNamespace(sim_time_s=25.0)
    book.ask(o1, np.zeros(gc.N_FEATURES), {})
    fix = FIX()
    book.bind(fix)
    book.on_feedback(fix, False, 'x')
    book.ask(o2, np.zeros(gc.N_FEATURES), {})
    assert book.decisions[0].abandoned and len(book.decisions) == 2


def test_rule_gate_threshold():
    g = gc.RuleGate(1.5)
    o = lambda rms: SimpleNamespace(georeference=SimpleNamespace(rms_horizontal_error_m=rms))   # noqa: E731
    assert g.decide(o(1.5), None, np.array([True, True]), {})[0] == gc.FIX_FIRST
    assert g.decide(o(1.4999), None, np.array([True, True]), {})[0] == gc.REPORT_NOW
    assert g.decide(o(3.0), None, np.array([True, False]), {})[0] == gc.REPORT_NOW


def test_model_interface_and_ppo_update_runs():
    torch.manual_seed(0)
    model = gc.GateActorCritic().double()
    counts = model.parameter_counts()
    assert counts['actor'] == (gc.N_FEATURES * 64 + 64) + (64 * 64 + 64) + (64 * 2 + 2)
    x = torch.zeros(1, 1, gc.N_FEATURES, dtype=torch.float64)
    p = torch.softmax(model(x)[0].reshape(-1), -1).detach()
    assert abs(float(p[1]) - 0.5) < 0.05                       # near a fair coin at init
    gate = gc.NetworkGate(model, generator=torch.Generator().manual_seed(3))
    buf = RolloutBuffer(gc.N_FEATURES, 2, 1)
    rng = np.random.default_rng(0)
    for ep in range(12):
        decisions, records, events, term, J = episode()
        for d in decisions:
            d.obs = rng.normal(size=gc.N_FEATURES)
            a, lp, v, _ = gate.decide(None, d.obs, np.array([True, True]), {})
            d.action_index, d.log_prob, d.value = a, lp, v
        s = gc.build_gate_segments(decisions, records, events, term, ep, GAMMA, TAU)
        s.add_to(buf)
    batch = compute_advantages(buf, GAMMA, TAU, 0.95, 'time', 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [q.detach().clone() for q in model.parameters()]
    m = ppo_update(model, opt, batch, 0.2, 0.5, 0.0, 0.5, 2, generator=torch.Generator().manual_seed(4),
                   minibatch_transitions=8, advantage_std_floor=0.0056)
    assert m['transitions'] == 24 and m['optimizer_steps'] == 6 and math.isfinite(m['approx_kl'])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_checkpoint_roundtrip(tmp=None):
    import tempfile
    torch.manual_seed(5)
    model = gc.GateActorCritic().double()
    with tempfile.TemporaryDirectory(dir='/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad') as d:
        path = Path(d) / 'c.pt'
        gc.save_gate_checkpoint(path, model, None, {'a': 1}, {'b': 2})
        loaded, payload = gc.load_gate_checkpoint(path)
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
        try:
            gc.save_gate_checkpoint(path, model, None, {}, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError('a checkpoint was overwritten')


def run_all():
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                results[name] = 'passed'
            except Exception:                                   # noqa: BLE001
                results[name] = 'FAILED: ' + traceback.format_exc()[-1500:]
    return results


if __name__ == '__main__':
    res = run_all()
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if all(v == 'passed' for v in res.values()) else 1)
