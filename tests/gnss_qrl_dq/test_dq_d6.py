"""D-6 design, sealed decision rules and an end-to-end analysis dry run on SYNTHETIC episodes (review 35 seal)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_d6 as d6                                                                                # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402


# -- schedule and nesting -------------------------------------------------------------------------------------------

def test_schedule_is_complete_nested_and_disjoint():
    s = d6.schedule()
    assert len(s) == len(d6.UNITS) * len(d6.SETS) * d6.OFFSETS * d6.CELLS == 2048
    assert len(set(s)) == len(s)
    ranges = {name: {base + o for o in range(d6.OFFSETS)} for name, base in d6.REALIZATION_BASE.items()}
    earlier = set(range(0, 22))                                  # D-1 .. D-5
    names = list(ranges)
    for i, a in enumerate(names):
        assert not ranges[a] & earlier and max(ranges[a]) < 1000
        for b in names[i + 1:]:
            assert not ranges[a] & ranges[b]
    for n in d6.BATCHES:
        assert sum(d6.in_prefix(o, n) for o in range(d6.OFFSETS)) * d6.CELLS == n
    for small, large in zip(d6.BATCHES, d6.BATCHES[1:]):
        assert all(d6.in_prefix(o, large) for o in range(d6.OFFSETS) if d6.in_prefix(o, small))
    assert {u['stream_unit'] for u in d6.UNITS.values()} == {60, 61}


def test_design_is_json_stable():
    assert json.loads(json.dumps(d6.design())) == d6.design()
    assert set(d6.DECISION_TABLE) == {'tau_rank', 'delta', 'batch', 'stress_5e-2', 'failures'}


# -- rule 1: tau_rank ---------------------------------------------------------------------------------------------------

def rel(*values):
    return torch.tensor(values, dtype=torch.float64)


def test_rank_rule_prefers_the_default_on_a_tie_and_refuses_ambiguity():
    spectra = {'a': rel(1., 0.05, 1e-17), 'b': rel(1., 0.03, 1e-18)}
    r = d6.rank_rule(spectra)
    assert r['decision'] is not None
    # a spectrum with a value near every candidate makes all ambiguous
    bad = {'x': rel(1., 3e-4, 3e-5, 3e-6, 3e-7, 3e-8, 3e-9, 0.)}
    assert d6.rank_rule(bad)['decision'] is None


def test_rank_rule_maximises_the_minimum_margin():
    # values at 1e-2 and 1e-12: the geometric middle (1e-7) has the largest margin among candidates
    r = d6.rank_rule({'a': rel(1., 1e-2, 1e-12)})
    assert r['decision'] == 1e-7
    # a symmetric tie including the default goes to the default
    k, m, amb = d6.spectrum_margin(rel(1., 1e-3, 1e-9), 1e-6)
    assert k == 2 and abs(m - 3.) < 1e-12 and not amb


# -- rule 2: delta ------------------------------------------------------------------------------------------------------

def test_delta_rule_takes_the_smallest_qualifying_and_errors_or_poor_transfer_disqualify():
    ok = {'cond': 1e5, 'residual': 1e-14, 'error': None}
    bad_cond = {'cond': 1e12, 'residual': 1e-14, 'error': None}
    good_t = {'arm': 'SPAN', 'status': 'ok', 'transfer': 1.1}
    bad_t = {'arm': 'SPAN', 'status': 'ok', 'transfer': 3.0}
    failed_t = {'arm': 'SPAN', 'status': 'unattained', 'transfer': None}
    checks = {1e-6: [bad_cond, good_t], 1e-5: [ok, {'error': 'LinAlgError'}, good_t], 1e-4: [ok, bad_t],
              1e-3: [ok, failed_t], 1e-2: [ok, good_t]}
    assert d6.delta_rule(checks)['decision'] == 1e-2
    checks[1e-3] = [ok, good_t]
    assert d6.delta_rule(checks)['decision'] == 1e-3
    assert d6.delta_rule({d: [ok] for d in d6.DELTA_CANDIDATES})['decision'] is None       # no transfer evidence
    assert d6.delta_rule({d: [bad_cond, good_t] for d in d6.DELTA_CANDIDATES})['decision'] is None
    assert d6.delta_rule({})['decision'] is None


# -- rule 3: batch ------------------------------------------------------------------------------------------------------

def test_batch_rule_takes_the_largest_feasible_and_flags_weak_signal():
    r = d6.batch_rule(3.0, 60.0, {u: {n: 0.5 for n in d6.BATCHES} for u in d6.UNITS})
    # 72 * (4 * n * 3 + 60) / 3600: 64 -> 16.56, 128 -> 31.92, 256 -> 62.64
    assert r['decision'] == 128 and r['signal_insufficient'] is False
    weak = d6.batch_rule(3.0, 60.0, {u: {n: 0.05 for n in d6.BATCHES} for u in d6.UNITS})
    assert weak['signal_insufficient'] is True
    assert d6.batch_rule(None, 60.0, {})['decision'] is None
    assert d6.batch_rule(100.0, 60.0, {})['decision'] is None


# -- rule 4: 5e-2 -------------------------------------------------------------------------------------------------------

def test_stress_rule_is_conservative():
    good = [{'unit': 'u', 'arm': 'SPAN', 'status': 'ok', 'rel_error': 0.1}]
    assert d6.stress_rule(good)['decision'] == 'local_eligible'
    assert d6.stress_rule(good + [{'unit': 'u', 'arm': 'TAN', 'status': 'ok', 'rel_error': 0.25}])['decision'] == 'stress'
    assert d6.stress_rule(good + [{'unit': 'u', 'arm': 'X', 'status': 'unattained', 'rel_error': None}])['decision'] == 'stress'
    assert d6.stress_rule([])['decision'] == 'stress'


# -- the seal -----------------------------------------------------------------------------------------------------------

def test_seal_verification_refuses_absence_self_hash_mismatch_and_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(d6, 'D6_ROOT', tmp_path / 'd6')
    with pytest.raises(d6.D6Error, match='no D-6 seal'):
        d6.verify_seal()
    payload = d6.seal_payload()
    payload['seal_sha256'] = dq.canonical_sha256(payload)
    (tmp_path / 'd6').mkdir()
    d6.seal_path().write_text(json.dumps(payload))
    assert d6.verify_seal() == payload['seal_sha256']
    tampered = copy.deepcopy(payload)
    tampered['design']['COST_CAP_CPU_H'] = 1e9
    d6.seal_path().unlink()
    d6.seal_path().write_text(json.dumps(tampered))
    with pytest.raises(d6.D6Error, match='its own hash'):
        d6.verify_seal()
    rehashed = {k: v for k, v in tampered.items() if k != 'seal_sha256'}
    rehashed['seal_sha256'] = dq.canonical_sha256(rehashed)
    d6.seal_path().unlink()
    d6.seal_path().write_text(json.dumps(rehashed))
    with pytest.raises(d6.D6Error, match='differs from the seal'):
        d6.verify_seal()


def test_run_and_analyze_refuse_without_a_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(d6, 'D6_ROOT', tmp_path / 'd6')
    with pytest.raises(d6.D6Error):
        d6.run(1)
    with pytest.raises(d6.D6Error):
        d6.analyze()


# -- end-to-end analysis dry run on synthetic episodes (no simulator episode, no real gain) -----------------------------

def synthetic_episodes(model, unit, set_name, seed):
    g = torch.Generator().manual_seed(seed)
    episodes = []
    for offset in range(d6.OFFSETS):
        for cell in range(d6.CELLS):
            n_rows = 4
            obs = torch.randn(n_rows, model.config['obs_dim'], generator=g, dtype=torch.float64)
            state = 0.3 * torch.randn(n_rows, model.hidden, generator=g, dtype=torch.float64)
            mask = torch.rand(n_rows, model.config['action_dim'], generator=g) > 0.3
            mask[:, 0] = True
            with torch.no_grad():
                logits = model.policy(obs.unsqueeze(1), state.unsqueeze(0), mask.unsqueeze(1))[0].squeeze(1)
            action = torch.stack([torch.multinomial(torch.softmax(l, -1), 1, generator=g)[0] for l in logits])
            lp = torch.log_softmax(logits, -1)[torch.arange(n_rows), action]
            reward = torch.randn(n_rows, generator=g, dtype=torch.float64) * 0.1
            rows = [{'obs': obs[i].numpy(), 'state': state[i].numpy(), 'mask': mask[i].numpy(), 'action': int(action[i]),
                     'log_prob': float(lp[i]), 'reward': float(reward[i])} for i in range(n_rows)]
            episodes.append({'cell_index': cell, 'offset': offset, 'realization': d6.REALIZATION_BASE[set_name] + offset,
                             'record': {}, 'rows': rows})
    return episodes


def test_analysis_dry_run_end_to_end_on_synthetic_episodes(tmp_path, monkeypatch):
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    models = {u: load_checkpoint(d6.E1 / s['checkpoint'])[0].eval() for u, s in d6.UNITS.items()}
    data = {(u, s): synthetic_episodes(models[u], u, s, 10 * i + j)
            for i, u in enumerate(d6.UNITS) for j, s in enumerate(d6.SETS)}
    root = tmp_path / 'd6'
    (root / 'collection').mkdir(parents=True)
    (root / 'collection' / 'collection.json').write_text(json.dumps(
        {'seal_sha256': 'TEST', 'scheduled': 2048, 'completed': 2048, 'complete': True, 'failures': [],
         's_per_episode': 3.0}))
    monkeypatch.setattr(d6, 'D6_ROOT', root)
    monkeypatch.setattr(d6, 'verify_seal', lambda: 'TEST')
    monkeypatch.setattr(d6, 'load_set', lambda u, s: data[(u, s)])
    monkeypatch.setattr(d6, 'RND_DRAWS', 2)
    monkeypatch.setattr(d6, 'TRANSFER_RANGE', (0., math.inf))      # synthetic rows: exercise the full path
    monkeypatch.setattr(dq, 'claim_output', lambda p, test_root=None, directory=False: _claim(p, tmp_path, directory))
    monkeypatch.setattr(dq, 'write_json', lambda p, payload, test_root=None: _write(p, payload, tmp_path))
    monkeypatch.setattr(dq, 'append_tree_watch', lambda step, test_root=None: None)
    monkeypatch.setattr(dq, 'provenance', lambda *a, **k: {'test': True})
    result = d6.analyze()
    decisions = result['decisions']
    assert decisions is not None and decisions['kappa_primary'] == 1e-2
    assert decisions['tau_rank'] in d6.TAU_CANDIDATES
    assert decisions['delta'] in d6.DELTA_CANDIDATES
    assert decisions['batch_per_set'] in d6.BATCHES
    assert decisions['kappa_5e-2'] in ('stress', 'local_eligible')
    stored = json.loads((root / 'analysis' / 'd6_results.json').read_text())
    assert stored['statement'] == dq.DEVELOPMENT_STATEMENT
    for u in d6.UNITS:
        for n in d6.BATCHES:
            arms = stored['displayed_diagnostics']['units'][u][str(n)]['arms']
            assert {'SPAN', 'TAN', 'ORTH', 'FIT-k', 'FIT-1', 'HQ-Fisher', 'HQ-plain', 'HQ-Adam-dir', 'RND-k#0'} <= set(arms)


def test_analysis_makes_no_decision_when_the_collection_is_incomplete(tmp_path, monkeypatch):
    root = tmp_path / 'd6'
    (root / 'collection').mkdir(parents=True)
    (root / 'collection' / 'collection.json').write_text(json.dumps(
        {'seal_sha256': 'TEST', 'scheduled': 2048, 'completed': 2047, 'complete': False,
         'failures': [{'task': ['b10_final', 'fit', 0, 0, 100], 'error': 'x'}], 's_per_episode': 3.0}))
    monkeypatch.setattr(d6, 'D6_ROOT', root)
    monkeypatch.setattr(d6, 'verify_seal', lambda: 'TEST')
    monkeypatch.setattr(dq, 'claim_output', lambda p, test_root=None, directory=False: _claim(p, tmp_path, directory))
    monkeypatch.setattr(dq, 'write_json', lambda p, payload, test_root=None: _write(p, payload, tmp_path))
    monkeypatch.setattr(dq, 'append_tree_watch', lambda step, test_root=None: None)
    result = d6.analyze()
    assert result['decisions'] is None and 'INCOMPLETE' in result['reason']


_REAL_CLAIM, _REAL_WRITE = dq.claim_output, dq.write_json


def _claim(path, root, directory):
    return _REAL_CLAIM(path, test_root=root, directory=directory)


def _write(path, payload, root):
    return _REAL_WRITE(path, payload, test_root=root)
