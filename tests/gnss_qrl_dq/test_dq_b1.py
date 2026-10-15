"""Freeze B1: units, seeds, D-6 binding, held-out twin parity, the primary family, the gate, stratified reporting,
reliability outputs, the E choice, failure handling (review 36 list)."""
from __future__ import annotations

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

import dq_b1 as b1                                                                                # noqa: E402
import dq_b1_analysis as an                                                                       # noqa: E402
import dq_circuits as qc                                                                          # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402

F64 = torch.float64


# -- D-6 binding ---------------------------------------------------------------------------------------------------

def test_the_four_d6_decisions_are_bound_and_any_change_is_refused(monkeypatch):
    d = b1.require_d6_decisions()
    assert d == {'tau_rank': 1e-8, 'delta': 1e-6, 'batch_per_set': 256, 'kappa_5e-2': 'stress', 'kappa_primary': 1e-2}
    assert b1.PER_CELL['fit'] * b1.CELLS == b1.BATCH == 256
    for name, value in (('DELTA', 1e-4), ('TAU_RANK', 1e-6), ('BATCH', 128), ('PRIMARY_KAPPA', 5e-2)):
        monkeypatch.setattr(b1, name, value)
        with pytest.raises(b1.B1Error, match='differ'):
            b1.require_d6_decisions()
        monkeypatch.undo()
    monkeypatch.setattr(b1, 'D6_RESULT_SHA256', '0' * 64)
    with pytest.raises(b1.B1Error, match='changed'):
        b1.require_d6_decisions()


def test_every_per_set_size_except_e_is_the_d6_batch_and_e_is_the_independent_80():
    for s in ('fit', 'oracle', 'calibration', 'heldout'):
        assert b1.PER_CELL[s] * b1.CELLS == 256
    assert b1.PER_CELL['fresh_eval'] * b1.CELLS == 80
    assert len(b1.E_SUBSET['shipped']) + len(b1.E_SUBSET['heldout']) == 24


# -- units ----------------------------------------------------------------------------------------------------------

def test_shipped_unit_is_the_e1_hq_construction_under_the_dq_seed():
    model = b1.build_unit('shipped', 3)
    torch.manual_seed(dq.DQ_STUDY.stream(b1.INIT_TAG, 3))
    config = {'critic': 'mlp', 'hidden': 64, 'critic_gradient': 'stop', 'memory': 'gru', 'dtype': 'float64',
              'study': 'dq'}
    from aerial_qrl.training.encoding import LAYOUT_V4_GNSS
    from aerial_qrl.training.smdp_ppo import RecurrentActorCritic
    torch.manual_seed(dq.DQ_STUDY.stream(b1.INIT_TAG, 3))
    ref = RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, 'mlp', hidden=64, actor_head='quantum').double()
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), ref.state_dict().values()))


def test_heldout_unit_shares_trunk_critic_and_projection_with_its_shipped_pair():
    s, h = b1.build_unit('shipped', 5), b1.build_unit('heldout', 5)
    for part in ('encoder', 'gru', 'critic'):
        for a, b in zip(getattr(s, part).state_dict().values(), getattr(h, part).state_dict().values()):
            assert torch.equal(a, b)
    assert torch.equal(s.actor.projection.weight, h.actor.projection.weight)
    assert tuple(h.actor.weights.shape) == (3, 4, 2)
    assert h.config['actor_head'] == 'dq_heldout_circuit'
    other = b1.build_unit('shipped', 6)
    assert not torch.equal(other.encoder[0].weight, s.encoder[0].weight)


def test_heldout_twin_is_certified_and_tampering_is_refused():
    support, _ = tp.support(qc.HELDOUT)
    model = b1.build_unit('heldout', 1)
    twin = b1.twin_of('heldout', model)
    record = b1.certify_unit('heldout', model, twin, support)
    assert record['static_failures'] == [] and record['P2_offgrid_max_abs'] <= 1e-12
    g = torch.Generator().manual_seed(0)
    n = 30
    raw = {'obs': torch.randn(n, model.config['obs_dim'], generator=g, dtype=F64),
           'state': 0.3 * torch.randn(n, 64, generator=g, dtype=F64),
           'mask': torch.ones(n, model.config['action_dim'], dtype=torch.bool)}
    assert tp.certify_rows(model, twin, raw)['P3_max_abs_logprob'] <= 1e-9
    with torch.no_grad():
        twin.actor.surrogate.weight[0, 0] += 1e-6
    with pytest.raises(tp.TransplantParityError):
        b1.certify_unit('heldout', model, twin, support)


# -- seeds ----------------------------------------------------------------------------------------------------------

def test_seeds_are_inside_their_namespaces_and_unique_per_set():
    m = b1.seed_manifest()
    for s in b1.PER_CELL:
        assert m['counts'][s] == 72 * 8 * b1.PER_CELL[s]
    for s, per in b1.PER_CELL.items():
        ns = dq.DQ_NAMESPACES[b1.SET_NAMESPACE[s]]
        seeds = {b1.episode_seed(b1.SET_NAMESPACE[s], c, i, cell, o, s) for c in b1.CIRCUITS for i in range(36)
                 for cell in range(8) for o in range(per)}
        assert len(seeds) == 72 * 8 * per and min(seeds) >= ns['start'] and max(seeds) < ns['end']
    with pytest.raises(b1.B1Error):
        b1.episode_seed('dq_fit', 'shipped', 0, 0, 32, 'fit')
    assert len({b1.stream_unit(c, i) for c in b1.CIRCUITS for i in range(36)}) == 72
    assert not {60, 61} & {b1.stream_unit(c, i) for c in b1.CIRCUITS for i in range(36)}


# -- gates and failure handling ---------------------------------------------------------------------------------------

def test_launch_and_units_refuse_while_b1_is_not_approved(tmp_path):
    with pytest.raises(dq.DQError, match='closed'):
        b1.launch(tmp_path / 'campaign', 1)
    record = b1.run_unit('shipped', 0)
    assert record['status'] == 'refused' and 'closed' in record['error']


# -- the primary family, the gate, strata, exclusions -----------------------------------------------------------------

def unit(index, circuit, span, hqf, rnd, status='complete', ambiguous=False, D=0.5, extra=None, bad=()):
    arms = {}
    values = {'SPAN': span, 'HQ-Fisher': hqf, **{f'RND-k#{r}': rnd[r] for r in range(20)}, 'TAN': hqf, 'FIT-k': span}
    for arm, v in values.items():
        arms[arm] = {k: ({'status': 'ok', 'G_H': v, 'G_H_jackknife_se': 1e-5} if arm not in bad else
                         {'status': 'unattained'}) for k in ('0.001', '0.01', '0.05')}
    return {'circuit': circuit, 'index': index, 'status': status,
            'instruments': {'rank_ambiguous': ambiguous, 'arms': arms,
                            'diagnostics': {'D': D, 'grad_norm': 1.0 + index, 'fisher_condition_tangent': 10. + index,
                                            'entropy': 2.0},
                            'reliability': {'split_half_cosine': 0.1, 'Y': span - hqf, 'Y_jackknife_se': 1e-5,
                                            'Y_snr': (span - hqf) / 1e-5}},
            **({'fresh_eval': {'delta_J': {'HQ-Fisher': 0.001, 'RND-k#0': 0.0, 'FIT-k': 0.002, 'SPAN': 0.003}}}
               if index < 12 else {}), **(extra or {})}


def stratum(circuit, span_mean, hqf_mean, rnd_mean, seed, noise=1e-5, **kw):
    rng = np.random.default_rng(seed)
    return [unit(i, circuit, span_mean + noise * rng.standard_normal(), hqf_mean + noise * rng.standard_normal(),
                 list(rnd_mean + noise * rng.standard_normal(20)), D=float(rng.random()), **kw) for i in range(36)]


def test_primary_family_detects_a_positive_mechanism_contrast_and_applies_holm():
    report = an.analyze({'shipped': stratum('shipped', 1e-4, 2e-5, 2e-5, 1),
                         'heldout': stratum('heldout', 1e-4, 2e-5, 2e-5, 2)})
    p = report['primary']
    assert p['M-H1']['rejected'] is True and p['M-H1']['mean'] > 0
    assert p['A-H1']['estimable'] and p['A-H1']['rejected'] is False           # HQ-Fisher = RND on average
    assert p['gate_SPAN']['passed'] and p['permitted_conclusion'] is None
    assert report['heldout_circuit']['cross_circuit_statement_permitted'] is True
    assert len(p['M-H1']['unit_values']) == 36 and 'standardised_effect' in p['M-H1']
    assert set(report['secondary']) >= {'shipped', 'heldout', 'P_H1_stratified'}
    assert report['secondary']['shipped']['reliability']['attenuation_correction'].startswith('none')


def test_holm_orders_and_stops():
    d = an.holm({'a': {'estimable': True, 'p': 0.03}, 'b': {'estimable': True, 'p': 0.02}})
    assert d['b']['rejected'] and d['a']['rejected']               # 0.02 <= 0.025, then 0.03 <= 0.05
    d = an.holm({'a': {'estimable': True, 'p': 0.04}, 'b': {'estimable': True, 'p': 0.03}})
    assert not d['b']['rejected'] and not d['a']['rejected']       # 0.03 > 0.025 stops
    d = an.holm({'a': {'estimable': False, 'p': None}, 'b': {'estimable': True, 'p': 0.03}})
    assert not d['a']['rejected'] and not d['b']['rejected']       # m = 2 even when one is not estimable


def test_failed_gate_restricts_the_conclusion_but_still_reports_the_tests():
    report = an.analyze({'shipped': stratum('shipped', 0.0, -1e-4, -1e-4, 3, noise=1e-4),
                         'heldout': stratum('heldout', 0.0, 0.0, 0.0, 4, noise=1e-4)})
    assert report['primary']['gate_SPAN']['passed'] is False
    assert report['primary']['permitted_conclusion'] == 'insensitive at the tested KL and batch scale'
    assert report['primary']['M-H1']['estimable'] and report['primary']['M-H1']['p'] is not None


def test_heldout_stratum_without_support_forbids_the_cross_circuit_statement():
    report = an.analyze({'shipped': stratum('shipped', 1e-4, 2e-5, 2e-5, 5),
                         'heldout': stratum('heldout', 2e-5, 2e-5, 2e-5, 6, noise=1e-4)})
    assert report['heldout_circuit']['cross_circuit_statement_permitted'] is False
    assert report['heldout_circuit']['note'].startswith('reported separately')


def test_exclusions_above_ten_percent_make_a_test_not_estimable_and_missing_units_count():
    units = stratum('shipped', 1e-4, 2e-5, 2e-5, 7)
    for u in units[:4]:
        u['instruments']['arms']['SPAN']['0.01'] = {'status': 'unattained'}
    report = an.analyze({'shipped': units, 'heldout': stratum('heldout', 1e-4, 2e-5, 2e-5, 8)})
    assert report['primary']['M-H1']['estimable'] is False and report['primary']['M-H1']['rejected'] is False
    ambiguous = stratum('shipped', 1e-4, 2e-5, 2e-5, 9)
    for u in ambiguous[:3]:
        u['instruments']['rank_ambiguous'] = True
    r = an.analyze({'shipped': ambiguous, 'heldout': stratum('heldout', 1e-4, 2e-5, 2e-5, 10)})
    assert r['primary']['A-H1']['n'] == 33 and r['primary']['A-H1']['estimable']
    assert r['primary']['M-H1']['n'] == 36


def test_load_units_marks_missing_units(tmp_path):
    (tmp_path / 'units').mkdir()
    (tmp_path / 'units' / 'shipped_00.json').write_text(json.dumps({'circuit': 'shipped', 'index': 0, 'status': 'episode_failure'}))
    units = an.load_units(tmp_path)
    assert len(units['shipped']) == 36 and len(units['heldout']) == 36
    assert an._counts(units['shipped']) == {'episode_failure': 1, 'missing': 35}


def test_fresh_eval_is_secondary_and_summarised_per_arm():
    report = an.analyze({'shipped': stratum('shipped', 1e-4, 2e-5, 2e-5, 11),
                         'heldout': stratum('heldout', 1e-4, 2e-5, 2e-5, 12)})
    e = report['secondary']['shipped']['fresh_eval_delta_J']
    assert set(e) == set(an.E_ARMS) and e['SPAN']['n'] == 12
    assert 'fresh' not in json.dumps(report['primary'])


# -- end-to-end pipeline smoke on dq_development with a shrunken design (development material; nothing written) -------

@pytest.mark.parametrize('circuit', ['shipped', 'heldout'])
def test_run_unit_end_to_end_on_the_development_namespace(monkeypatch, circuit):
    monkeypatch.setattr(b1, 'SET_NAMESPACE', {s: 'dq_development' for s in b1.SET_NAMESPACE})
    monkeypatch.setattr(b1, 'PER_CELL', {'fit': 3, 'oracle': 3, 'calibration': 3, 'heldout': 3, 'fresh_eval': 1})
    monkeypatch.setattr(b1, 'RND_DRAWS', 2)
    monkeypatch.setattr(b1, 'RND_EXTRA_K', (48,))
    monkeypatch.setattr(b1, 'UNITS_PER_CIRCUIT', 36)
    record = b1.run_unit(circuit, 0)
    assert record['status'] == 'complete', record.get('error', '') + record.get('traceback', '')
    cert = record['certificate']
    assert cert['P3_max_abs_logprob'] <= 1e-9 and cert['P4']['max_abs_dlogprob'] <= 1e-9
    inst = record['instruments']
    assert inst['k'] == (12 if circuit == 'shipped' else 20) and not inst['rank_ambiguous']
    assert {'SPAN', 'HQ-Fisher', 'TAN', 'ORTH', 'FIT-k', 'FIT-1', 'RND-k#0', 'RND-k#1', 'RND-48#0', 'HQ-plain',
            'HQ-Adam-dir', 'RND-spec-plain'} <= set(inst['arms'])
    assert inst['arms']['SPAN']['0.01']['status'] == 'ok'
    assert set(record['fresh_eval']['delta_J']) <= set(b1.E_ARMS[1:])
    assert 'Y' in inst['reliability'] and 'D' in inst['diagnostics']
    json.dumps(record, default=float)


def test_files_sealed_by_d6_are_unchanged():
    seal = json.loads((dq.DEVELOPMENT_ROOT / 'd6' / 'seal.json').read_text())
    for relative, digest in seal['files'].items():
        assert dq.sha256_file(ROOT / relative) == digest, relative
