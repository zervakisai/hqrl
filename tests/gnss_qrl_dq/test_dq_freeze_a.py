"""Freeze A: cells, seeds, CRN, bound finals, reset targets, gates, the scorer reproduction for A0, the analysis."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_freeze_a as fa                                                                          # noqa: E402
import dq_freeze_a_analysis as an                                                                 # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import run_p3_cells as runner                                                                     # noqa: E402


@pytest.fixture(scope='module')
def runtime():
    return fa.base_runtime()


@pytest.fixture(scope='module')
def cells(runtime):
    return fa.stress_cells(runtime)


def test_sixteen_cells_and_in_range_cells_reproduce_the_frozen_p3_georeference(runtime, cells):
    assert len(cells) == 16 and [c['cell_index'] for c in cells] == list(range(16))
    frozen = {c['cell_id']: c for c in runtime.cells}
    for c in cells:
        assert c['out_of_training_range'] == (c['d600_rms_m'] in (0.0, 8.0))
        assert c['pair_id'] == f'dqA__{c["budget_name"]}'
        if c['cell_id'] in frozen:
            f = frozen[c['cell_id']]
            assert c['georeference'] == f['georeference'] and c['georeference_sha256'] == f['georeference_sha256']
            assert c['mission_budget_s'] == f['mission_budget_s'] and c['n_ref'] == f['n_ref']
    assert sum(c['cell_id'] in frozen for c in cells) == 8


def test_seeds_share_crn_across_drift_and_presence_and_suites_are_disjoint(cells):
    ns = dq.DQ_NAMESPACES[fa.NAMESPACE]
    by_budget = {}
    for c in cells:
        for r in fa.STRESS_REALIZATIONS:
            s = fa.stress_seed(c, r)
            assert ns['start'] <= s < ns['end']
            group = runner.crn_group(c['pair_id'], fa.NAMESPACE, s)
            by_budget.setdefault((c['budget_name'], r), set()).add(group['crn_sha256'])
    assert all(len(v) == 1 for v in by_budget.values())             # one Brownian path per (budget, r)
    stress = {fa.stress_seed(c, r) for c in cells for r in fa.STRESS_REALIZATIONS}
    reset = {fa.stress_seed(c, r) for c in cells for r in fa.RESET_REALIZATIONS}
    assert not stress & reset
    assert len(fa.reset_episodes(cells)) == 160 and len(fa.stress_episodes(cells)) == 320


def test_the_48_finals_are_bound_by_sha256():
    finals = fa.selected_finals()
    assert len(finals) == 48
    assert {(f['family'], f['block']) for f in finals} == {(fam, b) for fam in fa.FAMILIES for b in fa.BLOCKS}
    for f in finals[::7]:
        assert dq.sha256_file(f['final']) == f['final_sha256']
    candidates = {f['family']: f['candidate'] for f in finals}
    assert candidates == {'quantum': 'lr3e-3_ent0', 'bottleneck': 'lr3e-3_ent0', 'mlp': 'lr3e-3_ent0',
                          'surrogate625': 'lr1e-3_ent0'}


@pytest.mark.parametrize('family', list(fa.FAMILIES))
def test_reset_changes_exactly_the_declared_tensors_to_the_runs_own_update0_values(family):
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    entry = next(f for f in fa.selected_finals() if f['family'] == family and f['block'] == 10)
    final, reset = fa.load_final(entry), fa.reset_model(entry)
    zero = load_checkpoint(entry['update0'])[0]
    fs, rs, zs = final.state_dict(), reset.state_dict(), zero.state_dict()
    for key in fs:
        if key in fa.RESET_TARGETS[family]:
            assert torch.equal(rs[key], zs[key])
        else:
            assert torch.equal(rs[key], fs[key])
    assert any(not torch.equal(fs[k], zs[k]) for k in fa.RESET_TARGETS[family])


def test_every_freeze_a_output_refuses_without_approval(tmp_path):
    with pytest.raises(dq.DQError, match='closed'):
        fa.a0()
    with pytest.raises(dq.DQError, match='closed'):
        fa.a1()
    with pytest.raises(dq.DQError, match='closed'):
        fa.launch(tmp_path / 'c', 1)
    assert fa.run_job({'job': 'x', 'kind': 'random_stress'})['status'] == 'refused'


def test_the_a0_scorer_reproduces_stored_never_fix_scores():
    never, rule = fa.a0_rows()
    assert len(never) == 160 and len(rule) == 160
    for key in sorted(never)[::16]:
        row = never[key]
        assert abs(fa.rescore(row, [s['global_position_enu'] for s in row['submissions']]) - row['score_J']) <= 1e-12


def test_fourier_power_shares_sum_to_one():
    C = torch.randn(625, 4, dtype=torch.float64)
    p = fa.fourier_power(C)
    assert abs(sum(p['harmonic_order'].values()) - 1) < 1e-12 and abs(sum(p['interaction_order'].values()) - 1) < 1e-12


def test_manifest_lists_every_job_and_seed(cells):
    m = fa.manifest()
    assert len(m['jobs']) == 48 * 3 + 2 and len(m['cells']) == 16
    assert m['episode_counts'] == {'stress_per_policy': 320, 'reset_suite_per_model': 160}


def test_stress_cells_run_including_0m_and_8m_on_the_development_namespace(monkeypatch, cells, runtime):
    """Development smoke: the new cells execute for a learned final, a reset model, the rule and random."""
    import e1_baselines
    monkeypatch.setattr(fa, 'NAMESPACE', 'dq_development')
    entry = next(f for f in fa.selected_finals() if f['family'] == 'quantum' and f['block'] == 10)
    model = fa.load_final(entry)
    for cell in (cells[0], cells[12]):                                    # d0 and d8, nominal, positive
        rec = fa.run_learned(runtime, cell, model, 900, f'smoke-{cell["cell_index"]}')
        assert rec['termination_reason'] and rec['cell_id'] == cell['cell_id']
    rec = fa.run_learned(runtime, cells[4], fa.reset_model(entry), 901, 'smoke-reset')
    assert rec['cell_id'] == 'd2__nominal_2700s__positive'
    rule = e1_baselines.run_policy_episode(runtime, cells[12], 'dq_development', fa.stress_seed(cells[12], 902),
                                           'p3_rule', 'smoke-rule', rule=e1_baselines.p3_rule())
    assert rule['fixes_attempted'] >= 1
    zero = e1_baselines.run_policy_episode(runtime, cells[0], 'dq_development', fa.stress_seed(cells[0], 903), 'p3_rule',
                                           'smoke-rule0', rule=e1_baselines.p3_rule())
    assert zero['fixes_attempted'] == 0                                  # no drift: the 1.5 m rule never fires


# -- analysis on synthetic job records ------------------------------------------------------------------------------

def synthetic_jobs(slope=0.0, beta=0.0, drop=()):
    rng = np.random.default_rng(0)
    jobs = {}
    cells = [(d, b, p) for d in an.D600 for b in an.BUDGETS for p in ('positive', 'no_human')]
    for fam in an.FAMILIES:
        for blk in an.BLOCKS:
            for kind in ('stress', 'reset_suite_final', 'reset_suite_reset'):
                name = f'{fam}__b{blk}__{kind}'
                if (fam, blk) in drop:
                    jobs[name] = {'job': name, 'status': 'failure'}
                    continue
                eps = []
                for d, b, p in cells:
                    if kind != 'stress' and d not in (2.0, 4.0):
                        continue
                    effect = (slope * d + (beta if b == 'tight_derived' else 0.)) if (fam == 'quantum' and p == 'positive') else 0.
                    for r in range(4):
                        eps.append({'cell_id': an.cell_id(d, b, p), 'J': 0.02 + effect + 0.001 * rng.standard_normal()
                                    + (0.01 if kind == 'reset_suite_reset' else 0.), 'fp_spatial': 0.3,
                                    'tp_timely': 0.5, 'return_ok': True, 'fixes_attempted': 10})
                jobs[name] = {'job': name, 'status': 'complete', 'episodes': eps}
    return jobs


def test_drift_interaction_is_detected_and_budget_is_not():
    report = an.analyze(synthetic_jobs(slope=0.002))
    inter = report['stress_map']['interactions']
    assert inter['I-drift']['rejected'] and inter['I-drift']['mean'] > 0
    assert not inter['I-budget']['rejected']
    assert report['stress_map']['hq_minus_cmatched_per_cell']['d8__nominal_2700s__positive']['out_of_training_range']
    assert report['reset_swaps']['quantum']['weighted_J']['mean'] == pytest.approx(0.01, abs=1e-3)


def test_dropped_blocks_beyond_one_make_the_stress_contrast_not_estimable():
    report = an.analyze(synthetic_jobs(drop={('quantum', 10), ('bottleneck', 11)}))
    assert report['stress_map']['blocks_dropped'] == [10, 11]
    assert report['stress_map']['estimable'] is False
    assert not report['stress_map']['interactions']['I-drift']['rejected']
    one = an.analyze(synthetic_jobs(drop={('quantum', 10)}))
    assert one['stress_map']['estimable'] and one['stress_map']['blocks_kept'][0] == 11
