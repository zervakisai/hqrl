"""Unit tests of the E1 analysis rules, the plan binding and the record handling (review B1-B3 and minor 2, 3, 4, 6,
7, 8, 13, 14). Synthetic records are built from the STORED RULE rows only (baselines.json); no run is read.

    python test_analysis.py            # runs every test_* function, prints a JSON summary, exit code 1 on any failure
    python -m pytest test_analysis.py  # the same tests under pytest
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import os
from pathlib import Path
import tempfile
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                                                # noqa: E402
from scipy import stats                                                                           # noqa: E402

import config_check as cc                                                                         # noqa: E402
import gate_analysis as ga                                                                        # noqa: E402

P1_KEY = 'P1_gate_minus_rule_e1_evaluation'
_CACHE = {}


def _ctx():
    if not _CACHE:
        import runner
        import p4_train as p4
        rt, agg = ga.aggregation()
        _CACHE.update(runner=runner, p4=p4, stored=ga.rule_rows(), rt=rt, agg=agg, weights=ga.cell_weights(rt))
    return SimpleNamespace(**_CACHE)


def _refused(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except SystemExit:
        return
    raise AssertionError(f'{getattr(fn, "__name__", fn)} accepted what it must refuse')


class _Records:
    """Redirect gate_analysis.RECORDS to a temporary directory for one test."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ga.scratch_dir())
        d = Path(self.tmp.name)
        (d / 'records').mkdir()
        self.saved = dict(ga.RECORDS)
        ga.RECORDS.update(dir=d / 'records', out=d / 'gate_results.json')
        return d

    def __exit__(self, *exc):
        ga.RECORDS.clear()
        ga.RECORDS.update(self.saved)
        self.tmp.cleanup()


def _put(d, rec):
    (d / 'records' / f'gate__b{rec["block"]}.json').write_text(json.dumps(rec, default=float))


def _synthetic_blocks(rng, n_blocks=12, positive_sd=0.03, shift=-0.003):
    cells = [f'c{i}__positive' for i in range(4)] + [f'c{i}__no_human' for i in range(4)]
    weights = {c: (0.2 if c.endswith('__positive') else 0.05) for c in cells}
    base = {(c, 1000 * i + r): float(rng.normal(0.2, 0.05)) for i, c in enumerate(cells) for r in range(10)}
    blocks = []
    for _ in range(n_blocks):
        blocks.append([{'cell_id': c, 'seed': s, 'rule_J': J,
                        'J': J + (float(rng.normal(shift, positive_sd)) if c.endswith('__positive') else 0.0)}
                       for (c, s), J in base.items()])
    return cells, weights, blocks


# ------------------------------------------------------------------ B1: the episode-level interval -----------------

def test_episode_level_equal_weights_is_scaled_paired_t():
    rng = np.random.default_rng(3)
    cells, weights, blocks = _synthetic_blocks(rng)
    ep = ga.episode_level(blocks, weights)
    per = {}
    for rows in blocks:
        for r in rows:
            per.setdefault((r['cell_id'], r['seed']), []).append(r['J'] - r['rule_J'])
    de = np.array([np.mean(v) for k, v in sorted(per.items()) if k[0].endswith('__positive')])
    t = stats.ttest_1samp(de, 0.0)
    half = stats.t.ppf(0.975, len(de) - 1) * de.std(ddof=1) / np.sqrt(len(de))
    assert ep['episodes'] == 40 and ep['weights_equal'] and abs(ep['weight_total'] - 0.8) < 1e-12
    assert abs(ep['mean'] - 0.8 * de.mean()) < 1e-15
    assert abs(ep['lo95'] - 0.8 * (de.mean() - half)) < 1e-12 and abs(ep['hi95'] - 0.8 * (de.mean() + half)) < 1e-12
    assert abs(ep['p_two_sided'] - t.pvalue) < 1e-12
    lo = stats.ttest_1samp(0.8 * de, -ga.DELTA, alternative='greater').pvalue
    hi = stats.ttest_1samp(0.8 * de, ga.DELTA, alternative='less').pvalue
    assert abs(ep['tost_p'] - max(lo, hi)) < 1e-12
    D = [sum(weights[c] * np.mean([r['J'] - r['rule_J'] for r in rows if r['cell_id'] == c]) for c in cells)
         for rows in blocks]
    assert abs(ep['mean'] - np.mean(D)) < 1e-12                     # the estimate is the P1 mean


def test_episode_level_all_80_when_a_no_human_row_differs():
    rng = np.random.default_rng(4)
    cells, weights, blocks = _synthetic_blocks(rng)
    blocks[0][-1]['J'] += 0.01                                      # a no-human row that is not the rule's
    ep = ga.episode_level(blocks, weights, positive_only=False)
    assert ep['episodes'] == 80 and not ep['weights_equal'] and abs(ep['weight_total'] - 1.0) < 1e-12
    D = [sum(weights[c] * np.mean([r['J'] - r['rule_J'] for r in rows if r['cell_id'] == c]) for c in cells)
         for rows in blocks]
    assert abs(ep['mean'] - np.mean(D)) < 1e-12


def test_episode_level_refuses_blocks_on_different_episodes():
    rng = np.random.default_rng(5)
    _, weights, blocks = _synthetic_blocks(rng)
    blocks[3] = blocks[3][1:]
    _refused(ga.episode_level, blocks, weights)


def test_generalises_rule():
    ep = lambda lo, hi, tost: {'lo95': lo, 'hi95': hi, 'tost_p': tost}           # noqa: E731
    assert ga.generalises('better', ep(0.001, 0.01, 0.9)) and not ga.generalises('better', ep(-0.001, 0.01, 0.9))
    assert ga.generalises('worse', ep(-0.01, -0.001, 0.9)) and not ga.generalises('worse', ep(-0.01, 0.001, 0.9))
    assert ga.generalises('equivalent', ep(-0.004, 0.004, 0.01))
    assert not ga.generalises('equivalent', ep(-0.004, 0.004, 0.06))
    assert ga.generalises(ga.EQUIVALENT_GAIN, ep(0.0005, 0.003, 0.01))
    assert not ga.generalises(ga.EQUIVALENT_GAIN, ep(-0.0005, 0.003, 0.01))
    assert not ga.generalises(ga.EQUIVALENT_GAIN, ep(0.0005, 0.003, 0.2))
    assert ga.generalises(ga.EQUIVALENT_LOSS, ep(-0.003, -0.0005, 0.01))
    assert not ga.generalises(ga.EQUIVALENT_LOSS, ep(-0.003, 0.0005, 0.01))
    for c in ('not resolved', ga.INSUFFICIENT):
        assert not ga.generalises(c, ep(0.001, 0.01, 0.0)) and not ga.generalises(c, ep(-0.01, -0.001, 0.0))
    assert not ga.generalises('better', None)


def test_reading_texts_are_conditional_and_qualified():
    P1 = {'mean': -0.01, 'lo95': -0.015, 'hi95': -0.005, 'lo90': -0.014, 'hi90': -0.006, 'tost_p': 0.6}
    ep = dict(P1, episodes=40)
    counts = {'completed': 11, 'algorithmic_failure': 1, 'infrastructure_failure': 0, 'missing': 0}
    cats = ('equivalent', ga.EQUIVALENT_GAIN, ga.EQUIVALENT_LOSS, 'better', 'worse', 'not resolved', ga.INSUFFICIENT)
    for c in cats:
        for agrees in (True, False):
            t = ga.reading_text(c, P1, ep, agrees and c not in ('not resolved', ga.INSUFFICIENT), counts)
            assert t.startswith('On the 80 E1 evaluation episodes (40 target-present)'), t
            assert '11 of 12 completed; 1 ended in an algorithmic failure and 0 in an infrastructure failure' in t
            assert 'not optimal' not in t
            if not agrees and c not in ('not resolved', ga.INSUFFICIENT):
                assert 'conditional on these episodes only' in t
            if agrees and c not in ('not resolved', ga.INSUFFICIENT):
                assert 'new episodes of these 8 cells' in t
    better = ga.reading_text('better', P1, ep, False, counts)
    assert '24 features' in better and 'rule 1' in better and 'one-feature rule on these episodes' in better
    assert 'rather than from the extra information' in better
    worse = ga.reading_text('worse', P1, ep, False, counts)
    for phrase in ('fixed a priori and not tuned', 'learning rate 1e-3', 'entropy coefficient 0.01',
                   '12,800 gate decisions', 'trained from scratch', 'tuned on a grid'):
        assert phrase in worse, phrase
    assert 'conditional on these episodes only' in ga.reading_text('worse', P1, None, False, counts)
    # round 2 (verifier): TOST without rejection does not exclude a gain up to the margin, so the 'equivalent'
    # sentence bounds the gain by the margin instead of denying any gain
    equiv = ga.reading_text('equivalent', P1, ep, False, counts)
    assert 'it does not improve on it by more than the margin (±0.0056 J)' in equiv, equiv
    assert 'does not improve on it.' not in equiv and 'does not improve on it;' not in equiv, equiv
    missing = dict(counts, missing=2)
    assert '2 have no record (provisional read-out)' in ga.reading_text('worse', P1, ep, True, missing)


# ------------------------------------------------------------------ minor 7: pairing and keys ----------------------

def test_block_contrast_requires_exactly_the_80_stored_keys():
    x = _ctx()
    rng = np.random.default_rng(6)
    rec = ga.synthetic_record(0, x.stored, rng)
    ga.block_contrast(rec, x.stored, x.rt, x.agg)
    rows = rec['final_evaluation']['episodes']
    _refused(ga.block_contrast, dict(rec, final_evaluation={'episodes': rows[:79] + [rows[0]]}), x.stored, x.rt, x.agg)
    _refused(ga.block_contrast, dict(rec, final_evaluation={'episodes': rows[:79]}), x.stored, x.rt, x.agg)
    foreign = [dict(r) for r in rows]
    foreign[5]['seed'] = foreign[5]['seed'] + 10 ** 6
    _refused(ga.block_contrast, dict(rec, final_evaluation={'episodes': foreign}), x.stored, x.rt, x.agg)
    crn = [dict(r) for r in rows]
    crn[7]['crn_sha256'] = '0' * 64
    _refused(ga.block_contrast, dict(rec, final_evaluation={'episodes': crn}), x.stored, x.rt, x.agg)
    rule = [dict(r) for r in rows]
    rule[9]['rule_J'] = rule[9]['rule_J'] + 1e-12
    _refused(ga.block_contrast, dict(rec, final_evaluation={'episodes': rule}), x.stored, x.rt, x.agg)


# ------------------------------------------------------------------ minor 6 / B3: plan binding ---------------------

def test_records_bound_to_plan_and_constants():
    x = _ctx()
    rng = np.random.default_rng(7)
    plan = ga.synthetic_plan()
    rec = ga.synthetic_record(1, x.stored, rng)
    ga.check_bound(rec, 1, plan)
    _refused(ga.check_bound, dict(rec, plan_sha256='another-plan'), 1, plan)
    _refused(ga.check_bound, rec, 2, plan)
    cfg = rec['config']
    for key, value in (('ppo', dict(cfg['ppo'], entropy_coef=0.0)), ('ppo', dict(cfg['ppo'], learning_rate=3e-4)),
                       ('budget_gate_decisions', 10_240), ('grid', [0, 2_560, 5_120, 7_680, 10_240]),
                       ('readout', 'sampled'), ('plan_sha256', 'another-plan'), ('initialisation', 'warm start'),
                       ('final_suite', 'e1_development')):
        _refused(ga.check_bound, dict(rec, config=dict(cfg, **{key: value})), 1, plan)
    _refused(ga.check_bound, dict(rec, config=dict(cfg, init_checkpoint='x.pt')), 1, plan)
    _refused(ga.check_bound, rec, 1, dict(plan, scripts=dict(plan['scripts'], **{'runner.py': '0' * 64})))
    _refused(ga.check_bound, dict(rec, code_tree_end='0' * 64), 1, plan)
    failed = ga.synthetic_record(2, x.stored, rng, status='algorithmic_failure')
    ga.check_bound(failed, 2, plan)
    _refused(ga.check_bound, dict(failed, plan_sha256=None), 2, plan)


def test_config_check_plan_constants():
    x = _ctx()
    runner = x.runner
    for b in (0, 5, 11):
        assert cc.config_mismatches(runner.config(b), b) == [], cc.config_mismatches(runner.config(b), b)
    cfg = runner.config(0)
    assert cc.config_mismatches(json.loads(json.dumps(cfg, default=str)), 0) == []
    assert cfg['ppo']['entropy_coef'] == 0.01 and cfg['ppo']['learning_rate'] == 1e-3
    assert cfg['budget_gate_decisions'] == 12_800 and cfg['grid'] == [0, 3_200, 6_400, 9_600, 12_800]
    assert cfg['readout'] == 'argmax' and cfg['sampled_readout']['role'] == 'descriptive'
    for over in ({'ppo': dict(cfg['ppo'], entropy_coef=0.0)}, {'budget_gate_decisions': 10_240},
                 {'grid': [0, 2_560, 5_120, 7_680, 10_240]}, {'transitions_per_update': 128},
                 {'rule': dict(cfg['rule'], parameters={'sigma_threshold_m': 1.4})}, {'extra_knob': 1},
                 {'salt': [20261025, 2]}, {'ppo': dict(cfg['ppo'], entropy_coef=0)}):
        assert cc.config_mismatches(dict(cfg, **over), 0), over
    assert cc.config_mismatches(cfg, 1)
    dev = runner.config(19, runner.STUDY_DEV, final_suite='e1_development')
    assert cc.config_mismatches(dev, 19)                                     # a development config is not a main one
    assert cc.module_mismatches(runner, x.p4) == []
    assert set(runner.code_hashes()) == set(cc.SCRIPTS)


def test_bound_plan_and_listed_scripts():
    with tempfile.TemporaryDirectory(dir=ga.scratch_dir()) as d:
        d = Path(d)
        for name in cc.SCRIPTS:
            (d / name).write_text(f'# {name}\n')
        table = '\n'.join(f'| `{name}` | `{cc.sha256(d / name)}` |' for name in cc.SCRIPTS)
        (d / 'PLAN.md').write_text(f'# plan\n\n| file | sha256 |\n|---|---|\n{table}\n')
        assert cc.listed_scripts((d / 'PLAN.md').read_text()) == {n: cc.sha256(d / n) for n in cc.SCRIPTS}
        _refused(cc.bound_plan, d)                                          # not hashed yet
        (d / 'PLAN.sha256').write_text(f'{cc.sha256(d / "PLAN.md")}  PLAN.md\n')
        plan = cc.bound_plan(d)
        assert plan['sha256'] == cc.sha256(d / 'PLAN.md') and set(plan['scripts']) == set(cc.SCRIPTS)
        (d / 'gate_analysis.py').write_text('# edited after hashing\n')
        _refused(cc.bound_plan, d)
        (d / 'gate_analysis.py').write_text('# gate_analysis.py\n')
        cc.bound_plan(d)
        (d / 'PLAN.sha256').write_text('0' * 64 + '  PLAN.md\n')
        _refused(cc.bound_plan, d)
        short = '\n'.join(line for line in (d / 'PLAN.md').read_text().splitlines() if 'config_check.py' not in line)
        (d / 'PLAN.md').write_text(short)
        (d / 'PLAN.sha256').write_text(f'{cc.sha256(d / "PLAN.md")}  PLAN.md\n')
        _refused(cc.bound_plan, d)                                          # a script missing from the listing


# ------------------------------------------------------------------ minor 13 / 14: readings and counts ------------

def test_fewer_than_two_blocks_still_writes_the_insufficient_reading():
    x = _ctx()
    rng = np.random.default_rng(8)
    for n_completed in (0, 1):
        with _Records() as d:
            for b in x.runner.BLOCKS:
                _put(d, ga.synthetic_record(b, x.stored, rng, status='completed' if b < n_completed
                                            else 'algorithmic_failure'))
            out = ga.main(False, plan=ga.synthetic_plan(), quiet=True)
            P1 = out[P1_KEY]
            assert P1['reading'] == ga.INSUFFICIENT and P1['n'] == n_completed and not P1['generalises']
            assert f'{12 - n_completed} ended in an algorithmic failure' in P1['reading_text']
            assert json.loads(ga.RECORDS['out'].read_text())[P1_KEY]['reading'] == ga.INSUFFICIENT


def test_main_reports_failures_beside_the_reading_and_extracts_descriptives():
    x = _ctx()
    rng = np.random.default_rng(9)
    with _Records() as d:
        for b in range(10):
            _put(d, ga.synthetic_record(b, x.stored, rng, shift=-0.01, noise=0.005))
        _put(d, ga.synthetic_record(10, x.stored, rng, status='algorithmic_failure'))
        retried = ga.synthetic_record(11, x.stored, rng, status='infrastructure_failure')
        retried.update(attempt=2, previous_attempts=[{'attempt': 1, 'status': 'infrastructure_failure'}])
        _put(d, retried)
        out = ga.main(False, plan=ga.synthetic_plan(), quiet=True)
    P1 = out[P1_KEY]
    assert P1['n'] == 10 and P1['reading'] == 'worse' and P1['generalises']
    assert out['counts']['algorithmic_failure'] == 1 and out['counts']['infrastructure_failure'] == 1
    assert out['counts']['retried_after_infrastructure_failure'] == [11]
    assert ('10 of 12 completed; 1 ended in an algorithmic failure and 1 in an infrastructure failure after one '
            'retry.') in P1['reading_text']
    ep = P1['episode_level']
    assert ep['episodes'] == 40 and abs(ep['mean'] - P1['mean']) < 1e-12 and P1['no_human_identical_to_rule_all_blocks']
    tr = out['descriptive']['training'][0]
    assert len(tr['approx_kl_pre_step_minibatch_mean_per_update']) == 50
    assert len(tr['p_fix_after_update_rms_lt_1.5']) == 50 and len(tr['p_fix_after_update_rms_ge_1.5']) == 50
    assert 'before its optimiser step' in out['descriptive']['approx_kl_label']
    s = out['descriptive']['sampled_readout']
    assert all(s['per_block'][b]['available'] for b in range(10)) and s['across_blocks']['n'] == 10
    assert out['S1'].startswith('dropped')


def test_common_policy_across_blocks_stays_conditional_on_the_episodes():
    """B1's failure mode: blocks converge to one policy that is not the rule's, so D_b hardly varies and P1 is
    decisive, while across episodes the difference is not resolved: the reading must stay conditional."""
    x = _ctx()
    rng = np.random.default_rng(10)
    positive = sorted(k for k in x.stored if k[0].endswith('__positive'))
    z = rng.normal(size=len(positive))
    z = (z - z.mean()) / z.std(ddof=1)
    common = {k: -0.009 + 0.04 * float(v) for k, v in zip(positive, z)}       # mean -0.009, SD 0.04 over episodes
    with _Records() as d:
        for b in x.runner.BLOCKS:
            rec = ga.synthetic_record(b, x.stored, rng, noise=0.0)
            for r in rec['final_evaluation']['episodes']:
                if r['cell_id'].endswith('__positive'):
                    r['J'] = r['rule_J'] + common[(r['cell_id'], r['seed'])] + float(rng.normal(0, 0.0005))
                    r['d_J'] = r['J'] - r['rule_J']
            _put(d, rec)
        out = ga.main(False, plan=ga.synthetic_plan(), quiet=True)
    P1 = out[P1_KEY]
    assert P1['reading'] == 'worse' and P1['p_two_sided'] < 1e-6, P1['reading']
    assert P1['episode_level']['lo95'] < 0 < P1['episode_level']['hi95']
    assert not P1['generalises'] and 'conditional on these episodes only' in P1['reading_text']


def test_incomplete_records_refused_unless_provisional():
    x = _ctx()
    rng = np.random.default_rng(11)
    with _Records() as d:
        for b in range(11):
            _put(d, ga.synthetic_record(b, x.stored, rng))
        _refused(ga.main, False, plan=ga.synthetic_plan(), quiet=True)
        out = ga.main(True, plan=ga.synthetic_plan(), quiet=True)
        assert out['provisional'] and out['counts']['missing'] == 1
        assert '1 have no record (provisional read-out)' in out[P1_KEY]['reading_text']


def test_unparsable_or_foreign_record_is_no_record():
    x = _ctx()
    rng = np.random.default_rng(12)
    with _Records() as d:
        good = ga.synthetic_record(0, x.stored, rng)
        _put(d, good)
        (d / 'records' / 'gate__b3.json').write_text(json.dumps(good)[:5000])            # truncated
        (d / 'records' / 'gate__b4.json').write_text(json.dumps(dict(good, block=5)))    # foreign block
        (d / 'records' / 'gate__b6.json').write_text(json.dumps(dict(good, block=6, status='started')))
        recs, unparsable = ga.load_records(range(12))
        assert sorted(recs) == [0] and unparsable == [3, 4, 6]
        runner = x.runner
        assert runner.read_record(d / 'records' / 'gate__b0.json', 0) is not None
        for b in (3, 4, 6):
            assert runner.read_record(d / 'records' / f'gate__b{b}.json', b) is None
        assert runner.read_record(d / 'records' / 'gate__b9.json', 9) is None             # missing


def test_sampled_contrast_is_descriptive_and_checked():
    x = _ctx()
    rng = np.random.default_rng(13)
    rec = ga.synthetic_record(0, x.stored, rng)
    s = ga.sampled_contrast(rec, x.stored, x.weights)
    rows = rec['final_evaluation_sampled']['episodes']
    cells = sorted({r['cell_id'] for r in rows})
    expected = sum(x.weights[c] * np.mean([r['J'] - r['rule_J'] for r in rows if r['cell_id'] == c]) for c in cells)
    assert s['available'] and len(cells) == 4 and abs(s['weighted_d_positive_cells'] - expected) < 1e-15
    short = dict(rec, final_evaluation_sampled=dict(rec['final_evaluation_sampled'], episodes=rows[1:]))
    assert not ga.sampled_contrast(short, x.stored, x.weights)['available']
    err = dict(rec, final_evaluation_sampled={'readout': 'sampled', 'error': 'RuntimeError: x'})
    assert not ga.sampled_contrast(err, x.stored, x.weights)['available']


# ------------------------------------------------------------------ runner: records, lock, retry, refusals --------

def test_atomic_record_write_leaves_no_partial_file():
    runner = _ctx().runner
    with tempfile.TemporaryDirectory(dir=ga.scratch_dir()) as d:
        path = Path(d) / 'gate__b2.json'
        runner.write_json_atomic(path, {'block': 2, 'status': 'completed', 'x': np.float64(0.5)})
        runner.write_json_atomic(path, {'block': 2, 'status': 'algorithmic_failure'})
        assert sorted(p.name for p in Path(d).iterdir()) == ['gate__b2.json']
        assert runner.read_record(path, 2)['status'] == 'algorithmic_failure'
    assert ga.FINAL_STATUSES == runner.FINAL_STATUSES


def test_launch_lock_is_exclusive_and_inherited_by_forked_workers():
    runner = _ctx().runner
    with tempfile.TemporaryDirectory(dir=ga.scratch_dir()) as d:
        path = Path(d) / '.launch.lock'
        handle = runner.acquire_lock(path)
        _refused(runner.acquire_lock, path)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:                                    # a surviving worker: holds the inherited descriptor
            os.close(w)
            os.read(r, 1)
            os._exit(0)
        os.close(r)
        handle.close()                                  # the parent is gone; the worker still holds the lock
        _refused(runner.acquire_lock, path)
        os.write(w, b'x')
        os.close(w)
        os.waitpid(pid, 0)
        runner.acquire_lock(path).close()               # all holders gone: a new launch may start


def _fake_run(task):
    """Stands in for runner.run_one in the retry test (module level, so the fork pool can call it)."""
    block, out, attempt = task
    if block == 4:
        return {'block': 4, 'attempt': attempt, 'status': 'refused', 'error': 'tree differs', 'wall_s': 0.0}
    (Path(out) / f'gate__b{block}' / 'run').mkdir(parents=True)
    if (block == 1 and attempt == 1) or block == 2:
        return {'block': block, 'attempt': attempt, 'status': 'infrastructure_failure', 'error': 'OSError: x',
                'wall_s': 0.0}
    if block == 3:
        return {'block': 3, 'attempt': attempt, 'status': 'algorithmic_failure', 'wall_s': 0.0}
    return {'block': block, 'attempt': attempt, 'status': 'completed', 'wall_s': 0.0}


def test_infrastructure_failure_retried_once_algorithmic_never():
    runner = _ctx().runner
    with tempfile.TemporaryDirectory(dir=ga.scratch_dir()) as d:
        d = Path(d)
        runs, records, aside = d / 'runs', d / 'records', d / 'interrupted' / 'utc'
        runs.mkdir()
        records.mkdir()
        (runs / 'gate__b5').mkdir()                     # a crashed run without a record
        (records / 'gate__b6.json').write_text('{"block": 6, "stat')                       # a truncated record
        runner.write_json_atomic(records / 'gate__b7.json', {'block': 7, 'status': 'completed'})
        (records / 'attempts').mkdir()                  # block 5: attempt 1 failed in an earlier launch, then a crash
        runner.write_json_atomic(records / 'attempts' / 'gate__b5__attempt1.json',
                                 {'block': 5, 'attempt': 1, 'status': 'infrastructure_failure', 'error': 'OSError: y'})
        todo = runner.pending_blocks(runs, records, aside, blocks=range(8))
        assert todo == [0, 1, 2, 3, 4, 5, 6]
        assert (aside / 'gate__b5').is_dir() and (aside / 'gate__b6.json.unparsable').exists()
        result = runner.drive(todo, 1, _fake_run, runs, records, aside, poll_s=0.02)
        assert result == {0: 'completed', 1: 'completed', 2: 'infrastructure_failure', 3: 'algorithmic_failure',
                          4: 'refused', 5: 'completed', 6: 'completed'}, result
        b5 = json.loads((records / 'gate__b5.json').read_text())
        assert b5['attempt'] == 2 and [a['error'] for a in b5['previous_attempts']] == ['OSError: y']
        rec = {b: json.loads((records / f'gate__b{b}.json').read_text()) for b in (0, 1, 2, 3)}
        assert rec[0]['attempt'] == 1 and rec[0]['previous_attempts'] == []
        assert rec[1]['attempt'] == 2 and [a['attempt'] for a in rec[1]['previous_attempts']] == [1]
        assert rec[2]['attempt'] == 2 and rec[2]['status'] == 'infrastructure_failure'
        assert rec[3]['attempt'] == 1 and rec[3]['previous_attempts'] == []              # never retried
        assert not (records / 'gate__b4.json').exists() and (aside / 'gate__b4__attempt1.refused.json').exists()
        assert (records / 'attempts' / 'gate__b1__attempt1.json').exists() and (aside / 'gate__b1__attempt1').is_dir()
        assert json.loads((records / 'gate__b7.json').read_text()) == {'block': 7, 'status': 'completed'}


def test_dropped_commands_refuse():
    runner = _ctx().runner
    _refused(runner.evaluate_stress, 1)
    _refused(runner.precheck_stress, 1)
    _refused(runner.pilot, [1e-3], [0], 6400, 1)


def run_all():
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                results[name] = 'passed'
            except (Exception, SystemExit):                     # noqa: BLE001  (a refusal is a failure here)
                results[name] = 'FAILED: ' + traceback.format_exc()[-1500:]
    return results


if __name__ == '__main__':
    res = run_all()
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if all(v == 'passed' for v in res.values()) else 1)
