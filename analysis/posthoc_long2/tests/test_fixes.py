"""Tests of the fixes after review (2026-10-25): runner.py (status classes, restart once, crash restart, single
instance), pilot_select.py (the amended rule B1) and e4_analysis.py (B2-B4 and the minor items).

    nice -n 10 /home/konstantinos/Documents/ChatGPT/idea/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/

Nothing here reads a pilot or main E4 result: the runner tests use fake runs, the selection tests synthetic summaries
and the analysis tests synthetic curves or stand-ins built from the stored LONG runs (already-analysed data).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import runner                                                                                     # noqa: E402


# ------------------------------------------------------------------ runner: classes -------------------------------
def rec(status, error=None, block=9, name='mlp__b9'):
    r = {'kind': 'main', 'name': name, 'arm': name.split('__')[0], 'block': block, 'status': status, 'wall_s': 1.0}
    if error is not None:
        r['error'] = error
    return r


def test_classify():
    assert runner.classify(rec('completed')) == 'completed'
    assert runner.classify(rec('algorithmic_failure')) == 'algorithmic_failure'
    # the namespace-end form of ScheduleExhausted (block 9 is the last block of dq_e2_training)
    assert runner.classify(rec('failure', 'P4Error: block 9 leaves the dq_e2_training namespace')) == 'namespace_end'
    # any other failure, or the message for another block, is non-algorithmic
    assert runner.classify(rec('failure', 'P4Error: block 8 leaves the dq_e2_training namespace')) == \
        'infrastructure_failure'
    assert runner.classify(rec('failure', 'P4Error: internal: 3 kept transitions collected, 2 used')) == \
        'infrastructure_failure'
    assert runner.classify(rec('failure', 'OSError: [Errno 28] No space left on device')) == 'infrastructure_failure'
    with pytest.raises(ValueError):
        runner.classify(rec('started'))


# ------------------------------------------------------------------ runner: single instance ----------------------
def test_single_instance_lock(tmp_path):
    lock = tmp_path / 'runner.lock'
    fd = runner.single_instance('test', lock_path=lock, here=tmp_path)
    try:
        with pytest.raises(SystemExit, match='held'):
            runner.single_instance('test2', lock_path=lock, here=tmp_path)
    finally:
        os.close(fd)
    fd = runner.single_instance('test3', lock_path=lock, here=tmp_path)       # released with its descriptor
    os.close(fd)


def test_other_launchers_finds_a_launcher_of_the_folder(tmp_path):
    fake = tmp_path / 'runner.py'
    fake.write_text('import time\ntime.sleep(60)\n')
    proc = subprocess.Popen([sys.executable, '-B', 'runner.py', 'launch', '--workers', '1'], cwd=tmp_path)
    try:
        deadline = time.time() + 10
        found = []
        while time.time() < deadline and not found:
            found = runner.other_launchers(tmp_path)
            time.sleep(0.1)
        assert [pid for pid, _ in found] == [proc.pid]
        with pytest.raises(SystemExit, match='another launcher'):
            runner.single_instance('x', lock_path=tmp_path / 'l', here=tmp_path)
        assert runner.other_launchers(tmp_path / 'elsewhere') == []
    finally:
        proc.kill()
        proc.wait()


# ------------------------------------------------------------------ runner: restart rules ---------------------------
def fake_run(t):
    """A fake run: makes its run directory; behaviour per name, counted by attempt files (the worker is a fork)."""
    out = Path(t['out'])
    attempts = out.parent / 'attempts'
    attempts.mkdir(exist_ok=True)
    n = len(list(attempts.glob(f'{t["name"]}.*')))
    (attempts / f'{t["name"]}.{n}').write_text('')
    run_dir = out / t['name']
    run_dir.mkdir(parents=True)
    (run_dir / 'metrics.jsonl').write_text(f'attempt {n}\n')
    r = {'kind': t['kind'], 'name': t['name'], 'arm': t['arm'], 'block': t['block'], 'task': t, 'wall_s': 0.0}
    mode = t['mode']
    if mode == 'ok' or (mode == 'fail_once' and n > 0):
        r['status'] = 'completed'
    elif mode == 'namespace':
        r.update(status='failure', error=f'P4Error: block {t["block"]} leaves the dq_e2_training namespace')
    else:                                                           # 'fail_once' first attempt, 'fail_always'
        r.update(status='failure', error='OSError: simulated')
    return r


def fake_tasks(root, modes):
    return [{'kind': 'main', 'name': f'mlp__b{b}', 'arm': 'mlp', 'block': b, 'out': str(root / 'runs'), 'mode': m}
            for b, m in modes.items()]


def statuses(root):
    return {p.stem: json.loads(p.read_text())['status'] for p in sorted((root / 'records').glob('*.json'))}


def test_restart_once_then_the_second_failure_stands(tmp_path):
    root = tmp_path
    tasks = fake_tasks(root, {4: 'ok', 5: 'fail_once', 6: 'fail_always', 9: 'namespace'})
    recs = runner.resumable(tasks, root, 1, 'test', fn=fake_run, poll_s=0.02)
    assert len(recs) == 6                                      # 4 first attempts + 2 restarts (b5, b6), b9 never
    assert statuses(root) == {'mlp__b4': 'completed', 'mlp__b5': 'completed', 'mlp__b6': 'failure',
                              'mlp__b9': 'failure'}
    ledger = runner.restarts(root)
    assert sorted((r['name'], r['kind']) for r in ledger) == [('mlp__b5', 'failure'), ('mlp__b6', 'failure')]
    for r in ledger:                                           # the failed copy and its record are kept
        assert (root / r['moved_to'] / 'metrics.jsonl').read_text() == 'attempt 0\n'
        assert json.loads((root / r['record_moved_to']).read_text())['status'] == 'failure'
    assert runner.classify(json.loads((root / 'records' / 'mlp__b9.json').read_text())) == 'namespace_end'
    # a relaunch changes nothing: every record stands (b6's second failure, b9's algorithmic end)
    assert runner.resumable(tasks, root, 1, 'test', fn=fake_run, poll_s=0.02) == []
    assert len(runner.restarts(root)) == 2


def test_relaunch_restarts_a_first_failure_and_a_crashed_run(tmp_path):
    root = tmp_path
    (root / 'records').mkdir()
    (root / 'runs' / 'mlp__b4').mkdir(parents=True)                       # crashed: a directory, no record
    (root / 'runs' / 'mlp__b4' / 'metrics.jsonl').write_text('crashed\n')
    (root / 'runs' / 'mlp__b5').mkdir(parents=True)                       # a first failure recorded by an old launcher
    (root / 'records' / 'mlp__b5.json').write_text(json.dumps(rec('failure', 'MemoryError: ', 5, 'mlp__b5')))
    tasks = fake_tasks(root, {4: 'ok', 5: 'ok'})
    recs = runner.resumable(tasks, root, 1, 'test', fn=fake_run, poll_s=0.02)
    assert len(recs) == 2 and statuses(root) == {'mlp__b4': 'completed', 'mlp__b5': 'completed'}
    kinds = {r['name']: r['kind'] for r in runner.restarts(root)}
    assert kinds == {'mlp__b4': 'interrupted', 'mlp__b5': 'failure'}
    crashed = [r for r in runner.restarts(root) if r['name'] == 'mlp__b4'][0]
    assert (root / crashed['moved_to'] / 'metrics.jsonl').read_text() == 'crashed\n'


def test_plan_names_frozen_hashes(tmp_path):
    plan = tmp_path / 'PLAN.md'
    plan.write_text('nothing')
    assert not any(runner.plan_names_frozen_hashes(plan).values())
    plan.write_text('\n'.join(f'{f} {runner.sha(HERE / f)}' for f in runner.FROZEN_FILES if (HERE / f).exists()))
    assert all(runner.plan_names_frozen_hashes(plan)[f] for f in runner.FROZEN_FILES if (HERE / f).exists())


def test_status_progress_reads_counts_only(tmp_path):
    m = tmp_path / 'metrics.jsonl'
    m.write_text('\n'.join(json.dumps(x, sort_keys=True) for x in (
        {'kind': 'arm_start'}, {'kind': 'checkpoint', 'points': [0]},
        {'kind': 'evaluation', 'points': [0], 'results': {'sample': {'weighted_J': 0.5}}},
        {'kind': 'episode', 'J': 0.4}, {'kind': 'update', 'update_index': 1, 'metrics': {'entropy': 1.5},
                                         'cumulative': {'policy_decisions': 512, 'episodes': 1}})) + '\n{"kind": "ep')
    p = runner.run_progress(m)
    assert p['updates'] == 1 and p['policy_decisions'] == 512 and p['last_evaluated_points'] == [0]
    assert p['lines'] == {'arm_start': 1, 'checkpoint': 1, 'evaluation': 1, 'episode': 1, 'update': 1, 'partial': 1}
    assert '0.5' not in json.dumps(p) and '1.5' not in json.dumps(p)


# ================================================================== pilot_select.py: the amended rule (B1) =========
import datetime                                                                                   # noqa: E402
import shutil                                                                                     # noqa: E402

import e4lib as e4                                                                                # noqa: E402
import e4_analysis as ea                                                                          # noqa: E402
import p4_train as p4                                                                             # noqa: E402
import pilot_select as ps                                                                         # noqa: E402


def psumm(status='completed', j=0.02, t80=0.5, t100=0.5, j80=None, j100=None):
    return {'status': status, 'J': {80_000: j if j80 is None else j80, 100_000: j if j100 is None else j100},
            'timely': {80_000: t80, 100_000: t100}}


def both(**kw):
    return {6: psumm(**kw), 7: psumm(**kw)}


def test_selection_rule_amended():
    sel = ps.select_entropy
    assert (ps.TIMELY_CUT, ps.TIE_BAND, ps.FALLBACK, ps.CANDIDATES) == (0.012, 0.0056, 0.01, (0.003, 0.01))
    # both eligible, clear difference: the larger S, in either direction
    assert sel({0.003: both(j=0.02), 0.01: both(j=0.03)})[0] == 0.01
    assert sel({0.003: both(j=0.04), 0.01: both(j=0.03)})[0] == 0.003
    # inside the tie band: the LARGER entropy, even when 0.003 scores higher
    c, d = sel({0.003: both(j=0.025), 0.01: both(j=0.02)})
    assert c == 0.01 and 'larger entropy' in d['rule_applied']
    # exactly at the band edge is not a tie
    assert sel({0.003: both(j=0.0056), 0.01: both(j=0.0)})[0] == 0.003
    # behavioural eligibility: <= 0.012 timely-credited reports per episode at either point in either block
    c, d = sel({0.003: {6: psumm(j=0.09), 7: psumm(j=0.09, t100=0.012)}, 0.01: both(j=0.01)})
    assert c == 0.01 and not d['values'][0.003]['eligible'] and d['values'][0.01]['eligible']
    assert sel({0.003: both(j=0.01), 0.01: {6: psumm(j=0.09, t80=0.0), 7: psumm(j=0.09)}})[0] == 0.003
    assert sel({0.003: both(j=0.05, t80=0.0121, t100=0.0121), 0.01: both(j=0.01, t80=0.0)})[0] == 0.003
    # none eligible: 0.01, whatever S
    c, d = sel({0.003: both(j=0.09, t80=0.0), 0.01: both(j=0.0, t100=0.0)})
    assert c == 0.01 and 'fallback' in d['rule_applied']
    # a failed or missing block is not 'completed in both blocks'
    assert sel({0.003: {6: psumm(status='algorithmic_failure'), 7: psumm(j=0.09)}, 0.01: both(j=0.0)})[0] == 0.01
    assert sel({0.003: both(j=0.0), 0.01: {6: psumm(status='code_failure'), 7: psumm(j=0.5)}})[0] == 0.003
    c, d = sel({0.003: {6: None, 7: psumm()}, 0.01: {6: psumm(status='namespace_end'), 7: psumm()}})
    assert c == 0.01 and 'fallback' in d['rule_applied']
    # S is the mean of the four values; entropy plays no role (the summaries carry none)
    _, d = sel({0.003: {6: psumm(j80=0.01, j100=0.03), 7: psumm(j80=0.02, j100=0.04)}, 0.01: both(j=0.0)})
    assert abs(d['values'][0.003]['S'] - 0.025) < 1e-15
    with pytest.raises(ValueError):
        sel({0.003: both(), 0.02: both()})


def dev_pairs():
    rt = ps.runtime()
    return [(c['cell_id'], p4.evaluation_seed(rt, 'dq_development', c, r)) for c in rt.cells for r in e4.DEV_REALIZATIONS]


def synth_run(arm, block, entropy, grid, values, pairs, rt, study='e4_dev', end='completed', step=20_000):
    """Synthetic events of one learned run on `grid`: values {point: (J, timely)}; updates of `step` transitions."""
    ev = [{'kind': 'arm_start', 'arm': arm, 'block': block, 'study': study,
           'identity': {'condition': {'obs_encoding_version': e4.OBS_VERSION}}, 'model': {'config': {'obs_dim': 270}}}]
    state = {'d': 0, 'u': 0, 'ep': 0}

    def evaluation(point):
        J, timely = values[point]
        eps = [{'cell_id': c, 'seed': s, 'J': J, 'tp_timely': timely, 'fp_spatial': 0, 'fixes_attempted': 0,
                'decisions': 50} for c, s in pairs]
        _, w = p4.aggregate_evaluation(rt, eps)
        ev.append({'kind': 'checkpoint', 'path': f'checkpoints/update{state["u"]:05d}.pt', 'points': [point],
                   'final': False, 'policy_decisions': state['d']})
        ev.append({'kind': 'evaluation', 'arm': arm, 'points': [point], 'policy_decisions': state['d'],
                   'results': {'sample': {'episodes': eps, 'weighted_J': w}}, 'wall_s': 1.0})
    evaluation(0)
    for point in grid[1:]:
        while state['d'] < point:
            ev.append({'kind': 'episode', 'episode': state['ep'], 'update_index': state['u'] + 1, 'cell_id': 'c',
                       'seed': state['ep'], 'policy_stats': {}})
            state['ep'] += 1
            n = min(step, point - state['d'])
            lr = e4.scheduled_lr(e4.LR0[arm], e4.LR_HORIZON, state['d'])
            state['u'] += 1
            ev.append({'kind': 'update', 'update_index': state['u'], 'transitions': n,
                       'metrics': {'decisions_before_update': state['d'], 'learning_rate': lr, 'entropy_coef': entropy,
                                   'entropy': 1.0},
                       'cumulative': {'policy_decisions': state['d'] + n, 'updates': state['u'], 'episodes': state['ep']}})
            state['d'] += n
        evaluation(point)
    ev.append({'kind': 'checkpoint', 'path': 'checkpoints/final.pt', 'points': [], 'final': True,
               'policy_decisions': state['d']})
    ev.append({'kind': 'arm_end', 'status': end, 'utc': '2026-10-25T12:00:00+00:00'})
    return ev


def write_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in events))


def write_plan(root):
    (root / 'PLAN.md').write_text('Stand-in plan (tests).\n' + ''.join(
        f'{f} {runner.sha(HERE / f)}\n' for f in runner.FROZEN_FILES))
    digest = runner.sha(root / 'PLAN.md')
    (root / 'PLAN.sha256').write_text(f'{digest}  PLAN.md\n')
    return digest


def synth_pilot(root, values, statuses=None):
    """A synthetic pilot under root/pilot: values {(arm, entropy, block): {point: (J, timely)}}."""
    rt, pairs = ps.runtime(), dev_pairs()
    proot = root / 'pilot'
    (proot / 'records').mkdir(parents=True)
    (proot / 'launches.jsonl').write_text(json.dumps({'runner_sha256': 'f' * 64}) + '\n')
    for t in runner.pilot_tasks():
        key = (t['arm'], t['entropy'], t['block'])
        status = (statuses or {}).get(key, 'completed')
        write_events(proot / 'runs' / t['name'] / 'run' / t['arm'] / 'metrics.jsonl',
                     synth_run(t['arm'], t['block'], t['entropy'], e4.PILOT_GRID, values[key], pairs, rt))
        rec_ = {'kind': 'pilot', 'name': t['name'], 'arm': t['arm'], 'block': t['block'], 'status': status,
                'task': t, 'config': {'ppo': {'entropy_coef': t['entropy'], 'learning_rate': t['lr0']}},
                'code': {'runner_sha256': 'f' * 64, 'e4lib_sha256': runner.E4LIB_SHA256,
                         'idea_dq_tree': e4.EXPECTED_TREE}, 'wall_s': 1.0}
        if status == 'failure':
            rec_['error'] = 'OSError: simulated'
        (proot / 'records' / f'{t["name"]}.json').write_text(json.dumps(rec_))


def pilot_values(j=None, timely=None):
    out = {}
    for t in runner.pilot_tasks():
        key = (t['arm'], t['entropy'], t['block'])
        jj, tt = (j or {}).get(key, 0.03), (timely or {}).get(key, 0.4)
        out[key] = {0: (0.0, 0.1), 50_000: (0.02, 0.3), 80_000: (jj, tt), 100_000: (jj, tt)}
    return out


def test_pilot_select_end_to_end(tmp_path):
    root = tmp_path
    # C-strong: 0.003 is higher by 0.004 (inside the band) -> 0.01; C-matched: 0.01 collapses in block 7 -> 0.003
    j = {('mlp', 0.003, 6): 0.034, ('mlp', 0.003, 7): 0.034}
    timely = {('bottleneck', 0.01, 7): 0.0}
    synth_pilot(root, pilot_values(j, timely))
    with pytest.raises(SystemExit, match='PLAN.md'):
        ps.main(root)
    write_plan(root)
    assert all(not v['problems'] for v in ps.check(root).values())
    sel = ps.main(root)
    assert sel['heads']['mlp']['entropy_coef'] == 0.01 and 'larger entropy' in sel['heads']['mlp']['rule_applied']
    assert sel['heads']['bottleneck']['entropy_coef'] == 0.003
    assert sel['pilot_select_sha256'] == runner.sha(HERE / 'pilot_select.py')
    assert sel['plan_sha256'] == runner.sha(root / 'PLAN.md') and 'original_rule' in sel['rule']
    assert all('entropy_last20_mean' in r for r in sel['heads']['mlp']['runs'].values())
    with pytest.raises(SystemExit, match='never overwritten'):
        ps.main(root)


def test_pilot_select_restart_rule_and_integrity(tmp_path):
    root = tmp_path / 'a'
    synth_pilot(root, pilot_values(), statuses={('mlp', 0.01, 7): 'failure'})
    write_plan(root)
    with pytest.raises(SystemExit, match='restarts it once'):
        ps.main(root)
    # after one restart the second failure stands: mlp 0.01 is not completed in both blocks -> 0.003 (eligible)
    (root / 'pilot' / 'restarts.jsonl').write_text(json.dumps({'name': 'mlp__ent0.01__b7', 'kind': 'failure'}) + '\n')
    sel = ps.main(root)
    assert sel['heads']['mlp']['entropy_coef'] == 0.003 and sel['heads']['mlp']['values'][0.01]['S'] is None
    # a broken learning-rate log or a duplicated line refuses
    root = tmp_path / 'b'
    synth_pilot(root, pilot_values())
    write_plan(root)
    m = root / 'pilot' / 'runs' / 'bottleneck__ent0.003__b6' / 'run' / 'bottleneck' / 'metrics.jsonl'
    m.write_text(m.read_text().replace('"entropy_coef": 0.003', '"entropy_coef": 0.01', 1))
    with pytest.raises(SystemExit, match='integrity'):
        ps.main(root)
    lines = m.read_text().splitlines(keepends=True)
    m.write_text(''.join(lines[:3] + lines[2:3] + lines[3:]).replace('"entropy_coef": 0.01', '"entropy_coef": 0.003', 1))
    assert ps.check(root)['bottleneck__ent0.003__b6']['problems']


# ================================================================== e4_analysis.py ==================================
def test_event_integrity_on_stored_long_and_mutations():
    lines, events = ea.read_lines(e4.LONG_DIR / 'runs' / 'mlp__b4' / 'run' / 'mlp' / 'metrics.jsonl')
    assert ea.event_problems(lines, events, ea.GRID, True) == []
    assert 'duplicated lines' in ea.event_problems(lines + lines[-2:-1], events, ea.GRID, True)
    assert ea.event_problems(lines[:-1], events[:-1], ea.GRID, True)            # arm_end missing
    assert ea.event_problems(lines, events, ea.GRID, False)                      # an unexpected arm_end
    ups = [i for i, e in enumerate(events) if e['kind'] == 'update']
    gap = events[:ups[3]] + events[ups[3] + 1:]
    assert any('update' in x or 'cumulative' in x or 'episode' in x
               for x in ea.event_problems(lines[:ups[3]] + lines[ups[3] + 1:], gap, ea.GRID, True))
    ev_i = [i for i, e in enumerate(events) if e['kind'] == 'evaluation']
    twice = events[:ev_i[2] + 1] + events[ev_i[1] - 1:ev_i[1] + 1] + events[ev_i[2] + 1:]
    assert ea.event_problems([json.dumps(e) + str(i) for i, e in enumerate(twice)], twice, ea.GRID, True)
    nock = events[:ev_i[1] - 1] + events[ev_i[1]:]
    assert any('checkpoint' in x for x in ea.event_problems([str(i) for i in range(len(nock))], nock, ea.GRID, True))


def test_schedule_check_on_the_precheck_e4_path():
    path = HERE / 'precheck' / 'runs' / 'c_mlp__b4' / 'run' / 'mlp' / 'metrics.jsonl'
    if not path.exists():
        pytest.skip('precheck (c) run not present')
    lines, events = ea.read_lines(path)
    assert ea.event_problems(lines, events, [0, 3000], True) == []
    assert ea.schedule_problems(events, 'mlp', 0.01) == []
    assert ea.schedule_problems(events, 'mlp', 0.003)                            # another entropy coefficient
    bad = [dict(e, metrics=dict(e['metrics'], learning_rate=0.003)) if e['kind'] == 'update' else e for e in events]
    assert ea.schedule_problems(bad, 'mlp', 0.01)                                # a constant learning rate


def test_partial_last_line_only_while_running(tmp_path):
    m = tmp_path / 'metrics.jsonl'
    m.write_text('{"kind": "arm_start"}\n{"kind": "epi')
    with pytest.raises(SystemExit):
        ea.read_lines(m)
    lines, events = ea.read_lines(m, partial_ok=True)
    assert events == [{'kind': 'arm_start'}]
    m.write_text('{"kind": "arm_start"}\nnot json\n{"kind": "x"}\n')
    with pytest.raises(SystemExit):
        ea.read_lines(m, partial_ok=True)


REF = {'weighted_J': 0.211329, 'episodes': [{'tp_timely': 1.0, 'fp_spatial': 0.0, 'fixes_attempted': 2.0,
                                              'decisions': 135}]}


def synth_curve(values):
    return {'curve': {p: {'J': j, 'timely': 0.5, 'false': 0.0, 'fixes': 1.0, 'decisions_per_episode': 100.0}
                      for p, j in values.items()}, 'entropy_last20_mean': 1.0, 'last_entropy': 1.0, 'last_lr': 0.0,
            'ev': {}}


def flat(j, T=300_000):
    return {p: j for p in ea.GRID if p <= T}


def analyse(newJ, oldJ=None, classes=None, T=300_000):
    """newJ {(arm, block): {point: J}}."""
    new = {k: synth_curve(v) for k, v in newJ.items()}
    old = {(a, b): synth_curve(flat((oldJ or {}).get((a, b), 0.01))) for a in ea.ARMS for b in ea.BLOCKS}
    cls = {(a, b): 'completed' for a in ea.ARMS for b in ea.BLOCKS}
    cls.update(classes or {})
    return ea.analyse(new, old, cls, T, REF)


def all_blocks(j=0.05, T=300_000):
    return {(a, b): flat(j, T) for a in ea.ARMS for b in ea.BLOCKS}


def test_b2_reading_threshold_is_min_of_0103_and_half_rule():
    assert ea.reading_threshold(0.211329) == 0.103 and ea.reading_threshold(0.18) == 0.09
    J = all_blocks(0.05)
    J[('mlp', 4)] = flat(0.104)                                  # between 0.103 and J_rule / 2 = 0.10566
    out = analyse(J)
    assert out['systems_reading'].startswith('restated')
    assert out['systems_reading_at_paired_half_rule_descriptive'].startswith('holds')
    assert out['heads']['C-strong']['P3']['blocks_at_or_above_reading_threshold'] == [4]
    assert out['heads']['C-strong']['P3']['blocks_at_or_above_paired_half_rule'] == []
    J = all_blocks(0.05)
    for b in ea.BLOCKS:
        J[('mlp', b)] = flat(0.104)
    out = analyse(J)
    assert out['systems_reading'].startswith('withdrawn')
    assert out['systems_reading_at_paired_half_rule_descriptive'].startswith('holds')
    assert analyse(all_blocks(0.05))['systems_reading'].startswith('holds for best-effort, evidence-count-augmented')
    t = analyse(all_blocks(0.05))['heads']['C-strong']['E4_minus_reading_threshold_at_T']
    assert abs(t['mean'] - (0.05 - 0.103)) < 1e-12


def test_readings_name_T_not_300000():
    T = 210_000
    out = analyse(all_blocks(0.05, T), T=T)
    assert '210,000' in out['systems_reading'] and '300,000' not in json.dumps(
        [out['systems_reading'], out['systems_reading_at_paired_half_rule_descriptive']]
        + [h['P1_reading'] for h in out['heads'].values()])
    J = all_blocks(0.05, T)
    for b in ea.BLOCKS:
        J[('mlp', b)] = flat(0.06 + 0.001 * b, T)
    out = analyse(J, T=T)
    assert out['heads']['C-strong']['P1_reading'].endswith('at 210,000 decisions')


def test_holm_keeps_family_size_2_with_one_head():
    s, m = 0.01, 0.012746
    diffs = [m - s, m + s, m - s, m + s, m - s, m + s]
    p = ea.t_sum(diffs)['p']
    assert 0.025 < p < 0.05
    J = {}
    for i, b in enumerate(ea.BLOCKS):
        J[('mlp', b)] = flat(0.01 + diffs[i])
        J[('bottleneck', b)] = flat(0.01) if b > 6 else {q: 0.01 for q in ea.GRID if q <= 120_000}
    classes = {('bottleneck', b): 'algorithmic_failure' for b in (4, 5, 6)}
    out = analyse(J, classes=classes)
    cm, cs = out['heads']['C-matched'], out['heads']['C-strong']
    assert cm['P1_E4_minus_LONG_at_T'] is None and \
        cm['P1_reading'] == 'not computed: failed in 3 of 6 blocks (no value at 300,000 decisions)'
    assert out['holm'] == {'family_size': 2, 'heads_with_P1': ['C-strong']}
    assert cs['P1_E4_minus_LONG_at_T']['holm_rejected'] is False          # p in (0.025, 0.05): not at alpha / 2
    assert ea.holm([p]) == [True] and ea.holm([p], m=2) == [False]
    assert cm['P1_sensitivity_LOCF']['n'] == 6 and cm['n'] == 3


def test_peak_reported_descriptively_final_of_record():
    J = all_blocks(0.02)
    J[("mlp", 9)] = {**flat(0.0), 60_000: 0.12}
    out = analyse(J)
    d = out['heads']['C-strong']['intermediate_checkpoints_descriptive']
    assert d['at_or_above_reading_threshold'] == [(9, 60_000, 0.12)]
    assert d['highest_block_J_at_any_checkpoint'] == (9, 60_000, 0.12)
    assert out['systems_reading'].startswith('holds')


def test_p3_failures_never_give_holds():
    """Round 2, defect 1: the systems reading under failures (PLAN.md, 'P3 when blocks lack a value at T')."""
    fail = 'algorithmic_failure'
    # verifier case 1: C-strong fails in all 6 blocks after reaching 0.12 at 60,000; C-matched at 0.05
    J = all_blocks(0.05)
    for b in ea.BLOCKS:
        J[('mlp', b)] = {p: (0.12 if p == 60_000 else 0.0) for p in ea.GRID if p <= 60_000}
    out = analyse(J, classes={('mlp', b): fail for b in ea.BLOCKS})
    s = out['systems_reading']
    assert out['systems_reading_category'] == 'withdrawn' and not s.startswith('holds')
    assert 'carried forward' in s and 'C-strong n = 0 of 6' in s and 'not computable at T for C-strong' in s
    assert out['heads']['C-strong']['P3']['mean_J_at_T'] is None and out['heads']['C-strong']['n'] == 0
    # verifier case 2: two C-strong blocks fail after reaching 0.15 (LOCF above the threshold), the rest at 0.05
    J = all_blocks(0.05)
    for b in (4, 5):
        J[('mlp', b)] = {p: 0.15 for p in ea.GRID if p <= 270_000}
    out = analyse(J, classes={('mlp', 4): fail, ('mlp', 5): fail})
    s, p3 = out['systems_reading'], out['heads']['C-strong']['P3']
    assert out['systems_reading_category'] == 'restated' and s.startswith('restated')
    assert '4 (0.15000 at 270,000 decisions)' in s and 'C-strong n = 4 of 6' in s
    assert p3['failed_blocks_last_evaluated_J_at_or_above_reading_threshold'] == [4, 5]
    assert p3['blocks_at_or_above_reading_threshold'] == []
    # failed blocks whose last evaluated J is below the threshold: qualified, never 'holds'
    J = all_blocks(0.05)
    for b in (4, 5):
        J[('mlp', b)] = {p: 0.02 for p in ea.GRID if p <= 150_000}
    out = analyse(J, classes={('mlp', 4): fail, ('mlp', 5): 'namespace_end'})
    s = out['systems_reading']
    assert out['systems_reading_category'] == 'qualified' and s.startswith('qualified (not "holds")')
    assert 'block 5 (namespace_end; last evaluated 150,000 decisions, J 0.02000)' in s
    assert out['systems_reading_at_paired_half_rule_descriptive_category'] == 'qualified'
    # a head with no value at T and nothing at or above the threshold: not computable, never 'holds'
    J = all_blocks(0.05)
    for b in ea.BLOCKS:
        J[('mlp', b)] = {p: 0.02 for p in ea.GRID if p <= 90_000}
    out = analyse(J, classes={('mlp', b): fail for b in ea.BLOCKS})
    assert out['systems_reading_category'] == 'not computable'
    assert out['systems_reading'].startswith('not computable at 300,000 decisions: C-strong has no block')
    # a failed run without any evaluated point (no metrics): named, no LOCF value
    J = all_blocks(0.05)
    del J[('bottleneck', 7)]
    out = analyse(J, classes={('bottleneck', 7): 'code_failure'})
    assert out['systems_reading_category'] == 'qualified'
    assert 'block 7 (code_failure; no evaluated point)' in out['systems_reading']
    # withdrawn on the values at T, with a failure qualifying it
    J = all_blocks(0.12)
    J[('mlp', 4)] = {p: 0.02 for p in ea.GRID if p <= 150_000}
    out = analyse(J, classes={('mlp', 4): fail})
    assert out['systems_reading_category'] == 'withdrawn' and 'on the values at T' in out['systems_reading']
    # every block with a value at T and below the threshold: plain 'holds' (unchanged)
    out = analyse(all_blocks(0.05))
    assert out['systems_reading_category'] == 'holds' and out['systems_reading'].startswith('holds for best-effort')


def test_three_way_core():
    assert ea.three_way({'A': {4: 0.2, 5: 0.0, 6: 0.0}, 'B': {4: 0.0}}, 0.1) == ('restated', {'A': [4]})
    assert ea.three_way({'A': {4: 0.2, 5: 0.0}, 'B': {4: 0.0}}, 0.1) == ('withdrawn', ['A'])     # mean at theta
    assert ea.three_way({'A': {4: 0.2, 5: 0.1}, 'B': {}}, 0.1) == ('withdrawn', ['A'])
    assert ea.three_way({'A': {4: 0.05}, 'B': {4: 0.0}}, 0.1) == ('holds', None)


def test_failed_run_with_a_value_at_T_counts():
    """Round 2, defect 2: a run that failed after evaluating T has a value at T; the 3-failure rule counts only
    failed blocks without a value at T."""
    T = 180_000
    J = all_blocks(0.05, T)
    for i, b in enumerate(ea.BLOCKS):
        J[('mlp', b)] = flat(0.05 + 0.002 * i, T)
    for b in (4, 5, 6):                                       # failed (before the deadline) after evaluating 240,000
        J[('bottleneck', b)] = {p: 0.04 for p in ea.GRID if p <= 240_000}
    out = analyse(J, classes={('bottleneck', b): 'algorithmic_failure' for b in (4, 5, 6)}, T=T)
    cm = out['heads']['C-matched']
    assert cm['n'] == 6 and cm['blocks_without_value_at_T'] == {} and cm['failed_blocks_with_value_at_T'] == [4, 5, 6]
    assert cm['P1_E4_minus_LONG_at_T'] is not None and cm['P1_E4_minus_LONG_at_T']['n'] == 6
    assert 'P1_sensitivity_LOCF' not in cm and 'not computed' not in cm['P1_reading']
    assert out['systems_reading_category'] == 'holds' and '180,000' in out['systems_reading']
    assert out['holm']['heads_with_P1'] == ['C-strong', 'C-matched']
    # the same three runs failing before T: no value at T, P1 not computed, the reading is qualified
    for b in (4, 5, 6):
        J[('bottleneck', b)] = {p: 0.04 for p in ea.GRID if p <= 120_000}
    out = analyse(J, classes={('bottleneck', b): 'algorithmic_failure' for b in (4, 5, 6)}, T=T)
    cm = out['heads']['C-matched']
    assert cm['n'] == 3 and sorted(cm['blocks_without_value_at_T']) == [4, 5, 6] and cm['P1_E4_minus_LONG_at_T'] is None
    assert cm['P1_reading'] == 'not computed: failed in 3 of 6 blocks (no value at 180,000 decisions)'
    assert out['systems_reading_category'] == 'qualified'
    # a running or completed block without a value at T is an inconsistency: refused
    J[('bottleneck', 4)] = {p: 0.04 for p in ea.GRID if p <= 120_000}
    with pytest.raises(SystemExit, match='running or completed block without a value'):
        analyse(J, classes={('bottleneck', b): 'algorithmic_failure' for b in (5, 6)}, T=T)


def test_slope_last3_for_E4_LONG_and_difference():
    """Round 2, defect 3: the plan's 'E4 - LONG on ... the slope over the last three checkpoints'."""
    J = {(a, b): {p: 0.01 + 0.003 * p / 30_000 for p in ea.GRID} for a in ea.ARMS for b in ea.BLOCKS}
    sl = analyse(J)['heads']['C-strong']['slope_last3_per_30k']
    assert sl['checkpoints'] == [240_000, 270_000, 300_000]
    assert abs(sl['E4']['mean'] - 0.003) < 1e-12 and abs(sl['LONG']['mean']) < 1e-12
    assert abs(sl['E4_minus_LONG']['mean'] - 0.003) < 1e-12
    T = 210_000
    J = {(a, b): {p: 0.01 + 0.003 * p / 30_000 for p in ea.GRID if p <= T} for a in ea.ARMS for b in ea.BLOCKS}
    assert analyse(J, T=T)['heads']['C-matched']['slope_last3_per_30k']['checkpoints'] == [150_000, 180_000, 210_000]


def test_deadline_constant_matches_the_plan():
    assert ea.DEADLINE_UTC == '2026-10-30T12:00:00+00:00'
    plan = HERE / 'PLAN.md'
    if plan.exists():
        assert '2026-10-30T12:00Z' in plan.read_text()


# ------------------------------------------------------------------ e4_analysis end to end on LONG stand-ins -----
D = datetime.datetime(2026, 10, 30, 12, tzinfo=datetime.timezone.utc)
ENT = {'mlp': 0.01, 'bottleneck': 0.003}
PRE = {'passed': True, 'runner_sha256': 'a' * 64, 'e4lib_sha256': 'b' * 64}


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec='seconds')


def make_standin(root, arm, block, digest, end_ts, eval_ts):
    """LONG's stored run copied as an E4 stand-in: only the identity, the schedule log, the freeze and the times are
    patched (no E4 data exists or is read). eval_ts(point) -> POSIX time at which point's checkpoint was saved."""
    src = e4.LONG_DIR / 'runs' / f'{arm}__b{block}' / 'run' / arm / 'metrics.jsonl'
    arm_dir = root / 'runs' / f'{arm}__b{block}' / 'run' / arm
    (arm_dir / 'checkpoints').mkdir(parents=True)
    out, before = [], 0
    for line in src.read_text().splitlines():
        e = json.loads(line)
        k = e['kind']
        if k == 'arm_start':
            e['identity']['condition']['obs_encoding_version'] = e4.OBS_VERSION
            e['model']['config']['obs_dim'] = 270
        elif k == 'update':
            e['metrics'].update(decisions_before_update=before, entropy_coef=ENT[arm],
                                learning_rate=e4.scheduled_lr(e4.LR0[arm], e4.LR_HORIZON, before))
            before = e['cumulative']['policy_decisions']
        elif k == 'checkpoint':
            f = arm_dir / e['path']
            f.write_text('')
            t = eval_ts(e['points'][0]) if e['points'] else end_ts - 1
            os.utime(f, (t, t))
        elif k == 'arm_end':
            e.update(protocol_freeze_sha256=digest, utc=iso(end_ts))
        else:
            out.append(line)
            continue
        out.append(json.dumps(e, sort_keys=True))
    (arm_dir / 'metrics.jsonl').write_text('\n'.join(out) + '\n')
    task = {'kind': 'main', 'arm': arm, 'block': block, 'entropy': ENT[arm], 'lr0': e4.LR0[arm], 'info': True,
            'schedule': True, 'study': 'dq_b2', 'budget': 300_000, 'grid': ea.GRID, 'freeze': digest,
            'name': f'{arm}__b{block}', 'out': str(root / 'runs')}
    r = {'kind': 'main', 'name': f'{arm}__b{block}', 'arm': arm, 'block': block, 'status': 'completed', 'task': task,
         'config': {'ppo': {'entropy_coef': ENT[arm]}, 'protocol_freeze_sha256': digest},
         'code': {'runner_sha256': PRE['runner_sha256'], 'e4lib_sha256': PRE['e4lib_sha256'],
                  'idea_dq_tree': e4.EXPECTED_TREE}, 'wall_s': 1.0}
    (root / 'records').mkdir(exist_ok=True)
    (root / 'records' / f'{arm}__b{block}.json').write_text(json.dumps(r))
    os.utime(root / 'records' / f'{arm}__b{block}.json', (end_ts, end_ts))


@pytest.fixture(scope='module')
def standin_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('standin')
    digest = write_plan(root)
    (root / 'pilot').mkdir()
    (root / 'pilot' / 'selection.json').write_text(json.dumps({'heads': {a: {'entropy_coef': e} for a, e in ENT.items()}}))
    (root / 'precheck.json').write_text(json.dumps(PRE))
    (root / 'launches.jsonl').write_text(json.dumps({'plan_sha256': digest, 'entropy_coef': ENT, **PRE}) + '\n')
    base = D.timestamp() - 3 * 86400
    for a in ea.ARMS:
        for b in ea.BLOCKS:
            make_standin(root, a, b, digest, D.timestamp() - 86400, lambda p: base + p / 10)
    return root


def fresh(standin_root, tmp_path):
    dest = tmp_path / 'root'
    shutil.copytree(standin_root, dest, copy_function=shutil.copy2)
    return dest


def metrics(root, arm, b):
    return root / 'runs' / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl'


def test_full_readout_and_no_overwrite(standin_root, tmp_path):
    root = fresh(standin_root, tmp_path)
    out = ea.main(root, 'record', now=D - datetime.timedelta(days=1), deadline=D)
    assert out['readout'] == 'full' and out['of_record'] and out['readout_checkpoint'] == 300_000
    assert out['systems_reading'].startswith('holds') and '300,000' in out['systems_reading']
    assert all(h['P1_E4_minus_LONG_at_T']['mean'] == 0 for h in out['heads'].values())       # stand-ins = LONG
    assert out['thresholds']['reading_threshold_min_0.103_and_half_rule'] == 0.103
    # the LONG slope is computed as long_analysis.py computed it (round 2, defect 3)
    stored = json.loads((e4.LONG_DIR / 'long_results.json').read_text())['LONG']
    for role, h in out['heads'].items():
        assert abs(h['slope_last3_per_30k']['LONG']['mean'] - stored[role]['slope_last3_per_30k']['mean']) < 1e-15
        assert h['slope_last3_per_30k']['E4_minus_LONG']['mean'] == 0
    with pytest.raises(SystemExit, match='never overwritten'):
        ea.main(root, 'record', now=D - datetime.timedelta(days=1), deadline=D)
    ea.main(root, 'record', force=True, now=D - datetime.timedelta(days=1), deadline=D)
    log = [json.loads(x) for x in (root / 'e4_analysis_log.jsonl').read_text().splitlines()]
    assert len(log) == 2 and log[1]['forced'] and (root / log[1]['replaced_kept_as']).exists()
    with pytest.raises(SystemExit, match='copy'):
        ea.main(ea.HERE, 'provisional')


@pytest.mark.parametrize('mutation, reason', [('entropy', 'entropy_coef'), ('duplicate', 'duplicated lines'),
                                              ('code', 'code hashes'), ('plan', 'PLAN.sha256'),
                                              ('pending_restart', 'awaits its one restart'), ('point0', 'point-0'),
                                              ('freeze', 'protocol_freeze')])
def test_checks_refuse(standin_root, tmp_path, mutation, reason):
    root = fresh(standin_root, tmp_path)
    m = metrics(root, 'mlp', 4)
    r = root / 'records' / 'mlp__b4.json'
    if mutation == 'entropy':
        m.write_text(m.read_text().replace('"entropy_coef": 0.01', '"entropy_coef": 0.003', 1))
    elif mutation == 'duplicate':
        lines = m.read_text().splitlines(keepends=True)
        m.write_text(''.join(lines[:10] + lines[9:10] + lines[10:]))
    elif mutation == 'code':
        rec_ = json.loads(r.read_text())
        rec_['code']['runner_sha256'] = 'c' * 64
        r.write_text(json.dumps(rec_))
    elif mutation == 'plan':
        (root / 'PLAN.md').write_text((root / 'PLAN.md').read_text() + 'edited after hashing\n')
    elif mutation == 'pending_restart':
        rec_ = json.loads(r.read_text())
        rec_.update(status='failure', error='OSError: simulated')
        r.write_text(json.dumps(rec_))
    elif mutation == 'point0':
        lines = m.read_text().splitlines(keepends=True)
        i = next(i for i, x in enumerate(lines) if '"kind": "evaluation"' in x)
        e = json.loads(lines[i])
        e['results']['sample']['episodes'][0]['J'] += 1e-12
        e['results']['sample']['weighted_J'] = p4.aggregate_evaluation(ea.runtime(), e['results']['sample']['episodes'])[1]
        lines[i] = json.dumps(e, sort_keys=True) + '\n'
        m.write_text(''.join(lines))
    elif mutation == 'freeze':
        rec_ = json.loads(r.read_text())
        rec_['config']['protocol_freeze_sha256'] = 'd' * 64
        r.write_text(json.dumps(rec_))
    with pytest.raises(SystemExit, match=reason):
        ea.main(root, 'record', now=D - datetime.timedelta(days=1), deadline=D)
    assert not (root / 'e4_results.json').exists()


def truncate_after_evaluation(path, point, extra=3, partial=True):
    lines = path.read_text().splitlines(keepends=True)
    i = next(i for i, x in enumerate(lines) if '"kind": "evaluation"' in x and json.loads(x)['points'] == [point])
    path.write_text(''.join(lines[:i + 1 + extra]) + ('{"kind": "epis' if partial else ''))


def test_deadline_readout(standin_root, tmp_path):
    root = fresh(standin_root, tmp_path)
    # C-strong block 5 is still running: evaluated up to 210,000, but 210,000 only after the deadline
    (root / 'records' / 'mlp__b5.json').unlink()
    truncate_after_evaluation(metrics(root, 'mlp', 5), 210_000)
    ck = metrics(root, 'mlp', 5).parent / 'checkpoints'
    for f in ck.iterdir():
        d = int(f.name.split('decisions')[1][:8])
        t = D.timestamp() + 3600 if d >= 210_000 else D.timestamp() - 36000
        os.utime(f, (t, t))
    # C-matched block 6 completed after the deadline; its 300,000 evaluation came after the deadline
    m6 = metrics(root, 'bottleneck', 6)
    m6.write_text(m6.read_text().replace(iso(D.timestamp() - 86400), iso(D.timestamp() + 5 * 3600)))
    for f in (m6.parent / 'checkpoints').iterdir():
        if f.name.startswith('final_') or int(f.name.split('decisions')[1][:8]) >= 300_000:
            os.utime(f, (D.timestamp() + 4 * 3600,) * 2)
    os.utime(root / 'records' / 'bottleneck__b6.json', (D.timestamp() + 5 * 3600,) * 2)
    with pytest.raises(SystemExit, match='waits'):
        ea.main(root, 'record', now=D - datetime.timedelta(hours=1), deadline=D)
    out = ea.main(root, 'record', now=D + datetime.timedelta(days=2), deadline=D)
    assert out['readout'] == 'deadline' and out['readout_checkpoint'] == 180_000
    assert out['deadline_cut']['mlp__b5'] == {'class_at_cut': 'running', 'reached_by_cut': 180_000, 'enters_T': True}
    assert out['deadline_cut']['bottleneck__b6']['class_at_cut'] == 'running'
    assert out['deadline_cut']['bottleneck__b6']['reached_by_cut'] == 270_000
    assert '180,000' in out['systems_reading'] and all(h['n'] == 6 for h in out['heads'].values())
    # once every run has ended, the 300,000 read-out is supplementary, never of record
    with pytest.raises(SystemExit, match='waits'):
        ea.main(root, 'supplementary', now=D + datetime.timedelta(days=2), deadline=D)


def test_deadline_readout_failed_run_keeps_its_value_at_T(standin_root, tmp_path):
    """Round 2, defect 2, end to end: a run that failed before the deadline after evaluating T keeps its J_T (n = 6),
    does not enter T, and its curve stops at T like every other run's."""
    root = fresh(standin_root, tmp_path)
    (root / 'records' / 'mlp__b5.json').unlink()                # running at the deadline, 180,000 reached by then
    truncate_after_evaluation(metrics(root, 'mlp', 5), 210_000)
    for f in (metrics(root, 'mlp', 5).parent / 'checkpoints').iterdir():
        t = D.timestamp() + 3600 if int(f.name.split('decisions')[1][:8]) >= 210_000 else D.timestamp() - 36000
        os.utime(f, (t, t))
    r = root / 'records' / 'bottleneck__b9.json'                # namespace end after its 240,000 evaluation
    rec_ = json.loads(r.read_text())
    rec_.update(status='failure', error='P4Error: block 9 leaves the dq_e2_training namespace')
    r.write_text(json.dumps(rec_))
    truncate_after_evaluation(metrics(root, 'bottleneck', 9), 240_000, extra=5, partial=False)
    os.utime(r, (D.timestamp() - 86400,) * 2)
    out = ea.main(root, 'record', now=D + datetime.timedelta(days=2), deadline=D)
    assert out['readout'] == 'deadline' and out['readout_checkpoint'] == 180_000
    assert out['deadline_cut']['bottleneck__b9'] == {'class_at_cut': 'namespace_end', 'reached_by_cut': 240_000,
                                                     'enters_T': False}
    cm = out['heads']['C-matched']
    assert cm['n'] == 6 and cm['failed_blocks'] == {9: 'namespace_end'} and cm['failed_blocks_with_value_at_T'] == [9]
    assert cm['blocks_without_value_at_T'] == {} and cm['P1_E4_minus_LONG_at_T']['n'] == 6
    assert 'P1_sensitivity_LOCF' not in cm
    assert out['runs']['bottleneck__b9']['evaluated_points'][-1] == 180_000
    assert out['systems_reading_category'] in ('holds', 'restated', 'withdrawn')     # every block has a value at T


def test_namespace_end_and_kept_copy_prefix(standin_root, tmp_path):
    root = fresh(standin_root, tmp_path)
    # C-matched block 9 ends by the namespace-end P4Error after its 240,000 evaluation: algorithmic, no 300k value
    r = root / 'records' / 'bottleneck__b9.json'
    rec_ = json.loads(r.read_text())
    rec_.update(status='failure', error='P4Error: block 9 leaves the dq_e2_training namespace')
    r.write_text(json.dumps(rec_))
    os.utime(r, (D.timestamp() - 86400,) * 2)
    truncate_after_evaluation(metrics(root, 'bottleneck', 9), 240_000, extra=5, partial=False)
    # C-strong block 4 has a kept copy from a crash (its first 400 events)
    kept = root / 'interrupted' / '2026-10-26T000000Z' / 'mlp__b4' / 'run' / 'mlp'
    kept.mkdir(parents=True)
    lines = metrics(root, 'mlp', 4).read_text().splitlines(keepends=True)
    (kept / 'metrics.jsonl').write_text(''.join(lines[:400]) + '{"kind": "ep')
    (root / 'restarts.jsonl').write_text(json.dumps({'name': 'mlp__b4', 'kind': 'interrupted',
                                                     'moved_to': 'interrupted/2026-10-26T000000Z/mlp__b4'}) + '\n')
    out = ea.main(root, 'record', now=D - datetime.timedelta(days=1), deadline=D)
    cm = out['heads']['C-matched']
    assert out['readout'] == 'full' and cm['failed_blocks'] == {9: 'namespace_end'} and cm['n'] == 5
    assert cm['P1_sensitivity_LOCF']['n'] == 6 and out['runs']['bottleneck__b9']['class'] == 'namespace_end'
    assert out['systems_reading_category'] in ('qualified', 'restated', 'withdrawn')   # never 'holds' with a failure
    assert 'C-matched n = 5 of 6 with a value at T' in out['systems_reading']
    assert 'block 9 (namespace_end; last evaluated 240,000 decisions' in out['systems_reading']
    assert out['restart_prefix_comparisons']['mlp__b4'] == [
        {'compared_events': 400, 'copy_events': 400, 'current_events': len(lines), 'identical': True,
         'first_mismatch_event': None, 'kind': 'interrupted', 'moved_to': 'interrupted/2026-10-26T000000Z/mlp__b4'}]
    # a kept copy that differs is reported (determinism documented), not a refusal
    j = next(i for i in range(100, 400) if '"kind": "episode"' in lines[i])
    (kept / 'metrics.jsonl').write_text(''.join(lines[:j]) + lines[j].replace('"seed": ', '"seed": 1', 1)
                                        + ''.join(lines[j + 1:400]))
    out = ea.main(root, 'record', force=True, now=D - datetime.timedelta(days=1), deadline=D)
    assert out['restart_prefix_comparisons']['mlp__b4'][0]['identical'] is False


def test_restart_comparison_on_two_precheck_generations():
    """The 2026-10-25 precheck re-run against the first precheck: identical except checkpoint file hashes."""
    prev = HERE / 'precheck_prev_2026-10-25T121716Z' / 'precheck' / 'runs'
    if not prev.exists() or not (HERE / 'precheck' / 'runs').exists():
        pytest.skip('precheck generations not present')
    for name in ('a_mlp__b4', 'a_bottleneck__b4', 'c_mlp__b4', 'c_bottleneck__b4'):
        arm = name.split('_', 1)[1].split('__')[0]
        _, old = ea.read_lines(prev / name / 'run' / arm / 'metrics.jsonl')
        _, new = ea.read_lines(HERE / 'precheck' / 'runs' / name / 'run' / arm / 'metrics.jsonl')
        cmp = ea.prefix_comparison(old, new)
        assert cmp['identical'] and cmp['compared_events'] == len(old) == len(new), (name, cmp)


def test_supplementary_after_deadline_and_provisional_on_a_copy(standin_root, tmp_path):
    root = fresh(standin_root, tmp_path)
    m5, r5 = metrics(root, 'mlp', 5), root / 'records' / 'mlp__b5.json'
    full_metrics, full_record = m5.read_text(), r5.read_text()
    r5.unlink()
    truncate_after_evaluation(m5, 210_000)
    prov = ea.main(root, 'provisional', now=D - datetime.timedelta(days=1), deadline=D)
    assert prov['readout'] == 'provisional' and not prov['of_record'] and prov['readout_checkpoint'] == 210_000
    assert (root / 'e4_results_PROVISIONAL.json').exists() and not (root / 'e4_results.json').exists()
    # the deadline passes with block 5 running (all its evaluations count as reached by the deadline here)
    out = ea.main(root, 'record', now=D + datetime.timedelta(hours=1), deadline=D)
    assert out['readout'] == 'deadline' and out['of_record'] and out['readout_checkpoint'] == 210_000
    # block 5 then completes after the deadline: its 300,000 read-out is supplementary, not of record
    m5.write_text(full_metrics.replace(iso(D.timestamp() - 86400), iso(D.timestamp() + 7200)))
    r5.write_text(full_record)
    os.utime(r5, (D.timestamp() + 7200,) * 2)
    sup = ea.main(root, 'supplementary', now=D + datetime.timedelta(hours=3), deadline=D)
    assert sup['readout'] == 'supplementary_full_after_deadline' and not sup['of_record']
    assert sup['readout_checkpoint'] == 300_000 and '300,000' in sup['systems_reading']
    assert json.loads((root / 'e4_results.json').read_text())['readout_checkpoint'] == 210_000   # unchanged
    # and the read-out of record is still the deadline read-out when re-derived (it would need --force anyway)
    with pytest.raises(SystemExit, match='never overwritten'):
        ea.main(root, 'record', now=D + datetime.timedelta(hours=3), deadline=D)
