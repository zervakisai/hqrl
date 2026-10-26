"""E3 (post hoc): early training at matched per-update KL in the B2 protocol (see PLAN_DRAFT.md, later PLAN.md).
Imports the B2 code tree of idea_dq unchanged; only Adam's constant learning rate differs from the stored B2 runs.

    python runner.py precheck                 # (a) HQ and (b) twin (with P4 parity) at B2's stored learning rates
                                              #     reproduce stored B2 block-0 events for 3,000 decisions (2 processes);
                                              #     (c)-(h) static checks (configs, namespaces, targets, slopes, tree)
    python runner.py pilot --workers 2        # development calibration pilot: STUDY_SMOKE / dq_development, no
                                              #     evaluation; resumable (records/ skip, interrupted/ move)
    python runner.py pilot-select             # the pre-declared selection rule -> selection.json (reads approx_kl only)
    python runner.py launch --workers 8       # the 72 main runs (needs PLAN.md + PLAN.sha256 written by the lead,
                                              #     PLAN.md recording the sha256 of runner.py and b2k_analysis.py,
                                              #     selection.json, a passed precheck.json of this runner.py)

Nothing is written outside this folder. dq_b2.run_one / launch / smoke (which pass dq.require_open and refuse after
2026-10-31T00:00Z) are never called; p4.train_arm is called directly, as in the other post-hoc runners.

Fixes after review (2026-10-25, REVIEW.md M7/M8): precheck, pilot and launch hold an exclusive flock on runner.lock
for their whole life (fork-pool workers inherit it, so a relaunch is refused while any worker of an earlier launch is
alive); every record is written atomically (temporary file, fsync, os.replace); a record that does not parse counts as
no record: it and its run directory are moved to interrupted/<UTC>/ and the run restarts from the beginning.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))

import numpy as np                                                                                # noqa: E402
import torch                                                                                      # noqa: E402
import dq_b2                                                                                      # noqa: E402
import p4_train as p4                                                                             # noqa: E402

B2_CAMPAIGN = IDEA_DQ / 'output/study/gnss_qrl_v1/dq/b2_campaign_v3'
B2_RUNS = B2_CAMPAIGN / 'runs'
POSTHOC = Path('/home/konstantinos/Documents/ChatGPT')
CMLR_RUNS = POSTHOC / 'idea_dq_posthoc_cmlr/runs'
LR_RUNS = POSTHOC / 'idea_dq_posthoc_lr/runs'
B2_TREE = 'ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952'
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PLAN = HERE / 'PLAN.md'
PLAN_SHA256 = hashlib.sha256(PLAN.read_bytes()).hexdigest() if PLAN.exists() else None
DRAFT_LABEL = 'e' * 64          # stands in for the plan hash in precheck/pilot configs; never a plan hash
LOCK = HERE / 'runner.lock'
ANALYSIS = HERE / 'b2k_analysis.py'

# ------------------------------------------------------------------ DESIGN (PLAN_DRAFT.md) ---------------------------
#: main sets: tag -> (arm, target name). T2 = K-H2 arm (twin), T1 = K-H1 arm (HQ).
SETS = {'T1': 'quantum', 'T2': 'surrogate625'}
#: matched statistic of the comparator: mean over blocks 0-35 (ascending) of the run mean of update approx_kl (stored)
TARGETS = {'surrogate625': {'value': 0.007823255686871039, 'source': 'stored B2 HQ (quantum, lr 3e-3), blocks 0-35'},
           'quantum': {'value': 0.00419994415996206, 'source': 'stored CMLR C-matched (bottleneck, lr 1e-2), blocks 0-35'}}
#: slope of ln(mean KL) on ln(lr) between the two stored learning rates 1e-3 and 3e-3 of the same arm (stored records)
SLOPES = {'surrogate625': 0.9385004171529107, 'quantum': 0.8421691057010121}
#: pilot candidates (log-spaced around the stored-point prediction: twin 7.76e-4, HQ 1.43e-3)
CANDIDATES = {'surrogate625': (5e-4, 7e-4, 1e-3), 'quantum': (1e-3, 1.4e-3, 2e-3)}
PILOT_PER_CANDIDATE = 3                      # disjoint development blocks per candidate learning rate
PILOT_FIRST_BLOCK = 90
PILOT_BUDGET = 30_000
PILOT_GRID = [0]                             # initial checkpoint only (plus the final one); no evaluation
RANGE_FACTOR = 1.5                           # range guard: selected lr within [min cand / 1.5, max cand * 1.5]
STAGE_1B_SPACING = 1.4
TOLERANCE = 0.25                             # achieved mean KL within +-25% of the target (analysis)
STORED_LR = {arm: dq_b2.ARMS[arm]['learning_rate'] for arm in SETS.values()}   # quantum 3e-3, surrogate625 1e-3
# ------------------------------------------------------------------ END DESIGN -------------------------------------


def pilot_blocks(index):
    """Disjoint development blocks of candidate `index` (0, 1, 2): 90-92, 93-95, 96-98."""
    start = PILOT_FIRST_BLOCK + PILOT_PER_CANDIDATE * index
    return tuple(range(start, start + PILOT_PER_CANDIDATE))


def lr_str(lr):
    """Canonical 3-significant-digit spelling of a learning rate (run names, selection.json, PLAN.md)."""
    return format(float(lr), '.2e')


def round3(lr):
    return float(lr_str(lr))


def candidate_name(lr):
    name = f'e3_lr{lr_str(lr)}_ent0'
    # E1's validator admits only listed candidates; this process adds the post-hoc one (as the hqlr/cmlr runners do).
    # STUDY_B2 and STUDY_SMOKE share dq_b2.CANDIDATES, in this process only.
    dq_b2.STUDY_B2.candidates.setdefault(name, {'learning_rate': float(lr), 'entropy_coef': 0.0})
    return name


def config_main(arm, block, lr, budget=dq_b2.BUDGET, grid=None, plan=None):
    c = dq_b2.config_for(arm, block, dq_b2.STUDY_B2, budget, grid or dq_b2.GRID, freeze_sha256=plan or DRAFT_LABEL)
    c['ppo'] = dict(c['ppo'], learning_rate=float(lr))
    c['candidate'] = candidate_name(lr)
    c['note'] = (f'post-hoc E3 matched-KL check (plan {(plan or "draft")[:8]}): {dq_b2.ARMS[arm]["role"]} '
                 f'lr {lr_str(lr)} block {block}')
    return c


def config_pilot(arm, block, lr):
    c = dq_b2.config_for(arm, block, dq_b2.STUDY_SMOKE, PILOT_BUDGET, PILOT_GRID, freeze_sha256=dq_b2.DEV_FREEZE_LABEL,
                         role='dq_b2_smoke')
    c['ppo'] = dict(c['ppo'], learning_rate=float(lr))
    c['candidate'] = candidate_name(lr)
    c['evaluation'] = None                    # no evaluation episode: the pilot yields no J curve and no AUC
    c['note'] = f'E3 calibration pilot (development): {dq_b2.ARMS[arm]["role"]} lr {lr_str(lr)} block {block}'
    return c


def run_name(kind, tag, arm, block, lr):
    return f'P_{arm}_lr{lr_str(lr)}__b{block}' if kind == 'pilot' else f'{tag}_{arm}__b{block}'


def kl_of(metrics_path):
    """The matched statistic of one run: mean over its update records of metrics.approx_kl (reads nothing else)."""
    kls = []
    with open(metrics_path) as handle:
        for line in handle:
            if '"kind": "update"' in line:
                kls.append(float(json.loads(line)['metrics']['approx_kl']))
    return {'mean': float(np.mean(kls)) if kls else None, 'median': float(np.median(kls)) if kls else None,
            'updates': len(kls)}


def run_one(task):
    """One run (main, pilot or precheck); returns its record. Failures are recorded, never retried."""
    torch.set_num_threads(1)                  # first, as dq_b2.run_one does, before any torch work in this process
    kind, tag, arm, block, lr, out = (task[k] for k in ('kind', 'tag', 'arm', 'block', 'lr', 'out'))
    rec = {'kind': kind, 'set': tag, 'arm': arm, 'block': block, 'learning_rate': float(lr), 'status': 'started',
           'plan_sha256': PLAN_SHA256, 'runner_sha256': RUNNER_SHA256}
    t0 = time.perf_counter()
    try:
        if kind == 'pilot':
            study, cfg = dq_b2.STUDY_SMOKE, config_pilot(arm, block, lr)
        else:
            study = dq_b2.STUDY_B2
            cfg = config_main(arm, block, lr, task.get('budget', dq_b2.BUDGET), task.get('grid'),
                              None if kind == 'precheck' else PLAN_SHA256)
        rt = dq_b2.runtime(study)
        if kind == 'pilot':                   # namespace guard: development seeds only
            ns = rt.namespaces[study.training_namespace]
            first, last = ns['start'] + block * p4.BLOCK_STRIDE, ns['start'] + block * p4.BLOCK_STRIDE + p4.BLOCK_STRIDE - 1
            assert study.training_namespace == 'dq_development' and ns['start'] <= first and last < ns['end'], ns
        run_dir = Path(out) / run_name(kind, tag, arm, block, lr)
        if run_dir.exists():
            raise RuntimeError(f'{run_dir} exists: no overwrite, no resume')
        rec['config'] = cfg
        rec['code_tree_sha256'] = p4.live_copy_tree()
        if arm == 'surrogate625':
            rec['twin_parity'] = {'P4': dq_b2.twin_trajectory_parity(study, rt, cfg, block)}
        summary = p4.train_arm(arm, cfg, rt, run_dir / 'run' / arm)
        if arm == 'surrogate625':
            rec['twin_parity'].update(dq_b2.TWIN_RECORDS[(study.name, block)])
        rec.update(status=summary['status'], summary=summary, kl=kl_of(run_dir / 'run' / arm / 'metrics.jsonl'))
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def single_instance(path=LOCK):
    """M7: one training command (precheck, pilot, launch) of this folder at a time. An exclusive, non-blocking flock
    on an open file description: the fork-pool workers inherit it, so if only the launcher is killed, the lock stays
    held until every surviving worker has exited, and a second launch is refused instead of writing into the same
    run directories. Returns the open handle; the caller keeps it for its whole life."""
    handle = open(path, 'a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit(f'{path} is locked: another precheck/pilot/launch of this folder, or a surviving worker of '
                         'one, is still running (stop its whole process group first)')
    handle.truncate(0)
    handle.write(json.dumps({'pid': os.getpid(), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             'argv': sys.argv}) + '\n')
    handle.flush()
    return handle


def write_json_atomic(path, obj):
    """M8: a record appears complete or not at all (temporary file in the same directory, fsync, os.replace)."""
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'w') as handle:
        handle.write(json.dumps(obj, indent=1, default=float))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_record(path):
    """A record that does not parse counts as no record (M8): returns None for it."""
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def prepare(names, runs, records, interrupted):
    """The resume rule, applied without looking at any partial result. A run with a parsable record of any status is
    never run again. Otherwise an unparsable record, the run directory (killed by a crash) and any stale temporary
    record are moved unchanged to `interrupted` and the run restarts from the beginning. Returns (to run, moved)."""
    runs, records, interrupted = Path(runs), Path(records), Path(interrupted)
    moved = []

    def move(src):
        interrupted.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(interrupted / src.name))
        moved.append(src.name)
    for stale in sorted(records.glob('.*.tmp')):
        move(stale)
    todo = []
    for name in names:
        rec = records / f'{name}.json'
        if rec.exists():
            if read_record(rec) is not None:
                continue
            move(rec)
        if (runs / name).exists():
            move(runs / name)
        todo.append(name)
    return todo, moved


def pool_run(tasks, workers, write):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_one, tasks):
            write(rec)
            print(rec['kind'], rec['set'], rec['arm'], rec['block'], lr_str(rec['learning_rate']), rec['status'],
                  round(rec['wall_s']), flush=True)


# -- stored statistics (KL only) -----------------------------------------------------------------------------------------

def stored_kl(root, pattern, arm, blocks=range(36)):
    out = {}
    for b in blocks:
        out[b] = kl_of(Path(root) / pattern.format(b=b) / 'run' / arm / 'metrics.jsonl')['mean']
    return out


def stored_targets_and_slopes():
    hq3 = stored_kl(B2_RUNS, 'quantum__b{b}', 'quantum')
    cm2 = stored_kl(CMLR_RUNS, 'C1_bottleneck__b{b}', 'bottleneck')
    hq1 = stored_kl(LR_RUNS, 'L2_quantum__b{b}', 'quantum')
    tw1 = stored_kl(B2_RUNS, 'surrogate625__b{b}', 'surrogate625')
    tw3 = stored_kl(LR_RUNS, 'L1_surrogate625__b{b}', 'surrogate625')
    mean = lambda d: float(np.mean([d[b] for b in range(36)]))                     # noqa: E731
    slope = lambda k1, k3: float(math.log(k3 / k1) / math.log(3e-3 / 1e-3))      # noqa: E731
    return {'targets': {'surrogate625': mean(hq3), 'quantum': mean(cm2)},
            'slopes': {'surrogate625': slope(mean(tw1), mean(tw3)), 'quantum': slope(mean(hq1), mean(hq3))},
            'stored_means': {'HQ_1e-3': mean(hq1), 'HQ_3e-3': mean(hq3), 'twin_1e-3': mean(tw1), 'twin_3e-3': mean(tw3),
                             'Cmatched_1e-2': mean(cm2)}}


# -- precheck ------------------------------------------------------------------------------------------------------------

def _load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def _max_rel(a, b):
    worst = 0.
    for k in set(a) | set(b):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if x != y:
                worst = max(worst, abs(x - y) / max(abs(y), 1e-300))
        elif x != y:
            return math.inf
    return worst


def compare_events(new_path, old_path):
    new, old = _load(new_path), _load(old_path)
    pick = lambda ev, kind: [e for e in ev if e['kind'] == kind]                   # noqa: E731
    strip = lambda e: {k: v for k, v in e.items() if k not in ('utc', 'wall_s', 'elapsed_wall_s')}      # noqa: E731
    checks = {}
    a, b = pick(new, 'evaluation'), pick(old, 'evaluation')
    checks['evaluation'] = {'compared': len(a), 'identical': all(x['results'] == y['results'] for x, y in zip(a, b))}
    a, b = pick(new, 'update'), pick(old, 'update')
    rel = max([_max_rel(x['metrics'], y['metrics']) for x, y in zip(a, b)] or [math.inf])
    checks['update'] = {'compared': len(a), 'identical': all(x['metrics'] == y['metrics'] for x, y in zip(a, b)),
                        'max_relative_difference': rel}
    a, b = pick(new, 'episode'), pick(old, 'episode')
    checks['episode'] = {'compared': len(a), 'identical': all(strip(x) == strip(y) for x, y in zip(a, b))}
    a, b = pick(new, 'arm_start')[0], pick(old, 'arm_start')[0]
    checks['arm_start'] = {'compared': 1, 'identical': all(a[k] == b[k] for k in ('model', 'identity', 'seeds', 'study'))}
    return checks


def compare_initial_checkpoint(new_dir, old_dir):
    ck = lambda d: next((Path(d) / 'checkpoints').glob('update00000_*.pt'))         # noqa: E731
    a = torch.load(ck(new_dir), map_location='cpu', weights_only=False)['model_state_dict']
    b = torch.load(ck(old_dir), map_location='cpu', weights_only=False)['model_state_dict']
    return {'compared': len(b), 'identical': set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in b)}


def static_checks():
    """(c)-(h): no training, no torch numerics (safe in the parent before forking)."""
    out = {}
    # (c) every main config differs from the stored B2 configuration only in the declared keys
    allowed = {'ppo.learning_rate', 'candidate', 'note', 'protocol_freeze_sha256'}
    diffs = set()
    for tag, arm in SETS.items():
        for b in dq_b2.BLOCKS:
            new = config_main(arm, b, CANDIDATES[arm][1], plan='f' * 64)
            old = dq_b2.config_for(arm, b, dq_b2.STUDY_B2, freeze_sha256='0' * 64)
            for k in set(new) | set(old):
                if k == 'ppo':
                    diffs |= {f'ppo.{q}' for q in set(new['ppo']) | set(old['ppo']) if new['ppo'].get(q) != old['ppo'].get(q)}
                elif new.get(k) != old.get(k):
                    diffs.add(k)
            p4.validate_config(new)
    out['c_main_configs'] = {'differing_keys': sorted(diffs), 'ok': diffs <= allowed and 'ppo.learning_rate' in diffs}
    # (d) pilot configs: development study, development namespace, no evaluation, B2's PPO settings except lr
    rt_dev = dq_b2.runtime(dq_b2.STUDY_SMOKE)
    dev = rt_dev.namespaces['dq_development']
    closed = [rt_dev.namespaces[n] for n in ('dq_e2_training', 'dq_e2_evaluation')]
    blocks = [b for i in range(3) for b in pilot_blocks(i)]
    seeds_ok = all(dev['start'] <= rt_dev.schedule(b, 0)[1] and rt_dev.schedule(b, p4.BLOCK_STRIDE - 1)[1] < dev['end']
                   and not any(n['start'] <= rt_dev.schedule(b, 0)[1] < n['end'] for n in closed) for b in blocks)
    pc = [config_pilot(arm, b, CANDIDATES[arm][0]) for arm in SETS.values() for b in blocks]
    out['d_pilot_configs'] = {
        'blocks': blocks, 'study': sorted({c['study'] for c in pc}), 'training_namespace': dq_b2.STUDY_SMOKE.training_namespace,
        'evaluation': sorted({str(c['evaluation']) for c in pc}), 'budget': sorted({c['total_policy_decisions'] for c in pc}),
        'seeds_in_dq_development_only': seeds_ok,
        'ppo_equal_to_b2_except_lr': all({k: v for k, v in c['ppo'].items() if k != 'learning_rate'} ==
                                         {k: v for k, v in dq_b2.PPO_BASE.items()} | {'entropy_coef': 0.0} for c in pc),
        'disjoint_from_b2_blocks_and_suite': dq_b2.STUDY_SMOKE.salt != dq_b2.STUDY_B2.salt}
    out['d_pilot_configs']['ok'] = (seeds_ok and out['d_pilot_configs']['study'] == ['dq_b2_smoke']
                                    and out['d_pilot_configs']['evaluation'] == ['None']
                                    and out['d_pilot_configs']['ppo_equal_to_b2_except_lr']
                                    and out['d_pilot_configs']['disjoint_from_b2_blocks_and_suite'])
    # (e)/(f) targets and slopes recomputed from the stored records equal the design constants
    st = stored_targets_and_slopes()
    out['e_targets'] = {'recomputed': st['targets'], 'declared': {a: t['value'] for a, t in TARGETS.items()},
                        'ok': all(st['targets'][a] == TARGETS[a]['value'] for a in TARGETS)}
    out['f_slopes'] = {'recomputed': st['slopes'], 'declared': SLOPES, 'stored_means': st['stored_means'],
                       'ok': all(abs(st['slopes'][a] - SLOPES[a]) < 1e-12 for a in SLOPES)}
    # (g) the twin build path and the studies are dq_b2's, in this process
    out['g_build_path'] = {'ok': p4.build_model is dq_b2.dq_build_model and p4.STUDIES['dq_b2'] is dq_b2.STUDY_B2
                           and p4.STUDIES['dq_b2_smoke'] is dq_b2.STUDY_SMOKE}
    # (h) the live B2 code tree is the recorded one
    tree = p4.live_copy_tree()
    out['h_code_tree'] = {'live': tree, 'ok': tree == B2_TREE}
    return out


def precheck():
    out = HERE / 'precheck'
    if out.exists():
        raise SystemExit('precheck/ exists')
    torch.set_num_threads(1)
    static = static_checks()
    out.mkdir()
    tasks = [{'kind': 'precheck', 'tag': 'PC', 'arm': arm, 'block': 0, 'lr': STORED_LR[arm], 'out': str(out),
              'budget': 3000, 'grid': [0, 3000]} for arm in ('quantum', 'surrogate625')]

    def write(rec):
        write_json_atomic(out / f'{rec["arm"]}.json', rec)
    pool_run(tasks, 2, write)
    assess_precheck(out, static)


def assess_precheck(out, static):
    """Compares the precheck runs in `out` with the stored B2 block-0 runs; writes precheck.json."""
    result = {'utc': p4.utc(), 'runner_sha256': RUNNER_SHA256, 'plan_sha256_at_precheck': PLAN_SHA256,
              'static': static}
    dyn, timing = {}, {}
    stored_twin = json.loads((B2_CAMPAIGN / 'records/surrogate625__b0.json').read_text())['twin_parity']
    for arm in ('quantum', 'surrogate625'):
        f = out / f'{arm}.json'
        rec = json.loads(f.read_text()) if f.exists() else {'status': 'missing'}
        if rec['status'] != 'completed':
            dyn[arm] = {'ok': False, 'status': rec['status'], 'error': rec.get('error')}
            continue
        if rec.get('runner_sha256') != RUNNER_SHA256:
            dyn[arm] = {'ok': False, 'status': 'run by another runner.py', 'runner_sha256': rec.get('runner_sha256')}
            continue
        new_dir = out / run_name('precheck', 'PC', arm, 0, STORED_LR[arm]) / 'run' / arm
        old_dir = B2_RUNS / f'{arm}__b0' / 'run' / arm
        checks = compare_events(new_dir / 'metrics.jsonl', old_dir / 'metrics.jsonl')
        checks['initial_checkpoint'] = compare_initial_checkpoint(new_dir, old_dir)
        exact = all(c['identical'] and c['compared'] > 0 for k, c in checks.items() if k != 'update') and \
            checks['update']['compared'] > 0
        upd = checks['update']['identical'] or (arm == 'surrogate625' and checks['update']['max_relative_difference'] <= 1e-12)
        entry = {'checks': checks, 'ok': exact and upd}
        if arm == 'surrogate625':
            p = rec.get('twin_parity', {})
            entry['twin_parity'] = p
            entry['twin_parity_equals_stored_b2_record'] = p == stored_twin
            entry['ok'] = entry['ok'] and p.get('P4', {}).get('max_abs_dlogprob', 1) <= 1e-9 and \
                not p.get('static', {}).get('static_failures', ['missing'])
        dyn[arm] = entry
        s = rec['summary']
        n_eval = sum(len(e['results']['sample']['episodes']) for e in _load(new_dir / 'metrics.jsonl')
                     if e['kind'] == 'evaluation')
        timing[arm] = {'evaluation_episodes': n_eval, 'policy_decisions': s['policy_decisions'], 'episodes': s['episodes'],
                       'rollout_wall_s_per_1000_decisions': 1000 * s['walls']['rollout_s'] / s['policy_decisions'],
                       'rollout_cpu_s_per_1000_decisions': 1000 * s['cpu']['rollout_s'] / s['policy_decisions'],
                       'update_wall_s_per_1000_decisions': 1000 * s['walls']['update_s'] / s['policy_decisions'],
                       'evaluation_wall_s_per_episode': s['walls'].get('evaluation_s', 0.) / n_eval,
                       'evaluation_cpu_s_per_episode': s['cpu']['evaluation_s'] / n_eval,
                       'run_wall_s': rec['wall_s'], 'load_average': s['load_average']}
    result['dynamic'] = dyn
    result['timing_current_load'] = timing
    result['passed'] = all(v['ok'] for v in static.values()) and all(v['ok'] for v in dyn.values())
    (HERE / 'precheck.json').write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))
    return result


# -- pilot ---------------------------------------------------------------------------------------------------------------

def pilot_tasks(stage, out):
    if stage == '1':
        table = {arm: list(CANDIDATES[arm]) for arm in SETS.values()}
    else:                                      # stage 1b: only for arms the stage-1 range guard flagged
        sel = json.loads((HERE / 'selection_stage1.json').read_text())
        table = {arm: [round3(sel['arms'][arm]['lr_selected'] * f) for f in (1 / STAGE_1B_SPACING, 1, STAGE_1B_SPACING)]
                 for arm in SETS.values() if sel['arms'][arm]['stage_1b_needed']}
    tasks = []
    for i in range(3):
        for b in pilot_blocks(i):
            for arm in ('quantum', 'surrogate625'):
                if arm in table:
                    tasks.append({'kind': 'pilot', 'tag': f'P{stage}', 'arm': arm, 'block': b, 'lr': table[arm][i],
                                  'out': str(out)})
    return tasks, table


def pilot(workers, stage):
    plan = PLAN if PLAN.exists() else HERE / 'PLAN_DRAFT.md'
    if not plan.exists():
        raise SystemExit('the selection rule must be written (PLAN_DRAFT.md) before the pilot')
    root = HERE / ('pilot' if stage == '1' else 'pilot_1b')
    runs, records = root / 'runs', root / 'records'
    runs.mkdir(parents=True, exist_ok=True); records.mkdir(exist_ok=True)
    utc = time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())
    shutil.copy2(plan, root / f'{plan.stem}_at_pilot_launch_{utc}.md')        # snapshot of the rule as launched
    tasks, table = pilot_tasks(stage, runs)
    by_name = {run_name('pilot', t['tag'], t['arm'], t['block'], t['lr']): t for t in tasks}
    names, moved = prepare(list(by_name), runs, records, root / 'interrupted' / utc)
    todo = [by_name[n] for n in names]
    with open(root / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': utc, 'stage': stage, 'workers': workers, 'runner_sha256': RUNNER_SHA256,
                            'candidates': table, 'to_run': len(todo), 'of': len(tasks), 'moved_to_interrupted': moved,
                            'b2_code_tree_live_copy': p4.live_copy_tree()}) + '\n')

    def write(rec):
        write_json_atomic(records / f'{run_name("pilot", rec["set"], rec["arm"], rec["block"], rec["learning_rate"])}.json',
                          rec)
    pool_run(todo, workers, write)


def select(kl_by_candidate, target, slope, candidates):
    """THE pre-declared rule. kl_by_candidate: {lr: [run statistic of each pilot block]} (completed runs only).
    Level from the pilot, slope fixed from the stored lr-KL points:  lr* = exp(xbar + (ln T - ybar) / s)."""
    lrs = [lr for lr in candidates if kl_by_candidate.get(lr)]
    if len(lrs) < 2:
        raise ValueError('fewer than two candidates with a completed pilot run')
    x = np.log(lrs)
    y = np.log([float(np.mean(kl_by_candidate[lr])) for lr in lrs])
    raw = float(math.exp(x.mean() + (math.log(target) - y.mean()) / slope))
    lo, hi = min(candidates) / RANGE_FACTOR, max(candidates) * RANGE_FACTOR
    ols = float(np.polyfit(x, y, 1)[0]) if len(lrs) >= 2 else None
    free = float(math.exp(x.mean() + (math.log(target) - y.mean()) / ols)) if ols and ols > 0 else None
    return {'candidates_used': lrs, 'mean_kl': {lr_str(lr): float(np.mean(kl_by_candidate[lr])) for lr in lrs},
            'level_a': float(np.mean(y - slope * x)), 'slope_fixed': slope, 'lr_selected_raw': raw,
            'lr_selected': round3(raw), 'lr_selected_str': lr_str(raw),
            'range_guard': {'lo': lo, 'hi': hi, 'inside': lo <= raw <= hi}, 'stage_1b_needed': not (lo <= raw <= hi),
            'descriptive_not_used': {'ols_slope_pilot': ols, 'lr_with_ols_slope': free,
                                     'predicted_kl_at_selected_lr': float(math.exp(np.mean(y - slope * x)) * round3(raw) ** slope)}}


def pilot_select(stage):
    root = HERE / ('pilot' if stage == '1' else 'pilot_1b')
    tasks, table = pilot_tasks(stage, root / 'runs')
    missing = [t for t in tasks if not (root / 'records' /
                                        f'{run_name("pilot", t["tag"], t["arm"], t["block"], t["lr"])}.json').exists()]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(tasks)} pilot runs have no record: the selection waits for all')
    out = {'utc': p4.utc(), 'stage': stage, 'runner_sha256': RUNNER_SHA256, 'statistic':
           'run mean of metrics.approx_kl over update records; arm value = mean over pilot blocks', 'arms': {}}
    for arm, cands in table.items():
        kl, per_block, failed = {}, {}, []
        for t in [t for t in tasks if t['arm'] == arm]:
            name = run_name('pilot', t['tag'], arm, t['block'], t['lr'])
            rec = json.loads((root / 'records' / f'{name}.json').read_text())
            if rec['status'] != 'completed':
                failed.append({'run': name, 'status': rec['status'], 'error': rec.get('error')})
                continue
            stat = kl_of(root / 'runs' / name / 'run' / arm / 'metrics.jsonl')    # update records only
            kl.setdefault(t['lr'], []).append(stat['mean'])
            per_block[name] = stat
        try:
            sel = select(kl, TARGETS[arm]['value'], SLOPES[arm], cands)
        except ValueError as exc:                            # rule 5: no selection, the arm's main set is not launched
            sel = {'error': str(exc), 'lr_selected': None, 'lr_selected_str': None, 'stage_1b_needed': False}
        sel.update(target=TARGETS[arm], candidates=[lr_str(c) for c in cands], per_run=per_block, failed_runs=failed)
        out['arms'][arm] = sel
    name = 'selection_stage1.json' if stage == '1' else 'selection_stage1b.json'
    (HERE / name).write_text(json.dumps(out, indent=1, default=float))
    # selection.json: the learning rate each main set uses (stage 1b replaces stage 1 only for the flagged arm)
    final = json.loads((HERE / 'selection.json').read_text()) if (HERE / 'selection.json').exists() else {'arms': {}}
    for arm, sel in out['arms'].items():
        final['arms'][arm] = {'lr': sel['lr_selected'], 'lr_str': sel['lr_selected_str'], 'stage': stage,
                              'stage_1b_needed': sel['stage_1b_needed'] if stage == '1' else False}
    final['ready'] = (set(final['arms']) == set(SETS.values()) and
                      all(v['lr'] is not None and not v['stage_1b_needed'] for v in final['arms'].values()))
    (HERE / 'selection.json').write_text(json.dumps(final, indent=1))
    print(json.dumps(out, indent=1, default=float))


# -- main launch ---------------------------------------------------------------------------------------------------------

def launch(workers):
    if PLAN_SHA256 is None or not (HERE / 'PLAN.sha256').exists() or \
            (HERE / 'PLAN.sha256').read_text().split()[0] != PLAN_SHA256:
        raise SystemExit('PLAN.md must exist and match PLAN.sha256 (written by the lead) before the launch')
    pre = json.loads((HERE / 'precheck.json').read_text())
    if not (pre.get('passed') and pre.get('runner_sha256') == RUNNER_SHA256):
        raise SystemExit('precheck not passed for this runner.py')
    sel = json.loads((HERE / 'selection.json').read_text())
    text = PLAN.read_text()
    if not sel.get('ready') or any(sel['arms'][arm]['lr_str'] not in text for arm in SETS.values()):
        raise SystemExit('selection.json not ready, or its learning rates are not written in PLAN.md')
    if any(float(sel['arms'][arm]['lr_str']) != sel['arms'][arm]['lr'] for arm in SETS.values()):
        raise SystemExit('selection.json: lr and lr_str disagree')
    if RUNNER_SHA256 not in text or sha256_file(ANALYSIS) not in text:     # review B3: the code is pinned by the plan
        raise SystemExit('PLAN.md does not record the sha256 of this runner.py and of the current b2k_analysis.py')
    tree = p4.live_copy_tree()
    if tree != B2_TREE:
        raise SystemExit(f'the B2 code tree changed: {tree}')
    runs, records = HERE / 'runs', HERE / 'records'
    runs.mkdir(exist_ok=True); records.mkdir(exist_ok=True)
    utc = time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())
    by_name = {}
    for b in dq_b2.BLOCKS:
        for tag in ('T1', 'T2'):                              # HQ (slower) first within each block
            arm = SETS[tag]
            by_name[run_name('main', tag, arm, b, None)] = {'kind': 'main', 'tag': tag, 'arm': arm, 'block': b,
                                                            'lr': sel['arms'][arm]['lr'], 'out': str(runs)}
    names, moved = prepare(list(by_name), runs, records, HERE / 'interrupted' / utc)
    todo = [by_name[n] for n in names]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': utc, 'plan_sha256': PLAN_SHA256, 'runner_sha256': RUNNER_SHA256, 'workers': workers,
                            'learning_rates': {a: sel['arms'][a]['lr'] for a in SETS.values()},
                            'to_run': len(todo), 'of': len(by_name), 'moved_to_interrupted': moved,
                            'b2_code_tree_live_copy': tree}) + '\n')

    def write(rec):
        write_json_atomic(records / f'{rec["set"]}_{rec["arm"]}__b{rec["block"]}.json', rec)
    pool_run(todo, workers, write)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    pp = sub.add_parser('pilot')
    pp.add_argument('--workers', type=int, default=2)
    pp.add_argument('--stage', choices=('1', '1b'), default='1')
    ps = sub.add_parser('pilot-select')
    ps.add_argument('--stage', choices=('1', '1b'), default='1')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=8)              # review M12: at most 8 while other campaigns run
    a = ap.parse_args()
    _lock = single_instance() if a.cmd in ('precheck', 'pilot', 'launch') else None   # held until this process exits
    if a.cmd == 'precheck':
        precheck()
    elif a.cmd == 'pilot':
        pilot(a.workers, a.stage)
    elif a.cmd == 'pilot-select':
        pilot_select(a.stage)
    else:
        launch(a.workers)
