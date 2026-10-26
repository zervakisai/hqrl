"""Post-hoc check E4 (see PLAN_DRAFT.md; PLAN.md once the lead has reviewed and hashed it): best-effort,
information-matched, re-tuned PPO for C-strong ('mlp') and C-matched ('bottleneck') at 300,000 decisions on B2 blocks
4-9, paired block by block with the stored LONG runs (idea_dq_posthoc_long). Imports the idea_dq tree unchanged; every
change lives in e4lib.py and is switched on per process (e4lib.configure).

    python -B runner.py precheck                # (a) pass-through path reproduces stored LONG block 4 (both heads),
                                                # (b) init: LONG's model bitwise, widened model's logits equal (<1e-12),
                                                # (c) E4 path, 3,000 decisions per head: completes, logs lr and entropy
    python -B runner.py pilot [--workers 2]     # development pilot: 2 heads x 2 entropies x 2 dev blocks, 100k each
    python -B runner.py pilot-select            # applies the selection rule fixed in the plan -> pilot/selection.json
    python -B runner.py launch [--workers 12]   # the 12 main runs; refuses without PLAN.md + PLAN.sha256, a passed
                                                # precheck of this exact code, the rule reference and the selection
    python -B runner.py status
Run with -B (no bytecode written into any tree). Every run is single-threaded (torch_threads 1), one process per run
(fork pool, maxtasksperchild=1), never overwritten, never resumed: an interrupted run is moved to interrupted/<utc>/
and restarted from scratch (identical seeds), a failure is recorded and never retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4lib as e4                                                                                # noqa: E402
import dq_b2                                                                                      # noqa: E402
import p4_train as p4                                                                             # noqa: E402
import torch                                                                                      # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


RUNNER_SHA256 = sha(__file__)
E4LIB_SHA256 = sha(HERE / 'e4lib.py')
PRECHECK_BLOCK, PRECHECK_BUDGET = 4, 3000
PRECHECK_ENTROPY = max(e4.ENTROPY_CANDIDATES)
STRIP = ('utc', 'wall_s', 'elapsed_wall_s')


def utc():
    return time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())


def plan_sha256():
    """(sha256 of PLAN.md, True iff PLAN.sha256 names exactly that hash). PLAN.md is written and hashed by the lead."""
    plan, record = HERE / 'PLAN.md', HERE / 'PLAN.sha256'
    if not plan.exists() or not record.exists():
        return None, False
    digest = sha(plan)
    return digest, record.read_text().split()[0] == digest


def code_identity():
    return {'runner_sha256': RUNNER_SHA256, 'e4lib_sha256': E4LIB_SHA256, 'idea_dq_tree': p4.live_copy_tree()}


# ------------------------------------------------------------------ one run ---------------------------------------
def task(kind, arm, block, entropy, info, schedule, out, freeze, budget=e4.BUDGET, grid=None, name=None):
    study = 'dq_b2' if kind in ('main', 'precheck_a', 'precheck_c') else 'e4_dev'
    return {'kind': kind, 'arm': arm, 'block': block, 'entropy': float(entropy), 'lr0': e4.LR0[arm], 'info': info,
            'schedule': schedule, 'study': study, 'budget': budget, 'grid': list(grid or e4.GRID), 'out': str(out),
            'freeze': freeze, 'name': name or f'{arm}__b{block}'}


def run_one(t):
    torch.set_num_threads(1)
    rec = {'kind': t['kind'], 'name': t['name'], 'arm': t['arm'], 'block': t['block'], 'status': 'started',
           'task': t, 'code': {'runner_sha256': RUNNER_SHA256, 'e4lib_sha256': E4LIB_SHA256}}
    t0 = time.perf_counter()
    try:
        tree = p4.live_copy_tree()
        rec['code']['idea_dq_tree'] = tree
        if tree != e4.EXPECTED_TREE:
            raise RuntimeError(f'idea_dq live tree {tree[:12]} is not {e4.EXPECTED_TREE[:12]}')
        rec['configure'] = e4.configure(t['info'], t['lr0'] if t['schedule'] else None,
                                        e4.LR_HORIZON if t['schedule'] else None)
        study = p4.STUDIES[t['study']]
        role = e4.ROLES[t['arm']]
        settings = (f'obs {p4.LAYOUT_V4_GNSS.obs_version} ({p4.LAYOUT_V4_GNSS.dim}); lr {t["lr0"]:g}'
                    + (f' linear to 0 over {e4.LR_HORIZON}' if t['schedule'] else ' constant')
                    + f'; entropy {t["entropy"]:g}')
        note = f'post-hoc E4 {t["kind"]} (code {RUNNER_SHA256[:8]}/{E4LIB_SHA256[:8]}): {role} block {t["block"]}; {settings}'
        cfg = e4.config(t['arm'], t['block'], study, t['budget'], t['grid'], t['entropy'], t['lr0'], t['freeze'], note)
        cfg['candidate'] = e4.candidate_label(t['entropy'], t['lr0'], t['info'], t['schedule'])
        rt = dq_b2.runtime(study)
        run_dir = Path(t['out']) / t['name']
        if run_dir.exists():
            raise RuntimeError(f'{run_dir} exists: no overwrite, no resume')
        lo, hi, space = e4.seed_range(study, rt, t['block'])
        if study is dq_b2.STUDY_B2:          # LONG's guard: [start + 40,000, start + 100,000), above every B2 seed
            ok = t['block'] >= 4 and space['start'] + 36_000 <= lo and hi <= space['end']
        else:                                # development: above the dev evaluation seeds (< start + 7,000)
            ok = space['start'] + 8_000 <= lo and hi <= space['end']
        if not ok:
            raise RuntimeError(f'training seeds [{lo}, {hi}) would overlap or leave {study.training_namespace}')
        rec.update(config=cfg, training_seed_range=[lo, hi], stride=p4.BLOCK_STRIDE)
        summary = p4.train_arm(t['arm'], cfg, rt, run_dir / 'run' / t['arm'])
        rec.update(status=summary['status'], summary=summary)
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def pool_run(tasks, workers, records):
    records.mkdir(parents=True, exist_ok=True)
    out = []
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_one, tasks):
            (records / f'{rec["name"]}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['kind'], rec['name'], rec['status'], round(rec['wall_s']), rec.get('error', ''), flush=True)
            out.append(rec)
    return out


def load_events(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def kind_of(events, kind):
    return [e for e in events if e['kind'] == kind]


def strip(e, drop_ev=False):
    out = {k: v for k, v in e.items() if k not in STRIP}
    if drop_ev and 'policy_stats' in out:
        out['policy_stats'] = {k: v for k, v in out['policy_stats'].items() if not k.startswith('ev_')}
    return out


def long_metrics(arm, block):
    return e4.LONG_DIR / 'runs' / f'{arm}__b{block}' / 'run' / arm / 'metrics.jsonl'


# ------------------------------------------------------------------ precheck --------------------------------------
def init_checks(arm, block=PRECHECK_BLOCK):
    """(b): LONG's model is rebuilt bitwise; the widened model equals it on every tensor but the zero columns; logits
    and values of the widened model equal the base model's to < 1e-12 with random extra inputs; the global torch
    generator is left where the base build leaves it."""
    cfg = dq_b2.config_for(arm, block, dq_b2.STUDY_B2, e4.BUDGET, e4.GRID, freeze_sha256=dq_b2.DEV_FREEZE_LABEL)
    e4.configure(False)
    base = p4.build_model(arm, cfg, block)
    rng_base = torch.get_rng_state()
    ck = next((long_metrics(arm, block).parent / 'checkpoints').glob('update00000_*.pt'))
    stored = torch.load(ck, map_location='cpu', weights_only=False)['model_state_dict']
    sb = base.state_dict()
    bitwise = set(stored) == set(sb) and all(torch.equal(stored[k], sb[k]) for k in stored)
    e4.configure(True, e4.LR0[arm], e4.LR_HORIZON)
    wide = p4.build_model(arm, cfg, block)
    rng_wide = torch.get_rng_state()
    sw = wide.state_dict()
    others = all(torch.equal(sw[k], sb[k]) for k in sb if k != 'encoder.0.weight')
    w = sw['encoder.0.weight']
    cols = torch.equal(w[:, :e4.BASE.dim], sb['encoder.0.weight']) and bool((w[:, e4.BASE.dim:] == 0).all())
    g = torch.Generator().manual_seed(20261025)
    x = torch.rand(4, 9, e4.BASE.dim, generator=g, dtype=torch.float64)
    extra = torch.rand(4, 9, e4.K, generator=g, dtype=torch.float64)
    with torch.no_grad():
        l0, v0, h0 = base(x)
        l1, v1, h1 = wide(torch.cat([x, extra], dim=-1))
    dl, dv, dh = (float((a - b).abs().max()) for a, b in ((l0, l1), (v0, v1), (h0, h1)))
    e4.restore_originals()
    return {'base_equals_stored_LONG_init_bitwise': bitwise, 'stored_checkpoint': str(ck),
            'wide_obs_dim': wide.config['obs_dim'], 'wide_other_tensors_equal': others,
            'wide_encoder_columns_ok': cols, 'max_abs_dlogit': dl, 'max_abs_dvalue': dv, 'max_abs_dstate': dh,
            'global_rng_after_build_equal': bool(torch.equal(rng_base, rng_wide)),
            'passed': bitwise and others and cols and max(dl, dv, dh) < 1e-12 and wide.config['obs_dim'] == e4.WIDE.dim
            and bool(torch.equal(rng_base, rng_wide))}


def check_a(arm, run_dir):
    """(a): with every change disabled, the runner's code path re-runs stored LONG block 4 for 3,000 decisions."""
    new = load_events(run_dir / 'run' / arm / 'metrics.jsonl')
    old = load_events(long_metrics(arm, PRECHECK_BLOCK))
    out = {}
    a, b = kind_of(new, 'evaluation'), kind_of(old, 'evaluation')
    out['evaluation_point0'] = {'compared': len(a), 'identical': len(a) == 1 and a[0]['points'] == [0] == b[0]['points']
                                and a[0]['results'] == b[0]['results']}
    a, b = kind_of(new, 'update'), kind_of(old, 'update')
    out['update'] = {'compared': len(a), 'identical': all(x['metrics'] == y['metrics'] and x['cumulative'] == y['cumulative']
                                                          for x, y in zip(a, b))}
    a, b = kind_of(new, 'episode'), kind_of(old, 'episode')
    out['episode'] = {'compared': len(a), 'identical': all(strip(x) == strip(y) for x, y in zip(a, b))}
    a, b = kind_of(new, 'arm_start')[0], kind_of(old, 'arm_start')[0]
    out['arm_start'] = {'identical': all(a[k] == b[k] for k in ('model', 'identity', 'seeds'))}
    end = kind_of(new, 'arm_end')
    out['status'] = end[0]['status'] if end else None
    out['passed'] = (out['status'] == 'completed' and out['arm_start']['identical'] and
                     all(out[k]['identical'] and out[k]['compared'] > 0 for k in ('evaluation_point0', 'update', 'episode')))
    return out


def check_c(arm, run_dir, seed_range):
    """(c): the E4 path for 3,000 decisions. Before the first update the policy is LONG's (point-0 evaluation and the
    episodes of update 1 identical to stored LONG); every update logs the scheduled learning rate and the entropy
    coefficient; the evidence features are populated; the final checkpoint carries the wide model and the last lr."""
    events = load_events(run_dir / 'run' / arm / 'metrics.jsonl')
    old = load_events(long_metrics(arm, PRECHECK_BLOCK))
    out = {}
    end = kind_of(events, 'arm_end')
    out['status'] = end[0]['status'] if end else None
    start = kind_of(events, 'arm_start')[0]
    out['obs_dim'] = start['model']['config']['obs_dim']
    out['obs_encoding_version'] = start['identity']['condition']['obs_encoding_version']
    ev = kind_of(events, 'evaluation')
    out['evaluations'] = [{'points': e['points'], 'episodes': len(e['results']['sample']['episodes']),
                           'weighted_J': e['results']['sample']['weighted_J'], 'wall_s': e['wall_s']} for e in ev]
    out['evaluation_point0_identical_to_LONG'] = bool(ev) and ev[0]['results'] == kind_of(old, 'evaluation')[0]['results']
    eps, old_eps = kind_of(events, 'episode'), kind_of(old, 'episode')
    first = [e for e in eps if e['update_index'] == 1]
    out['update1_episodes'] = len(first)
    out['update1_episodes_identical_to_LONG'] = bool(first) and all(
        strip(x, drop_ev=True) == strip(y) for x, y in zip(first, old_eps))
    out['schedule_paired_with_LONG'] = all((x['cell_id'], x['seed']) == (y['cell_id'], y['seed'])
                                           for x, y in zip(eps, old_eps))
    lo, hi = seed_range
    out['all_seeds_in_range'] = bool(eps) and all(lo <= e['seed'] < hi for e in eps)
    ups = kind_of(events, 'update')
    lr_rows, lr_ok = [], True
    for u in ups:
        m = u['metrics']
        before = u['cumulative']['policy_decisions'] - u['transitions']
        want = e4.scheduled_lr(e4.LR0[arm], e4.LR_HORIZON, before)
        lr_rows.append({'update': u['update_index'], 'decisions_before': before, 'learning_rate': m.get('learning_rate'),
                        'expected': want, 'entropy_coef': m.get('entropy_coef'), 'entropy': m['entropy'],
                        'approx_kl': m['approx_kl']})
        lr_ok = lr_ok and m.get('learning_rate') == want and m.get('decisions_before_update') == before and \
            m.get('entropy_coef') == PRECHECK_ENTROPY and math.isfinite(m['entropy']) and m['entropy'] > 0
    out['updates'] = lr_rows
    out['lr_and_entropy_logged_as_intended'] = bool(ups) and lr_ok
    stats = {k: sum(e['policy_stats'].get(k, 0) for e in eps) for k in e4.EV_STATS}
    stats['decisions'] = sum(e['policy_stats']['decisions'] for e in eps)
    out['evidence_stats'] = stats
    out['evidence_populated'] = stats['ev_viewpoint_feature_nonzero'] > 0 and stats['ev_feedback_errors'] == 0 and \
        stats['ev_encodes'] >= stats['decisions'] > 0
    final = [c for c in kind_of(events, 'checkpoint') if c['final']]
    ck_ok = False
    if final:
        payload = torch.load(run_dir / 'run' / arm / final[0]['path'], map_location='cpu', weights_only=False)
        lrs = {g['lr'] for g in payload['optimizer_state_dict']['param_groups']}
        w = payload['model_state_dict']['encoder.0.weight']
        out['final_checkpoint'] = {'obs_dim': payload['model_config']['obs_dim'], 'optimizer_lr': sorted(lrs),
                                   'entropy_coef_in_config': payload['train_config']['ppo']['entropy_coef'],
                                   'extra_columns_max_abs': float(w[:, e4.BASE.dim:].abs().max())}
        ck_ok = payload['model_config']['obs_dim'] == e4.WIDE.dim and lrs == {lr_rows[-1]['learning_rate']} and \
            payload['train_config']['ppo']['entropy_coef'] == PRECHECK_ENTROPY
    out['final_checkpoint_ok'] = ck_ok
    out['throughput'] = {
        'rollout_s_per_1000_decisions': 1000 * sum(u['wall']['rollout_s'] for u in ups) / max(1, sum(u['transitions'] for u in ups)),
        'update_s_per_1000_decisions': 1000 * sum(u['wall']['update_s'] for u in ups) / max(1, sum(u['transitions'] for u in ups)),
        'evaluation_s_per_episode': [e['wall_s'] / max(1, len(e['results']['sample']['episodes'])) for e in ev],
        'training_episode_wall_s_mean': sum(e['wall_s'] for e in eps) / max(1, len(eps))}
    out['passed'] = (out['status'] == 'completed' and out['obs_dim'] == e4.WIDE.dim and
                     out['obs_encoding_version'] == e4.OBS_VERSION and out['evaluation_point0_identical_to_LONG'] and
                     out['update1_episodes_identical_to_LONG'] and out['schedule_paired_with_LONG'] and
                     out['all_seeds_in_range'] and out['lr_and_entropy_logged_as_intended'] and
                     out['evidence_populated'] and ck_ok and len(ev) == 2 and all(e['episodes'] == 80 for e in out['evaluations']))
    return out


def precheck(workers=2):
    out = HERE / 'precheck'
    if out.exists():
        raise SystemExit('precheck/ exists (never overwritten); move it aside to re-run')
    out.mkdir()
    t0 = time.perf_counter()
    result = {'utc': utc(), **code_identity(),
              'plan_draft_sha256': sha(HERE / 'PLAN_DRAFT.md') if (HERE / 'PLAN_DRAFT.md').exists() else None}
    if result['idea_dq_tree'] != e4.EXPECTED_TREE:
        raise SystemExit('idea_dq live tree moved')
    torch.set_num_threads(1)
    result['b_init'] = {arm: init_checks(arm) for arm in e4.ARMS}
    print(json.dumps(result['b_init'], indent=1), flush=True)
    tasks = []
    for arm in e4.ARMS:
        tasks.append(task('precheck_a', arm, PRECHECK_BLOCK, 0.0, False, False, out / 'runs', dq_b2.DEV_FREEZE_LABEL,
                          budget=PRECHECK_BUDGET, grid=[0], name=f'a_{arm}__b{PRECHECK_BLOCK}'))
        tasks.append(task('precheck_c', arm, PRECHECK_BLOCK, PRECHECK_ENTROPY, True, True, out / 'runs',
                          dq_b2.DEV_FREEZE_LABEL, budget=PRECHECK_BUDGET, grid=[0, PRECHECK_BUDGET],
                          name=f'c_{arm}__b{PRECHECK_BLOCK}'))
    recs = {r['name']: r for r in pool_run(tasks, workers, out / 'records')}
    result['a_reproduction'], result['c_e4_path'] = {}, {}
    for arm in e4.ARMS:
        ra, rc = recs[f'a_{arm}__b{PRECHECK_BLOCK}'], recs[f'c_{arm}__b{PRECHECK_BLOCK}']
        result['a_reproduction'][arm] = (check_a(arm, out / 'runs' / ra['name']) if ra['status'] == 'completed'
                                         else {'passed': False, 'status': ra['status'], 'error': ra.get('error')})
        result['a_reproduction'][arm]['wall_s'] = ra['wall_s']
        result['c_e4_path'][arm] = (check_c(arm, out / 'runs' / rc['name'], rc['training_seed_range'])
                                    if rc['status'] == 'completed'
                                    else {'passed': False, 'status': rc['status'], 'error': rc.get('error')})
        result['c_e4_path'][arm]['wall_s'] = rc['wall_s']
    result['idea_dq_tree_after'] = p4.live_copy_tree()
    result['passed'] = (all(v['passed'] for v in result['b_init'].values()) and
                        all(v['passed'] for v in result['a_reproduction'].values()) and
                        all(v['passed'] for v in result['c_e4_path'].values()) and
                        result['idea_dq_tree_after'] == e4.EXPECTED_TREE)
    result['wall_s'] = time.perf_counter() - t0
    (HERE / 'precheck.json').write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({'passed': result['passed'],
                      'a': {k: v['passed'] for k, v in result['a_reproduction'].items()},
                      'b': {k: v['passed'] for k, v in result['b_init'].items()},
                      'c': {k: v['passed'] for k, v in result['c_e4_path'].items()}}, indent=1), flush=True)


def require_precheck():
    path = HERE / 'precheck.json'
    if not path.exists():
        raise SystemExit('no precheck.json')
    pre = json.loads(path.read_text())
    if not (pre.get('passed') and pre.get('runner_sha256') == RUNNER_SHA256 and pre.get('e4lib_sha256') == E4LIB_SHA256):
        raise SystemExit('the precheck has not passed for this exact runner.py and e4lib.py')
    return pre


# ------------------------------------------------------------------ resumable launcher ----------------------------
def resumable(tasks, root, workers, label):
    runs, records = root / 'runs', root / 'records'
    runs.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    stamp = utc()
    todo = []
    for t in tasks:
        if (records / f'{t["name"]}.json').exists():
            continue
        if (runs / t['name']).exists():                       # killed by a crash: moved aside, restarted from scratch
            (root / 'interrupted' / stamp).mkdir(parents=True, exist_ok=True)
            shutil.move(str(runs / t['name']), str(root / 'interrupted' / stamp / t['name']))
        todo.append(dict(t, out=str(runs)))
    print(f'{label}: {len(todo)} of {len(tasks)} runs to do, {workers} workers', flush=True)
    return pool_run(todo, workers, records) if todo else []


def pilot_tasks():
    return [task('pilot', arm, block, ent, True, True, HERE / 'pilot' / 'runs', dq_b2.DEV_FREEZE_LABEL,
                 budget=e4.PILOT_BUDGET, grid=e4.PILOT_GRID, name=f'{arm}__ent{ent:g}__b{block}')
            for block in e4.DEV_BLOCKS for arm in e4.ARMS for ent in e4.ENTROPY_CANDIDATES]


def pilot(workers):
    require_precheck()
    root = HERE / 'pilot'
    root.mkdir(exist_ok=True)
    stamp = utc()
    draft = HERE / 'PLAN_DRAFT.md'
    snapshot = root / f'PLAN_DRAFT_at_pilot_launch_{stamp}.md'
    shutil.copyfile(draft, snapshot)
    with open(root / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': stamp, 'workers': workers, **code_identity(), 'plan_draft_snapshot': snapshot.name,
                            'plan_draft_snapshot_sha256': sha(snapshot),
                            'note': 'provenance of the draft the pilot ran under; not the plan freeze'}) + '\n')
    resumable(pilot_tasks(), root, workers, 'pilot')


def pilot_summary(path):
    events = load_events(path)
    end = kind_of(events, 'arm_end')
    status = end[0]['status'] if end else None
    J = {}
    for e in kind_of(events, 'evaluation'):
        for p in e['points']:
            J[p] = e['results']['sample']['weighted_J']
    ups = kind_of(events, 'update')
    return {'status': status, 'J': J, 'last_update_entropy': ups[-1]['metrics']['entropy'] if ups else None,
            'last_update_lr': ups[-1]['metrics'].get('learning_rate') if ups else None,
            'policy_decisions': ups[-1]['cumulative']['policy_decisions'] if ups else 0}


def select_entropy(by_value):
    """THE selection rule of the plan, per head. by_value: {entropy: {block: summary}}. Returns (choice, details)."""
    details = {}
    for ent, blocks in by_value.items():
        completed = all(s['status'] == 'completed' for s in blocks.values())
        S = (sum(s['J'][p] for s in blocks.values() for p in e4.SELECTION_POINTS) /
             (len(blocks) * len(e4.SELECTION_POINTS))) if completed else None
        collapsed = completed and any(s['last_update_entropy'] < e4.COLLAPSE_ENTROPY_NATS for s in blocks.values())
        details[ent] = {'completed_all_blocks': completed, 'S': S, 'collapse_guard_hit': collapsed,
                        'eligible': completed and not collapsed}
    small, large = min(details), max(details)
    eligible = [e for e, d in details.items() if d['eligible']]
    if not eligible:                                    # guardrail lifted when no value passes it; failures still out
        eligible = [e for e, d in details.items() if d['completed_all_blocks']]
        rule = 'collapse guard lifted (no value passed it)' if eligible else 'every value failed: 0.01'
        if not eligible:
            return large, {'values': details, 'rule_applied': rule}
    else:
        rule = 'eligible values only'
    if len(eligible) == 1:
        return eligible[0], {'values': details, 'rule_applied': rule + '; one eligible value'}
    d = details[large]['S'] - details[small]['S']
    if abs(d) < e4.SELECTION_TIE_J:
        return small, {'values': details, 'rule_applied': rule + f'; |S diff| {abs(d):.4f} < {e4.SELECTION_TIE_J}: smaller'}
    return (large if d > 0 else small), {'values': details, 'rule_applied': rule + f'; larger S (diff {d:+.4f})'}


def pilot_select():
    out = HERE / 'pilot' / 'selection.json'
    if out.exists():
        raise SystemExit(f'{out} exists; never overwritten')
    recs = {}
    for t in pilot_tasks():
        rec_path = HERE / 'pilot' / 'records' / f'{t["name"]}.json'
        if not rec_path.exists():
            raise SystemExit(f'pilot incomplete: no record for {t["name"]}')
        rec = json.loads(rec_path.read_text())
        if rec['status'] not in ('completed', 'algorithmic_failure'):
            raise SystemExit(f'{t["name"]}: status {rec["status"]} (infrastructure); re-run it before selecting')
        recs[t['name']] = (t, pilot_summary(Path(t['out']) / t['name'] / 'run' / t['arm'] / 'metrics.jsonl'))
    selection = {'utc': utc(), **code_identity(), 'rule': {
        'score': f'mean weighted development J at {list(e4.SELECTION_POINTS)} over dev blocks {list(e4.DEV_BLOCKS)}',
        'eligible': f'completed in every dev block and last-update policy entropy >= {e4.COLLAPSE_ENTROPY_NATS} nats '
                    'in every dev block (lifted if no value passes; failures never eligible)',
        'tie': f'|S(0.01) - S(0.003)| < {e4.SELECTION_TIE_J} J -> the smaller entropy'}, 'heads': {}}
    for arm in e4.ARMS:
        by_value = {ent: {t['block']: s for t, s in recs.values() if t['arm'] == arm and t['entropy'] == ent}
                    for ent in e4.ENTROPY_CANDIDATES}
        choice, details = select_entropy(by_value)
        selection['heads'][arm] = {'role': e4.ROLES[arm], 'entropy_coef': choice, 'lr0': e4.LR0[arm], **details,
                                   'runs': {t['name']: s for t, s in recs.values() if t['arm'] == arm}}
    out.write_text(json.dumps(selection, indent=1, default=float))
    print(json.dumps({a: {'entropy_coef': v['entropy_coef'], 'rule_applied': v['rule_applied'],
                          'S': {e: d['S'] for e, d in v['values'].items()}} for a, v in selection['heads'].items()},
                     indent=1, default=float))


# ------------------------------------------------------------------ main runs -------------------------------------
def launch(workers):
    digest, ok = plan_sha256()
    if not ok:
        raise SystemExit('PLAN.md and a matching PLAN.sha256 (written by the lead) are required before any main run')
    pre = require_precheck()
    ref = json.loads((HERE / 'rule_reference.json').read_text())
    if not ref['checks']['passed']:
        raise SystemExit('the rule reference did not pass its checks')
    sel_path = HERE / 'pilot' / 'selection.json'
    if not sel_path.exists():
        raise SystemExit('no pilot/selection.json: run the pilot and pilot-select first')
    sel = json.loads(sel_path.read_text())
    entropy = {arm: float(sel['heads'][arm]['entropy_coef']) for arm in e4.ARMS}
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': utc(), 'plan_sha256': digest, 'workers': workers, **code_identity(),
                            'precheck_utc': pre['utc'], 'entropy_coef': entropy,
                            'selection_sha256': sha(sel_path), 'rule_reference_sha256': sha(HERE / 'rule_reference.json')})
                + '\n')
    tasks = [task('main', arm, block, entropy[arm], True, True, HERE / 'runs', digest)
             for block in e4.BLOCKS for arm in e4.ARMS]
    resumable(tasks, HERE, workers, 'main')


def status():
    for label, root, n in (('pilot', HERE / 'pilot', len(pilot_tasks())), ('main', HERE, 12)):
        recs = sorted((root / 'records').glob('*.json')) if (root / 'records').exists() else []
        alive = sorted(p.name for p in (root / 'runs').glob('*')) if (root / 'runs').exists() else []
        print(f'{label}: {len(recs)}/{n} records; run dirs {len(alive)}')
        for p in alive:
            mets = list((root / 'runs' / p / 'run').glob('*/metrics.jsonl'))
            if mets:
                last = None
                for line in mets[0].read_text().splitlines():
                    if '"kind": "update"' in line:
                        last = line
                if last:
                    u = json.loads(last)
                    print(f'  {p}: update {u["update_index"]}, decisions {u["cumulative"]["policy_decisions"]}, '
                          f'lr {u["metrics"].get("learning_rate")}, entropy {u["metrics"]["entropy"]:.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    pc = sub.add_parser('precheck')
    pc.add_argument('--workers', type=int, default=2)
    pi = sub.add_parser('pilot')
    pi.add_argument('--workers', type=int, default=2)
    sub.add_parser('pilot-select')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=12)
    sub.add_parser('status')
    a = ap.parse_args()
    {'precheck': lambda: precheck(a.workers), 'pilot': lambda: pilot(a.workers), 'pilot-select': pilot_select,
     'launch': lambda: launch(a.workers), 'status': status}[a.cmd]()
