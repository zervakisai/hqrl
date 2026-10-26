"""Post-hoc B2 runs of HQ with frozen circuit angles (see PLAN.md). Imports the B2 code tree of idea_dq unchanged.

    python runner.py precheck                 # (a) code path reproduces stored HQ block 0 for 3,000 decisions,
                                              # (b) built model equals the stored initial checkpoint bitwise,
                                              # (c) a frozen 3,000-decision run keeps the angles bit-identical
    python runner.py launch --workers 10      # the 36 runs of PLAN.md, written under runs/ and records/
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))

import torch                                                                                      # noqa: E402
import dq_b2                                                                                      # noqa: E402
import p4_train as p4                                                                             # noqa: E402

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
B2_RUNS = IDEA_DQ / 'output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs'
TAG, ARM = 'F', 'quantum'

#: freezing of the 16 circuit angles in this process
FREEZE = {'on': True}
_build_model = p4.build_model


def build_model_frozen(arm, config, block):
    """p4_train.build_model, then actor.weights (the 16 circuit angles) excluded from training (PLAN.md)."""
    model = _build_model(arm, config, block)
    if arm == ARM and FREEZE['on']:
        model.actor.weights.requires_grad_(False)
    return model


p4.build_model = build_model_frozen


def config(block, budget=dq_b2.BUDGET, grid=None):
    c = dq_b2.config_for(ARM, block, dq_b2.STUDY_B2, budget, grid or dq_b2.GRID, freeze_sha256=PLAN_SHA256)
    c['note'] = f'post-hoc HQ with frozen circuit angles (PLAN.md {PLAN_SHA256[:8]}): block {block}'
    return c


def run_one(task):
    block, out = task
    torch.set_num_threads(1)
    FREEZE['on'] = True
    rec = {'set': TAG, 'arm': ARM, 'block': block, 'frozen_angles': True, 'status': 'started',
           'plan_sha256': PLAN_SHA256}
    t0 = time.perf_counter()
    try:
        cfg = config(block)
        rt = dq_b2.runtime(dq_b2.STUDY_B2)
        run_dir = Path(out) / f'{TAG}_{ARM}__b{block}'
        if run_dir.exists():
            raise RuntimeError(f'{run_dir} exists: no overwrite, no resume')
        rec['config'] = cfg
        summary = p4.train_arm(ARM, cfg, rt, run_dir / 'run' / ARM)
        rec.update(status=summary['status'], summary=summary)
        rec['angles'] = angle_check(run_dir / 'run' / ARM)
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def angle_check(run_dir):
    cks = sorted((Path(run_dir) / 'checkpoints').glob('*.pt'))
    ws = [torch.load(c, map_location='cpu', weights_only=False)['model_state_dict']['actor.weights'] for c in cks]
    return {'checkpoints': len(ws), 'all_equal_initial': all(torch.equal(w, ws[0]) for w in ws),
            'max_abs_change': float(max((w - ws[0]).abs().max() for w in ws))}


def precheck():
    out = HERE / 'precheck'
    if out.exists():
        raise SystemExit('precheck/ exists')
    torch.set_num_threads(1)
    result = {'plan_sha256': PLAN_SHA256}
    # (b) the frozen build equals the stored initial checkpoint bitwise
    FREEZE['on'] = True
    built_model = build_model_frozen(ARM, config(0), 0)
    built = built_model.state_dict()
    ck = next((B2_RUNS / 'quantum__b0/run/quantum/checkpoints').glob('update00000_*.pt'))
    stored = torch.load(ck, map_location='cpu', weights_only=False)['model_state_dict']
    result['b_init'] = {'parameters': len(stored), 'differ': sorted(k for k in stored if not torch.equal(built[k], stored[k])),
                        'angles_requires_grad': bool(built_model.actor.weights.requires_grad),
                        'trainable_actor_parameters': int(sum(p.numel() for p in built_model.actor.parameters() if p.requires_grad))}
    b_ok = result['b_init']['differ'] == [] and not result['b_init']['angles_requires_grad']
    # (a) with freezing disabled, the code path reproduces the stored HQ block 0 for 3,000 decisions
    FREEZE['on'] = False
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    cfg = dq_b2.config_for(ARM, 0, dq_b2.STUDY_B2, 3000, [0, 3000], freeze_sha256=PLAN_SHA256)
    p4.train_arm(ARM, cfg, rt, out / 'unfrozen' / ARM)
    load = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines()]          # noqa: E731
    new = load(out / f'unfrozen/{ARM}/metrics.jsonl')
    old = load(B2_RUNS / f'quantum__b0/run/{ARM}/metrics.jsonl')
    pick = lambda ev, kind: [e for e in ev if e['kind'] == kind]                           # noqa: E731
    checks = {}
    for kind, key in (('evaluation', 'results'), ('update', 'metrics')):
        a, b = pick(new, kind), pick(old, kind)
        checks[kind] = {'compared': len(a), 'identical': all(x[key] == y[key] for x, y in zip(a, b[:len(a)]))}
    a, b = pick(new, 'episode'), pick(old, 'episode')
    strip = lambda e: {k: v for k, v in e.items() if k not in ('utc', 'wall_s', 'elapsed_wall_s')}      # noqa: E731
    checks['episode'] = {'compared': len(a), 'identical': all(strip(x) == strip(y) for x, y in zip(a, b[:len(a)]))}
    result['a_reproduction'] = checks
    a_ok = all(c['identical'] and c['compared'] > 0 for c in checks.values())
    # (c) a frozen 3,000-decision run keeps the angles bit-identical; its point-0 evaluation equals the stored one
    FREEZE['on'] = True
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    p4.train_arm(ARM, cfg, rt, out / 'frozen' / ARM)
    result['c_frozen'] = angle_check(out / 'frozen' / ARM)
    fz = load(out / f'frozen/{ARM}/metrics.jsonl')
    e0n, e0o = pick(fz, 'evaluation')[0], pick(old, 'evaluation')[0]
    result['c_frozen']['point0_evaluation_identical'] = e0n['results'] == e0o['results']
    starts = [e for e in fz if e['kind'] == 'arm_start']
    result['c_frozen']['arm_start_parameter_counts'] = starts[0].get('model', {}).get('parameter_counts') if starts else None
    c_ok = result['c_frozen']['all_equal_initial'] and result['c_frozen']['checkpoints'] >= 2 \
        and result['c_frozen']['point0_evaluation_identical']
    result['passed'] = b_ok and a_ok and c_ok
    (HERE / 'precheck.json').write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


def launch(workers):
    pre = json.loads((HERE / 'precheck.json').read_text())
    if not (pre.get('passed') and pre.get('plan_sha256') == PLAN_SHA256):
        raise SystemExit('precheck not passed for this plan')
    runs, records = HERE / 'runs', HERE / 'records'
    runs.mkdir(exist_ok=True); records.mkdir(exist_ok=True)
    utc = time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': utc, 'plan_sha256': PLAN_SHA256, 'workers': workers,
                            'b2_code_tree_live_copy': p4.live_copy_tree()}) + '\n')
    task_list = []
    for b in dq_b2.BLOCKS:
        name = f'{TAG}_{ARM}__b{b}'
        if (records / f'{name}.json').exists():
            continue
        if (runs / name).exists():                           # killed by a crash: moved aside, restarted from scratch
            import shutil
            (HERE / 'interrupted' / utc).mkdir(parents=True, exist_ok=True)
            shutil.move(str(runs / name), str(HERE / 'interrupted' / utc / name))
        task_list.append((b, str(runs)))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_one, task_list):
            (records / f'{rec["set"]}_{rec["arm"]}__b{rec["block"]}.json').write_text(
                json.dumps(rec, indent=1, default=float))
            print(rec['set'], rec['arm'], rec['block'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers)
