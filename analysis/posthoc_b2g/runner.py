"""Post-hoc B2 runs of HQ from generic circuit angles (see PLAN.md). Imports the B2 code tree of idea_dq unchanged.

    python runner.py precheck                 # (a) code path reproduces stored HQ block 0 for 3,000 decisions,
                                              # (b) only actor.weights differ from the stored initial checkpoint
    python runner.py launch --workers 12      # the 36 runs of PLAN.md, written under runs/ and records/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
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

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
B2_RUNS = IDEA_DQ / 'output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs'
TAG, ARM = 'G', 'quantum'

#: block whose angles are replaced in this process (None: replacement disabled)
ANGLES = {'block': None}
_build_model = p4.build_model


def angle_seed(block):
    return int(np.random.SeedSequence([20261024, int(block)]).generate_state(1)[0])


def build_model_generic(arm, config, block):
    """p4_train.build_model, then the 16 circuit angles from U(-pi, pi) drawn from a separate generator (PLAN.md)."""
    model = _build_model(arm, config, block)
    if arm == ARM and ANGLES['block'] is not None:
        assert ANGLES['block'] == block, (ANGLES['block'], block)
        w = model.actor.weights
        g = torch.Generator().manual_seed(angle_seed(block))
        with torch.no_grad():
            w.copy_((torch.rand(w.shape, generator=g, dtype=torch.float64) * 2 - 1).to(w.dtype) * math.pi)
    return model


p4.build_model = build_model_generic


def config(block, budget=dq_b2.BUDGET, grid=None):
    c = dq_b2.config_for(ARM, block, dq_b2.STUDY_B2, budget, grid or dq_b2.GRID, freeze_sha256=PLAN_SHA256)
    c['note'] = f'post-hoc HQ from generic circuit angles (PLAN.md {PLAN_SHA256[:8]}): block {block}'
    return c


def run_one(task):
    block, out = task
    torch.set_num_threads(1)
    ANGLES['block'] = block
    rec = {'set': TAG, 'arm': ARM, 'block': block, 'angle_seed': angle_seed(block), 'status': 'started',
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
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def precheck():
    out = HERE / 'precheck'
    if out.exists():
        raise SystemExit('precheck/ exists')
    torch.set_num_threads(1)
    result = {'plan_sha256': PLAN_SHA256}
    # (b) the replacement changes only actor.weights
    ANGLES['block'] = 0
    cfg = config(0)
    built = build_model_generic(ARM, cfg, 0).state_dict()
    ck = next((B2_RUNS / 'quantum__b0/run/quantum/checkpoints').glob('update00000_*.pt'))
    stored = torch.load(ck, map_location='cpu', weights_only=False)['model_state_dict']
    same = {k: bool(torch.equal(built[k], stored[k])) for k in stored}
    w = built['actor.weights']
    result['b_init'] = {'parameters': len(stored), 'differ': sorted(k for k, v in same.items() if not v),
                        'angles_within_pi': bool((w.abs() < math.pi).all()), 'angles_mean_abs': float(w.abs().mean()),
                        'stored_angles_mean_abs': float(stored['actor.weights'].abs().mean())}
    b_ok = result['b_init']['differ'] == ['actor.weights'] and result['b_init']['angles_within_pi']
    # (a) with the replacement disabled, the code path reproduces the stored HQ block 0 for 3,000 decisions
    ANGLES['block'] = None
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    cfg = dq_b2.config_for(ARM, 0, dq_b2.STUDY_B2, 3000, [0, 3000], freeze_sha256=PLAN_SHA256)
    p4.train_arm(ARM, cfg, rt, out / 'run' / ARM)
    load = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines()]          # noqa: E731
    new = load(out / f'run/{ARM}/metrics.jsonl')
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
    result['passed'] = b_ok and all(c['identical'] and c['compared'] > 0 for c in checks.values())
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
