"""Post-hoc long-training runs of HQ (see PLAN.md), the companion of idea_dq_posthoc_long.
Imports the B2 code tree of idea_dq unchanged; in its own process it (i) adds C-strong's E1 configuration to the B2 arm
table and (ii) widens the training-schedule stride from 1,000 to 10,000 seeds per block (PLAN.md).

    python runner.py precheck              # C-strong block 4, 3,000 decisions: seeds in the new range, run completes
    python runner.py launch --workers 6    # the 6 runs, written under runs/ and records/
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
BLOCKS = (4, 5, 6, 7, 8, 9)
BUDGET = 300_000
GRID = list(range(0, BUDGET + 1, 30_000))
STRIDE = 10_000
ARMS = ('quantum',)
#: C-strong's E1 selection (E1 report: lr3e-3_ent0), added to the B2 arm table in this process only
dq_b2.ARMS.setdefault('mlp', {'learning_rate': 3e-3, 'entropy_coef': 0.0, 'role': 'C-strong', 'candidate': 'lr3e-3_ent0'})
p4.BLOCK_STRIDE = STRIDE


def config(arm, block, budget=BUDGET, grid=None):
    c = dq_b2.config_for(arm, block, dq_b2.STUDY_B2, budget, grid or GRID, freeze_sha256=PLAN_SHA256)
    c['note'] = f'post-hoc long training (PLAN.md {PLAN_SHA256[:8]}): {dq_b2.ARMS[arm]["role"]} block {block}'
    return c


def run_one(task):
    arm, block, out, budget, grid = task
    torch.set_num_threads(1)
    rec = {'arm': arm, 'block': block, 'status': 'started', 'plan_sha256': PLAN_SHA256, 'stride': p4.BLOCK_STRIDE}
    t0 = time.perf_counter()
    try:
        cfg = config(arm, block, budget, grid)
        rt = dq_b2.runtime(dq_b2.STUDY_B2)
        run_dir = Path(out) / f'{arm}__b{block}'
        if run_dir.exists():
            raise RuntimeError(f'{run_dir} exists: no overwrite, no resume')
        rec['config'] = cfg
        space = rt.namespaces[dq_b2.STUDY_B2.training_namespace]
        lo, hi = space['start'] + block * STRIDE, space['start'] + (block + 1) * STRIDE
        if not (block >= 4 and space['start'] + 36_000 <= lo and hi <= space['end']):
            raise RuntimeError('training seeds would overlap B2 or leave the namespace')
        rec['training_seed_range'] = [lo, hi]
        rec['summary'] = p4.train_arm(arm, cfg, rt, run_dir / 'run' / arm)
        rec['status'] = rec['summary']['status']
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def precheck():
    out = HERE / 'precheck'
    if out.exists():
        raise SystemExit('precheck/ exists')
    rec = run_one(('mlp', 4, str(out), 3000, [0, 3000]))
    seeds = []
    f = out / 'mlp__b4' / 'run' / 'mlp' / 'metrics.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            e = json.loads(line)
            if e.get('kind') == 'episode' and 'seed' in e:
                seeds.append(e['seed'])
    lo, hi = rec.get('training_seed_range', [None, None])
    result = {'plan_sha256': PLAN_SHA256, 'status': rec['status'], 'error': rec.get('error'),
              'training_seed_range': [lo, hi], 'episode_seeds_seen': len(seeds),
              'all_seeds_in_range': bool(seeds) and all(lo <= s < hi for s in seeds),
              'first_seeds': seeds[:3], 'wall_s': rec['wall_s']}
    (HERE / 'precheck.json').write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


def launch(workers):
    runs, records = HERE / 'runs', HERE / 'records'
    runs.mkdir(exist_ok=True)
    records.mkdir(exist_ok=True)
    utc = time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': utc, 'plan_sha256': PLAN_SHA256, 'workers': workers,
                            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'b2_code_tree_live_copy': p4.live_copy_tree()}) + '\n')
    tasks = []
    for block in BLOCKS:
        for arm in ARMS:
            name = f'{arm}__b{block}'
            if (records / f'{name}.json').exists():
                continue
            if (runs / name).exists():                      # killed by a crash: moved aside, restarted from scratch
                import shutil
                (HERE / 'interrupted' / utc).mkdir(parents=True, exist_ok=True)
                shutil.move(str(runs / name), str(HERE / 'interrupted' / utc / name))
            tasks.append((arm, block, str(runs), BUDGET, GRID))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_one, tasks):
            (records / f'{rec["arm"]}__b{rec["block"]}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['arm'], rec['block'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers)
