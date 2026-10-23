"""Post-hoc matched-learning-rate runs for B2 (see PLAN.md). Imports the B2 code tree of idea_dq unchanged.

    python runner.py precheck                 # determinism pre-check: B2 twin (1e-3), block 0, first 3,000 decisions
    python runner.py launch --workers 12      # the 72 runs of PLAN.md, written under runs/ and records/
"""
from __future__ import annotations

import argparse
import copy
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
#: (arm, learning rate) of the two new run sets; the candidate names exist in dq_b2.CANDIDATES
NEW = {'L1': ('surrogate625', 3e-3, 'lr3e-3_ent0'), 'L2': ('quantum', 1e-3, 'lr1e-3_ent0')}


def config(arm, block, lr, candidate, budget=dq_b2.BUDGET, grid=None):
    c = dq_b2.config_for(arm, block, dq_b2.STUDY_B2, budget, grid or dq_b2.GRID, freeze_sha256=PLAN_SHA256)
    c['ppo'] = dict(c['ppo'], learning_rate=lr)
    c['candidate'] = candidate
    c['note'] = f'post-hoc matched-LR check (PLAN.md {PLAN_SHA256[:8]}): {dq_b2.ARMS[arm]["role"]} lr {lr} block {block}'
    return c


def run_one(task):
    tag, arm, block, lr, candidate, out = task
    torch.set_num_threads(1)
    rec = {'set': tag, 'arm': arm, 'block': block, 'learning_rate': lr, 'status': 'started', 'plan_sha256': PLAN_SHA256}
    t0 = time.perf_counter()
    try:
        cfg = config(arm, block, lr, candidate)
        rt = dq_b2.runtime(dq_b2.STUDY_B2)
        run_dir = Path(out) / f'{tag}_{arm}__b{block}'
        if run_dir.exists():
            raise RuntimeError(f'{run_dir} exists: no overwrite, no resume')
        rec['config'] = cfg
        if arm == 'surrogate625':
            rec['twin_parity'] = {'P4': dq_b2.twin_trajectory_parity(dq_b2.STUDY_B2, rt, cfg, block)}
        summary = p4.train_arm(arm, cfg, rt, run_dir / 'run' / arm)
        if arm == 'surrogate625':
            rec['twin_parity'].update(dq_b2.TWIN_RECORDS[('dq_b2', block)])
        rec.update(status=summary['status'], summary=summary)
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def precheck():
    """B2's stored twin (1e-3) of block 0, re-run for the first 3,000 nominal decisions, against the stored records."""
    out = HERE / 'precheck'
    if out.exists():
        raise SystemExit('precheck/ exists')
    cfg = dq_b2.config_for('surrogate625', 0, dq_b2.STUDY_B2, 3000, [0, 3000], freeze_sha256=PLAN_SHA256)
    rt = dq_b2.runtime(dq_b2.STUDY_B2)
    parity = dq_b2.twin_trajectory_parity(dq_b2.STUDY_B2, rt, cfg, 0)
    p4.train_arm('surrogate625', cfg, rt, out / 'run' / 'surrogate625')
    load = lambda f: [json.loads(x) for x in Path(f).read_text().splitlines()]          # noqa: E731
    new = load(out / 'run/surrogate625/metrics.jsonl')
    old = load(B2_RUNS / 'surrogate625__b0/run/surrogate625/metrics.jsonl')
    pick = lambda ev, kind: [e for e in ev if e['kind'] == kind]                           # noqa: E731
    checks = {}
    for kind, key in (('evaluation', 'results'), ('update', 'metrics')):
        a, b = pick(new, kind), pick(old, kind)
        n = len(a)
        checks[kind] = {'compared': n, 'identical': all(x[key] == y[key] for x, y in zip(a, b[:n]))}
    a, b = pick(new, 'episode'), pick(old, 'episode')
    strip = lambda e: {k: v for k, v in e.items() if k not in ('utc', 'wall_s', 'elapsed_wall_s')}      # noqa: E731
    checks['episode'] = {'compared': len(a), 'identical': all(strip(x) == strip(y) for x, y in zip(a, b[:len(a)]))}
    result = {'plan_sha256': PLAN_SHA256, 'twin_parity_P4': parity, 'checks': checks,
              'passed': all(c['identical'] and c['compared'] > 0 for c in checks.values())}
    (HERE / 'precheck.json').write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


def launch(workers):
    runs, records = HERE / 'runs', HERE / 'records'
    if runs.exists() or records.exists():
        raise SystemExit('runs/ or records/ exists: a second launch or resume is refused')
    runs.mkdir(); records.mkdir()
    (HERE / 'launch.json').write_text(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                  'plan_sha256': PLAN_SHA256, 'workers': workers,
                                                  'b2_code_tree_live_copy': p4.live_copy_tree()}, indent=1))
    # interleave the two sets so both progress together; slower HQ runs first within each block
    task_list = [(tag, arm, b, lr, cand, str(runs)) for b in dq_b2.BLOCKS
                 for tag, (arm, lr, cand) in sorted(NEW.items(), key=lambda kv: kv[1][0] != 'quantum')]
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
