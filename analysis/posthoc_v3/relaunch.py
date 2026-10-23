"""Restart the runs of PLAN.md that a machine crash interrupted (see PLAN_ADDENDUM.md). Safe to repeat.

    python relaunch.py --workers 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time

import runner

HERE = runner.HERE
ADDENDUM_SHA256 = hashlib.sha256((HERE / 'PLAN_ADDENDUM.md').read_bytes()).hexdigest()


def main(workers):
    launch = json.loads((HERE / 'launch.json').read_text())
    tree = runner.p4.live_copy_tree()
    if launch['plan_sha256'] != runner.PLAN_SHA256 or launch['b2_code_tree_live_copy'] != tree:
        raise SystemExit('PLAN.md or the B2 code tree changed since launch.json: refused')
    runs, records = HERE / 'runs', HERE / 'records'
    utc = time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())
    tasks, kept, moved = [], [], []
    # the task order of runner.launch
    for b in runner.dq_b2.BLOCKS:
        for tag, (arm, lr, cand) in sorted(runner.NEW.items(), key=lambda kv: kv[1][0] != 'quantum'):
            name = f'{tag}_{arm}__b{b}'
            if (records / f'{name}.json').exists():
                kept.append(name)
                continue
            if (runs / name).exists():
                dest = HERE / 'interrupted' / utc / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(runs / name), str(dest))
                moved.append(name)
            tasks.append((tag, arm, b, lr, cand, str(runs)))
    entry = {'utc': utc, 'workers': workers, 'plan_sha256': runner.PLAN_SHA256, 'addendum_sha256': ADDENDUM_SHA256,
             'b2_code_tree_live_copy': tree, 'kept': kept, 'moved_to_interrupted': moved, 'to_run': len(tasks)}
    with open(HERE / 'relaunches.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f'kept {len(kept)}, interrupted {len(moved)}, to run {len(tasks)}', flush=True)
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(runner.run_one, tasks):
            rec['relaunch_utc'] = utc
            (records / f'{rec["set"]}_{rec["arm"]}__b{rec["block"]}.json').write_text(
                json.dumps(rec, indent=1, default=float))
            print(rec['set'], rec['arm'], rec['block'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    main(ap.parse_args().workers)
