"""Post-hoc B1 extension T (see PLAN.md): the identical-policy local comparison at trained policies, the 36 B2 final
checkpoints of HQ (part T, every B1 arm) and of C-matched (part TD, extension G's classical-head arms).

    python runner.py precheck              # T 0 and TD 0: checkpoint identity, certificate, self-check, arm status
    python runner.py launch --workers 3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_b1_tree_a66f8979')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))

import torch                                                                                      # noqa: E402
import dq_b1 as b1                                                                                # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402
from aerial_qrl.training.rl_policy import RecurrentActorCritic                                    # noqa: E402

_spec = importlib.util.spec_from_file_location('b1g', '/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1g/runner.py')
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)                      # extension G's classical-head functions, unchanged

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
B2_RUNS = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs')
F64 = torch.float64
ARM_OF = {'T': 'quantum', 'TD': 'bottleneck'}


def final_checkpoint(arm, block):
    run = B2_RUNS / f'{arm}__b{block}' / 'run' / arm
    last = json.loads((run / 'metrics.jsonl').read_text().splitlines()[-1])
    finals = [c for c in last['checkpoints'] if c.get('final')]
    assert len(finals) == 1
    path = run / finals[0]['path']
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != finals[0]['sha256']:
        raise RuntimeError(f'{path}: SHA-256 differs from the run record')
    return path, sha, finals[0]['policy_decisions']


def trained_model(arm, block):
    path, sha, decisions = final_checkpoint(arm, block)
    model = RecurrentActorCritic(b1.LAYOUT_V4_GNSS.dim, b1.LAYOUT_V4_GNSS.n_actions, 'mlp', hidden=64,
                                 critic_gradient='stop', memory='gru', actor_head=arm).to(F64)
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model.eval(), {'checkpoint': str(path), 'sha256': sha, 'policy_decisions': decisions}


def run_T(block, runtime):
    rec = {}
    model, rec['checkpoint'] = trained_model('quantum', block)
    support = tp.support(b1.SPEC['shipped'])[0]
    twin = b1.twin_of('shipped', model)
    rec['certificate'] = b1.certify_unit('shipped', model, twin, support)
    rec['circuit_weights'] = [float(x) for x in b1.circuit_weights(model).reshape(-1)]
    episodes, rows = {}, {}
    for s in ('fit', 'oracle', 'calibration', 'heldout'):
        episodes[s], _, _ = b1.collect_set(model, runtime, s, 'shipped', block)
        rows[s], _ = b1.rows_of(model, episodes[s], 'shipped')
    rec['rows'] = {s: rows[s].n for s in rows}
    rec['episode_J'] = {s: [e['record']['J'] for e in episodes[s]] for s in episodes}
    t = time.perf_counter()
    rec['instruments'], _ = b1.unit_instruments('shipped', block, model, rows, support)
    rec['instrument_wall_s'] = time.perf_counter() - t
    return rec


def run_TD(block, runtime):
    rec = {}
    model, rec['checkpoint'] = trained_model('bottleneck', block)
    rows, info = {}, {}
    for s in ('fit', 'calibration', 'heldout'):
        eps, _, _ = b1.collect_set(model, runtime, s, 'shipped', block)
        rows[s], info[s] = g.crows_of(model, eps)
        rec.setdefault('episode_J', {})[s] = [e['record']['J'] for e in eps]
    rec['rows'] = {s: rows[s].n for s in rows}
    rec['row_info'] = info
    rows['theta0'] = g.theta_of(model)
    rec['self_check'] = g.self_check(rows['fit'], rows['theta0'])
    t = time.perf_counter()
    rec['arms'] = g.classical_arms(rows)
    rec['instrument_wall_s'] = time.perf_counter() - t
    return rec


def run_task(task):
    part, block = task
    torch.set_num_threads(1)
    rec = {'part': part, 'block': block, 'status': 'started', 'plan_sha256': PLAN_SHA256}
    t0 = time.perf_counter()
    try:
        runtime = dq.cell_runtime()
        rec.update(run_T(block, runtime) if part == 'T' else run_TD(block, runtime))
        rec['status'] = 'complete'
    except Exception as exc:                                                        # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def precheck():
    out = HERE / 'precheck.json'
    if out.exists():
        raise SystemExit('precheck.json exists')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(2, maxtasksperchild=1) as pool:
        recs = pool.map(run_task, [('T', 0), ('TD', 0)])
    keep = []
    for rec in recs:
        k = {x: rec.get(x) for x in ('part', 'block', 'status', 'checkpoint', 'rows', 'row_info', 'self_check',
                                     'instrument_wall_s', 'wall_s', 'error', 'traceback')}
        k['certificate_failures'] = (rec.get('certificate') or {}).get('static_failures')
        if rec.get('status') == 'complete':
            arms = rec['instruments']['arms'] if rec['part'] == 'T' else {a: v for a, v in rec['arms'].items()
                                                                          if a != 'diagnostics'}
            k['arm_status'] = {a: {kk: e['status'] for kk, e in v.items()} for a, v in arms.items()
                               if a in ('HQ-Fisher', 'TAN', 'SPAN', 'C-Fisher', 'C-TAN', 'C-plain')}
        keep.append(k)
    out.write_text(json.dumps({'plan_sha256': PLAN_SHA256, 'tasks': keep}, indent=1, default=float))
    print(json.dumps({'plan_sha256': PLAN_SHA256, 'tasks': keep}, indent=1, default=float))


def launch(workers):
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    tasks = [(p, b) for b in range(36) for p in ('T', 'TD') if not (records / f'{p}_{b:02d}.json').exists()]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'workers': workers,
                            'plan_sha256': PLAN_SHA256,
                            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'b1_code_tree_live_copy': p4.live_copy_tree(), 'tasks': len(tasks)}) + '\n')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_task, tasks):
            (records / f'{rec["part"]}_{rec["block"]:02d}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['part'], rec['block'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers)
