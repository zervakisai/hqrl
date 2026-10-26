"""Post-hoc B1 extension N (see PLAN.md): fit-set size 64 / 128 / 256 / 1024 on the 36 two-layer B1 units.
K and H are extension C's saved rows; the fit set is regenerated from its frozen seeds and extended by 768 new episodes.

    python runner.py precheck              # unit 1, 8 new episodes only: reproduction and code path (no gain printed)
    python runner.py launch --workers 9    # every two-layer unit that extension C kept and whose rows it saved
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
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_b1_tree_a66f8979')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))

import torch                                                                                      # noqa: E402
import dq_b1 as b1                                                                                # noqa: E402
import dq_batches as db                                                                           # noqa: E402
import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
B1C = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1c')
B1_UNITS = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units')
F64 = torch.float64
CIRCUIT = 'shipped'
KAPPAS = (1e-3, 1e-2)
DELTA = b1.DELTA
NEW_PER_CELL = 96
SEED_OFFSET = 50_000
ACTION_OFFSET = 1000
SIZES = (64, 128, 256, 1024)


def load_KH(index):
    d = torch.load(B1C / 'rows' / f'{CIRCUIT}_{index:02d}.pt', weights_only=False)
    spec = b1.SPEC[CIRCUIT]
    return {s: ins.Rows(phi=qc.basis(spec, v['angles']), W=d['readout_W'], b=d['readout_b'], mask=v['mask'],
                        action=v['action'], advantage=v['advantage'], episode=v['episode'])
            for s, v in d['sets'].items() if s in ('calibration', 'heldout')}


def new_fit_episodes(model, runtime, index, per_cell):
    namespace = dq.require_open('dq_fit')
    space = dq.DQ_NAMESPACES[namespace]
    out = []
    for offset in range(per_cell):
        for cell_index in range(b1.CELLS):
            seed = space['start'] + SEED_OFFSET + (b1.slot(CIRCUIT, index) * b1.CELLS + cell_index) * NEW_PER_CELL + offset
            if not space['start'] + SEED_OFFSET <= seed < space['end']:
                raise RuntimeError('new fit seed outside its block')
            gen = torch.Generator().manual_seed(dq.DQ_STUDY.stream(db.SET_TAG['fit'], b1.stream_unit(CIRCUIT, index),
                                                                   cell_index, ACTION_OFFSET + offset))
            run = p4.run_learned_episode(runtime, runtime.cells[cell_index], namespace, seed, model, gen, False,
                                         f'b1n-{CIRCUIT}{index}-fitx-c{cell_index}-o{offset}', 0, db.GAMMA_REF, db.TAU)
            rec = run['record']
            if rec['infrastructure'] or rec['invalid_rows']:
                raise db.BatchError(f'infrastructure/invalid new fit episode c{cell_index} o{offset}')
            out.append({'cell': runtime.cells[cell_index], 'realization': 1000 + offset, 'record': rec,
                        'rows': [r for r in run['transitions'].rows if r['valid']]})
    return out


def single(ray, c0, K_, H_, curvature, kappa):
    cal = ins.calibrate_kl(ray, c0, K_, curvature, kappa)
    e = {'status': cal.status, 'alpha': cal.alpha, 'kl_K': cal.kl}
    if cal.status == 'ok':
        cn = ray(cal.alpha)
        e.update(G_H=ins.heldout_gain(H_, c0, cn), kl_H=ins.mean_kl(H_, c0, cn))
    return e


def arms_at(F_, K_, H_, model, index, support):
    spec = b1.SPEC[CIRCUIT]
    w0 = b1.circuit_weights(model)
    c0 = qc.coefficients(spec, w0).reshape(-1)
    A0 = qc.jacobian(spec, w0)[support]
    k, _, _ = ins.chart_rank(A0, b1.TAU_RANK)
    g, F = ins.gradient(F_, c0, support), ins.fisher(F_, c0, support)
    g_t, F_t = ins.theta_quantities(A0, g, F)
    F_K = ins.fisher(K_, c0, support)
    F_Kt = A0.T @ F_K @ A0
    dt, _ = ins.theta_natural(g_t, F_t, DELTA)
    dim = int(support.sum())
    su = b1.stream_unit(CIRCUIT, index)
    lin = (lambda d: ((lambda a: ins.full(c0[support] + a * d, support, c0)), float(d @ F_K @ d)))
    arms = {'HQ-Fisher': ((lambda a: qc.coefficients(spec, w0 + a * dt.reshape(spec.weight_shape)).reshape(-1)),
                          float(dt @ F_Kt @ dt)),
            'TAN': lin(A0 @ dt), 'SPAN': lin(ins.span_step(g, F, DELTA)[0])}
    for r in range(b1.RND_DRAWS):
        B = ins.haar_subspace(dim, k, torch.Generator().manual_seed(dq.DQ_STUDY.stream(b1.RND_TAG, su, k, r)))
        arms[f'RND-k#{r}'] = lin(ins.natural_step(B, g, F, DELTA)[0])
    return {arm: {repr(kappa): single(ray, c0, K_, H_, curv, kappa) for kappa in KAPPAS} for arm, (ray, curv) in arms.items()}


def run_unit(task, per_cell=NEW_PER_CELL):
    index = task
    torch.set_num_threads(1)
    rec = {'circuit': CIRCUIT, 'index': index, 'status': 'started', 'plan_sha256': PLAN_SHA256, 'new_per_cell': per_cell}
    t0 = time.perf_counter()
    try:
        runtime = dq.cell_runtime()
        support = tp.support(b1.SPEC[CIRCUIT])[0]
        model = b1.build_unit(CIRCUIT, index)
        fit256, _, _ = b1.collect_set(model, runtime, 'fit', CIRCUIT, index)
        stored = json.loads((B1_UNITS / f'{CIRCUIT}_{index:02d}.json').read_text())
        same_J = [e['record']['J'] for e in fit256] == [r['J'] for r in stored['episode_records']['fit']]
        extra = new_fit_episodes(model, runtime, index, per_cell)
        rec['data_wall_s'] = time.perf_counter() - t0
        KH = load_KH(index)
        sets = {64: [e for e in fit256 if e['realization'] < 8], 128: [e for e in fit256 if e['realization'] < 16],
                256: fit256, 1024: fit256 + extra}
        rec['episodes'] = {str(s): len(v) for s, v in sets.items()}
        rec['arms'] = {}
        for s, eps in sets.items():
            rows, _ = b1.rows_of(model, eps, CIRCUIT)
            rec['arms'][str(s)] = arms_at(rows, KH['calibration'], KH['heldout'], model, index, support)
            if s == 1024:
                c0 = qc.coefficients(b1.SPEC[CIRCUIT], b1.circuit_weights(model)).reshape(-1)
                rec['split_half_cosine_1024'] = b1.split_half(rows, c0, support)
        b1c = json.loads((B1C / 'records' / f'{CIRCUIT}_{index:02d}.json').read_text())
        worst = 0.
        for arm in ('HQ-Fisher', 'TAN'):
            for kappa in KAPPAS:
                a, b = rec['arms']['256'][arm][repr(kappa)].get('G_H'), b1c['arms'][arm][repr(kappa)].get('G_H')
                if a is not None and b is not None:
                    worst = max(worst, abs(a - b) / max(abs(b), 1e-9))
        rec['reproduction'] = {'same_fit_episode_J': same_J, 'max_rel_diff_G_H_F256_vs_C': worst,
                               'passed': same_J and worst <= 1e-6}
        rec['status'] = 'complete'
    except Exception as exc:                                                        # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def eligible():
    out = []
    for i in range(b1.UNITS_PER_CIRCUIT):
        f = B1C / 'records' / f'{CIRCUIT}_{i:02d}.json'
        if f.exists():
            r = json.loads(f.read_text())
            if r.get('status') == 'complete' and r['reproduction']['passed'] and r.get('rows_saved', {}).get('saved'):
                out.append(i)
    return out


def precheck():
    out = HERE / 'precheck.json'
    if out.exists():
        raise SystemExit('precheck.json exists')
    rec = run_unit(eligible()[0], per_cell=1)
    keep = {k: rec.get(k) for k in ('index', 'status', 'episodes', 'reproduction', 'data_wall_s', 'wall_s', 'error',
                                     'traceback')}
    if rec.get('status') == 'complete':
        keep['arm_status'] = {s: {a: {k: e['status'] for k, e in v.items()} for a, v in arms.items()
                                  if a in ('HQ-Fisher', 'TAN', 'SPAN', 'RND-k#0')} for s, arms in rec['arms'].items()}
    keep['plan_sha256'] = PLAN_SHA256
    out.write_text(json.dumps(keep, indent=1, default=float))
    print(json.dumps(keep, indent=1, default=float))


def launch(workers):
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    tasks = [i for i in eligible() if not (records / f'{CIRCUIT}_{i:02d}.json').exists()]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'workers': workers,
                            'plan_sha256': PLAN_SHA256,
                            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'b1_code_tree_live_copy': p4.live_copy_tree(), 'tasks': len(tasks)}) + '\n')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_unit, tasks):
            (records / f'{CIRCUIT}_{rec["index"]:02d}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['index'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=9)
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers)
