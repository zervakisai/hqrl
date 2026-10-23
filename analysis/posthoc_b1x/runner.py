"""Post-hoc B1 extension (see PLAN.md): re-linearised circuit steps and a damping ladder on the 72 B1 units.
Imports the B1 code tree of idea_dq unchanged; regenerates the F, K and H sets of each unit from their frozen seeds.

    python runner.py precheck                    # unit shipped 0: regenerated sets reproduce the stored B1 record
    python runner.py launch --workers 4 [--circuit shipped|heldout]     # one record per unit under records/; skips done
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
#: the B1 code tree of record (a66f8979), extracted unchanged from its source lock; output/ and docs/ are links to
#: the live checkout's (data, freezes and approvals)
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_b1_tree_a66f8979')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))

import torch                                                                                      # noqa: E402
import dq_b1 as b1                                                                                # noqa: E402
import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
B1_UNITS = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units')
F64 = torch.float64
SETS = ('fit', 'calibration', 'heldout')          # the oracle set O is not needed
DELTA = b1.DELTA                                  # 1e-6, the frozen B1 relative damping
DAMPING = (1e-4, 1e-2, 1.)                        # damping ladder; 1e-6 is B1 itself, delta -> inf is HQ-plain (stored)
STEPS = (4, 16)                                   # re-linearised sub-steps: per-step KL target kappa / m^2
KAPPAS = (1e-3, 1e-2)                             # B1's secondary and primary KL targets
MAX_STEPS_FACTOR = 25                             # at most 25 m sub-steps before 'unattained'


# -- instruments built only from dq_instruments ------------------------------------------------------------------------

def chart_fisher(rows, c, support, A):
    """A^T F_S(c) A without forming F_S: the Fisher pulled back through a chart or basis A (|S|, p)."""
    out = torch.zeros(A.shape[1], A.shape[1], dtype=F64)
    for start in range(0, rows.n, ins.FISHER_CHUNK):
        ZA = ins._fisher_factor(rows, c, support, start, min(start + ins.FISHER_CHUNK, rows.n)) @ A
        out += ZA.T @ ZA
    out /= rows.n
    return 0.5 * (out + out.T)


def is_gradient(rows, c, c0, support):
    """Gradient at c of (1/N) sum_i pi_c(a_i)/pi_c0(a_i) Ahat_i, the importance-weighted surrogate on the fit set;
    equal to ins.gradient at c = c0."""
    idx = torch.arange(rows.n)
    ratio = (ins.log_policy(rows, c)[idx, rows.action] - ins.log_policy(rows, c0)[idx, rows.action]).exp()
    return ins.gradient(dataclasses.replace(rows, advantage=rows.advantage * ratio), c, support)


def coeffs(spec, w):
    return qc.coefficients(spec, w.reshape(spec.weight_shape)).reshape(-1)


def single(ray, c0, K_, H_, curvature, kappa):
    cal = ins.calibrate_kl(ray, c0, K_, curvature, kappa)
    entry = {'status': cal.status, 'alpha': cal.alpha, 'kl_K': cal.kl}
    if cal.status == 'ok':
        cn = ray(cal.alpha)
        entry.update(G_H=ins.heldout_gain(H_, c0, cn), kl_H=ins.mean_kl(H_, c0, cn),
                     quadratic_rel_error=abs(0.5 * cal.alpha ** 2 * curvature - cal.kl) / cal.kl)
    return entry


def iterate(kind, spec, w0, c0, support, F_, K_, H_, kappa, m, T0, d_tan):
    """m-resolution re-linearised natural steps from c0 until the mean forward KL(pi_0 || pi) on K reaches kappa.

    kind 'HQ': each sub-step is the circuit's damped natural step at the current weights, realised through the circuit.
    kind 'TAN': each sub-step is the damped natural step inside the fixed initial tangent span T0, applied linearly.
    Each sub-step is calibrated to KL(pi_current || pi_next) = kappa / m^2 on K; the sub-step that crosses kappa is
    bisected to the first crossing. Gradient and Fisher are recomputed on the fit set at every sub-step."""
    eps = kappa / m ** 2
    c, w = c0.clone(), w0.reshape(-1).clone()
    for step in range(1, MAX_STEPS_FACTOR * m + 1):
        g = is_gradient(F_, c, c0, support)
        if kind == 'HQ':
            A = qc.jacobian(spec, w)[support]
            Ft = chart_fisher(F_, c, support, A)
            dt, _ = ins.theta_natural(A.T @ g, Ft, DELTA)
            ray = (lambda a, w=w, dt=dt: coeffs(spec, w + a * dt))
        else:
            Ft = chart_fisher(F_, c, support, T0)
            coef, _ = ins.theta_natural(T0.T @ g, Ft, DELTA)
            dt, d = coef, T0 @ coef
            ray = (lambda a, c=c, d=d: ins.full(c[support] + a * d, support, c))
        cal = ins.calibrate_kl(ray, c, K_, float(dt @ Ft @ dt), eps)
        if cal.status != 'ok':
            return {'status': f'substep_{cal.status}', 'steps': step}
        c_next = ray(cal.alpha)
        total = ins.mean_kl(K_, c0, c_next)
        if total > kappa * (1 + ins.KL_REL_TOL):
            lo, hi = 0., cal.alpha
            for _ in range(ins.MAX_BISECTIONS):
                mid = 0.5 * (lo + hi)
                c_mid = ray(mid)
                total = ins.mean_kl(K_, c0, c_mid)
                if abs(total - kappa) <= ins.KL_REL_TOL * kappa:
                    break
                lo, hi = (mid, hi) if total < kappa else (lo, mid)
            else:
                return {'status': 'no_convergence', 'steps': step}
            c_next, cal_alpha = c_mid, mid
        else:
            cal_alpha = cal.alpha
        if kind == 'HQ':
            w = w + cal_alpha * dt
        c = c_next
        if abs(total - kappa) <= ins.KL_REL_TOL * kappa:
            disp = (c - c0)[support]
            return {'status': 'ok', 'steps': step, 'kl_K': total, 'G_H': ins.heldout_gain(H_, c0, c),
                    'kl_H': ins.mean_kl(H_, c0, c),
                    'cos_to_TAN': float(disp @ d_tan / (disp.norm() * d_tan.norm())),
                    'weight_displacement': float((w - w0.reshape(-1)).norm()) if kind == 'HQ' else None}
    return {'status': 'unattained', 'steps': MAX_STEPS_FACTOR * m}


def unit_arms(circuit, model, rows, support):
    spec = b1.SPEC[circuit]
    w0 = b1.circuit_weights(model)
    c0 = qc.coefficients(spec, w0).reshape(-1)
    A0 = qc.jacobian(spec, w0)[support]
    k, sv, _ = ins.chart_rank(A0, b1.TAU_RANK)
    F_, K_, H_ = rows['fit'], rows['calibration'], rows['heldout']
    g = ins.gradient(F_, c0, support)
    g_t, F_t = A0.T @ g, chart_fisher(F_, c0, support, A0)
    F_Kt = chart_fisher(K_, c0, support, A0)
    T0 = ins.tangent_basis(A0, k)
    out = {'k': k}
    for delta in (DELTA,) + DAMPING:
        dt, _ = ins.theta_natural(g_t, F_t, delta)
        curv = float(dt @ F_Kt @ dt)
        tag = 'B1' if delta == DELTA else f'{delta:g}'
        rays = {f'HQ-Fisher@{tag}': (lambda a, dt=dt: coeffs(spec, w0.reshape(-1) + a * dt)),
                f'TAN@{tag}': (lambda a, d=A0 @ dt: ins.full(c0[support] + a * d, support, c0))}
        for arm, ray in rays.items():
            out[arm] = {repr(kappa): single(ray, c0, K_, H_, curv, kappa) for kappa in b1.KAPPAS}
    d_tan = A0 @ ins.theta_natural(g_t, F_t, DELTA)[0]
    for m in STEPS:
        out[f'HQ-Fisher-m{m}'] = {repr(kappa): iterate('HQ', spec, w0, c0, support, F_, K_, H_, kappa, m, T0, d_tan)
                                  for kappa in KAPPAS}
    m = max(STEPS)
    out[f'TAN-m{m}'] = {repr(kappa): iterate('TAN', spec, w0, c0, support, F_, K_, H_, kappa, m, T0, d_tan)
                        for kappa in KAPPAS}
    return out


def reproduction(circuit, index, rows, episodes, arms):
    """The regenerated sets and the two re-computed B1 arms against the stored B1 unit record."""
    stored = json.loads((B1_UNITS / f'{circuit}_{index:02d}.json').read_text())
    same_rows = all(rows[s].n == stored['rows'][s] for s in SETS)
    same_J = all([e['record']['J'] for e in episodes[s]] == [r['J'] for r in stored['episode_records'][s]] for s in SETS)
    worst = 0.
    for new, old in (('HQ-Fisher@B1', 'HQ-Fisher'), ('TAN@B1', 'TAN')):
        for kappa in b1.KAPPAS:
            a, b = arms[new][repr(kappa)], stored['instruments']['arms'][old][repr(kappa)]
            if a.get('G_H') is not None and b.get('G_H') is not None:
                worst = max(worst, abs(a['G_H'] - b['G_H']) / max(abs(b['G_H']), 1e-9))
    return {'same_rows': same_rows, 'same_episode_J': same_J, 'max_rel_diff_G_H_B1_arms': worst,
            'passed': same_rows and same_J and worst <= 1e-6}


def run_unit(task):
    circuit, index = task
    torch.set_num_threads(1)
    rec = {'circuit': circuit, 'index': index, 'status': 'started', 'plan_sha256': PLAN_SHA256}
    t0 = time.perf_counter()
    try:
        runtime = dq.cell_runtime()
        support = tp.support(b1.SPEC[circuit])[0]
        model = b1.build_unit(circuit, index)
        episodes, rows = {}, {}
        for s in SETS:
            episodes[s], _, _ = b1.collect_set(model, runtime, s, circuit, index)
            rows[s], _ = b1.rows_of(model, episodes[s], circuit)
        rec['rows'] = {s: rows[s].n for s in SETS}
        t1 = time.perf_counter()
        rec['arms'] = unit_arms(circuit, model, rows, support)
        rec['instrument_wall_s'] = time.perf_counter() - t1
        rec['reproduction'] = reproduction(circuit, index, rows, episodes, rec['arms'])
        rec['status'] = 'complete'
    except Exception as exc:                                                        # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def precheck():
    out = HERE / 'precheck.json'
    if out.exists():
        raise SystemExit('precheck.json exists')
    rec = run_unit(('shipped', 0))
    keep = {k: rec.get(k) for k in ('status', 'rows', 'reproduction', 'instrument_wall_s', 'wall_s', 'error')}
    keep['plan_sha256'] = PLAN_SHA256
    out.write_text(json.dumps(keep, indent=1, default=float))
    print(json.dumps(keep, indent=1, default=float))
    # the unit's arm values are not printed or kept here; the launch recomputes this unit like every other


def launch(workers, circuits):
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    tasks = [(c, i) for c in circuits for i in range(b1.UNITS_PER_CIRCUIT)
             if not (records / f'{c}_{i:02d}.json').exists()]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'workers': workers,
                            'plan_sha256': PLAN_SHA256, 'b1_code_tree_live_copy': p4.live_copy_tree(),
                            'tasks': len(tasks)}) + '\n')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_unit, tasks):
            (records / f'{rec["circuit"]}_{rec["index"]:02d}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['circuit'], rec['index'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=4)
    la.add_argument('--circuit', choices=b1.CIRCUITS, action='append')
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers, a.circuit or list(b1.CIRCUITS))
