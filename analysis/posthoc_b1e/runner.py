"""Post-hoc B1 extension E (see PLAN.md): optimised circuit steps against the best linear tangent step, on the rows
that extension C regenerated and saved; plus first-crossing and angle-displacement checks of B1's calibration.

    python runner.py launch --workers 12       # every unit whose extension-C record passed and whose rows were saved
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
IDEA_DQ = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_b1_tree_a66f8979')
for extra in (IDEA_DQ / 'src', IDEA_DQ / 'scripts', IDEA_DQ / 'scripts' / 'gnss_qrl_runtime'):
    sys.path.insert(0, str(extra))

import numpy as np                                                                                # noqa: E402
from scipy.optimize import minimize                                                               # noqa: E402
import torch                                                                                      # noqa: E402
import dq_b1 as b1                                                                                # noqa: E402
import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
B1C = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1c')
F64 = torch.float64
KAPPAS = (1e-3, 1e-2)
DELTA = b1.DELTA
MAXITER = 200
GRID = 4000
FEAS = 1e-3


def seeded(tag):
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], 'big'))


def load_rows(circuit, index):
    d = torch.load(B1C / 'rows' / f'{circuit}_{index:02d}.pt', weights_only=False)
    spec = b1.SPEC[circuit]
    return {s: ins.Rows(phi=qc.basis(spec, v['angles']), W=d['readout_W'], b=d['readout_b'], mask=v['mask'],
                        action=v['action'], advantage=v['advantage'], episode=v['episode'])
            for s, v in d['sets'].items()}


def chart_fisher(rows, c, support, A):
    out = torch.zeros(A.shape[1], A.shape[1], dtype=F64)
    for start in range(0, rows.n, ins.FISHER_CHUNK):
        ZA = ins._fisher_factor(rows, c, support, start, min(start + ins.FISHER_CHUNK, rows.n)) @ A
        out += ZA.T @ ZA
    out /= rows.n
    return 0.5 * (out + out.T)


def surrogate(rows, lp0_a, c):
    idx = torch.arange(rows.n)
    return (((ins.log_policy(rows, c)[idx, rows.action] - lp0_a).exp() - 1.) * rows.advantage).mean()


def kl_torch(rows, lp0, c):
    lp = ins.log_policy(rows, c)
    return torch.where(rows.mask, lp0.exp() * (lp0 - lp), torch.zeros_like(lp0)).sum(-1).mean()


def optimise(c_of_x, x_starts, F_, K_, lp0F_a, lp0K, kappa):
    """SLSQP: max L_F(c(x)) s.t. KL_K(c(x)) <= kappa, from each start; returns the best feasible end point."""
    cache = {}

    def evaluate(x):
        key = x.tobytes()
        if key not in cache:
            xt = torch.tensor(x, dtype=F64, requires_grad=True)
            c = c_of_x(xt)
            L = surrogate(F_, lp0F_a, c)
            gL, = torch.autograd.grad(L, xt, retain_graph=True)
            kl = kl_torch(K_, lp0K, c)
            gk, = torch.autograd.grad(kl, xt)
            cache.clear()
            cache[key] = (float(L), gL.numpy().copy(), float(kl), gk.numpy().copy())
        return cache[key]
    best, ends = None, []
    for name, x0 in x_starts:
        res = minimize(lambda x: (-evaluate(x)[0], -evaluate(x)[1]), x0, jac=True, method='SLSQP',
                       constraints=[{'type': 'ineq', 'fun': lambda x: kappa - evaluate(x)[2],
                                     'jac': lambda x: -evaluate(x)[3]}],
                       options={'maxiter': MAXITER, 'ftol': 1e-14})
        L, _, kl, _ = evaluate(res.x)
        end = {'start': name, 'L_F': L, 'kl_K': kl, 'feasible': kl <= kappa * (1 + FEAS), 'iterations': int(res.nit),
               'message': str(res.message)}
        ends.append(end)
        if end['feasible'] and (best is None or L > best[1]['L_F']):
            best = (res.x.copy(), end)
    return best, ends


def unit(task):
    circuit, index = task
    torch.set_num_threads(1)
    rec = {'circuit': circuit, 'index': index, 'status': 'started', 'plan_sha256': PLAN_SHA256}
    t0 = time.perf_counter()
    try:
        spec = b1.SPEC[circuit]
        support = tp.support(spec)[0]
        rows = load_rows(circuit, index)
        F_, K_, H_ = rows['fit'], rows['calibration'], rows['heldout']
        model = b1.build_unit(circuit, index)
        w0 = b1.circuit_weights(model).reshape(-1).to(F64)
        active = [i for i in range(spec.n_weights) if i not in spec.inert_weights()]
        E = torch.zeros(spec.n_weights, len(active), dtype=F64)
        E[active, range(len(active))] = 1.
        c0 = qc.coefficients(spec, w0.reshape(spec.weight_shape)).reshape(-1)
        A0 = qc.jacobian(spec, w0.reshape(spec.weight_shape))[support]
        k, _, _ = ins.chart_rank(A0, b1.TAU_RANK)
        g = ins.gradient(F_, c0, support)
        F_t, F_Kt = chart_fisher(F_, c0, support, A0), chart_fisher(K_, c0, support, A0)
        d, _ = ins.theta_natural(A0.T @ g, F_t, DELTA)
        g_t = A0.T @ g
        T = ins.tangent_basis(A0, k)
        TFT = chart_fisher(F_, c0, support, T)
        lam_T = DELTA * float(torch.trace(TFT)) / k
        d_tc = T @ torch.linalg.solve(TFT + lam_T * torch.eye(k, dtype=F64), T.T @ g)
        idx = torch.arange(F_.n)
        lp0F_a = ins.log_policy(F_, c0)[idx, F_.action].detach()
        lp0K = ins.log_policy(K_, c0).detach()
        circ = (lambda w: qc.coefficients(spec, w.reshape(spec.weight_shape)).reshape(-1))
        lin = (lambda u: ins.full(c0[support] + A0[:, active] @ u, support, c0))
        rays = {'HQ-Fisher': ((lambda a: circ(w0 + a * d)), float(d @ F_Kt @ d)),
                'HQ-plain': ((lambda a: circ(w0 + a * g_t)), float(g_t @ F_Kt @ g_t)),
                'TAN': ((lambda a: ins.full(c0[support] + a * (A0 @ d), support, c0)), float(d @ F_Kt @ d)),
                'TAN-c': ((lambda a: ins.full(c0[support] + a * d_tc, support, c0)),
                          float(d_tc @ ins.fisher_vector(K_, c0, support, d_tc)))}
        stored = json.loads((B1C / 'records' / f'{circuit}_{index:02d}.json').read_text())
        out = {'k': k}
        for kappa in KAPPAS:
            kk = repr(kappa)
            res = {}
            alphas = {}
            for arm, (ray, curv) in rays.items():
                cal = ins.calibrate_kl(ray, c0, K_, curv, kappa)
                e = {'status': cal.status, 'alpha': cal.alpha, 'kl_K': cal.kl}
                if cal.status == 'ok':
                    cn = ray(cal.alpha)
                    e.update(G_H=ins.heldout_gain(H_, c0, cn), L_F=float(surrogate(F_, lp0F_a, cn)))
                    alphas[arm] = cal.alpha
                res[arm] = e
            # reproduction against extension C
            worst = 0.
            for arm in ('HQ-Fisher', 'TAN'):
                a, b = res[arm].get('G_H'), stored['arms'][arm][kk].get('G_H')
                if a is not None and b is not None:
                    worst = max(worst, abs(a - b) / max(abs(b), 1e-9))
            res['reproduction_max_rel_diff'] = worst
            # displacement and first crossing of the circuit rays
            for arm, v in (('HQ-Fisher', d), ('HQ-plain', g_t)):
                if arm in alphas:
                    step = (alphas[arm] * v)[active].abs()
                    res[arm].update(max_abs_angle_step=float(step.max()), mean_abs_angle_step=float(step.mean()))
            if 'HQ-Fisher' in alphas:
                ray = rays['HQ-Fisher'][0]
                grid = np.linspace(0., alphas['HQ-Fisher'], GRID + 1)[1:]
                kls = [ins.mean_kl(K_, c0, ray(float(a))) for a in grid]
                over = [i for i, v in enumerate(kls) if v > kappa * (1 + ins.KL_REL_TOL)]
                fc = {'monotone_on_grid': bool(np.all(np.diff(kls) >= -1e-9 * kappa)), 'earlier_crossing': bool(over)}
                if over:
                    i = over[0]
                    lo, hi = (grid[i - 1] if i > 0 else 0.), grid[i]
                    for _ in range(ins.MAX_BISECTIONS):
                        mid = 0.5 * (lo + hi)
                        v = ins.mean_kl(K_, c0, ray(mid))
                        if abs(v - kappa) <= ins.KL_REL_TOL * kappa:
                            break
                        lo, hi = (mid, hi) if v < kappa else (lo, mid)
                    fc.update(alpha_first=mid, G_H_first=ins.heldout_gain(H_, c0, ray(mid)))
                res['first_crossing_HQ-Fisher'] = fc
            # optimised steps
            n_act = len(active)

            def random_start(kind, j):
                r = torch.randn(n_act, generator=seeded(f'b1e|{circuit}|{index}|{kind}{j}|{kk}'), dtype=F64)
                ray = (lambda a: circ(w0 + a * (E @ r))) if kind == 'circ' else (lambda a: lin(a * r))
                curv = float((E @ r) @ F_Kt @ (E @ r))           # the same initial curvature for both rays
                cal = ins.calibrate_kl(ray, c0, K_, curv, kappa / 4)
                return (cal.alpha * r).numpy() if cal.status == 'ok' else None
            starts_c = []
            if 'HQ-Fisher' in alphas:
                starts_c.append(('HQ-Fisher', (alphas['HQ-Fisher'] * d)[active].numpy()))
            if 'HQ-plain' in alphas:
                starts_c.append(('HQ-plain', (alphas['HQ-plain'] * g_t)[active].numpy()))
            for j in range(3):
                x = random_start('circ', j)
                if x is not None:
                    starts_c.append((f'random{j}', x))
            best, ends = optimise(lambda x: circ(w0 + E @ x), starts_c, F_, K_, lp0F_a, lp0K, kappa)
            res['OPT-CIRC'] = {'ends': ends}
            if best is not None:
                cn = circ(w0 + E @ torch.tensor(best[0], dtype=F64)).detach()
                res['OPT-CIRC'].update(best[1], G_H=ins.heldout_gain(H_, c0, cn), kl_H=ins.mean_kl(H_, c0, cn),
                                       max_abs_angle_step=float(np.abs(best[0]).max()))
            starts_t = []
            if 'TAN' in alphas:
                starts_t.append(('TAN', (alphas['TAN'] * d)[active].numpy()))
            for j in range(3):
                x = random_start('tan', j)
                if x is not None:
                    starts_t.append((f'random{j}', x))
            best, ends = optimise(lin, starts_t, F_, K_, lp0F_a, lp0K, kappa)
            res['OPT-TAN'] = {'ends': ends}
            if best is not None:
                cn = lin(torch.tensor(best[0], dtype=F64)).detach()
                res['OPT-TAN'].update(best[1], G_H=ins.heldout_gain(H_, c0, cn), kl_H=ins.mean_kl(H_, c0, cn))
            out[kk] = res
        rec['arms'] = out
        rec['rows_sha256'] = hashlib.sha256((B1C / 'rows' / f'{circuit}_{index:02d}.pt').read_bytes()).hexdigest()
        rec['status'] = 'complete'
    except Exception as exc:                                                        # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def eligible():
    tasks = []
    for f in sorted((B1C / 'records').glob('*.json')):
        r = json.loads(f.read_text())
        if r.get('status') == 'complete' and r['reproduction']['passed'] and r.get('rows_saved', {}).get('saved'):
            tasks.append((r['circuit'], r['index']))
    return tasks


def launch(workers):
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    tasks = [t for t in eligible() if not (records / f'{t[0]}_{t[1]:02d}.json').exists()]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'workers': workers,
                            'plan_sha256': PLAN_SHA256,
                            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'tasks': len(tasks)}) + '\n')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(unit, tasks):
            (records / f'{rec["circuit"]}_{rec["index"]:02d}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['circuit'], rec['index'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    launch(a.workers)
