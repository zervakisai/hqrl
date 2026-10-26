"""Post-hoc B1 extension C (see PLAN.md): matched-tangent classical charts and the mission effect of local updates.
Imports the B1 code tree of record unchanged; regenerates the F, K and H sets of each unit from their frozen seeds.

    python runner.py precheck                    # shipped 0 and heldout 0 with reduced draws/episodes: reproduction
                                                 # and timing only (no arm or mission value is printed or kept)
    python runner.py launch --workers 9 [--circuit shipped|heldout]     # one record per unit under records/
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
#: the B1 code tree of record (a66f8979), extracted unchanged from its source lock (as for the B1 extension)
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
B1_UNITS = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units')
F64 = torch.float64
SETS = ('fit', 'calibration', 'heldout')          # the oracle set O is not needed
DELTA = b1.DELTA                                  # 1e-6, the frozen B1 relative damping
KAPPAS = (1e-3, 1e-2)
DRAWS = 20                                        # CTRIG draws per unit, gains averaged within the unit
FEATURES = 64                                     # m, rows of W
LADDER = (0.1, 10.)                               # CTRIG curvature ladder (multiples of the matched scale s)
SUB_M, SUB_KAPPA, MAX_SUB = 16, 1e-2, 400         # CTRIG-16: per-sub-step KL kappa / m^2, at most 400 sub-steps
N_DIRECTIONS = 200                                # random unit directions for the curvature-concentration ratio
MISSION_KAPPA = 1e-2
MISSION_PER_CELL = 20                             # 160 fresh episodes per arm and unit
MISSION_SEED_OFFSET = 50_000                      # B1's own fresh episodes occupy the first 5,760 seeds
ACTION_OFFSET = 1000                              # B1's fresh episodes use action-stream offsets 0..9
MISSION_ARMS = ('BASE', 'HQ-Fisher', 'TAN', 'CTRIG#0')


# -- instruments built only from dq_instruments ------------------------------------------------------------------------

def chart_fisher(rows, c, support, A):
    """A^T F_S(c) A without forming F_S (the B1 extension's function)."""
    out = torch.zeros(A.shape[1], A.shape[1], dtype=F64)
    for start in range(0, rows.n, ins.FISHER_CHUNK):
        ZA = ins._fisher_factor(rows, c, support, start, min(start + ins.FISHER_CHUNK, rows.n)) @ A
        out += ZA.T @ ZA
    out /= rows.n
    return 0.5 * (out + out.T)


def is_gradient(rows, c, c0, support):
    """Gradient at c of the importance-weighted surrogate on the fit set (the B1 extension's function)."""
    idx = torch.arange(rows.n)
    ratio = (ins.log_policy(rows, c)[idx, rows.action] - ins.log_policy(rows, c0)[idx, rows.action]).exp()
    return ins.gradient(dataclasses.replace(rows, advantage=rows.advantage * ratio), c, support)


def coeffs(spec, w):
    return qc.coefficients(spec, w.reshape(spec.weight_shape)).reshape(-1)


def circuit_hessian(spec, w0, support):
    """Hc (|S|, n_w, n_w): exact second derivatives of the supported coefficients in the circuit weights."""
    flat = (lambda v: coeffs(spec, v))
    return torch.func.jacfwd(torch.func.jacfwd(flat))(w0.reshape(-1).to(F64))[support]


def draw_generator(circuit, index, draw):
    seed = int.from_bytes(hashlib.sha256(f'b1c|{circuit}|{index}|{draw}'.encode()).digest()[:8], 'big')
    return torch.Generator().manual_seed(seed)


def ctrig_draw(circuit, index, draw, n_w, inert, size):
    gen = draw_generator(circuit, index, draw)
    W = torch.randint(-1, 2, (FEATURES, n_w), generator=gen).to(F64)
    W[:, list(inert)] = 0.
    R = torch.randn(size, FEATURES, generator=gen, dtype=F64)
    return W, R


def ctrig_norm(W, G_R):
    """sum_ij H_ij^T F H_ij for H_ij = sum_k R_k W_ki W_kj, from G_R = R^T F R: sum_kl G_kl (W_k . W_l)^2."""
    P = W @ W.T
    return float((G_R * P ** 2).sum())


def ctrig_along(W, G_R, u):
    """||H[u, u]||_F^2 = sum_kl G_kl (W_k . u)^2 (W_l . u)^2 (unscaled)."""
    q = (W @ u) ** 2
    return float(q @ G_R @ q)


def single(ray, c0, K_, H_, curvature, kappa):
    cal = ins.calibrate_kl(ray, c0, K_, curvature, kappa)
    entry = {'status': cal.status, 'alpha': cal.alpha, 'kl_K': cal.kl}
    if cal.status == 'ok':
        cn = ray(cal.alpha)
        entry.update(G_H=ins.heldout_gain(H_, c0, cn), kl_H=ins.mean_kl(H_, c0, cn),
                     quadratic_rel_error=abs(0.5 * cal.alpha ** 2 * curvature - cal.kl) / cal.kl)
    return entry, (ray(cal.alpha) if cal.status == 'ok' else None)


def ctrig_chart(c0, support, A, W, R, s):
    """phi -> vec(C) of the CTRIG chart and its Jacobian at phi."""
    base = c0[support]

    def chart(phi):
        return ins.full(base + A @ torch.sin(phi) + s * (R @ (1. - torch.cos(W @ phi))), support, c0)

    def jac(phi):
        return A * torch.cos(phi)[None, :] + s * (R * torch.sin(W @ phi)[None, :]) @ W
    return chart, jac


def iterate_chart(chart, jac, n_w, c0, support, F_, K_, H_, kappa, m, d_tan):
    """Re-linearised damped natural sub-steps through a chart (the extension's HQ-Fisher-m with the chart in place of the
    circuit): per-sub-step KL(pi_current || pi_next) = kappa / m^2 on K, gradient and Fisher recomputed on F, the
    sub-step that crosses the total KL(pi_0 || pi) = kappa on K bisected to the first crossing."""
    eps = kappa / m ** 2
    phi = torch.zeros(n_w, dtype=F64)
    c = c0.clone()
    for step in range(1, MAX_SUB + 1):
        g = is_gradient(F_, c, c0, support)
        J = jac(phi)
        Ft = chart_fisher(F_, c, support, J)
        dt, _ = ins.theta_natural(J.T @ g, Ft, DELTA)
        ray = (lambda a, phi=phi, dt=dt: chart(phi + a * dt))
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
            c_next, a_used = c_mid, mid
        else:
            a_used = cal.alpha
        phi = phi + a_used * dt
        c = c_next
        if abs(total - kappa) <= ins.KL_REL_TOL * kappa:
            disp = (c - c0)[support]
            return {'status': 'ok', 'steps': step, 'kl_K': total, 'G_H': ins.heldout_gain(H_, c0, c),
                    'kl_H': ins.mean_kl(H_, c0, c),
                    'cos_to_TAN': float(disp @ d_tan / (disp.norm() * d_tan.norm()))}
    return {'status': 'unattained', 'steps': MAX_SUB}


def unit_arms(circuit, index, model, rows, support, draws=DRAWS, sub_steps=True):
    spec = b1.SPEC[circuit]
    n_w = spec.n_weights
    inert = spec.inert_weights()
    active = [i for i in range(n_w) if i not in inert]
    w0 = b1.circuit_weights(model).reshape(-1)
    c0 = coeffs(spec, w0)
    A0 = qc.jacobian(spec, w0.reshape(spec.weight_shape))[support]
    size = int(support.sum())
    F_, K_, H_ = rows['fit'], rows['calibration'], rows['heldout']
    g = ins.gradient(F_, c0, support)
    F_t = chart_fisher(F_, c0, support, A0)
    d, _ = ins.theta_natural(A0.T @ g, F_t, DELTA)
    curv = float(d @ chart_fisher(K_, c0, support, A0) @ d)
    v = A0 @ d                                                     # the shared initial coefficient direction
    Hc = circuit_hessian(spec, w0, support)                        # (|S|, n_w, n_w)
    h_c = torch.einsum('sij,i,j->s', Hc, d, d)                     # Hc[d, d]

    # one Fisher pass for every second-order quantity: [A0 | Hc columns | h_c | R of every draw]
    draws_WR = [ctrig_draw(circuit, index, k, n_w, inert, size) for k in range(draws)]
    big = torch.cat([A0, Hc.reshape(size, n_w * n_w), h_c[:, None]] + [R for _, R in draws_WR], dim=1)
    G = chart_fisher(F_, c0, support, big)
    o1, o2 = n_w, n_w + n_w * n_w
    G_H = G[o1:o2, o1:o2]                                          # Gram of the circuit Hessian columns
    norm_c = float(torch.trace(G_H))
    hFh = float(G[o2, o2])
    AFh = G[:n_w, o2]
    F_act = G[:n_w, :n_w][active][:, active]
    tangential_share = float(AFh[active] @ torch.linalg.pinv(F_act, hermitian=True) @ AFh[active]) / hFh
    gen_u = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(
        f'b1c-directions|{circuit}|{index}'.encode()).digest()[:8], 'big'))
    U = torch.randn(N_DIRECTIONS, n_w, generator=gen_u, dtype=F64)
    U[:, list(inert)] = 0.
    U = U / U.norm(dim=1, keepdim=True)
    d_hat = d / d.norm()

    def along_c(u):
        x = torch.outer(u, u).reshape(-1)
        return float(x @ G_H @ x)
    ratio_c = math.sqrt(along_c(d_hat)) / float(torch.tensor([math.sqrt(along_c(u)) for u in U]).mean())

    out = {'k_rank': ins.chart_rank(A0, b1.TAU_RANK)[0], 'alpha0_curvature': curv,
           'diagnostics': {'circuit_hessian_fisher_norm2': norm_c, 'circuit_h_dd_fisher_norm2': hFh,
                           'circuit_h_dd_tangential_share': tangential_share,
                           'circuit_along_d_ratio': ratio_c, 'd_norm': float(d.norm()),
                           'v_fisher_norm2': float(d @ F_t @ d)}}
    mission_c = {'BASE': c0}
    rays = {'HQ-Fisher': (lambda a: coeffs(spec, w0 + a * d)),
            'TAN': (lambda a: ins.full(c0[support] + a * v, support, c0)),
            'QTAY': (lambda a: ins.full(c0[support] + a * v + 0.5 * a * a * h_c, support, c0))}
    for arm, ray in rays.items():
        out[arm] = {}
        for kappa in KAPPAS:
            entry, cn = single(ray, c0, K_, H_, curv, kappa)
            out[arm][repr(kappa)] = entry
            if kappa == MISSION_KAPPA and arm in MISSION_ARMS:
                mission_c[arm] = cn

    out['CTRIG_draws'] = []
    for k, (W, R) in enumerate(draws_WR):
        sl = slice(o2 + 1 + k * FEATURES, o2 + 1 + (k + 1) * FEATURES)
        G_R = G[sl, sl]
        s = math.sqrt(norm_c / ctrig_norm(W, G_R))
        h_ct = s * (R @ ((W @ d) ** 2))                            # H[d, d] of the draw
        ratio_ct = math.sqrt(ctrig_along(W, G_R, d_hat)) / float(torch.tensor(
            [math.sqrt(ctrig_along(W, G_R, u)) for u in U]).mean())
        rec = {'draw': k, 'scale_s': s, 'along_d_ratio': ratio_ct, 'h_dd_fisher_norm2': float(
            s * s * ((W @ d) ** 2) @ G_R @ ((W @ d) ** 2))}
        variants = {'CTRIG': s, 'CTRIGx0.1': LADDER[0] * s, 'CTRIGx10': LADDER[1] * s}
        for name, sc in variants.items():
            chart, _ = ctrig_chart(c0, support, A0, W, R, sc)
            rec[name] = {}
            for kappa in KAPPAS:
                entry, cn = single((lambda a, chart=chart: chart(a * d)), c0, K_, H_, curv, kappa)
                rec[name][repr(kappa)] = entry
                if name == 'CTRIG' and k == 0 and kappa == MISSION_KAPPA:
                    mission_c['CTRIG#0'] = cn
        rec['CQR'] = {}
        for kappa in KAPPAS:
            ray = (lambda a, h=h_ct: ins.full(c0[support] + a * v + 0.5 * a * a * h, support, c0))
            rec['CQR'][repr(kappa)] = single(ray, c0, K_, H_, curv, kappa)[0]
        if k == 0 and sub_steps:
            chart, jac = ctrig_chart(c0, support, A0, W, R, s)
            rec['CTRIG-16'] = {repr(SUB_KAPPA): iterate_chart(chart, jac, n_w, c0, support, F_, K_, H_, SUB_KAPPA,
                                                              SUB_M, v)}
        out['CTRIG_draws'].append(rec)
    return out, mission_c


def reproduction(circuit, index, rows, episodes, arms):
    """The regenerated sets and the two re-computed B1 arms against the stored B1 unit record (kappa <= 0.01)."""
    stored = json.loads((B1_UNITS / f'{circuit}_{index:02d}.json').read_text())
    same_rows = all(rows[s].n == stored['rows'][s] for s in SETS)
    same_J = all([e['record']['J'] for e in episodes[s]] == [r['J'] for r in stored['episode_records'][s]] for s in SETS)
    worst = 0.
    for arm in ('HQ-Fisher', 'TAN'):
        for kappa in KAPPAS:
            a, b = arms[arm][repr(kappa)], stored['instruments']['arms'][arm][repr(kappa)]
            if a.get('G_H') is not None and b.get('G_H') is not None:
                worst = max(worst, abs(a['G_H'] - b['G_H']) / max(abs(b['G_H']), 1e-9))
    return {'same_rows': same_rows, 'same_episode_J': same_J, 'max_rel_diff_G_H_B1_arms': worst,
            'passed': same_rows and same_J and worst <= 1e-6}


# -- mission evaluation ---------------------------------------------------------------------------------------------------

def mission_episode(policy, runtime, circuit, index, cell_index, offset, label):
    namespace = dq.require_open('dq_fresh_eval')
    space = dq.DQ_NAMESPACES[namespace]
    seed = (space['start'] + MISSION_SEED_OFFSET
            + (b1.slot(circuit, index) * b1.CELLS + cell_index) * MISSION_PER_CELL + offset)
    if not space['start'] + MISSION_SEED_OFFSET <= seed < space['end']:
        raise RuntimeError(f'mission seed {seed} outside its block')
    generator = torch.Generator().manual_seed(dq.DQ_STUDY.stream(db.SET_TAG['fresh_eval'], b1.stream_unit(circuit, index),
                                                                 cell_index, ACTION_OFFSET + offset))
    return p4.run_learned_episode(runtime, runtime.cells[cell_index], namespace, seed, policy, generator, False,
                                  f'b1c-{circuit}{index}-mission-c{cell_index}-o{offset}-{label.replace("#", "_")}', 0,
                                  db.GAMMA_REF, db.TAU)


def mission(circuit, index, model, runtime, mission_c, per_cell=MISSION_PER_CELL):
    spec = b1.SPEC[circuit]
    out = {}
    for arm in MISSION_ARMS:
        c = mission_c.get(arm)
        if c is None:
            out[arm] = {'status': 'no_coefficients'}
            continue
        policy = b1.twin_of(circuit, model, c.reshape(spec.features, spec.wires))
        episodes = []
        for offset in range(per_cell):
            for cell_index in range(b1.CELLS):
                run = mission_episode(policy, runtime, circuit, index, cell_index, offset, arm)
                rec = run['record']
                episodes.append({'cell_id': rec['cell_id'], 'offset': offset, 'J': rec['J'],
                                 'tp_timely': rec['tp_timely'], 'fp_spatial': rec['fp_spatial'],
                                 'return_ok': rec['return_ok'], 'fixes_attempted': rec['fixes_attempted'],
                                 'infrastructure': rec['infrastructure'], 'decisions': len(run['policy'].decisions)})
        per_cell_J, weighted = p4.aggregate_evaluation(runtime, episodes)
        out[arm] = {'status': 'ok', 'weighted_J': weighted, 'per_cell_J': per_cell_J, 'episodes': episodes,
                    'coefficients_sha256': hashlib.sha256(c.numpy().astype('<f8').tobytes()).hexdigest()}
    return out


# -- one unit ---------------------------------------------------------------------------------------------------------

def save_rows(circuit, index, model, episodes, rows):
    """Compact copy of the regenerated F, K and H rows (head angles instead of the Fourier features, which
    qc.basis rebuilds bitwise) for later descriptive analyses without new episodes. Never fails the unit."""
    try:
        spec = b1.SPEC[circuit]
        out = {'readout_W': rows['fit'].W, 'readout_b': rows['fit'].b, 'sets': {}}
        for s in SETS:
            raw = db.raw_rows(episodes[s])
            features, _ = db.replay(model, raw)
            with torch.no_grad():
                angles = math.pi * torch.tanh(model.actor.projection(features))
            if not torch.equal(qc.basis(spec, angles), rows[s].phi):
                return {'saved': False, 'reason': f'{s}: rebuilt features differ'}
            r = rows[s]
            out['sets'][s] = {'angles': angles, 'mask': r.mask, 'action': r.action, 'advantage': r.advantage,
                              'episode': r.episode}
        path = HERE / 'rows' / f'{circuit}_{index:02d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(out, path)
        return {'saved': True, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    except Exception as exc:                                                       # recorded, never fatal
        return {'saved': False, 'reason': f'{type(exc).__name__}: {exc}'}


def run_unit(task, draws=DRAWS, per_cell=MISSION_PER_CELL, sub_steps=True):
    circuit, index = task
    torch.set_num_threads(1)
    rec = {'circuit': circuit, 'index': index, 'status': 'started', 'plan_sha256': PLAN_SHA256,
           'settings': {'draws': draws, 'mission_per_cell': per_cell, 'sub_steps': sub_steps}}
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
        rec['rows_saved'] = save_rows(circuit, index, model, episodes, rows)
        t1 = time.perf_counter()
        rec['data_wall_s'] = t1 - t0
        rec['arms'], mission_c = unit_arms(circuit, index, model, rows, support, draws, sub_steps)
        t2 = time.perf_counter()
        rec['instrument_wall_s'] = t2 - t1
        rec['reproduction'] = reproduction(circuit, index, rows, episodes, rec['arms'])
        rec['mission'] = mission(circuit, index, model, runtime, mission_c, per_cell)
        rec['mission_wall_s'] = time.perf_counter() - t2
        rec['status'] = 'complete'
    except Exception as exc:                                                        # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def _precheck_task(task):
    return run_unit(task, draws=2, per_cell=1, sub_steps=True)


def precheck():
    out = HERE / 'precheck.json'
    if out.exists():
        raise SystemExit('precheck.json exists')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(2, maxtasksperchild=1) as pool:
        recs = pool.map(_precheck_task, [('shipped', 0), ('heldout', 0)])
    keep = []
    for rec in recs:
        k = {x: rec.get(x) for x in ('circuit', 'index', 'status', 'rows', 'reproduction', 'data_wall_s',
                                     'instrument_wall_s', 'mission_wall_s', 'wall_s', 'error', 'traceback')}
        if rec.get('status') == 'complete':
            arms = rec['arms']
            k['arm_status'] = {a: {kk: e['status'] for kk, e in arms[a].items()} for a in ('HQ-Fisher', 'TAN', 'QTAY')}
            k['ctrig_status'] = [{n: {kk: e['status'] for kk, e in d[n].items()}
                                  for n in ('CTRIG', 'CTRIGx0.1', 'CTRIGx10', 'CQR', 'CTRIG-16') if n in d}
                                 for d in arms['CTRIG_draws']]
            k['mission_status'] = {a: m['status'] for a, m in rec['mission'].items()}
            k['mission_episodes'] = {a: len(m.get('episodes', [])) for a, m in rec['mission'].items()}
        keep.append(k)
    result = {'plan_sha256': PLAN_SHA256, 'units': keep}
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))
    # no arm gain or mission score is printed or kept; the launch recomputes these units like every other


def launch(workers, circuits):
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    tasks = [(c, i) for i in range(b1.UNITS_PER_CIRCUIT) for c in circuits
             if not (records / f'{c}_{i:02d}.json').exists()]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'workers': workers,
                            'plan_sha256': PLAN_SHA256, 'runner_sha256': hashlib.sha256(
                                Path(__file__).read_bytes()).hexdigest(),
                            'b1_code_tree_live_copy': p4.live_copy_tree(), 'tasks': len(tasks)}) + '\n')
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
    la.add_argument('--workers', type=int, default=9)
    la.add_argument('--circuit', choices=b1.CIRCUITS, action='append')
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers, a.circuit or list(b1.CIRCUITS))
