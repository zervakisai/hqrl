"""Post-hoc B1 extension G (see PLAN.md): the path comparison at generic circuit angles (part G) and on the classical
C-matched head (part D). Imports the B1 code tree of record unchanged.

    python runner.py precheck                  # G 0 and D 0: data, self-checks and arm status only (no gain printed)
    python runner.py launch --workers 5        # one record per task under records/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

import torch                                                                                      # noqa: E402
import dq_b1 as b1                                                                                # noqa: E402
import dq_batches as db                                                                           # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402
from aerial_qrl.training.rl_policy import RecurrentActorCritic                                    # noqa: E402

PLAN_SHA256 = hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
F64 = torch.float64
KAPPAS = (1e-3, 1e-2)
DELTA = b1.DELTA
CHECK_ROWS = 400                                  # rows used by the per-unit automatic-differentiation self-check


def seeded(tag):
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], 'big'))


# -- part G: B1 at generic circuit angles -------------------------------------------------------------------------------

def generic_unit(index):
    model = b1.build_unit('shipped', index)
    w = (torch.rand(model.actor.weights.shape, generator=seeded(f'b1g|shipped|{index}'), dtype=F64) * 2 - 1) * math.pi
    with torch.no_grad():
        model.actor.weights.copy_(w)
    return model


def run_generic(index, runtime):
    rec = {}
    model = generic_unit(index)
    support = tp.support(b1.SPEC['shipped'])[0]
    rec['circuit_weights_sha256'] = hashlib.sha256(b1.circuit_weights(model).numpy().astype('<f8').tobytes()).hexdigest()
    twin = b1.twin_of('shipped', model)
    rec['certificate'] = b1.certify_unit('shipped', model, twin, support)
    episodes, rows = {}, {}
    for s in ('fit', 'oracle', 'calibration', 'heldout'):
        episodes[s], _, _ = b1.collect_set(model, runtime, s, 'shipped', index)
        rows[s], _ = b1.rows_of(model, episodes[s], 'shipped')
    rec['rows'] = {s: rows[s].n for s in rows}
    rec['episode_J'] = {s: [e['record']['J'] for e in episodes[s]] for s in episodes}
    t = time.perf_counter()
    rec['instruments'], _ = b1.unit_instruments('shipped', index, model, rows, support)
    rec['instrument_wall_s'] = time.perf_counter() - t
    return rec


# -- part D: the classical C-matched head ---------------------------------------------------------------------------------

@dataclass
class CRows:
    u: torch.Tensor            # (N, 4) tanh(first layer(h))
    Wr: torch.Tensor           # (A, 4) readout
    br: torch.Tensor           # (A,)
    mask: torch.Tensor         # (N, A) bool
    action: torch.Tensor       # (N,)
    advantage: torch.Tensor    # (N,)

    @property
    def n(self):
        return int(self.u.shape[0])


def classical_unit(index):
    torch.manual_seed(dq.DQ_STUDY.stream(b1.INIT_TAG, index))
    return RecurrentActorCritic(b1.LAYOUT_V4_GNSS.dim, b1.LAYOUT_V4_GNSS.n_actions, 'mlp', hidden=64,
                                critic_gradient='stop', memory='gru', actor_head='bottleneck').to(F64).eval()


def crows_of(model, episodes):
    raw = db.raw_rows(episodes)
    features, replay_error = db.replay(model, raw)
    head = model.actor
    with torch.no_grad():
        u = torch.tanh(head[0](features))
        logits_model = head(features).masked_fill(~raw['mask'], torch.finfo(F64).min)
    advantage, fallback = ins.mc_advantages(raw['reward'], raw['episode'].numpy(), raw['cell'])
    rows = CRows(u=u, Wr=head[4].weight.detach().clone(), br=head[4].bias.detach().clone(), mask=raw['mask'],
                 action=raw['action'], advantage=advantage)
    theta0 = theta_of(model)
    mine = logits_of(rows, z_of(rows, theta0))
    agree = float((torch.log_softmax(mine, -1) - torch.log_softmax(logits_model, -1))[rows.mask].abs().max())
    return rows, {'replay_max_abs_logprob': replay_error, 'baseline_fallback_rows': fallback,
                  'logits_vs_model_max_abs_logprob': agree}


def theta_of(model):
    return torch.cat([model.actor[2].weight.detach().reshape(-1), model.actor[2].bias.detach()]).clone()


def z_of(rows, theta):
    M, b = theta[:16].reshape(4, 4), theta[16:]
    return torch.tanh(rows.u @ M.T + b)


def logits_of(rows, z):
    return (z @ rows.Wr.T + rows.br).masked_fill(~rows.mask, torch.finfo(F64).min)


def log_policy(rows, z):
    return torch.log_softmax(logits_of(rows, z), dim=-1)


def jac_z(rows, theta):
    """J (N, 4, 20) = dz/dtheta, theta = (vec(M) row-major, b)."""
    z = z_of(rows, theta)
    s = 1. - z ** 2
    J = torch.zeros(rows.n, 4, 20, dtype=F64)
    for j in range(4):
        J[:, j, 4 * j:4 * j + 4] = s[:, j:j + 1] * rows.u
        J[:, j, 16 + j] = s[:, j]
    return J


def grad_fisher(rows, theta):
    """(g_theta, F_theta): exact surrogate gradient and exact Fisher of the row-mean forward KL at theta."""
    z = z_of(rows, theta)
    pi = torch.softmax(logits_of(rows, z), -1)
    J = jac_z(rows, theta)
    residual = -pi
    residual[torch.arange(rows.n), rows.action] += 1.
    g = torch.einsum('na,aw,nwp->p', residual * rows.advantage[:, None], rows.Wr, J) / rows.n
    cov = torch.diag_embed(pi) - pi[:, :, None] * pi[:, None, :]
    M = rows.Wr.T @ cov @ rows.Wr                                                    # (N, 4, 4)
    F = torch.einsum('nwp,nwv,nvq->pq', J, M, J) / rows.n
    return g, 0.5 * (F + F.T)


def mean_kl(rows, lp_old, lp_new):
    p_old = lp_old.exp()
    terms = torch.where(rows.mask, p_old * (lp_old - lp_new), torch.zeros_like(p_old))
    return float(terms.sum(-1).mean())


def heldout_gain(rows, lp_old, lp_new):
    idx = torch.arange(rows.n)
    return float(((lp_new[idx, rows.action] - lp_old[idx, rows.action]).exp() - 1.) @ rows.advantage / rows.n)


def calibrate(kl_at, curvature, target):
    """B1's rule (dq_instruments.calibrate_kl) on a KL function of the scale: alpha0 from the quadratic prediction,
    doubling until the target is reached, then bisection to 1e-3 relative."""
    if not math.isfinite(curvature) or curvature <= ins.NULL_CURVATURE:
        return 'null_direction', 0., 0.
    lo, hi = 0., math.sqrt(2. * target / curvature)
    kl_hi = kl_at(hi)
    doublings = 0
    while kl_hi < target:
        if doublings >= ins.MAX_DOUBLINGS:
            return 'unattained', hi, kl_hi
        lo, hi = hi, 2. * hi
        kl_hi = kl_at(hi)
        doublings += 1
    if abs(kl_hi - target) <= ins.KL_REL_TOL * target:
        return 'ok', hi, kl_hi
    for _ in range(ins.MAX_BISECTIONS):
        mid = 0.5 * (lo + hi)
        kl_mid = kl_at(mid)
        if abs(kl_mid - target) <= ins.KL_REL_TOL * target:
            return 'ok', mid, kl_mid
        lo, hi = (mid, hi) if kl_mid < target else (lo, mid)
    return 'no_convergence', mid, kl_mid


def self_check(rows, theta0):
    """Closed-form J, g and F against automatic differentiation on the first CHECK_ROWS rows."""
    sub = CRows(rows.u[:CHECK_ROWS], rows.Wr, rows.br, rows.mask[:CHECK_ROWS], rows.action[:CHECK_ROWS],
                rows.advantage[:CHECK_ROWS])
    J_auto = torch.func.jacfwd(lambda t: z_of(sub, t))(theta0)
    lp0 = log_policy(sub, z_of(sub, theta0)).detach()

    def surrogate(t):
        idx = torch.arange(sub.n)
        return (((log_policy(sub, z_of(sub, t))[idx, sub.action] - lp0[idx, sub.action]).exp() - 1.)
                * sub.advantage).mean()

    def kl(t):
        lp = log_policy(sub, z_of(sub, t))
        return torch.where(sub.mask, lp0.exp() * (lp0 - lp), torch.zeros_like(lp0)).sum(-1).mean()
    g_auto = torch.func.grad(surrogate)(theta0)
    F_auto = torch.func.hessian(kl)(theta0)
    g, F = grad_fisher(sub, theta0)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())                          # noqa: E731
    out = {'J_rel': rel(jac_z(sub, theta0), J_auto), 'g_rel': rel(g, g_auto), 'F_rel': rel(F, F_auto)}
    out['passed'] = all(v <= 1e-9 for v in out.values())
    return out


def classical_arms(rows):
    F_, K_, H_ = rows['fit'], rows['calibration'], rows['heldout']
    theta0 = rows['theta0']
    z0 = {s: z_of(r, theta0) for s, r in (('K', K_), ('H', H_))}
    lp0 = {s: log_policy(r, z0[s]) for s, r in (('K', K_), ('H', H_))}
    g, F = grad_fisher(F_, theta0)
    lam = DELTA * float(torch.trace(F)) / 20
    d = torch.linalg.solve(F + lam * torch.eye(20, dtype=F64), g)
    _, F_K = grad_fisher(K_, theta0)
    J = {s: jac_z(r, theta0) for s, r in (('K', K_), ('H', H_))}
    ev = torch.linalg.eigvalsh(F)
    sv = torch.linalg.svdvals(jac_z(F_, theta0).reshape(-1, 20))
    out = {'diagnostics': {'fisher_condition': float(ev[-1] / ev[0]), 'jacobian_cond2': float((sv[0] / sv[-1]) ** 2),
                           'jacobian_singular_values_rel': [float(x) for x in sv / sv[0]],
                           'theta0_norm': float(theta0.norm()), 'd_norm': float(d.norm()), 'g_norm': float(g.norm())}}
    paths = {'C-Fisher': (d, 'through'), 'C-TAN': (d, 'linear'), 'C-plain': (g, 'through')}
    for arm, (v, kind) in paths.items():
        curv = float(v @ F_K @ v)

        def z_at(a, s, r, v=v, kind=kind):
            return z_of(r, theta0 + a * v) if kind == 'through' else z0[s] + a * (J[s] @ v)
        out[arm] = {}
        for kappa in KAPPAS:
            status, alpha, kl = calibrate(lambda a: mean_kl(K_, lp0['K'], log_policy(K_, z_at(a, 'K', K_))), curv, kappa)
            entry = {'status': status, 'alpha': alpha, 'kl_K': kl}
            if status == 'ok':
                lpH = log_policy(H_, z_at(alpha, 'H', H_))
                entry.update(G_H=heldout_gain(H_, lp0['H'], lpH), kl_H=mean_kl(H_, lp0['H'], lpH),
                             quadratic_rel_error=abs(0.5 * alpha ** 2 * curv - kl) / kl,
                             parameter_displacement=alpha * float(v.norm()))
            out[arm][repr(kappa)] = entry
    return out


def run_classical(index, runtime):
    rec = {}
    model = classical_unit(index)
    hq = b1.build_unit('shipped', index)
    rec['trunk_identical_to_B1_unit'] = all(
        all(torch.equal(a, b) for a, b in zip(getattr(model, p).state_dict().values(), getattr(hq, p).state_dict().values()))
        for p in ('encoder', 'gru', 'critic')) and torch.equal(model.actor[0].weight, hq.actor.projection.weight)
    rows, info = {}, {}
    for s in ('fit', 'calibration', 'heldout'):
        eps, _, _ = b1.collect_set(model, runtime, s, 'shipped', index)
        rows[s], info[s] = crows_of(model, eps)
        rec.setdefault('episode_J', {})[s] = [e['record']['J'] for e in eps]
    rec['rows'] = {s: rows[s].n for s in rows}
    rec['row_info'] = info
    rows['theta0'] = theta_of(model)
    rec['self_check'] = self_check(rows['fit'], rows['theta0'])
    t = time.perf_counter()
    rec['arms'] = classical_arms(rows)
    rec['instrument_wall_s'] = time.perf_counter() - t
    return rec


# -- tasks ------------------------------------------------------------------------------------------------------------

def run_task(task):
    part, index = task
    torch.set_num_threads(1)
    rec = {'part': part, 'index': index, 'status': 'started', 'plan_sha256': PLAN_SHA256}
    t0 = time.perf_counter()
    try:
        runtime = dq.cell_runtime()
        rec.update(run_generic(index, runtime) if part == 'G' else run_classical(index, runtime))
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
        recs = pool.map(run_task, [('G', 0), ('D', 0)])
    keep = []
    for rec in recs:
        k = {x: rec.get(x) for x in ('part', 'index', 'status', 'rows', 'row_info', 'self_check', 'certificate',
                                     'trunk_identical_to_B1_unit', 'instrument_wall_s', 'wall_s', 'error', 'traceback')}
        if rec.get('status') == 'complete':
            arms = rec['instruments']['arms'] if rec['part'] == 'G' else {a: v for a, v in rec['arms'].items()
                                                                          if a != 'diagnostics'}
            k['arm_status'] = {a: {kk: e['status'] for kk, e in v.items()} for a, v in arms.items()
                               if a in ('HQ-Fisher', 'TAN', 'SPAN', 'C-Fisher', 'C-TAN', 'C-plain')}
        keep.append(k)
    out.write_text(json.dumps({'plan_sha256': PLAN_SHA256, 'tasks': keep}, indent=1, default=float))
    print(json.dumps({'plan_sha256': PLAN_SHA256, 'tasks': keep}, indent=1, default=float))


def launch(workers):
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    tasks = [(p, i) for i in range(b1.UNITS_PER_CIRCUIT) for p in ('G', 'D')
             if not (records / f'{p}_{i:02d}.json').exists()]
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'workers': workers,
                            'plan_sha256': PLAN_SHA256,
                            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            'b1_code_tree_live_copy': p4.live_copy_tree(), 'tasks': len(tasks)}) + '\n')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_task, tasks):
            (records / f'{rec["part"]}_{rec["index"]:02d}.json').write_text(json.dumps(rec, indent=1, default=float))
            print(rec['part'], rec['index'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('precheck')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    precheck() if a.cmd == 'precheck' else launch(a.workers)
