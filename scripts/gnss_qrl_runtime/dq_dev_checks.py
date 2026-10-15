"""Development-only engineering checks D-1 .. D-5 of the DQ protocol section 5.

    .venv/bin/python scripts/gnss_qrl_runtime/dq_dev_checks.py {d1,d2,d3,d4,d5}

Each check writes ONE new directory output/study/gnss_qrl_v1/dq/development/<check>_<utc>/ holding its record and a
provenance.json, through the DQ write guard. Material: E1 HQ checkpoints read-only (development material) and episodes
from the `dq_development` namespace only. No check computes a held-out gain, a fresh-episode return difference between
arms or a diagnostic correlation: D-1 .. D-5 produce certificates, numerical errors, counts, spectra and timings.
D-6 (the instrument pilot) is not implemented here; it waits for review 35.

Realisation ranges inside dq_development, one per check so no two checks share an episode:
    D-1 row parity 0-1, trajectory parity 2;   D-2 rows 10-11;   D-4 rows 12-13;   D-5 cost 20-21
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_batches as db                                                                           # noqa: E402
import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402

E1 = ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'e1' / 'e1_campaign_v2'
DEV_CHECKPOINTS = [
    *(f'evaluation/quantum__lr3e-3_ent0__b{b}/run/quantum/checkpoints/{name}'
      for b in (10, 11) for name in ('update00000_decisions00000000.pt', 'update000[0-9][0-9]_decisions00015*.pt', 'final_update00056_*.pt')),
    *(f'tuning/quantum__lr3e-3_ent0__b0/run/quantum/checkpoints/{name}'
      for name in ('update00000_decisions00000000.pt', 'final_update00056_*.pt')),
]
UNTRAINED = [f'evaluation/quantum__lr3e-3_ent0__b{b}/run/quantum/checkpoints/update00000_decisions00000000.pt'
             for b in range(10, 22)]
REAL = {'d1_rows': (0, 1), 'd1_trajectory': 2, 'd2': (10, 11), 'd4': (12, 13), 'd5': (20, 21)}
TAU_CANDIDATE, DELTA_CANDIDATE = 1e-6, 1e-4
HELDOUT_HEADS = 8


def resolve(patterns):
    paths = []
    for pattern in patterns:
        found = sorted(E1.glob(pattern))
        if len(found) != 1:
            raise dq.DQError(f'{pattern}: expected exactly one checkpoint, found {len(found)}')
        paths.append(found[0])
    return paths


def out_dir(check):
    stamp = dq.utc().replace(':', '').replace('-', '').replace('+0000', 'Z')
    return dq.claim_output(dq.DEVELOPMENT_ROOT / f'{check}_{stamp}', directory=True)


def finish(check, out, record, sources, started):
    record = dict(record, check=check, wall_s=time.perf_counter() - started, statement=dq.DEVELOPMENT_STATEMENT)
    dq.write_json(out / f'{check}.json', record)
    dq.write_json(out / 'provenance.json', dq.provenance('dq_development', sources=sources))
    dq.append_tree_watch(f'{check} done')
    print(json.dumps({'check': check, 'out': str(out), 'passed': record.get('passed')}))
    return record


def dev_rows(model, runtime, realizations, unit=0):
    episodes, timing = db.collect(model, runtime, 'development', unit, realizations, cells=runtime.cells[:2])
    raw = db.raw_rows(episodes)
    features, replay_error = db.replay(model, raw)
    return raw, features, replay_error, timing


# -- D-1 -----------------------------------------------------------------------------------------------------------

def d1():
    started = time.perf_counter()
    dq.require_open('dq_development')
    dq.append_tree_watch('d1 start')
    sources = resolve(DEV_CHECKPOINTS)
    out = out_dir('d1_transplant_certificate')
    runtime = dq.cell_runtime()
    support, support_record = tp.support(qc.SHIPPED)
    certificates, refused = [], []
    for path in sources:
        try:
            model, twin, C, record = tp.transplant_checkpoint(path, support)
            raw, _, replay_error, timing = dev_rows(model, runtime, REAL['d1_rows'])
            record.update(tp.certify_rows(model, twin, raw), replay_max_abs_logprob=replay_error,
                          rows_episodes=timing['episodes'])
            record['P4'] = db.trajectory_parity(model, twin, runtime, 0, runtime.cells[0], REAL['d1_trajectory'])
            certificates.append(record)
        except (tp.TransplantParityError, db.BatchError) as exc:
            refused.append({'source': str(path), 'error': str(exc)})
    # held-out circuit: algebraic parity of 8 random heads (P1, P2 against its own exact simulation, P6, P7)
    h_support, h_support_record = tp.support(qc.HELDOUT)
    heldout = []
    g = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, 97))
    for i in range(HELDOUT_HEADS):
        w = (0.1 if i % 2 == 0 else 1.0) * torch.randn(*qc.HELDOUT.weight_shape, generator=g, dtype=torch.float64)
        C = qc.coefficients(qc.HELDOUT, w)
        angles = (torch.rand(tp.P2_ROWS, 4, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
        z = qc.expectations(qc.HELDOUT, angles, w)
        rec = {'P1_solve_residual': qc.solve_residual(qc.HELDOUT, w, C),
               'P2_offgrid_max_abs': float((qc.basis(qc.HELDOUT, angles) @ C - z).abs().max()),
               'P6_max_off_support': float(C.reshape(-1)[~h_support].abs().max()),
               'P7_max_abs_z': float(z.abs().max())}
        rec['passed'] = (rec['P1_solve_residual'] <= tp.TOL['P1'] and rec['P2_offgrid_max_abs'] <= tp.TOL['P2']
                         and rec['P6_max_off_support'] <= tp.TOL['P6'] and rec['P7_max_abs_z'] <= 1 + tp.TOL['P7'])
        heldout.append(rec)
    passed = not refused and all(h['passed'] for h in heldout)
    return finish('d1_transplant_certificate', out, {
        'passed': passed, 'tolerances': tp.TOL, 'support_shipped': support_record, 'support_heldout': h_support_record,
        'certified': certificates, 'refused': refused, 'heldout_circuit_algebra': heldout,
        'worst': {k: max(c[k] for c in certificates) if certificates else None
                  for k in ('P1_solve_residual', 'P2_offgrid_max_abs', 'P3_max_abs_logit', 'P3_max_abs_logprob',
                            'P6_max_off_support', 'P7_max_abs_z', 'replay_max_abs_logprob')},
        'worst_P4_dlogprob': max(c['P4']['max_abs_dlogprob'] for c in certificates) if certificates else None},
        sources, started)


# -- D-2 and D-4 need replayed development rows ---------------------------------------------------------------------------

def _unit(path, runtime, support, realizations):
    model, twin, C, _ = tp.transplant_checkpoint(path, support)
    raw, features, _, timing = dev_rows(model, runtime, realizations)
    rows, info = db.head_rows(model, raw, features)
    return model, C.reshape(-1).detach(), rows, info, timing


def d2():
    started = time.perf_counter()
    dq.require_open('dq_development')
    dq.append_tree_watch('d2 start')
    sources = resolve([DEV_CHECKPOINTS[0], DEV_CHECKPOINTS[2]])       # block 10: untrained and final
    out = out_dir('d2_instrument_correctness')
    runtime = dq.cell_runtime()
    support, _ = tp.support(qc.SHIPPED)
    results = []
    for path in sources:
        model, c, rows, info, timing = _unit(path, runtime, support, REAL['d2'])
        weights = model.actor.weights.detach().clone()
        r = {'source': str(path), 'rows': rows.n, **info}
        # score vs autograd
        c_s = c[support].clone().requires_grad_(True)
        lp = ins.log_policy(rows, ins.full(c_s, support, torch.zeros_like(c)))[torch.arange(rows.n), rows.action]
        (lp @ rows.advantage / rows.n).backward()
        g = ins.gradient(rows, c, support)
        r['gradient_vs_autograd'] = float((g - c_s.grad).abs().max())
        # Fisher: explicit vs matrix-free, and vs the KL second difference
        F = ins.fisher(rows, c, support)
        gen = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, 96))
        mf, fd = 0., 0.
        for _ in range(5):
            v = torch.randn(int(support.sum()), generator=gen, dtype=torch.float64)
            mf = max(mf, float((ins.fisher_vector(rows, c, support, v) - F @ v).abs().max()) / float((F @ v).abs().max()))
            eps = 1e-4 / float(v.norm())
            kl = ins.mean_kl(rows, c, ins.full(c[support] + eps * v, support, c))
            quad = 0.5 * eps ** 2 * float(v @ F @ v)
            fd = max(fd, abs(kl - quad) / quad)
        r['fisher_matrix_free_rel'] = mf
        r['fisher_vs_kl_second_difference_rel'] = fd
        r['fisher_min_eig'] = float(torch.linalg.eigvalsh(F)[0])
        # pull-back vs autograd through the circuit
        A = qc.jacobian(qc.SHIPPED, weights)[support]
        g_t, F_t = ins.theta_quantities(A, g, F)
        w = weights.reshape(-1).clone().requires_grad_(True)
        lp = ins.log_policy(rows, qc.coefficients(qc.SHIPPED, w.reshape(2, 4, 2)).reshape(-1))[torch.arange(rows.n), rows.action]
        (lp @ rows.advantage / rows.n).backward()
        r['theta_gradient_vs_autograd'] = float((g_t - w.grad).abs().max())
        r['inert_theta_gradient_max'] = max(abs(float(g_t[i])) for i in qc.SHIPPED.inert_weights())
        # natural-step span invariance (delta -> 0)
        B = ins.haar_subspace(int(support.sum()), 12, gen)
        R = torch.randn(12, 12, generator=gen, dtype=torch.float64) + 3 * torch.eye(12, dtype=torch.float64)
        d1_, _ = ins.natural_step(B, g, F, 0.)
        d2_, _ = ins.natural_step(B @ R, g, F, 0.)
        r['natural_step_basis_invariance_rel'] = float((d1_ - d2_).abs().max() / d1_.abs().max())
        k, s, amb = ins.chart_rank(A, TAU_CANDIDATE)
        r['chart_rank'], r['rank_ambiguous'] = k, amb
        r['passed'] = (r['gradient_vs_autograd'] <= 1e-12 and r['fisher_matrix_free_rel'] <= 1e-10
                       and r['fisher_vs_kl_second_difference_rel'] <= 1e-2 and r['theta_gradient_vs_autograd'] <= 1e-12
                       and r['inert_theta_gradient_max'] <= 1e-13 and r['natural_step_basis_invariance_rel'] <= 1e-6
                       and r['fisher_min_eig'] >= -1e-10)
        results.append(r)
    # FIT-k on a planted subspace (instrument sanity; synthetic)
    n = int(support.sum())
    planted = ins.haar_subspace(n, 12, torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, 95)))
    U = torch.randn(200, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) @ planted.T
    Q, _, fit_record = ins.fit_subspace(U, torch.eye(n, dtype=torch.float64), 12, 1e-12)
    fit_min_cos = float(torch.linalg.svdvals(Q.T @ planted).min())
    passed = all(r['passed'] for r in results) and fit_min_cos >= 1 - 1e-8
    return finish('d2_instrument_correctness', out, {'passed': passed, 'units': results,
                                                     'fit_k_planted_min_cosine': fit_min_cos, 'fit_record': fit_record},
                  sources, started)


def d3():
    started = time.perf_counter()
    dq.append_tree_watch('d3 start')
    records = {}
    for spec in (qc.SHIPPED, qc.HELDOUT):
        mask, rec = tp.support(spec)
        records[spec.name] = rec
    support = tp.support(qc.SHIPPED)[0]
    sources = resolve(UNTRAINED + DEV_CHECKPOINTS[2:3] + DEV_CHECKPOINTS[5:6])
    out = out_dir('d3_support_rank')
    spectra = []
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    for path in sources:
        model, _, _ = load_checkpoint(path)
        w = model.actor.weights.detach()
        A = qc.jacobian(qc.SHIPPED, w)[support]
        k, s, amb = ins.chart_rank(A, TAU_CANDIDATE)
        group = {}
        for name, idx in (('ry_offsets', [l * 8 + j * 2 for l in range(2) for j in range(4)]),
                          ('rz_layer1', [j * 2 + 1 for j in range(4)]), ('rz_layer2_inert', [8 + j * 2 + 1 for j in range(4)])):
            group[name] = [float(x) for x in torch.linalg.svdvals(A[:, idx])]
        spectra.append({'source': str(path), 'rank': k, 'ambiguous': amb, 'singular_values': [float(x) for x in s],
                        'relative': [float(x / s[0]) for x in s], 'by_group': group})
    heldout_ranks = []
    g = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, 94))
    h_support = tp.support(qc.HELDOUT)[0]
    for _ in range(4):
        A = qc.jacobian(qc.HELDOUT, 0.1 * torch.randn(*qc.HELDOUT.weight_shape, generator=g, dtype=torch.float64))[h_support]
        k, s, amb = ins.chart_rank(A, TAU_CANDIDATE)
        heldout_ranks.append({'rank': k, 'ambiguous': amb, 'relative': [float(x / s[0]) for x in s]})
    passed = all(not s['ambiguous'] for s in spectra + heldout_ranks)
    return finish('d3_support_rank', out, {'passed': passed, 'tau_candidate': TAU_CANDIDATE, 'support': records,
                                           'shipped_spectra': spectra, 'heldout_random_init_ranks': heldout_ranks},
                  sources, started)


def d4():
    started = time.perf_counter()
    dq.require_open('dq_development')
    dq.append_tree_watch('d4 start')
    sources = resolve([DEV_CHECKPOINTS[2]])
    out = out_dir('d4_kl_mechanics')
    runtime = dq.cell_runtime()
    support, _ = tp.support(qc.SHIPPED)
    model, c, rows, _, _ = _unit(sources[0], runtime, support, REAL['d4'])
    weights = model.actor.weights.detach().clone()
    F, g = ins.fisher(rows, c, support), ins.gradient(rows, c, support)
    d, _ = ins.span_step(g, F, DELTA_CANDIDATE)
    ray = lambda a: ins.full(c[support] + a * d, support, c)                   # noqa: E731
    curv = float(d @ F @ d)
    results = {}
    for target in (1e-3, 1e-2, 5e-2):
        cal = ins.calibrate_kl(ray, c, rows, curv, target)
        quad = 0.5 * cal.alpha ** 2 * curv
        results[f'linear_{target}'] = dict(vars(cal), quadratic_prediction_rel_error=abs(quad - cal.kl) / cal.kl)
    A = qc.jacobian(qc.SHIPPED, weights)[support]
    g_t, F_t = ins.theta_quantities(A, g, F)
    dt, _ = ins.theta_natural(g_t, F_t, DELTA_CANDIDATE)
    mray = lambda a: qc.coefficients(qc.SHIPPED, weights + a * dt.reshape(2, 4, 2)).reshape(-1)   # noqa: E731
    for target in (1e-3, 1e-2, 5e-2):
        cal = ins.calibrate_kl(mray, c, rows, float(dt @ F_t @ dt), target)
        results[f'manifold_{target}'] = vars(cal)
    results['forced_null'] = vars(ins.calibrate_kl(ray, c, rows, 0., 1e-2))
    results['forced_unattained'] = vars(ins.calibrate_kl(ray, c, rows, curv * 1e12, 1e-2, max_doublings=3))
    results['forced_no_convergence'] = vars(ins.calibrate_kl(ray, c, rows, curv * 1e6, 1e-2, max_bisections=1))
    ok = [v for k, v in results.items() if not k.startswith('forced')]
    passed = (all(v['status'] == 'ok' and abs(v['kl'] - v['target']) <= ins.KL_REL_TOL * v['target'] for v in ok)
              and results['forced_null']['status'] == 'null_direction'
              and results['forced_unattained']['status'] == 'unattained'
              and results['forced_no_convergence']['status'] == 'no_convergence')
    return finish('d4_kl_mechanics', out, {'passed': passed, 'delta_candidate': DELTA_CANDIDATE, 'rows': rows.n,
                                           'calibrations': results}, sources, started)


def d5():
    started = time.perf_counter()
    dq.require_open('dq_development')
    dq.append_tree_watch('d5 start')
    sources = resolve([DEV_CHECKPOINTS[2]])
    out = out_dir('d5_cost')
    runtime = dq.cell_runtime()
    support, _ = tp.support(qc.SHIPPED)
    model, twin, C, _ = tp.transplant_checkpoint(sources[0], support)
    timing = {}
    for name, m in (('hq', model), ('twin', twin)):
        episodes, t = db.collect(m, runtime, 'development', 0, REAL['d5'], cells=runtime.cells[:2], label=name)
        timing[f'{name}_episodes'] = dict(t, s_per_episode=t['wall_s'] / t['episodes'], s_per_row=t['wall_s'] / t['rows'])
    raw = db.raw_rows(episodes)
    features, _ = db.replay(twin, raw)
    rows, _ = db.head_rows(twin, raw, features)
    c = C.reshape(-1)
    # instrument cost at the shipped support, scaled to 2000 rows by tiling the development rows
    reps = max(1, 2000 // rows.n)
    big = ins.Rows(phi=rows.phi.repeat(reps, 1), W=rows.W, b=rows.b, mask=rows.mask.repeat(reps, 1),
                   action=rows.action.repeat(reps), advantage=rows.advantage.repeat(reps),
                   episode=rows.episode.repeat(reps))
    t0 = time.perf_counter(); F = ins.fisher(big, c, support); timing['fisher_shipped_s'] = time.perf_counter() - t0
    timing['fisher_shipped_rows'] = big.n
    g = ins.gradient(big, c, support)
    t0 = time.perf_counter(); ins.span_step(g, F, DELTA_CANDIDATE); timing['span_solve_shipped_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    B = ins.haar_subspace(int(support.sum()), 12, torch.Generator().manual_seed(0))
    ins.natural_step(B, g, F, DELTA_CANDIDATE)
    timing['rnd12_step_s'] = time.perf_counter() - t0
    # held-out circuit support: Fisher on synthetic rows of the same size (algebra only)
    h_support, _ = tp.support(qc.HELDOUT)
    gen = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, 93))
    angles = (torch.rand(big.n, 4, generator=gen, dtype=torch.float64) * 2 - 1) * math.pi
    hrows = ins.Rows(phi=qc.basis(qc.HELDOUT, angles), W=rows.W, b=rows.b, mask=big.mask, action=big.action,
                     advantage=big.advantage, episode=big.episode)
    hc = qc.coefficients(qc.HELDOUT, 0.1 * torch.randn(*qc.HELDOUT.weight_shape, generator=gen, dtype=torch.float64)).reshape(-1)
    t0 = time.perf_counter(); HF = ins.fisher(hrows, hc, h_support); timing['fisher_heldout_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); ins.span_step(ins.gradient(hrows, hc, h_support), HF, DELTA_CANDIDATE)
    timing['span_solve_heldout_s'] = time.perf_counter() - t0
    timing['torch_threads'] = torch.get_num_threads()
    return finish('d5_cost', out, {'passed': True, 'timing': timing}, sources, started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check', choices=['d1', 'd2', 'd3', 'd4', 'd5'])
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    record = {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5}[args.check]()
    return 0 if record.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
