"""Freeze B1: the local same-policy study on fresh validation units (DQ protocol section 4; reviews 34-36).

    .venv/bin/python scripts/gnss_qrl_runtime/dq_b1.py seed-manifest --out <new dir>     # frozen seeds (no episode)
    .venv/bin/python scripts/gnss_qrl_runtime/dq_b1.py freeze        --out <new dir>     # the empty freeze package
    .venv/bin/python scripts/gnss_qrl_runtime/dq_b1.py launch --campaign <dir> --workers 8   # REFUSES without approval

A unit is one fresh model initialisation of one circuit. Nothing here reads a D-6 gain; the four D-6 decisions are
read from the sealed D-6 result and compared with the constants below, and any difference is a refusal.

Units          SHIPPED unit i (i = 0..35): the E1 recurrent policy with the shipped 2-layer circuit head, initialised by
               torch.manual_seed(Study('dq').stream(INIT_TAG, i)) exactly as p4_train.build_model builds an HQ model.
               HELDOUT unit i: the SAME seed; the encoder, GRU and critic are identical to shipped unit i, and the
               held-out 3-layer circuit head is built from the generator state at which the shipped head is built
               (so its projection is also identical); only the circuit weights and the readout differ.
Slots          shipped i -> slot i, held-out i -> slot 36 + i; episode stream unit = 1000 + slot.
Sets           F, O, K, H: 256 episodes each = 8 frozen core cells x 32 offsets, namespaces dq_fit, dq_oracle,
               dq_calibration, dq_heldout; E (secondary): 80 = 8 cells x 10 offsets in dq_fresh_eval, only for the E
               subset (shipped 0-11 and held-out 0-11), shared by every evaluated arm of a unit (CRN).
Seeds          disturbance seed = namespace.start + (slot * 8 + cell_index) * per_cell + offset   (per_cell 32; E 10)
               action stream    = SeedSequence([DQ_SALT, SET_TAG[set], 1000 + slot, cell_index, offset])
Frozen settings (D-6 result of record, review 36): tau_rank 1e-8, delta 1e-6, n = 256, kappa 1e-2 primary,
               5e-2 stress; KL tolerance 1e-3 kappa, 60 doublings, 100 bisections; 20 RND draws per k.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_batches as db                                                                           # noqa: E402
import dq_circuits as qc                                                                          # noqa: E402
import dq_instruments as ins                                                                      # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS                                           # noqa: E402
from aerial_qrl.training.smdp_ppo import RecurrentActorCritic                                    # noqa: E402

# ------------------------------------------------------------------ FROZEN DESIGN ----------------------------------
TOOL = 'gnss-dq-b1/1'
UNITS_PER_CIRCUIT = 36
CIRCUITS = ('shipped', 'heldout')
SPEC = {'shipped': qc.SHIPPED, 'heldout': qc.HELDOUT}
INIT_TAG = 50                                   # Study('dq').stream(INIT_TAG, i): the unit initialisation
CELLS = 8
SET_NAMESPACE = {'fit': 'dq_fit', 'oracle': 'dq_oracle', 'calibration': 'dq_calibration', 'heldout': 'dq_heldout',
                 'fresh_eval': 'dq_fresh_eval'}
PER_CELL = {'fit': 32, 'oracle': 32, 'calibration': 32, 'heldout': 32, 'fresh_eval': 10}
E_SUBSET = {'shipped': tuple(range(12)), 'heldout': tuple(range(12))}
E_ARMS = ('BASE', 'HQ-Fisher', 'RND-k#0', 'FIT-k', 'SPAN')
D6_RESULT = ROOT / 'output/study/gnss_qrl_v1/dq/development/d6/analysis/d6_results.json'
D6_RESULT_SHA256 = '413797fe8a9a32afa7e5847ba15ada5391271211a1325bd770c307915902e303'
D6_SEAL_SHA256 = 'b244bf03d0efa1e05b48ef9d084533fc406c593e74d657948d6145acc21f3a58'
TAU_RANK, DELTA, BATCH = 1e-8, 1e-6, 256
KAPPAS = (1e-3, 1e-2, 5e-2)
PRIMARY_KAPPA, STRESS_KAPPA = 1e-2, 5e-2
RND_DRAWS = 20
RND_EXTRA_K = (48, 192)                         # descriptive only
RND_TAG = 51                                    # Study('dq').stream(RND_TAG, slot, k, draw)
P2_TAG = 52
JACKKNIFE_ARMS = ('SPAN', 'HQ-Fisher', 'RND-k#0', 'FIT-k')
TORCH_THREADS = 1
# ------------------------------------------------------------------ END FROZEN DESIGN -------------------------------


class B1Error(dq.DQError):
    """A B1 refusal: D-6 decisions changed, a closed namespace, a unit parity failure."""


def require_d6_decisions():
    """The four D-6 decisions of record, read from the sealed result; any change is a refusal (review 36)."""
    if dq.sha256_file(D6_RESULT) != D6_RESULT_SHA256:
        raise B1Error('the D-6 result of record changed on disk')
    result = json.loads(D6_RESULT.read_text())
    if result['seal_sha256'] != D6_SEAL_SHA256:
        raise B1Error('the D-6 result names another seal')
    d = result['decisions']
    wanted = {'tau_rank': TAU_RANK, 'delta': DELTA, 'batch_per_set': BATCH, 'kappa_5e-2': 'stress',
              'kappa_primary': PRIMARY_KAPPA}
    if d != wanted:
        raise B1Error(f'the D-6 decisions {d} differ from the frozen B1 settings {wanted}')
    return d


def slot(circuit, index):
    if circuit not in CIRCUITS or not 0 <= index < UNITS_PER_CIRCUIT:
        raise B1Error(f'no unit {circuit} {index}')
    return index + (0 if circuit == 'shipped' else UNITS_PER_CIRCUIT)


def stream_unit(circuit, index):
    return 1000 + slot(circuit, index)


def episode_seed(namespace, circuit, index, cell_index, offset, set_name):
    per_cell = PER_CELL[set_name]
    if not 0 <= offset < per_cell or not 0 <= cell_index < CELLS:
        raise B1Error(f'offset {offset} / cell {cell_index} outside the {set_name} design')
    space = dq.DQ_NAMESPACES[namespace]
    seed = space['start'] + (slot(circuit, index) * CELLS + cell_index) * per_cell + offset
    if not space['start'] <= seed < space['end']:
        raise B1Error(f'seed {seed} outside {namespace}')
    return seed


# -- unit models --------------------------------------------------------------------------------------------------------

class DQModel(RecurrentActorCritic):
    """A RecurrentActorCritic with a DQ head (held-out circuit or a twin), trunk and critic copied from `source`."""

    def __init__(self, source, head, label):
        with torch.random.fork_rng(devices=[]):
            super().__init__(**dict(source.config, actor_head='linear'))
        self.to(torch.float64)
        for part in ('encoder', 'gru', 'critic'):
            getattr(self, part).load_state_dict(getattr(source, part).state_dict())
        self.actor = head.to(torch.float64)
        self.config['actor_head'] = label

    def actor_shots(self):
        return None

    def describe(self):
        return {'config': dict(self.config), 'dq_head': self.config['actor_head'],
                'parameter_counts': self.parameter_counts()}


def build_unit(circuit, index):
    """The unit's own policy (float64, eval mode)."""
    seed = dq.DQ_STUDY.stream(INIT_TAG, index)
    torch.manual_seed(seed)
    shipped = RecurrentActorCritic(LAYOUT_V4_GNSS.dim, LAYOUT_V4_GNSS.n_actions, 'mlp', hidden=64,
                                   critic_gradient='stop', memory='gru', actor_head='quantum').to(torch.float64).eval()
    if circuit == 'shipped':
        return shipped
    torch.manual_seed(seed)                              # replay the trunk draws to reach the head's generator state
    nn.Sequential(nn.Linear(LAYOUT_V4_GNSS.dim, 64), nn.Tanh())
    nn.GRU(64, 64, batch_first=True)
    with torch.random.fork_rng(devices=[]):
        head = qc.CircuitHead(64, LAYOUT_V4_GNSS.n_actions, qc.HELDOUT)
    return DQModel(shipped, head, 'dq_heldout_circuit').eval()


def circuit_weights(model):
    return model.actor.weights.detach().clone()


def coefficients_of(circuit, model):
    return qc.coefficients(SPEC[circuit], circuit_weights(model))


def twin_of(circuit, model, C=None):
    """The unit's exact Fourier twin (a surrogate625 model for shipped; a TwinHead DQModel for held-out)."""
    if circuit == 'shipped':
        twin, C0 = tp.make_twin(model)
        if C is not None:
            with torch.no_grad():
                twin.actor.surrogate.weight.copy_(C.T)
        return twin
    spec = qc.HELDOUT
    head = qc.TwinHead(64, LAYOUT_V4_GNSS.n_actions, spec).to(torch.float64)
    with torch.no_grad():
        head.projection.load_state_dict(model.actor.projection.state_dict())
        head.readout.load_state_dict(model.actor.readout.state_dict())
        head.surrogate.weight.copy_((coefficients_of(circuit, model) if C is None else C).T)
        head.surrogate.bias.zero_()
    return DQModel(model, head, 'dq_heldout_twin').eval()


def pennylane_heldout(weights, angles):
    """The PennyLane reference of the held-out circuit (P2): the QuantumActorHead layer structure with 3 layers."""
    import pennylane as qml
    device = qml.device('default.qubit', wires=4, shots=None)

    @qml.qnode(device, interface='torch', diff_method='backprop')
    def circuit(a, w):
        for layer in range(3):
            for wire in range(4):
                qml.RY(a[..., wire], wires=wire)
                qml.RY(w[layer, wire, 0], wires=wire)
                qml.RZ(w[layer, wire, 1], wires=wire)
            for wire in range(4):
                qml.CNOT(wires=[wire, (wire + 1) % 4])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]
    return torch.stack(circuit(angles, weights), dim=-1)


def certify_unit(circuit, model, twin, support):
    """Static certificate P1, P2, P5, P6, P7 for either circuit (fail closed)."""
    if circuit == 'shipped':
        return tp.certify_static(model, twin, coefficients_of(circuit, model), support)
    spec = qc.HELDOUT
    w = circuit_weights(model)
    C = coefficients_of(circuit, model)
    g = torch.Generator().manual_seed(dq.DQ_STUDY.stream(P2_TAG, 1))
    angles = (torch.rand(tp.P2_ROWS, 4, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    with torch.no_grad():
        reference = pennylane_heldout(w, angles)
        twin_z = twin.actor.surrogate(qc.basis(spec, angles))
    shared_src = {**{f'encoder.{k}': v for k, v in model.encoder.state_dict().items()},
                  **{f'gru.{k}': v for k, v in model.gru.state_dict().items()},
                  **{f'critic.{k}': v for k, v in model.critic.state_dict().items()},
                  **{f'projection.{k}': v for k, v in model.actor.projection.state_dict().items()},
                  **{f'readout.{k}': v for k, v in model.actor.readout.state_dict().items()}}
    shared_twin = {**{f'encoder.{k}': v for k, v in twin.encoder.state_dict().items()},
                   **{f'gru.{k}': v for k, v in twin.gru.state_dict().items()},
                   **{f'critic.{k}': v for k, v in twin.critic.state_dict().items()},
                   **{f'projection.{k}': v for k, v in twin.actor.projection.state_dict().items()},
                   **{f'readout.{k}': v for k, v in twin.actor.readout.state_dict().items()}}
    record = {'P1_solve_residual': qc.solve_residual(spec, w, C),
              'P2_offgrid_max_abs': float((twin_z - reference).abs().max()),
              'P5_unequal': [k for k in shared_src if not torch.equal(shared_src[k], shared_twin[k])],
              'P6_max_off_support': float(C.reshape(-1)[~support].abs().max()),
              'P7_max_abs_z': float(reference.abs().max())}
    failures = [name for name, ok in (('P1', record['P1_solve_residual'] <= tp.TOL['P1']),
                                      ('P2', record['P2_offgrid_max_abs'] <= tp.TOL['P2']),
                                      ('P5', not record['P5_unequal']),
                                      ('P6', record['P6_max_off_support'] <= tp.TOL['P6']),
                                      ('P7', record['P7_max_abs_z'] <= 1 + tp.TOL['P7'])) if not ok]
    record['static_failures'] = failures
    if failures:
        raise tp.TransplantParityError(f'static parity failed {failures}: {json.dumps(record)}')
    return record


# -- episodes -----------------------------------------------------------------------------------------------------

def run_unit_episode(model, runtime, set_name, circuit, index, cell_index, offset, label):
    namespace = dq.require_open(SET_NAMESPACE[set_name])
    cell = runtime.cells[cell_index]
    seed = episode_seed(namespace, circuit, index, cell_index, offset, set_name)
    generator = torch.Generator().manual_seed(dq.DQ_STUDY.stream(db.SET_TAG[set_name], stream_unit(circuit, index),
                                                                 cell_index, offset))
    return p4.run_learned_episode(runtime, cell, namespace, seed, model, generator, False,
                                  f'b1-{circuit}{index}-{set_name}-c{cell_index}-o{offset}-{label.replace("#", "_")}', 0,
                                  db.GAMMA_REF, db.TAU)


def collect_set(model, runtime, set_name, circuit, index, keep_first_decisions=False):
    episodes, wall, first = [], 0., None
    for offset in range(PER_CELL[set_name]):
        for cell_index in range(CELLS):
            run = run_unit_episode(model, runtime, set_name, circuit, index, cell_index, offset, 'unit')
            rec = run['record']
            if rec['infrastructure'] or rec['invalid_rows']:
                raise db.BatchError(f'infrastructure/invalid episode {set_name} c{cell_index} o{offset}: '
                                    f'{rec["termination_reason"]}')
            if keep_first_decisions and first is None:
                first = run
            episodes.append({'cell': runtime.cells[cell_index], 'realization': offset, 'record': rec,
                             'rows': [r for r in run['transitions'].rows if r['valid']]})
            wall += run['wall_s']
    return episodes, wall, first


def trajectory_parity_b1(first_run, twin, runtime, circuit, index):
    """P4: the twin re-flies F episode (cell 0, offset 0) with the same seed and action stream."""
    rerun = run_unit_episode(twin, runtime, 'fit', circuit, index, 0, 0, 'twin')
    a, b = first_run['policy'].decisions, rerun['policy'].decisions
    if len(a) != len(b):
        raise tp.TransplantParityError(f'P4: {len(a)} vs {len(b)} decisions')
    worst = 0.
    for x, y in zip(a, b):
        if x.action_index != y.action_index or not np.array_equal(x.obs, y.obs) or \
                not np.array_equal(x.public_mask, y.public_mask):
            raise tp.TransplantParityError(f'P4: decision {x.index} differs')
        if x.log_prob is not None:
            worst = max(worst, abs(x.log_prob - y.log_prob))
    if first_run['record']['J'] != rerun['record']['J'] or not worst <= tp.TOL['P4']:
        raise tp.TransplantParityError(f'P4: outcome or log-prob differs ({worst:.3e})')
    return {'decisions': len(a), 'max_abs_dlogprob': worst}


# -- instruments at one unit -----------------------------------------------------------------------------------------

def rows_of(model, episodes, circuit):
    raw = db.raw_rows(episodes)
    features, replay_error = db.replay(model, raw)
    rows, info = db.head_rows(model, raw, features, spec=SPEC[circuit])
    return rows, dict(info, replay_max_abs_logprob=replay_error)


def entropy(rows, c):
    lp = ins.log_policy(rows, c)
    p = lp.exp()
    return float(torch.where(rows.mask, -p * lp, torch.zeros_like(p)).sum(-1).mean())


def jackknife_pair(rows_H, c, c_a, c_b=None):
    """Episode-level jackknife SE of G_H(a) (or of G_H(a) - G_H(b))."""
    idx = torch.arange(rows_H.n)
    lp0 = ins.log_policy(rows_H, c)[idx, rows_H.action]

    def terms(cn):
        return ((ins.log_policy(rows_H, cn)[idx, rows_H.action] - lp0).exp() - 1.) * rows_H.advantage
    t = terms(c_a) - (terms(c_b) if c_b is not None else 0.)
    episodes = torch.unique(rows_H.episode)
    loo = torch.stack([t[rows_H.episode != e].mean() for e in episodes])
    m = len(episodes)
    return float(math.sqrt((m - 1) / m * float(((loo - loo.mean()) ** 2).sum())))


def unit_instruments(circuit, index, model, rows, support):
    """Every arm, calibration and gain of one unit at the frozen settings; plus diagnostics and reliability."""
    spec = SPEC[circuit]
    su = stream_unit(circuit, index)
    w = circuit_weights(model)
    c = qc.coefficients(spec, w).reshape(-1)
    A = qc.jacobian(spec, w)[support]
    k, sv, ambiguous = ins.chart_rank(A, TAU_RANK)
    F_, O_, K_, H_ = rows['fit'], rows['oracle'], rows['calibration'], rows['heldout']
    g, F = ins.gradient(F_, c, support), ins.fisher(F_, c, support)
    g_t, F_t = ins.theta_quantities(A, g, F)
    T = ins.tangent_basis(A, k)
    U, _ = ins.episode_contributions(O_, c, support)
    F_O = ins.fisher(O_, c, support)
    dim = int(support.sum())

    def lin(d):
        return (lambda a: ins.full(c[support] + a * d, support, c)), float(d @ F_K @ d)

    F_K = ins.fisher(K_, c, support)
    F_Kt = A.T @ F_K @ A
    arms = {}
    arms['SPAN'] = lin(ins.span_step(g, F, DELTA)[0])
    dt_f, _ = ins.theta_natural(g_t, F_t, DELTA)
    arms['TAN'] = lin(A @ dt_f)
    arms['ORTH'] = lin(ins.natural_step(ins.orthogonal_complement(T), g, F, DELTA)[0])
    Q, mu, fit_record = ins.fit_subspace(U, F_O, k, DELTA)
    arms['FIT-k'] = lin(ins.natural_step(Q, g, F, DELTA)[0])
    B1v, _ = ins.fit_one(U, F_O, DELTA)
    arms['FIT-1'] = lin(ins.natural_step(B1v, g, F, DELTA)[0])
    for kk, tag in [(k, 'RND-k')] + [(e, f'RND-{e}') for e in RND_EXTRA_K]:
        for r in range(RND_DRAWS):
            B = ins.haar_subspace(dim, kk, torch.Generator().manual_seed(dq.DQ_STUDY.stream(RND_TAG, su, kk, r)))
            arms[f'{tag}#{r}'] = lin(ins.natural_step(B, g, F, DELTA)[0])
    Br = ins.haar_subspace(dim, k, torch.Generator().manual_seed(dq.DQ_STUDY.stream(RND_TAG, su, k, 999)))
    arms['RND-spec-plain'] = lin(Br @ (sv[:k] ** 2 * (Br.T @ g)))
    for name, dt in (('HQ-Fisher', dt_f), ('HQ-plain', g_t), ('HQ-Adam-dir', g_t / (g_t.abs() + 1e-8))):
        shape = spec.weight_shape
        arms[name] = ((lambda a, dt=dt: qc.coefficients(spec, w + a * dt.reshape(shape)).reshape(-1)),
                      float(dt @ F_Kt @ dt))
    out, new_c = {}, {}
    for arm, (ray, curv) in arms.items():
        out[arm] = {}
        for kappa in KAPPAS:
            cal = ins.calibrate_kl(ray, c, K_, curv, kappa)
            entry = {'status': cal.status, 'alpha': cal.alpha, 'kl_K': cal.kl, 'doublings': cal.doublings,
                     'bisections': cal.bisections}
            if cal.status == 'ok':
                cn = ray(cal.alpha)
                entry['G_H'] = ins.heldout_gain(H_, c, cn)
                entry['kl_H'] = ins.mean_kl(H_, c, cn)
                entry['quadratic_rel_error'] = abs(0.5 * cal.alpha ** 2 * curv - cal.kl) / cal.kl
                if kappa == PRIMARY_KAPPA:
                    new_c[arm] = cn
                    if arm in JACKKNIFE_ARMS:
                        entry['G_H_jackknife_se'] = jackknife_pair(H_, c, cn)
            out[arm][repr(kappa)] = entry
    # diagnostics (section 4.8 and P-H1b)
    q_tan = ins.accessible(T, g, F, DELTA)
    q_span = float(g @ ins.span_step(g, F, DELTA)[0])
    ev_t = torch.linalg.eigvalsh(F_t[:, [i for i in range(spec.n_weights) if i not in spec.inert_weights()]]
                                 [[i for i in range(spec.n_weights) if i not in spec.inert_weights()], :])
    diagnostics = {'D': 1. - q_tan / q_span, 'rho_hat': q_tan / q_span, 'q_tan': q_tan, 'q_span': q_span,
                   'grad_norm': float(g.norm()), 'fisher_condition_tangent': float(ev_t[-1] / ev_t[0]) if float(ev_t[0]) > 0 else math.inf,
                   'entropy': entropy(F_, c)}
    # reliability (review 36)
    reliability = {'split_half_cosine': split_half(F_, c, support)}
    if 'SPAN' in new_c and 'HQ-Fisher' in new_c:
        y = out['SPAN'][repr(PRIMARY_KAPPA)]['G_H'] - out['HQ-Fisher'][repr(PRIMARY_KAPPA)]['G_H']
        se = jackknife_pair(H_, c, new_c['SPAN'], new_c['HQ-Fisher'])
        reliability.update(Y=y, Y_jackknife_se=se, Y_snr=y / se if se > 0 else None)
    return {'k': k, 'rank_ambiguous': ambiguous, 'singular_values': [float(x) for x in sv], 'support': dim,
            'fit_record': fit_record, 'arms': out, 'diagnostics': diagnostics, 'reliability': reliability}, new_c


def split_half(rows, c, support):
    F = ins.fisher(rows, c, support)
    halves = []
    for parity in (0, 1):
        keep = (rows.episode % 2) == parity
        sub = ins.Rows(phi=rows.phi[keep], W=rows.W, b=rows.b, mask=rows.mask[keep], action=rows.action[keep],
                       advantage=rows.advantage[keep], episode=rows.episode[keep])
        halves.append(ins.span_step(ins.gradient(sub, c, support), ins.fisher(sub, c, support), DELTA)[0])
    a, b = halves
    return float(a @ F @ b / math.sqrt(float(a @ F @ a) * float(b @ F @ b)))


def fresh_eval(circuit, index, model, new_c, runtime):
    """E (secondary): weighted J of the unit and each E arm (as exact twins) on 80 shared CRN episodes."""
    spec = SPEC[circuit]
    base_c = qc.coefficients(spec, circuit_weights(model))
    policies = {'BASE': twin_of(circuit, model)}
    for arm in E_ARMS[1:]:
        if arm in new_c:
            policies[arm] = twin_of(circuit, model, new_c[arm].reshape(spec.features, spec.wires))
    results = {}
    for arm, policy in policies.items():
        episodes = []
        for offset in range(PER_CELL['fresh_eval']):
            for cell_index in range(CELLS):
                run = run_unit_episode(policy, runtime, 'fresh_eval', circuit, index, cell_index, offset, arm)
                rec = run['record']
                episodes.append({'cell_id': rec['cell_id'], 'J': rec['J'], 'tp_timely': rec['tp_timely'],
                                 'fp_spatial': rec['fp_spatial'], 'return_ok': rec['return_ok'],
                                 'fixes_attempted': rec['fixes_attempted'], 'infrastructure': rec['infrastructure']})
        per_cell, weighted = p4.aggregate_evaluation(runtime, episodes)
        results[arm] = {'weighted_J': weighted, 'per_cell_J': per_cell, 'episodes': episodes}
    base = results['BASE']['weighted_J']
    return {'arms': results, 'delta_J': {a: r['weighted_J'] - base for a, r in results.items() if a != 'BASE'},
            'base_coefficients_sha256': hashlib.sha256(base_c.numpy().astype('<f8').tobytes()).hexdigest()}


def run_unit(circuit, index, runtime=None, support=None):
    """The whole per-unit pipeline. Returns the unit record (never raises: failures are recorded, not retried)."""
    torch.set_num_threads(TORCH_THREADS)
    started = time.perf_counter()
    record = {'tool': TOOL, 'circuit': circuit, 'index': index, 'slot': slot(circuit, index),
              'init_seed': dq.DQ_STUDY.stream(INIT_TAG, index), 'status': 'started'}
    try:
        runtime = runtime or dq.cell_runtime()
        support = support if support is not None else tp.support(SPEC[circuit])[0]
        model = build_unit(circuit, index)
        record['circuit_weights_sha256'] = hashlib.sha256(circuit_weights(model).numpy().astype('<f8').tobytes()).hexdigest()
        episodes, wall = {}, {}
        first = None
        for s in ('fit', 'oracle', 'calibration', 'heldout'):
            episodes[s], wall[s], f = collect_set(model, runtime, s, circuit, index, keep_first_decisions=(s == 'fit'))
            first = first or f
        record['collection_wall_s'] = wall
        twin = twin_of(circuit, model)
        record['certificate'] = certify_unit(circuit, model, twin, support)
        rows, infos = {}, {}
        for s in episodes:
            rows[s], infos[s] = rows_of(model, episodes[s], circuit)
        record['certificate'].update(tp.certify_rows(model, twin, db.raw_rows(episodes['fit'])))
        record['certificate']['P4'] = trajectory_parity_b1(first, twin, runtime, circuit, index)
        record['rows'] = {s: rows[s].n for s in rows}
        record['row_info'] = infos
        record['episode_records'] = {s: [e['record'] for e in episodes[s]] for s in episodes}
        del episodes
        t0 = time.perf_counter()
        record['instruments'], new_c = unit_instruments(circuit, index, model, rows, support)
        record['instrument_wall_s'] = time.perf_counter() - t0
        if index in E_SUBSET[circuit]:
            record['fresh_eval'] = fresh_eval(circuit, index, model, new_c, runtime)
        record['status'] = 'complete'
    except tp.TransplantParityError as exc:
        record.update(status='parity_refused', error=str(exc))
    except db.BatchError as exc:
        record.update(status='episode_failure', error=str(exc))
    except dq.DQError as exc:
        record.update(status='refused', error=str(exc))
    except Exception as exc:                                                        # algorithmic; recorded
        record.update(status='algorithmic_failure', error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc()[-4000:])
    record['wall_s'] = time.perf_counter() - started
    return record


# -- seed manifest, freeze and campaign ---------------------------------------------------------------------------------

def seed_manifest():
    units = []
    for circuit in CIRCUITS:
        for i in range(UNITS_PER_CIRCUIT):
            units.append({'circuit': circuit, 'index': i, 'slot': slot(circuit, i), 'stream_unit': stream_unit(circuit, i),
                          'init_seed': dq.DQ_STUDY.stream(INIT_TAG, i), 'fresh_eval': i in E_SUBSET[circuit],
                          'first_seeds': {s: episode_seed(SET_NAMESPACE[s], circuit, i, 0, 0, s) for s in PER_CELL}})
    all_seeds = {s: sorted(episode_seed(SET_NAMESPACE[s], c, i, cell, o, s) for c in CIRCUITS
                           for i in range(UNITS_PER_CIRCUIT) for cell in range(CELLS) for o in range(PER_CELL[s]))
                 for s in PER_CELL}
    return {'manifest_version': 'gnss-dq-b1-seed-manifest/1', 'study': 'dq', 'salt': [dq.DQ_SALT],
            'namespaces': {s: dq.DQ_NAMESPACES[ns] for s, ns in SET_NAMESPACE.items()},
            'rules': {'seed': 'namespace.start + (slot * 8 + cell_index) * per_cell + offset',
                      'action_stream': 'SeedSequence([DQ_SALT, SET_TAG[set], 1000 + slot, cell_index, offset])',
                      'init': 'torch.manual_seed(SeedSequence([DQ_SALT, 50, index]))'},
            'per_cell': PER_CELL, 'units': units,
            'seed_sets_sha256': {s: hashlib.sha256(json.dumps(v).encode()).hexdigest() for s, v in all_seeds.items()},
            'counts': {s: len(v) for s, v in all_seeds.items()}}


FROZEN_FILES = ('scripts/gnss_qrl_runtime/dq_b1.py', 'scripts/gnss_qrl_runtime/dq_b1_analysis.py',
                'scripts/gnss_qrl_runtime/dq_instruments.py', 'scripts/gnss_qrl_runtime/dq_batches.py',
                'scripts/gnss_qrl_runtime/dq_circuits.py', 'scripts/gnss_qrl_runtime/dq_transplant.py',
                'scripts/gnss_qrl_runtime/dq_runtime.py', 'tests/gnss_qrl_dq/test_dq_b1.py',
                'docs/study/gnss_qrl_v1/frozen/DQ_B1_PROTOCOL_EN.md')


def write_seed_manifest(out):
    dq.claim_output(out, directory=True)
    manifest = seed_manifest()
    manifest['manifest_sha256'] = dq.canonical_sha256(manifest)
    path = out / 'b1_seed_manifest.json'
    dq.write_json(path, {'utc': dq.utc(), 'seed_manifest': manifest})
    return path, manifest['manifest_sha256']


def write_freeze(out, seed_manifest_path):
    require_d6_decisions()
    dq.watch_trees()
    dq.claim_output(out, directory=True)
    manifest = json.loads(Path(seed_manifest_path).read_text())['seed_manifest']
    record = {'tool': TOOL, 'utc': dq.utc(), 'protocol': 'docs/study/gnss_qrl_v1/frozen/DQ_B1_PROTOCOL_EN.md',
              'files': {f: dq.sha256_file(ROOT / f) for f in FROZEN_FILES},
              'seed_manifest': str(Path(seed_manifest_path).relative_to(ROOT)),
              'seed_manifest_sha256': manifest['manifest_sha256'],
              'copy_code_tree_sha256': dq.dq_tree()[0],
              'd6': {'result_sha256': D6_RESULT_SHA256, 'seal_sha256': D6_SEAL_SHA256,
                     'decisions': require_d6_decisions()},
              'units': 2 * UNITS_PER_CIRCUIT, 'statement': 'Freeze B1 package; nothing has run under it'}
    path = out / 'freeze.json'
    dq.write_json(path, record)
    return path, dq.sha256_file(path)


def launch(campaign, workers):
    """Run every unit under an APPROVED freeze (dq.require_open refuses every B1 namespace otherwise)."""
    import multiprocessing as mp
    require_d6_decisions()
    for ns in ('dq_fit', 'dq_oracle', 'dq_calibration', 'dq_heldout', 'dq_fresh_eval'):
        dq.require_open(ns)
    campaign = Path(campaign)
    units_dir = campaign / 'units'
    dq.claim_output(units_dir, directory=True)
    dq.append_tree_watch('b1 launch')
    tasks = [(c, i) for i in range(UNITS_PER_CIRCUIT) for c in CIRCUITS]       # paired circuits interleaved
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(_unit_task, tasks):
            dq.write_json(units_dir / f'{record["circuit"]}_{record["index"]:02d}.json',
                          json.loads(json.dumps(record, default=float)))
    dq.append_tree_watch('b1 done')


def _unit_task(task):
    return run_unit(*task)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sm = sub.add_parser('seed-manifest')
    sm.add_argument('--out', required=True, type=Path)
    fr = sub.add_parser('freeze')
    fr.add_argument('--out', required=True, type=Path)
    fr.add_argument('--seed-manifest', required=True, type=Path)
    la = sub.add_parser('launch')
    la.add_argument('--campaign', required=True, type=Path)
    la.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'seed-manifest':
        print(json.dumps(dict(zip(('path', 'manifest_sha256'), map(str, write_seed_manifest(args.out))))))
    elif args.command == 'freeze':
        print(json.dumps(dict(zip(('path', 'sha256'), map(str, write_freeze(args.out, args.seed_manifest))))))
    else:
        launch(args.campaign, args.workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
