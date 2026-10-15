"""D-6: the two-unit instrument pilot (development material), under the pre-run seal of review 35.

    .venv/bin/python scripts/gnss_qrl_runtime/dq_d6.py seal                       # writes development/d6/seal.json
    .venv/bin/python scripts/gnss_qrl_runtime/dq_d6.py run     --workers 10        # collection (verifies the seal)
    .venv/bin/python scripts/gnss_qrl_runtime/dq_d6.py analyze                     # decisions (verifies the seal)

Purpose (DQ protocol section 5, DQ protocol amendment, section 6, review 35): fix, BEFORE Freeze B1 and with deterministic rules
written here, the batch size per set, the damping delta, the numerical-rank threshold tau_rank and the classification
of the KL level 5e-2. Development held-out gains are DISPLAYED for estimator diagnosis only; no rule reads them.

Everything below the line "SEALED DESIGN" is bound by the seal (the file hash, the live idea_dq tree, and the seal's
own copy of the design). `run` and `analyze` refuse unless the seal verifies and the watched checkouts are unchanged.

Units         two E1 HQ development checkpoints of block 10 (untrained, update 0; final, update 56); stream units 60, 61
Sets          F (fit), O (FIT-k fit), K (KL calibration), H (held-out), each from namespace dq_development with its
              own disjoint realisation range; every one of the 8 frozen core cells; realisation OFFSET o = 0..31
Nesting       batch n in {64, 128, 256} episodes per set = offsets o < n/8 on every cell, so 64 is a prefix of 128 and
              128 a prefix of 256 for each set (the episode for (cell, offset) is the same in every prefix)
Order of the rules  tau_rank -> batch size -> delta -> 5e-2 classification; each reads only its own inputs, and
              delta and 5e-2 are decided at the chosen batch size (the one Freeze B1 will use).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
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

# ---------------------------------------------------------------- SEALED DESIGN ----------------------------------
TOOL = 'gnss-dq-d6/1'
D6_ROOT = dq.DEVELOPMENT_ROOT / 'd6'
E1 = ROOT / 'output' / 'study' / 'gnss_qrl_v1' / 'e1' / 'e1_campaign_v2'
UNITS = {
    'b10_untrained': {'stream_unit': 60,
                      'checkpoint': 'evaluation/quantum__lr3e-3_ent0__b10/run/quantum/checkpoints/update00000_decisions00000000.pt'},
    'b10_final': {'stream_unit': 61,
                  'checkpoint': 'evaluation/quantum__lr3e-3_ent0__b10/run/quantum/checkpoints/final_update00056_decisions00030181.pt'},
}
SETS = ('fit', 'oracle', 'calibration', 'heldout')
#: realisation = base + offset inside dq_development; ranges are disjoint from each other and from D-1..D-5 (0-21)
REALIZATION_BASE = {'fit': 100, 'oracle': 200, 'calibration': 300, 'heldout': 400}
OFFSETS = 32
CELLS = 8
BATCHES = (64, 128, 256)                       # episodes per set; n / CELLS offsets on every cell
KAPPAS = (1e-3, 1e-2, 5e-2)
PRIMARY_KAPPA, STRESS_KAPPA = 1e-2, 5e-2
TAU_CANDIDATES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
TAU_DEFAULT = 1e-6
DELTA_CANDIDATES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
DELTA_COND_MAX = 1e10
DELTA_RESIDUAL_MAX = 1e-10
TRANSFER_RANGE = (0.5, 2.0)                     # KL on H / KL on K for a step calibrated on K
TRANSFER_ARMS = ('SPAN', 'TAN', 'ORTH', 'FIT-k', 'RND-k#0', 'HQ-Fisher')
COST_CAP_CPU_H = 48.0                           # projected B1 collection + instruments for the planned units
PLANNED_UNITS = 72                              # 36 shipped + 36 held-out circuit initialisations (planning value)
STRESS_REL_ERROR_MAX = 0.20
SIGNAL_FLAG_COSINE = 0.1
RND_DRAWS = 20
RND_STREAM = 70                                 # Study('dq').stream(10, RND_STREAM, stream_unit, n, draw)
HELDOUT_SPECTRA_DRAWS = 16
HELDOUT_SPECTRA_STREAM = 71
DECISION_TABLE = {
    'tau_rank': ('Inputs: relative singular values sigma_j/sigma_1 of A(theta) restricted to S for (i) the 12 untrained '
                 'E1 evaluation-block HQ checkpoints, (ii) the two D-6 units, (iii) 16 held-out-circuit initialiser-law '
                 'draws (stream HELDOUT_SPECTRA_STREAM). For each candidate tau: a spectrum is AMBIGUOUS if any '
                 'relative value lies in [tau/100, 100 tau]; margin(spectrum) = min(log10(s_k/tau), log10(tau/s_{k+1})) '
                 'with k = #{s > tau} (s_{k+1} = 0 counts as +inf margin on that side). A candidate QUALIFIES if no '
                 'spectrum is ambiguous. Choose the qualifying candidate with the largest minimum margin; ties (equal '
                 'to 1e-12) go to TAU_DEFAULT if it is tied, else to the smaller tau. No qualifying candidate: '
                 'decision NONE (B1 cannot rank-match; stop and hand off). Held-out gain is not an input.'),
    'delta': ('Inputs, at the chosen tau and the chosen batch n, for each unit. (a) Numerics, for each subspace B in '
              '{tangent (k_u), SPAN, ORTH, FIT-k (fit on O_n), RND draw 0}: M = B^T F_n B, lambda = delta tr(M)/dim; '
              'cond(M + lambda I) from eigvalsh and the relative solve residual |(M + lambda I) x - B^T g| / |B^T g|. '
              '(b) KL transfer, for each arm in TRANSFER_ARMS: the step calibrated to kappa = 1e-2 on K_n must move '
              'the policy on the disjoint set H_n by a comparable amount, KL_H / KL_K in TRANSFER_RANGE; a non-ok '
              'calibration fails (b). This is a stability property of the local step (does the Fisher of K describe '
              'the step on other states?), not a gain. A candidate QUALIFIES if every cond <= DELTA_COND_MAX, every '
              'residual <= DELTA_RESIDUAL_MAX, every transfer ratio lies in TRANSFER_RANGE and nothing raised. Choose '
              'the SMALLEST qualifying delta (least damping bias). No qualifying candidate: decision NONE (stop and '
              'hand off). Held-out gain is not an input.'),
    'batch': ('Inputs: the measured single-process collection seconds per episode (sum of per-episode wall time of all '
              'D-6 episodes / their count) and the measured instrument seconds per unit (the larger of the two units\' '
              'wall time for one full instrument pass at n = 256: Fisher on F, O, K; FIT; every arm; calibration at '
              'every kappa; delta fixed at 1e-4 for this timing only, since the cost does not depend on delta). '
              'projected_cpu_h(n) = PLANNED_UNITS * (4 * n * s_per_episode + instrument_s) / 3600. Choose the LARGEST '
              'n in BATCHES with projected_cpu_h(n) <= COST_CAP_CPU_H; the same n for F, O, K and H. A missing '
              'measurement: decision NONE. No feasible n: decision NONE. Reported, not used: the split-half Fisher-'
              'metric cosine of the SPAN natural direction (chosen delta) fitted on the even and odd episodes of F_n; '
              'if it is below SIGNAL_FLAG_COSINE for BOTH units at the chosen n, the flag signal_insufficient is raised '
              '(kill criterion K5 of the plan). Held-out gain is not an input.'),
    'stress_5e-2': ('Inputs, at the chosen delta and the chosen n, for both units and the arms SPAN, TAN, ORTH, FIT-k, '
                    'HQ-Fisher, HQ-plain and each of the RND_DRAWS RND-k draws: the relative error '
                    '|0.5 alpha^2 d^T F_K d - D(alpha)| / D(alpha) at the calibrated alpha for kappa = 5e-2 (for the '
                    'manifold arms d and F are in theta). 5e-2 is LOCAL-ELIGIBLE iff every error <= '
                    'STRESS_REL_ERROR_MAX and every calibration status is ok; a missing value or a non-ok status '
                    'counts as a failure; otherwise it stays STRESS. 1e-2 stays primary whatever happens (review 34). '
                    'Held-out gain is not an input.'),
    'failures': ('Any scheduled episode that fails, any seal or tree-watch mismatch, or any refused batch makes the run '
                 'INCOMPLETE: analyze then writes the completeness report and NO decisions; nothing is re-run or '
                 'chosen by hand.'),
}
OUTPUT_SCHEMA = {
    'collection': 'development/d6/collection/<unit>__<set>.pt (raw rows) + collection.json (per-episode records, failures, timing)',
    'results': 'development/d6/analysis/d6_results.json: seal_sha256, completeness, rank_rule, delta_rule, batch_rule, '
               'stress_rule, flags, displayed_diagnostics (gains labelled development), failures, statement',
}
# ---------------------------------------------------------------- END SEALED DESIGN -------------------------------

DESIGN_KEYS = ('TOOL', 'UNITS', 'SETS', 'REALIZATION_BASE', 'OFFSETS', 'CELLS', 'BATCHES', 'KAPPAS', 'PRIMARY_KAPPA',
               'STRESS_KAPPA', 'TAU_CANDIDATES', 'TAU_DEFAULT', 'DELTA_CANDIDATES', 'DELTA_COND_MAX',
               'DELTA_RESIDUAL_MAX', 'TRANSFER_RANGE', 'TRANSFER_ARMS', 'COST_CAP_CPU_H', 'PLANNED_UNITS', 'STRESS_REL_ERROR_MAX', 'SIGNAL_FLAG_COSINE',
               'RND_DRAWS', 'RND_STREAM', 'HELDOUT_SPECTRA_DRAWS', 'HELDOUT_SPECTRA_STREAM', 'DECISION_TABLE',
               'OUTPUT_SCHEMA')
SEALED_FILES = ('scripts/gnss_qrl_runtime/dq_d6.py', 'scripts/gnss_qrl_runtime/dq_instruments.py',
                'scripts/gnss_qrl_runtime/dq_batches.py', 'scripts/gnss_qrl_runtime/dq_circuits.py',
                'scripts/gnss_qrl_runtime/dq_transplant.py', 'scripts/gnss_qrl_runtime/dq_runtime.py',
                'tests/gnss_qrl_dq/test_dq_d6.py')


class D6Error(dq.DQError):
    """A D-6 refusal: seal mismatch, incomplete evidence, overwrite."""


def design():
    return json.loads(json.dumps({k: globals()[k] for k in DESIGN_KEYS}, default=list))


def schedule():
    """Every scheduled episode: (unit, set, cell_index, offset, realisation), in collection order."""
    out = []
    for unit in UNITS:
        for s in SETS:
            for offset in range(OFFSETS):
                for cell in range(CELLS):
                    out.append((unit, s, cell, offset, REALIZATION_BASE[s] + offset))
    return out


def in_prefix(offset, n):
    return offset < n // CELLS


# -- the seal -----------------------------------------------------------------------------------------------------

def seal_payload():
    runtime = dq.cell_runtime()
    if len(runtime.cells) != CELLS:
        raise D6Error(f'expected {CELLS} core cells, found {len(runtime.cells)}')
    checkpoints = {u: {'path': str(E1 / spec['checkpoint']), 'sha256': dq.sha256_file(E1 / spec['checkpoint']),
                       'stream_unit': spec['stream_unit']} for u, spec in UNITS.items()}
    return {'tool': TOOL, 'statement': dq.DEVELOPMENT_STATEMENT, 'review': 'review 35',
            'dq_tree': dq.dq_tree()[0], 'files': {f: dq.sha256_file(ROOT / f) for f in SEALED_FILES},
            'checkpoints': checkpoints, 'namespace': 'dq_development',
            'stream_rule': "SeedSequence([DQ_SALT, SET_TAG['development'] = 0, stream_unit, cell_index, realisation])",
            'disturbance_rule': 'p3_design.seed_for(dq_development, seed_cell_index, realisation)',
            'cells': [c['cell_id'] for c in runtime.cells],
            'schedule_count': len(schedule()),
            'schedule_sha256': hashlib.sha256(json.dumps(schedule()).encode()).hexdigest(),
            'nesting_rule': 'batch n uses offsets o < n / 8 on each of the 8 cells: 64 is a prefix of 128, 128 of 256',
            'design': design()}


def seal_path():
    return D6_ROOT / 'seal.json'


def write_seal():
    dq.require_open('dq_development')
    dq.watch_trees()
    payload = seal_payload()
    payload['seal_sha256'] = dq.canonical_sha256(payload)
    dq.write_json(seal_path(), payload)
    dq.append_tree_watch('d6 seal written')
    return payload['seal_sha256'], dq.sha256_file(seal_path())


def verify_seal():
    """The seal on disk equals a fresh recomputation (tree, files, checkpoints, schedule, design); else D6Error."""
    if not seal_path().exists():
        raise D6Error('no D-6 seal exists; write it before any D-6 episode')
    stored = json.loads(seal_path().read_text())
    claimed = stored.pop('seal_sha256')
    if dq.canonical_sha256(stored) != claimed:
        raise D6Error('the seal file does not match its own hash')
    fresh = seal_payload()
    differ = [k for k in fresh if k != 'utc' and fresh[k] != stored.get(k)]
    if differ:
        raise D6Error(f'the live D-6 state differs from the seal in {differ}')
    dq.watch_trees()
    return claimed


# -- collection ---------------------------------------------------------------------------------------------------

_WORKER = {}


def _worker_init():
    torch.set_num_threads(1)
    _WORKER['runtime'] = dq.cell_runtime()
    _WORKER['models'] = {}


def _worker_model(unit):
    if unit not in _WORKER['models']:
        from aerial_qrl.training.smdp_ppo import load_checkpoint
        model, _, _ = load_checkpoint(E1 / UNITS[unit]['checkpoint'])
        _WORKER['models'][unit] = model.eval()
    return _WORKER['models'][unit]


def _collect_one(task):
    unit, set_name, cell_index, offset, realization = task
    runtime = _WORKER['runtime']
    cell = runtime.cells[cell_index]
    try:
        run = db.run_episode(_worker_model(unit), runtime, 'development', UNITS[unit]['stream_unit'], cell, realization,
                             f'd6-{unit}-{set_name}')
        rec = run['record']
        if rec['infrastructure'] or rec['invalid_rows']:
            return {'task': task, 'ok': False, 'error': f'infrastructure/invalid: {rec["termination_reason"]}'}
        rows = [{'obs': np.asarray(r['obs'], np.float64), 'state': np.asarray(r['state'], np.float64),
                 'mask': np.asarray(r['mask'], bool), 'action': int(r['action']), 'log_prob': float(r['log_prob']),
                 'reward': float(r['reward'])} for r in run['transitions'].rows if r['valid']]
        return {'task': task, 'ok': True, 'rows': rows, 'wall_s': run['wall_s'],
                'record': {k: rec[k] for k in ('cell_id', 'seed', 'J', 'termination_reason', 'decisions', 'valid_rows')}}
    except Exception as exc:                                                    # reported, never retried
        return {'task': task, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}


def run(workers):
    torch.set_num_threads(1)
    seal = verify_seal()
    collection = D6_ROOT / 'collection'
    dq.claim_output(collection, directory=True)
    dq.append_tree_watch('d6 collection start')
    tasks = schedule()
    started = time.perf_counter()
    with mp.get_context('fork').Pool(workers, initializer=_worker_init) as pool:
        results = pool.map(_collect_one, tasks, chunksize=1)
    wall = time.perf_counter() - started
    failures = [{'task': r['task'], 'error': r['error']} for r in results if not r['ok']]
    by_key = {}
    for r in results:
        if r['ok']:
            by_key.setdefault((r['task'][0], r['task'][1]), []).append(r)
    for (unit, set_name), items in sorted(by_key.items()):
        payload = {'episodes': [{'cell_index': it['task'][2], 'offset': it['task'][3], 'realization': it['task'][4],
                                 'record': it['record'], 'rows': it['rows']} for it in items]}
        with dq.claim_output(collection / f'{unit}__{set_name}.pt') as handle:
            torch.save(payload, handle)
    episodes_ok = [r for r in results if r['ok']]
    record = {'seal_sha256': seal, 'scheduled': len(tasks), 'completed': len(episodes_ok), 'failures': failures,
              'complete': not failures and len(episodes_ok) == len(tasks), 'workers': workers, 'wall_s': wall,
              'sum_episode_wall_s': sum(r['wall_s'] for r in episodes_ok),
              's_per_episode': (sum(r['wall_s'] for r in episodes_ok) / len(episodes_ok)) if episodes_ok else None,
              'episodes': [{'task': r['task'], 'ok': r['ok'], **({'record': r['record'], 'wall_s': r['wall_s']}
                                                                 if r['ok'] else {'error': r['error']})}
                           for r in results],
              'statement': dq.DEVELOPMENT_STATEMENT}
    dq.write_json(collection / 'collection.json', record)
    dq.append_tree_watch('d6 collection done')
    print(json.dumps({k: record[k] for k in ('scheduled', 'completed', 'complete', 'wall_s', 's_per_episode')}))
    return record


# -- analysis: rules ----------------------------------------------------------------------------------------------

def spectrum_margin(relative, tau):
    k = int((relative > tau).sum())
    upper = math.log10(float(relative[k - 1]) / tau) if k > 0 else math.inf
    lower = math.log10(tau / float(relative[k])) if k < len(relative) and float(relative[k]) > 0 else math.inf
    ambiguous = bool(((relative >= tau / 100.) & (relative <= tau * 100.)).any())
    return k, min(upper, lower), ambiguous


def rank_rule(spectra):
    """spectra: {name: relative singular values (1-d tensor, descending)}. Returns the decision record."""
    table = {}
    for tau in TAU_CANDIDATES:
        rows = {name: spectrum_margin(rel, tau) for name, rel in spectra.items()}
        table[tau] = {'qualifies': not any(r[2] for r in rows.values()),
                      'min_margin': min(r[1] for r in rows.values()),
                      'ambiguous': sorted(n for n, r in rows.items() if r[2]),
                      'ranks': {n: r[0] for n, r in rows.items()}}
    qualifying = [t for t in TAU_CANDIDATES if table[t]['qualifies']]
    if not qualifying:
        return {'decision': None, 'table': _keys(table), 'reason': 'no candidate qualifies'}
    best = max(table[t]['min_margin'] for t in qualifying)
    tied = [t for t in qualifying if abs(table[t]['min_margin'] - best) <= 1e-12]
    choice = TAU_DEFAULT if TAU_DEFAULT in tied else min(tied)
    return {'decision': choice, 'table': _keys(table), 'tied': tied}


def delta_rule(checks):
    """checks: {delta: list of items}; a numeric item has 'cond', 'residual', 'error'; a transfer item has 'transfer'
    (KL_H / KL_K or None) and 'status'. Items over units x subspaces/arms at the chosen batch."""
    table = {}
    for delta in DELTA_CANDIDATES:
        items = checks.get(delta, [])
        numeric = [c for c in items if 'transfer' not in c]
        transfer = [c for c in items if 'transfer' in c]
        failed = [c for c in numeric if c.get('error') is not None]
        worst_cond = max((c['cond'] for c in numeric if c.get('error') is None), default=math.inf)
        worst_res = max((c['residual'] for c in numeric if c.get('error') is None), default=math.inf)
        bad_transfer = [c for c in transfer if c['status'] != 'ok' or c['transfer'] is None
                        or not TRANSFER_RANGE[0] <= c['transfer'] <= TRANSFER_RANGE[1]]
        ratios = [c['transfer'] for c in transfer if c['transfer'] is not None]
        table[delta] = {'numeric_items': len(numeric), 'transfer_items': len(transfer), 'errors': len(failed),
                        'worst_cond': worst_cond, 'worst_residual': worst_res, 'transfer_failures': bad_transfer,
                        'transfer_min': min(ratios, default=None), 'transfer_max': max(ratios, default=None),
                        'qualifies': bool(numeric) and bool(transfer) and not failed and not bad_transfer
                        and worst_cond <= DELTA_COND_MAX and worst_res <= DELTA_RESIDUAL_MAX}
    qualifying = [d for d in DELTA_CANDIDATES if table[d]['qualifies']]
    return {'decision': min(qualifying) if qualifying else None, 'table': _keys(table),
            **({} if qualifying else {'reason': 'no candidate qualifies'})}


def batch_rule(s_per_episode, instrument_s, split_half):
    """split_half: {unit: {n: cosine}} (reported; only the flag reads it)."""
    if s_per_episode is None or instrument_s is None:
        return {'decision': None, 'reason': 'a cost measurement is missing'}
    projected = {n: PLANNED_UNITS * (4 * n * s_per_episode + instrument_s) / 3600. for n in BATCHES}
    feasible = [n for n in BATCHES if projected[n] <= COST_CAP_CPU_H]
    if not feasible:
        return {'decision': None, 'projected_cpu_h': _keys(projected), 'reason': 'no feasible batch'}
    n = max(feasible)
    flag = all(split_half.get(u, {}).get(n, -math.inf) < SIGNAL_FLAG_COSINE for u in UNITS)
    return {'decision': n, 'projected_cpu_h': _keys(projected), 'split_half_cosine': split_half,
            'signal_insufficient': flag}


def stress_rule(errors):
    """errors: list of {'unit', 'arm', 'status', 'rel_error'} at kappa 5e-2, n 256, chosen delta."""
    if not errors:
        return {'decision': 'stress', 'reason': 'no inputs', 'failures': []}
    failures = [e for e in errors if e['status'] != 'ok' or e['rel_error'] is None
                or not e['rel_error'] <= STRESS_REL_ERROR_MAX]
    worst = max((e['rel_error'] for e in errors if e['rel_error'] is not None), default=None)
    return {'decision': 'local_eligible' if not failures else 'stress', 'inputs': len(errors),
            'failures': failures, 'worst_rel_error': worst}


def _keys(table):
    return {repr(k): v for k, v in table.items()}


# -- analysis: per-unit computations --------------------------------------------------------------------------------

def load_set(unit, set_name):
    path = D6_ROOT / 'collection' / f'{unit}__{set_name}.pt'
    return torch.load(path, weights_only=False)['episodes']


def prefix_raw(episodes, n, cells):
    chosen = sorted((e for e in episodes if in_prefix(e['offset'], n)), key=lambda e: (e['offset'], e['cell_index']))
    if len(chosen) != n:
        raise D6Error(f'prefix {n}: {len(chosen)} episodes')
    obs, state, mask, action, logp, reward, episode, cell = [], [], [], [], [], [], [], []
    for i, e in enumerate(chosen):
        for r in e['rows']:
            obs.append(r['obs']); state.append(r['state']); mask.append(r['mask']); action.append(r['action'])
            logp.append(r['log_prob']); reward.append(r['reward']); episode.append(i); cell.append(cells[e['cell_index']])
    return {'obs': torch.as_tensor(np.stack(obs)), 'state': torch.as_tensor(np.stack(state)),
            'mask': torch.as_tensor(np.stack(mask)), 'action': torch.as_tensor(action),
            'log_prob': torch.as_tensor(logp, dtype=torch.float64), 'reward': np.asarray(reward),
            'episode': torch.as_tensor(episode), 'cell': np.asarray(cell)}


def unit_rows(model, episodes_by_set, n, cells):
    out = {}
    for s in SETS:
        raw = prefix_raw(episodes_by_set[s], n, cells)
        features, _ = db.replay(model, raw)
        out[s], _ = db.head_rows(model, raw, features)
    return out


def split_half_cosine(rows, c, support, delta):
    F = ins.fisher(rows, c, support)
    halves = []
    for parity in (0, 1):
        keep = (rows.episode % 2) == parity
        sub = ins.Rows(phi=rows.phi[keep], W=rows.W, b=rows.b, mask=rows.mask[keep], action=rows.action[keep],
                       advantage=rows.advantage[keep], episode=rows.episode[keep])
        d, _ = ins.span_step(ins.gradient(sub, c, support), ins.fisher(sub, c, support), delta)
        halves.append(d)
    a, b = halves
    return float(a @ F @ b / math.sqrt(float(a @ F @ a) * float(b @ F @ b)))


def subspace_checks(B, g, F, delta):
    M = B.T @ F @ B
    M = 0.5 * (M + M.T)
    lam = delta * float(torch.trace(M)) / M.shape[0]
    K = M + lam * torch.eye(M.shape[0], dtype=torch.float64)
    ev = torch.linalg.eigvalsh(K)
    rhs = B.T @ g
    x = torch.linalg.solve(K, rhs)
    return {'cond': float(ev[-1] / ev[0]) if float(ev[0]) > 0 else math.inf,
            'residual': float((K @ x - rhs).norm() / rhs.norm()), 'error': None}


def arm_rays(model, rows_F, rows_O, c, support, delta, k, rnd_bases):
    """{arm: (ray alpha -> full vec(C), curvature d^T F_K d function of rows_K)} at the chosen delta."""
    weights = model.actor.weights.detach().clone()
    A = qc.jacobian(qc.SHIPPED, weights)[support]
    g, F = ins.gradient(rows_F, c, support), ins.fisher(rows_F, c, support)
    T = ins.tangent_basis(A, k)
    rays = {}

    def linear(d):
        return (lambda a: ins.full(c[support] + a * d, support, c)), ('c', d)

    rays['SPAN'] = linear(ins.span_step(g, F, delta)[0])
    rays['TAN'] = linear(A @ ins.theta_natural(*ins.theta_quantities(A, g, F), delta)[0])
    rays['ORTH'] = linear(ins.natural_step(ins.orthogonal_complement(T), g, F, delta)[0])
    U, _ = ins.episode_contributions(rows_O, c, support)
    F_O = ins.fisher(rows_O, c, support)
    Q, _, fit_record = ins.fit_subspace(U, F_O, k, delta)
    rays['FIT-k'] = linear(ins.natural_step(Q, g, F, delta)[0])
    B1, _ = ins.fit_one(U, F_O, delta)
    rays['FIT-1'] = linear(ins.natural_step(B1, g, F, delta)[0])
    for i, B in enumerate(rnd_bases):
        rays[f'RND-k#{i}'] = linear(ins.natural_step(B, g, F, delta)[0])
    g_t, F_t = ins.theta_quantities(A, g, F)
    for name, dt in (('HQ-Fisher', ins.theta_natural(g_t, F_t, delta)[0]), ('HQ-plain', g_t),
                     ('HQ-Adam-dir', g_t / (g_t.abs() + 1e-8))):
        rays[name] = ((lambda a, dt=dt: qc.coefficients(qc.SHIPPED, weights + a * dt.reshape(2, 4, 2)).reshape(-1)),
                      ('theta', dt))
    return rays, A, fit_record


def curvature(kind_d, rows_K, c, support, A):
    kind, d = kind_d
    F_K = ins.fisher(rows_K, c, support)
    if kind == 'c':
        return float(d @ F_K @ d), F_K
    F_t = A.T @ F_K @ A
    return float(d @ F_t @ d), F_K


def jackknife_se(rows_H, c_old, c_new):
    """Episode-level jackknife SE of the held-out gain (displayed only)."""
    idx = torch.arange(rows_H.n)
    ratio = (ins.log_policy(rows_H, c_new)[idx, rows_H.action] - ins.log_policy(rows_H, c_old)[idx, rows_H.action]).exp() - 1.
    terms = ratio * rows_H.advantage
    episodes = torch.unique(rows_H.episode)
    loo = torch.stack([terms[rows_H.episode != e].mean() for e in episodes])
    m = len(episodes)
    return float(math.sqrt((m - 1) / m * float(((loo - loo.mean()) ** 2).sum())))


def instrument_pass(model, pu, n, support, delta, diagnostics=False):
    """Every arm at batch n: directions, calibration on K at every kappa; with diagnostics, the displayed development
    gains, held-out KL, jackknife SE and rho. Returns (record, stress inputs at 5e-2)."""
    rows, c, k, A = pu['rows'][n], pu['c'], pu['k'], pu['A']
    rnd = [ins.haar_subspace(int(support.sum()), k, torch.Generator().manual_seed(
        dq.DQ_STUDY.stream(10, RND_STREAM, pu['stream_unit'], n, i))) for i in range(RND_DRAWS)]
    rays, A, fit_record = arm_rays(model, rows['fit'], rows['oracle'], c, support, delta, k, rnd)
    F_K = ins.fisher(rows['calibration'], c, support)
    F_Kt = A.T @ F_K @ A
    arms, stress = {}, []
    for arm, (ray, (kind, d)) in rays.items():
        curv = float(d @ (F_K if kind == 'c' else F_Kt) @ d)
        arms[arm] = {}
        for kappa in KAPPAS:
            cal = ins.calibrate_kl(ray, c, rows['calibration'], curv, kappa)
            entry = {'status': cal.status, 'alpha': cal.alpha, 'kl_K': cal.kl,
                     'rel_error': (abs(0.5 * cal.alpha ** 2 * curv - cal.kl) / cal.kl) if cal.status == 'ok' else None}
            if diagnostics and cal.status == 'ok':
                c_new = ray(cal.alpha)
                entry['heldout_gain_DEVELOPMENT'] = ins.heldout_gain(rows['heldout'], c, c_new)
                entry['heldout_kl'] = ins.mean_kl(rows['heldout'], c, c_new)
                if arm in ('SPAN', 'HQ-Fisher', 'RND-k#0') and kappa == PRIMARY_KAPPA:
                    entry['heldout_gain_jackknife_se'] = jackknife_se(rows['heldout'], c, c_new)
            arms[arm][repr(kappa)] = entry
            if kappa == STRESS_KAPPA and arm not in ('FIT-1', 'HQ-Adam-dir'):
                stress.append({'arm': arm, 'status': cal.status, 'rel_error': entry['rel_error']})
    record = {'arms': arms, 'fit_record': fit_record, 'rows': {s: rows[s].n for s in SETS}}
    if diagnostics:
        g, F = ins.gradient(rows['fit'], c, support), ins.fisher(rows['fit'], c, support)
        q_tan = ins.accessible(ins.tangent_basis(A, k), g, F, delta)
        q_span = float(g @ ins.span_step(g, F, delta)[0])
        record['rho_hat_DEVELOPMENT'] = q_tan / q_span
    return record, stress


def analyze():
    torch.set_num_threads(1)
    seal = verify_seal()
    collection = json.loads((D6_ROOT / 'collection' / 'collection.json').read_text())
    out = D6_ROOT / 'analysis'
    dq.claim_output(out, directory=True)
    dq.append_tree_watch('d6 analysis start')
    result = {'seal_sha256': seal, 'statement': dq.DEVELOPMENT_STATEMENT,
              'completeness': {k: collection[k] for k in ('scheduled', 'completed', 'complete', 'failures')}}
    if not collection['complete'] or collection['seal_sha256'] != seal:
        result['decisions'] = None
        result['reason'] = 'INCOMPLETE or seal mismatch: no decision is made (DECISION_TABLE failures)'
        dq.write_json(out / 'd6_results.json', result)
        return result
    runtime = dq.cell_runtime()
    cells = [c['cell_id'] for c in runtime.cells]
    support, _ = tp.support(qc.SHIPPED)
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    models = {u: load_checkpoint(E1 / UNITS[u]['checkpoint'])[0].eval() for u in UNITS}

    # rule 1: tau_rank (algebra only)
    spectra = {}
    for b in range(10, 22):
        m, _, _ = load_checkpoint(E1 / f'evaluation/quantum__lr3e-3_ent0__b{b}/run/quantum/checkpoints/update00000_decisions00000000.pt')
        s = torch.linalg.svdvals(qc.jacobian(qc.SHIPPED, m.actor.weights.detach())[support])
        spectra[f'e1_b{b}_untrained'] = s / s[0]
    for u, m in models.items():
        s = torch.linalg.svdvals(qc.jacobian(qc.SHIPPED, m.actor.weights.detach())[support])
        spectra[f'unit_{u}'] = s / s[0]
    h_support, _ = tp.support(qc.HELDOUT)
    gen = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, HELDOUT_SPECTRA_STREAM))
    for i in range(HELDOUT_SPECTRA_DRAWS):
        s = torch.linalg.svdvals(qc.jacobian(qc.HELDOUT, 0.1 * torch.randn(3, 4, 2, generator=gen, dtype=torch.float64))[h_support])
        spectra[f'heldout_draw{i}'] = s / s[0]
    result['rank_rule'] = rank_rule(spectra)
    tau = result['rank_rule']['decision']
    if tau is None:
        result['decisions'] = None
        dq.write_json(out / 'd6_results.json', result)
        return result

    # per-unit data: rows for every nested batch
    episodes = {u: {s: load_set(u, s) for s in SETS} for u in UNITS}
    per_unit = {}
    for u, model in models.items():
        c = qc.coefficients(qc.SHIPPED, model.actor.weights.detach()).reshape(-1)
        A = qc.jacobian(qc.SHIPPED, model.actor.weights.detach())[support]
        k, _, _ = ins.chart_rank(A, tau)
        per_unit[u] = {'k': k, 'c': c, 'A': A, 'stream_unit': UNITS[u]['stream_unit'], 'rows': {n: unit_rows(model, episodes[u], n, cells) for n in BATCHES}}

    # rule 2: batch size (cost only); the instrument pass is timed at n = 256 with delta 1e-4 (cost is delta-free)
    instrument_s = []
    for u, model in models.items():
        t0 = time.perf_counter()
        instrument_pass(model, per_unit[u], max(BATCHES), support, 1e-4)
        instrument_s.append(time.perf_counter() - t0)
    result['batch_rule'] = batch_rule(collection['s_per_episode'], max(instrument_s), {})
    n = result['batch_rule']['decision']
    result['instrument_s_per_unit_at_256'] = instrument_s
    if n is None:
        result['decisions'] = None
        dq.write_json(out / 'd6_results.json', json.loads(json.dumps(result, default=float)))
        return result

    # rule 3: delta at the chosen n (numerics + KL transfer)
    delta_checks = {d: [] for d in DELTA_CANDIDATES}
    for u, model in models.items():
        pu = per_unit[u]
        rows, c, k, A = pu['rows'][n], pu['c'], pu['k'], pu['A']
        g, F = ins.gradient(rows['fit'], c, support), ins.fisher(rows['fit'], c, support)
        U, _ = ins.episode_contributions(rows['oracle'], c, support)
        F_O = ins.fisher(rows['oracle'], c, support)
        T = ins.tangent_basis(A, k)
        rnd0 = ins.haar_subspace(int(support.sum()), k, torch.Generator().manual_seed(
            dq.DQ_STUDY.stream(10, RND_STREAM, UNITS[u]['stream_unit'], n, 0)))
        for delta in DELTA_CANDIDATES:
            try:
                Q, _, _ = ins.fit_subspace(U, F_O, k, delta)
                for name, B in (('tangent', T), ('ORTH', ins.orthogonal_complement(T)), ('FIT-k', Q), ('RND0', rnd0)):
                    delta_checks[delta].append(dict(subspace_checks(B, g, F, delta), unit=u, subspace=name))
                lam = delta * float(torch.trace(F)) / F.shape[0]
                K = F + lam * torch.eye(F.shape[0], dtype=torch.float64)
                ev = torch.linalg.eigvalsh(K)
                x = torch.linalg.solve(K, g)
                delta_checks[delta].append({'unit': u, 'subspace': 'SPAN', 'error': None,
                                            'cond': float(ev[-1] / ev[0]) if float(ev[0]) > 0 else math.inf,
                                            'residual': float((K @ x - g).norm() / g.norm())})
                rays, _, _ = arm_rays(model, rows['fit'], rows['oracle'], c, support, delta, k, [rnd0])
                F_K = ins.fisher(rows['calibration'], c, support)
                for arm in TRANSFER_ARMS:
                    ray, (kind, d) = rays[arm]
                    curv = float(d @ (F_K if kind == 'c' else A.T @ F_K @ A) @ d)
                    cal = ins.calibrate_kl(ray, c, rows['calibration'], curv, PRIMARY_KAPPA)
                    ratio = ins.mean_kl(rows['heldout'], c, ray(cal.alpha)) / cal.kl if cal.status == 'ok' else None
                    delta_checks[delta].append({'unit': u, 'arm': arm, 'status': cal.status, 'transfer': ratio})
            except Exception as exc:                                              # disqualifies the candidate
                delta_checks[delta].append({'unit': u, 'error': f'{type(exc).__name__}: {exc}'})
    result['delta_rule'] = delta_rule(delta_checks)
    delta = result['delta_rule']['decision']
    if delta is None:
        result['decisions'] = None
        dq.write_json(out / 'd6_results.json', json.loads(json.dumps(result, default=float)))
        return result

    # rule 4 inputs and the displayed diagnostics, at the chosen delta, for every batch (rule 4 reads n only)
    displayed, stress_inputs, split_half = {}, [], {}
    for u, model in models.items():
        pu = per_unit[u]
        displayed[u], split_half[u] = {}, {}
        for nb in BATCHES:
            record, stress = instrument_pass(model, pu, nb, support, delta, diagnostics=True)
            split_half[u][nb] = split_half_cosine(pu['rows'][nb]['fit'], pu['c'], support, delta)
            displayed[u][nb] = record
            if nb == n:
                stress_inputs += [dict(e, unit=u) for e in stress]
    result['batch_rule']['split_half_cosine'] = split_half
    result['batch_rule']['signal_insufficient'] = all(split_half[u][n] < SIGNAL_FLAG_COSINE for u in UNITS)
    result['stress_rule'] = stress_rule(stress_inputs)
    result['decisions'] = {'tau_rank': tau, 'delta': delta, 'batch_per_set': n,
                           'kappa_5e-2': result['stress_rule']['decision'], 'kappa_primary': PRIMARY_KAPPA}
    result['flags'] = {'signal_insufficient': result['batch_rule'].get('signal_insufficient')}
    result['displayed_diagnostics'] = {'label': 'DEVELOPMENT held-out gains and rho: estimator diagnosis only; never '
                                                'validation observations; no rule reads them', 'units': displayed}
    dq.write_json(out / 'd6_results.json', json.loads(json.dumps(result, default=float)))
    dq.write_json(out / 'provenance.json', dq.provenance('dq_development', sources=[E1 / s['checkpoint'] for s in UNITS.values()]))
    dq.append_tree_watch('d6 analysis done')
    print(json.dumps(result['decisions']))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('seal')
    r = sub.add_parser('run')
    r.add_argument('--workers', type=int, default=10)
    sub.add_parser('analyze')
    args = parser.parse_args(argv)
    if args.command == 'seal':
        print(json.dumps(dict(zip(('seal_sha256', 'file_sha256'), write_seal()))))
    elif args.command == 'run':
        run(args.workers)
    else:
        analyze()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
