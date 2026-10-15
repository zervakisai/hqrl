"""Exact HQ -> Fourier-twin transplant with a fail-closed parity certificate (the DQ protocol section 3).

    .venv/bin/python scripts/gnss_qrl_runtime/dq_transplant.py certify --checkpoint <E1 HQ .pt> [...] --out <new dir>

The twin copies the encoder, GRU, projection, readout and critic BITWISE from the source model and sets its surrogate
weights to C(theta)^T, solved on the tensor grid (dq_circuits.coefficients), bias 0. Optimiser state is never carried:
the twin is saved with optimizer None.

Certificate (all in float64; any failure raises TransplantParityError, nothing is written, nothing is retried):
    P1 solve residual  max|Phi_G C - Z_G(theta)|                                       <= 1e-12
    P2 off-grid parity on 4096 dq_development angle rows, twin basis vs the PennyLane circuit <= 1e-12
    P3 row parity on replayed decisions: max|logit diff| and max|log-prob diff| (feasible)  <= 1e-9   (certify_rows)
    P4 trajectory parity on dq_development episodes (dq_batches.trajectory_parity)          exact, <= 1e-9
    P5 every non-head tensor and the projection/readout torch.equal
    P6 off-support coefficients |C_k| <= 1e-12 for k outside S (an excursion is recorded, then refused)
    P7 physical range max|z| <= 1 + 1e-12 on the P2 rows (recorded, never clipped)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
from aerial_qrl.training.smdp_ppo import (RecurrentActorCritic, apply_action_mask, load_checkpoint,  # noqa: E402
                                          save_checkpoint)

TOOL = 'gnss-dq-transplant/1'
TOL = dict(P1=1e-12, P2=1e-12, P3=1e-9, P4=1e-9, P6=1e-12, P7=1e-12)
P2_ROWS = 4096
SUPPORT_THRESHOLD = 1e-10
SUPPORT_DRAWS = 256            # of each law: 0.1*N(0,1) (the initialiser's) and U[-pi, pi]
#: stream tags inside Study('dq') for the algebraic draws (never an episode seed)
STREAM_SUPPORT, STREAM_P2 = 99, 98


class TransplantParityError(dq.DQError):
    """The twin is not the same policy as its source: the unit is refused, nothing is written."""


def support(spec, draws=SUPPORT_DRAWS, threshold=SUPPORT_THRESHOLD):
    """S = {k : max_r |C_k(theta_r)| > threshold} over `draws` initialiser-law and `draws` uniform-law weights from the
    dq_development stream (DQ protocol section 4.2). Returns (bool mask over vec(C), record)."""
    g = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, STREAM_SUPPORT, qc.SPEC_INDEX[spec.name]))
    weights = [0.1 * torch.randn(*spec.weight_shape, generator=g, dtype=torch.float64) for _ in range(draws)]
    weights += [(torch.rand(*spec.weight_shape, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
                for _ in range(draws)]
    magnitude = torch.zeros(spec.features * spec.wires, dtype=torch.float64)
    for w in weights:
        magnitude = torch.maximum(magnitude, qc.coefficients(spec, w).reshape(-1).abs())
    mask = magnitude > threshold
    record = {'circuit': spec.name, 'size': int(mask.sum()), 'coordinates': int(mask.numel()), 'draws': 2 * draws,
              'threshold': threshold, 'min_on_support': float(magnitude[mask].min()),
              'max_off_support': float(magnitude[~mask].max()) if bool((~mask).any()) else 0.,
              'index_sha256': hashlib.sha256(np.flatnonzero(mask.numpy()).astype('<i8').tobytes()).hexdigest()}
    return mask, record


def _require_f64(model, what):
    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float64}:
        raise TransplantParityError(f'{what}: parity is certified in float64 only, got {sorted(map(str, dtypes))}')


def make_twin(model):
    """The Surrogate625 twin of a shipped-circuit HQ RecurrentActorCritic: (twin, C (625, 4))."""
    if model.config['actor_head'] != 'quantum':
        raise TransplantParityError(f'source actor head {model.config["actor_head"]!r} is not the shipped circuit')
    _require_f64(model, 'source')
    with torch.no_grad():
        C = qc.coefficients(qc.SHIPPED, model.actor.weights.detach())
    twin = RecurrentActorCritic(**dict(model.config, actor_head='surrogate625')).to(torch.float64)
    state = {k: v.clone() for k, v in model.state_dict().items() if k != 'actor.weights'}
    state['actor.surrogate.weight'] = C.T.contiguous().clone()
    state['actor.surrogate.bias'] = torch.zeros(qc.SHIPPED.wires, dtype=torch.float64)
    twin.load_state_dict(state, strict=True)
    twin.eval()
    return twin, C


def certify_static(model, twin, C, support_mask):
    """P1, P2, P5, P6, P7. Returns the record; raises TransplantParityError on any failure."""
    spec = qc.SHIPPED
    record = {}
    weights = model.actor.weights.detach()
    record['P1_solve_residual'] = qc.solve_residual(spec, weights, C)
    g = torch.Generator().manual_seed(dq.DQ_STUDY.stream(10, STREAM_P2))
    angles = (torch.rand(P2_ROWS, spec.wires, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    with torch.no_grad():
        reference = torch.stack(model.actor.circuit(angles, weights), dim=-1)      # the PennyLane head itself
        twin_z = twin.actor.surrogate(qc.basis(spec, angles))
    record['P2_offgrid_max_abs'] = float((twin_z - reference).abs().max())
    source_state, twin_state = model.state_dict(), twin.state_dict()
    shared = [k for k in source_state if k != 'actor.weights']
    unequal = [k for k in shared if not torch.equal(source_state[k], twin_state[k])]
    record['P5_shared_tensors'] = len(shared)
    record['P5_unequal'] = unequal
    off = C.reshape(-1)[~support_mask]
    record['P6_max_off_support'] = float(off.abs().max()) if off.numel() else 0.
    record['P7_max_abs_z'] = float(reference.abs().max())
    failures = []
    if not record['P1_solve_residual'] <= TOL['P1']:
        failures.append('P1')
    if not record['P2_offgrid_max_abs'] <= TOL['P2']:
        failures.append('P2')
    if unequal:
        failures.append('P5')
    if not record['P6_max_off_support'] <= TOL['P6']:
        failures.append('P6')
    if not record['P7_max_abs_z'] <= 1. + TOL['P7']:
        failures.append('P7')
    record['static_failures'] = failures
    if failures:
        raise TransplantParityError(f'static parity failed {failures}: {json.dumps(record)}')
    return record


def masked_logits(model, obs, state, mask):
    """Logits of single decisions (N rows) from stored (obs, GRU input state, mask), trunk replayed one step each."""
    with torch.no_grad():
        logits, _ = model.policy(obs.unsqueeze(1), state.unsqueeze(0), mask.unsqueeze(1))
    return logits.squeeze(1)


def certify_rows(model, twin, rows):
    """P3 on stored decisions: rows = dict(obs (N, d), state (N, h), mask (N, A) bool)."""
    a = masked_logits(model, rows['obs'], rows['state'], rows['mask'])
    b = masked_logits(twin, rows['obs'], rows['state'], rows['mask'])
    feasible = rows['mask']
    logit = float((a - b).abs()[feasible].max())
    lp = float((torch.log_softmax(a, -1) - torch.log_softmax(b, -1)).abs()[feasible].max())
    record = {'P3_rows': int(a.shape[0]), 'P3_max_abs_logit': logit, 'P3_max_abs_logprob': lp}
    if not (logit <= TOL['P3'] and lp <= TOL['P3']):
        raise TransplantParityError(f'row parity failed: {record}')
    return record


def transplant_checkpoint(path, support_mask):
    """Load an HQ checkpoint read-only, build and statically certify its twin. Returns (model, twin, C, record)."""
    path = Path(path)
    before = dq.sha256_file(path)
    model, _, payload = load_checkpoint(path)
    model.eval()
    twin, C = make_twin(model)
    record = {'tool': TOOL, 'source': str(path), 'source_sha256': before,
              'source_model_config': payload['model_config'],
              'twin_coefficients_sha256': hashlib.sha256(C.detach().numpy().astype('<f8').tobytes()).hexdigest(),
              'optimizer_state': None, **certify_static(model, twin, C, support_mask)}
    if dq.sha256_file(path) != before:
        raise dq.DQError(f'{path} changed while it was read')
    return model, twin, C, record


def save_twin(twin, record, out_path, test_root=None):
    """Write the twin (never overwriting; optimizer None) through the DQ write guard."""
    dq.vet_output(out_path, test_root=test_root)     # save_checkpoint then opens 'xb' itself
    save_checkpoint(out_path, twin, None, train_config={'tool': TOOL}, metadata={'transplant': record})
    return dq.sha256_file(out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    certify = sub.add_parser('certify', help='static certificate P1, P2, P5-P7 for E1 HQ development checkpoints')
    certify.add_argument('--checkpoint', nargs='+', required=True, type=Path)
    certify.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    dq.require_open('dq_development')
    dq.claim_output(args.out, directory=True)
    mask, support_record = support(qc.SHIPPED)
    results, refused = [], []
    for path in args.checkpoint:
        try:
            _, _, _, record = transplant_checkpoint(path, mask)
            results.append(record)
        except TransplantParityError as exc:
            refused.append({'source': str(path), 'error': str(exc)})
    dq.write_json(args.out / 'certificate.json', {'support': support_record, 'certified': results, 'refused': refused,
                                                  'tolerances': TOL})
    dq.write_json(args.out / 'provenance.json', dq.provenance('dq_development', sources=args.checkpoint))
    print(json.dumps({'certified': len(results), 'refused': len(refused), 'out': str(args.out)}))
    return 0 if not refused else 1


if __name__ == '__main__':
    raise SystemExit(main())
