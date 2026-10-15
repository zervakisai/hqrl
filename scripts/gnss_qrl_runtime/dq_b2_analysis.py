"""Freeze B2 analysis: integrity proof of every run, AUC re-derived from raw events, the two-test primary family.

    .venv/bin/python scripts/gnss_qrl_runtime/dq_b2_analysis.py --campaign <abs dir> --freeze <abs freeze.json>
        --freeze-sha256 <sha> --out <abs new dir>

review 40 (B40-1, B40-2, B40-5). The analysis PROVES its inputs before it uses them:

IDENTITY      the campaign's records/ holds exactly the manifest's 108 `<arm>__b<block>.json` files: no extra,
              missing, duplicate or misnamed record. Each record's arm, role, block and study match its name; its
              config is IDENTICAL to dq_b2.config_for(arm, block, STUDY_B2, freeze_sha256) (canonical hash); its
              config_sha256, protocol_freeze_sha256 and code_tree_sha256 equal the freeze's.
RAW EVENTS    for every run whose record is `completed`: metrics.jsonl has exactly one `arm_start` (first) and one
              `arm_end` (last); update indices are 1..U without gaps; the evaluation points are the frozen grid exactly
              once, in order; every sampled weighted J is finite; the AUC is RECOMPUTED with p4_train.learning_curve_auc
              and must equal, exactly, the arm_end summary's AUC and the parent record's; status, decisions, updates
              and episodes agree across record, summary and arm_end; exactly one final checkpoint.
TWIN          a completed S625-twin run enters B-H2 only with a present, failure-free static certificate and a P4
              trajectory parity <= 1e-9.
FREEZE        before any record is read: p4_train.verify_freeze(freeze, sha) (members, seed manifest, LIVE tree);
              the campaign must be exactly the manifest's; launch.json must name this freeze, hash and frozen tree;
              the manifest must list 108 unique run identities (review 41, R41-2).
OUTCOME PROOF a non-completed status must agree with what exists on disk (review 41, R41-1): an outcome may never
              coexist with a completed arm_end, a final checkpoint or a completed summary; refused / parity_refused
              runs have no run directory; an algorithmic failure has its arm_end with that status and no final
              checkpoint; an infrastructure failure has no arm_end and no final checkpoint; every outcome carries its
              error text.
Any failure of FREEZE, IDENTITY, RAW EVENTS, OUTCOME PROOF or TWIN is an IntegrityError: the analysis REFUSES and writes nothing. It is
never converted into a missing block.

OUTCOMES      A run whose record status is `algorithmic_failure`, `infrastructure_failure`, `parity_refused` or `refused`
              is an outcome: its block is dropped from every contrast that needs it (counted). More than 3 of 36 blocks
              dropped: the contrast is NOT ESTIMABLE (not rejected).
PRIMARY       Holm over two, familywise alpha 0.05, two-sided paired t over blocks: B-H1 AUC_HQ - AUC_C-matched;
              B-H2 AUC_HQ - AUC_S625-twin. Degenerate case (sd = 0): mean 0 -> t = 0, p = 1; a non-zero constant
              effect -> t = +/-inf, p = 0 (predeclared).
TOST (B-H1)   equivalent within +/-0.0056 only if the 90% t interval lies inside; it does not change the Holm decision.
READING       B-H2 compares two PRESELECTED TRAINING CONFIGURATIONS (HQ lr 3e-3, twin lr 1e-3) from an identical
              starting policy; it is not an equal-KL isolation of the parameterisation (review 40). Realised KL is
              secondary diagnostic evidence only.
SECONDARY     final outcomes, realised KL, per-cell HQ - C-matched with drift/budget contrasts, twin divergence
              (descriptive); the B1 accessibility index is not re-tested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_runtime as dq                                                                           # noqa: E402

TOOL = 'gnss-dq-b2-analysis/3'
ALPHA = 0.05
BLOCKS = tuple(range(36))
MAX_DROPPED = 3
TOST_DELTA = 0.0056
ARMS = ('quantum', 'bottleneck', 'surrogate625')
OUTCOME_STATUSES = ('algorithmic_failure', 'infrastructure_failure', 'parity_refused', 'refused')
TARGET_CELLS = ('d2__nominal_2700s__positive', 'd2__tight_derived__positive', 'd4__nominal_2700s__positive',
                'd4__tight_derived__positive')
READING = ('B-H2 compares two preselected training configurations (HQ lr 3e-3, S625-twin lr 1e-3) from an identical '
           'starting policy; it is not an equal-KL isolation of the parameterisation. B1 remains the equal-KL local '
           'mechanism experiment; realised KL here is secondary diagnostic evidence.')


class IntegrityError(dq.DQError):
    """A contradictory, tampered, incomplete or foreign artifact: the analysis refuses."""


# -- identity and raw-event proof ----------------------------------------------------------------------------------------

def _expected_names():
    return {f'{arm}__b{b}' for b in BLOCKS for arm in ARMS}


def load_records(campaign):
    files = sorted((Path(campaign) / 'records').glob('*'))
    names = [p.name for p in files]
    expected = {f'{n}.json' for n in _expected_names()}
    extra, missing = sorted(set(names) - expected), sorted(expected - set(names))
    if extra or missing or len(names) != len(set(names)):
        raise IntegrityError(f'records: extra {extra[:5]}, missing {missing[:5]}')
    records = {}
    for path in files:
        try:
            r = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise IntegrityError(f'{path.name}: malformed JSON') from exc
        key = (r.get('arm'), r.get('block'))
        if f'{key[0]}__b{key[1]}.json' != path.name:
            raise IntegrityError(f'{path.name}: names ({key}) inside the record')
        if key in records:
            raise IntegrityError(f'duplicate identity {key}')
        records[key] = r
    return records


def check_identity(record, freeze_sha256, tree_sha256, config_for):
    arm, block = record['arm'], record['block']
    import dq_b2
    if record.get('study') != 'dq_b2' or record.get('role') != dq_b2.ARMS[arm]['role']:
        raise IntegrityError(f'{arm} b{block}: study/role {record.get("study")}/{record.get("role")}')
    if record.get('status') == 'refused' and 'config' not in record:
        return                                             # refused before any configuration existed (proven below)
    expected = config_for(arm, block)
    if dq.canonical_sha256(record.get('config')) != dq.canonical_sha256(expected):
        raise IntegrityError(f'{arm} b{block}: the configuration is not the frozen one')
    if record.get('config_sha256') != dq.canonical_sha256(expected):
        raise IntegrityError(f'{arm} b{block}: config_sha256 mismatch')
    if record.get('protocol_freeze_sha256') != freeze_sha256 or record.get('code_tree_sha256') != tree_sha256:
        raise IntegrityError(f'{arm} b{block}: freeze or code-tree identity mismatch')


def metrics_path(campaign, arm, block):
    return Path(campaign) / 'runs' / f'{arm}__b{block}' / 'run' / arm / 'metrics.jsonl'


def read_events(campaign, arm, block):
    path = metrics_path(campaign, arm, block)
    if not path.exists():
        raise IntegrityError(f'{arm} b{block}: metrics.jsonl missing for a completed run')
    events = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IntegrityError(f'{arm} b{block}: malformed event') from exc
    return events


def recompute_auc(events, grid, budget, arm, block):
    import p4_train as p4
    points, values = [], []
    for e in events:
        if e.get('kind') != 'evaluation':
            continue
        v = e['results']['sample']['weighted_J']
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise IntegrityError(f'{arm} b{block}: non-finite evaluation value')
        for p in e['points']:
            points.append(p)
            values.append(v)
    if points != list(grid):
        raise IntegrityError(f'{arm} b{block}: evaluation points {points} are not the frozen grid exactly once, in order')
    return p4.learning_curve_auc(points, values, budget), list(zip(points, values))


def check_completed_run(campaign, record, grid, budget):
    arm, block = record['arm'], record['block']
    events = read_events(campaign, arm, block)
    if not events or events[0].get('kind') != 'arm_start' or events[-1].get('kind') != 'arm_end':
        raise IntegrityError(f'{arm} b{block}: the event log does not start with arm_start and end with arm_end')
    if sum(e.get('kind') == 'arm_start' for e in events) != 1 or sum(e.get('kind') == 'arm_end' for e in events) != 1:
        raise IntegrityError(f'{arm} b{block}: more than one arm_start / arm_end')
    updates = [e['update_index'] for e in events if e.get('kind') == 'update']
    if updates != list(range(1, len(updates) + 1)):
        raise IntegrityError(f'{arm} b{block}: update indices are not contiguous')
    finals = [e for e in events if e.get('kind') == 'checkpoint' and e.get('final')]
    if len(finals) != 1:
        raise IntegrityError(f'{arm} b{block}: {len(finals)} final checkpoints')
    auc, curve = recompute_auc(events, grid, budget, arm, block)
    end, summary = events[-1], record.get('summary') or {}
    for source, s in (('arm_end', end), ('record', summary)):
        if s.get('status') != 'completed' or (s.get('auc') or {}).get('sample') != auc:
            raise IntegrityError(f'{arm} b{block}: the {source} AUC or status disagrees with the raw evaluation events')
        if s.get('updates') != len(updates):
            raise IntegrityError(f'{arm} b{block}: the {source} update count disagrees with the events')
    for key in ('policy_decisions', 'episodes', 'updates', 'block', 'arm'):
        if end.get(key) != summary.get(key):
            raise IntegrityError(f'{arm} b{block}: record and arm_end disagree on {key}')
    return {'auc': auc, 'curve': curve, 'events': events}


def run_dir(campaign, arm, block):
    return Path(campaign) / 'runs' / f'{arm}__b{block}'


def check_outcome(campaign, record):
    """R41-1: a non-completed status must be consistent with every artifact that exists."""
    arm, block, status = record['arm'], record['block'], record['status']
    if not isinstance(record.get('error'), str) and status != 'algorithmic_failure':
        raise IntegrityError(f'{arm} b{block}: outcome {status} without its error text')
    if (record.get('summary') or {}).get('status') == 'completed':
        raise IntegrityError(f'{arm} b{block}: outcome {status} with a completed summary')
    d = run_dir(campaign, arm, block)
    if status in ('refused', 'parity_refused'):
        if d.exists():
            raise IntegrityError(f'{arm} b{block}: {status} but a run directory exists')
        return
    if record.get('config') is None:
        raise IntegrityError(f'{arm} b{block}: {status} after training started must carry its configuration')
    checkpoints = d / 'run' / arm / 'checkpoints'
    if checkpoints.exists() and any(p.name.startswith('final_') for p in checkpoints.iterdir()):
        raise IntegrityError(f'{arm} b{block}: {status} but a final checkpoint exists')
    events = read_events(campaign, arm, block) if metrics_path(campaign, arm, block).exists() else []
    ends = [e for e in events if e.get('kind') == 'arm_end']
    if status == 'algorithmic_failure':
        if len(ends) != 1 or ends[0].get('status') != 'algorithmic_failure' or \
                (record.get('summary') or {}).get('status') != 'algorithmic_failure':
            raise IntegrityError(f'{arm} b{block}: algorithmic failure not recorded as such by the trainer')
    elif status == 'infrastructure_failure' and ends:
        raise IntegrityError(f'{arm} b{block}: infrastructure failure but the trainer finished (arm_end present)')


def check_twin(record):
    tw = record.get('twin_parity') or {}
    static, p4_ = tw.get('static') or {}, tw.get('P4') or {}
    if static.get('static_failures') != [] or not isinstance(p4_.get('max_abs_dlogprob'), (int, float)) or \
            not p4_['max_abs_dlogprob'] <= 1e-9:
        raise IntegrityError(f'surrogate625 b{record["block"]}: completed without a valid twin certificate')


def prove(campaign, freeze_path, freeze_sha256):
    """FREEZE, IDENTITY, RAW EVENTS, OUTCOME PROOF and TWIN; returns (records, {(arm, block): proven run or None})."""
    import dq_b2
    import p4_train as p4
    freeze_path = Path(freeze_path).resolve()
    try:
        freeze = p4.verify_freeze(freeze_path, freeze_sha256)
    except (p4.FreezeBindingError, FileNotFoundError) as exc:
        raise IntegrityError(f'the freeze does not verify: {exc}') from exc
    manifest = json.loads((dq.ROOT / freeze['seed_manifest']).read_text())['seed_manifest']
    runs = manifest['runs']
    if len(runs) != 108 or len(set(runs)) != 108 or set(runs) != _expected_names():
        raise IntegrityError('the frozen manifest does not list exactly 108 unique B2 run identities')
    if Path(campaign).resolve() != (dq.ROOT / manifest['campaign']).resolve():
        raise IntegrityError(f'{campaign} is not the frozen campaign {manifest["campaign"]}')
    launch_path = Path(campaign) / 'launch.json'
    if not launch_path.exists():
        raise IntegrityError('launch.json is missing')
    launch = json.loads(launch_path.read_text())
    if Path(launch.get('freeze', '')).resolve() != freeze_path or launch.get('freeze_sha256') != freeze_sha256 or \
            launch.get('code_tree_sha256') != freeze['copy_code_tree_sha256']:
        raise IntegrityError('launch.json does not name this freeze, hash and frozen tree')
    grid, budget = manifest['grid'], manifest['budget']
    records = load_records(campaign)
    proven = {}
    for (arm, block), record in records.items():
        check_identity(record, freeze_sha256, freeze['copy_code_tree_sha256'],
                       lambda a, b: dq_b2.config_for(a, b, dq_b2.STUDY_B2, freeze_sha256=freeze_sha256))
        status = record.get('status')
        if status == 'completed':
            run = check_completed_run(campaign, record, grid, budget)
            if arm == 'surrogate625':
                check_twin(record)
            proven[(arm, block)] = run
        elif status in OUTCOME_STATUSES:
            check_outcome(campaign, record)
            proven[(arm, block)] = None
        else:
            raise IntegrityError(f'{arm} b{block}: unknown status {status!r}')
    return records, proven


# -- statistics -------------------------------------------------------------------------------------------------------------

def paired(proven, a, b):
    values, dropped = {}, []
    for blk in BLOCKS:
        x, y = proven.get((a, blk)), proven.get((b, blk))
        if x is None or y is None:
            dropped.append(blk)
        else:
            values[blk] = x['auc'] - y['auc']
    return values, dropped


def t_test(values, dropped):
    x = np.asarray(list(values.values()), float)
    n = x.size
    record = {'n': int(n), 'dropped_blocks': dropped, 'block_values': {str(k): v for k, v in values.items()}}
    if len(dropped) > MAX_DROPPED or n < 3:
        return dict(record, estimable=False, p=None)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    if sd == 0:
        t, p, degenerate = (0., 1., True) if mean == 0 else (math.copysign(math.inf, mean), 0., True)
    else:
        t, degenerate = mean / se, False
        p = float(2 * stats.t.sf(abs(t), n - 1))
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return dict(record, estimable=True, mean=mean, sd=sd, se=se, t=float(t), p=float(p), degenerate=degenerate,
                ci95=[mean - q95 * se, mean + q95 * se], ci90=[mean - q90 * se, mean + q90 * se],
                standardised_effect=mean / sd if sd > 0 else None)


def holm(tests):
    ordered = sorted(((r['p'], n) for n, r in tests.items() if r.get('estimable')), key=lambda z: z[0])
    out = {n: {'rejected': False, 'holm_threshold': None} for n in tests}
    for rank, (p, n) in enumerate(ordered):
        thr = ALPHA / (len(tests) - rank)
        out[n]['holm_threshold'] = thr
        if p <= thr:
            out[n]['rejected'] = True
        else:
            break
    return out


def tost(test):
    if not test.get('estimable'):
        return {'equivalent_within_delta': False, 'delta': TOST_DELTA, 'reason': 'not estimable'}
    lo, hi = test['ci90']
    return {'equivalent_within_delta': bool(-TOST_DELTA < lo and hi < TOST_DELTA), 'delta': TOST_DELTA, 'ci90': [lo, hi]}


def t_summary(v):
    x = np.asarray(v, float)
    if x.size < 3:
        return {'n': int(x.size)}
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(x.size)
    q = stats.t.ppf(0.975, x.size - 1)
    return {'n': int(x.size), 'mean': mean, 'ci95': [mean - q * se, mean + q * se],
            'p_two_sided': float(2 * stats.t.sf(abs(mean / se), x.size - 1)) if se > 0 else None}


def secondary(proven, records):
    final, kl, per_cell = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: {} for a in ARMS}
    for (arm, blk), run in proven.items():
        if run is None:
            continue
        evals = [e for e in run['events'] if e.get('kind') == 'evaluation']
        updates = [e for e in run['events'] if e.get('kind') == 'update']
        res = evals[-1]['results']['sample']
        eps = res['episodes']
        final[arm].append({'block': blk, 'weighted_J': res['weighted_J'],
                           **{k: float(np.mean([float(e[k]) for e in eps]))
                              for k in ('tp_timely', 'fp_spatial', 'return_ok', 'fixes_attempted')}})
        per_cell[arm][blk] = res['per_cell_J']
        ks = [u['metrics']['approx_kl'] for u in updates]
        kl[arm].append({'block': blk, 'mean_approx_kl': float(np.mean(ks)), 'cumulative_approx_kl': float(np.sum(ks)),
                        'final_weighted_J': res['weighted_J']})
    common = [b for b in BLOCKS if b in per_cell['quantum'] and b in per_cell['bottleneck']]
    diff = {c: [per_cell['quantum'][b][c] - per_cell['bottleneck'][b][c] for b in common] for c in TARGET_CELLS}
    idx = range(len(common))
    drift = [np.mean([diff[TARGET_CELLS[2]][i], diff[TARGET_CELLS[3]][i]]) - np.mean([diff[TARGET_CELLS[0]][i], diff[TARGET_CELLS[1]][i]]) for i in idx]
    budget = [np.mean([diff[TARGET_CELLS[1]][i], diff[TARGET_CELLS[3]][i]]) - np.mean([diff[TARGET_CELLS[0]][i], diff[TARGET_CELLS[2]][i]]) for i in idx]
    divergence = {}
    for blk in BLOCKS:
        a, b = proven.get(('quantum', blk)), proven.get(('surrogate625', blk))
        if a is None or b is None:
            continue
        ea = [e for e in a['events'] if e.get('kind') == 'episode']
        eb = [e for e in b['events'] if e.get('kind') == 'episode']
        divergence[str(blk)] = next((i for i, (x, y) in enumerate(zip(ea, eb))
                                     if (x['J'], x['decisions'], x['termination_reason']) !=
                                     (y['J'], y['decisions'], y['termination_reason'])), None)
    return {'final_outcomes': {a: ({k: float(np.mean([f[k] for f in final[a]])) for k in final[a][0] if k != 'block'}
                                   if final[a] else None) for a in ARMS},
            'kl': {a: ({'mean_approx_kl': float(np.mean([k['mean_approx_kl'] for k in kl[a]])),
                        'cumulative_approx_kl': float(np.mean([k['cumulative_approx_kl'] for k in kl[a]])),
                        'per_block': kl[a]} if kl[a] else None) for a in ARMS},
            'hq_minus_cmatched_final_per_cell': {c: t_summary(v) for c, v in diff.items()},
            'cell_contrasts': {'drift_d4_minus_d2': t_summary(drift), 'budget_tight_minus_nominal': t_summary(budget)},
            'twin_parity': {str(b): records[('surrogate625', b)].get('twin_parity') for b in BLOCKS},
            'hq_twin_first_divergent_training_episode': divergence,
            'omitted': 'B1 P-H1 accessibility index not re-tested (review 38 addendum)'}


def analyze(campaign, freeze_path, freeze_sha256):
    records, proven = prove(campaign, freeze_path, freeze_sha256)
    status = {}
    for r in records.values():
        status[r['status']] = status.get(r['status'], 0) + 1
    h1 = t_test(*paired(proven, 'quantum', 'bottleneck'))
    h2 = t_test(*paired(proven, 'quantum', 'surrogate625'))
    decisions = holm({'B-H1': h1, 'B-H2': h2})
    return {'tool': TOOL, 'freeze_sha256': freeze_sha256, 'status_counts': status, 'reading': READING,
            'primary': {'B-H1': dict(h1, **decisions['B-H1'], tost=tost(h1)), 'B-H2': dict(h2, **decisions['B-H2'])},
            'per_arm_auc_recomputed': {a: [None if proven.get((a, b)) is None else proven[(a, b)]['auc'] for b in BLOCKS]
                                       for a in ARMS},
            'secondary': secondary(proven, records)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.campaign.resolve(), args.freeze.resolve(), args.freeze_sha256)
    dq.claim_output(args.out.resolve(), directory=True)
    dq.write_json(args.out.resolve() / 'b2_report.json', json.loads(json.dumps(report, default=float)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
