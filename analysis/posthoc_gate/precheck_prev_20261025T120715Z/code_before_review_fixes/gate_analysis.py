"""Analysis of E1 (learned correction trigger), exactly as fixed in PLAN.md before any main run.

    python gate_analysis.py [--provisional]   # writes gate_results.json; refuses incomplete data unless --provisional
    python gate_analysis.py --selftest        # stored rule J of record re-derived; test statistics checked (no new run)
    python gate_analysis.py --dryrun          # main() on synthetic records in the scratchpad (no new run read)

Primary (P1): per block b, D_b = cell-weighted J(gate, argmax, final) - cell-weighted J(rule) over the 80 paired
e1_evaluation episodes (frozen cell weights 0.2 / 0.05, p4_train.aggregate_evaluation; pairs checked on cell_id, seed
and crn_sha256). Across blocks: mean, 95% paired t interval, two-sided paired t p-value, dz; TOST at +-0.0056 J
(two one-sided t tests at 0.05 = 90% interval inside the margin). Holm over the primary family (one member).
Secondary (S1, only if the stress read-out exists): the same on the 320-episode stress map (map weights 0.4 / 0.1).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DELTA = 0.0056          # 0.1 report-equivalent per episode (dq_b2.TOST_DELTA), fixed before any gate data
ALPHA = 0.05
MIN_BLOCKS = 8          # fewer completed blocks: P1 is 'not resolved (insufficient completed blocks)'
N_REF = 18


def paired(d, delta=DELTA):
    d = np.asarray(d, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float('nan')
    se = sd / np.sqrt(n)
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    if se == 0:
        p = 1.0 if m == 0 else 0.0
        p_lo = p_hi = 0.0 if abs(m) < delta else 1.0
    else:
        p = float(2 * stats.t.sf(abs(m / se), n - 1))
        p_lo = float(stats.t.sf((m + delta) / se, n - 1))       # H0: mu <= -delta
        p_hi = float(stats.t.cdf((m - delta) / se, n - 1))      # H0: mu >= +delta
    return {'n': n, 'mean': m, 'sd': sd, 'se': se, 'lo95': m - q95 * se, 'hi95': m + q95 * se,
            'lo90': m - q90 * se, 'hi90': m + q90 * se, 'p_two_sided': p, 'dz': m / sd if sd else None,
            'tost_p': max(p_lo, p_hi), 'tost_p_lower': p_lo, 'tost_p_upper': p_hi, 'delta': delta,
            'mean_report_equivalents': m * N_REF, 'share_negative': float(np.mean(d < 0)),
            'share_zero': float(np.mean(d == 0))}


def holm(ps, alpha=ALPHA):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, still = [False] * len(ps), True
    for rank, i in enumerate(order):
        still = still and ps[i] <= alpha / (len(ps) - rank)
        rej[i] = still
    return rej


def reading(s, diff_rejected, equivalent):
    """The pre-declared reading (PLAN.md, 'Reading rules')."""
    if s['n'] < MIN_BLOCKS:
        return 'not resolved (insufficient completed blocks)'
    if equivalent and diff_rejected:
        return ('equivalent and better (a gain smaller than the margin)' if s['mean'] > 0 else
                'equivalent and worse (a loss smaller than the margin)')
    if equivalent:
        return 'equivalent'
    if diff_rejected:
        return 'better' if s['mean'] > 0 else 'worse'
    return 'not resolved'


def rule_rows(suite='e1_evaluation'):
    import runner
    return runner.stored_rule_rows(suite)


def aggregation():
    import runner
    import p4_train as p4
    rt = runner.runtime(runner.STUDY_MAIN)
    return rt, p4.aggregate_evaluation


RECORDS = {'dir': HERE / 'records', 'stress': HERE / 'stress_records', 'out': HERE / 'gate_results.json'}


def load_records(blocks):
    recs = {}
    for b in blocks:
        f = RECORDS['dir'] / f'gate__b{b}.json'
        if f.exists():
            recs[b] = json.loads(f.read_text())
    return recs


def block_contrast(rec, stored, rt, agg):
    rows = rec['final_evaluation']['episodes']
    if len(rows) != 80:
        raise SystemExit(f'block {rec["block"]}: {len(rows)} evaluation rows, not 80')
    for r in rows:
        s = stored[(r['cell_id'], r['seed'])]
        if s['crn_sha256'] != r['crn_sha256'] or s['J'] != r['rule_J']:
            raise SystemExit(f'block {rec["block"]}: row {r["cell_id"]} {r["seed"]} is not paired with the stored rule')
    _, w_gate = agg(rt, rows)
    _, w_rule = agg(rt, [dict(r, J=stored[(r['cell_id'], r['seed'])]['J']) for r in rows])
    pos = [r for r in rows if r['cell_id'].endswith('__positive')]
    gates = [g for r in pos for g in r['gates']]
    bins = [(0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]
    fix_by_rms = {f'[{a},{b})': (float(np.mean([g[1] for g in gates if a <= g[2] < b]))
                                 if any(a <= g[2] < b for g in gates) else None) for a, b in bins}
    return {'block': rec['block'], 'D': w_gate - w_rule, 'J_gate': w_gate, 'J_rule': w_rule,
            'unweighted_mean_d': float(np.mean([r['J'] - r['rule_J'] for r in rows])),
            'per_cell_d': {c: float(np.mean([r['J'] - r['rule_J'] for r in rows if r['cell_id'] == c]))
                           for c in sorted({r['cell_id'] for r in rows})},
            'fixes_per_positive_episode': {'gate': float(np.mean([r['fixes_attempted'] for r in pos])),
                                           'rule': float(np.mean([r['rule_fixes'] for r in pos]))},
            'fp_per_positive_episode': {'gate': float(np.mean([r['fp_spatial'] for r in pos])),
                                        'rule': float(np.mean([r['rule_fp'] for r in pos]))},
            'tp_timely_per_positive_episode': {'gate': float(np.mean([r['tp_timely'] for r in pos])),
                                               'rule': float(np.mean([r['rule_tp_timely'] for r in pos]))},
            'return_rate': float(np.mean([r['return_ok'] for r in rows])),
            'infrastructure_episodes': sum(1 for r in rows if r['infrastructure']),
            'gate_rows_per_positive_episode': float(np.mean([r['gate_valid_rows'] for r in pos])),
            'fix_rate_by_rms_m': fix_by_rms,
            'no_human_identical_to_rule': all(r['J'] == r['rule_J'] for r in rows if r['cell_id'].endswith('__no_human'))}


def stress_contrast(blocks):
    out = {}
    for b in blocks:
        f = RECORDS['stress'] / f'gate__b{b}.json'
        if not f.exists():
            continue
        rows = json.loads(f.read_text())['episodes']
        if len(rows) != 320 or any('error' in r for r in rows):
            raise SystemExit(f'stress block {b}: incomplete or failed rows')
        cells = sorted({r['cell_id'] for r in rows})
        w = {c: next(r['weight'] for r in rows if r['cell_id'] == c) for c in cells}
        dc = {c: float(np.mean([r['d_J'] for r in rows if r['cell_id'] == c])) for c in cells}
        by_drift = {}
        for d600 in sorted({r['d600'] for r in rows}):
            cs = [c for c in cells if next(r['d600'] for r in rows if r['cell_id'] == c) == d600]
            by_drift[str(d600)] = sum(w[c] * dc[c] for c in cs) / sum(w[c] for c in cs)
        out[b] = {'D': sum(w[c] * dc[c] for c in cells) / sum(w.values()), 'by_drift': by_drift}
    return out


def main(provisional=False):
    import runner
    blocks = runner.BLOCKS
    recs = load_records(blocks)
    done = {b: r for b, r in recs.items() if r.get('status') == 'completed'}
    complete = len(recs) == len(blocks)
    if not complete and not provisional:
        raise SystemExit(f'incomplete: {len(recs)} of {len(blocks)} records; the read-out waits for all of them')
    if len(done) < 2:
        raise SystemExit(f'{len(done)} completed blocks: nothing to test')
    stored = rule_rows()
    rt, agg = aggregation()
    per_block = [block_contrast(done[b], stored, rt, agg) for b in sorted(done)]
    P1 = paired([x['D'] for x in per_block])
    P1['holm_rejected'] = holm([P1['p_two_sided']])[0]
    P1['equivalent'] = P1['tost_p'] < ALPHA
    P1['reading'] = reading(P1, P1['holm_rejected'], P1['equivalent'])
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()
           if (HERE / 'PLAN.md').exists() else None,
           'provisional': not complete, 'blocks_recorded': sorted(recs), 'blocks_completed': sorted(done),
           'blocks_not_completed': {b: recs[b].get('status') for b in recs if b not in done},
           'P1_gate_minus_rule_e1_evaluation': P1,
           'A0_note': ('A0 (freeze_a a0.json, +0.40 report-equivalents) is a conditional replay ceiling of never_fix '
                       'trajectories with perfect georeferencing; it is not an upper bound for a gate that changes '
                       'timing and is not called headroom.'),
           'descriptive': {
               'per_block': per_block,
               'mean_unweighted_80_episode_d': paired([x['unweighted_mean_d'] for x in per_block]),
               'per_cell_d_mean_over_blocks': {c: float(np.mean([x['per_cell_d'][c] for x in per_block]))
                                               for c in per_block[0]['per_cell_d']} if per_block else {},
               'fixes_per_positive_episode_gate_mean': float(np.mean([x['fixes_per_positive_episode']['gate']
                                                                      for x in per_block])) if per_block else None,
               'fixes_per_positive_episode_rule': per_block[0]['fixes_per_positive_episode']['rule'] if per_block else None,
               'fp_per_positive_episode_gate_mean': float(np.mean([x['fp_per_positive_episode']['gate']
                                                                   for x in per_block])) if per_block else None,
               'learning_curve_dev_weighted_d': {b: done[b]['summary']['learning_curve'] for b in sorted(done)},
               'training': {b: {k: done[b]['summary'][k] for k in ('episodes', 'gate_decisions', 'updates',
                                                                    'infrastructure_episodes', 'empty_episodes',
                                                                    'pre_gate_reward', 'dropped_reward', 'wall_s')}
                            for b in sorted(done)}}}
    st = stress_contrast(sorted(done))
    if st:
        S1 = paired([st[b]['D'] for b in sorted(st)])
        S1['equivalent'] = S1['tost_p'] < ALPHA
        S1['note'] = 'secondary; no Holm coupling with P1; drifts 0 and 8 m are outside the training range'
        S1['by_drift_mean'] = {d: float(np.mean([st[b]['by_drift'][d] for b in st])) for d in st[min(st)]['by_drift']}
        out['S1_gate_minus_rule_stress_map'] = S1
    RECORDS['out'].write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({'provisional': out['provisional'], 'blocks_completed': len(done),
                      'P1': {k: P1[k] for k in ('n', 'mean', 'lo95', 'hi95', 'p_two_sided', 'tost_p', 'reading')}},
                     indent=1, default=float))


def selftest():
    import runner
    stored = rule_rows()
    rt, agg = aggregation()
    rows = list(stored.values())
    _, w = agg(rt, rows)
    assert w == 0.20564463190957744, w                     # baselines.json p3_rule/e1_evaluation weighted_J
    rng = np.random.default_rng(1)
    d = rng.normal(0.001, 0.004, 12)
    s = paired(d)
    t = stats.ttest_1samp(d, 0.0)
    assert abs(s['p_two_sided'] - t.pvalue) < 1e-12
    lo = stats.ttest_1samp(d, -DELTA, alternative='greater').pvalue
    hi = stats.ttest_1samp(d, DELTA, alternative='less').pvalue
    assert abs(s['tost_p'] - max(lo, hi)) < 1e-12
    assert (s['tost_p'] < 0.05) == (s['lo90'] > -DELTA and s['hi90'] < DELTA)
    assert holm([0.01, 0.04]) == [True, True] and holm([0.03, 0.04]) == [False, False]
    assert reading({'n': 12, 'mean': -0.01}, True, False) == 'worse'
    assert reading({'n': 12, 'mean': 0.0}, False, True) == 'equivalent'
    assert reading({'n': 7, 'mean': 0.0}, False, True).startswith('not resolved')
    print(json.dumps({'selftest': 'passed', 'rule_weighted_J_e1_evaluation': w, 'synthetic': s}, indent=1,
                     default=float))


def dryrun():
    """The whole read-out on SYNTHETIC records built from the stored rule rows (gate J = rule J + noise on the
    positive cells), written to a scratch directory; exercises every code path of main() without any run."""
    import tempfile
    import runner
    stored = rule_rows()
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad'))
    (tmp / 'records').mkdir()
    for b in runner.BLOCKS:
        shift = rng.normal(-0.002, 0.004)
        rows = []
        for (cell, seed), s in stored.items():
            pos = cell.endswith('__positive')
            J = s['J'] + (shift + rng.normal(0, 0.02) if pos else 0.0)
            rows.append(dict(s, rule_J=s['J'], d_J=J - s['J'], J=J, rule_fixes=s['fixes_attempted'],
                             rule_fp=s['fp_spatial'], rule_tp_timely=s['tp_timely'], infrastructure=False,
                             gate_valid_rows=8 if pos else 0,
                             gates=[[100.0 * k, int(rng.random() < 0.5), float(rng.uniform(0, 3))]
                                    for k in range(8 if pos else 0)]))
        rec = {'block': b, 'status': 'completed', 'final_evaluation': {'episodes': rows},
               'summary': {'learning_curve': [], 'episodes': 2000, 'gate_decisions': 12800, 'updates': 50,
                           'infrastructure_episodes': 0, 'empty_episodes': 400, 'pre_gate_reward': 0.0,
                           'dropped_reward': 0.0, 'wall_s': 1.0}}
        (tmp / 'records' / f'gate__b{b}.json').write_text(json.dumps(rec, default=float))
    RECORDS.update(dir=tmp / 'records', stress=tmp / 'stress_records', out=tmp / 'gate_results.json')
    main(False)
    res = json.loads(RECORDS['out'].read_text())
    assert res['P1_gate_minus_rule_e1_evaluation']['n'] == 12 and not res['provisional']
    assert all(x['no_human_identical_to_rule'] for x in res['descriptive']['per_block'])
    print('dryrun passed; synthetic results in', RECORDS['out'])


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--dryrun' in sys.argv:
        dryrun()
    else:
        main('--provisional' in sys.argv)
