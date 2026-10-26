"""Analysis of post-hoc check E4, exactly as fixed in PLAN.md (drafted as PLAN_DRAFT.md before any E4 run).

    python -B e4_analysis.py               # writes e4_results.json; refuses unless all 12 runs have finished
    python -B e4_analysis.py --provisional # partial data allowed; writes e4_results_PROVISIONAL.json (never of record)
    python -B e4_analysis.py --deadline-readout   # only after DEADLINE_UTC (plan's stopping rule): largest common
                                                  # checkpoint, written to e4_results.json and labelled as such
    python -B e4_analysis.py --selftest    # reads NO E4 run: recomputes LONG's stored 300k values and the rule reference

Primary (per head, n = 6 blocks, paired by block): P1 = J300k(E4) - J300k(LONG) (paired t, two-sided, dz; Holm over
the two heads); P2 = share of LONG's gap to the paired rule reference closed by E4 (ratio of block means, percentile
bootstrap over blocks); P3 = half-rule check. Readings are computed here from the rules fixed in the plan.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4lib as e4                                                                                # noqa: E402
import dq_b2                                                                                      # noqa: E402
import p4_train as p4                                                                             # noqa: E402

ARMS, BLOCKS, GRID, T_PLANNED = e4.ARMS, e4.BLOCKS, e4.GRID, e4.BUDGET
LONG_HALF_RULE = 0.103                     # LONG's planned threshold (half of min(0.206, 0.2104)), reported alongside
NO_TIMELY_CUT = 0.012                      # LONG's post-result forensic: <= 0.012 timely-credited reports per episode
COLLAPSE_ENTROPY = e4.COLLAPSE_ENTROPY_NATS
BOOT_SEED, BOOT_N = 20261025, 20_000
#: PLAN.md stopping rule (the lead may change it before hashing): read-out at the largest common checkpoint after this
DEADLINE_UTC = '2026-10-30T12:00:00+00:00'
RT = None


def runtime():
    global RT
    if RT is None:
        RT = dq_b2.runtime(dq_b2.STUDY_B2)
    return RT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path):
    """{'curve': {point: {...}}, 'status', 'last_entropy', 'last_lr', 'ev': {...}} from one metrics.jsonl."""
    curve, status, ups, ev = {}, None, [], {k: 0 for k in e4.EV_STATS}
    ev['decisions'] = 0
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        k = e['kind']
        if k == 'evaluation':
            res = e['results']['sample']
            per_cell, weighted = p4.aggregate_evaluation(runtime(), res['episodes'])
            if weighted != res['weighted_J']:
                raise SystemExit(f'{path}: stored weighted J differs from its episodes')
            eps = res['episodes']
            row = {'J': weighted, 'policy_decisions': e['policy_decisions'],
                   'timely': float(np.mean([x['tp_timely'] for x in eps])),
                   'false': float(np.mean([x['fp_spatial'] for x in eps])),
                   'fixes': float(np.mean([x['fixes_attempted'] for x in eps])),
                   'decisions_per_episode': float(np.mean([x['decisions'] for x in eps])),
                   'pairs': [(x['cell_id'], x['seed']) for x in eps]}
            for p in e['points']:
                curve[p] = row
        elif k == 'update':
            ups.append(e['metrics'])
        elif k == 'episode':
            for s in e4.EV_STATS:
                ev[s] += e['policy_stats'].get(s, 0)
            ev['decisions'] += e['policy_stats'].get('decisions', 0)
        elif k == 'arm_end':
            status = e['status']
    return {'curve': curve, 'status': status, 'last_entropy': ups[-1]['entropy'] if ups else None,
            'last_lr': ups[-1].get('learning_rate') if ups else None, 'updates': len(ups), 'ev': ev}


def t_sum(x):
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return {'n': int(n), 'mean': float(x.mean()) if n else None}
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    return {'n': int(n), 'mean': float(m), 'sd': float(sd), 'lo': float(m - q * se), 'hi': float(m + q * se), 'p': p,
            'dz': float(m / sd) if sd > 0 else None, 'share_negative': float(np.mean(x < 0)), 'values': x.tolist()}


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, still = [False] * len(ps), True
    for rank, i in enumerate(order):
        still = still and ps[i] <= alpha / (len(ps) - rank)
        rej[i] = still
    return rej


def gap_closed(e4_j, long_j, rule_j):
    """Ratio of block means (J_E4 - J_LONG) / (J_rule - J_LONG); bootstrap over blocks (percentile 95%)."""
    a, b = np.asarray(e4_j, float), np.asarray(long_j, float)
    den = rule_j - b
    if den.mean() <= 0:
        return {'estimate': None, 'note': 'LONG at or above the rule: undefined'}
    est = float((a - b).mean() / den.mean())
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, a.size, size=(BOOT_N, a.size))
    num, dd = (a - b)[idx].mean(1), den[idx].mean(1)
    boots = num[dd > 0] / dd[dd > 0]
    return {'estimate': est, 'lo': float(np.percentile(boots, 2.5)), 'hi': float(np.percentile(boots, 97.5)),
            'per_block': ((a - b) / den).tolist(), 'bootstrap': {'seed': BOOT_SEED, 'resamples': BOOT_N}}


def auc(curve, T):
    pts = [p for p in GRID if p <= T]
    J = np.array([curve[p]['J'] for p in pts])
    return float(np.sum((J[1:] + J[:-1]) / 2 * np.diff(pts)) / T)


def rule_reference():
    ref = json.loads((HERE / 'rule_reference.json').read_text())
    per_cell, weighted = p4.aggregate_evaluation(runtime(), ref['episodes'])
    if weighted != ref['weighted_J'] or not ref['checks']['passed']:
        raise SystemExit('rule_reference.json does not verify')
    return ref


def p1_reading(t, rejected):
    if rejected and t['lo'] > 0:
        return 'above zero: the best-effort configuration ended higher than LONG at 300,000 decisions'
    if rejected and t['hi'] < 0:
        return 'below zero: the best-effort configuration ended lower than LONG at 300,000 decisions'
    return 'not resolved at n = 6 (not evidence of equality)'


def main(provisional=False, deadline=False):
    ref = rule_reference()
    J_rule, half = ref['weighted_J'], ref['weighted_J'] / 2
    pairs = [(r['cell_id'], r['seed']) for r in ref['episodes']]
    new, old = {}, {}
    for arm in ARMS:
        for b in BLOCKS:
            f = HERE / 'runs' / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl'
            if f.exists():
                new[(arm, b)] = read_run(f)
            old[(arm, b)] = read_run(e4.LONG_DIR / 'runs' / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl')
    finished = {k: v['status'] in ('completed', 'algorithmic_failure') for k, v in new.items()}
    complete = len(new) == 12 and all(finished.values()) and \
        all((HERE / 'records' / f'{a}__b{b}.json').exists() for a in ARMS for b in BLOCKS)
    if not complete and not (provisional or deadline):
        raise SystemExit(f'incomplete: {sum(finished.values())} of 12 runs finished; the read-out waits (PLAN.md)')
    for v in list(new.values()) + list(old.values()):          # every evaluation flew the rule reference's episodes
        for p, row in v['curve'].items():
            if row['pairs'] != pairs:
                raise SystemExit('an evaluation is not paired with the rule reference')
    # read-out checkpoint: 300,000 once every run has finished (an algorithmically failed run simply has no value
    # there); otherwise (provisional, or the plan's deadline read-out) the largest checkpoint every unfailed run reached
    reached = [max(v['curve']) for v in new.values() if v['curve'] and v['status'] != 'algorithmic_failure']
    T = T_PLANNED if complete else (min(reached) if reached else 0)
    if deadline and not complete:
        import datetime
        if datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(DEADLINE_UTC):
            raise SystemExit(f'--deadline-readout is allowed only after {DEADLINE_UTC} (PLAN.md stopping rule)')
    out = {'plan_sha256': sha(HERE / 'PLAN.md') if (HERE / 'PLAN.md').exists() else None,
           'analysis_sha256': sha(__file__), 'provisional': not complete and not deadline,
           'deadline_readout': bool(deadline and not complete), 'readout_checkpoint': T,
           'rule_reference': {'weighted_J': J_rule, 'half': half, 'LONG_half_rule_threshold': LONG_HALF_RULE,
                              'sha256': sha(HERE / 'rule_reference.json')},
           'status': {f'{a}__b{b}': new[(a, b)]['status'] if (a, b) in new else 'absent' for a in ARMS for b in BLOCKS},
           'heads': {}}
    ps = []
    for arm in ARMS:
        role = e4.ROLES[arm]
        ok = [b for b in BLOCKS if (arm, b) in new and T in new[(arm, b)]['curve']]
        failed = [b for b in BLOCKS if (arm, b) in new and new[(arm, b)]['status'] == 'algorithmic_failure']
        jn = [new[(arm, b)]['curve'][T]['J'] for b in ok]
        jo = [old[(arm, b)]['curve'][T]['J'] for b in ok]
        h = {'role': role, 'blocks_with_value': ok, 'algorithmic_failures': failed}
        if len(ok) >= 2 and len(failed) < 3:
            h['P1_E4_minus_LONG_at_T'] = t_sum(np.subtract(jn, jo))
            ps.append((arm, h['P1_E4_minus_LONG_at_T']['p']))
        else:
            h['P1_E4_minus_LONG_at_T'] = None
        if failed:                                             # sensitivity: last evaluated J carried forward
            locf = [new[(arm, b)]['curve'][max(new[(arm, b)]['curve'])]['J'] if b in failed else
                    new[(arm, b)]['curve'][T]['J'] for b in BLOCKS if (arm, b) in new]
            h['P1_sensitivity_LOCF'] = t_sum(np.subtract(locf, [old[(arm, b)]['curve'][T]['J'] for b in BLOCKS
                                                                if (arm, b) in new]))
        h['P2_share_of_LONG_gap_to_rule_closed'] = gap_closed(jn, jo, J_rule) if ok else None
        h['P3'] = {'max_block_J_at_T': float(max(jn)) if jn else None,
                   'blocks_at_or_above_half_rule': [b for b, j in zip(ok, jn) if j >= half],
                   'blocks_at_or_above_LONG_threshold_0.103': [b for b, j in zip(ok, jn) if j >= LONG_HALF_RULE],
                   'mean_J_at_T': float(np.mean(jn)) if jn else None}
        h['E4_J_at_T'] = t_sum(jn)
        h['LONG_J_at_T'] = t_sum(jo)
        h['E4_minus_rule_at_T'] = t_sum(np.subtract(jn, J_rule)) if len(jn) >= 2 else None
        h['E4_share_of_rule_J'] = float(np.mean(jn) / J_rule) if jn else None
        # descriptive (no test; not in the Holm family)
        pts = [p for p in GRID if p <= T]
        h['curve_mean'] = {p: {'E4': float(np.mean([new[(arm, b)]['curve'][p]['J'] for b in ok])),
                               'LONG': float(np.mean([old[(arm, b)]['curve'][p]['J'] for b in ok]))} for p in pts}
        h['E4_minus_LONG_by_checkpoint'] = {p: t_sum([new[(arm, b)]['curve'][p]['J'] - old[(arm, b)]['curve'][p]['J']
                                                      for b in ok]) for p in pts}
        if T > 0 and ok:
            h['AUC_E4_minus_LONG'] = t_sum([auc(new[(arm, b)]['curve'], T) - auc(old[(arm, b)]['curve'], T) for b in ok])
            last3 = pts[-3:]
            h['slope_last3_per_30k_E4'] = t_sum([np.polyfit(last3, [new[(arm, b)]['curve'][p]['J'] for p in last3], 1)[0]
                                                  * 30000 for b in ok])
        if 30000 <= T and ok:
            h['J30k_E4_minus_LONG'] = t_sum([new[(arm, b)]['curve'][30000]['J'] - old[(arm, b)]['curve'][30000]['J']
                                             for b in ok])
        h['components_at_T'] = {src: {c: float(np.mean([runs[(arm, b)]['curve'][T][c] for b in ok]))
                                      for c in ('timely', 'false', 'fixes', 'decisions_per_episode')}
                                for src, runs in (('E4', new), ('LONG', old))} if ok else None
        h['rule_components'] = {'timely': float(np.mean([r['tp_timely'] for r in ref['episodes']])),
                                'false': float(np.mean([r['fp_spatial'] for r in ref['episodes']])),
                                'fixes': float(np.mean([r['fixes_attempted'] for r in ref['episodes']])),
                                'decisions_per_episode': float(np.mean([r['decisions'] for r in ref['episodes']]))}
        h['collapse'] = {src: {'no_timely_at_T': [b for b in ok if runs[(arm, b)]['curve'][T]['timely'] <= NO_TIMELY_CUT],
                               'last_update_entropy_below_0.30': [b for b in ok if runs[(arm, b)]['last_entropy'] is not None
                                                                  and runs[(arm, b)]['last_entropy'] < COLLAPSE_ENTROPY],
                               'last_update_entropy': {b: runs[(arm, b)]['last_entropy'] for b in ok}}
                         for src, runs in (('E4', new), ('LONG', old))}
        h['max_block_J_any_checkpoint_E4'] = max((new[(arm, b)]['curve'][p]['J'], b, p) for b in ok for p in pts) if ok else None
        h['evidence_stats'] = {b: new[(arm, b)]['ev'] for b in ok}
        h['last_update_lr'] = {b: new[(arm, b)]['last_lr'] for b in ok}
        out['heads'][role] = h
    rej = holm([p for _, p in ps]) if ps else []
    for (arm, _), r in zip(ps, rej):
        h = out['heads'][e4.ROLES[arm]]
        h['P1_E4_minus_LONG_at_T']['holm_rejected'] = r
        h['P1_reading'] = p1_reading(h['P1_E4_minus_LONG_at_T'], r)
    for h in out['heads'].values():
        h.setdefault('P1_reading', 'not computed: fewer than 4 blocks with a value (algorithmic failures)')
    # the systems reading (P3), fixed in the plan
    blocks_half = {r: h['P3']['blocks_at_or_above_half_rule'] for r, h in out['heads'].items()}
    means = {r: h['P3']['mean_J_at_T'] for r, h in out['heads'].items()}
    if any(m is not None and m >= half for m in means.values()):
        sys_read = ('withdrawn at 300,000 decisions for best-effort classical PPO: a head\'s mean J reached half the '
                    'rule (' + ', '.join(r for r, m in means.items() if m is not None and m >= half) + ')')
    elif any(blocks_half.values()):
        sys_read = ('restated: no learned controller approached the rule on average, but best-effort classical PPO '
                    'reached half the rule in ' + '; '.join(f'{r} block(s) {v}' for r, v in blocks_half.items() if v))
    else:
        sys_read = ('holds for best-effort, information-matched, re-tuned classical PPO at 300,000 decisions (every '
                    'block of both heads below half the paired rule reference)')
    out['systems_reading'] = sys_read
    out['not_shown_below_rule'] = [r for r, h in out['heads'].items()
                                   if h['E4_minus_rule_at_T'] and h['E4_minus_rule_at_T'].get('hi', -1) >= 0]
    name = 'e4_results.json' if (complete or deadline) else 'e4_results_PROVISIONAL.json'
    (HERE / name).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({'file': name, 'provisional': not complete, 'T': T, 'J_rule': J_rule,
                      'systems_reading': sys_read,
                      'P1': {r: (h['P1_E4_minus_LONG_at_T'] or {}).get('mean') for r, h in out['heads'].items()},
                      'P1_reading': {r: h['P1_reading'] for r, h in out['heads'].items()},
                      'P2': {r: (h['P2_share_of_LONG_gap_to_rule_closed'] or {}).get('estimate')
                             for r, h in out['heads'].items()}}, indent=1, default=float))


def selftest():
    """No E4 run is read. LONG's stored 300k means are recomputed through read_run and compared with
    long_results.json; the rule reference is verified and every stored LONG evaluation is checked to be paired."""
    stored = json.loads((e4.LONG_DIR / 'long_results.json').read_text())['LONG']
    ref = rule_reference()
    pairs = [(r['cell_id'], r['seed']) for r in ref['episodes']]
    for arm in ARMS:
        runs = [read_run(e4.LONG_DIR / 'runs' / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl') for b in BLOCKS]
        m = float(np.mean([r['curve'][300000]['J'] for r in runs]))
        want = stored[e4.ROLES[arm]]['J_at_T']['mean']
        print(arm, 'LONG J300k mean recomputed', m, 'stored', want)
        assert abs(m - want) < 1e-12, (m, want)
        assert all(row['pairs'] == pairs for r in runs for row in r['curve'].values())
    t = t_sum([0.01, 0.02, -0.005, 0.03, 0.0, 0.015])
    assert abs(t['mean'] - 0.0116666666) < 1e-9 and t['lo'] < t['mean'] < t['hi']
    assert holm([0.01, 0.04]) == [True, True] and holm([0.03, 0.04]) == [False, False] and holm([0.001, 0.2]) == [True, False]
    g = gap_closed([0.1] * 6, [0.0] * 6, 0.2)
    assert abs(g['estimate'] - 0.5) < 1e-12
    print('selftest passed; rule reference weighted J', ref['weighted_J'])


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        main('--provisional' in sys.argv, '--deadline-readout' in sys.argv)
