"""Analysis of E1 (learned correction trigger), exactly as fixed in PLAN.md before any main run.

    python gate_analysis.py [--provisional]   # writes gate_results.json. Refuses unless PLAN.md is hashed and every
                                              # script has the SHA-256 PLAN.md lists (config_check.bound_plan), and
                                              # unless every record carries this plan's SHA-256, a config equal to the
                                              # PLAN constants, the listed script hashes and the idea_dq tree
                                              # ffe9dfef...; waits for all 12 records unless --provisional
    python gate_analysis.py --selftest        # stored rule J of record re-derived; test statistics checked (no run read)
    python gate_analysis.py --dryrun          # main() on synthetic records in the scratchpad (no run read)

Primary (P1): per block b, D_b = cell-weighted J(gate, argmax, final) - cell-weighted J(rule) over the 80 paired
e1_evaluation episodes (frozen cell weights 0.2 / 0.05, p4_train.aggregate_evaluation; pairs checked on cell_id, seed
and crn_sha256; the 80 keys must equal the 80 stored keys). Across the completed blocks: mean, 95% paired t interval,
two-sided paired t p-value, dz; TOST at +-0.0056 J (two one-sided t tests at 0.05 = 90% interval inside the margin).
Holm over the primary family (one member). P1 is conditional on these 80 episodes (40 target-present): every block is
read on the same episodes, so the training seed is its only source of variation.

Episode-level interval (review B1): per target-present episode e, d_e = mean over the completed blocks of
(J_gate - J_rule); the cell-weighted paired t interval of sum_e w_e d_e with w_e = w_cell / n_cell (0.2 / 10, the
weights of the weighted J), computed as the one-sample t of u_e = n * w_e * d_e with df = n - 1 (= 0.8 x the ordinary
paired t interval of the d_e). Its point estimate equals the P1 mean exactly. A reading is stated beyond these episodes
only if this interval agrees with it (generalises()); otherwise it is stated as conditional on these episodes only.
S1 (the stress map) was dropped: it would have needed additional runs.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config_check as cc                                                                         # noqa: E402

DELTA = 0.0056          # 0.1 report-equivalent per episode (dq_b2.TOST_DELTA), fixed before any gate data
ALPHA = 0.05
MIN_BLOCKS = 8          # fewer completed blocks: P1 is 'not resolved (insufficient completed blocks)'
N_REF = 18
PLANNED_BLOCKS = len(cc.BLOCKS)
EVALUATION_EPISODES, TARGET_PRESENT_EPISODES = 80, 40
FINAL_STATUSES = ('completed', 'algorithmic_failure', 'infrastructure_failure')
INSUFFICIENT = 'not resolved (insufficient completed blocks)'
EQUIVALENT_GAIN = 'equivalent and better (a gain smaller than the margin)'
EQUIVALENT_LOSS = 'equivalent and worse (a loss smaller than the margin)'
PREFIX = 'On the 80 E1 evaluation episodes (40 target-present)'
SETTINGS = ('learning rate 1e-3, entropy coefficient 0.01, 10 epochs, minibatches of 64 transitions, clip 0.2, '
            '256 gate decisions per update, 12,800 gate decisions, trained from scratch')
APPROX_KL_LABEL = ("approx_kl is smdp_ppo's (dq_b2's) pre-step minibatch statistic: the mean over an update's "
                   "minibatch steps of the k3 estimate ((r - 1) - log r) of KL(rollout policy || current policy), each "
                   "taken before its optimiser step; it is not D_KL after the update")
A0_NOTE = ('A0 (freeze_a a0.json, +0.40 report-equivalents) is a conditional replay ceiling of never_fix trajectories '
           'with perfect georeferencing; it is not an upper bound for a gate that changes timing and is not called '
           'headroom.')


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
    """The pre-declared reading category (PLAN.md, 'Reading rules')."""
    if s['n'] < MIN_BLOCKS:
        return INSUFFICIENT
    if equivalent and diff_rejected:
        return EQUIVALENT_GAIN if s['mean'] > 0 else EQUIVALENT_LOSS
    if equivalent:
        return 'equivalent'
    if diff_rejected:
        return 'better' if s['mean'] > 0 else 'worse'
    return 'not resolved'


def episode_level(blocks_rows, weights, positive_only=True, delta=DELTA):
    """Review B1: the episode-level interval. blocks_rows = one list of 80 paired rows per completed block;
    weights = {cell_id: frozen cell weight}. d_e = mean over blocks of (J - rule_J); w_e = weight / n_cell;
    the one-sample t of u_e = n * w_e * d_e (mean = sum_e w_e d_e, df = n - 1), with the TOST at +-delta.
    positive_only: the 40 target-present episodes (the planned case: every no-human row equals the rule's);
    otherwise all 80 episodes with their weights (0.02 / 0.005)."""
    per_key = {}
    for rows in blocks_rows:
        for r in rows:
            per_key.setdefault((r['cell_id'], r['seed']), []).append(r['J'] - r['rule_J'])
    if any(len(v) != len(blocks_rows) for v in per_key.values()):
        raise SystemExit('episode-level interval: the blocks were not read on the same episodes')
    n_cell = {}
    for cell, _ in per_key:
        n_cell[cell] = n_cell.get(cell, 0) + 1
    keys = sorted(k for k in per_key if k[0].endswith('__positive') or not positive_only)
    d = np.array([float(np.mean(per_key[k])) for k in keys])
    w = np.array([weights[k[0]] / n_cell[k[0]] for k in keys])
    u = len(keys) * w * d
    out = paired(u, delta)
    out.update(episodes=len(keys), episodes_used='target-present' if positive_only else 'all 80 (no-human rows differ)',
               blocks=len(blocks_rows), weight_total=float(w.sum()), weights_equal=bool(np.ptp(w) == 0),
               d_mean_unweighted=float(d.mean()), d_sd=float(d.std(ddof=1)) if len(d) > 1 else None,
               share_episodes_d_negative=float(np.mean(d < 0)), share_episodes_d_zero=float(np.mean(d == 0)),
               method='one-sample t of u_e = n w_e d_e, df = n - 1; w_e = cell weight / episodes in the cell')
    return out


def generalises(category, ep, alpha=ALPHA):
    """Review B1 reading rule: a statement beyond these episodes needs the episode-level interval to agree with P1.
    better: its 95% interval > 0; worse: < 0; equivalent: its 90% interval inside +-delta (TOST p < alpha); a
    detectable gain/loss within the margin: both. 'not resolved' (either kind) states nothing beyond the episodes."""
    if ep is None:
        return False
    if category == 'better':
        return ep['lo95'] > 0
    if category == 'worse':
        return ep['hi95'] < 0
    if category == 'equivalent':
        return ep['tost_p'] < alpha
    if category == EQUIVALENT_GAIN:
        return ep['tost_p'] < alpha and ep['lo95'] > 0
    if category == EQUIVALENT_LOSS:
        return ep['tost_p'] < alpha and ep['hi95'] < 0
    return False


def _f(x):
    return f'{x:+.4f}'


def reading_text(category, P1, ep, agrees, counts):
    """The pre-declared sentence of each reading (PLAN.md, 'Reading rules'), its scope and the block tally."""
    if category == INSUFFICIENT:
        text = (f'{PREFIX}, P1 is not resolved: {counts["completed"]} of {PLANNED_BLOCKS} blocks completed, fewer '
                f'than the {MIN_BLOCKS} required.')
    else:
        m = f'mean {_f(P1["mean"])} J'
        ci95 = f'95% interval [{_f(P1["lo95"])}, {_f(P1["hi95"])}] J'
        ci90 = f'90% interval [{_f(P1["lo90"])}, {_f(P1["hi90"])}] J'
        if category == 'equivalent':
            text = (f"{PREFIX}, the learned correction trigger inside the rule's executive (PPO with settings fixed a "
                    f"priori and not tuned: {SETTINGS}) matches the tuned rule within 0.1 report-equivalent per "
                    f"episode (±0.0056 J): {m}, {ci90}, TOST p = {P1['tost_p']:.3g}. Learning recovers the tuned "
                    f"rule's performance on these episodes within the margin; it does not improve on it by more "
                    f"than the margin (±0.0056 J).")
        elif category in (EQUIVALENT_GAIN, EQUIVALENT_LOSS):
            kind = 'gain' if category == EQUIVALENT_GAIN else 'loss'
            text = (f"{PREFIX}, the learned trigger matches the tuned rule within ±0.0056 J (TOST p = "
                    f"{P1['tost_p']:.3g}), with a detectable {kind} smaller than the margin: {m}, {ci95}.")
        elif category == 'better':
            text = (f"{PREFIX}, the learned trigger exceeds the tuned rule: {m}, {ci95}. The gate sees 24 features "
                    f"and the rule 1 (the RMS error), so this shows only that a richer-information trigger beat the "
                    f"one-feature rule on these episodes; it does not show that the rule's threshold is suboptimal "
                    f"within its own one-feature class, nor that the gain comes from learning rather than from the "
                    f"extra information.")
        elif category == 'worse':
            text = (f"{PREFIX}, the learned trigger falls short of the tuned rule: {m}, {ci95}. The PPO settings were "
                    f"fixed a priori and not tuned ({SETTINGS}), whereas the rule's sigma was tuned on a grid; this "
                    f"shows that PPO at these settings and this budget did not recover the tuned trigger, not that no "
                    f"learned trigger could.")
        elif category == 'not resolved':
            text = (f"{PREFIX}, neither a difference (two-sided paired t, Holm, 0.05) nor equivalence within "
                    f"±0.0056 J (TOST, 0.05) was shown: {m}, {ci95}. This is not evidence of equality.")
        else:
            raise ValueError(category)
    if ep is None:
        scope = ' The reading is conditional on these episodes only.'
    else:
        e = (f'episode-level estimate {_f(ep["mean"])} J, 95% interval [{_f(ep["lo95"])}, {_f(ep["hi95"])}], '
             f'90% interval [{_f(ep["lo90"])}, {_f(ep["hi90"])}], {ep["episodes"]} episodes')
        if agrees:
            scope = (f' The episode-level interval agrees ({e}), so the reading is also stated for new episodes of '
                     f'these 8 cells; nothing is claimed beyond these cells and this one tuned comparator.')
        elif category in ('not resolved', INSUFFICIENT):
            scope = f' Nothing is stated beyond these episodes ({e}).'
        else:
            scope = (f' The episode-level interval does not agree ({e}), so the reading is conditional on these '
                     f'episodes only.')
    tally = (f' Blocks: {counts["completed"]} of {PLANNED_BLOCKS} completed; {counts["algorithmic_failure"]} ended in '
             f'an algorithmic failure and {counts["infrastructure_failure"]} in an infrastructure failure after one '
             f'retry')
    tally += f'; {counts["missing"]} have no record (provisional read-out).' if counts['missing'] else '.'
    return text + scope + tally


def rule_rows(suite='e1_evaluation'):
    import runner
    return runner.stored_rule_rows(suite)


def aggregation():
    import runner
    import p4_train as p4
    rt = runner.runtime(runner.STUDY_MAIN)
    return rt, p4.aggregate_evaluation


def cell_weights(rt):
    return {c['cell_id']: float(rt.probabilities[i]) for i, c in enumerate(rt.cells)}


RECORDS = {'dir': HERE / 'records', 'out': HERE / 'gate_results.json'}


def load_records(blocks):
    """{block: final record}; an unparsable file, or one that is not a final record of its block, is no record."""
    recs, unparsable = {}, []
    for b in blocks:
        f = RECORDS['dir'] / f'gate__b{b}.json'
        if not f.exists():
            continue
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            unparsable.append(b)
            continue
        if not isinstance(rec, dict) or rec.get('block') != b or rec.get('status') not in FINAL_STATUSES:
            unparsable.append(b)
            continue
        recs[b] = rec
    return recs, unparsable


def check_bound(rec, block, plan):
    """Review minor 6: a record enters only if it was produced under this plan (its SHA-256), with a config equal to
    the PLAN constants, the scripts PLAN.md lists and the idea_dq tree ffe9dfef...; anything else stops the analysis."""
    where = f'block {block}'
    if rec.get('plan_sha256') != plan['sha256']:
        raise SystemExit(f'{where}: record plan_sha256 {rec.get("plan_sha256")} is not this plan ({plan["sha256"]})')
    cfg = rec.get('config')
    if cfg is None:
        if rec['status'] == 'completed':
            raise SystemExit(f'{where}: a completed record without its config')
        return
    bad = cc.config_mismatches(cfg, block)
    if bad:
        raise SystemExit(f'{where}: the record config differs from PLAN.md: {bad}')
    if cfg.get('plan_sha256') != plan['sha256']:
        raise SystemExit(f'{where}: config plan_sha256 is not this plan')
    code = cfg.get('code_sha256') or {}
    if any(code.get(k) != v for k, v in plan['scripts'].items()):
        raise SystemExit(f'{where}: the run used scripts other than those PLAN.md lists')
    if rec['status'] == 'completed':
        if rec.get('code_tree_start') != cc.EXPECTED_TREE or rec.get('code_tree_end') != cc.EXPECTED_TREE:
            raise SystemExit(f'{where}: the idea_dq tree was not {cc.EXPECTED_TREE} for the whole run')
        if rec['summary']['gate_decisions'] < cc.PLAN_CONSTANTS['budget_gate_decisions']:
            raise SystemExit(f'{where}: completed below the budget')


def block_contrast(rec, stored, rt, agg):
    rows = rec['final_evaluation']['episodes']
    keys = [(r['cell_id'], r['seed']) for r in rows]
    if len(rows) != EVALUATION_EPISODES or len(set(keys)) != len(keys) or set(keys) != set(stored):
        raise SystemExit(f'block {rec["block"]}: the evaluation rows are not exactly the {EVALUATION_EPISODES} stored '
                         f'rule keys ({len(rows)} rows, {len(set(keys))} distinct)')
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


def sampled_contrast(rec, stored, weights):
    """The sampled read-out (descriptive only): the weighted gate-minus-rule difference over the 40 target-present
    e1_evaluation episodes (no-human episodes are not run: the gate is never asked there)."""
    s = rec.get('final_evaluation_sampled')
    if not s or 'error' in s:
        return {'available': False, 'error': (s or {}).get('error', 'no sampled read-out')}
    rows = s['episodes']
    positive = {k for k in stored if k[0].endswith('__positive')}
    keys = [(r['cell_id'], r['seed']) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != positive:
        return {'available': False, 'error': 'the sampled rows are not the 40 stored target-present keys'}
    if any(stored[k]['crn_sha256'] != r['crn_sha256'] or stored[k]['J'] != r['rule_J'] for k, r in zip(keys, rows)):
        return {'available': False, 'error': 'a sampled row is not paired with the stored rule'}
    cells = sorted({k[0] for k in keys})
    per_cell = {c: float(np.mean([r['J'] - r['rule_J'] for r in rows if r['cell_id'] == c])) for c in cells}
    return {'available': True, 'weighted_d_positive_cells': float(sum(weights[c] * per_cell[c] for c in cells)),
            'unweighted_mean_d': float(np.mean([r['J'] - r['rule_J'] for r in rows])), 'per_cell_d': per_cell,
            'fixes_per_episode': float(np.mean([r['fixes_attempted'] for r in rows]))}


def training_descriptives(rec):
    s = rec['summary']
    series = s.get('update_series') or []
    p_after = [u['p_fix_after_update'] for u in series]
    return {**{k: s.get(k) for k in ('episodes', 'gate_decisions', 'updates', 'infrastructure_episodes',
                                     'empty_episodes', 'pre_gate_reward', 'dropped_reward', 'wall_s')},
            'approx_kl_pre_step_minibatch_mean_per_update': [u['approx_kl_pre_step_minibatch_mean'] for u in series],
            'p_fix_after_update_rms_lt_1.5': [p['rms_lt_1.5'] for p in p_after],
            'p_fix_after_update_rms_ge_1.5': [p['rms_ge_1.5'] for p in p_after],
            'rollout_fix_rate_per_update': [u['rollout_fix_rate'] for u in series],
            'clip_fraction_per_update': [u['clip_fraction'] for u in series],
            'explained_variance_per_update': [u['explained_variance'] for u in series]}


def _write(out):
    RECORDS['out'].write_text(json.dumps(out, indent=1, default=float))


def main(provisional=False, plan=None, quiet=False):
    import runner
    plan = cc.bound_plan(HERE) if plan is None else plan
    blocks = runner.BLOCKS
    recs, unparsable = load_records(blocks)
    for b, rec in recs.items():
        check_bound(rec, b, plan)
    complete = len(recs) == len(blocks)
    if not complete and not provisional:
        raise SystemExit(f'incomplete: {len(recs)} of {len(blocks)} records; the read-out waits for all of them')
    done = {b: r for b, r in recs.items() if r['status'] == 'completed'}
    counts = {'planned': len(blocks), 'recorded': len(recs), 'completed': len(done),
              'algorithmic_failure': sum(1 for r in recs.values() if r['status'] == 'algorithmic_failure'),
              'infrastructure_failure': sum(1 for r in recs.values() if r['status'] == 'infrastructure_failure'),
              'missing': len(blocks) - len(recs), 'unparsable_records': unparsable,
              'retried_after_infrastructure_failure': sorted(b for b, r in recs.items() if r.get('previous_attempts'))}
    out = {'plan_sha256': plan['sha256'], 'analysis_sha256': cc.sha256(Path(__file__).resolve()),
           'provisional': not complete, 'counts': counts, 'blocks_completed': sorted(done),
           'blocks_not_completed': {b: recs[b]['status'] for b in recs if b not in done}, 'A0_note': A0_NOTE,
           'S1': 'dropped in PLAN.md: it would have needed additional runs'}
    if len(done) < 2:                                    # review minor 13: the reading is written anyway
        P1 = {'n': len(done), 'reading': INSUFFICIENT, 'generalises': False, 'episode_level': None,
              'scope': 'these 80 E1 evaluation episodes only', 'blocks': counts}
        P1['reading_text'] = reading_text(INSUFFICIENT, P1, None, False, counts)
        out['P1_gate_minus_rule_e1_evaluation'] = P1
        _write(out)
        if not quiet:
            print(json.dumps({'provisional': out['provisional'], 'P1': P1}, indent=1, default=float))
        return out
    stored = rule_rows()
    if len(stored) != EVALUATION_EPISODES:
        raise SystemExit(f'{len(stored)} stored rule rows, not {EVALUATION_EPISODES}')
    rt, agg = aggregation()
    weights = cell_weights(rt)
    per_block = [block_contrast(done[b], stored, rt, agg) for b in sorted(done)]
    P1 = paired([x['D'] for x in per_block])
    P1['holm_rejected'] = holm([P1['p_two_sided']])[0]
    P1['equivalent'] = P1['tost_p'] < ALPHA
    P1['reading'] = reading(P1, P1['holm_rejected'], P1['equivalent'])
    no_human_identical = all(x['no_human_identical_to_rule'] for x in per_block)
    ep = episode_level([done[b]['final_evaluation']['episodes'] for b in sorted(done)], weights,
                       positive_only=no_human_identical)
    if abs(ep['mean'] - P1['mean']) > 1e-12:
        raise SystemExit(f'internal: episode-level estimate {ep["mean"]!r} differs from the P1 mean {P1["mean"]!r}')
    P1['episode_level'] = ep
    P1['no_human_identical_to_rule_all_blocks'] = no_human_identical
    P1['generalises'] = generalises(P1['reading'], ep)
    P1['scope'] = ('these 80 E1 evaluation episodes and new episodes of these 8 cells' if P1['generalises'] else
                   'these 80 E1 evaluation episodes only')
    P1['blocks'] = counts
    P1['reading_text'] = reading_text(P1['reading'], P1, ep, P1['generalises'], counts)
    out['P1_gate_minus_rule_e1_evaluation'] = P1
    sampled = {b: sampled_contrast(done[b], stored, weights) for b in sorted(done)}
    ok = [sampled[b]['weighted_d_positive_cells'] for b in sorted(done) if sampled[b]['available']]
    out['descriptive'] = {
        'per_block': per_block,
        'mean_unweighted_80_episode_d': paired([x['unweighted_mean_d'] for x in per_block]),
        'per_cell_d_mean_over_blocks': {c: float(np.mean([x['per_cell_d'][c] for x in per_block]))
                                        for c in per_block[0]['per_cell_d']},
        'fixes_per_positive_episode_gate_mean': float(np.mean([x['fixes_per_positive_episode']['gate']
                                                               for x in per_block])),
        'fixes_per_positive_episode_rule': per_block[0]['fixes_per_positive_episode']['rule'],
        'fp_per_positive_episode_gate_mean': float(np.mean([x['fp_per_positive_episode']['gate'] for x in per_block])),
        'sampled_readout': {'note': 'descriptive only; target-present e1_evaluation episodes; no test, no reading',
                            'per_block': sampled,
                            'across_blocks': paired(ok) if len(ok) >= 2 else None},
        'learning_curve_dev_weighted_d': {b: done[b]['summary']['learning_curve'] for b in sorted(done)},
        'training': {b: training_descriptives(done[b]) for b in sorted(done)},
        'approx_kl_label': APPROX_KL_LABEL}
    _write(out)
    if quiet:
        return out
    print(json.dumps({'provisional': out['provisional'], 'counts': counts,
                      'P1': {k: P1[k] for k in ('n', 'mean', 'lo95', 'hi95', 'p_two_sided', 'tost_p', 'reading',
                                                'generalises', 'reading_text')}}, indent=1, default=float))
    return out


# ------------------------------------------------------------------ synthetic records (selftest, dryrun, tests) ------

def synthetic_record(block, stored, rng, shift=0.0, noise=0.02, plan_sha='synthetic-plan', status='completed',
                     no_human_shift=0.0):
    """A record shaped as runner.launch writes it, built from the STORED RULE rows (gate J = rule J + noise on the
    target-present cells). No run is read."""
    import runner
    cfg = dict(runner.config(block), plan_sha256=plan_sha)
    rec = {'block': block, 'attempt': 1, 'status': status, 'plan_sha256': plan_sha, 'config': cfg,
           'code_tree_start': cc.EXPECTED_TREE, 'code_tree_end': cc.EXPECTED_TREE, 'previous_attempts': []}
    if status != 'completed':
        rec['summary'] = {'status': status, 'failure': {'kind': 'FloatingPointError'}} \
            if status == 'algorithmic_failure' else None
        rec['error'] = None if status == 'algorithmic_failure' else 'OSError: synthetic'
        return rec
    rows = []
    for (cell, seed), s in stored.items():
        pos = cell.endswith('__positive')
        J = s['J'] + (shift + rng.normal(0, noise) if pos else no_human_shift)
        rows.append(dict(s, rule_J=s['J'], d_J=J - s['J'], J=J, rule_fixes=s['fixes_attempted'],
                         rule_fp=s['fp_spatial'], rule_tp_timely=s['tp_timely'], infrastructure=False,
                         gate_valid_rows=8 if pos else 0,
                         gates=[[100.0 * k, int(rng.random() < 0.5), float(rng.uniform(0, 3))]
                                for k in range(8 if pos else 0)]))
    sampled = [dict(r, J=r['J'] + rng.normal(0, noise)) for r in rows if r['cell_id'].endswith('__positive')]
    for r in sampled:
        r['d_J'] = r['J'] - r['rule_J']
    series = [{'update': k + 1, 'gate_decisions': 256 * (k + 1), 'episodes': 41,
               'approx_kl_pre_step_minibatch_mean': float(rng.uniform(0, 0.02)), 'clip_fraction': 0.1,
               'entropy': 0.5, 'explained_variance': 0.5, 'rollout_fix_rate': 0.5,
               'p_fix_after_update': {'mean': 0.5, 'rms_lt_1.5': 0.4, 'rms_ge_1.5': 0.6, 'n_rms_lt_1.5': 100,
                                      'n_rms_ge_1.5': 156}} for k in range(50)]
    rec.update(final_evaluation={'episodes': rows, 'n': len(rows)},
               final_evaluation_sampled={'readout': 'sampled', 'episodes': sampled, 'n': len(sampled)},
               summary={'status': 'completed', 'learning_curve': [], 'episodes': 2000, 'gate_decisions': 12_800,
                        'updates': 50, 'infrastructure_episodes': 0, 'empty_episodes': 400, 'pre_gate_reward': 0.0,
                        'dropped_reward': 0.0, 'wall_s': 1.0, 'update_series': series})
    return rec


def synthetic_plan(plan_sha='synthetic-plan'):
    import runner
    return {'sha256': plan_sha, 'scripts': runner.code_hashes()}


def scratch_dir():
    """The scratch directory for synthetic output: this task's scratchpad folder if present, else the system temp."""
    import tempfile
    for p in (Path('/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad/fix_e1_gate'),
              Path('/tmp/claude-1000/-home-konstantinos/8edd25a0-e7ee-45e2-8e79-5d8e3e867e92/scratchpad')):
        if p.is_dir():
            return p
    return Path(tempfile.gettempdir())


def selftest():
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
    assert reading({'n': 7, 'mean': 0.0}, False, True) == INSUFFICIENT
    assert not generalises('not resolved', {'lo95': 1, 'hi95': 2, 'tost_p': 0.0})
    print(json.dumps({'selftest': 'passed', 'rule_weighted_J_e1_evaluation': w, 'synthetic': s}, indent=1,
                     default=float))


def dryrun():
    """The whole read-out on SYNTHETIC records built from the stored rule rows, written to a scratch directory;
    exercises every code path of main() without any run."""
    import tempfile
    import runner
    stored = rule_rows()
    rng = np.random.default_rng(7)
    tmp = Path(tempfile.mkdtemp(dir=scratch_dir()))
    (tmp / 'records').mkdir()
    for b in runner.BLOCKS:
        rec = synthetic_record(b, stored, rng, shift=rng.normal(-0.002, 0.004))
        (tmp / 'records' / f'gate__b{b}.json').write_text(json.dumps(rec, default=float))
    RECORDS.update(dir=tmp / 'records', out=tmp / 'gate_results.json')
    main(False, plan=synthetic_plan())
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
