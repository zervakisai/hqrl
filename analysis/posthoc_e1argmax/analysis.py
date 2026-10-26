"""Analysis of the post-hoc E1 argmax read-out, exactly as fixed in PLAN.md. Run ONCE, after all 48 records exist.

    <E1 python> -B analysis.py              # writes argmax_results.json once (refuses if it exists)
    <E1 python> -B analysis.py --selftest   # statistics checked against scipy; the rule's J of record, its timely
                                            # credited reports and Table 5's sampled values re-derived (no record read)
    <E1 python> -B analysis.py --dryrun     # main() on SYNTHETIC records in a scratch directory (no record read)

The real read-out refuses unless PLAN.md is hashed and every script has the SHA-256 PLAN.md lists
(runner.bound_plan), precheck.json (the precheck of record, untagged) passed for this plan and these scripts, the 48
policies re-prove through the frozen E1 code with the table digest PLAN.md lists, and every one of the 48 records
exists, is a completed argmax read-out under this plan (runner.check_record: plan, scripts and both code trees at start
and end, checkpoint, the 80 episode keys in the stored order), re-derives its weighted J by
p4_train.aggregate_evaluation, and - for block 10 of each family - reproduces the precheck's outcome-blind argmax digest.
There is no provisional or partial read-out.

Quantities (per family f, blocks b = 10..21; all J are frozen-cell-weighted, p4_train.aggregate_evaluation):
  J_arg(f,b)   weighted J of the argmax read-out of the final policy on the 80 e1_evaluation episodes
  J_smp(f,b)   the stored E1 sampled read-out of the same policy at 30,000 decisions (Table 5's), from the episode rows
               proven by e1_study.proven_stage; the precheck re-ran all 48 of them exactly through this study's path
  J_rule       0.20564463190957744, the P3 rule on the same 80 episodes (baselines.json), a constant
  family mean over blocks of J_arg with the 95 % t interval (df 11); pooled per-episode timely credited reports
               (e1_report.credited: tp_timely if the episode returned, else 0) and false reports (fp_spatial), as in
               Table 5 (pooled over 960 episodes = the mean of the 12 block means)
  D_as(f,b)    J_arg - J_smp, paired by block: mean, 95 % t interval (descriptive)
  D_rule(f,b)  J_arg - J_rule: mean, 95 % t interval, two-sided one-sample t p; Holm over the 4 families (R-A)
  episode level (descriptive, as in the gate study): d_e = mean_b J_arg(f,b,e) - J_rule(e) per episode, the
               one-sample t interval of u_e = 80 w_e d_e (w_e = cell weight / 10, df 79), whose mean equals mean D_rule
  R-B          rule credited / best family's argmax credited (best = highest pooled point estimate), with a paired
               block bootstrap (10,000 resamples, seed 20261026; the same resampled blocks for every family; the best
               family re-chosen in every resample) of best/rule, inverted to a ratio interval
The block-level intervals condition on the 80 episodes: every block is read on the same episodes and the rule's J is a
constant on them, so they hold training-seed variation only, not episode-sampling variation.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json
import math
import os
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner                                                                                     # noqa: E402

import numpy as np                                                                                # noqa: E402
from scipy import stats                                                                           # noqa: E402

import p4_train as p4                                                                             # noqa: E402

VERSION = 'e1-argmax-posthoc-analysis/1'
ALPHA = 0.05
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10_000, 20261026
RATIO_THRESHOLDS = (2.0, 3.0)
N_EPISODES, N_BLOCKS = 80, 12
#: Table 5's sampled values of record (e1_report_v1: the mean over blocks of each family's weighted J at 30,000
#: decisions); --selftest re-derives them from the stored rows. Not used by the read-out.
TABLE5_SAMPLED_J = {'quantum': 0.024285791296910626, 'bottleneck': 0.03743572490896784, 'mlp': 0.052016273841715234,
                    'surrogate625': 0.044911764424285364}
E1_REPORT = runner.IDEA_E1 / 'output/study/gnss_qrl_v1/e1/e1_report_v1/report.json'
PANEL = ('credited_timely', 'tp_timely', 'tp_spatial', 'fp_spatial', 'return_ok', 'fixes_attempted', 'decisions')
PATHS = {'records': runner.RECORDS, 'out': HERE / 'argmax_results.json', 'precheck': HERE / 'precheck.json'}
SCOPE = ('Post hoc: planned and run after the referees\' comments on Table 5. Evaluation only: the 48 E1 selected final '
         'policies (4 families x blocks 10-21) are read out with argmax instead of sampled actions on the same 80 '
         'e1_evaluation episodes; no training, no selection, nothing else changed. The block-level intervals '
         'condition on these 80 episodes (every block is read on the same episodes and the rule\'s J of record, '
         '0.20564463190957744, is a constant on them): they hold training-seed variation only, not episode-sampling '
         'variation. The episode-level interval is descriptive. No re-run with changed settings is reported as this '
         'study.')


class Refused(SystemExit):
    pass


def credited(e):
    return runner.e1_report_credited(e)


# ------------------------------------------------------------------ statistics -----------------------------------------

def t_summary(values, alpha=ALPHA):
    """Mean, sd, se, the (1 - alpha) t interval and the two-sided one-sample t p against 0 (df n - 1)."""
    values = [float(v) for v in values]
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 else float('nan')
    q = float(stats.t.ppf(1 - alpha / 2, n - 1)) if n > 1 else float('nan')
    if n > 1 and se > 0:
        p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    elif n > 1:
        p = 1.0 if mean == 0 else 0.0
    else:
        p = float('nan')
    return {'n': n, 'mean': mean, 'sd': sd, 'se': se, 'lo95': mean - q * se, 'hi95': mean + q * se,
            'p_two_sided': p, 'df': n - 1}


def holm_adjusted(ps):
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out, running = [0.] * m, 0.
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def reading_ra(adjusted_p, mean, alpha=ALPHA):
    """R-A: 'holds' (the gap holds under argmax), 'reversed', or 'not resolved' - Holm over the 4 families."""
    if adjusted_p <= alpha and mean < 0:
        return 'holds'
    if adjusted_p <= alpha and mean > 0:
        return 'reversed'
    return 'not resolved'


def reading_rb(point, lo, hi, k):
    """R-B for 'the rule earns at least k x the best family's timely credited reports'."""
    if point >= k and lo >= k:
        return 'holds'
    if point < k and hi < k:
        return 'fails'
    return 'not resolved'


def inverse(q):
    return math.inf if q == 0 else 1.0 / q


def ratio_bootstrap(block_credited, rule_credited, families, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Paired block bootstrap of best/rule (best re-chosen per resample), inverted to an interval of rule/best."""
    data = np.asarray([block_credited[f] for f in families], float)            # families x blocks
    n = data.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    best = data[:, idx].mean(axis=2).max(axis=0)                               # resamples
    q = best / rule_credited
    q_lo, q_hi = (float(x) for x in np.quantile(q, [0.025, 0.975]))
    return {'resamples': resamples, 'seed': seed, 'method': 'paired percentile bootstrap over blocks; the best family '
            're-chosen in every resample; interval of best/rule inverted', 'share_lo': q_lo, 'share_hi': q_hi,
            'ratio_lo': inverse(q_hi), 'ratio_hi': inverse(q_lo),
            'best_family_frequency': {f: float(np.mean(data[:, idx].mean(axis=2).argmax(axis=0) == i))
                                      for i, f in enumerate(families)}}


def episode_level(arg_rows_by_block, rule_rows, weights):
    """Descriptive episode-level interval of mean_b J_arg - J_rule: u_e = n w_e d_e, one-sample t, df n - 1."""
    keys = [(e['cell_id'], e['seed']) for e in rule_rows]
    n = len(keys)
    d = []
    for i, (cell, _) in enumerate(keys):
        arg = statistics.mean(float(rows[i]['J']) for rows in arg_rows_by_block)
        d.append(arg - float(rule_rows[i]['J']))
    u = [n * weights[cell] * x for (cell, _), x in zip(keys, d)]
    s = t_summary(u)
    s['method'] = 'u_e = n w_e d_e, w_e = cell weight / episodes of the cell, one-sample t (df n - 1)'
    return s


# ------------------------------------------------------------------ panels ---------------------------------------------

def panel(episodes):
    rows = [dict(e, credited_timely=credited(e)) for e in episodes]
    out = {k: statistics.mean(float(r[k]) for r in rows) for k in PANEL}
    out['episodes'] = len(rows)
    out['termination_reasons'] = {}
    for r in rows:
        out['termination_reasons'][r['termination_reason']] = out['termination_reasons'].get(r['termination_reason'], 0) + 1
    return out


def block_mean(episodes, key):
    return statistics.mean(credited(e) if key == 'credited_timely' else float(e[key]) for e in episodes)


# ------------------------------------------------------------------ the read-out ---------------------------------------

def compute(table, arg_records, rule_rows, rt, rule_j=runner.RULE_J_OF_RECORD):
    """Every number of the read-out from the 48 argmax records, the 48 stored sampled read-outs and the rule rows."""
    weights = {c['cell_id']: float(rt.probabilities[i]) / sum(1 for e in rule_rows if e['cell_id'] == c['cell_id'])
               for i, c in enumerate(rt.cells)}
    rule_credited = statistics.mean(credited(e) for e in rule_rows)
    by = {(e['family'], e['block']): e for e in table}
    families = {}
    for f in runner.FAMILIES:
        blocks = list(runner.BLOCKS)
        arg_eps = [arg_records[by[(f, b)]['job']]['episodes'] for b in blocks]
        smp_eps = [by[(f, b)]['stored']['episodes'] for b in blocks]
        j_arg = [p4.aggregate_evaluation(rt, eps)[1] for eps in arg_eps]
        j_smp = [p4.aggregate_evaluation(rt, eps)[1] for eps in smp_eps]
        d_as = [a - s for a, s in zip(j_arg, j_smp)]
        d_rule = [a - rule_j for a in j_arg]
        cred_arg = [block_mean(eps, 'credited_timely') for eps in arg_eps]
        cred_smp = [block_mean(eps, 'credited_timely') for eps in smp_eps]
        fp_arg = [block_mean(eps, 'fp_spatial') for eps in arg_eps]
        fp_smp = [block_mean(eps, 'fp_spatial') for eps in smp_eps]
        pooled_arg = panel([e for eps in arg_eps for e in eps])
        pooled_smp = panel([e for eps in smp_eps for e in eps])
        if abs(pooled_arg['credited_timely'] - statistics.mean(cred_arg)) > 1e-12:
            raise RuntimeError('pooled credited reports differ from the mean of the block means')
        families[f] = {
            'role': runner.ROLE[f], 'candidate': runner.EXPECTED_CANDIDATES[f], 'blocks': blocks,
            'argmax_weighted_J': t_summary(j_arg), 'sampled_weighted_J': t_summary(j_smp),
            'argmax_minus_sampled': dict(t_summary(d_as), blocks_argmax_higher=sum(x > 0 for x in d_as),
                                         blocks_equal=sum(x == 0 for x in d_as), label='descriptive'),
            'argmax_minus_rule': dict(t_summary(d_rule), rule_weighted_J=rule_j),
            'episode_level_argmax_minus_rule': episode_level(arg_eps, rule_rows, weights),
            'panel_argmax_pooled': pooled_arg, 'panel_sampled_pooled': pooled_smp,
            'credited_timely_argmax_blocks': t_summary(cred_arg), 'fp_spatial_argmax_blocks': t_summary(fp_arg),
            'credited_timely_argmax_minus_sampled': t_summary([a - s for a, s in zip(cred_arg, cred_smp)]),
            'fp_spatial_argmax_minus_sampled': t_summary([a - s for a, s in zip(fp_arg, fp_smp)]),
            'per_block': [{'block': b, 'argmax_J': ja, 'sampled_J': js, 'argmax_credited_timely': ca,
                           'sampled_credited_timely': cs, 'argmax_fp_spatial': fa, 'sampled_fp_spatial': fs}
                          for b, ja, js, ca, cs, fa, fs in zip(blocks, j_arg, j_smp, cred_arg, cred_smp, fp_arg,
                                                               fp_smp)],
            '_cred_blocks': cred_arg}
    # R-A: Holm over the four family contrasts learned(argmax) - rule
    names = list(runner.FAMILIES)
    adjusted = holm_adjusted([families[f]['argmax_minus_rule']['p_two_sided'] for f in names])
    for f, p_adj in zip(names, adjusted):
        c = families[f]['argmax_minus_rule']
        c['holm_adjusted_p'] = p_adj
        c['reading_R_A'] = reading_ra(p_adj, c['mean'])
        c['interval_below_0'] = c['hi95'] < 0
        ep = families[f]['episode_level_argmax_minus_rule']
        c['episode_level_agrees'] = (ep['hi95'] < 0 if c['reading_R_A'] == 'holds' else
                                     ep['lo95'] > 0 if c['reading_R_A'] == 'reversed' else None)
    # R-B: the rule's timely credited reports against the best family's under argmax
    point = {f: families[f]['panel_argmax_pooled']['credited_timely'] for f in names}
    best = max(names, key=lambda f: (point[f], -names.index(f)))
    ratio = inverse(point[best] / rule_credited)
    boot = ratio_bootstrap({f: families[f].pop('_cred_blocks') for f in names}, rule_credited, names)
    smp_point = {f: families[f]['panel_sampled_pooled']['credited_timely'] for f in names}
    smp_best = max(names, key=lambda f: (smp_point[f], -names.index(f)))
    rb = {'rule_credited_timely_per_episode': rule_credited, 'rule_fp_spatial_per_episode':
          statistics.mean(float(e['fp_spatial']) for e in rule_rows),
          'best_family_argmax': best, 'best_role_argmax': runner.ROLE[best],
          'best_credited_timely_argmax': point[best], 'ratio_rule_over_best': ratio, 'bootstrap': boot,
          'readings': {f'{k:g}x': reading_rb(ratio, boot['ratio_lo'], boot['ratio_hi'], k) for k in RATIO_THRESHOLDS},
          'per_family_ratio_argmax': {f: inverse(point[f] / rule_credited) for f in names},
          'sampled_reference': {'best_family': smp_best, 'best_credited_timely': smp_point[smp_best],
                                'ratio_rule_over_best': inverse(smp_point[smp_best] / rule_credited)},
          'rule': 'holds if the point ratio >= k and the bootstrap lower bound >= k; fails if the point ratio < k and '
                  'the upper bound < k; otherwise not resolved'}
    return {'families': families, 'R_A_family_contrasts': names, 'R_B': rb,
            'rule_weighted_J': rule_j, 'reading_text': reading_text(families, rb)}


def _n(x, d=3):
    return 'inf' if x == math.inf else f'{x:.{d}f}'


def _s(x, d=3):
    return 'inf' if x == math.inf else f'{x:+.{d}f}'


def reading_text(families, rb):
    """The sentences of record, generated from the numbers (no bracketed integer intervals)."""
    out = []
    for f, v in families.items():
        a, s, d, r = (v['argmax_weighted_J'], v['sampled_weighted_J'], v['argmax_minus_sampled'],
                      v['argmax_minus_rule'])
        ep = v['episode_level_argmax_minus_rule']
        verdict = {'holds': 'the systems gap holds under the argmax read-out',
                   'reversed': 'the systems gap is reversed under the argmax read-out',
                   'not resolved': 'the systems gap is not resolved under the argmax read-out'}[r['reading_R_A']]
        scope = ('; the episode-level interval agrees' if r['episode_level_agrees'] else
                 '; the episode-level interval does not agree, so this holds for these 80 episodes only'
                 if r['episode_level_agrees'] is False else '')
        out.append(f'{v["role"]} ({f}), argmax: weighted J {_n(a["mean"])} (95% t interval {_n(a["lo95"])} to '
                   f'{_n(a["hi95"])}, 12 blocks; sampled {_n(s["mean"])}; argmax minus sampled {_s(d["mean"])}, '
                   f'{_s(d["lo95"])} to {_s(d["hi95"])}, descriptive). Minus the rule (0.206 on the same 80 episodes): '
                   f'{_s(r["mean"])} ({_s(r["lo95"])} to {_s(r["hi95"])}), Holm-adjusted p = {r["holm_adjusted_p"]:.2g}'
                   f': {verdict}{scope} (episode-level {_s(ep["lo95"])} to {_s(ep["hi95"])}). Per episode: '
                   f'{_n(v["panel_argmax_pooled"]["credited_timely"], 2)} timely credited and '
                   f'{_n(v["panel_argmax_pooled"]["fp_spatial"], 2)} false reports (sampled '
                   f'{_n(v["panel_sampled_pooled"]["credited_timely"], 2)} and '
                   f'{_n(v["panel_sampled_pooled"]["fp_spatial"], 2)}).')
    b = rb['bootstrap']
    out.append(f'R-B: the rule earns {_n(rb["rule_credited_timely_per_episode"], 2)} timely credited reports per episode, '
               f'{_n(rb["ratio_rule_over_best"], 2)} times the best family under argmax ({rb["best_role_argmax"]}, '
               f'{_n(rb["best_credited_timely_argmax"], 2)}; block bootstrap {_n(b["ratio_lo"], 2)} to '
               f'{_n(b["ratio_hi"], 2)}); at least 2x: {rb["readings"]["2x"]}; at least 3x: {rb["readings"]["3x"]} '
               f'(sampled: {_n(rb["sampled_reference"]["ratio_rule_over_best"], 2)} times '
               f'{runner.ROLE[rb["sampled_reference"]["best_family"]]}).')
    out.append('These block-level intervals condition on the 80 e1_evaluation episodes (post hoc, evaluation only).')
    return out


# ------------------------------------------------------------------ inputs and refusals --------------------------------

def load_records(table, records_dir):
    recs, missing, unparsable = {}, [], []
    for e in table:
        path = Path(records_dir) / f'{e["job"]}.json'
        if not path.exists():
            missing.append(e['job'])
            continue
        try:
            recs[e['job']] = json.loads(path.read_text())
        except (OSError, ValueError):
            unparsable.append(e['job'])
    return recs, missing, unparsable


def check_inputs(table, recs, plan, rt, precheck):
    """Refuse (SystemExit) on any record that is not this plan's completed argmax read-out, or that does not re-derive
    its J, or (block 10) does not reproduce the precheck's outcome-blind argmax digest."""
    for e in table:
        rec = recs[e['job']]
        problems = runner.check_record(rec, e, plan['sha256'], plan['scripts'])
        if problems:
            raise Refused(f'{e["job"]}: {problems}')
        per_cell, weighted = p4.aggregate_evaluation(rt, rec['episodes'])
        if runner.canonical([per_cell, weighted]) != runner.canonical([rec['per_cell_J'], rec['weighted_J']]):
            raise Refused(f'{e["job"]}: its stored weighted J is not its episodes\'')
    probe = precheck['d_argmax_probe']['families']
    checks = {}
    for f in runner.FAMILIES:
        e = next(x for x in table if x['family'] == f and x['block'] == runner.PROBE_BLOCK)
        rows = {(r['cell_id'], r['seed']): r for r in recs[e['job']]['episodes']}
        keys = [tuple(k) for k in probe[f]['argmax_keys']]
        digest = runner.rows_digest([rows[k] for k in keys])
        checks[f] = digest == probe[f]['argmax_digest']
        if not checks[f]:
            raise Refused(f'{e["job"]}: the argmax rows of the probe episodes differ from the precheck\'s (the argmax '
                          'read-out did not reproduce across processes)')
    return checks


def main(plan=None, precheck=None, ctx=None, table=None, paths=None, quiet=False):
    paths = dict(PATHS, **(paths or {}))
    out = Path(paths['out'])
    if out.exists():
        raise Refused(f'{out} exists: the read-out is run once')
    plan = runner.bound_plan() if plan is None else plan
    if precheck is None:
        pf = Path(paths['precheck'])
        precheck = json.loads(pf.read_text()) if pf.exists() else {}
        if not (precheck.get('passed') and precheck.get('plan_sha256') == plan['sha256'] and precheck.get('tag') == ''
                and precheck.get('code_sha256') == plan['scripts']):
            raise Refused('precheck.json (the precheck of record) has not passed for this plan and these scripts')
    ctx = runner.e1_context() if ctx is None else ctx
    table = runner.policy_table(ctx) if table is None else table
    digest = runner.table_digest(table)
    if digest != plan['policy_table_digest']:
        raise Refused(f'the policy table digest {digest} is not the one PLAN.md lists')
    recs, missing, unparsable = load_records(table, paths['records'])
    if missing or unparsable:
        raise Refused(f'incomplete: {len(recs)} of 48 records ({len(missing)} missing, {len(unparsable)} unparsable); '
                      'the read-out waits for all 48 (no partial read-out)')
    rt = ctx['runtime']
    rule = ctx['baselines']['p3_rule/e1_evaluation']
    if runner.sha(runner.BASELINES) != runner.BASELINES_SHA256:
        raise Refused('baselines.json is not the file of record')
    _, rule_j = p4.aggregate_evaluation(rt, rule['episodes'])
    if repr(rule_j) != repr(runner.RULE_J_OF_RECORD):
        raise Refused(f'the rule\'s weighted J {rule_j!r} is not the J of record')
    keys = [(e['cell_id'], e['seed']) for e in rule['episodes']]
    if any([(x['cell_id'], x['seed']) for x in e['stored']['episodes']] != keys for e in table):
        raise Refused('the rule rows and the stored E1 read-outs are not the same 80 episodes in the same order')
    probe_checks = check_inputs(table, recs, plan, rt, precheck)
    result = compute(table, recs, rule['episodes'], rt, rule_j)
    result = {'version': VERSION, 'utc': p4.utc(), 'scope': SCOPE, 'plan_sha256': plan['sha256'],
              'scripts': plan['scripts'], 'policy_table_digest': digest, 'precheck_utc': precheck.get('utc'),
              'idea_e1_tree': runner.tree(runner.IDEA_E1), 'idea_dq_tree': runner.tree(runner.IDEA_DQ),
              'baselines_sha256': runner.sha(runner.BASELINES),
              'records_sha256': {e['job']: runner.sha(Path(paths['records']) / f'{e["job"]}.json') for e in table},
              'probe_digests_reproduced': probe_checks, 'synthetic': bool(paths.get('synthetic')), **result}
    runner.write_json_atomic(out, result)
    if not quiet:
        print(json.dumps({'R_A': {f: {k: v['argmax_minus_rule'][k] for k in ('mean', 'lo95', 'hi95', 'holm_adjusted_p',
                                                                            'reading_R_A', 'episode_level_agrees')}
                                  for f, v in result['families'].items()},
                          'R_B': {k: result['R_B'][k] for k in ('best_family_argmax', 'ratio_rule_over_best',
                                                                'readings')},
                          'reading_text': result['reading_text']}, indent=1, default=float))
        print('written', out)
    return result


# ------------------------------------------------------------------ synthetic records (selftest, dryrun, tests) --------

def synthetic_record(entry, plan, rt, rng, shift=0.0, noise=0.02):
    """A synthetic completed argmax record built from the stored sampled rows of `entry` (J shifted and jittered)."""
    rows = []
    for e in entry['stored']['episodes']:
        r = dict(e)
        r['J'] = float(e['J']) + shift + float(rng.normal(0, noise))
        r['tp_timely'] = max(0.0, float(e['tp_timely']) + float(rng.normal(0, 0.1)))
        rows.append(r)
    per_cell, weighted = p4.aggregate_evaluation(rt, rows)
    prov = {'idea_e1_tree': runner.EXPECTED_E1_TREE, 'idea_dq_tree': runner.EXPECTED_DQ_TREE,
            'scripts': plan['scripts'], 'utc': 'synthetic'}
    return {'version': runner.VERSION, 'kind': 'read_out', 'status': 'completed', 'mode': 'argmax',
            'job': entry['job'], 'family': entry['family'], 'block': entry['block'], 'plan_sha256': plan['sha256'],
            'start': prov, 'end': prov, 'checkpoint': {'sha256': entry['final_checkpoint']['sha256']},
            'episodes': [runner.json_row(r) for r in rows], 'per_cell_J': per_cell, 'weighted_J': weighted}


def synthetic_precheck(table, recs):
    fams = {}
    for f in runner.FAMILIES:
        e = next(x for x in table if x['family'] == f and x['block'] == runner.PROBE_BLOCK)
        first = {}
        for r in recs[e['job']]['episodes']:                 # realisation 0 of every cell (the stored order)
            first.setdefault(r['cell_id'], r)
        chosen = list(first.values())
        fams[f] = {'argmax_keys': [[r['cell_id'], r['seed']] for r in chosen],
                   'argmax_digest': runner.rows_digest(chosen)}
    return {'passed': True, 'tag': '', 'utc': 'synthetic', 'd_argmax_probe': {'families': fams}}


def scratch_dir():
    import tempfile
    base = os.environ.get('E1ARGMAX_SCRATCH')
    return Path(base) if base and Path(base).is_dir() else Path(tempfile.gettempdir())


def _context():
    ctx = runner.e1_context()
    return ctx, runner.policy_table(ctx)


def selftest():
    # statistics against scipy
    rng = np.random.default_rng(1)
    x = rng.normal(-0.15, 0.02, 12)
    s = t_summary(x)
    t = stats.ttest_1samp(x, 0.0)
    lo, hi = stats.t.interval(0.95, 11, loc=np.mean(x), scale=stats.sem(x))
    assert abs(s['p_two_sided'] - t.pvalue) < 1e-12 and abs(s['lo95'] - lo) < 1e-12 and abs(s['hi95'] - hi) < 1e-12
    assert t_summary([1.0, 1.0, 1.0])['p_two_sided'] == 0.0 and t_summary([0.0, 0.0])['p_two_sided'] == 1.0
    adj = holm_adjusted([0.01, 0.04, 0.03, 0.005])
    assert all(abs(a - b) < 1e-12 for a, b in zip(adj, [0.03, 0.06, 0.06, 0.02])), adj
    assert holm_adjusted([0.5, 0.9]) == [1.0, 1.0]
    assert reading_ra(0.001, -0.1) == 'holds' and reading_ra(0.001, 0.1) == 'reversed'
    assert reading_ra(0.2, -0.1) == 'not resolved' and reading_ra(0.05, -0.1) == 'holds'
    assert reading_rb(3.2, 3.05, 3.4, 3) == 'holds' and reading_rb(3.2, 2.8, 3.4, 3) == 'not resolved'
    assert reading_rb(1.5, 1.2, 1.9, 2) == 'fails' and reading_rb(1.5, 1.2, 2.1, 2) == 'not resolved'
    const = ratio_bootstrap({'a': [0.5] * 12, 'b': [0.25] * 12}, 2.0, ['a', 'b'], resamples=200)
    assert abs(const['ratio_lo'] - 4.0) < 1e-12 and abs(const['ratio_hi'] - 4.0) < 1e-12
    zero = ratio_bootstrap({'a': [0.0] * 12}, 2.0, ['a'], resamples=50)
    assert zero['ratio_lo'] == math.inf and zero['ratio_hi'] == math.inf
    # E1 numbers of record re-derived from the frozen evidence (read-only)
    ctx, table = _context()
    rt = ctx['runtime']
    rule = ctx['baselines']['p3_rule/e1_evaluation']['episodes']
    _, rule_j = p4.aggregate_evaluation(rt, rule)
    assert repr(rule_j) == repr(runner.RULE_J_OF_RECORD), rule_j
    rule_credited = statistics.mean(credited(e) for e in rule)
    assert abs(rule_credited - 2.388502108982746) < 1e-12 and round(rule_credited, 2) == 2.39, rule_credited
    report = json.loads(E1_REPORT.read_text())
    sampled = {}
    for f in runner.FAMILIES:
        eps = [e['stored']['episodes'] for e in table if e['family'] == f]
        js = [p4.aggregate_evaluation(rt, x)[1] for x in eps]
        assert js == [e['stored']['weighted_J'] for e in table if e['family'] == f]
        sampled[f] = statistics.mean(js)
        assert sampled[f] == TABLE5_SAMPLED_J[f], (f, sampled[f])
        p = panel([r for x in eps for r in x])
        fp = report['families'][f]['final_panel']
        assert abs(p['credited_timely'] - fp['credited_timely']) < 1e-12 and abs(p['fp_spatial'] - fp['fp_spatial']) < 1e-12
    assert abs(report['reference_policies']['p3_rule']['e1_evaluation']['weighted_J'] - rule_j) == 0
    # the episode-level point estimate equals the block-level mean difference
    recs = {e['job']: synthetic_record(e, {'sha256': 's', 'scripts': {}}, rt, np.random.default_rng(e['block']))
            for e in table if e['family'] == 'mlp'}
    arg_eps = [recs[e['job']]['episodes'] for e in table if e['family'] == 'mlp']
    weights = {c['cell_id']: float(rt.probabilities[i]) / 10 for i, c in enumerate(rt.cells)}
    ep = episode_level(arg_eps, rule, weights)
    block = statistics.mean(p4.aggregate_evaluation(rt, x)[1] - rule_j for x in arg_eps)
    assert abs(ep['mean'] - block) < 1e-12, (ep['mean'], block)
    print(json.dumps({'selftest': 'passed', 'rule_weighted_J_e1_evaluation': rule_j,
                      'rule_credited_timely_per_episode': rule_credited, 'table5_sampled_J': sampled,
                      'holm_example': adj}, indent=1, default=float))


def dryrun():
    """The whole read-out on SYNTHETIC records (stored sampled rows, J shifted and jittered) in a scratch directory;
    exercises every path of main() and two refusals, without any run and without touching records/."""
    import tempfile
    ctx, table = _context()
    rt = ctx['runtime']
    tmp = Path(tempfile.mkdtemp(prefix='e1argmax_dryrun_', dir=scratch_dir()))
    (tmp / 'records').mkdir()
    plan = {'sha256': 'synthetic-plan', 'scripts': runner.code_hashes(), 'policy_table_digest': runner.table_digest(table)}
    rng = np.random.default_rng(7)
    shifts = {'quantum': -0.01, 'bottleneck': 0.0, 'mlp': 0.03, 'surrogate625': 0.01}
    recs = {}
    for e in table:
        recs[e['job']] = synthetic_record(e, plan, rt, rng, shift=shifts[e['family']] + rng.normal(0, 0.005))
        (tmp / 'records' / f'{e["job"]}.json').write_text(json.dumps(recs[e['job']], default=float))
    pre = synthetic_precheck(table, recs)
    paths = {'records': tmp / 'records', 'out': tmp / 'argmax_results.json', 'synthetic': True}
    # refusal 1: a missing record
    gone = tmp / 'records' / f'{table[0]["job"]}.json'
    held = gone.read_text()
    gone.unlink()
    try:
        main(plan=plan, precheck=pre, ctx=ctx, table=table, paths=paths, quiet=True)
        raise AssertionError('a missing record was not refused')
    except Refused as exc:
        assert 'incomplete' in str(exc)
    gone.write_text(held)
    # refusal 2: a record of another plan
    bad = dict(recs[table[1]['job']], plan_sha256='another-plan')
    (tmp / 'records' / f'{table[1]["job"]}.json').write_text(json.dumps(bad, default=float))
    try:
        main(plan=plan, precheck=pre, ctx=ctx, table=table, paths=paths, quiet=True)
        raise AssertionError('a foreign record was not refused')
    except Refused as exc:
        assert 'plan_sha256' in str(exc)
    (tmp / 'records' / f'{table[1]["job"]}.json').write_text(json.dumps(recs[table[1]['job']], default=float))
    res = main(plan=plan, precheck=pre, ctx=ctx, table=table, paths=paths)
    assert set(res['families']) == set(runner.FAMILIES) and res['synthetic']
    assert all(v['argmax_weighted_J']['n'] == 12 for v in res['families'].values())
    assert all(v['argmax_minus_rule']['reading_R_A'] in ('holds', 'reversed', 'not resolved')
               for v in res['families'].values())
    try:
        main(plan=plan, precheck=pre, ctx=ctx, table=table, paths=paths, quiet=True)
        raise AssertionError('a second read-out was not refused')
    except Refused as exc:
        assert 'run once' in str(exc)
    print('dryrun passed; synthetic results in', paths['out'])


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--dryrun' in sys.argv:
        dryrun()
    else:
        main()
