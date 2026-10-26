"""POST-HOC RE-ANALYSIS OF STORED RECORDS (no new runs): uncertainty for the headline systems numbers.

    ~/Documents/ChatGPT/idea/.venv/bin/python -B reanalysis_panel.py [out.json]
    (default output: ../work/reanalysis_panel.json)

Answers the referees' point that the systems numbers carry no uncertainty. Read-only on every record; nothing is
re-run, no frozen test or reading changes. Every input is first checked by recomputing the point values the paper
prints (assert, at the printed precision); the checks are listed in the output under 'assertion_checks'.

1. Table 5 (E1 evaluation episodes: 80 = 8 cells x 10 realisations, shared by all 12 blocks, the rule and random).
   Learned rows: t interval over the 12 blocks, and a pair-preserving percentile bootstrap that resamples blocks and
   evaluation realisations together, with one realisation index list per drift-budget pair, shared by that pair's
   target-present and empty cells (they fly the same common-random-number seeds) and by every policy (the convention of
   Table 11 / posthoc_v2). Rule and random rows: an episode bootstrap stratified by cell (same index lists; a variant
   with independent lists per cell is also given). Learned - rule differences and the rule/learned ratio of
   timely-credited reports use the same replicates, so the rule's episode variability is included.
2. Stress study (Freeze A, Section 4.8): at D600 = 8 m, learned - random J per family x budget in the target-present
   cell (Figure 9's quantity), paired on the 20 common-random-number realisations: block-level t interval and the
   pair-preserving bootstrap (secondary: the budget's two cells weighted 0.8/0.2 as in the frozen manifest). Also the
   0 -> 2 m change behind 'changed by at most 0.006'.
3. E4 (S7): each re-tuned head's mean J at 300,000 decisions as a share of the rule's 0.211 on the same 80 episodes,
   with the t interval (rule fixed) and, secondarily, the pair-preserving bootstrap over blocks x realisations.
4. B1 / B1R (Section 4.4, S9): OPT-CIRC / OPT-TAN fit-set objective (ratio of means) and the matched-chart shortfall
   share (CTRIG - TAN) / (HQ-Fisher - TAN), each with a unit-bootstrap percentile interval, on B1's and on the fresh
   units (kappa = 0.01, two-layer).
Percentile intervals, 10,000 draws, fixed seeds (one generator per section).
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np                                                                                # noqa: E402
from scipy import stats                                                                           # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'work' / 'reanalysis_panel.json'
CHATGPT = Path('/home/konstantinos/Documents/ChatGPT')
E1 = CHATGPT / 'idea_e1/output/study/gnss_qrl_v1/e1/e1_campaign_v2'
E1_BASELINES_SHA = '00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7'
FA = CHATGPT / 'idea_dq/output/study/gnss_qrl_v1/dq'
FA_JOBS = FA / 'freeze_a_campaign_v1/jobs'
FA_REPORT = FA / 'freeze_a_analysis_v1/freeze_a_report.json'
FA_MANIFEST = FA / 'freeze_a_manifest_v1/freeze_a_manifest.json'
E4 = CHATGPT / 'idea_dq_posthoc_long2'
B1R = CHATGPT / 'idea_dq_posthoc_b1r'
SBS_JSON = HERE.parent / 'work' / 'b1r_side_by_side.json'

N_BOOT = 10_000
SEED = 20261026
FAMILY = {'mlp': 'C-strong', 'surrogate625': 'Surrogate625', 'bottleneck': 'C-matched', 'quantum': 'HQ'}
E1_BLOCKS = tuple(range(10, 22))
RE = 18.0
BUDGET_S = {'nominal_2700s': '2,700 s', 'tight_derived': '1,881 s'}

CHECKS = []


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(what, value, printed, fmt):
    """Assert that value, formatted as the paper prints it, equals the printed string (Unicode minus accepted)."""
    s = format(value, fmt)
    ok = s == printed.replace('−', '-')
    CHECKS.append({'what': what, 'value': float(value), 'printed': printed, 'formatted': s, 'pass': ok})
    assert ok, f'{what}: recomputed {s} ({value!r}) but the paper prints {printed}'


def t_ci(x):
    x = np.asarray(x, float)
    n = x.size
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {'n': n, 'mean': m, 'lo': m - h, 'hi': m + h}


def pct(rep):
    lo, hi = np.percentile(rep, [2.5, 97.5])
    return {'lo': float(lo), 'hi': float(hi)}


def counts(idx, n):
    """(N, k) resampled indices -> (N, n) multiplicities."""
    return (idx[..., None] == np.arange(n)).sum(1).astype(float)


def rep_learned(X, cb, cr, pair_of):
    """X (blocks, cells, realisations) -> per-replicate cell means (N, cells) over resampled blocks x realisations."""
    nb, nc, nr = X.shape
    return np.stack([np.einsum('nb,br,nr->n', cb, X[:, c, :], cr[pair_of[c]]) / (nb * nr) for c in range(nc)], 1)


def rep_fixed(x, cr, pair_of):
    """x (cells, realisations) of a single policy -> per-replicate cell means (N, cells)."""
    nc, nr = x.shape
    return np.stack([cr[pair_of[c]] @ x[c] / nr for c in range(nc)], 1)


def episode_array(episodes, cells, seeds, field, realisation_key='seed'):
    by = {(e['cell_id'], e[realisation_key]): e for e in episodes}
    assert len(by) == len(episodes), 'duplicated (cell, realisation)'
    return np.array([[float(by[c, s][field]) for s in seeds[c]] for c in cells])


# -- 1. Table 5 -----------------------------------------------------------------------------------------------------

STATS5 = {'weighted_J': 'J', 'timely_credited_per_pooled_episode': 'tp_timely',
          'false_reports_per_pooled_episode': 'fp_spatial', 'people_located_per_target_present_episode': 'tp_spatial'}


def table5():
    assert sha256(E1 / 'baselines/baselines.json') == E1_BASELINES_SHA
    base = json.loads((E1 / 'baselines/baselines.json').read_text())['results']
    rule_eps, rnd_eps = base['p3_rule/e1_evaluation']['episodes'], base['random/e1_evaluation']['episodes']
    cells = sorted({e['cell_id'] for e in rule_eps})
    assert len(cells) == 8 and len(rule_eps) == len(rnd_eps) == 80
    pair_of = [c.rsplit('__', 1)[0] for c in cells]
    pairs = sorted(set(pair_of))
    seeds = {c: sorted(e['seed'] for e in rule_eps if e['cell_id'] == c) for c in cells}
    assert all(len(v) == 10 for v in seeds.values())
    for c, p in zip(cells, pair_of):           # common random numbers: the two cells of a pair fly the same seeds
        assert seeds[c] == seeds[f'{p}__positive'] == seeds[f'{p}__no_human']
    wJ = np.array([0.2 if c.endswith('positive') else 0.05 for c in cells])
    W = {'weighted_J': wJ, 'timely_credited_per_pooled_episode': np.full(8, 1 / 8),
         'false_reports_per_pooled_episode': np.full(8, 1 / 8),
         'people_located_per_target_present_episode': np.array([0.25 if c.endswith('positive') else 0. for c in cells])}

    fixed = {'Tuned rule': rule_eps, 'Uniform random': rnd_eps}
    for name, eps in fixed.items():
        assert sorted((e['cell_id'], e['seed']) for e in eps) == sorted((c, s) for c in cells for s in seeds[c]), name
    learned = {}
    for arm in FAMILY:
        blocks = {}
        for f in glob.glob(str(E1 / f'evaluation/{arm}__*__b*/run/{arm}/metrics.jsonl')):
            b = int(f.split('__b')[1].split('/')[0])
            ev = [json.loads(x) for x in Path(f).read_text().splitlines()]
            last = [e for e in ev if e.get('kind') == 'evaluation'][-1]
            assert last['points'] == [30000], (f, last['points'])
            eps = last['results']['sample']['episodes']
            assert sorted((e['cell_id'], e['seed']) for e in eps) == sorted((c, s) for c in cells for s in seeds[c])
            blocks[b] = eps
            jw = sum(wJ[i] * np.mean([e['J'] for e in eps if e['cell_id'] == c]) for i, c in enumerate(cells))
            assert abs(jw - last['results']['sample']['weighted_J']) < 1e-12
        assert tuple(sorted(blocks)) == E1_BLOCKS, arm
        learned[FAMILY[arm]] = [blocks[b] for b in E1_BLOCKS]

    X = {n: {k: np.stack([episode_array(eps, cells, seeds, f) for eps in bl]) for k, f in STATS5.items()}
         for n, bl in learned.items()}                                                  # (12, 8, 10)
    x = {n: {k: episode_array(eps, cells, seeds, f) for k, f in STATS5.items()} for n, eps in fixed.items()}
    point = {n: {k: float(W[k] @ v[k].mean(-1).mean(0)) for k in STATS5} for n, v in X.items()}
    per_block = {n: {k: v[k].mean(-1) @ W[k] for k in STATS5} for n, v in X.items()}   # (12,)
    point.update({n: {k: float(W[k] @ v[k].mean(-1)) for k in STATS5} for n, v in x.items()})
    assert abs(point['Tuned rule']['weighted_J'] - base['p3_rule/e1_evaluation']['weighted_J']) < 1e-12
    assert abs(point['Uniform random']['weighted_J'] - base['random/e1_evaluation']['weighted_J']) < 1e-12

    # printed Table 5 and its note
    printed = {'Tuned rule': ('0.206', '2.39', '0.075', '0', '7.8'), 'C-strong': ('0.052', '0.79', '0.25', '2.8', '2.0'),
               'Surrogate625': ('0.045', '0.73', '0.29', '2.9', '2.1'), 'C-matched': ('0.037', '0.65', '0.33', '3.0', '1.9'),
               'HQ': ('0.024', '0.49', '0.33', '3.3', '1.6'), 'Uniform random': ('−0.002', '0.17', '0.29', '3.7', '1.0')}
    for n, (pj, pt, pf, pg, pl) in printed.items():
        p = point[n]
        check(f'Table 5 {n} weighted J', p['weighted_J'], pj, '.3f')
        check(f'Table 5 {n} timely-credited reports', p['timely_credited_per_pooled_episode'], pt, '.2f')
        check(f'Table 5 {n} false reports', p['false_reports_per_pooled_episode'], pf, f'.{len(pf) - 2}f')
        if n != 'Tuned rule':
            check(f'Table 5 {n} gap to rule (RE)', RE * (point['Tuned rule']['weighted_J'] - p['weighted_J']), pg, '.1f')
        check(f'Table 5 note: {n} people located per target-present episode (of 18)',
              p['people_located_per_target_present_episode'], pl, '.1f')
    check('Table 5 note: rule timely-credited per target-present episode',
          2 * point['Tuned rule']['timely_credited_per_pooled_episode'], '4.8', '.1f')

    rng = np.random.default_rng([SEED, 5])
    cb = counts(rng.integers(0, 12, (N_BOOT, 12)), 12)
    cr = {p: counts(rng.integers(0, 10, (N_BOOT, 10)), 10) for p in pairs}
    rng_ind = np.random.default_rng([SEED, 55])
    cr_ind = {c: counts(rng_ind.integers(0, 10, (N_BOOT, 10)), 10) for c in cells}
    rep = {n: {k: rep_learned(v[k], cb, cr, pair_of) @ W[k] for k in STATS5} for n, v in X.items()}
    rep.update({n: {k: rep_fixed(v[k], cr, pair_of) @ W[k] for k in STATS5} for n, v in x.items()})
    rep_ind = {n: {k: rep_fixed(v[k], cr_ind, cells) @ W[k] for k in STATS5} for n, v in x.items()}

    rows = {}
    for n in ('Tuned rule', 'C-strong', 'Surrogate625', 'C-matched', 'HQ', 'Uniform random'):
        r = {}
        for k in STATS5:
            e = {'point': point[n][k]}
            if n in X:
                tt = t_ci(per_block[n][k])
                e['t95_over_12_blocks'] = [tt['lo'], tt['hi']]
                e['pair_preserving_bootstrap95_blocks_x_realisations'] = list(pct(rep[n][k]).values())
            else:
                e['episode_bootstrap95_stratified_by_cell'] = list(pct(rep[n][k]).values())
                e['episode_bootstrap95_independent_lists_per_cell'] = list(pct(rep_ind[n][k]).values())
            r[k] = e
        rows[n] = r

    R = 'Tuned rule'
    diffs, ratios = {}, {}
    for n in list(X) + ['Uniform random']:
        d = {}
        for k in ('weighted_J', 'timely_credited_per_pooled_episode', 'false_reports_per_pooled_episode'):
            e = {'point': point[n][k] - point[R][k],
                 'pair_preserving_bootstrap95': list(pct(rep[n][k] - rep[R][k]).values())}
            if n in X:
                tt = t_ci(per_block[n][k] - point[R][k])
                e['t95_over_12_blocks_rule_fixed'] = [tt['lo'], tt['hi']]
            d[k] = e
        d['weighted_J_in_RE_per_episode'] = {'point': RE * d['weighted_J']['point'],
                                             'pair_preserving_bootstrap95': [RE * v for v in
                                                                             d['weighted_J']['pair_preserving_bootstrap95']]}
        diffs[n] = d
        if n in X:
            tk, fk = 'timely_credited_per_pooled_episode', 'false_reports_per_pooled_episode'
            q = rep[R][tk] / rep[n][tk]
            qf = rep[R][fk] / rep[n][fk]
            ratios[n] = {'rule_over_learned_timely': {'point': point[R][tk] / point[n][tk], **pct(q),
                                                      'share_of_replicates_ge_3': float(np.mean(q >= 3))},
                         'rule_over_learned_false_reports': {'point': point[R][fk] / point[n][fk], **pct(qf)}}
    best = np.max([rep[n]['timely_credited_per_pooled_episode'] for n in X], 0)
    qb = rep[R]['timely_credited_per_pooled_episode'] / best
    ratios['rule_over_best_learned_family_timely (the "every learned family" reading)'] = {
        'point': point[R]['timely_credited_per_pooled_episode'] / max(point[n]['timely_credited_per_pooled_episode'] for n in X),
        **pct(qb), 'share_of_replicates_ge_3': float(np.mean(qb >= 3))}
    return {'episodes': '80 = 8 cells x 10 realisations (40 target-present), shared by all 12 blocks, rule and random',
            'cells': cells, 'drift_budget_pairs': pairs, 'rows': rows, 'learned_minus_rule': diffs,
            'ratios': ratios, 'inputs': {'baselines.json_sha256': E1_BASELINES_SHA,
                                         'learned': 'evaluation/<arm>__<cand>__b10..21/run/<arm>/metrics.jsonl, last '
                                                    'evaluation event (30,000 decisions)'}}


# -- 2. Stress study (Freeze A) -------------------------------------------------------------------------------------

def stress():
    rep_frozen = json.loads(FA_REPORT.read_text())
    manifest = json.loads(FA_MANIFEST.read_text())
    mcells = {c['cell_id']: c for c in manifest_cells(manifest)}
    assert {c['budget_name']: c['mission_budget_s'] for c in mcells.values()} == {'nominal_2700s': 2700.0,
                                                                                 'tight_derived': 1881.0}

    def load(name):
        d = json.loads((FA_JOBS / f'{name}__stress.json').read_text())
        assert d['status'] == 'complete' and len(d['episodes']) == 320, name
        return d['episodes']
    pols = {FAMILY[a]: [load(f'{a}__b{b}') for b in E1_BLOCKS] for a in FAMILY}
    rnd, rule = load('random'), load('p3_rule')
    cells = sorted({e['cell_id'] for e in rnd})
    assert len(cells) == 16
    crn = {(e['cell_id'], e['realization']): (e['seed'], e['crn_sha256']) for e in rnd}
    for eps in [rule] + [x for v in pols.values() for x in v]:
        assert {(e['cell_id'], e['realization']): (e['seed'], e['crn_sha256']) for e in eps} == crn
    reals = {c: list(range(20)) for c in cells}
    for c in cells:                 # common random numbers across drift and presence within a budget
        bud = c.split('__')[1]
        assert [crn[c, r] for r in range(20)] == [crn[f'd0__{bud}__positive', r] for r in range(20)]
    A = {n: np.stack([episode_array(eps, cells, reals, 'J', 'realization') for eps in v]) for n, v in pols.items()}
    a_rnd = episode_array(rnd, cells, reals, 'J', 'realization')
    a_rule = episode_array(rule, cells, reals, 'J', 'realization')
    ci = {c: i for i, c in enumerate(cells)}
    # the frozen report's descriptive cell means are reproduced exactly
    arm_of = {v: k for k, v in FAMILY.items()}
    for n, X in A.items():
        for c in cells:
            assert abs(X[:, ci[c]].mean() - rep_frozen['descriptive'][arm_of[n]]['mean_J'][c]) < 1e-12
    for key, arr in (('random', a_rnd), ('p3_rule', a_rule)):
        for c in cells:
            assert abs(arr[ci[c]].mean() - rep_frozen['descriptive'][key]['mean_J'][c]) < 1e-12

    # printed point values of Section 4.8
    d0 = [A[n][:, ci[f'd0__{b}__positive']].mean() for n in A for b in BUDGET_S]
    check('[p158] learned at 0 m, target-present, lowest', min(d0), '0.037', '.3f')
    check('[p158] learned at 0 m, target-present, highest', max(d0), '0.082', '.3f')
    r0 = [a_rule[ci[f'd0__{b}__positive']].mean() for b in BUDGET_S]
    check('[p158] rule at 0 m, target-present, lowest', min(r0), '0.26', '.2f')
    check('[p158] rule at 0 m, target-present, highest', max(r0), '0.32', '.2f')
    check('[p158] random at 8 m, 2,700 s, target-present', a_rnd[ci['d8__nominal_2700s__positive']].mean(),
          '−0.022', '.3f')
    check('[p158] random at 8 m, 1,881 s, target-present', a_rnd[ci['d8__tight_derived__positive']].mean(),
          '−0.026', '.3f')

    rng = np.random.default_rng([SEED, 8])
    cb = counts(rng.integers(0, 12, (N_BOOT, 12)), 12)
    cr = {b: counts(rng.integers(0, 20, (N_BOOT, 20)), 20) for b in BUDGET_S}
    out8, below = {}, 0
    for n, X in A.items():
        for b in BUDGET_S:
            cp, ca = ci[f'd8__{b}__positive'], ci[f'd8__{b}__no_human']
            L, Rn = X[:, cp], a_rnd[cp]                                                 # (12, 20), (20,)
            lt = t_ci(L.mean(1))
            below += lt['hi'] < 0
            dt = t_ci(L.mean(1) - Rn.mean())
            boot = np.einsum('nb,br,nr->n', cb, L - Rn[None], cr[b]) / (12 * 20)
            lw = 0.8 * X[:, cp] + 0.2 * X[:, ca]
            rw = 0.8 * a_rnd[cp] + 0.2 * a_rnd[ca]
            wt = t_ci(lw.mean(1) - rw.mean())
            wboot = np.einsum('nb,br,nr->n', cb, lw - rw[None], cr[b]) / (12 * 20)
            out8[f'{n} | {BUDGET_S[b]}'] = {
                'learned_mean_target_present': lt['mean'], 'learned_t95_over_blocks': [lt['lo'], lt['hi']],
                'random_mean_target_present': float(Rn.mean()),
                'learned_minus_random': {'point': dt['mean'], 't95_over_12_blocks_random_fixed': [dt['lo'], dt['hi']],
                                         'pair_preserving_bootstrap95_blocks_x_realisations': list(pct(boot).values()),
                                         'share_of_blocks_below_random': float(np.mean(L.mean(1) < Rn.mean()))},
                'secondary_budget_cells_weighted_0.8_0.2_learned_minus_random': {
                    'point': wt['mean'], 't95_over_12_blocks_random_fixed': [wt['lo'], wt['hi']],
                    'pair_preserving_bootstrap95': list(pct(wboot).values())}}
    assert all(v['learned_mean_target_present'] < 0 for v in out8.values()), '[p158] negative mean J at 8 m'
    check('[p158] 8 m family-budget cells whose block-level 95% interval lies below zero', below, '6', 'd')
    for k in ('C-strong | 1,881 s', 'Surrogate625 | 1,881 s'):
        lo, hi = out8[k]['learned_t95_over_blocks']
        assert lo < 0 < hi, k

    change = {}
    for n, X in A.items():
        for b in BUDGET_S:
            dd = X[:, ci[f'd2__{b}__positive']].mean(1) - X[:, ci[f'd0__{b}__positive']].mean(1)
            w2 = 0.8 * X[:, ci[f'd2__{b}__positive']] + 0.2 * X[:, ci[f'd2__{b}__no_human']]
            w0 = 0.8 * X[:, ci[f'd0__{b}__positive']] + 0.2 * X[:, ci[f'd0__{b}__no_human']]
            tt = t_ci(dd)
            change[f'{n} | {BUDGET_S[b]}'] = {'target_present_2m_minus_0m': tt['mean'], 't95_over_blocks': [tt['lo'], tt['hi']],
                                             'budget_cells_weighted_2m_minus_0m': float((w2 - w0).mean())}
    mx = max(change.items(), key=lambda kv: abs(kv[1]['target_present_2m_minus_0m']))
    mxw = max(abs(v['budget_cells_weighted_2m_minus_0m']) for v in change.values())
    return {'records': 'Freeze A stress jobs: 48 E1 finals x 16 cells x 20 realisations, random and rule (same CRN)',
            'report_of_record_sha256': sha256(FA_REPORT), 'manifest_sha256': sha256(FA_MANIFEST),
            'd8_learned_minus_random': out8, 'change_0_to_2m': change,
            'paper_at_most_0.006_check': {
                'max_abs_change_target_present': abs(mx[1]['target_present_2m_minus_0m']), 'at': mx[0],
                'max_abs_change_budget_weighted_cells': mxw,
                'within_0.006_unrounded': bool(abs(mx[1]['target_present_2m_minus_0m']) <= 0.006),
                'within_0.006_at_3_decimals': float(format(abs(mx[1]['target_present_2m_minus_0m']), '.3f')) <= 0.006}}


def manifest_cells(m):
    """Every cell dict in the Freeze A manifest (located by its keys, wherever it is nested)."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            if 'cell_id' in x and 'mission_budget_s' in x:
                found.append(x)
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(m)
    assert found
    return found


# -- 3. E4 (S7) -----------------------------------------------------------------------------------------------------

def e4():
    res = json.loads((E4 / 'e4_results.json').read_text())
    assert res['of_record'] and not res['provisional'] and res['readout_checkpoint'] == 300000
    rr = json.loads((E4 / 'rule_reference.json').read_text())
    assert sha256(E4 / 'rule_reference.json') == res['rule_reference']['sha256']
    reps = rr['episodes']
    cells = sorted({e['cell_id'] for e in reps})
    pair_of = [c.rsplit('__', 1)[0] for c in cells]
    pairs = sorted(set(pair_of))
    seeds = {c: sorted(e['seed'] for e in reps if e['cell_id'] == c) for c in cells}
    w = np.array([rr['cell_weights'][c] for c in cells])
    x_rule = episode_array(reps, cells, seeds, 'J')
    J_rule = float(w @ x_rule.mean(1))
    assert abs(J_rule - res['rule_reference']['weighted_J']) < 1e-12
    check('S7 rule weighted J on the 80 B2 episodes', J_rule, '0.211', '.3f')
    rng = np.random.default_rng([SEED, 4])
    cb = counts(rng.integers(0, 6, (N_BOOT, 6)), 6)
    cr = {p: counts(rng.integers(0, 10, (N_BOOT, 10)), 10) for p in pairs}
    rule_rep = rep_fixed(x_rule, cr, pair_of) @ w
    printed_blocks = {'C-strong': ['0.075', '0.156', '0.144', '0.117', '0.170', '0.135'],
                      'C-matched': ['0.050', '0.043', '0.072', '0.066', '0.069', '0.075']}
    printed_share = {'C-strong': '63', 'C-matched': '30'}
    out = {}
    for arm, head in (('mlp', 'C-strong'), ('bottleneck', 'C-matched')):
        X = []
        for b in range(4, 10):
            ev = [json.loads(z) for z in (E4 / f'runs/{arm}__b{b}/run/{arm}/metrics.jsonl').read_text().splitlines()]
            e = [z for z in ev if z.get('kind') == 'evaluation' and z['points'] == [300000]]
            assert len(e) == 1
            X.append(episode_array(e[0]['results']['sample']['episodes'], cells, seeds, 'J'))
        X = np.stack(X)                                                                   # (6, 8, 10)
        pb = X.mean(-1) @ w
        stored = res['heads'][head]['E4_J_at_T']['values']
        assert np.max(np.abs(pb - np.array(stored))) < 1e-12
        for i, (v, p) in enumerate(zip(pb, printed_blocks[head])):
            check(f'S7 {head} J at 300,000 decisions, block {4 + i}', v, p, '.3f')
        tt = t_ci(pb)
        share = tt['mean'] / J_rule
        check(f'[p96] {head} share of the rule (%)', 100 * share, printed_share[head], '.0f')
        assert abs(share - res['heads'][head]['E4_share_of_rule_J']) < 1e-12
        jrep = rep_learned(X, cb, cr, pair_of) @ w
        out[head] = {'mean_J': tt['mean'], 'mean_J_t95': [tt['lo'], tt['hi']], 'share_of_rule': share,
                     'share_t95_rule_fixed': [tt['lo'] / J_rule, tt['hi'] / J_rule],
                     'share_pair_preserving_bootstrap95_blocks_x_realisations': list(pct(jrep / rule_rep).values()),
                     'J_minus_rule_pair_preserving_bootstrap95': list(pct(jrep - rule_rep).values()),
                     'share_bootstrap_replicates_at_or_above_half': float(np.mean(jrep / rule_rep >= 0.5)),
                     'n_blocks': 6}
    return {'rule_weighted_J': J_rule, 'heads': out, 'e4_results_sha256': sha256(E4 / 'e4_results.json')}


# -- 4. B1 / B1R ratios -----------------------------------------------------------------------------------------------

def b1r():
    sys.path.insert(0, str(B1R))
    import b1r_analysis as A                                                                     # noqa: E402
    K = '0.01'
    r = json.loads((B1R / 'b1r_results.json').read_text())
    assert r['analysis_sha256'] == A.ANALYSIS_SHA256 and not r['provisional']
    recs = [json.loads(p.read_text()) for p in sorted((B1R / 'records').glob('shipped_*.json'))]
    recs = [x for x in recs if x['index'] in A.FRESH and x['status'] == 'complete' and not A.consistency(x)]
    assert len(recs) == r['units_analysed'] == 36
    sets = {"B1's units": A.original_units(K), 'fresh units (B1R)': [A.unit_values(x) for x in recs]}
    sbs = json.loads(SBS_JSON.read_text())
    printed_t = {"B1's units": {'L_F': ('−7.98', '−9.34', '−6.62')},
                 'fresh units (B1R)': {'L_F': ('−7.52', '−8.53', '−6.52'),
                                       'R1': ('−6.35', '−8.91', '−3.80'),
                                       'R2': ('−3.56', '−5.08', '−2.04')}}
    printed_ratio = {"B1's units": {'lf': '49', 'chart': '76'}, 'fresh units (B1R)': {'lf': '50', 'chart': '56'}}
    sbs_key = {"B1's units": 'b1_units', 'fresh units (B1R)': 'fresh_units'}
    rng = np.random.default_rng([SEED, 1])
    idx = rng.integers(0, 36, (N_BOOT, 36))
    out = {}
    for name, U in sets.items():
        c = {a: A.col(U, (a, K)) for a in ('HQ-Fisher', 'TAN', 'CTRIG', 'OPT-CIRC_L_F', 'OPT-TAN_L_F')}
        d1, d2 = c['HQ-Fisher'] - c['TAN'], c['CTRIG'] - c['TAN']
        dl = c['OPT-CIRC_L_F'] - c['OPT-TAN_L_F']
        for key, d in (('L_F', dl), ('R1', d1), ('R2', d2)):
            if key in printed_t[name]:
                tt = A.t_test(d)
                for v, p, lab in zip((tt['mean'], tt['lo'], tt['hi']), printed_t[name][key], ('mean', 'lo', 'hi')):
                    check(f'{name} {key} {lab} (x 1e-5)', v, p, '.2f')
        lf = float(np.nanmean(c['OPT-CIRC_L_F']) / np.nanmean(c['OPT-TAN_L_F']))
        ch = float(np.nanmean(d2) / np.nanmean(d1))
        assert abs(lf - sbs['opt_circ_fit_objective_share_of_opt_tan'][sbs_key[name]]) < 1e-12
        assert abs(ch - sbs['chart_shortfall_share_of_circuit'][sbs_key[name]]) < 1e-12
        check(f'{name} OPT-CIRC / OPT-TAN fit-set objective (%)', 100 * lf, printed_ratio[name]['lf'], '.0f')
        check(f'{name} chart shortfall as a share of the circuit\'s (%)', 100 * ch, printed_ratio[name]['chart'], '.0f')
        with np.errstate(invalid='ignore'):
            lf_b = np.nanmean(c['OPT-CIRC_L_F'][idx], 1) / np.nanmean(c['OPT-TAN_L_F'][idx], 1)
            ch_b = np.nanmean(d2[idx], 1) / np.nanmean(d1[idx], 1)
        out[name] = {'n_units': 36,
                     'units_with_nan': {k: int(np.isnan(v).sum()) for k, v in (('L_F pair', dl), ('R1', d1), ('R2', d2))},
                     'opt_circ_over_opt_tan_fit_objective': {'point': lf, **pct(lf_b)},
                     'chart_shortfall_share_of_circuit': {'point': ch, **pct(ch_b),
                                                          'share_of_replicates_below_1': float(np.mean(ch_b < 1))}}
    return {'kappa': K, 'stratum': 'two-layer', 'unit_bootstrap': 'units resampled with replacement (36), ratio of means',
            'b1r_results_sha256_prefix': sha256(B1R / 'b1r_results.json')[:12], 'sets': out}


# -- readings against the printed text --------------------------------------------------------------------------------

def readings(r):
    t5, st, e, b = r['table5_E1'], r['stress_freeze_A'], r['E4_S7']['heads'], r['B1_B1R_ratios']['sets']
    q = t5['ratios']['C-strong']['rule_over_learned_timely']
    qb = next(v for k, v in t5['ratios'].items() if k.startswith('rule_over_best'))
    fr = {n: t5['ratios'][n]['rule_over_learned_false_reports'] for n in FAMILY.values()}
    d8 = st['d8_learned_minus_random']
    excl = [k for k, v in d8.items() if (lambda i: i[0] > 0 or i[1] < 0)(
        v['learned_minus_random']['pair_preserving_bootstrap95_blocks_x_realisations'])]
    above = [k for k, v in d8.items() if v['learned_minus_random']['point'] > 0]
    ch = st['paper_at_most_0.006_check']
    cs = e['C-strong']
    return [
        {'text': '[p16], [p177], Cover letter: "earned at least three times the mean timely-report credit of every '
                 'learned family"',
         'reanalysis': f"rule / C-strong {q['point']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}]; rule / best family per "
                       f"replicate {qb['point']:.2f} [{qb['lo']:.2f}, {qb['hi']:.2f}], >= 3 in "
                       f"{100 * qb['share_of_replicates_ge_3']:.0f}% of replicates",
         'consistent': qb['lo'] >= 3},
        {'text': '[p177]: "with less than a third of their false reports"',
         'reanalysis': '; '.join(f"{n} {v['point']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" for n, v in fr.items()),
         'consistent': all(v['hi'] < 1 / 3 for v in fr.values())},
        {'text': '[p158]: "so at 8 m the learned families were close to or below random"; Abstract: "near or below '
                 'random"; Highlight 3: "near or below random at 8 m"',
         'reanalysis': f"learned - random intervals excluding zero: {excl or 'none'}; point estimates above random: "
                       f"{above}",
         'consistent': 'close to random: yes (no interval excludes zero); below random: not established'},
        {'text': '[p158]: "their scores changed by at most 0.006 between 0 and 2 m"',
         'reanalysis': f"max |change| {ch['max_abs_change_target_present']:.4f} ({ch['at']}, target-present)",
         'consistent': 'at 3 decimals only' if ch['within_0.006_at_3_decimals'] and not ch['within_0.006_unrounded']
         else ch['within_0.006_unrounded']},
        {'text': '[p96]: "63% of the rule\'s 0.211"; [p203]: "reached 63% of the rule\'s score ..., above the half-rule '
                 'threshold"; Cover letter: "reached 63% of the rule\'s score after 300,000 decisions, above half of it"',
         'reanalysis': f"share {100 * cs['share_of_rule']:.1f}% t [{100 * cs['share_t95_rule_fixed'][0]:.1f}%, "
                       f"{100 * cs['share_t95_rule_fixed'][1]:.1f}%] (rule fixed)",
         'consistent': 'point estimate only: the t interval includes half'},
        {'text': 'S9 [p52]: "its mean reaching about half of the linear step\'s (49% on B1\'s units, 50% on the fresh '
                 'units"',
         'reanalysis': '; '.join(f"{k} {100 * v['opt_circ_over_opt_tan_fit_objective']['point']:.0f}% "
                                 f"[{100 * v['opt_circ_over_opt_tan_fit_objective']['lo']:.0f}%, "
                                 f"{100 * v['opt_circ_over_opt_tan_fit_objective']['hi']:.0f}%]" for k, v in b.items()),
         'consistent': True},
        {'text': '[p124]: "the charts\' mean shortfall was about 56% as large as the circuit\'s on the fresh units (76% '
                 'on B1\'s units; ratios of means, post hoc)"',
         'reanalysis': '; '.join(f"{k} {100 * v['chart_shortfall_share_of_circuit']['point']:.0f}% "
                                 f"[{100 * v['chart_shortfall_share_of_circuit']['lo']:.0f}%, "
                                 f"{100 * v['chart_shortfall_share_of_circuit']['hi']:.0f}%]" for k, v in b.items()),
         'consistent': 'yes; on B1\'s units the interval includes 100% (as "not resolved on B1\'s units" says)'},
    ]


# -- main -------------------------------------------------------------------------------------------------------------

def main(out=OUT):
    result = {'label': 'POST-HOC RE-ANALYSIS OF STORED RECORDS (no new runs); descriptive intervals; no frozen test '
                       'or reading changes',
              'bootstrap': {'draws': N_BOOT, 'seed': SEED, 'interval': 'percentile 95%',
                            'note': 'with few top-level units the percentile bootstrap is narrower than the t interval '
                                    '(no small-sample t quantile, resampling variance (n-1)/n): about 15% for 12 '
                                    'blocks before episode variability is added, about 30% for E4\'s 6 blocks, so for '
                                    'E4 the t interval is the one to quote'},
              'table5_E1': table5(), 'stress_freeze_A': stress(), 'E4_S7': e4(), 'B1_B1R_ratios': b1r()}
    result['paper_text_vs_reanalysis'] = readings(result)
    result['assertion_checks'] = {'n': len(CHECKS), 'all_passed': all(c['pass'] for c in CHECKS), 'checks': CHECKS}
    result['script_sha256'] = sha256(__file__)
    Path(out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float) + '\n')
    summary(result)
    print(f'\n{len(CHECKS)} printed values reproduced; written {out}')


def f3(v):
    return f'{v:+.3f}'.replace('-', '−')


def summary(r):
    t5 = r['table5_E1']
    print('Table 5 (E1, 80 episodes): point [t over 12 blocks] {pair-preserving bootstrap} | rule/random: {episode bootstrap}')
    for n, row in t5['rows'].items():
        parts = []
        for k, lab in (('weighted_J', 'J'), ('timely_credited_per_pooled_episode', 'timely'),
                       ('false_reports_per_pooled_episode', 'false'), ('people_located_per_target_present_episode', 'located')):
            e = row[k]
            b = e.get('pair_preserving_bootstrap95_blocks_x_realisations') or e['episode_bootstrap95_stratified_by_cell']
            t = e.get('t95_over_12_blocks')
            parts.append(f"{lab} {e['point']:.3f}" + (f" [{t[0]:.3f}, {t[1]:.3f}]" if t else '') + f" {{{b[0]:.3f}, {b[1]:.3f}}}")
        print(f'  {n:15s}', ' | '.join(parts))
    for n, d in t5['learned_minus_rule'].items():
        j, tm = d['weighted_J'], d['timely_credited_per_pooled_episode']
        print(f"  {n:15s} J - rule {f3(j['point'])} {{{f3(j['pair_preserving_bootstrap95'][0])}, "
              f"{f3(j['pair_preserving_bootstrap95'][1])}}}  timely - rule {tm['point']:+.2f} "
              f"{{{tm['pair_preserving_bootstrap95'][0]:+.2f}, {tm['pair_preserving_bootstrap95'][1]:+.2f}}}")
    for n, q in t5['ratios'].items():
        q = q.get('rule_over_learned_timely', q)
        print(f"  rule/learned timely, {n}: {q['point']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}]")
    print('Stress at 8 m, target-present: learned - random, point [t] {bootstrap}')
    for n, v in r['stress_freeze_A']['d8_learned_minus_random'].items():
        d = v['learned_minus_random']
        print(f"  {n:24s} {f3(d['point'])} [{f3(d['t95_over_12_blocks_random_fixed'][0])}, "
              f"{f3(d['t95_over_12_blocks_random_fixed'][1])}] {{{f3(d['pair_preserving_bootstrap95_blocks_x_realisations'][0])}, "
              f"{f3(d['pair_preserving_bootstrap95_blocks_x_realisations'][1])}}}")
    print('  0 -> 2 m:', r['stress_freeze_A']['paper_at_most_0.006_check'])
    for h, v in r['E4_S7']['heads'].items():
        print(f"E4 {h}: share {v['share_of_rule']:.3f} t [{v['share_t95_rule_fixed'][0]:.3f}, {v['share_t95_rule_fixed'][1]:.3f}]"
              f" bootstrap [{v['share_pair_preserving_bootstrap95_blocks_x_realisations'][0]:.3f}, "
              f"{v['share_pair_preserving_bootstrap95_blocks_x_realisations'][1]:.3f}]")
    for s, v in r['B1_B1R_ratios']['sets'].items():
        a, b = v['opt_circ_over_opt_tan_fit_objective'], v['chart_shortfall_share_of_circuit']
        print(f"{s}: OPT-CIRC/OPT-TAN L_F {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]; "
              f"chart/circuit shortfall {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]")


if __name__ == '__main__':
    main(*sys.argv[1:])
