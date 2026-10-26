"""Analysis of B1R, exactly as fixed in PLAN.md (written before any fresh-unit record existed).

    python b1r_analysis.py [--provisional]     # writes b1r_results.json; refuses incomplete data unless --provisional
                                               # (then writes b1r_results_provisional.json, labelled as such)

Primary family R (two-layer, kappa = 0.01, Holm over two at familywise 0.05), paired over fresh units:
    R1  G_H(HQ-Fisher) - G_H(TAN)          one-sided, H1: mean < 0
    R2  G_H(CTRIG) - G_H(TAN)              one-sided, H1: mean < 0   (CTRIG = within-unit mean over its ok draws)
Secondary family S (Holm over three at familywise 0.05; never read as primary):
    S1  G_H(CTRIG) - G_H(HQ-Fisher)        two-sided
    S2  G_H(OPT-CIRC) - G_H(HQ-Fisher)     one-sided, H1: mean > 0
    R3  G_H(OPT-CIRC) - G_H(OPT-TAN)       one-sided margin test, H1: mean > -DELTA_R3 (secondary; see PLAN.md for its
                                           scale and wording: never 'non-inferior' or 'replicated' in a headline)
Descriptive: R3's two-sided difference and TOST, OPT-CIRC - OPT-TAN on the fit-set objective L_F, and the rest.
R3 is a held-out-gain test, never a test of reachability (which concerns L_F); every R3 and joint R1/R3 reading carries
the L_F difference in the same sentence, and the joint reading says 'mainly' only when B1R's own R1 mean is at or below
-2 delta (PLAN.md, reading rules).
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
CHATGPT = Path('/home/konstantinos/Documents/ChatGPT')
B1_UNITS = CHATGPT / 'idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units'
B1C_RECORDS = CHATGPT / 'idea_dq_posthoc_b1c/records'
B1E_RECORDS = CHATGPT / 'idea_dq_posthoc_b1e/records'
ANALYSIS_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FRESH = tuple(range(100, 136))                    # the 36 fresh units (= runner.FRESH; a test checks it)
KAPPAS = ('0.001', '0.01')
PRIMARY = '0.01'
SCALE = 1e5                                       # every reported difference in units of 1e-5
DELTA_R3 = 2.85e-5                                # R3 margin: half of B1's HQ-Fisher - TAN mean on the ORIGINAL units
                                                  # (5.70e-5), PLAN.md; fixed, never rescaled to B1R's own shortfall
#: joint R1/R3 reading (PLAN.md, round 3): 'mainly' is read only when B1R's own R1 mean is at or below -2 delta, i.e.
#: when delta is at most half of the fresh shortfall's point estimate; otherwise the split is not settled
JOINT_R1_LIMIT = -2 * DELTA_R3                    # -5.70e-5
ALPHA = 0.05
REPRO_TOL = 1e-6                                  # extension E's recomputation of HQ-Fisher / TAN vs extension C's
CTRIG_DRAWS = 20
MIN_COMPLETE = 30                                 # a primary contrast with fewer units: the replication is incomplete
#: R3's scale, as disclosed in PLAN.md (original units, kappa = 0.01, mean gains x 1e-5: TAN 4.38, OPT-TAN 3.99,
#: OPT-CIRC 3.39, HQ-Fisher -1.31); every R3 reading carries it
R3_SCALE = ('delta = 2.85e-5 is fixed at half of B1\'s natural-step shortfall on the original units (HQ-Fisher - TAN, '
            '5.70e-5; not rescaled to the fresh units\' own shortfall) and is about 71% of OPT-TAN\'s mean held-out gain '
            'on the original units (3.99e-5), so passing the margin means that OPT-CIRC keeps at least about 29% of '
            'OPT-TAN\'s held-out gain; the margin was chosen knowing that the original units pass (p = 0.0038). '
            'Secondary test on held-out gain G_H, which also mixes in overfitting; it is not a test of reachability, '
            'which concerns the fit-set objective L_F (on the original units OPT-CIRC - OPT-TAN on L_F was -7.98e-5 '
            '[-9.34, -6.62], n = 36, about 2.8 delta). The optimiser is local (SLSQP from five starts for OPT-CIRC, four '
            'for OPT-TAN); never worded as non-inferior or replicated in a headline.')
#: R3's comparison, in held-out terms (round 3): never called 'the reachability gap'
HELD_OUT_GAP = ('the held-out gap between the best optimised circuit step and the best linear tangent step found '
                '(OPT-CIRC - OPT-TAN on G_H; local optimiser)')
DELTA_ANCHOR = ('delta = 2.85e-5, fixed at half of B1\'s natural-step shortfall on the original units (5.70e-5), not '
                'rescaled to the fresh units\' own')


# -- statistics -------------------------------------------------------------------------------------------------------

def t_test(d, alternative='two-sided', margin=0.0, scale=SCALE):
    """Paired one-sample t on unit differences d (NaN dropped). H0: mean = margin (two-sided) or the one-sided
    complement of H1: mean < margin ('less') / mean > margin ('greater'). Mean, CI and dz are reported in 1/scale."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = int(d.size)
    if n < 3:
        return {'n': n, 'mean': None, 'p': None}
    m, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0.:
        return {'n': n, 'mean': m * scale, 'sd': 0., 'p': None, 'note': 'zero variance: no test'}
    se = sd / np.sqrt(n)
    t = (m - margin) / se
    if alternative == 'less':
        p = float(stats.t.cdf(t, n - 1))
    elif alternative == 'greater':
        p = float(stats.t.sf(t, n - 1))
    else:
        p = float(2 * stats.t.sf(abs(t), n - 1))
    q = stats.t.ppf(0.975, n - 1)
    q1 = stats.t.ppf(0.95, n - 1)
    return {'n': n, 'mean': m * scale, 'sd': sd * scale, 'lo': (m - q * se) * scale, 'hi': (m + q * se) * scale,
            'one_sided_95_upper': (m + q1 * se) * scale, 'one_sided_95_lower': (m - q1 * se) * scale,
            'alternative': alternative, 'margin': margin * scale, 't': float(t), 'p': p, 'dz': m / sd,
            'share_negative': float(np.mean(d < 0))}


def tost(d, delta, scale=SCALE):
    """Two one-sided tests of |mean| < delta; p = max of the two one-sided p values; 90% CI."""
    lo = t_test(d, 'greater', -delta, scale)
    hi = t_test(d, 'less', delta, scale)
    if lo['p'] is None:
        return {'n': lo['n'], 'p': None}
    return {'n': lo['n'], 'delta': delta * scale, 'p': max(lo['p'], hi['p']), 'p_lower': lo['p'], 'p_upper': hi['p'],
            'ci90': [lo['one_sided_95_lower'], hi['one_sided_95_upper']]}


def holm(ps, alpha=ALPHA):
    """Holm step-down; a missing p (None) is never rejected and counts in the family size."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i] is None, ps[i] if ps[i] is not None else 0.))
    out = [False] * m
    for rank, i in enumerate(order):
        if ps[i] is not None and ps[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def opposite(c):
    """B2 rule (1): the two-sided 95% CI lies entirely above 0, the direction opposite to B1's and B1C's."""
    return c.get('lo') is not None and c['lo'] > 0


def two_sided_sign(c, alpha=ALPHA):
    """-1 / +1 when the two-sided difference test is significant at alpha in that direction, else 0."""
    if c.get('p') is None or c['p'] > alpha:
        return 0
    return -1 if c['mean'] < 0 else 1


def at_most_half(c1):
    """Joint R1/R3 rule (PLAN.md): True when B1R's own R1 mean (x 1e-5) is at or below -2 delta (-5.70e-5), so that
    delta is at most half of the fresh shortfall's point estimate. A fixed, deterministic rule; no test is read."""
    return c1.get('mean') is not None and c1['mean'] <= JOINT_R1_LIMIT * SCALE


def lf_clause(p):
    """The descriptive L_F difference, reported in the same sentence as every R3 and joint R1/R3 reading."""
    c = p.get('OPT-CIRC_minus_OPT-TAN_L_F') or {}
    head = 'on the fit-set objective L_F, which reachability concerns, OPT-CIRC - OPT-TAN is '
    if c.get('lo') is None:
        return head + f'not available (descriptive, n = {c.get("n", 0)})'
    return head + f'{c["mean"]:+.2f} [{c["lo"]:+.2f}, {c["hi"]:+.2f}] x 1e-5 (descriptive, n = {c["n"]})'


# -- unit values ------------------------------------------------------------------------------------------------------

def g(entry):
    return entry['G_H'] if isinstance(entry, dict) and entry.get('status', 'ok') == 'ok' and 'G_H' in entry else np.nan


def draws_mean(draws, arm, k):
    vals = [g(d[arm][k]) for d in draws if arm in d and k in d[arm]]
    ok = [v for v in vals if not np.isnan(v)]
    return (float(np.mean(ok)) if ok else np.nan), len(ok)


def unit_values(r):
    """Per-unit G_H of every arm (and L_F of the optimisers and TAN); record layout: arms (C), opt.arms (E)."""
    v = {}
    a, o = r['arms'], r['opt']['arms']
    for k in KAPPAS:
        for arm in ('HQ-Fisher', 'TAN', 'QTAY'):
            v[(arm, k)] = g(a[arm][k])
        for arm in ('CTRIG', 'CTRIGx0.1', 'CTRIGx10', 'CQR'):
            v[(arm, k)], v[(arm + '_ok_draws', k)] = draws_mean(a['CTRIG_draws'], arm, k)
        v[('CTRIG#0', k)] = g(a['CTRIG_draws'][0]['CTRIG'][k])
        v[('CTRIG_draw_sd', k)] = float(np.nanstd([g(d['CTRIG'][k]) for d in a['CTRIG_draws']], ddof=1))
        e = o[k]
        for arm in ('OPT-CIRC', 'OPT-TAN'):
            feasible = e.get(arm, {}).get('feasible')
            v[(arm, k)] = e[arm].get('G_H', np.nan) if feasible else np.nan
            v[(arm + '_L_F', k)] = e[arm].get('L_F', np.nan) if feasible else np.nan
        for arm in ('HQ-plain', 'TAN-c'):
            v[(arm, k)] = g(e[arm])
        v[('E_HQ-Fisher', k)], v[('E_TAN', k)] = g(e['HQ-Fisher']), g(e['TAN'])
        v[('TAN_L_F', k)] = (e['TAN'].get('L_F', np.nan) if isinstance(e.get('TAN'), dict)
                             and e['TAN'].get('status') == 'ok' else np.nan)
        v[('repro', k)] = e.get('reproduction_max_rel_diff', np.nan)
    return v


def consistency(r):
    """Pre-declared internal check (PLAN.md, exclusions): at both kappa, extension E's recomputed HQ-Fisher and TAN
    have the same calibration status as extension C's, and their gains agree within relative 1e-6.
    Returns the list of reasons for exclusion (empty: consistent)."""
    a, o = r['arms'], r['opt']['arms']
    out = []
    for k in KAPPAS:
        for arm in ('HQ-Fisher', 'TAN'):
            sc, se = a[arm][k].get('status'), o[k].get(arm, {}).get('status')
            if sc != se:
                out.append(f'{arm} at kappa {k}: status {se!r} in E, {sc!r} in C')
        rd = o[k].get('reproduction_max_rel_diff', np.inf)
        if not rd <= REPRO_TOL:
            out.append(f'reproduction at kappa {k}: relative difference {rd!r} > {REPRO_TOL}')
    return out


def consistent(r):
    return not consistency(r)


def col(U, key):
    return np.array([u.get(key, np.nan) for u in U], float)


def contrasts(U, k):
    c = lambda a: col(U, (a, k))                                                     # noqa: E731
    return {
        'R1_HQ-Fisher_minus_TAN': t_test(c('HQ-Fisher') - c('TAN'), 'less'),
        'R2_CTRIG_minus_TAN': t_test(c('CTRIG') - c('TAN'), 'less'),
        'S1_CTRIG_minus_HQ-Fisher': t_test(c('CTRIG') - c('HQ-Fisher')),
        'S2_OPT-CIRC_minus_HQ-Fisher': t_test(c('OPT-CIRC') - c('HQ-Fisher'), 'greater'),
        'R3_OPT-CIRC_minus_OPT-TAN_margin': t_test(c('OPT-CIRC') - c('OPT-TAN'), 'greater', -DELTA_R3),
        'R3_two_sided_difference': t_test(c('OPT-CIRC') - c('OPT-TAN')),
        'R3_tost': tost(c('OPT-CIRC') - c('OPT-TAN'), DELTA_R3),
        # descriptive (no test is read)
        'OPT-CIRC_minus_OPT-TAN_L_F': t_test(c('OPT-CIRC_L_F') - c('OPT-TAN_L_F')),
        'OPT-TAN_minus_TAN_L_F': t_test(c('OPT-TAN_L_F') - c('TAN_L_F')),
        'QTAY_minus_HQ-Fisher': t_test(c('QTAY') - c('HQ-Fisher')),
        'CQR_minus_CTRIG': t_test(c('CQR') - c('CTRIG')),
        'CTRIGx0.1_minus_TAN': t_test(c('CTRIGx0.1') - c('TAN')),
        'CTRIGx10_minus_TAN': t_test(c('CTRIGx10') - c('TAN')),
        'CTRIG#0_minus_TAN': t_test(c('CTRIG#0') - c('TAN')),
        'OPT-TAN_minus_TAN': t_test(c('OPT-TAN') - c('TAN')),
        'OPT-CIRC_minus_TAN': t_test(c('OPT-CIRC') - c('TAN')),
        'TAN-c_minus_TAN': t_test(c('TAN-c') - c('TAN')),
        'HQ-plain_minus_TAN': t_test(c('HQ-plain') - c('TAN')),
        'arm_means': {a: t_test(c(a)) for a in ('HQ-Fisher', 'TAN', 'QTAY', 'CTRIG', 'CTRIG#0', 'CQR', 'OPT-CIRC',
                                                 'OPT-TAN', 'HQ-plain', 'TAN-c')},
        'arm_means_L_F': {a: t_test(c(a + '_L_F')) for a in ('OPT-CIRC', 'OPT-TAN', 'TAN')},
        'ctrig_within_unit_draw_sd_median': float(np.nanmedian(col(U, ('CTRIG_draw_sd', k)))) * SCALE,
        'ctrig_units_with_fewer_than_20_ok_draws': int(np.sum(col(U, ('CTRIG_ok_draws', k)) < CTRIG_DRAWS)),
    }


def opt_diagnostics(recs, k):
    R = [r['opt']['arms'][k] for r in recs]
    fc = [x.get('first_crossing_HQ-Fisher', {}) for x in R]
    out = {'first_crossing': {'units_checked': sum(1 for f in fc if f),
                              'not_monotone_on_grid': sum(1 for f in fc if f and not f['monotone_on_grid']),
                              'earlier_crossing': sum(1 for f in fc if f.get('earlier_crossing'))},
           'angle_displacement': {}, 'optimiser': {}}
    early = [f['G_H_first'] - x['HQ-Fisher']['G_H'] for f, x in zip(fc, R) if f.get('earlier_crossing')]
    out['first_crossing']['G_H_first_minus_calibrated'] = t_test(early) if len(early) > 2 else None
    for arm in ('HQ-Fisher', 'HQ-plain', 'OPT-CIRC'):
        mx = [x.get(arm, {}).get('max_abs_angle_step') for x in R]
        mx = np.array([m for m in mx if m is not None], float)
        out['angle_displacement'][arm] = {'median_max_abs_angle_step': float(np.median(mx)) if mx.size else None,
                                          'share_beyond_pi': float(np.mean(mx > np.pi)) if mx.size else None}
    for arm in ('OPT-CIRC', 'OPT-TAN'):
        starts = [x[arm].get('start') for x in R if x[arm].get('start')]
        its = [x[arm]['iterations'] for x in R if x[arm].get('iterations') is not None]
        out['optimiser'][arm] = {'feasible_units': sum(1 for x in R if x[arm].get('feasible')),
                                 'best_start_counts': {s: starts.count(s) for s in sorted(set(starts))},
                                 'median_iterations_best': float(np.median(its)) if its else None}
    return out


# -- the original B1 units (descriptive side-by-side; stored records, nothing recomputed) -----------------------------

_STORED = {}


def stored_original(i):
    if i not in _STORED:
        _STORED[i] = tuple(json.loads((d / f'shipped_{i:02d}.json').read_text()) for d in (B1_UNITS, B1C_RECORDS, B1E_RECORDS))
    return _STORED[i]


def original_units(k):
    U = []
    for i in range(36):
        b, c, e = stored_original(i)
        a = b['instruments']['arms']
        u = {('HQ-Fisher', k): g(a['HQ-Fisher'][k]), ('TAN', k): g(a['TAN'][k]),
             ('CTRIG', k): draws_mean(c['arms']['CTRIG_draws'], 'CTRIG', k)[0]}
        for arm in ('OPT-CIRC', 'OPT-TAN'):
            x = e['arms'][k][arm]
            u[(arm, k)] = x.get('G_H', np.nan) if x.get('feasible') else np.nan
            u[(arm + '_L_F', k)] = x.get('L_F', np.nan) if x.get('feasible') else np.nan
        U.append(u)
    return U


def side_by_side(U_new, k):
    U_old = original_units(k)
    out = {}
    for name, (a, b) in {'HQ-Fisher_minus_TAN': ('HQ-Fisher', 'TAN'), 'CTRIG_minus_TAN': ('CTRIG', 'TAN'),
                         'OPT-CIRC_minus_OPT-TAN': ('OPT-CIRC', 'OPT-TAN'),
                         'OPT-CIRC_minus_OPT-TAN_L_F': ('OPT-CIRC_L_F', 'OPT-TAN_L_F')}.items():
        dn = col(U_new, (a, k)) - col(U_new, (b, k))
        do = col(U_old, (a, k)) - col(U_old, (b, k))
        dn, do = dn[~np.isnan(dn)], do[~np.isnan(do)]
        w = stats.ttest_ind(dn, do, equal_var=False) if dn.size > 1 and do.size > 1 else None
        out[name] = {'original_b1_units': t_test(do), 'fresh_units': t_test(dn),
                     'fresh_minus_original_mean': (dn.mean() - do.mean()) * SCALE if dn.size and do.size else None,
                     'welch_p_descriptive': float(w.pvalue) if w is not None else None,
                     'pooled_72_units_descriptive': t_test(np.concatenate([do, dn]))}
    tan_old = col(U_old, ('OPT-TAN', k))
    tan_new = col(U_new, ('OPT-TAN', k))
    out['delta_r3_share_of_mean_OPT-TAN_gain'] = {
        'original_units': float(DELTA_R3 / np.nanmean(tan_old)) if np.isfinite(np.nanmean(tan_old)) else None,
        'fresh_units_descriptive': (float(DELTA_R3 / np.nanmean(tan_new))
                                    if tan_new.size and np.isfinite(np.nanmean(tan_new)) else None)}
    return out


# -- readings (fixed in PLAN.md) --------------------------------------------------------------------------------------

def readings(p, rejected_R, rejected_S, incomplete=None):
    """The pre-written reading of every outcome (PLAN.md, reading rules). rejected_R = (R1, R2), Holm in family R;
    rejected_S = (S1, S2, R3), Holm in family S."""
    r1, r2 = rejected_R
    s1, s2, r3 = rejected_S
    c1, c2 = p['R1_HQ-Fisher_minus_TAN'], p['R2_CTRIG_minus_TAN']
    out = {}
    if incomplete:
        out['incomplete'] = incomplete
    # R1
    if r1:
        out['R1'] = ('replicated: on fresh units, in a prespecified test, the circuit-realised natural step (HQ-Fisher) '
                     'gains less than the same initial direction applied linearly (TAN)')
    elif opposite(c1):
        out['R1'] = ('contradicts B1: on fresh units the circuit-realised natural step gains more than the same '
                     'direction applied linearly (two-sided 95% CI entirely above 0). The manuscript states that the '
                     'prespecified replication contradicts B1 and removes the path claim; it is not qualified')
    else:
        out['R1'] = ('not replicated at this sample size: the manuscript states that the B1 path shortfall was not '
                     'replicated in a prespecified replication, reports B1 and B1R side by side and qualifies every '
                     'path claim in the abstract, the summary and the conclusions; no pooling or post-hoc analysis '
                     'replaces R1')
    r1_word = 'was contradicted' if opposite(c1) else 'did not replicate'
    # R2
    if r2 and r1:
        out['R2'] = ('replicated: the matched classical trigonometric chart (CTRIG) also falls short of the linear '
                     'tangent step on fresh units. This is an existence claim (a matched classical chart also shows '
                     'the shortfall, so the path effect is not specific to the circuit for this matching); it does not '
                     'say that the chart and the circuit fall short by the same amount: S1 not rejected is not '
                     'equivalence')
    elif r2:
        out['R2'] = (f'shown for the matched chart only (conditional on R1): the matched classical chart falls short '
                     f'of the linear tangent step on fresh units, but the circuit\'s own shortfall {r1_word} (R1). '
                     'The generic reading is not confirmed: there is no replicated circuit shortfall for the chart '
                     'result to generalise, so R2 is reported as a finding about the matched chart only. S1 not '
                     'rejected is not equivalence')
    elif opposite(c2):
        out['R2'] = ('contradicts B1C: on fresh units the matched classical chart gains more than the linear tangent '
                     'step (two-sided 95% CI entirely above 0). The manuscript states that the replication contradicts '
                     'B1C and removes the matched-chart path claim (that a matched classical chart also falls short, '
                     'the generic reading of the path effect); it is not qualified')
    else:
        out['R2'] = ('not replicated: the matched-chart shortfall of extension C stays a single post-hoc finding; the '
                     'manuscript does not call the path effect generic, and says that a matched classical chart fell '
                     'short in a post-hoc study and that this was not replicated')
    # R3 (secondary, family S). Round 3: worded in held-out gain only, delta anchored to B1's original shortfall, and
    # the L_F difference in the same sentence; G_H never stands for reachability, which concerns L_F (PLAN.md)
    sign = two_sided_sign(p['R3_two_sided_difference'])
    lf = lf_clause(p)
    held_out_only = 'this is a held-out reading: reachability concerns L_F and is not tested'
    if r3 and sign < 0:
        out['R3'] = ('within the margin, yet worse (secondary): on held-out gain, OPT-CIRC\'s mean is shown to be less '
                     'than delta below OPT-TAN\'s, and the two-sided difference is negative at 0.05, so the best '
                     'optimised circuit step found gains less on H than the best linear tangent step found (local '
                     f'optimiser), by less than delta ({DELTA_ANCHOR}); {lf}; {held_out_only}')
    elif r3:
        out['R3'] = ('margin passed (secondary): on held-out gain, OPT-CIRC\'s mean is shown to be less than delta below '
                     f'OPT-TAN\'s, so {HELD_OUT_GAP} is smaller than delta ({DELTA_ANCHOR}); ' +
                     ('the two-sided difference is positive at 0.05 (OPT-CIRC gains more on H)' if sign > 0 else
                      'the two-sided difference is not resolved') + f'; {lf}; {held_out_only}')
    else:
        out['R3'] = ('margin not passed (secondary): a held-out gap of delta or more between the best optimised circuit '
                     'step and the best linear tangent step found (OPT-CIRC - OPT-TAN on G_H; local optimiser) is not '
                     f'excluded ({DELTA_ANCHOR}); {lf}; ' +
                     ('the two-sided difference is negative at 0.05, reported as secondary evidence that the best '
                      'optimised circuit step found gains less on H than the best linear tangent step found; it is not '
                      'read as a reachability limit, because reachability concerns L_F' if sign < 0 else
                      'the two-sided difference is positive at 0.05 (reported as computed)' if sign > 0 else
                      'the two-sided difference is not resolved, so how the shortfall splits between the natural '
                      'step\'s path and what circuit policies can reach stays unresolved'))
    out['R3_scale'] = R3_SCALE
    # R1 with the secondary R3. Round 3: 'mainly' only when delta is at most half of B1R's own shortfall (R1 mean at or
    # below -2 delta); otherwise the split is not settled. Never 'rather than' (reachability concerns L_F)
    if r1 and r3:
        m1 = c1.get('mean')
        r1_mean = f'R1 mean {m1:+.2f} x 1e-5' if m1 is not None else 'R1 mean not available'
        joint = ('the B1 shortfall replicates (R1); secondary: on held-out gain, the best optimised circuit step found '
                 'comes within delta of the best linear tangent step found (local optimiser)' +
                 (' but gains significantly less, so a held-out gap smaller than delta is shown' if sign < 0 else '') +
                 f' ({DELTA_ANCHOR}; {lf}); ')
        if at_most_half(c1):
            joint += (f'delta is at most half of the fresh shortfall\'s point estimate ({r1_mean}, at or below -2 delta '
                      '= -5.70 x 1e-5), so on held-out gain this gap accounts for less than half of the fresh '
                      'shortfall, which lies mainly' + (', not wholly,' if sign < 0 else '') + ' outside the gap '
                      f'between the best optimised steps found; {held_out_only}')
        else:
            joint += (f'delta is more than half of the fresh shortfall\'s point estimate ({r1_mean}, above -2 delta = '
                      '-5.70 x 1e-5), so how the fresh shortfall splits between the natural step\'s path and what '
                      'circuit policies can reach is not settled')
        out['joint_R1_R3'] = joint
    elif r1:
        out['joint_R1_R3'] = 'the B1 shortfall replicates (R1); path against reachability is not settled'
    elif r3:
        out['joint_R1_R3'] = ('the natural-step shortfall did not replicate; the secondary held-out margin test passed '
                              f'({lf}), with no replicated shortfall for it to explain')
    else:
        out['joint_R1_R3'] = 'neither the natural-step shortfall nor the secondary margin result is shown'
    out['generic_vs_specific'] = ('shown for the circuit and for a matched classical chart (an existence claim; S1 not '
                                  'rejected is not equivalence)' if r1 and r2 else
                                  'shown for the circuit only; the generic reading is not confirmed' if r1 else
                                  'conditional: the matched chart falls short while the circuit\'s shortfall '
                                  f'{r1_word}; the generic reading is not confirmed' if r2 else
                                  'neither is shown')
    # S1, S2 (secondary)
    s1c = p['S1_CTRIG_minus_HQ-Fisher']
    out['S1'] = ((('CTRIG gains more than HQ-Fisher' if s1c['mean'] > 0 else 'CTRIG gains less than HQ-Fisher') +
                  ' (secondary, two-sided)') if s1 else
                 'not resolved (secondary): this is not evidence that CTRIG and HQ-Fisher gain the same')
    out['S2'] = ('OPT-CIRC gains more than HQ-Fisher (secondary; B1E\'s E2 direction)' if s2 else
                 'not resolved (secondary)')
    out['rule'] = ('B1R is reported beside B1, never merged with it; its p values are never combined with B1\'s; '
                   'secondary results (family S, including R3) are never read as primary')
    return out


# -- main -------------------------------------------------------------------------------------------------------------

def recorded(sha, root):
    """True when the hash is written in PLAN.md or (a disclosed deviation) in PLAN_LOG.md."""
    return any((root / n).exists() and sha in (root / n).read_text() for n in ('PLAN.md', 'PLAN_LOG.md'))


def main(provisional):
    plan = HERE / 'PLAN.md'
    plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest() if plan.exists() else None
    analysis_recorded = recorded(ANALYSIS_SHA256, HERE)
    if not analysis_recorded and not provisional:
        raise SystemExit(f'b1r_analysis.py {ANALYSIS_SHA256} is recorded neither in PLAN.md nor in PLAN_LOG.md')
    files = sorted(glob.glob(str(HERE / 'records' / 'shipped_*.json')))
    recs = [json.loads(Path(f).read_text()) for f in files]
    recs = [r for r in recs if r['index'] in FRESH]
    missing = sorted(set(FRESH) - {r['index'] for r in recs})
    if missing and not provisional:
        raise SystemExit(f'{len(recs)} of {len(FRESH)} fresh unit records; use --provisional for a partial look')
    wrong_plan = sorted(r['index'] for r in recs if r.get('plan_sha256') != plan_sha)
    if wrong_plan and not provisional:
        raise SystemExit(f'records not run under this PLAN.md: {wrong_plan}')
    runner_hashes = sorted({str(r.get('runner_sha256')) for r in recs})
    if not provisional and (len(runner_hashes) > 1 or not all(recorded(h, HERE) for h in runner_hashes)):
        raise SystemExit(f'records come from runner.py {runner_hashes}: one runner, recorded in PLAN.md or PLAN_LOG.md, '
                         'must have run every unit (failure rule 2: every unit is rerun after a fix)')
    failed = [(r['index'], r.get('failure_class'), r.get('error')) for r in recs if r['status'] != 'complete']
    good = [r for r in recs if r['status'] == 'complete']
    inconsistent = {r['index']: consistency(r) for r in good if consistency(r)}
    good = [r for r in good if r['index'] not in inconsistent]
    U = [unit_values(r) for r in good]
    out = {'plan_sha256': plan_sha, 'analysis_sha256': ANALYSIS_SHA256, 'analysis_recorded': analysis_recorded,
           'runner_sha256_of_records': runner_hashes, 'provisional': provisional, 'records': len(recs),
           'missing': missing, 'failed': failed, 'excluded_internal_consistency': sorted(inconsistent),
           'excluded_internal_consistency_reasons': inconsistent, 'units_analysed': len(U),
           'delta_r3': DELTA_R3, 'scale': 'differences in units of 1e-5', 'wrong_plan': wrong_plan}
    for k in KAPPAS:
        out[k] = contrasts(U, k)
        out[k]['opt_diagnostics'] = opt_diagnostics(good, k) if good else None
        out[k]['side_by_side_with_original_b1_units'] = side_by_side(U, k)
    p = out[PRIMARY]
    ps_R = [p['R1_HQ-Fisher_minus_TAN']['p'], p['R2_CTRIG_minus_TAN']['p']]
    ps_S = [p['S1_CTRIG_minus_HQ-Fisher']['p'], p['S2_OPT-CIRC_minus_HQ-Fisher']['p'],
            p['R3_OPT-CIRC_minus_OPT-TAN_margin']['p']]
    rej_R, rej_S = holm(ps_R), holm(ps_S)
    n_R = {'R1': p['R1_HQ-Fisher_minus_TAN']['n'], 'R2': p['R2_CTRIG_minus_TAN']['n']}
    short = {c: n for c, n in n_R.items() if n < MIN_COMPLETE}
    out['incomplete'] = bool(short)
    out['incomplete_statement'] = (f'The replication is incomplete: {", ".join(f"{c} has {n} units" for c, n in short.items())}'
                                   f' (fewer than {MIN_COMPLETE}). The tests are unchanged.' if short else None)
    out['R3_units'] = p['R3_OPT-CIRC_minus_OPT-TAN_margin']['n']
    out['family_R'] = {'p': ps_R, 'n': n_R, 'holm_rejected': dict(zip(('R1', 'R2'), rej_R)),
                       'opposite_direction': {'R1': opposite(p['R1_HQ-Fisher_minus_TAN']),
                                              'R2': opposite(p['R2_CTRIG_minus_TAN'])}}
    out['family_S'] = {'p': ps_S, 'holm_rejected': dict(zip(('S1', 'S2', 'R3'), rej_S)),
                       'R3_two_sided_sign_at_0.05': two_sided_sign(p['R3_two_sided_difference'])}
    out['joint_R1_R3_rule'] = {'R1_mean_x1e-5': p['R1_HQ-Fisher_minus_TAN'].get('mean'),
                               'limit_minus_2_delta_x1e-5': JOINT_R1_LIMIT * SCALE,
                               'R1_mean_at_or_below_limit': at_most_half(p['R1_HQ-Fisher_minus_TAN'])}
    out['readings'] = readings(p, rej_R, rej_S, out['incomplete_statement'])
    name = 'b1r_results_provisional.json' if provisional else 'b1r_results.json'
    (HERE / name).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('records', 'missing', 'failed', 'excluded_internal_consistency',
                                          'units_analysed', 'incomplete', 'family_R', 'family_S', 'readings')},
                     indent=1, default=float))
    for k in KAPPAS:
        for key in ('R1_HQ-Fisher_minus_TAN', 'R2_CTRIG_minus_TAN', 'S1_CTRIG_minus_HQ-Fisher',
                    'S2_OPT-CIRC_minus_HQ-Fisher', 'R3_OPT-CIRC_minus_OPT-TAN_margin', 'R3_two_sided_difference',
                    'OPT-CIRC_minus_OPT-TAN_L_F'):
            v = out[k][key]
            if v.get('p') is not None:
                print(f"kappa {k} {key}: {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] p={v['p']:.2g} "
                      f"({v['alternative']}) dz={v['dz']:+.2f} n={v['n']}")
    return out


if __name__ == '__main__':
    main('--provisional' in sys.argv)
