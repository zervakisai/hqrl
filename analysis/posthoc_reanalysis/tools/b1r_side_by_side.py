"""B1R beside B1: every reported local contrast on B1's 36 two-layer units and on the 36 fresh units (descriptive).

    ~/Documents/ChatGPT/idea/.venv/bin/python -B b1r_side_by_side.py [out.json]

Reads only stored records, through the unchanged functions of idea_dq_posthoc_b1r/b1r_analysis.py (ef5c0515…):
original_units() for B1's units (B1, B1C and B1E records) and the fresh-unit values of b1r_results.json. The analysis
of record already gives four of the six contrasts side by side; this adds CTRIG − HQ-Fisher (S1) and OPT-CIRC −
HQ-Fisher (S2) on B1's units, and checks that the four agree with b1r_results.json and the two with the published B1C
and B1E values. It also gives, descriptively, the Welch difference fresh minus B1's units for all six (the analysis of
record has four), the 72-unit pooled means of record, and the chart's shortfall as a share of the circuit's. No claim
or test rests on pooling or on a B1-versus-B1R comparison (PLAN.md: reported beside B1, never merged).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
B1R = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_b1r')
sys.path.insert(0, str(B1R))
import b1r_analysis as A  # noqa: E402

K = '0.01'
CONTRASTS = {'R1': ('HQ-Fisher', 'TAN', 'R1_HQ-Fisher_minus_TAN', 'HQ-Fisher_minus_TAN'),
             'R2': ('CTRIG', 'TAN', 'R2_CTRIG_minus_TAN', 'CTRIG_minus_TAN'),
             'S1': ('CTRIG', 'HQ-Fisher', 'S1_CTRIG_minus_HQ-Fisher', None),
             'S2': ('OPT-CIRC', 'HQ-Fisher', 'S2_OPT-CIRC_minus_HQ-Fisher', None),
             'R3': ('OPT-CIRC', 'OPT-TAN', 'R3_two_sided_difference', 'OPT-CIRC_minus_OPT-TAN'),
             'L_F': ('OPT-CIRC_L_F', 'OPT-TAN_L_F', 'OPT-CIRC_minus_OPT-TAN_L_F', 'OPT-CIRC_minus_OPT-TAN_L_F')}
#: published values on B1's units (Supplementary Material S2: B1C's Y1 and B1E's E2), x 1e-5, 2 decimals
PUBLISHED = {'S1': (1.38, -0.54, 3.31), 'S2': (4.71, 2.91, 6.50)}


def main(out=None):
    r = json.loads((B1R / 'b1r_results.json').read_text())
    assert r['analysis_sha256'] == A.ANALYSIS_SHA256 and not r['provisional']
    U = A.original_units(K)
    sbs = r[K]['side_by_side_with_original_b1_units']
    rows = {}
    for name, (a, b, fresh_key, sbs_key) in CONTRASTS.items():
        d = A.col(U, (a, K)) - A.col(U, (b, K))
        alt = {'R1': 'less', 'R2': 'less'}.get(name, 'two-sided')
        orig = A.t_test(d, alt)
        if sbs_key:
            ref = sbs[sbs_key]['original_b1_units']
            assert abs(ref['mean'] - orig['mean']) < 1e-9 and abs(ref['lo'] - orig['lo']) < 1e-9, name
        else:
            pm, plo, phi = PUBLISHED[name]
            assert (round(orig['mean'], 2), round(orig['lo'], 2), round(orig['hi'], 2)) == (pm, plo, phi), (name, orig)
        rows[name] = {'contrast': f'{a.replace("_L_F", "")} − {b.replace("_L_F", "")}'
                                  + (' (fit-set objective)' if name == 'L_F' else ''),
                      'b1_units': {k: orig[k] for k in ('n', 'mean', 'lo', 'hi', 'p', 'dz', 'share_negative')},
                      'fresh_units': {k: r[K][fresh_key][k] for k in ('n', 'mean', 'lo', 'hi', 'p', 'dz',
                                                                       'share_negative')}}
    # fresh minus B1's units for every row (Welch, descriptive, post hoc for S1 and S2; the analysis of record gives the
    # other four, which must agree), and the chart's shortfall as a share of the circuit's (ratio of means, descriptive)
    from scipy import stats
    recs = [json.loads(p.read_text()) for p in sorted((B1R / 'records').glob('shipped_*.json'))]
    recs = [x for x in recs if x['index'] in A.FRESH and x['status'] == 'complete' and not A.consistency(x)]
    assert len(recs) == r['units_analysed'] == 36
    Un = [A.unit_values(x) for x in recs]
    for name, (a, b, fresh_key, sbs_key) in CONTRASTS.items():
        dn = A.col(Un, (a, K)) - A.col(Un, (b, K))
        do = A.col(U, (a, K)) - A.col(U, (b, K))
        dn, do = dn[~np.isnan(dn)], do[~np.isnan(do)]
        assert abs(dn.mean() * A.SCALE - r[K][fresh_key]['mean']) < 1e-9, name
        w = stats.ttest_ind(dn, do, equal_var=False)
        diff = (dn.mean() - do.mean()) * A.SCALE
        se = np.sqrt(dn.var(ddof=1) / dn.size + do.var(ddof=1) / do.size) * A.SCALE
        q = stats.t.ppf(0.975, w.df)
        rows[name]['fresh_minus_b1_welch'] = {'diff': diff, 'lo': diff - q * se, 'hi': diff + q * se,
                                              'p': float(w.pvalue)}
        if sbs_key:
            assert abs(sbs[sbs_key]['welch_p_descriptive'] - float(w.pvalue)) < 1e-9, name
            rows[name]['pooled_72_units_descriptive'] = {k: sbs[sbs_key]['pooled_72_units_descriptive'][k]
                                                         for k in ('mean', 'lo', 'hi')}
    share = {s_: rows['R2'][s_]['mean'] / rows['R1'][s_]['mean'] for s_ in ('b1_units', 'fresh_units')}
    # the best circuit step's fit-set objective as a share of the best linear tangent step's (ratio of means)
    lf_share = {'b1_units': float(np.nanmean(A.col(U, ('OPT-CIRC_L_F', K))) / np.nanmean(A.col(U, ('OPT-TAN_L_F', K)))),
                'fresh_units': float(np.nanmean(A.col(Un, ('OPT-CIRC_L_F', K))) /
                                     np.nanmean(A.col(Un, ('OPT-TAN_L_F', K))))}
    # single chart draws whose held-out gain fell below the circuit's natural step (post hoc, descriptive; written after
    # the third round of independent checks): CTRIG is a 20-draw mean, so this shows where the circuit lies among them
    def below(draws, hq):
        vals = [A.g(dr['CTRIG'][K]) for dr in draws]
        vals = [v for v in vals if not np.isnan(v)]
        return sum(v < hq for v in vals), len(vals)
    fb = [below(x['arms']['CTRIG_draws'], A.g(x['arms']['HQ-Fisher'][K])) for x in recs]
    ob = []
    for i in range(36):
        b_, c_, _ = A.stored_original(i)
        ob.append(below(c_['arms']['CTRIG_draws'], A.g(b_['instruments']['arms']['HQ-Fisher'][K])))
    draws_below = {'fresh_units': [sum(a_ for a_, _ in fb), sum(n_ for _, n_ in fb)],
                   'b1_units': [sum(a_ for a_, _ in ob), sum(n_ for _, n_ in ob)]}
    res = {'label': 'descriptive side-by-side; kappa = 0.01, two-layer stratum; x 1e-5; never pooled for any claim',
           'single_chart_draws_below_circuit_post_hoc': draws_below,
           'chart_shortfall_share_of_circuit': share, 'opt_circ_fit_objective_share_of_opt_tan': lf_share,
           'b1r_results_sha256_prefix': None, 'rows': rows}
    import hashlib
    res['b1r_results_sha256_prefix'] = hashlib.sha256((B1R / 'b1r_results.json').read_bytes()).hexdigest()[:12]
    text = json.dumps(res, indent=1)
    if out:
        Path(out).write_text(text)
    print(text)


if __name__ == '__main__':
    main(*sys.argv[1:])
