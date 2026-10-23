"""Analysis of the post-hoc B1 extension, exactly as fixed in PLAN.md (written before the runs).

    python b1x_analysis.py [--provisional]     # writes b1x_results.json; refuses incomplete data unless --provisional

The manuscript wording is generated here from the numbers and the reading rules of PLAN.md, so it cannot drift.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
B1_UNITS = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b1_campaign_v1/units')
STRATA = {'shipped': 'two-layer', 'heldout': 'three-layer'}
K1, K2 = '0.01', '0.001'
DAMPING = ('B1', '0.0001', '0.01', '1')
DLABEL = {'B1': '10⁻⁶', '0.0001': '10⁻⁴', '0.01': '10⁻²', '1': '1'}


def load(circuit):
    """{index: record} of complete units that passed the reproduction check, and the excluded indices."""
    keep, excluded = {}, []
    for f in sorted((HERE / 'records').glob(f'{circuit}_*.json')):
        r = json.loads(f.read_text())
        if r['status'] == 'complete' and r['reproduction']['passed']:
            keep[r['index']] = r
        else:
            excluded.append(r['index'])
    return keep, excluded


def stored_plain(circuit, index, kappa):
    s = json.loads((B1_UNITS / f'{circuit}_{index:02d}.json').read_text())
    return s['instruments']['arms']['HQ-plain'][kappa].get('G_H')


def gain(r, arm, kappa):
    e = r['arms'][arm][kappa]
    return e.get('G_H') if e.get('status') == 'ok' else None


def paired(units, fa, fb):
    """Complete-case paired statistics of fa(unit) - fb(unit), in units of 1e-5 J per decision."""
    d = []
    for i, r in units.items():
        a, b = fa(i, r), fb(i, r)
        if a is not None and b is not None:
            d.append((a - b) * 1e5)
    d = np.array(d)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return {'n': n, 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': 2 * stats.t.sf(abs(m / se), n - 1),
            'dz': m / sd}


def arm_mean(units, f):
    x = np.array([v for v in (f(i, r) for i, r in units.items()) if v is not None]) * 1e5
    n = len(x)
    q = stats.t.ppf(0.975, n - 1)
    se = x.std(ddof=1) / np.sqrt(n)
    return {'n': n, 'mean': x.mean(), 'lo': x.mean() - q * se, 'hi': x.mean() + q * se}


def num(x, nd=2):
    return f'{x:+.{nd}f}'.replace('-', '−')


def cell(r, nd=2, n_full=36):
    s = f'{num(r["mean"], nd)} [{num(r["lo"], nd)}, {num(r["hi"], nd)}]'
    return s + (f' (n = {r["n"]})' if r['n'] != n_full else '')


def pstr(p):
    if p >= 0.001:
        return f'{p:.2g}'
    m, e = f'{p:.1e}'.split('e')
    return f'{m} × 10⁻' + str(-int(e)).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def sci(x):
    m, e = f'{x:.0e}'.split('e')
    return f'{m} × 10' + ('⁻' if int(e) < 0 else '') + str(abs(int(e))).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


WORDS = dict(enumerate(('no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')))


def all_units(circuit):
    return {r['index']: r for r in (json.loads(f.read_text()) for f in sorted((HERE / 'records').glob(f'{circuit}_*.json')))}


def reproduction_detail():
    """Where the re-computed B1 arms differ from the stored ones: worst relative difference at each kappa, over all
    72 units, and the calibrated step scale of the excluded units at the failing kappa."""
    worst, alpha = {}, 0.
    for circuit in STRATA:
        for i, r in all_units(circuit).items():
            s = json.loads((B1_UNITS / f'{circuit}_{i:02d}.json').read_text())['instruments']['arms']
            for new, old in (('HQ-Fisher@B1', 'HQ-Fisher'), ('TAN@B1', 'TAN')):
                for k, e in r['arms'][new].items():
                    a, b = e.get('G_H'), s[old][k].get('G_H')
                    if a is None or b is None:
                        continue
                    rel = abs(a - b) / max(abs(b), 1e-9)
                    worst[(new, k)] = max(worst.get((new, k), 0.), rel)
                    if rel > 1e-6:
                        alpha = max(alpha, e['alpha'])
    failing = sorted({key for key, v in worst.items() if v > 1e-6})
    low = max(v for (arm, k), v in worst.items() if float(k) <= 0.01)
    return {'failing_arm_kappa': failing, 'worst_failing': max(worst[f] for f in failing) if failing else 0.,
            'worst_kappa_le_0.01': low, 'max_alpha_failing': alpha}


def g(arm, kappa):
    return lambda i, r: gain(r, arm, kappa)


def main(provisional=False):
    res, units_all, excl = {}, {}, {}
    for circuit in STRATA:
        units_all[circuit], excl[circuit] = load(circuit)
    n_rec = sum(len(units_all[c]) + len(excl[c]) for c in STRATA)
    if not provisional and n_rec < 72:
        raise SystemExit(f'incomplete: {n_rec}/72 unit records')
    repro = max((r['reproduction']['max_rel_diff_G_H_B1_arms'] for c in STRATA for r in units_all[c].values()),
                default=float('nan'))
    for circuit, units in units_all.items():
        out = {'n_units': len(units), 'excluded': excl[circuit]}
        for kappa in (K1, K2):
            out[f'X1@{kappa}'] = paired(units, g('HQ-Fisher-m16', kappa), g('HQ-Fisher@B1', kappa))
            out[f'X2@{kappa}'] = paired(units, g('HQ-Fisher-m16', kappa), g('TAN@B1', kappa))
            out[f'm4-TAN@{kappa}'] = paired(units, g('HQ-Fisher-m4', kappa), g('TAN@B1', kappa))
            out[f'TAN16-TAN@{kappa}'] = paired(units, g('TAN-m16', kappa), g('TAN@B1', kappa))
            for arm in ('HQ-Fisher-m4', 'HQ-Fisher-m16', 'TAN-m16'):
                out[f'arm:{arm}@{kappa}'] = arm_mean(units, g(arm, kappa))
            steps = [r['arms']['HQ-Fisher-m16'][kappa].get('steps') for r in units.values()
                     if r['arms']['HQ-Fisher-m16'][kappa].get('status') == 'ok']
            cos = [r['arms']['HQ-Fisher-m16'][kappa].get('cos_to_TAN') for r in units.values()
                   if r['arms']['HQ-Fisher-m16'][kappa].get('status') == 'ok']
            m4s = [r['arms']['HQ-Fisher-m4'][kappa].get('steps') for r in units.values()
                   if r['arms']['HQ-Fisher-m4'][kappa].get('status') == 'ok']
            out[f'diag_m4@{kappa}'] = {'steps_median': float(np.median(m4s)), 'steps_max': int(max(m4s))}
            out[f'diag@{kappa}'] = {'steps_median': float(np.median(steps)), 'steps_max': int(max(steps)),
                                    'steps_min': int(min(steps)),
                                    'cos_to_TAN_median': float(np.median(cos)),
                                    'status': {s: sum(1 for r in units.values()
                                                      if r['arms']['HQ-Fisher-m16'][kappa].get('status') == s)
                                               for s in {r['arms']['HQ-Fisher-m16'][kappa].get('status')
                                                         for r in units.values()}}}
            tan = [r['arms']['TAN-m16'][kappa] for r in units.values() if r['arms']['TAN-m16'][kappa].get('status') == 'ok']
            out[f'diag_tan@{kappa}'] = {'steps_median': float(np.median([e['steps'] for e in tan])),
                                        'cos_to_TAN_median': float(np.median([e['cos_to_TAN'] for e in tan]))}
        for kappa in ('0.001', '0.01', '0.05'):
            for dl in DAMPING:
                out[f'damp:{dl}@{kappa}'] = paired(units, g(f'HQ-Fisher@{dl}', kappa), g(f'TAN@{dl}', kappa))
                for arm in ('HQ-Fisher', 'TAN'):
                    if dl != 'B1':
                        out[f'arm:{arm}@{dl}@{kappa}'] = arm_mean(units, g(f'{arm}@{dl}', kappa))
            out[f'damp:plain@{kappa}'] = paired(units, lambda i, r, k=kappa: stored_plain(circuit, i, k),
                                                g('TAN@B1', kappa))
        res[circuit] = out
    # Holm over X1 and X2 in the two-layer stratum (PLAN.md)
    two = res['shipped']
    ps = sorted([('X1', two[f'X1@{K1}']['p']), ('X2', two[f'X2@{K1}']['p'])], key=lambda x: x[1])
    holm = {ps[0][0]: bool(ps[0][1] <= 0.025), ps[1][0]: bool(ps[0][1] <= 0.025 and ps[1][1] <= 0.05)}
    x1, x2 = two[f'X1@{K1}'], two[f'X2@{K1}']
    x1pos, x1neg = holm['X1'] and x1['mean'] > 0, holm['X1'] and x1['mean'] < 0
    x2pos, x2neg = holm['X2'] and x2['mean'] > 0, holm['X2'] and x2['mean'] < 0
    if x2pos:
        reading = ('Re-linearised circuit steps therefore exceeded the linear path along the initial direction.')
        verdict = 'exceeded'
    elif x1pos and not x2neg:
        reading = ('Re-linearising the circuit step therefore recovered the linear-path gain: the B1 shortfall is '
                   'attributable to realising one natural step through the curved circuit map, although an '
                   'unresolved difference is not evidence of equivalence.')
        verdict = 'recovered'
    elif x1pos and x2neg:
        reading = ('Re-linearising the circuit step therefore recovered part of the shortfall, and part remained.')
        verdict = 'partial'
    elif x1neg:
        reading = ('Re-linearising the circuit step therefore made the circuit update worse.')
        verdict = 'worse'
    elif x2neg:
        reading = ('Re-linearised circuit steps therefore did not recover the shortfall, which persisted along the '
                   'circuit path.')
        verdict = 'not_recovered'
    else:
        reading = ('Neither contrast was resolved, so the extension does not decide whether re-linearisation '
                   'recovers the shortfall.')
        verdict = 'unresolved'
    th = res['heldout']
    t1, t2 = th[f'X1@{K1}'], th[f'X2@{K1}']
    n2, n3, e2, e3 = len(units_all['shipped']), len(units_all['heldout']), len(excl['shipped']), len(excl['heldout'])
    nn = {'shipped': n2, 'heldout': n3}
    damp = [two[f'damp:{dl}@{K1}'] for dl in DAMPING]
    all_neg = all(r['hi'] < 0 for r in damp)
    damp_list = '; '.join(f'δ = {DLABEL[dl]}: {cell(two[f"damp:{dl}@{K1}"], n_full=n2)}' for dl in DAMPING)
    if all_neg:
        damp_read = 'at this KL, the shortfall of the circuit-realised step is therefore not specific to near-zero damping.'
    else:
        unres = [DLABEL[dl] for dl, r in zip(DAMPING, damp) if not r['hi'] < 0]
        damp_read = ('the interval includes zero at δ = ' + ', '.join(unres) +
                     ', so at that damping the path difference is not resolved.')
    # outside the planned cell (two-layer, kappa = 0.01): where the damped or plain step does not trail TAN resolvably
    off = []
    for c, clab in (('shipped', 'two-layer'), ('heldout', 'three-layer')):
        for kappa in (K2, K1):
            for dl in DAMPING + ('plain',):
                if c == 'shipped' and kappa == K1 and dl != 'plain':
                    continue
                r = res[c][f'damp:{dl}@{kappa}']
                if not r['hi'] < 0:
                    lab = 'the plain step' if dl == 'plain' else f'δ = {DLABEL[dl]}'
                    off.append(f'{lab} at κ = {kappa} in the {clab} stratum ({cell(r, n_full=nn[c])})')
    off_text = (' Outside this primary cell the difference was unresolved for ' + ' and for '.join(off) + '.') if off else ''

    # reproduction and exclusions (PLAN.md: a unit failing the check is reported and excluded from every contrast)
    rd = reproduction_detail()
    repro_text = ('Their fit, calibration and held-out sets were regenerated from the frozen B1 seeds with the B1 code '
                  'tree of record, and at κ ≤ 0.01 the re-computed HQ-Fisher and TAN gains matched the stored values '
                  f'in every unit (largest relative difference {sci(rd["worst_kappa_le_0.01"])}).')
    if e2 + e3:
        arms_k = sorted({(a.split('@')[0], k) for a, k in rd['failing_arm_kappa']})
        where = ' and '.join(f'{a} at κ = {k}' for a, k in arms_k)
        repro_text += (f' {WORDS.get(e2 + e3, str(e2 + e3)).capitalize()} units ({WORDS.get(e2, e2)} two-layer, '
                       f'{WORDS.get(e3, e3)} three-layer) exceeded the planned tolerance of 10⁻⁶ only for {where}, '
                       f'where the calibrated step scale reached {sci(rd["max_alpha_failing"])}; the circuit angles are '
                       'periodic, so such steps are numerically ill-posed, and this stress level enters no test of the '
                       f'extension. As the plan requires, the {WORDS.get(e2 + e3, str(e2 + e3))} units were excluded '
                       f'from every contrast, leaving {n2} two-layer and {n3} three-layer units.')
    sens = {c: {'X1': paired(all_units(c), g('HQ-Fisher-m16', K1), g('HQ-Fisher@B1', K1)),
                'X2': paired(all_units(c), g('HQ-Fisher-m16', K1), g('TAN@B1', K1))} for c in STRATA}

    # descriptive share of the single-step gap to TAN closed by HQ-Fisher-16 (a ratio not in the plan)
    def share_of(units, kappa, boot=0):
        pairs = [(gain(r, 'HQ-Fisher-m16', kappa), gain(r, 'HQ-Fisher@B1', kappa), gain(r, 'TAN@B1', kappa))
                 for r in units.values()]
        a = np.array([p for p in pairs if None not in p])
        ratio = lambda x: 100 * (x[:, 0] - x[:, 1]).mean() / (x[:, 2] - x[:, 1]).mean()        # noqa: E731
        if not boot:
            return ratio(a)
        rng = np.random.default_rng(20261023)
        reps = [ratio(a[rng.integers(0, len(a), len(a))]) for _ in range(boot)]
        return ratio(a), *np.percentile(reps, [2.5, 97.5])
    share = {c: share_of(units_all[c], K1, boot=20000) for c in STRATA}
    share_low = {c: share_of(units_all[c], K2) for c in STRATA}
    gap = -two[f'damp:B1@{K1}']['mean']

    d16, d4 = two[f'diag@{K1}'], two[f'diag_m4@{K1}']
    cos, cos_t = d16['cos_to_TAN_median'], two[f'diag_tan@{K1}']['cos_to_TAN_median']
    signed = lambda x: f'{x:.2f}'.replace('-', '−')                                            # noqa: E731
    p_intro = (
        'Re-linearised steps and intermediate damping (post hoc). A post-hoc extension, with its analysis plan fixed '
        f'by hash before any run, tested this interpretation on the 72 B1 units. {repro_text}')
    p_steps = (
        "The circuit's damped natural step was then iterated in re-linearised sub-steps (HQ-Fisher-m, m = 4 or 16): "
        'at each sub-step the importance-weighted gradient and the Fisher were recomputed on the fit set at the '
        'current policy, the damped natural step was taken through the circuit with a KL of κ/m² on the calibration '
        'set, and sub-steps continued until the total mean forward KL from the initial policy reached κ (for m = 16 at '
        f'κ = 0.01, a median of {d16["steps_median"]:.0f} sub-steps, range {d16["steps_min"]}–{d16["steps_max"]}). '
        f'At κ = 0.01, HQ-Fisher-16 differed from the single step by {cell(x1, n_full=n2)} × 10⁻⁵ (two-sided '
        f'p = {pstr(x1["p"])}) and from TAN by {cell(x2, n_full=n2)} × 10⁻⁵ (p = {pstr(x2["p"])}); '
        + ('both contrasts were rejected' if holm['X1'] and holm['X2'] else
           'X1 was ' + ('' if holm['X1'] else 'not ') + 'rejected and X2 was ' + ('' if holm['X2'] else 'not ') +
           'rejected') +
        f' after Holm adjustment over the two in the two-layer stratum. In the three-layer stratum they were '
        f'{cell(t1, n_full=n3)} (p = {pstr(t1["p"])}) and {cell(t2, n_full=n3)} (p = {pstr(t2["p"])}), and with the '
        'excluded units reinstated (not in the plan) the two-layer contrasts were '
        f'{cell(sens["shipped"]["X1"])} and {cell(sens["shipped"]["X2"])}. {reading} Descriptively (a ratio not in the '
        f'plan), HQ-Fisher-16 closed {share["shipped"][0]:.0f}% (95% bootstrap interval {share["shipped"][1]:.0f}–'
        f'{share["shipped"][2]:.0f}%) of the single-step gap to TAN in the two-layer stratum ({x1["mean"]:.2f} of '
        f'{gap:.2f} × 10⁻⁵ in the same units) and {share["heldout"][0]:.0f}% ({share["heldout"][1]:.0f}–'
        f'{share["heldout"][2]:.0f}%) in the three-layer stratum; at κ = 0.001 the shares were '
        f'{share_low["shipped"]:.0f}% and {share_low["heldout"]:.0f}%. With m = 4 (median {d4["steps_median"]:.0f} '
        f'sub-steps) the difference to TAN at κ = 0.01 was {cell(two[f"m4-TAN@{K1}"], n_full=n2)} × 10⁻⁵, against '
        f'{num(-gap)} for the single step, so the recovered share was still rising at m = 16. The net coefficient '
        'displacement of HQ-Fisher-16 was ' + ('nearly orthogonal to' if abs(cos) < 0.2 else 'at an angle to') +
        f' the initial direction (median cosine {signed(cos)}), whereas TAN-16, the same iteration inside the initial '
        f'tangent span applied linearly, stayed aligned with it ({signed(cos_t)}), reached κ in a median of '
        f'{two[f"diag_tan@{K1}"]["steps_median"]:.0f} sub-steps and differed from TAN by '
        f'{cell(two[f"TAN16-TAN@{K1}"], n_full=n2)} × 10⁻⁵. The iterated circuit update therefore does not retrace the initial direction, '
        'whereas TAN continues along it in coefficient space, where the circuit need not be able to follow; this '
        'experiment does not separate the size of the realised steps from the curvature of the set of policies the '
        'circuit can reach as causes of the remaining gap.')
    p_damp = (
        f'Over a damping ladder, HQ-Fisher − TAN at κ = 0.01 in the {n2} retained two-layer units was {damp_list} '
        f'(× 10⁻⁵; δ = 10⁻⁶ is the B1 setting); {damp_read}{off_text} Appendix A lists every arm.')
    paragraphs = [p_intro, p_steps, p_damp]
    paragraph = ' '.join(paragraphs)
    rows = []
    for kappa, klab in ((K2, '0.001'), (K1, '0.01')):
        rows.append([klab, 'HQ-Fisher-16 − HQ-Fisher, post hoc', cell(res['shipped'][f'X1@{kappa}']),
                     cell(res['heldout'][f'X1@{kappa}'])])
        rows.append([klab, 'HQ-Fisher-16 − TAN, post hoc', cell(res['shipped'][f'X2@{kappa}']),
                     cell(res['heldout'][f'X2@{kappa}'])])
    appendix = {}
    for circuit in STRATA:
        o = res[circuit]
        rows_a = []
        for arm, lab in (('HQ-Fisher-m4', 'HQ-Fisher-4 (post hoc)'), ('HQ-Fisher-m16', 'HQ-Fisher-16 (post hoc)'),
                         ('TAN-m16', 'TAN-16 (post hoc)')):
            rows_a.append([lab, cell(o[f'arm:{arm}@{K2}']), cell(o[f'arm:{arm}@{K1}']), '—'])
        for arm in ('HQ-Fisher', 'TAN'):
            for dl in DAMPING[1:]:
                rows_a.append([f'{arm}, δ = {DLABEL[dl]} (post hoc)'] +
                              [cell(o[f'arm:{arm}@{dl}@{k}']) for k in ('0.001', '0.01', '0.05')])
        appendix[circuit] = rows_a
    pct = f'{share["shipped"][0]:.0f}%'
    summary = {
        'recovered': 'smaller re-linearised circuit steps recovered the gain of the linear path at the primary KL',
        'partial': f'smaller re-linearised circuit steps recovered {pct} of the gap to the linear path at the primary KL',
        'not_recovered': 'smaller re-linearised circuit steps did not recover the gain of the linear path',
        'worse': 'smaller re-linearised circuit steps lowered the gain of the circuit update further',
        'exceeded': 'smaller re-linearised circuit steps exceeded the linear path at the primary KL',
        'unresolved': 'whether smaller re-linearised circuit steps recover the gain of the linear path was not resolved',
    }[verdict]
    damp_clause = ('and at that KL the single-step shortfall persisted at every tested damping from 10⁻⁶ to 1'
                   if all_neg else 'and the single-step shortfall was not resolved at every tested damping')
    discussion = f'In a post-hoc extension, {summary}, {damp_clause} (Section 4.4).'
    conclusion = {
        'recovered': ('A post-hoc extension located the difference in step realisation: smaller re-linearised circuit '
                      'steps recovered the gain of the linear path at the primary KL.'),
        'partial': (f'A post-hoc extension found that smaller re-linearised circuit steps recovered {pct} of the gap to '
                    'the linear path at the primary KL' + (' and that no tested damping removed it there' if all_neg
                                                           else '') +
                    '; at the tested step sizes most of the difference remained, and its cause is not identified.'),
        'not_recovered': ('A post-hoc extension found that smaller re-linearised circuit steps did not recover the gain '
                          'of the linear path, so the difference is not an artefact of one large step; its cause is '
                          'not identified.'),
        'worse': ('A post-hoc extension found that smaller re-linearised circuit steps lowered the gain of the circuit '
                  'update further; the cause of the difference is not identified.'),
        'exceeded': ('A post-hoc extension found that smaller re-linearised circuit steps exceeded the linear path, '
                     'locating the difference in step realisation.'),
        'unresolved': ('A post-hoc extension with smaller re-linearised circuit steps did not resolve whether step '
                       'realisation explains the difference.'),
    }[verdict]
    out = {'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(), 'provisional': provisional,
           'reproduction_max_rel_diff': repro, 'holm': holm, 'verdict': verdict, 'damping_all_negative': all_neg,
           'results': res, 'paragraph': paragraph, 'paragraphs': paragraphs, 'table7_rows': rows,
           'appendix_rows': appendix, 'summary_clause': summary,
           'share_recovered_pct': {c: {'point': v[0], 'boot95': [v[1], v[2]]} for c, v in share.items()},
           'share_recovered_pct_kappa_0.001': share_low, 'sensitivity_all_units': sens,
           'abstract_clause': {'partial': f'smaller re-linearised circuit steps recovered {pct} of that gap at the '
                                          'primary KL'}.get(verdict, summary),
           'reproduction_detail': {k: (v if not isinstance(v, list) else [list(x) for x in v]) for k, v in rd.items()},
           'discussion_sentence': discussion, 'conclusion_sentence': conclusion}
    (HERE / 'b1x_results.json').write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    print('\n\n'.join(paragraphs))


if __name__ == '__main__':
    main('--provisional' in sys.argv)
