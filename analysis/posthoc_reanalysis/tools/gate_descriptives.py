"""Descriptive summaries across blocks of the E1-gate read-out of record (post hoc, descriptive only).

Reads gate_results.json (written once by idea_dq_posthoc_gate/gate_analysis.py) and averages its per-block
descriptive panel over the 12 blocks. No test and no reading is added here; P1 and its reading come only from
gate_results.json. Usage: python gate_descriptives.py <gate_results.json> <out.json>
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats

#: the stored rule rows the plan binds (idea_dq_posthoc_gate/runner.py BASELINES, BASELINES_SHA256)
BASELINES = ('/home/konstantinos/Documents/ChatGPT/idea_e1/output/study/gnss_qrl_v1/e1/e1_campaign_v2/baselines/'
             'baselines.json')
BASELINES_SHA256 = '00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7'


def per_episode(gate_dir, P):
    """Post-hoc, descriptive: which target-present episodes carry the block-level difference (raw records)."""
    raw = open(BASELINES, 'rb').read()
    assert hashlib.sha256(raw).hexdigest() == BASELINES_SHA256
    rule = {(e['cell_id'], e['seed']): e for e in json.loads(raw)['results']['p3_rule/e1_evaluation']['episodes']}
    pos = sorted(k for k in rule if k[0].endswith('positive'))
    D, FP = np.zeros((12, len(pos))), np.zeros((12, len(pos)))
    for b in range(12):
        rec = json.load(open(f'{gate_dir}/records/gate__b{b}.json'))
        ep = {(e['cell_id'], e['seed']): e for e in rec['final_evaluation']['episodes']}
        for i, k in enumerate(pos):
            D[b, i] = ep[k]['J'] - rule[k]['J']
            FP[b, i] = ep[k]['fp_spatial'] - rule[k]['fp_spatial']
    w = 0.2 / 10
    assert abs((D * w).sum(1).mean() - P['mean']) < 1e-12
    de = D.mean(0)
    order = list(np.argsort(-de))

    def row(i):
        return {'cell_id': pos[i][0], 'seed': pos[i][1], 'realization': pos[i][1] % 1000, 'mean_d': float(de[i]),
                'weighted_contribution': float(w * de[i]), 'blocks_above_rule': int((D[:, i] > 1e-12).sum()),
                'blocks_below_rule': int((D[:, i] < -1e-12).sum()), 'rule_fp': rule[pos[i]]['fp_spatial'],
                'mean_fp_change': float(FP[:, i].mean()),
                'blocks_fp_lower': int((FP[:, i] < 0).sum()), 'blocks_fp_higher': int((FP[:, i] > 0).sum()),
                'fp_change_values': sorted(set(float(x) for x in FP[:, i]))}
    keep = order[2:]
    Dk = (D[:, keep] * w).sum(1)
    keep3 = order[2:-1]
    Dk3 = (D[:, keep3] * w).sum(1)
    third = order[-1]
    fpj = lambda cols: float(-FP[:, cols].sum(1).mean() * w / 18)
    others = [i for i in range(len(pos)) if i not in (order[0], order[1], third)]
    fp_split = {'two_favourable': fpj(order[:2]), 'third': fpj([third]), 'other_episodes': fpj(others),
                'other_episodes_with_change': [f'{pos[i][0]} r{pos[i][1] % 1000}' for i in others
                                               if abs(FP[:, i].mean()) > 1e-12]}
    fp_split['total'] = fp_split['two_favourable'] + fp_split['third'] + fp_split['other_episodes']
    return {'note': 'post hoc, descriptive; not in the plan', 'false_report_J_split': fp_split,
            'top2': [row(i) for i in order[:2]], 'largest_negative': row(order[-1]),
            'top2_weighted_contribution': float(w * de[order[:2]].sum()),
            'without_top2': {'mean': float(Dk.mean()), 'p_two_sided': float(stats.ttest_1samp(Dk, 0).pvalue)},
            'without_top2_and_largest_negative': {'mean': float(Dk3.mean()),
                                                  'p_two_sided': float(stats.ttest_1samp(Dk3, 0).pvalue)},
            'episodes_mean_d_positive': int((de > 1e-12).sum()), 'episodes_mean_d_negative': int((de < -1e-12).sum())}


def main(src, out):
    raw = open(src, 'rb').read()
    r = json.loads(raw)
    assert r['provisional'] is False and r['counts']['completed'] == 12
    pb = r['descriptive']['per_block']
    P = r['P1_gate_minus_rule_e1_evaluation']

    def mm(xs):
        xs = [x for x in xs if x is not None]
        return {'mean': float(np.mean(xs)), 'min': float(np.min(xs)), 'max': float(np.max(xs)), 'n': len(xs)}

    res = {
        'source': src, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'plan_sha256': r['plan_sha256'],
        'note': 'descriptive, post hoc summaries over blocks of gate_results.json; no test, no reading',
        'blocks_D_positive': int(sum(b['D'] > 0 for b in pb)), 'blocks': len(pb),
        'D_min': float(min(b['D'] for b in pb)), 'D_max': float(max(b['D'] for b in pb)),
        'rule_J': pb[0]['J_rule'], 'gate_J': mm([b['J_gate'] for b in pb]),
        'fixes_per_positive_episode': {'gate': mm([b['fixes_per_positive_episode']['gate'] for b in pb]),
                                       'rule': pb[0]['fixes_per_positive_episode']['rule']},
        'fp_per_positive_episode': {'gate': mm([b['fp_per_positive_episode']['gate'] for b in pb]),
                                    'rule': pb[0]['fp_per_positive_episode']['rule']},
        'tp_timely_per_positive_episode': {'gate': mm([b['tp_timely_per_positive_episode']['gate'] for b in pb]),
                                           'rule': pb[0]['tp_timely_per_positive_episode']['rule']},
        'return_rate': mm([b['return_rate'] for b in pb]),
        'infrastructure_episodes': int(sum(b['infrastructure_episodes'] for b in pb)),
        'gate_rows_per_positive_episode': mm([b['gate_rows_per_positive_episode'] for b in pb]),
        'fix_rate_by_rms_m': {k: mm([b['fix_rate_by_rms_m'][k] for b in pb]) for k in pb[0]['fix_rate_by_rms_m']},
        'per_cell_d_mean_over_blocks': r['descriptive']['per_cell_d_mean_over_blocks'],
        'per_cell_blocks_positive': {c: int(sum(b['per_cell_d'][c] > 0 for b in pb))
                                     for c in pb[0]['per_cell_d'] if c.endswith('positive')},
        'sampled_readout_across_blocks': r['descriptive']['sampled_readout']['across_blocks'],
    }
    pos = r['descriptive']['per_cell_d_mean_over_blocks']
    w = {c: (0.2 if c.endswith('positive') else 0.05) for c in pos}
    contrib = {c: w[c] * pos[c] for c in pos}
    res['per_cell_weighted_contribution'] = contrib
    assert abs(sum(contrib.values()) - P['mean']) < 1e-12, (sum(contrib.values()), P['mean'])
    lc = r['descriptive']['learning_curve_dev_weighted_d']
    grid = [pt[0] for pt in lc['0']]
    res['learning_curve_dev'] = {
        'grid': grid,
        'gate_minus_rule_mean': [float(np.mean([lc[b][i][3] for b in lc])) for i in range(len(grid))],
        'blocks_above_rule': [int(sum(lc[b][i][3] > 0 for b in lc)) for i in range(len(grid))],
        'note': 'e1_development read-outs (development episodes), argmax'}
    tr = r['descriptive']['training']
    res['p_fix_after_last_update'] = {
        'rms_lt_1.5': mm([tr[b]['p_fix_after_update_rms_lt_1.5'][-1] for b in tr]),
        'rms_ge_1.5': mm([tr[b]['p_fix_after_update_rms_ge_1.5'][-1] for b in tr])}
    res['approx_kl_last10_updates_mean'] = mm([float(np.mean(tr[b]['approx_kl_pre_step_minibatch_mean_per_update'][-10:]))
                                              for b in tr])
    res['training_gate_decisions'] = mm([tr[b]['gate_decisions'] for b in tr])
    res['training_episodes'] = mm([tr[b]['episodes'] for b in tr])
    gate_dir = os.path.dirname(os.path.abspath(src))
    res['per_episode'] = per_episode(gate_dir, P)
    pc = json.load(open(os.path.join(gate_dir, 'precheck.json')))
    assert pc['plan_sha256'] == r['plan_sha256']
    res['precheck'] = {'passed': pc['passed'], 'unit_tests': len(pc['b_unit_tests']['results']),
                       'unit_tests_passed': pc['b_unit_tests']['passed'],
                       'rule_reproduction': f"{pc['c_reproduction']['identical']}/{pc['c_reproduction']['episodes']}",
                       'rule_gate_rows_per_positive_episode': pc['c_reproduction']['gate_rows_per_positive_episode'],
                       'smoke_passed': pc['d_smoke']['passed']}
    tr = r['descriptive']['training']
    res['pre_gate_reward_all_zero'] = all(tr[b]['pre_gate_reward'] == 0 for b in tr)
    res['dropped_reward_all_zero'] = all(tr[b]['dropped_reward'] == 0 for b in tr)
    res['unweighted_80_episode_d'] = r['descriptive']['mean_unweighted_80_episode_d']
    res['episode_level'] = P['episode_level']
    # by Eq. (1), J = (timely-credited - false reports) / 18 per episode: split of the weighted gain (descriptive)
    tg, tr_ = res['tp_timely_per_positive_episode']['gate']['mean'], res['tp_timely_per_positive_episode']['rule']
    fg, fr_ = res['fp_per_positive_episode']['gate']['mean'], res['fp_per_positive_episode']['rule']
    share_pos = 0.8      # cell weight of the four target-present cells; target-absent episodes equal the rule's
    split = {'timely_J': share_pos * (tg - tr_) / 18, 'false_report_J': share_pos * (fr_ - fg) / 18}
    assert abs(split['timely_J'] + split['false_report_J'] - P['mean']) < 1e-9, split
    res['gain_split_eq1'] = split
    open(out, 'w').write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
