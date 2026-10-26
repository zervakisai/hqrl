"""Analysis of LONG (classical heads) and LONG-HQ (HQ), exactly as fixed in their PLAN.md files and addenda.

    python long_analysis.py [--provisional]   # writes long_results.json (both plans); refuses incomplete runs otherwise
"""
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

HERE = Path(__file__).resolve().parent
HQ_DIR = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_longhq')
B2 = Path('/home/konstantinos/Documents/ChatGPT/idea_dq/output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs')
BLOCKS = (4, 5, 6, 7, 8, 9)
GRID = list(range(0, 300001, 30000))
RULE_E1 = 0.206                       # tuned rule, weighted J on the E1 evaluation episodes (E1 report)
SENSL = Path('/home/konstantinos/Documents/ChatGPT/idea_dq_posthoc_sensl/sensl_results.json')
W = {'positive': 0.2, 'absent': 0.05}
RNG = np.random.default_rng(20261023)


def rule_core():
    """Tuned rule's weighted J on the Freeze A core cells (D600 = 2 and 4 m, both budgets and presences), with B2's cell
    weights: the mean of SENS-lambda's per-drift weighted values at lambda = 1, which reproduce the stored Freeze A values."""
    s = json.loads(SENSL.read_text())['stress']['1.0']
    return (s['2']['rule'] + s['4']['rule']) / 2


def curve(path, decisions=None):
    """{nominal point: (weighted J, timely, false, fixes)} from a run's evaluation events; completion flag.
    If a dict is passed as decisions, it receives {nominal point: policy decisions at that evaluation}."""
    out, done = {}, False
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e.get('kind') == 'evaluation':
            eps = e['results']['sample']['episodes']
            cells = {}
            for x in eps:
                cells.setdefault(x['cell_id'], []).append(x['J'])
            J = sum(W['positive' if c.endswith('positive') else 'absent'] * np.mean(v) for c, v in cells.items())
            comp = tuple(float(np.mean([x[k] for x in eps])) for k in ('tp_timely', 'fp_spatial', 'fixes_attempted'))
            for p in e['points']:
                out[p] = (J,) + comp
                if decisions is not None:
                    decisions[p] = e['policy_decisions']
        if e.get('status') == 'completed':
            done = True
    return out, done


def t_sum(x):
    x = np.asarray(x, float)
    n = x.size
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return {'n': int(n), 'mean': m, 'lo': m - q * se, 'hi': m + q * se, 'p': float(2 * stats.t.sf(abs(m / se), n - 1)),
            'share_negative': float(np.mean(x < 0))}


def boot_mean(x, n=20_000):
    x = np.asarray(x, float)
    b = x[RNG.integers(0, x.size, size=(n, x.size))].mean(1)
    return {'mean': float(x.mean()), 'lo': float(np.percentile(b, 2.5)), 'hi': float(np.percentile(b, 97.5))}


def auc(c, T):
    pts = [p for p in GRID if p <= T]
    J = np.array([c[p][0] for p in pts])
    return float(np.sum((J[1:] + J[:-1]) / 2 * np.diff(pts)) / T)


def angle_change(run_dir, dec, T):
    """Circuit-angle displacement of an HQ run from its first checkpoint to those at 30,000 and at T decisions."""
    def w(d):
        f = [x for x in (run_dir / 'checkpoints').glob(f'update*_decisions{d:08d}.pt')]
        assert len(f) == 1, (run_dir, d, f)
        return torch.load(f[0], map_location='cpu', weights_only=False)['model_state_dict']['actor.weights'].double()
    w0 = w(dec[0])
    out = {'init_mean_abs_rad': float(w0.abs().mean())}
    for tag, p in (('30k', 30000), ('T', T)):
        dw = w(dec[p]) - w0
        out[f'mean_abs_change_rad_{tag}'] = float(dw.abs().mean())
        out[f'rel_norm_change_{tag}'] = float(dw.norm() / w0.norm())
    return out


def main(provisional):
    runs, done, decs = {}, {}, {}
    for arm, root in (('mlp', HERE), ('bottleneck', HERE), ('quantum', HQ_DIR)):
        for b in BLOCKS:
            f = root / 'runs' / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl'
            if f.exists():
                decs[(arm, b)] = {}
                runs[(arm, b)], done[(arm, b)] = curve(f, decs[(arm, b)])
    complete = all(done.get((a, b)) for a in ('mlp', 'bottleneck', 'quantum') for b in BLOCKS)
    if not complete and not provisional:
        raise SystemExit('not every run has completed; the addenda remove the time cut-off, so the read-out waits')
    reached = {k: max(v) for k, v in runs.items()}
    T_cls = min(reached.get((a, b), 0) for a in ('mlp', 'bottleneck') for b in BLOCKS)
    T_hq = min(reached.get(('quantum', b), 0) for b in BLOCKS + ())
    T_pair = min(T_hq, min(reached.get(('bottleneck', b), 0) for b in BLOCKS))
    out = {'plans': {'LONG': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest(),
                     'LONG-HQ': hashlib.sha256((HQ_DIR / 'PLAN.md').read_bytes()).hexdigest()},
           'provisional': not complete, 'all_runs_completed': complete,
           'largest_common_checkpoint': {'classical': T_cls, 'HQ_vs_C-matched': T_pair},
           'completed_300k': complete and T_cls == 300000 and T_pair == 300000}
    # LONG (classical): L1, L2, slope, components, reference B2 30k
    rc = rule_core()
    out['rule_references'] = {'E1_evaluation_episodes': RULE_E1, 'FreezeA_core_cells': rc,
                              'half_rule_threshold_used': min(RULE_E1, rc) / 2}
    b2_30k = {arm: [curve(B2 / f'{arm}__b{b}' / 'run' / arm / 'metrics.jsonl')[0][30000][0] for b in BLOCKS]
              for arm in ('bottleneck', 'quantum')}
    cls = {}
    for arm, role in (('mlp', 'C-strong'), ('bottleneck', 'C-matched')):
        J30 = [runs[(arm, b)][30000][0] for b in BLOCKS]
        JT = [runs[(arm, b)][T_cls][0] for b in BLOCKS]
        pts = [p for p in GRID if p <= T_cls][-3:]
        slopes = [np.polyfit(pts, [runs[(arm, b)][p][0] for p in pts], 1)[0] * 30000 for b in BLOCKS]
        closed = [(jt - j30) / (RULE_E1 - j30) for jt, j30 in zip(JT, J30)]
        closed_core = [(jt - j30) / (rc - j30) for jt, j30 in zip(JT, J30)]
        cls[role] = {'J_at_T': t_sum(JT), 'J_at_30k': t_sum(J30), 'L1_JT_minus_J30k': t_sum(np.subtract(JT, J30)),
                     'L2_fraction_of_30k_gap_to_rule_closed': boot_mean(closed),
                     'L2_fraction_of_30k_gap_to_rule_closed_FreezeA_core': boot_mean(closed_core),
                     'slope_last3_per_30k': t_sum(slopes),
                     'below_half_rule_all_blocks': bool(max(JT) < min(RULE_E1, rc) / 2),
                     'max_block_J_at_T': float(max(JT)),
                     'components_at_T': {k: float(np.mean([runs[(arm, b)][T_cls][i] for b in BLOCKS]))
                                         for i, k in ((1, 'timely'), (2, 'false'), (3, 'fixes'))},
                     'curve_mean': {p: float(np.mean([runs[(arm, b)][p][0] for b in BLOCKS])) for p in GRID if p <= T_cls}}
        if arm in b2_30k:
            cls[role]['B2_stored_J_at_30k_same_blocks'] = t_sum(b2_30k[arm])
            cls[role]['this_run_30k_minus_B2_30k'] = t_sum(np.subtract(J30, b2_30k[arm]))
        else:
            cls[role]['B2_stored_J_at_30k_same_blocks'] = None      # C-strong was not an arm of B2
    out['LONG'] = cls
    out['LONG_reading'] = ('systems conclusion holds at ' + f'{T_cls:,}' + ' decisions for these classical heads'
                           if all(v['below_half_rule_all_blocks'] for v in cls.values())
                           else 'an arm exceeded half the rule\'s J: application conclusion is budget-specific')
    # LONG-HQ: H1, H2
    hq = [runs[('quantum', b)][T_pair][0] for b in BLOCKS]
    cm = [runs[('bottleneck', b)][T_pair][0] for b in BLOCKS]
    hq30 = [runs[('quantum', b)][30000][0] for b in BLOCKS]
    cm30 = [runs[('bottleneck', b)][30000][0] for b in BLOCKS]
    H1 = t_sum(np.subtract(hq, cm))
    H2 = t_sum(np.subtract(hq, cm) - np.subtract(hq30, cm30))
    out['LONG-HQ'] = {'T': T_pair, 'H1_HQ_minus_Cmatched_at_T': H1, 'gap_at_30k': t_sum(np.subtract(hq30, cm30)),
                      'H2_change_of_gap_30k_to_T': H2,
                      'AUC_over_T': {'HQ': t_sum([auc(runs[('quantum', b)], T_pair) for b in BLOCKS]),
                                     'C-matched': t_sum([auc(runs[('bottleneck', b)], T_pair) for b in BLOCKS]),
                                     'HQ_minus_C-matched': t_sum([auc(runs[('quantum', b)], T_pair)
                                                                  - auc(runs[('bottleneck', b)], T_pair) for b in BLOCKS])},
                      'HQ_minus_C-strong_at_T': t_sum([runs[('quantum', b)][T_pair][0] - runs[('mlp', b)][T_pair][0]
                                                       for b in BLOCKS]) if T_pair <= T_cls else None,
                      'H1_reading': ('below zero: the HQ deficit persists' if H1['hi'] < 0 else
                                     'above zero: HQ exceeds C-matched' if H1['lo'] > 0 else
                                     'not resolved at n = 6 (not evidence of equality)'),
                      'curve_mean_HQ': {p: float(np.mean([runs[('quantum', b)][p][0] for b in BLOCKS])) for p in GRID if p <= T_pair},
                      'B2_stored_J_at_30k_same_blocks': t_sum(b2_30k['quantum']),
                      'this_run_30k_minus_B2_30k': t_sum(np.subtract(hq30, b2_30k['quantum'])),
                      'B2_stored_gap_at_30k_same_blocks': t_sum(np.subtract(b2_30k['quantum'], b2_30k['bottleneck']))}
    ac = [angle_change(HQ_DIR / 'runs' / f'quantum__b{b}' / 'run' / 'quantum', decs[('quantum', b)], T_pair)
          for b in BLOCKS]
    out['LONG-HQ']['angle_change_median'] = {k: float(np.median([a[k] for a in ac])) for k in ac[0]}
    # wall time per 30k decisions, from the run summaries when complete
    walls = {}
    for (arm, b), _ in runs.items():
        rec = (HQ_DIR if arm == 'quantum' else HERE) / 'records' / f'{arm}__b{b}.json'
        if rec.exists():
            r = json.loads(rec.read_text())
            if r.get('summary'):
                walls.setdefault(arm, []).append(r['summary']['wall_s'] / r['summary']['policy_decisions'] * 30000)
    out['wall_s_per_30k'] = {a: float(np.median(v)) for a, v in walls.items()}
    (HERE / 'long_results.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('provisional', 'largest_common_checkpoint', 'completed_300k', 'LONG_reading')},
                     indent=1))
    for role, v in cls.items():
        print(role, 'J30k', round(v['J_at_30k']['mean'], 4), 'J_T', round(v['J_at_T']['mean'], 4),
              'L1', round(v['L1_JT_minus_J30k']['mean'], 4), 'closed', round(v['L2_fraction_of_30k_gap_to_rule_closed']['mean'], 3))
    print('HQ - C-matched at T', out['LONG-HQ']['H1_HQ_minus_Cmatched_at_T'], out['LONG-HQ']['H1_reading'])


if __name__ == '__main__':
    main('--provisional' in sys.argv)
