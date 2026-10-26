"""Analysis of E3 (matched per-update KL, B2 protocol), exactly as fixed in PLAN.md before any main run.

    python b2k_analysis.py [--provisional]     # writes b2k_results.json; refuses incomplete data unless --provisional
    python b2k_analysis.py --stored-table      # descriptive table from STORED runs only -> stored_kl_table.json
    python b2k_analysis.py --selftest          # B-H1, B-H2 of record and the two KL targets from stored runs only

Writes only into this folder. Imports `runner` (this folder) for the configuration the runs must carry and, through it,
p4_train.learning_curve_auc of the unchanged idea_dq tree.

Fixes after review (2026-10-25, REVIEW.md):
B1   B2's frozen rules (idea_dq dq_b2_analysis.py: MAX_DROPPED, t_test, holm, tost), copied: a failed run's block is
     dropped from the contrasts that need it; more than 3 of 36 blocks dropped (or n < 3) -> the contrast is NOT
     ESTIMABLE (not 'not rejected'); a not-estimable contrast is left out of the Holm ordering but the Holm denominator
     stays the family size 2; degenerate sd = 0 -> p = 1 if the mean is 0, p = 0 for a non-zero constant; TOST of a
     not-estimable contrast is reported as 'not estimable'.
B2   INTEGRITY, before any outcome is used; any failure is an IntegrityError: the analysis refuses and writes nothing.
     FROZEN    PLAN.md matches PLAN.sha256 and records the sha256 of this file and of runner.py; precheck.json passed
               with the imported runner.py; selection.json is ready and its lr strings are in PLAN.md; live tree ffe9dfef.
     IDENTITY  records/ holds only <T1_quantum|T2_surrogate625>__b<0-35>.json; each record names its set, arm, block
               and kind 'main'; learning_rate = selection.json; plan_sha256 = PLAN.sha256; runner_sha256 = the
               precheck's; code_tree_sha256 = ffe9dfef...; config = runner.config_main(arm, block, lr, PLAN hash)
               (canonical JSON).
     RAW       for a completed run: exactly one arm_start (first) and one arm_end (last); update indices 1..U without
               gap or duplicate, U = arm_end's and the record's update count; the evaluation points are the grid
               exactly once, in order; finite J; AUC recomputed with p4_train.learning_curve_auc equal to the arm_end's
               and the record's; status, decisions, episodes, updates agree; one final checkpoint; the record's KL
               statistic equals the one recomputed from these update events.
     OUTCOME   a non-completed record ('algorithmic_failure' from the trainer, 'failure' from the runner) never
               coexists with a completed arm_end or a final checkpoint, and carries its error text.
     TWIN      a completed twin run needs a failure-free static certificate and P4 parity <= 1e-9.
     An unparsable record counts as no record (runner.py moves it to interrupted/ and re-runs the run).
M1   descriptive contrasts carry no p-value.
M2   K-H2 is reported beside B-H2 of record (B_H2_RECORD); 'inconclusive' if its 95% interval contains both 0 and
     B_H2_RECORD; 'partial reduction' if Holm-rejected below zero with the interval entirely above B_H2_RECORD;
     AUC(twin new) - AUC(twin B2, 1e-3) is the direct descriptive measure (paired, same initial policy).
Lead, after B2F: descriptive AUC(twin new) - AUC(HQ frozen angles, idea_dq_posthoc_b2f), read as b2f_analysis.py does.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
CH = Path('/home/konstantinos/Documents/ChatGPT')
B2 = CH / 'idea_dq/output/study/gnss_qrl_v1/dq/b2_campaign_v3/runs'
CMLR, LR, HQLR = CH / 'idea_dq_posthoc_cmlr/runs', CH / 'idea_dq_posthoc_lr/runs', CH / 'idea_dq_posthoc_hqlr/runs'
B2F = CH / 'idea_dq_posthoc_b2f/runs'
BLOCKS = tuple(range(36))
B2_TREE = 'ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952'
#: fixed in the plan (identical to runner.TARGETS; --selftest recomputes them from the stored records)
T_H2 = 0.007823255686871039          # stored B2 HQ (lr 3e-3): target of the new twin (K-H2)
T_H1 = 0.00419994415996206           # stored CMLR C-matched (lr 1e-2): target of the new HQ (K-H1)
#: B-H2 of record: mean over blocks 0-35 of AUC(HQ, B2, 3e-3) - AUC(twin, B2, 1e-3) (--selftest recomputes it exactly)
B_H2_RECORD = -0.007772148908780312
TOLERANCE = 0.25
TOST_DELTA = 0.0056                  # dq_b2.TOST_DELTA
ALPHA = 0.05                         # dq_b2_analysis.ALPHA
MAX_DROPPED = 3                      # dq_b2_analysis.MAX_DROPPED
SETS = {'T1': 'quantum', 'T2': 'surrogate625'}
EXPECTED = tuple(f'{tag}_{arm}__b{b}' for b in BLOCKS for tag, arm in SETS.items())
OUTCOME_STATUSES = ('algorithmic_failure', 'failure')
#: stored run sets: label -> (root, directory pattern, arm, learning rate)
STORED = {'HQ_1e-3': (LR, 'L2_quantum__b{b}', 'quantum', 1e-3), 'HQ_3e-3': (B2, 'quantum__b{b}', 'quantum', 3e-3),
          'HQ_1e-2': (HQLR, 'H1_quantum__b{b}', 'quantum', 1e-2),
          'Cmatched_3e-3': (B2, 'bottleneck__b{b}', 'bottleneck', 3e-3),
          'Cmatched_1e-2': (CMLR, 'C1_bottleneck__b{b}', 'bottleneck', 1e-2),
          'twin_1e-3': (B2, 'surrogate625__b{b}', 'surrogate625', 1e-3),
          'twin_3e-3': (LR, 'L1_surrogate625__b{b}', 'surrogate625', 3e-3)}


class IntegrityError(Exception):
    """A contradictory, tampered, incomplete or foreign artifact: the analysis refuses and writes nothing."""


# -- stored runs (results of record, proven by their own analyses) -------------------------------------------------------

def summary(path):
    """AUC, J at first/final checkpoint and the KL statistic of one completed STORED run; None if not completed."""
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    last = json.loads(lines[-1])
    if last.get('kind') != 'arm_end' or last.get('status') != 'completed':
        return None
    kls = [float(json.loads(x)['metrics']['approx_kl']) for x in lines if '"kind": "update"' in x]
    lc = last['learning_curve']['sample']
    return {'auc': last['auc']['sample'], 'J0': lc[0][2], 'Jfinal': lc[-1][2], 'kl': float(np.mean(kls)),
            'kl_updates': kls, 'updates': len(kls)}


def run_set(root, pattern, arm, blocks=BLOCKS):
    out = {}
    for b in blocks:
        s = summary(Path(root) / pattern.format(b=b) / 'run' / arm / 'metrics.jsonl')
        if s is not None:
            out[b] = s
    return out


def load_stored():
    return {k: run_set(r, p, a) for k, (r, p, a, _) in STORED.items()}


def b2f_summary(path):
    """As idea_dq_posthoc_b2f/b2f_analysis.summary: the run's summary line (the last line of metrics.jsonl) only."""
    path = Path(path)
    if not path.exists():
        return None
    last = json.loads(path.read_text().splitlines()[-1])
    if last.get('status') != 'completed':
        return None
    return {'auc': last['auc']['sample']}


def load_b2f(root=B2F, blocks=BLOCKS):
    """HQ with frozen circuit angles (B2F, lr 3e-3, PLAN a5e573b3...): {block: {'auc'}} of its completed runs."""
    out = {}
    for b in blocks:
        s = b2f_summary(Path(root) / f'F_quantum__b{b}' / 'run' / 'quantum' / 'metrics.jsonl')
        if s is not None:
            out[b] = s
    return out


# -- statistics: B2's frozen rules (dq_b2_analysis.py), copied --------------------------------------------------------

def differences(a, b, key='auc', blocks=BLOCKS):
    """Per-block a - b; a block missing from either side (failed or absent run) is dropped (listed)."""
    values, dropped = {}, []
    for blk in blocks:
        if blk in a and blk in b:
            values[blk] = a[blk][key] - b[blk][key]
        else:
            dropped.append(blk)
    return values, dropped


def t_test(values, dropped):
    """dq_b2_analysis.t_test (two-sided paired t over blocks), plus dz and the share of blocks below zero."""
    x = np.asarray(list(values.values()), float)
    n = x.size
    record = {'n': int(n), 'dropped_blocks': list(dropped), 'block_values': {str(k): v for k, v in values.items()}}
    if len(dropped) > MAX_DROPPED or n < 3:
        return dict(record, estimable=False, p=None)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    if sd == 0:
        t, p, degenerate = (0., 1., True) if mean == 0 else (math.copysign(math.inf, mean), 0., True)
    else:
        t, degenerate = mean / se, False
        p = float(2 * stats.t.sf(abs(t), n - 1))
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return dict(record, estimable=True, mean=mean, sd=sd, se=se, t=float(t), p=float(p), degenerate=degenerate,
                ci95=[mean - q95 * se, mean + q95 * se], ci90=[mean - q90 * se, mean + q90 * se],
                standardised_effect=mean / sd if sd > 0 else None, dz=mean / sd if sd > 0 else None,
                share_negative=float(np.mean(x < 0)))


def holm(tests):
    """dq_b2_analysis.holm: estimable tests in p order; the denominator is the FAMILY size (len(tests))."""
    ordered = sorted(((r['p'], n) for n, r in tests.items() if r.get('estimable')), key=lambda z: z[0])
    out = {n: {'rejected': False, 'holm_threshold': None} for n in tests}
    for rank, (p, n) in enumerate(ordered):
        thr = ALPHA / (len(tests) - rank)
        out[n]['holm_threshold'] = thr
        if p <= thr:
            out[n]['rejected'] = True
        else:
            break
    return out


def tost(test):
    """dq_b2_analysis.tost: equivalent within +-0.0056 only if the 90% t interval lies inside."""
    if not test.get('estimable'):
        return {'equivalent_within_delta': False, 'delta': TOST_DELTA, 'reason': 'not estimable'}
    lo, hi = test['ci90']
    return {'equivalent_within_delta': bool(-TOST_DELTA < lo and hi < TOST_DELTA), 'delta': TOST_DELTA, 'ci90': [lo, hi]}


def interval(a, b, key='auc'):
    """Descriptive (M1): mean, 95% paired t interval, dz, share below zero; NO p-value, no decision."""
    values, dropped = differences(a, b, key)
    x = np.asarray(list(values.values()), float)
    n = int(x.size)
    if n < 3:
        return {'n': n, 'dropped_blocks': dropped}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    se, q = sd / np.sqrt(n), stats.t.ppf(0.975, n - 1)          # the arithmetic of the earlier paired()/interval()
    return {'n': n, 'dropped_blocks': dropped, 'mean': m, 'lo': float(m - q * se), 'hi': float(m + q * se),
            'dz': m / sd if sd > 0 else None, 'share_negative': float(np.mean(x < 0))}


def kl_summary(runs):
    if len(runs) < 2:
        return {'n': len(runs)}
    v = np.array([s['kl'] for s in runs.values()])
    pooled = np.concatenate([s['kl_updates'] for s in runs.values()])
    se = v.std(ddof=1) / np.sqrt(len(v))
    q = stats.t.ppf(0.975, len(v) - 1)
    return {'n': len(v), 'mean': float(v.mean()), 'lo': float(v.mean() - q * se), 'hi': float(v.mean() + q * se),
            'median_of_run_means': float(np.median(v)), 'pooled_update_median': float(np.median(pooled)),
            'pooled_update_p90': float(np.percentile(pooled, 90))}


# -- reading rules (PLAN.md "Reading rules") ---------------------------------------------------------------------------

NOT_ESTIMABLE = ('NOT ESTIMABLE: more than 3 of 36 blocks dropped (or fewer than 3 paired blocks); not tested and not '
                 'rejected; left out of the Holm ordering, whose denominator stays 2')
READINGS = {
    ('K-H2', 'below zero'): 'at matched mean per-update KL HQ still learned less than its exact twin, and the 95% '
                            'interval does not lie above B-H2 of record; B-H2 is not explained by the twin\'s larger '
                            'updates as measured by this statistic',
    ('K-H2', 'partial reduction'): 'at matched mean per-update KL HQ still learned less than its exact twin, but the '
                                   'whole 95% interval lies above B-H2 of record (-0.00777): the twin\'s larger updates '
                                   'account for part of B-H2, not all of it, as measured by this statistic',
    ('K-H2', 'inconclusive'): 'the 95% interval contains both 0 and B-H2 of record (-0.00777): at matched per-update '
                              'KL the check neither removes nor confirms B-H2 (inconclusive; not evidence of equality)',
    ('K-H2', 'not resolved'): 'at matched per-update KL the HQ - twin difference is not resolved and its 95% interval '
                              'excludes B-H2 of record; the B-H2 difference is consistent with an update-size account '
                              '(not evidence of equality)',
    ('K-H2', 'not resolved after Holm'): 'not resolved after Holm, although the unadjusted 95% interval excludes 0; no '
                                         'update-size statement',
    ('K-H2', 'above zero'): 'at matched per-update KL HQ learned more than its twin; B-H2 reverses and is reported as '
                            'dependent on update size',
    ('K-H1', 'below zero'): 'at matched mean per-update KL HQ learned less than C-matched; B-H1 is not explained by '
                            'update size as measured by this statistic',
    ('K-H1', 'not resolved'): 'at matched per-update KL the HQ - C-matched difference is not resolved; B-H1 is '
                              'consistent with a dependence on update size (not evidence of equality)',
    ('K-H1', 'not resolved after Holm'): 'not resolved after Holm, although the unadjusted 95% interval excludes 0; no '
                                         'update-size statement',
    ('K-H1', 'above zero'): 'at matched per-update KL HQ learned more than C-matched; B-H1 is reported as dependent '
                            'on update size'}


def beside_b_h2(test):
    """M2: K-H2's 95% interval beside B-H2 of record."""
    if not test.get('estimable'):
        return {'B-H2_of_record': B_H2_RECORD, 'K-H2_estimable': False}
    lo, hi = test['ci95']
    return {'B-H2_of_record': B_H2_RECORD, 'K-H2_mean': test['mean'], 'K-H2_ci95': [lo, hi],
            'contains_0': bool(lo <= 0 <= hi), 'contains_B-H2_of_record': bool(lo <= B_H2_RECORD <= hi),
            'entirely_above_B-H2_of_record': bool(lo > B_H2_RECORD)}


def direction(name, test):
    """Both contrasts: 'not estimable' | 'below zero' | 'above zero' (Holm-rejected) | 'not resolved' (not rejected,
    unadjusted 95% interval contains 0) | 'not resolved after Holm' (not rejected although the unadjusted 95% interval
    excludes 0: tested at Holm's 0.025, or not tested because Holm stopped at the other contrast). K-H2 only (M2):
    'partial reduction' (instead of 'below zero') and 'inconclusive' (instead of 'not resolved')."""
    if not test.get('estimable'):
        return 'not estimable'
    rejected, mean = test['rejected'], test['mean']
    lo, hi = test['ci95']
    if rejected and mean < 0:
        return 'partial reduction' if name == 'K-H2' and lo > B_H2_RECORD else 'below zero'
    if rejected and mean > 0:
        return 'above zero'
    if lo <= 0 <= hi:
        return 'inconclusive' if name == 'K-H2' and lo <= B_H2_RECORD else 'not resolved'
    return 'not resolved after Holm'


def reading(name, test, matched, ratio):
    d = direction(name, test)
    if d == 'not estimable':
        return d, NOT_ESTIMABLE
    if matched:
        text = READINGS[(name, d)]
        if d == 'not resolved' and test['tost']['equivalent_within_delta']:
            text += '; the difference at matched KL is also within B2\'s +-0.0056 equivalence margin'
        return d, text
    label = f'KL not matched (R = {ratio:.2f})'
    below = test['rejected'] and test['mean'] < 0
    if name == 'K-H2' and ratio < 1 - TOLERANCE and below:
        return d, f'{label}: HQ learned less than a twin whose updates were smaller'
    if name == 'K-H1' and ratio > 1 + TOLERANCE and below:
        return d, f'{label}: HQ learned less than C-matched although its updates were larger'
    return d, f'{label}: no update-size statement'


# -- integrity (review B2) ---------------------------------------------------------------------------------------------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=float)


def read_record(path):
    """A record that does not parse counts as no record (as runner.read_record)."""
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_records(root):
    """{run name: record} of the parsable records, and the names of unparsable ones; refuses foreign entries."""
    d = Path(root) / 'records'
    entries = sorted(d.iterdir()) if d.exists() else []
    foreign = [p.name for p in entries if not p.name.endswith('.json') or p.name[:-5] not in EXPECTED or not p.is_file()]
    if foreign:
        raise IntegrityError(f'records/: unexpected entries {foreign[:5]}')
    records, unparsable = {}, []
    for p in entries:
        r = read_record(p)
        if r is None:
            unparsable.append(p.name)
        else:
            records[p.name[:-5]] = r
    return records, unparsable


def frozen_identity(root):
    """FROZEN: what every record must carry, from PLAN.md/PLAN.sha256, precheck.json and selection.json."""
    import runner
    root = Path(root)
    try:
        plan = (root / 'PLAN.md').read_bytes()
        declared = (root / 'PLAN.sha256').read_text().split()[0]
    except (FileNotFoundError, IndexError) as exc:
        raise IntegrityError('PLAN.md / PLAN.sha256 missing or empty') from exc
    if sha256_bytes(plan) != declared:
        raise IntegrityError('PLAN.md does not match PLAN.sha256')
    text = plan.decode()
    own = sha256_bytes(Path(__file__).read_bytes())
    if own not in text:
        raise IntegrityError(f'PLAN.md does not record the sha256 of this b2k_analysis.py ({own})')
    if runner.RUNNER_SHA256 not in text:
        raise IntegrityError(f'PLAN.md does not record the sha256 of runner.py ({runner.RUNNER_SHA256})')
    pre = json.loads((root / 'precheck.json').read_text())
    if not pre.get('passed') or pre.get('runner_sha256') != runner.RUNNER_SHA256:
        raise IntegrityError('precheck.json: not passed, or made by another runner.py than the one imported here')
    sel = json.loads((root / 'selection.json').read_text())
    if not sel.get('ready'):
        raise IntegrityError('selection.json is not ready')
    lr = {}
    for arm in SETS.values():
        s = sel['arms'][arm]
        if s['lr_str'] not in text or float(s['lr_str']) != s['lr']:
            raise IntegrityError(f'selection.json {arm}: lr_str not in PLAN.md, or lr and lr_str disagree')
        lr[arm] = s['lr']
    if runner.B2_TREE != B2_TREE or runner.p4.live_copy_tree() != B2_TREE:
        raise IntegrityError('the live idea_dq tree is not ffe9dfef...')
    return {'plan_sha256': declared, 'runner_sha256': pre['runner_sha256'], 'analysis_sha256': own, 'lr': lr,
            'tree': B2_TREE}


def parse_name(name):
    tag, rest = name.split('_', 1)
    arm, block = rest.split('__b')
    return tag, arm, int(block)


def check_identity(name, r, ident, expected_config):
    tag, arm, block = parse_name(name)
    if (r.get('kind'), r.get('set'), r.get('arm'), r.get('block')) != ('main', tag, arm, block):
        raise IntegrityError(f'{name}: the record names {(r.get("kind"), r.get("set"), r.get("arm"), r.get("block"))}')
    if r.get('learning_rate') != ident['lr'][arm]:
        raise IntegrityError(f'{name}: learning rate {r.get("learning_rate")} is not the selected {ident["lr"][arm]}')
    if r.get('plan_sha256') != ident['plan_sha256'] or r.get('runner_sha256') != ident['runner_sha256']:
        raise IntegrityError(f'{name}: plan or runner identity mismatch')
    if r.get('status') == 'completed' and ('config' not in r or 'code_tree_sha256' not in r):
        raise IntegrityError(f'{name}: completed without its configuration or code-tree identity')
    if 'config' in r and canonical(r['config']) != canonical(expected_config):
        raise IntegrityError(f'{name}: the configuration is not runner.config_main(arm, block, selected lr, PLAN hash)')
    if 'code_tree_sha256' in r and r['code_tree_sha256'] != ident['tree']:
        raise IntegrityError(f'{name}: code tree {r["code_tree_sha256"]}')


def read_events(path, name):
    if not Path(path).exists():
        raise IntegrityError(f'{name}: metrics.jsonl missing')
    events = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IntegrityError(f'{name}: malformed event line') from exc
    return events


def check_completed_run(root, name, r, cfg):
    """RAW EVENTS of a completed run (dq_b2_analysis.check_completed_run, adapted); returns its proven summary."""
    import runner
    _, arm, _ = parse_name(name)
    arm_dir = Path(root) / 'runs' / name / 'run' / arm
    events = read_events(arm_dir / 'metrics.jsonl', name)
    if not events or events[0].get('kind') != 'arm_start' or events[-1].get('kind') != 'arm_end':
        raise IntegrityError(f'{name}: the event log does not start with arm_start and end with arm_end')
    if sum(e.get('kind') == 'arm_start' for e in events) != 1 or sum(e.get('kind') == 'arm_end' for e in events) != 1:
        raise IntegrityError(f'{name}: more than one arm_start / arm_end')
    updates = [e for e in events if e.get('kind') == 'update']
    end, summ = events[-1], r.get('summary') or {}
    if [e.get('update_index') for e in updates] != list(range(1, len(updates) + 1)):
        raise IntegrityError(f'{name}: update indices are not 1..U without gap or duplicate')
    if end.get('updates') != len(updates) or summ.get('updates') != len(updates):
        raise IntegrityError(f'{name}: U = {len(updates)} update events, arm_end {end.get("updates")}, record '
                             f'{summ.get("updates")}')
    finals = [e for e in events if e.get('kind') == 'checkpoint' and e.get('final')]
    if len(finals) != 1:
        raise IntegrityError(f'{name}: {len(finals)} final checkpoints')
    points, values = [], []
    for e in events:
        if e.get('kind') != 'evaluation':
            continue
        v = e['results']['sample']['weighted_J']
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise IntegrityError(f'{name}: non-finite evaluation value')
        for p in e['points']:
            points.append(p)
            values.append(v)
    grid, budget = cfg['checkpoint_decisions'], cfg['total_policy_decisions']
    if points != list(grid):
        raise IntegrityError(f'{name}: evaluation points {points} are not the grid exactly once, in order')
    auc = runner.p4.learning_curve_auc(points, values, budget)
    for source, s in (('arm_end', end), ('record', summ)):
        if s.get('status') != 'completed' or (s.get('auc') or {}).get('sample') != auc:
            raise IntegrityError(f'{name}: the {source} AUC or status disagrees with the raw evaluation events')
    for key in ('policy_decisions', 'episodes', 'updates', 'block', 'arm'):
        if end.get(key) != summ.get(key):
            raise IntegrityError(f'{name}: record and arm_end disagree on {key}')
    kls = [float(e['metrics']['approx_kl']) for e in updates]
    kl = float(np.mean(kls))
    if (r.get('kl') or {}).get('mean') != kl or (r.get('kl') or {}).get('updates') != len(kls):
        raise IntegrityError(f'{name}: the record\'s KL statistic differs from the update events')
    return {'auc': auc, 'J0': values[0], 'Jfinal': values[-1], 'kl': kl, 'kl_updates': kls, 'updates': len(kls)}


def check_twin(name, r):
    tw = r.get('twin_parity') or {}
    static, p4_ = tw.get('static') or {}, tw.get('P4') or {}
    if static.get('static_failures') != [] or not isinstance(p4_.get('max_abs_dlogprob'), (int, float)) or \
            not p4_['max_abs_dlogprob'] <= 1e-9:
        raise IntegrityError(f'{name}: completed without a failure-free static certificate and P4 <= 1e-9')


def check_outcome(root, name, r):
    _, arm, _ = parse_name(name)
    status = r.get('status')
    if status == 'failure' and not isinstance(r.get('error'), str):
        raise IntegrityError(f'{name}: failure without its error text')
    if (r.get('summary') or {}).get('status') == 'completed':
        raise IntegrityError(f'{name}: outcome {status} with a completed summary')
    arm_dir = Path(root) / 'runs' / name / 'run' / arm
    ck = arm_dir / 'checkpoints'
    if ck.exists() and any(p.name.startswith('final_') for p in ck.iterdir()):
        raise IntegrityError(f'{name}: outcome {status} but a final checkpoint exists')
    events = read_events(arm_dir / 'metrics.jsonl', name) if (arm_dir / 'metrics.jsonl').exists() else []
    ends = [e for e in events if e.get('kind') == 'arm_end']
    if any(e.get('status') == 'completed' for e in ends):
        raise IntegrityError(f'{name}: outcome {status} but the trainer finished (completed arm_end)')
    if status == 'algorithmic_failure' and (len(ends) != 1 or ends[0].get('status') != 'algorithmic_failure'):
        raise IntegrityError(f'{name}: algorithmic failure not recorded as such by the trainer')


def prove(root, records, ident):
    """IDENTITY, RAW EVENTS, OUTCOME and TWIN for every parsable record: {(tag, block): proven summary or None}."""
    import runner
    proven = {}
    for name, r in sorted(records.items()):
        tag, arm, block = parse_name(name)
        cfg = runner.config_main(arm, block, ident['lr'][arm], plan=ident['plan_sha256'])
        check_identity(name, r, ident, cfg)
        status = r.get('status')
        if status == 'completed':
            run = check_completed_run(root, name, r, json.loads(canonical(cfg)))
            if arm == 'surrogate625':
                check_twin(name, r)
            proven[(tag, block)] = run
        elif status in OUTCOME_STATUSES:
            check_outcome(root, name, r)
            proven[(tag, block)] = None
        else:
            raise IntegrityError(f'{name}: unknown status {status!r}')
    return proven


def twin_parity_summary(records):
    p4 = [r['twin_parity']['P4']['max_abs_dlogprob'] for n, r in records.items()
          if n.startswith('T2_') and r.get('status') == 'completed']
    return {'completed_twin_records': len(p4), 'P4_max_abs_dlogprob_max': max(p4) if p4 else None,
            'all_P4_within_1e-9_and_static_clean': True if p4 else None}     # otherwise prove() refused


# -- the read-out --------------------------------------------------------------------------------------------------------

def main(provisional=False, root=HERE, stored=None, b2f=None):
    root = Path(root)
    records, unparsable = load_records(root)
    complete = len(records) == len(EXPECTED)
    if not complete and not provisional:
        raise SystemExit(f'incomplete: {len(records)} of 72 parsable records (unparsable, counted as no record: '
                         f'{unparsable}); the read-out waits for all 72 (runner.py launch moves unparsable ones aside)')
    ident = frozen_identity(root)
    proven = prove(root, records, ident)
    twin_new = {b: v for (tag, b), v in proven.items() if tag == 'T2' and v is not None}
    hq_new = {b: v for (tag, b), v in proven.items() if tag == 'T1' and v is not None}
    st = load_stored() if stored is None else stored
    hq_b2, cm_cmlr = st['HQ_3e-3'], st['Cmatched_1e-2']
    failed = [{'record': n, 'status': r.get('status'), 'error': r.get('error')} for n, r in sorted(records.items())
              if r.get('status') != 'completed']
    tests = {'K-H1': t_test(*differences(hq_new, cm_cmlr)), 'K-H2': t_test(*differences(hq_b2, twin_new))}
    decisions = holm(tests)
    for name, t in tests.items():
        t.update(decisions[name], tost=tost(t))
    KH1, KH2 = tests['K-H1'], tests['K-H2']
    kl = {'twin_new': kl_summary(twin_new), 'HQ_new': kl_summary(hq_new),
          'HQ_B2_3e-3 (target K-H2)': kl_summary(hq_b2), 'Cmatched_CMLR_1e-2 (target K-H1)': kl_summary(cm_cmlr)}
    match = {}
    for name, arm_key, target in (('K-H2', 'twin_new', T_H2), ('K-H1', 'HQ_new', T_H1)):
        achieved = kl[arm_key].get('mean')
        ratio = achieved / target if achieved is not None else None
        match[name] = {'target': target, 'achieved': achieved, 'R': ratio,
                       'matched': bool(ratio is not None and 1 - TOLERANCE <= ratio <= 1 + TOLERANCE)}
    read = {}
    for name, c in (('K-H2', KH2), ('K-H1', KH1)):
        d, text = reading(name, c, match[name]['matched'], match[name]['R'] if match[name]['R'] is not None else math.nan)
        read[name] = {'direction': d, 'reading': text}
    b2f_runs = load_b2f() if b2f is None else b2f
    sets = {'HQ_new': hq_new, 'twin_new': twin_new, **st}
    common = sorted(set(twin_new) & set(hq_b2) & set(st['twin_1e-3']))
    step = interval(twin_new, st['twin_1e-3'])
    bh2_same = interval({b: hq_b2[b] for b in common}, {b: st['twin_1e-3'][b] for b in common})
    desc = {'label': 'descriptive, no p-values, no decisions',
            'AUC_mean': {k: float(np.mean([s['auc'] for s in v.values()])) for k, v in sets.items() if v},
            'J0_mean': {k: float(np.mean([s['J0'] for s in v.values()])) for k, v in sets.items() if v},
            'Jfinal_mean': {k: float(np.mean([s['Jfinal'] for s in v.values()])) for k, v in sets.items() if v},
            'K-H2_beside_B-H2_of_record': beside_b_h2(KH2),
            'twin_new_minus_twin_B2_1e-3': dict(step, note=(
                'direct measure of the twin step-size share of B-H2: paired, same initial policy and streams, only the '
                'learning rate differs (KL-matched lr vs 1e-3); per block it equals B-H2 - K-H2, so a negative value '
                'is the part of B-H2 that the twin\'s larger step accounts for')),
            'share_of_B-H2_removed_point': (step['mean'] / bh2_same['mean']
                                            if 'mean' in step and bh2_same.get('mean') else None),
            'twin_new_minus_HQ_frozen_B2F': dict(interval(twin_new, b2f_runs), note=(
                'AUC(twin, KL-matched lr) - AUC(HQ with frozen circuit angles, idea_dq_posthoc_b2f, lr 3e-3), same '
                'blocks; HQ frozen is the twin with its coefficients frozen (same model, same gradients), but the '
                'classical layers\' learning rates differ (7.79e-04 vs 3e-3), so this is not an exact decomposition'),
                n_b2f_completed=len(b2f_runs)),
            'HQ_new_minus_HQ_B2_3e-3': interval(hq_new, hq_b2), 'HQ_new_minus_HQ_1e-3': interval(hq_new, st['HQ_1e-3']),
            'B-H1_of_record_same_blocks': interval(hq_b2, st['Cmatched_3e-3']),
            'B-H2_of_record_same_blocks': interval(hq_b2, st['twin_1e-3']),
            'per_block_KL_ratio_median': {
                'K-H2 twin_new / HQ_B2': (float(np.median([twin_new[b]['kl'] / hq_b2[b]['kl'] for b in twin_new
                                                           if b in hq_b2])) if twin_new else None),
                'K-H1 HQ_new / Cmatched_CMLR': (float(np.median([hq_new[b]['kl'] / cm_cmlr[b]['kl'] for b in hq_new
                                                                 if b in cm_cmlr])) if hq_new else None)},
            'twin_parity': twin_parity_summary(records)}
    out = {'plan': 'PLAN.md', 'plan_sha256': ident['plan_sha256'], 'analysis_sha256': ident['analysis_sha256'],
           'runner_sha256': ident['runner_sha256'], 'provisional': not complete, 'records': len(records),
           'unparsable_records_counted_as_none': unparsable,
           'integrity': 'FROZEN, IDENTITY, RAW EVENTS, OUTCOME, TWIN: all records proven',
           'learning_rates': ident['lr'],
           'completed': {'T1_HQ': len(hq_new), 'T2_twin': len(twin_new)}, 'failed_runs': failed,
           'K-H2_HQ_B2_minus_twin_new': KH2, 'K-H1_HQ_new_minus_Cmatched_CMLR': KH1, 'kl_match': match,
           'readings': read, 'kl': kl, 'descriptive': desc}
    name = 'b2k_results.json' if complete else 'b2k_results_PROVISIONAL.json'
    (root / name).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('provisional', 'records', 'completed', 'kl_match', 'readings')}, indent=1))
    for k in ('K-H2_HQ_B2_minus_twin_new', 'K-H1_HQ_new_minus_Cmatched_CMLR'):
        print(k, json.dumps({q: v for q, v in out[k].items() if q != 'block_values'}))
    return out


def stored_table(root=HERE):
    """Descriptive post-hoc table from STORED runs only: no p-values, no decisions."""
    st = load_stored()
    rows = []
    for comp in ('Cmatched_3e-3', 'Cmatched_1e-2', 'twin_1e-3', 'twin_3e-3'):
        for hq in ('HQ_1e-3', 'HQ_3e-3', 'HQ_1e-2'):
            a, b = st[hq], st[comp]
            kla = float(np.mean([a[k]['kl'] for k in sorted(set(a) & set(b))]))
            klb = float(np.mean([b[k]['kl'] for k in sorted(set(a) & set(b))]))
            iv = interval(a, b)
            rows.append({'contrast': f'{hq} - {comp}', 'HQ_lr': STORED[hq][3], 'comparator_lr': STORED[comp][3],
                         'HQ_mean_KL': kla, 'comparator_mean_KL': klb, 'KL_ratio_HQ_over_comparator': kla / klb,
                         **{k: iv[k] for k in ('n', 'mean', 'lo', 'hi', 'dz', 'share_negative')}})
    out = {'label': 'descriptive, post hoc, stored runs only; no p-values, no decisions',
           'statistic': 'AUC (sample); KL = mean over blocks of the run mean of update approx_kl',
           'sets': {k: {'root': str(r), 'pattern': p, 'arm': a, 'lr': lr, 'n_completed': len(st[k])}
                    for k, (r, p, a, lr) in STORED.items()},
           'excluded': 'idea_dq_posthoc_b2g (HQ from generic angles: another initialisation)', 'rows': rows}
    (Path(root) / 'stored_kl_table.json').write_text(json.dumps(out, indent=1, default=float))
    for r in rows:
        print(f"{r['contrast']:<28} KL {r['HQ_mean_KL']:.4f} vs {r['comparator_mean_KL']:.4f} (x{r['KL_ratio_HQ_over_comparator']:.2f})"
              f"  dAUC {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  HQ lower in {100 * r['share_negative']:.0f}%  n={r['n']}")
    return out


def selftest():
    st = load_stored()
    assert all(len(v) == 36 for v in st.values()), {k: len(v) for k, v in st.items()}
    bh1 = t_test(*differences(st['HQ_3e-3'], st['Cmatched_3e-3']))
    bh2 = t_test(*differences(st['HQ_3e-3'], st['twin_1e-3']))
    print('B-H1', round(bh1['mean'], 5), f"{bh1['p']:.2g}", ' B-H2', round(bh2['mean'], 5), f"{bh2['p']:.2g}")
    assert abs(bh1['mean'] + 0.0087) < 5e-5 and abs(bh1['p'] - 0.0017) < 2e-4, bh1
    assert abs(bh2['mean'] + 0.0078) < 5e-5 and abs(bh2['p'] - 0.0012) < 2e-4, bh2
    assert bh2['mean'] == B_H2_RECORD, (bh2['mean'], B_H2_RECORD)
    t2 = float(np.mean([st['HQ_3e-3'][b]['kl'] for b in BLOCKS]))
    t1 = float(np.mean([st['Cmatched_1e-2'][b]['kl'] for b in BLOCKS]))
    print('targets', t2, t1)
    assert t2 == T_H2 and t1 == T_H1
    n_b2f = len(load_b2f())
    print('B2F completed runs readable:', n_b2f)
    assert n_b2f == 36
    print('selftest passed')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--stored-table' in sys.argv:
        stored_table()
    else:
        main('--provisional' in sys.argv)
