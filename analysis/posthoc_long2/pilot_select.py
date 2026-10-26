"""Pilot selection of post-hoc check E4: THE rule of PLAN.md, section 'Development pilot' (amended 2026-10-25 after
review item B1, while the pilot was running and before any pilot result was read). Writes pilot/selection.json once.

    python -B pilot_select.py           # refuses unless PLAN.md and PLAN.sha256 match and PLAN.md names the SHA-256 of
                                        # this file (and the other frozen files), all 8 pilot runs stand (the restart
                                        # rule applied) and pilot/selection.json does not exist
    python -B pilot_select.py --check   # integrity of the pilot runs that have a record (others: 'absent'); prints
                                        # pass/fail only, computes no selection statistic and prints no outcome

The rule, per head (C-strong 'mlp', C-matched 'bottleneck'), candidates e in {0.003, 0.01}:
  eligible(e)  the run of e completed in both development blocks (6 and 7) AND, in both blocks, its development-suite
               timely-credited reports per episode (mean tp_timely over the 80 episodes of the development evaluation)
               exceed 0.012 at 80,000 AND at 100,000 decisions. 0.012 is the per-episode level at or below which
               LONG's PLAN_LOG (2026-10-24T16:37:30Z) classed the C-strong blocks as having ended without timely
               reports.
  S(e)         mean weighted development J at 80,000 and 100,000 decisions over both development blocks.
  choice       one eligible value: it. Two: the larger S, except |S(0.01) - S(0.003)| < 0.0056 (dq_b2.TOST_DELTA)
               chooses 0.01 (the larger entropy). None eligible: 0.01 (fallback, fixed in PLAN.md).
  descriptive  the mean policy entropy over the last 20 PPO updates of every run (no role in the choice).
A pilot run whose record is a non-algorithmic failure must have been restarted once (runner.py pilot does that); a
second failure stands and the value is then not 'completed in both blocks'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner                                                                                     # noqa: E402
import e4_analysis as ea                                                                          # noqa: E402
import e4lib as e4                                                                                # noqa: E402
import dq_b2                                                                                      # noqa: E402
import p4_train as p4                                                                             # noqa: E402

CANDIDATES = tuple(sorted(e4.ENTROPY_CANDIDATES))
assert CANDIDATES == (0.003, 0.01)
DEV_BLOCKS = tuple(e4.DEV_BLOCKS)
POINTS = tuple(e4.SELECTION_POINTS)
assert DEV_BLOCKS == (6, 7) and POINTS == (80_000, 100_000)
TIMELY_CUT = 0.012                       # LONG PLAN_LOG 2026-10-24T16:37:30Z: <= 0.012 = ended without timely reports
TIE_BAND = dq_b2.TOST_DELTA              # 0.0056 J = 0.1 report-equivalent per episode
assert TIE_BAND == 0.0056 == e4.SELECTION_TIE_J
FALLBACK = max(CANDIDATES)               # 0.01
ENTROPY_WINDOW = 20
ORIGINAL_RULE = (
    "Eligible: completed in both blocks (no algorithmic failure) and last-update mean policy entropy >= 0.30 nats in "
    "both blocks (LONG's collapse level). If no value is eligible, the entropy guard is lifted (failures stay "
    "ineligible); if both values failed, e = 0.01. One eligible value: it is chosen. Two: the larger S, except that "
    "|S(0.01) - S(0.003)| < 0.0056 J (0.1 report-equivalent per episode, dq_b2.TOST_DELTA) chooses 0.003 (the "
    "smaller entropy).")
RULE = {
    'score': f'S(e) = mean weighted development J at {list(POINTS)} over development blocks {list(DEV_BLOCKS)}',
    'eligible': (f'completed in both development blocks and development-suite timely-credited reports per episode '
                 f'> {TIMELY_CUT} at {POINTS[0]} AND {POINTS[1]} decisions in both blocks (the level at or below which '
                 "LONG's PLAN_LOG 2026-10-24T16:37:30Z classed C-strong blocks as having ended without timely reports)"),
    'choice': (f'one eligible: it; two: the larger S, except |S(0.01) - S(0.003)| < {TIE_BAND} chooses 0.01 (the larger '
               f'entropy); none eligible: {FALLBACK} (fallback)'),
    'descriptive_only': f'mean policy entropy over the last {ENTROPY_WINDOW} PPO updates of every run',
    'amended': ('2026-10-25, after review item B1, while the pilot was running and before any pilot result was read; '
                'it replaces the rule of PLAN_DRAFT.md quoted in original_rule'),
    'original_rule': ORIGINAL_RULE}
_RT = None


def runtime():
    global _RT
    if _RT is None:
        _RT = dq_b2.runtime(e4.STUDY_DEV)
    return _RT


# ------------------------------------------------------------------ the rule (pure) -------------------------------
def select_entropy(by_value):
    """THE rule, one head. by_value: {e: {block: summary or None}}, summary = {'status', 'J': {point: J},
    'timely': {point: timely-credited reports per episode}}. Returns (choice, details)."""
    if tuple(sorted(by_value)) != CANDIDATES:
        raise ValueError(f'candidates {sorted(by_value)} are not {CANDIDATES}')
    details = {}
    for e in CANDIDATES:
        runs = by_value[e]
        completed = all(runs.get(b) is not None and runs[b]['status'] == 'completed' for b in DEV_BLOCKS)
        S = float(np.mean([runs[b]['J'][p] for b in DEV_BLOCKS for p in POINTS])) if completed else None
        timely = {b: ({p: runs[b]['timely'][p] for p in POINTS} if runs.get(b) is not None and
                      runs[b]['status'] == 'completed' else None) for b in DEV_BLOCKS}
        above = completed and all(timely[b][p] > TIMELY_CUT for b in DEV_BLOCKS for p in POINTS)
        details[e] = {'completed_both_blocks': completed, 'S': S, 'timely': timely,
                      'timely_above_cut_at_both_points_in_both_blocks': bool(above), 'eligible': bool(above)}
    eligible = [e for e in CANDIDATES if details[e]['eligible']]
    if not eligible:
        return FALLBACK, {'values': details, 'rule_applied': f'no value eligible: fallback {FALLBACK}'}
    if len(eligible) == 1:
        return eligible[0], {'values': details, 'rule_applied': f'one eligible value: {eligible[0]}'}
    small, large = CANDIDATES
    d = details[large]['S'] - details[small]['S']
    if abs(d) < TIE_BAND:
        return large, {'values': details, 'rule_applied': f'both eligible; |S diff| {abs(d):.5f} < {TIE_BAND}: '
                                                          f'the larger entropy {large}'}
    return (large if d > 0 else small), {'values': details,
                                         'rule_applied': f'both eligible; the larger S (S(0.01) - S(0.003) = {d:+.5f})'}


# ------------------------------------------------------------------ reading the pilot ----------------------------
def summarize(events):
    """{'status', 'J', 'timely', 'entropy_last20_mean', ...} of one pilot run's events (integrity checked first)."""
    end = [e for e in events if e.get('kind') == 'arm_end']
    J, timely, pairs = {}, {}, None
    for e in events:
        if e.get('kind') != 'evaluation':
            continue
        res = e['results']['sample']
        eps = res['episodes']
        _, weighted = p4.aggregate_evaluation(runtime(), eps)
        if weighted != res['weighted_J'] or len(eps) != 80:
            raise ea.Refusal('a pilot evaluation does not verify (weighted J from its episodes, 80 episodes)')
        here = [(x['cell_id'], x['seed']) for x in eps]
        if pairs is None:
            pairs = here
        elif here != pairs:
            raise ea.Refusal('pilot evaluations of one run fly different episodes')
        for p in e['points']:
            J[p] = weighted
            timely[p] = float(np.mean([x['tp_timely'] for x in eps]))
    ent = [e['metrics']['entropy'] for e in events if e.get('kind') == 'update']
    return {'status': end[0]['status'] if end else None, 'J': J, 'timely': timely, 'pairs': pairs,
            'entropy_last20_mean': float(np.mean(ent[-ENTROPY_WINDOW:])) if ent else None,
            'updates': len(ent), 'policy_decisions': next((e['cumulative']['policy_decisions'] for e in
                                                            reversed(events) if e.get('kind') == 'update'), 0)}


def pilot_problems(t, rec, events, launch_runners):
    """Integrity and identity of one pilot run (structure only)."""
    p = list(ea.event_problems(*events, e4.PILOT_GRID, expect_arm_end=True))
    lines, evs = events
    code = rec.get('code', {})
    if code.get('e4lib_sha256') != runner.E4LIB_SHA256 or code.get('idea_dq_tree') != e4.EXPECTED_TREE:
        p.append('record e4lib or idea_dq tree is not the pilot\'s')
    if code.get('runner_sha256') not in launch_runners:
        p.append('record runner hash is not that of any pilot launch')
    task = rec.get('task', {})
    for k in ('kind', 'arm', 'block', 'entropy', 'lr0', 'info', 'schedule', 'study', 'budget', 'grid', 'name'):
        if task.get(k) != t[k]:
            p.append(f'record task {k} differs from the pilot design')
    ppo = rec.get('config', {}).get('ppo', {})
    if ppo.get('entropy_coef') != t['entropy'] or ppo.get('learning_rate') != t['lr0']:
        p.append('record config entropy or lr0 differs from the task')
    start = evs[0] if evs else {}
    if start.get('identity', {}).get('condition', {}).get('obs_encoding_version') != e4.OBS_VERSION or \
            start.get('model', {}).get('config', {}).get('obs_dim') != e4.WIDE.dim:
        p.append('not the evidence-count-augmented observation')
    p += ea.schedule_problems(evs, t['arm'], t['entropy'])
    if rec['status'] != (evs[-1].get('status') if evs else None):
        p.append('record status differs from arm_end')
    return p


def load_pilot(root=HERE, require_all=True):
    """[(task, record, class, (lines, events) or None)] for the 8 pilot runs; refuses unless every run stands
    (require_all False, for --check: a run without a record is listed as 'absent')."""
    proot = Path(root) / 'pilot'
    out = []
    for t in runner.pilot_tasks():
        rec_path = proot / 'records' / f'{t["name"]}.json'
        if not rec_path.exists():
            if not require_all:
                out.append((t, None, 'absent', None))
                continue
            raise ea.Refusal(f'pilot incomplete: no record for {t["name"]}')
        rec = json.loads(rec_path.read_text())
        cls = runner.classify(rec)
        if cls == 'infrastructure_failure':
            if runner.failure_restarts(proot, t['name']) == 0:
                raise ea.Refusal(f'{t["name"]}: a first non-algorithmic failure; relaunch the pilot '
                                 '(runner.py pilot restarts it once) before selecting')
            cls = 'code_failure'                                      # a second failure stands
        metrics = proot / 'runs' / t['name'] / 'run' / t['arm'] / 'metrics.jsonl'
        ev = ea.read_lines(metrics) if cls in ('completed', 'algorithmic_failure') else None
        out.append((t, rec, cls, ev))
    return out


def check(root=HERE, require_all=True):
    """Integrity of every standing pilot run; returns {name: {'class', 'problems'}}. Prints nothing."""
    proot = Path(root) / 'pilot'
    launches = [json.loads(x) for x in (proot / 'launches.jsonl').read_text().splitlines() if x.strip()]
    launch_runners = {x['runner_sha256'] for x in launches}
    out = {}
    for t, rec, cls, ev in load_pilot(root, require_all):
        out[t['name']] = {'class': cls, 'problems': pilot_problems(t, rec, ev, launch_runners) if ev else []}
    return out


def main(root=HERE):
    root = Path(root)
    out_path = root / 'pilot' / 'selection.json'
    if out_path.exists():
        raise ea.Refusal(f'{out_path} exists; never overwritten')
    digest, ok = ea.plan_digest(root)
    if not ok:
        raise ea.Refusal('PLAN.md and a matching PLAN.sha256 are required: the lead hashes PLAN.md first')
    named = runner.plan_names_frozen_hashes(root / 'PLAN.md')
    if not all(named.values()):
        raise ea.Refusal(f'PLAN.md does not name the SHA-256 of {[f for f, v in named.items() if not v]}')
    checked = check(root)
    bad = {k: v['problems'] for k, v in checked.items() if v['problems']}
    if bad:
        raise ea.Refusal(f'pilot integrity: {bad}')
    runs = {}
    for t, rec, cls, ev in load_pilot(root):
        s = summarize(ev[1]) if cls == 'completed' else {'status': cls}
        s['status'] = cls
        runs[t['name']] = (t, s)
    pairs = {tuple(map(tuple, s['pairs'])) for _, s in runs.values() if s.get('pairs')}
    if len(pairs) > 1:
        raise ea.Refusal('pilot runs fly different development evaluation episodes')
    for cell_seed in (next(iter(pairs)) if pairs else ()):
        space = runtime().namespaces['dq_development']
        if not space['start'] <= cell_seed[1] < space['start'] + 7_000:
            raise ea.Refusal('a development evaluation seed is outside dq_development [start, start + 7,000)')
    selection = {'utc': runner.utc(), 'plan_sha256': digest, 'pilot_select_sha256': runner.sha(Path(__file__)),
                 **runner.code_identity(), 'rule': RULE, 'heads': {}}
    for arm in e4.ARMS:
        by_value = {e: {t['block']: s for t, s in runs.values() if t['arm'] == arm and t['entropy'] == e}
                    for e in CANDIDATES}
        choice, details = select_entropy(by_value)
        selection['heads'][arm] = {
            'role': e4.ROLES[arm], 'entropy_coef': choice, 'lr0': e4.LR0[arm], **details,
            'runs': {t['name']: {k: v for k, v in s.items() if k != 'pairs'} for t, s in runs.values() if t['arm'] == arm}}
    out_path.write_text(json.dumps(selection, indent=1, default=float))
    print(json.dumps({a: {'entropy_coef': v['entropy_coef'], 'rule_applied': v['rule_applied'],
                          'S': {e: d['S'] for e, d in v['values'].items()},
                          'eligible': {e: d['eligible'] for e, d in v['values'].items()}}
                      for a, v in selection['heads'].items()}, indent=1, default=float))
    return selection


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        res = check(require_all=False)
        print(json.dumps({k: {'class': v['class'], 'passed': None if v['class'] == 'absent' else not v['problems'],
                              'problems': v['problems']}
                          for k, v in res.items()}, indent=1))
    else:
        main()
