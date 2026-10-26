"""Analysis of post-hoc check E4, exactly as fixed in PLAN.md (which freezes this file by SHA-256 together with
runner.py, e4lib.py, pilot_select.py and rule_reference.json; the draft analysis is kept in provenance/).

    python -B e4_analysis.py                      # THE read-out of record -> e4_results.json (see below)
    python -B e4_analysis.py --supplementary-full # after a deadline read-out of record, once every run has ended:
                                                  # the 300,000 read-out -> e4_results_full_after_deadline.json
    python -B e4_analysis.py --provisional --root DIR   # code testing on a COPY only; refused on this folder
    python -B e4_analysis.py --selftest           # reads NO E4 run (stored LONG values, rule reference, checkers)
    add --force to replace an existing output: the old file is kept (renamed) and the replacement is logged in
    e4_analysis_log.jsonl (every write is logged there)

Which read-out is of record (PLAN.md, stopping rule): if every one of the 12 runs has ended (its record stands) by
DEADLINE_UTC, the full read-out at T = 300,000. Otherwise, from DEADLINE_UTC on, the deadline read-out: T is the
largest grid point that every run still running or completed at the deadline had evaluated by the deadline (the time
of an evaluation is the mtime of the checkpoint saved just before it plus the evaluation's wall_s; a run's end is its
arm_end utc, or its record's mtime); it is computed from the files whenever the script is run after the deadline, and
it stays of record if the runs finish later (their 300,000 read-out is then supplementary). Before the deadline with
a run unfinished the script refuses.

Every run is checked before anything is computed, and the script refuses on any failure (B4): raw-event integrity (one
arm_start first, one arm_end last iff the run ended through train_arm, contiguous updates and episodes, no duplicated
line, every grid point evaluated once and in order, each evaluation preceded by its checkpoint); the point-0
evaluation equals LONG's of the same block and head; the (cell_id, seed) sequence of the training episodes equals
LONG's on the overlapping episode numbers; obs_version obs-encoding/4-gnss+ev16/1 and 270 inputs; the entropy
coefficient equals pilot/selection.json (record, task and every update); the learning rate of every update equals the
schedule; protocol_freeze_sha256 equals sha256(PLAN.md); the record code hashes equal precheck.json's; every
evaluation flies the rule reference's 80 episodes.

Primary (per head, n = 6 blocks, paired by block): P1 = J_T(E4) - J_T(LONG) (paired t, two-sided, dz; Holm with the
family size kept at 2 even if only one head has P1); P2 = share of LONG's gap to the paired rule reference closed (an
estimate with a percentile bootstrap interval, no test); P3 = the half-rule check at min(0.103, J_rule / 2), with the
paired J_rule / 2 reported alongside. Readings are computed here from the rules fixed in the plan, with T in every
string.

Failures (PLAN.md, reading rules): a block has a value at T iff its run evaluated T (by the deadline under the deadline
read-out), even if the run later ended with a failure; a block without one is a failed block. P1 is not computed for a
head with 3 or more such blocks. P3: 'holds' needs all 12 blocks with a value at T, every one below the threshold;
otherwise the reading of record is the more severe of the reading on the values at T and the reading with the failed
blocks' last evaluated J carried forward (withdrawn > restated > holds), and a remaining 'holds' becomes 'not
computable' (a head without any value at T) or 'qualified'; the text names n per head, the failed blocks and every
last evaluated J at or above the threshold.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner                                                                                     # noqa: E402
import e4lib as e4                                                                                # noqa: E402
import dq_b2                                                                                      # noqa: E402
import p4_train as p4                                                                             # noqa: E402

ARMS, BLOCKS, GRID, T_PLANNED = e4.ARMS, e4.BLOCKS, list(e4.GRID), e4.BUDGET
LONG_HALF_RULE = 0.103                     # LONG's planned threshold (half of min(0.206, 0.2104)), in the manuscript
NO_TIMELY_CUT = 0.012                      # LONG PLAN_LOG 2026-10-24T16:37:30Z: <= 0.012 = ended without timely reports
ENTROPY_WINDOW = 20                        # mean policy entropy over the last 20 PPO updates (descriptive only)
HOLM_FAMILY = 2                            # {P1_C-strong, P1_C-matched}, kept at 2 when only one head has P1
BOOT_SEED, BOOT_N = 20261025, 20_000
#: PLAN.md stopping rule: the deadline (the plan states the same instant)
DEADLINE_UTC = '2026-10-30T12:00:00+00:00'
DEADLINE = datetime.datetime.fromisoformat(DEADLINE_UTC)
OUTPUTS = {'record': 'e4_results.json', 'supplementary': 'e4_results_full_after_deadline.json',
           'provisional': 'e4_results_PROVISIONAL.json'}
FAILED = ('algorithmic_failure', 'namespace_end', 'code_failure')
#: top-level fields that carry time, not the run's trajectory (excluded when copies are compared)
TIMING_FIELDS = ('utc', 'wall_s', 'elapsed_wall_s', 'wall', 'sha256', 'cpu', 'walls', 'load_average', 'process_rusage',
                 'rollout_per_decision_s', 'update_per_decision_s')
RT = None


class Refusal(SystemExit):
    """The analysis refuses (exit status 1 with the reason)."""


def runtime():
    global RT
    if RT is None:
        RT = dq_b2.runtime(dq_b2.STUDY_B2)
    return RT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def plan_digest(root):
    """(sha256 of root/PLAN.md, True iff root/PLAN.sha256 names exactly that hash)."""
    plan, record = Path(root) / 'PLAN.md', Path(root) / 'PLAN.sha256'
    if not plan.exists() or not record.exists():
        return None, False
    digest = sha(plan)
    return digest, record.read_text().split()[0] == digest


# ------------------------------------------------------------------ reading and integrity -------------------------
def read_lines(path, partial_ok=False):
    """(lines, events) of a metrics.jsonl. A last line without its newline (being written) is dropped only if
    partial_ok (a run still running); any other unreadable line is a refusal."""
    raw = Path(path).read_bytes().decode()
    lines = raw.split('\n')
    tail = lines.pop()
    if tail and not partial_ok:
        raise Refusal(f'{path}: the last line has no newline')
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            raise Refusal(f'{path}: line {i + 1} is not JSON')
    return lines, events


def event_problems(lines, events, grid, expect_arm_end):
    """Raw-event integrity of one run (structure only; empty list = passed)."""
    p = []
    if len(set(lines)) != len(lines):
        p.append('duplicated lines')
    kinds = [e.get('kind') for e in events]
    if not events or kinds[0] != 'arm_start' or kinds.count('arm_start') != 1:
        p.append('not exactly one arm_start, first')
    if expect_arm_end:
        if kinds.count('arm_end') != 1 or kinds[-1] != 'arm_end':
            p.append('not exactly one arm_end, last')
    elif 'arm_end' in kinds:
        p.append('an arm_end where the record (or its absence) says the run did not end through train_arm')
    ups = [e['update_index'] for e in events if e.get('kind') == 'update']
    if ups != list(range(1, len(ups) + 1)):
        p.append('update indices not contiguous from 1')
    n_updates, next_episode, total = 0, 0, 0
    for e in events:
        k = e.get('kind')
        if k == 'update':
            n_updates += 1
            total += e['transitions']
            if e['cumulative']['policy_decisions'] != total or e['cumulative']['updates'] != n_updates:
                p.append(f'update {e["update_index"]}: cumulative counts inconsistent')
                break
        elif k == 'episode':
            if e['episode'] != next_episode or e['update_index'] != n_updates + 1:
                p.append(f'episode numbering broken at episode {e.get("episode")}')
                break
            next_episode += 1
    points = [q for e in events if e.get('kind') == 'evaluation' for q in e['points']]
    if points != list(grid[:len(points)]):
        p.append('evaluated points are not the grid, in order, each once')
    for i, e in enumerate(events):
        if e.get('kind') == 'evaluation':
            prev = events[i - 1] if i else {}
            if prev.get('kind') != 'checkpoint' or prev.get('points') != e['points'] or prev.get('final'):
                p.append(f'evaluation of {e["points"]} not preceded by its checkpoint')
                break
    if expect_arm_end and kinds and kinds[-1] == 'arm_end':
        status = events[-1].get('status')
        if status == 'completed':
            if points != list(grid):
                p.append('a completed run lacks grid points')
            if not any(e.get('kind') == 'checkpoint' and e.get('final') for e in events) or 'algorithmic_failure' in kinds:
                p.append('a completed run without its final checkpoint, or with a failure event')
        elif status == 'algorithmic_failure':
            if kinds.count('algorithmic_failure') != 1:
                p.append('an algorithmic failure without exactly one failure event')
        else:
            p.append(f'arm_end status {status!r}')
    return p


def schedule_problems(events, arm, entropy):
    """Every update ran at the scheduled learning rate (lr0 (1 - D / 300,000), D = decisions before the update) and
    logged the entropy coefficient `entropy`."""
    p, before = [], 0
    for e in events:
        if e.get('kind') != 'update':
            continue
        m = e['metrics']
        want = e4.scheduled_lr(e4.LR0[arm], e4.LR_HORIZON, before)
        if m.get('decisions_before_update') != before or m.get('learning_rate') != want:
            p.append(f'update {e["update_index"]}: learning rate not the schedule')
            break
        if m.get('entropy_coef') != entropy:
            p.append(f'update {e["update_index"]}: entropy_coef {m.get("entropy_coef")} is not {entropy}')
            break
        before = e['cumulative']['policy_decisions']
    return p


def main_run_problems(arm, block, events, rec, ctx):
    """B4 (i)-(iii) and the identity of one main run. rec is None for a run still running (then the launches log
    stands in for the record). ctx: entropy {arm: e}, digest, precheck, launches, long_events {(arm, block): events}."""
    p = []
    lng = ctx['long_events'][(arm, block)]
    ev0 = [e for e in events if e.get('kind') == 'evaluation' and e['points'] == [0]]
    lev0 = [e for e in lng if e.get('kind') == 'evaluation' and e['points'] == [0]]
    if not ev0 or not lev0 or ev0[0]['results'] != lev0[0]['results'] or \
            ev0[0]['policy_decisions'] != lev0[0]['policy_decisions']:
        p.append('(i) the point-0 evaluation is not LONG\'s')
    eps = [(e['episode'], e['cell_id'], e['seed']) for e in events if e.get('kind') == 'episode']
    leps = [(e['episode'], e['cell_id'], e['seed']) for e in lng if e.get('kind') == 'episode']
    if eps[:len(leps)] != leps[:len(eps)]:
        p.append('(ii) the training (cell_id, seed) sequence differs from LONG\'s on the overlapping episodes')
    start = events[0] if events else {}
    if start.get('identity', {}).get('condition', {}).get('obs_encoding_version') != e4.OBS_VERSION or \
            start.get('model', {}).get('config', {}).get('obs_dim') != e4.WIDE.dim:
        p.append('(iii) not obs-encoding/4-gnss+ev16/1 with 270 inputs')
    if start.get('study') != 'dq_b2' or start.get('block') != block or start.get('arm') != arm:
        p.append('arm_start study, block or arm')
    p += schedule_problems(events, arm, ctx['entropy'][arm])
    pre = ctx['precheck']
    if rec is not None:
        task = rec.get('task', {})
        want = {'kind': 'main', 'arm': arm, 'block': block, 'entropy': ctx['entropy'][arm], 'lr0': e4.LR0[arm],
                'info': True, 'schedule': True, 'study': 'dq_b2', 'budget': T_PLANNED, 'grid': GRID,
                'freeze': ctx['digest'], 'name': f'{arm}__b{block}'}
        if any(task.get(k) != v for k, v in want.items()):
            p.append('record task differs from the plan')
        cfg = rec.get('config', {})
        if cfg.get('ppo', {}).get('entropy_coef') != ctx['entropy'][arm]:
            p.append('(iii) config entropy differs from pilot/selection.json')
        if cfg.get('protocol_freeze_sha256') != ctx['digest']:
            p.append('(iii) config protocol_freeze_sha256 is not sha256(PLAN.md)')
        end = [e for e in events if e.get('kind') == 'arm_end']
        if end and end[0].get('protocol_freeze_sha256') != ctx['digest']:
            p.append('(iii) arm_end protocol_freeze_sha256 is not sha256(PLAN.md)')
        if end and end[0].get('status') != rec.get('status'):
            p.append('record status differs from arm_end')
        code = rec.get('code', {})
        if code.get('runner_sha256') != pre['runner_sha256'] or code.get('e4lib_sha256') != pre['e4lib_sha256'] or \
                code.get('idea_dq_tree') != e4.EXPECTED_TREE:
            p.append('(iii) record code hashes differ from the precheck\'s')
    if not ctx['launches']:
        p.append('no main launch is logged')
    for la in ctx['launches']:
        if la.get('plan_sha256') != ctx['digest'] or la.get('entropy_coef', {}).get(arm) != ctx['entropy'][arm] or \
                la.get('runner_sha256') != pre['runner_sha256'] or la.get('e4lib_sha256') != pre['e4lib_sha256']:
            p.append('a logged launch differs in plan, entropy or code')
            break
    return p


def canon(e):
    """An event without its timing fields. Checkpoint FILE hashes are dropped too (top level and inside arm_end's
    checkpoint list): torch.save output is not byte-reproducible, while every trajectory field is (the 2026-10-25
    precheck re-run equals the first precheck in every other field)."""
    out = {k: v for k, v in e.items() if k not in TIMING_FIELDS}
    if isinstance(out.get('checkpoints'), list):
        out['checkpoints'] = [{k: v for k, v in c.items() if k != 'sha256'} if isinstance(c, dict) else c
                              for c in out['checkpoints']]
    return out


def prefix_comparison(copy_events, current_events):
    """Compare a kept (interrupted or failed) copy with the current run over their common prefix of events."""
    n = min(len(copy_events), len(current_events))
    first = next((i for i in range(n) if canon(copy_events[i]) != canon(current_events[i])), None)
    return {'compared_events': n, 'copy_events': len(copy_events), 'current_events': len(current_events),
            'identical': first is None, 'first_mismatch_event': first}


def evaluation_times(arm_dir, events):
    """{point: POSIX time at which its evaluation was logged} = mtime of the checkpoint saved right before it (the
    trainer saves, then evaluates) + the evaluation's wall_s."""
    out = {}
    for i, e in enumerate(events):
        if e.get('kind') == 'evaluation':
            ck = Path(arm_dir) / events[i - 1]['path']
            if not ck.exists():
                raise Refusal(f'{ck} is missing: the evaluation time cannot be established')
            for q in e['points']:
                out[q] = ck.stat().st_mtime + e['wall_s']
    return out


def curve_of(events, pairs):
    """{point: row} from the evaluation events; every stored weighted J is re-derived from its episodes and every
    evaluation must fly the rule reference's 80 episodes."""
    curve = {}
    for e in events:
        if e.get('kind') != 'evaluation':
            continue
        res = e['results']['sample']
        eps = res['episodes']
        _, weighted = p4.aggregate_evaluation(runtime(), eps)
        if weighted != res['weighted_J']:
            raise Refusal('a stored weighted J differs from its episodes')
        if [(x['cell_id'], x['seed']) for x in eps] != pairs:
            raise Refusal('an evaluation is not paired with the rule reference')
        row = {'J': weighted, 'policy_decisions': e['policy_decisions'],
               'timely': float(np.mean([x['tp_timely'] for x in eps])),
               'false': float(np.mean([x['fp_spatial'] for x in eps])),
               'fixes': float(np.mean([x['fixes_attempted'] for x in eps])),
               'decisions_per_episode': float(np.mean([x['decisions'] for x in eps]))}
        for q in e['points']:
            curve[q] = row
    return curve


def run_summary(events, pairs):
    ups = [e['metrics'] for e in events if e.get('kind') == 'update']
    ev = {k: 0 for k in e4.EV_STATS}
    ev['decisions'] = 0
    for e in events:
        if e.get('kind') == 'episode':
            for s in e4.EV_STATS:
                ev[s] += e['policy_stats'].get(s, 0)
            ev['decisions'] += e['policy_stats'].get('decisions', 0)
    ent = [m['entropy'] for m in ups]
    return {'curve': curve_of(events, pairs), 'updates': len(ups), 'ev': ev,
            'last_entropy': ent[-1] if ent else None,
            'entropy_last20_mean': float(np.mean(ent[-ENTROPY_WINDOW:])) if ent else None,
            'last_lr': ups[-1].get('learning_rate') if ups else None}


@functools.lru_cache(maxsize=None)
def long_run(arm, block):
    """(events, summary) of the stored LONG run (already-analysed data; integrity-checked as a completed run)."""
    lines, events = read_lines(e4.LONG_DIR / 'runs' / f'{arm}__b{block}' / 'run' / arm / 'metrics.jsonl')
    bad = event_problems(lines, events, GRID, expect_arm_end=True)
    if bad or events[-1]['status'] != 'completed':
        raise Refusal(f'stored LONG {arm} b{block}: {bad}')
    return events, run_summary(events, rule_pairs())


@functools.lru_cache(maxsize=None)
def rule_reference():
    ref = json.loads((HERE / 'rule_reference.json').read_text())
    _, weighted = p4.aggregate_evaluation(runtime(), ref['episodes'])
    if weighted != ref['weighted_J'] or not ref['checks']['passed']:
        raise Refusal('rule_reference.json does not verify')
    return ref


def rule_pairs():
    return [(r['cell_id'], r['seed']) for r in rule_reference()['episodes']]


# ------------------------------------------------------------------ the state of every run -------------------------
def run_states(root):
    """{(arm, block): state}. state: class ('completed', 'algorithmic_failure', 'namespace_end', 'code_failure',
    'running'), end time, record, current copy (lines, events, arm_dir) and kept copies. Refuses while a first
    non-algorithmic failure awaits its one restart."""
    root = Path(root)
    ledger = runner.restarts(root)
    out = {}
    for arm in ARMS:
        for b in BLOCKS:
            name = f'{arm}__b{b}'
            rec_path = root / 'records' / f'{name}.json'
            rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
            cls = runner.classify(rec) if rec else 'running'
            if cls == 'infrastructure_failure':
                if runner.failure_restarts(root, name) == 0:
                    raise Refusal(f'{name}: a first non-algorithmic failure awaits its one restart (relaunch)')
                cls = 'code_failure'
            arm_dir = root / 'runs' / name / 'run' / arm
            metrics = arm_dir / 'metrics.jsonl'
            st = {'name': name, 'class': cls, 'record': rec, 'arm_dir': arm_dir, 'end': None, 'copies': []}
            if metrics.exists():
                st['lines'], st['events'] = read_lines(metrics, partial_ok=rec is None)
            else:
                st['lines'], st['events'] = None, None
            if rec is not None:
                end = [e for e in (st['events'] or []) if e.get('kind') == 'arm_end']
                st['end'] = (datetime.datetime.fromisoformat(end[0]['utc']).timestamp() if end
                             else rec_path.stat().st_mtime)
            for entry in ledger:
                if entry['name'] == name and entry.get('moved_to'):
                    cdir = root / entry['moved_to'] / 'run' / arm
                    if (cdir / 'metrics.jsonl').exists():
                        cl, ce = read_lines(cdir / 'metrics.jsonl', partial_ok=True)
                        st['copies'].append({'entry': entry, 'arm_dir': cdir, 'lines': cl, 'events': ce})
            out[(arm, b)] = st
    return out


def check_runs(states, ctx):
    """Every check of the docstring on every current copy; refuses on any failure. Returns the restart comparisons."""
    problems, restarts = {}, {}
    for (arm, b), st in states.items():
        if st['events'] is None:
            if st['record'] is not None and st['class'] not in ('namespace_end', 'code_failure'):
                problems[st['name']] = ['a record without metrics']
            continue
        expect_end = st['class'] in ('completed', 'algorithmic_failure')
        p = event_problems(st['lines'], st['events'], GRID, expect_end)
        p += main_run_problems(arm, b, st['events'], st['record'], ctx)
        if p:
            problems[st['name']] = p
        restarts[st['name']] = [dict(prefix_comparison(c['events'], st['events']), kind=c['entry']['kind'],
                                     moved_to=c['entry']['moved_to']) for c in st['copies']]
    if problems:
        raise Refusal(f'integrity or identity checks failed: {problems}')
    return restarts


def readout_point(states, kind, deadline):
    """(T, per-run detail). 'full': 300,000. 'deadline': the largest grid point every run running or completed AT the
    deadline had evaluated BY the deadline (current copy, or a kept copy whose common prefix with the current copy is
    identical). 'provisional': the same over the runs running or completed now."""
    if kind == 'full':
        return T_PLANNED, {}
    cut = deadline.timestamp() if kind == 'deadline' else float('inf')
    detail, reached = {}, []
    for key, st in states.items():
        ended = st['end'] is not None and st['end'] <= cut
        cls = st['class'] if ended else 'running'
        best = -1
        copies = [(st['arm_dir'], st['events'])] if st['events'] else []
        copies += [(c['arm_dir'], c['events']) for c in st['copies']
                   if st['events'] and prefix_comparison(c['events'], st['events'])['identical']]
        for arm_dir, events in copies:
            times = evaluation_times(arm_dir, events) if kind == 'deadline' else \
                {q: 0. for e in events if e.get('kind') == 'evaluation' for q in e['points']}
            best = max([best] + [q for q, t in times.items() if t <= cut])
        detail[st['name']] = {'class_at_cut': cls, 'reached_by_cut': best if best >= 0 else None,
                              'enters_T': cls in ('running', 'completed')}
        if cls in ('running', 'completed'):
            if best < 0:
                raise Refusal(f'{st["name"]}: running or completed at the cut but no evaluated point by then')
            reached.append(best)
    if not reached:
        raise Refusal('no run was running or completed at the cut: no read-out point exists')
    return min(reached), detail


def merged_curve(st, pairs):
    """The run's curve; a grid point missing in the current copy is taken from a kept copy with an identical common
    prefix (the same deterministic trajectory), marked as such."""
    curve = dict(run_summary(st['events'], pairs)['curve']) if st['events'] else {}
    sources = {q: 'current' for q in curve}
    for c in st['copies']:
        if st['events'] and prefix_comparison(c['events'], st['events'])['identical']:
            for q, row in curve_of(c['events'], pairs).items():
                if q not in curve:
                    curve[q], sources[q] = row, c['entry']['moved_to']
    return curve, sources


# ------------------------------------------------------------------ statistics --------------------------------------
def t_sum(x):
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return {'n': int(n), 'mean': float(x.mean()) if n else None, 'values': x.tolist()}
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    p = float(2 * stats.t.sf(abs(m / se), n - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    return {'n': int(n), 'mean': float(m), 'sd': float(sd), 'lo': float(m - q * se), 'hi': float(m + q * se), 'p': p,
            'dz': float(m / sd) if sd > 0 else None, 'share_negative': float(np.mean(x < 0)), 'values': x.tolist()}


def holm(ps, alpha=0.05, m=None):
    """Holm step-down; m = family size (default len(ps)). With m = 2 and one p-value the test is at alpha / 2."""
    m = len(ps) if m is None else m
    if len(ps) > m:
        raise ValueError('more p-values than the family size')
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej, still = [False] * len(ps), True
    for rank, i in enumerate(order):
        still = still and ps[i] <= alpha / (m - rank)
        rej[i] = still
    return rej


def gap_closed(e4_j, long_j, rule_j):
    """ESTIMATE (no test): ratio of block means (J_E4 - J_LONG) / (J_rule - J_LONG), percentile bootstrap over blocks."""
    a, b = np.asarray(e4_j, float), np.asarray(long_j, float)
    den = rule_j - b
    if den.mean() <= 0:
        return {'estimate': None, 'note': 'LONG at or above the rule: undefined'}
    est = float((a - b).mean() / den.mean())
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, a.size, size=(BOOT_N, a.size))
    num, dd = (a - b)[idx].mean(1), den[idx].mean(1)
    boots = num[dd > 0] / dd[dd > 0]
    return {'estimate': est, 'lo': float(np.percentile(boots, 2.5)), 'hi': float(np.percentile(boots, 97.5)),
            'per_block': ((a - b) / den).tolist(), 'bootstrap': {'seed': BOOT_SEED, 'resamples': BOOT_N},
            'note': 'an estimate with a bootstrap interval; no test, no reading rule'}


def auc(curve, T):
    pts = [p for p in GRID if p <= T]
    J = np.array([curve[p]['J'] for p in pts])
    return float(np.sum((J[1:] + J[:-1]) / 2 * np.diff(pts)) / T)


def reading_threshold(J_rule):
    """B2: the holds / restated / withdrawn readings use the smaller of LONG's 0.103 and the paired J_rule / 2."""
    return min(LONG_HALF_RULE, J_rule / 2)


def p1_reading(t, rejected, T):
    if rejected and t['lo'] > 0:
        return f'above zero: the best-effort configuration ended higher than LONG at {T:,} decisions'
    if rejected and t['hi'] < 0:
        return f'below zero: the best-effort configuration ended lower than LONG at {T:,} decisions'
    return f'not resolved at n = {t["n"]} (not evidence of equality)'


SEVERITY = {'holds': 0, 'restated': 1, 'withdrawn': 2}


def three_way(vals, theta):
    """The plan's three readings over vals {role: {block: J}}, using only the heads with at least one value:
    ('withdrawn', [heads whose mean >= theta]) | ('restated', {role: [blocks >= theta]}) | ('holds', None)."""
    vals = {r: v for r, v in vals.items() if v}
    hi = [r for r, v in vals.items() if float(np.mean(list(v.values()))) >= theta]
    if hi:
        return 'withdrawn', hi
    at = {r: [b for b in sorted(v) if v[b] >= theta] for r, v in vals.items()}
    at = {r: bl for r, bl in at.items() if bl}
    return ('restated', at) if at else ('holds', None)


def systems_reading(at_T, without, theta, T):
    """P3 reading (PLAN.md, reading rules and 'P3 when blocks lack a value at T'). Returns (category, text).

    at_T {role: {block: J_T}}: the blocks with a value at T. without {role: {block: {'class', 'last_point',
    'last_J'}}}: the blocks without one (each a failed run; last_point / last_J of its last evaluation, or None).
    - Every block has a value: the three readings (holds / restated / withdrawn) on the values at T.
    - Otherwise 'holds' is never given. The three readings are applied twice: (1) to the values at T, over the heads
      that have one; (2) with every failed block's last evaluated J carried forward (LOCF). The reading of record is
      the more severe of the two (withdrawn > restated > holds). If both are 'holds' it is 'not computable' when a
      head has no value at T, and 'qualified' otherwise. The text always names n per head, every failed block (class,
      last evaluated point, J) and every last evaluated J at or above theta."""
    roles = list(at_T)
    if not any(without.get(r) for r in roles):
        if not all(at_T[r] for r in roles):
            return 'not computable', f'not computable: no head has a value at {T:,} decisions'
        cat, d = three_way(at_T, theta)
        if cat == 'withdrawn':
            return cat, (f'withdrawn at {T:,} decisions for best-effort classical PPO: a head\'s mean J reached half '
                         f'the rule ({", ".join(d)}; threshold {theta:.5f})')
        if cat == 'restated':
            return cat, ('restated: no learned controller approached the rule on average, but best-effort classical '
                         f'PPO reached half the rule (threshold {theta:.5f}) at {T:,} decisions in '
                         + '; '.join(f'{r} block(s) {v}' for r, v in d.items()))
        return cat, (f'holds for best-effort, evidence-count-augmented, re-tuned classical PPO at {T:,} decisions '
                     f'(every block of both heads below half the rule, threshold {theta:.5f})')
    locf = {r: {**at_T[r], **{b: w['last_J'] for b, w in without.get(r, {}).items() if w['last_J'] is not None}}
            for r in roles}
    c1, d1 = three_way(at_T, theta)
    c2, d2 = three_way(locf, theta)
    no_head = [r for r in roles if not at_T[r]]
    locf_hi = {r: [b for b in sorted(without.get(r, {})) if without[r][b]['last_J'] is not None
                   and without[r][b]['last_J'] >= theta] for r in roles}
    locf_hi = {r: v for r, v in locf_hi.items() if v}
    cat = max((c1, c2), key=SEVERITY.get)
    if cat == 'withdrawn':
        heads = sorted(set((d1 if c1 == 'withdrawn' else []) + (d2 if c2 == 'withdrawn' else [])), key=roles.index)
        basis = ' and '.join(x for x, c in (('on the values at T', c1), ('with the failed blocks\' last evaluated J '
                                                                          'carried forward', c2)) if c == 'withdrawn')
        head = (f'withdrawn at {T:,} decisions for best-effort classical PPO: a head\'s mean J reached half the rule '
                f'({", ".join(heads)}; threshold {theta:.5f}; {basis})')
    elif cat == 'restated':
        parts = [f'at {T:,} decisions in {r} block(s) {v}' for r, v in (d1 or {}).items()]
        parts += [f'in the last evaluation before a failure in {r} block(s) '
                  + ', '.join(f'{b} ({without[r][b]["last_J"]:.5f} at {without[r][b]["last_point"]:,} decisions)'
                              for b in v) for r, v in locf_hi.items()]
        head = ('restated: no learned controller approached the rule on average (on the values at T, and with the '
                'failed blocks\' last evaluated J carried forward), but best-effort classical PPO reached half the '
                f'rule (threshold {theta:.5f}) ' + '; '.join(parts))
    elif no_head:
        cat = 'not computable'
        head = (f'not computable at {T:,} decisions: {", ".join(no_head)} has no block with a value at T, so the '
                'systems conclusion is not stated as holding')
    else:
        cat = 'qualified'
        head = (f'qualified (not "holds"): at {T:,} decisions every block with a value at T is below half the rule '
                f'(threshold {theta:.5f}), and so is every failed block\'s last evaluated J, but not every block has a '
                'value at T')

    def fail(r, b):
        w = without[r][b]
        last = (f'last evaluated {w["last_point"]:,} decisions, J {w["last_J"]:.5f}' if w['last_J'] is not None
                else 'no evaluated point')
        return f'block {b} ({w["class"]}; {last})'
    def head_n(r):
        w = without.get(r, {})
        return (f'{r} n = {len(at_T[r])} of {len(at_T[r]) + len(w)} with a value at T'
                + (', without one: ' + ', '.join(fail(r, b) for b in sorted(w)) if w else ''))
    qual = '; qualified by failures: ' + '; '.join(head_n(r) for r in roles)
    qual += ('; last evaluated J at or above the threshold: '
             + '; '.join(f'{r} block(s) {v}' for r, v in locf_hi.items())) if locf_hi else \
        '; no failed block\'s last evaluated J is at or above the threshold'
    if no_head and cat != 'not computable':
        qual += '; not computable at T for ' + ', '.join(no_head) + ' (no block has a value at T)'
    with_value = [r for r in roles if at_T[r]]
    on_T = (f'{c1}' + (f' (over {", ".join(with_value)} only)' if no_head else '')) if with_value else \
        'none (no block has a value at T)'
    qual += (f'; reading on the values at T: {on_T}; sensitivity reading with LOCF: '
             + (c2 if any(locf.values()) else 'none (no evaluated point)'))
    return cat, head + qual


def analyse(new, old, classes, T, ref):
    """The planned statistics. new / old {(arm, block): {'curve', 'entropy_last20_mean', 'last_entropy', 'last_lr',
    'ev'}} (new: E4 runs that exist); classes {(arm, block): class at the read-out}; T the read-out point."""
    J_rule = ref['weighted_J']
    half, theta = J_rule / 2, reading_threshold(J_rule)
    out = {'readout_checkpoint': T,
           'thresholds': {'reading_threshold_min_0.103_and_half_rule': theta, 'paired_half_rule': half,
                          'LONG_half_rule_threshold': LONG_HALF_RULE, 'J_rule': J_rule},
           'heads': {}}
    ps, at_T, without = [], {}, {}
    for arm in ARMS:
        role = e4.ROLES[arm]
        # a block has a value at T iff its run evaluated T (by the deadline under the deadline read-out), even if the
        # run later ended with a failure; a block without one is a failed block (PLAN.md, reading rules, Failures)
        ok = [b for b in BLOCKS if (arm, b) in new and T in new[(arm, b)]['curve']]
        failed = [b for b in BLOCKS if classes.get((arm, b)) in FAILED]
        no_value = [b for b in BLOCKS if b not in ok]
        if any(b not in failed for b in no_value):
            raise Refusal(f'{role}: a running or completed block without a value at {T:,} decisions '
                          f'{[b for b in no_value if b not in failed]}')
        jn = [new[(arm, b)]['curve'][T]['J'] for b in ok]
        jo = [old[(arm, b)]['curve'][T]['J'] for b in ok]

        def last(b):
            c = new[(arm, b)]['curve'] if (arm, b) in new else {}
            return (max(c), c[max(c)]['J']) if c else (None, None)
        without[role] = {b: dict(zip(('last_point', 'last_J'), last(b)), **{'class': classes[(arm, b)]})
                         for b in no_value}
        at_T[role] = dict(zip(ok, jn))
        h = {'role': role, 'blocks_with_value_at_T': ok, 'failed_blocks': {b: classes[(arm, b)] for b in failed},
             'failed_blocks_with_value_at_T': [b for b in failed if b in ok],
             'blocks_without_value_at_T': without[role], 'n': len(ok)}
        if len(ok) >= 2 and len(no_value) < 3:
            h['P1_E4_minus_LONG_at_T'] = t_sum(np.subtract(jn, jo))
            ps.append((arm, h['P1_E4_minus_LONG_at_T']['p']))
        else:
            h['P1_E4_minus_LONG_at_T'] = None
        if no_value:                                           # sensitivity: last evaluated J carried forward
            locf = [new[(arm, b)]['curve'][max(new[(arm, b)]['curve'])]['J'] if b in no_value else
                    new[(arm, b)]['curve'][T]['J'] for b in BLOCKS
                    if (arm, b) in new and (b in ok or (b in no_value and new[(arm, b)]['curve']))]
            blocks = [b for b in BLOCKS if (arm, b) in new and (b in ok or (b in no_value and new[(arm, b)]['curve']))]
            h['P1_sensitivity_LOCF'] = dict(t_sum(np.subtract(locf, [old[(arm, b)]['curve'][T]['J'] for b in blocks])),
                                            blocks=blocks)
        h['P2_estimate_share_of_LONG_gap_to_rule_closed'] = gap_closed(jn, jo, J_rule) if ok else None
        h['P3'] = {'mean_J_at_T': float(np.mean(jn)) if jn else None, 'max_block_J_at_T': float(max(jn)) if jn else None,
                   'blocks_at_or_above_reading_threshold': [b for b, j in zip(ok, jn) if j >= theta],
                   'blocks_at_or_above_paired_half_rule': [b for b, j in zip(ok, jn) if j >= half],
                   'blocks_at_or_above_LONG_0.103': [b for b, j in zip(ok, jn) if j >= LONG_HALF_RULE],
                   'failed_blocks_last_evaluated_J_at_or_above_reading_threshold':
                       [b for b, w in without[role].items() if w['last_J'] is not None and w['last_J'] >= theta],
                   'failed_blocks_last_evaluated_J_at_or_above_paired_half_rule':
                       [b for b, w in without[role].items() if w['last_J'] is not None and w['last_J'] >= half]}
        h['E4_J_at_T'] = t_sum(jn)
        h['LONG_J_at_T'] = t_sum(jo)
        h['E4_minus_rule_at_T'] = t_sum(np.subtract(jn, J_rule)) if len(jn) >= 2 else None
        h['E4_minus_reading_threshold_at_T'] = t_sum(np.subtract(jn, theta)) if len(jn) >= 2 else None
        h['E4_minus_paired_half_rule_at_T'] = t_sum(np.subtract(jn, half)) if len(jn) >= 2 else None
        h['E4_share_of_rule_J'] = float(np.mean(jn) / J_rule) if jn else None
        # descriptive (no test; not in the Holm family)
        pts = [p for p in GRID if p <= T]
        h['curve_mean'] = {p: {'E4': float(np.mean([new[(arm, b)]['curve'][p]['J'] for b in ok])),
                               'LONG': float(np.mean([old[(arm, b)]['curve'][p]['J'] for b in ok]))}
                           for p in pts} if ok else {}
        h['E4_minus_LONG_by_checkpoint'] = {p: t_sum([new[(arm, b)]['curve'][p]['J'] - old[(arm, b)]['curve'][p]['J']
                                                      for b in ok]) for p in pts} if ok else {}
        if T > 0 and ok:
            h['AUC_E4_minus_LONG'] = t_sum([auc(new[(arm, b)]['curve'], T) - auc(old[(arm, b)]['curve'], T) for b in ok])
            last3 = pts[-3:]
            if len(last3) == 3:                    # least-squares slope per 30,000 decisions, as long_analysis.py
                sl = {src: [np.polyfit(last3, [runs[(arm, b)]['curve'][p]['J'] for p in last3], 1)[0] * 30000
                            for b in ok] for src, runs in (('E4', new), ('LONG', old))}
                h['slope_last3_per_30k'] = {'checkpoints': last3, 'E4': t_sum(sl['E4']), 'LONG': t_sum(sl['LONG']),
                                            'E4_minus_LONG': t_sum(np.subtract(sl['E4'], sl['LONG']))}
        if 30000 <= T and ok:
            h['J30k_E4_minus_LONG'] = t_sum([new[(arm, b)]['curve'][30000]['J'] - old[(arm, b)]['curve'][30000]['J']
                                             for b in ok])
        h['components_at_T'] = {src: {c: float(np.mean([runs[(arm, b)]['curve'][T][c] for b in ok]))
                                      for c in ('timely', 'false', 'fixes', 'decisions_per_episode')}
                                for src, runs in (('E4', new), ('LONG', old))} if ok else None
        h['rule_components'] = {c: float(np.mean([r[k] for r in ref['episodes']])) for c, k in
                                (('timely', 'tp_timely'), ('false', 'fp_spatial'), ('fixes', 'fixes_attempted'),
                                 ('decisions_per_episode', 'decisions'))}
        h['collapse_descriptive'] = {
            src: {'no_timely_at_T_le_0.012': [b for b in ok if runs[(arm, b)]['curve'][T]['timely'] <= NO_TIMELY_CUT],
                  'entropy_last20_mean': {b: runs[(arm, b)]['entropy_last20_mean'] for b in ok},
                  'last_update_entropy': {b: runs[(arm, b)]['last_entropy'] for b in ok}}
            for src, runs in (('E4', new), ('LONG', old))}
        # peak versus final: intermediate checkpoints at or above a threshold, reported descriptively; the value of
        # record is J_T (no checkpoint is selected on the evaluation suite)
        inter = [(b, p, new[(arm, b)]['curve'][p]['J']) for b in BLOCKS if (arm, b) in new
                 for p in GRID if p < T and p in new[(arm, b)]['curve']]
        h['intermediate_checkpoints_descriptive'] = {
            'at_or_above_reading_threshold': [x for x in inter if x[2] >= theta],
            'at_or_above_paired_half_rule': [x for x in inter if x[2] >= half],
            'highest_block_J_at_any_checkpoint': max(inter + [(b, T, j) for b, j in zip(ok, jn)], key=lambda x: x[2])
            if inter or ok else None}
        h['evidence_stats'] = {b: new[(arm, b)]['ev'] for b in ok}
        h['last_update_lr'] = {b: new[(arm, b)]['last_lr'] for b in ok}
        out['heads'][role] = h
    rej = holm([p for _, p in ps], m=HOLM_FAMILY) if ps else []
    out['holm'] = {'family_size': HOLM_FAMILY, 'heads_with_P1': [e4.ROLES[a] for a, _ in ps]}
    for (arm, _), r in zip(ps, rej):
        h = out['heads'][e4.ROLES[arm]]
        h['P1_E4_minus_LONG_at_T']['holm_rejected'] = r
        h['P1_reading'] = p1_reading(h['P1_E4_minus_LONG_at_T'], r, T)
    for h in out['heads'].values():
        if 'P1_reading' not in h:
            k = len(h['blocks_without_value_at_T'])
            h['P1_reading'] = (f'not computed: failed in {k} of 6 blocks (no value at {T:,} decisions)' if k >= 3 else
                               f'not computed: {h["n"]} block(s) with a value at {T:,} decisions')
    out['systems_reading_category'], out['systems_reading'] = systems_reading(at_T, without, theta, T)
    (out['systems_reading_at_paired_half_rule_descriptive_category'],
     out['systems_reading_at_paired_half_rule_descriptive']) = systems_reading(at_T, without, half, T)
    out['not_shown_below_rule'] = [r for r, h in out['heads'].items()
                                   if h['E4_minus_rule_at_T'] and h['E4_minus_rule_at_T'].get('hi', -1) >= 0]
    return out


# ------------------------------------------------------------------ the read-out ------------------------------------
def write_output(root, name, out, force, mode):
    root = Path(root)
    path = root / name
    log = {'utc': utc_now().isoformat(timespec='seconds'), 'mode': mode, 'file': name, 'argv': sys.argv[1:],
           'analysis_sha256': sha(__file__)}
    if path.exists():
        if not force:
            raise Refusal(f'{name} exists: never overwritten without --force')
        kept = root / f'{path.stem}.superseded_{runner.utc()}.json'
        shutil.move(str(path), str(kept))
        log.update(forced=True, replaced_sha256=sha(kept), replaced_kept_as=kept.name)
    path.write_text(json.dumps(out, indent=1, default=float))
    log['sha256'] = sha(path)
    with open(root / 'e4_analysis_log.jsonl', 'a') as f:
        f.write(json.dumps(log) + '\n')
    return path


def main(root=HERE, mode='record', force=False, now=None, deadline=DEADLINE):
    root = Path(root).resolve()
    now = now or utc_now()
    if mode == 'provisional' and root == HERE:
        raise Refusal('--provisional is for a copy (--root); it is refused on this folder (no peeking)')
    ref = rule_reference()
    digest, ok = plan_digest(root)
    if not ok:
        raise Refusal('PLAN.md and a matching PLAN.sha256 are required')
    named = runner.plan_names_frozen_hashes(root / 'PLAN.md')
    if not all(named.values()):
        raise Refusal(f'PLAN.md does not name the SHA-256 of {[f for f, v in named.items() if not v]}')
    sel_path = root / 'pilot' / 'selection.json'
    if not sel_path.exists():
        raise Refusal('no pilot/selection.json')
    sel = json.loads(sel_path.read_text())
    pre = json.loads((root / 'precheck.json').read_text())
    if not pre.get('passed'):
        raise Refusal('precheck.json did not pass')
    launches = [json.loads(x) for x in (root / 'launches.jsonl').read_text().splitlines() if x.strip()] \
        if (root / 'launches.jsonl').exists() else []
    pairs = rule_pairs()
    ctx = {'entropy': {arm: float(sel['heads'][arm]['entropy_coef']) for arm in ARMS}, 'digest': digest,
           'precheck': pre, 'launches': launches,
           'long_events': {(arm, b): long_run(arm, b)[0] for arm in ARMS for b in BLOCKS}}
    states = run_states(root)
    restarts = check_runs(states, ctx)
    all_ended = all(st['record'] is not None for st in states.values())
    ended_by_deadline = all_ended and all(st['end'] <= deadline.timestamp() for st in states.values())
    if mode == 'record':
        if ended_by_deadline:
            kind = 'full'
        elif now >= deadline:
            kind = 'deadline'
        else:
            raise Refusal(f'incomplete: {sum(st["record"] is not None for st in states.values())} of 12 runs ended; '
                          f'the read-out waits until every run has ended or {DEADLINE_UTC} (PLAN.md)')
    elif mode == 'supplementary':
        prior = root / OUTPUTS['record']
        if not prior.exists() or json.loads(prior.read_text()).get('readout') != 'deadline':
            raise Refusal('--supplementary-full follows a deadline read-out of record only')
        if not all_ended:
            raise Refusal('--supplementary-full waits until every run has ended')
        kind = 'full'
    else:
        kind = 'full' if all_ended else 'provisional'
    T, cut_detail = readout_point(states, kind, deadline)
    classes = {}
    for key, st in states.items():
        if kind == 'deadline':
            classes[key] = cut_detail[st['name']]['class_at_cut']
        else:
            classes[key] = st['class']
    new, sources = {}, {}
    for key, st in states.items():
        if st['events'] is None:
            continue
        s = run_summary(st['events'], pairs)
        s['curve'], sources[st['name']] = merged_curve(st, pairs)
        # every run's curve stops at T (PLAN.md, Failures). Under the deadline read-out a run running or completed at
        # the deadline evaluated every point <= T by then, and a run that failed before it evaluated all its points
        # before it; under the full read-out T = 300,000 and nothing is cut
        s['curve'] = {q: v for q, v in s['curve'].items() if q <= T}
        new[key] = s
    old = {(arm, b): long_run(arm, b)[1] for arm in ARMS for b in BLOCKS}
    res = analyse(new, old, classes, T, ref)
    of_record = mode == 'record'
    out = {'readout': kind if mode != 'supplementary' else 'supplementary_full_after_deadline',
           'of_record': of_record, 'provisional': mode == 'provisional', 'run_utc': now.isoformat(timespec='seconds'),
           'deadline_utc': DEADLINE_UTC, 'plan_sha256': digest, 'analysis_sha256': sha(__file__),
           'frozen_files': {f: sha(HERE / f) for f in runner.FROZEN_FILES}, 'selection_sha256': sha(sel_path),
           'rule_reference': {'weighted_J': ref['weighted_J'], 'sha256': sha(HERE / 'rule_reference.json')},
           'entropy_coef': ctx['entropy'],
           'runs': {st['name']: {'class': st['class'], 'class_at_readout': classes[key],
                                 'end_utc': (datetime.datetime.fromtimestamp(st['end'], datetime.timezone.utc)
                                             .isoformat(timespec='seconds') if st['end'] else None),
                                 'evaluated_points': sorted(new[key]['curve']) if key in new else [],
                                 'point_sources_not_current': {q: s for q, s in sources.get(st['name'], {}).items()
                                                               if s != 'current'}}
                    for key, st in states.items()},
           'deadline_cut': cut_detail, 'restart_prefix_comparisons': restarts,
           'checks': 'passed (integrity, LONG point-0 and schedule pairing, obs_version, entropy, learning-rate '
                     'schedule, protocol freeze, code hashes, rule-reference pairing)', **res}
    path = write_output(root, OUTPUTS[mode], out, force, mode)
    print(json.dumps({'file': path.name, 'readout': out['readout'], 'of_record': of_record, 'T': T,
                      'systems_reading_category': res['systems_reading_category'],
                      'systems_reading': res['systems_reading'],
                      'P1': {r: (h['P1_E4_minus_LONG_at_T'] or {}).get('mean') for r, h in res['heads'].items()},
                      'P1_reading': {r: h['P1_reading'] for r, h in res['heads'].items()},
                      'P2_estimate': {r: (h['P2_estimate_share_of_LONG_gap_to_rule_closed'] or {}).get('estimate')
                                      for r, h in res['heads'].items()}}, indent=1, default=float))
    return out


def selftest():
    """No E4 run is read. LONG's stored 300k means are recomputed and compared with long_results.json (every LONG run
    passes the integrity checker as a completed run); the rule reference verifies; the statistics helpers work."""
    stored = json.loads((e4.LONG_DIR / 'long_results.json').read_text())['LONG']
    ref = rule_reference()
    for arm in ARMS:
        m = float(np.mean([long_run(arm, b)[1]['curve'][300000]['J'] for b in BLOCKS]))
        want = stored[e4.ROLES[arm]]['J_at_T']['mean']
        print(arm, 'LONG J300k mean recomputed', m, 'stored', want)
        assert abs(m - want) < 1e-12, (m, want)
    t = t_sum([0.01, 0.02, -0.005, 0.03, 0.0, 0.015])
    assert abs(t['mean'] - 0.0116666666) < 1e-9 and t['lo'] < t['mean'] < t['hi']
    assert holm([0.01, 0.04]) == [True, True] and holm([0.03, 0.04]) == [False, False]
    assert holm([0.001, 0.2]) == [True, False] and holm([0.03], m=2) == [False] and holm([0.02], m=2) == [True]
    assert abs(gap_closed([0.1] * 6, [0.0] * 6, 0.2)['estimate'] - 0.5) < 1e-12
    assert reading_threshold(ref['weighted_J']) == LONG_HALF_RULE < ref['weighted_J'] / 2
    print('selftest passed; rule reference weighted J', ref['weighted_J'], 'reading threshold',
          reading_threshold(ref['weighted_J']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--provisional', action='store_true')
    ap.add_argument('--supplementary-full', action='store_true')
    ap.add_argument('--deadline-readout', action='store_true',
                    help='the read-out of record, refused before the deadline (the plain command decides by itself)')
    ap.add_argument('--root', default=str(HERE))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if a.deadline_readout and utc_now() < DEADLINE:
            raise Refusal(f'--deadline-readout is allowed only after {DEADLINE_UTC}')
        main(a.root, 'provisional' if a.provisional else 'supplementary' if a.supplementary_full else 'record',
             a.force)
