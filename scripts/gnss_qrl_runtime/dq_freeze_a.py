"""Freeze A: evaluation only (A0 conditional replay ceiling, A1 read-out, stress map, reset swaps). No training.

    .venv/bin/python scripts/gnss_qrl_runtime/dq_freeze_a.py manifest --out <new dir>       # frozen manifest (no episode)
    .venv/bin/python scripts/gnss_qrl_runtime/dq_freeze_a.py freeze   --out <new dir> --manifest <file>
    .venv/bin/python scripts/gnss_qrl_runtime/dq_freeze_a.py a0       --out <new dir>        # REFUSES without approval
    .venv/bin/python scripts/gnss_qrl_runtime/dq_freeze_a.py a1       --out <new dir>        # REFUSES without approval
    .venv/bin/python scripts/gnss_qrl_runtime/dq_freeze_a.py launch   --campaign <dir> --workers 10  # REFUSES likewise

Design (DQ protocol section 8; reviews 34 and 36). Every output is gated by the Freeze A approval
(`DQ_A_PROTOCOL_APPROVAL.json`, key `dq_a_protocol_approval`), including the two computations that draw no seed (A0,
A1): nothing of Freeze A is computed as an outcome before approval.

A0  CONDITIONAL REPLAY CEILING. The 160 P3 `never_fix` rows of the tuning campaign of record (p3_tuning_campaign_v18;
    8 core cells x 20 realisations, D600 2 and 4 m) are re-scored with every recorded report placed at its LOCAL
    (drone-frame) position, i.e. as if georeferencing were perfect, with the scorer's own matching
    (reward_gnss.terminal_score, 3 m xyz, the row's own budget, n_ref and return flag). The search, inspection and
    report decisions stay those recorded. It bounds what coordinate correction ALONE would have recovered on those
    recorded reports; it does not bound the best attainable policy, and it is never called an oracle or headroom.
    Reported per cell and cell-weighted: recorded J, A0 J, their difference in report-equivalents (x N_ref = 18), and
    A0 minus the tuned P3 rule (uncertainty_fix 1.5 m) on the SAME tuning seeds, paired by (cell, realisation).
A1  DESCRIPTIVE READ-OUT of E1 checkpoints (algebra, no episode): for each of the 12 HQ evaluation blocks at update 0
    and at the final, the Fourier power of C(theta) by harmonic order and by interaction order (constant excluded),
    the chart spectrum by parameter group, and the circuit displacement |theta_final - theta_0| (mean, max); for each
    of the 12 Surrogate625 finals, the share of coefficient power outside the circuit support S.
STRESS MAP. A new prospective evaluation of the 48 E1 selected finals (4 families x 12 blocks, bound by SHA-256 from
    each run's summary.json), the P3 reference rule and the uniform-random policy on 16 cells: D600 in {0, 2, 4, 8} m
    x {nominal_2700s, tight_derived} x {positive, no_human}. Each cell is the frozen P3 d2 cell of the same budget and
    presence with only the drift changed (georeference_for(D600); for 2 and 4 m this reproduces the frozen P3 cell's
    georeference exactly, which a test checks). 0 m and 8 m are OUTSIDE the training range: generalisation scenarios.
    20 realisations per cell. CRN: pair_id = 'dqA__<budget>' does not contain the drift, so for a given budget and
    realisation every drift level and both presences meet the same Brownian path, scaled by sqrt(q) (and the same
    re-observation seed); learned policies share the action stream SeedSequence([DQ_SALT, 80, cell_index, r]).
    seed = dq_stress_eval.start + budget_index * 1000 + r.
RESET SWAPS (descriptive; a replication of Chen 2309.07339, not causal identification). On a separate 160-episode suite
    (the 8 cells with D600 2 and 4 m, realisations 500-519, same seed rule), each final is evaluated as is and with one
    part reset to THE SAME RUN's update-0 checkpoint values: HQ `actor.weights` (all 16 circuit angles); Surrogate625
    `actor.surrogate.weight` and `.bias`; C-matched (bottleneck) inner layer `actor.2.weight` and `.bias`; C-strong
    (mlp) both hidden layers `actor.0.*` and `actor.2.*`. Outcome vector: weighted J, timely true reports, false
    reports, return rate, fixes attempted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'src', ROOT / 'scripts', ROOT / 'scripts' / 'gnss_qrl_runtime'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dq_circuits as qc                                                                          # noqa: E402
import dq_runtime as dq                                                                           # noqa: E402
import dq_transplant as tp                                                                        # noqa: E402
import p4_train as p4                                                                             # noqa: E402

# ------------------------------------------------------------------ FROZEN DESIGN ----------------------------------
TOOL = 'gnss-dq-freeze-a/1'
NAMESPACE = 'dq_stress_eval'
STRESS_D600 = (0.0, 2.0, 4.0, 8.0)
IN_TRAINING_RANGE = (2.0, 4.0)
BUDGETS = ('nominal_2700s', 'tight_derived')
PRESENCE = ('positive', 'no_human')
STRESS_REALIZATIONS = tuple(range(20))
RESET_REALIZATIONS = tuple(range(500, 520))
RESET_D600 = (2.0, 4.0)
LEARNED_STREAM, RANDOM_STREAM = 80, 81
FAMILIES = {'quantum': 'HQ', 'bottleneck': 'C-matched', 'mlp': 'C-strong', 'surrogate625': 'Surrogate625'}
BLOCKS = tuple(range(10, 22))
E1_CAMPAIGN = ROOT / 'output/study/gnss_qrl_v1/e1/e1_campaign_v2'
E1_SELECTION = E1_CAMPAIGN / 'tuning' / 'selection.json'
RESET_TARGETS = {'quantum': ('actor.weights',), 'surrogate625': ('actor.surrogate.weight', 'actor.surrogate.bias'),
                 'bottleneck': ('actor.2.weight', 'actor.2.bias'),
                 'mlp': ('actor.0.weight', 'actor.0.bias', 'actor.2.weight', 'actor.2.bias')}
P3_TUNING_RAW = ROOT / 'output/study/gnss_qrl_v1/frozen/p3_tuning_campaign_v18/raw'
NEVER_FIX = 'never_fix'
N_REF = 18
# ------------------------------------------------------------------ END FROZEN DESIGN -------------------------------


class FreezeAError(dq.DQError):
    """A Freeze A refusal."""


def require_open():
    """Every Freeze A outcome (episodes AND the two algebraic read-outs) needs the Freeze A approval."""
    return dq.require_open(NAMESPACE)


# -- cells, seeds and streams -----------------------------------------------------------------------------------------

def base_runtime():
    return dq.cell_runtime()


def stress_cells(runtime):
    """The 16 stress cells, in drift -> budget -> presence order, each derived from the frozen P3 d2 cell."""
    from p3_freeze import georeference_for
    from aerial_qrl.training.checkpoint_identity import canonical_sha256
    base = {(c['budget_name'], c['presence']): c for c in runtime.cells if c['d600_rms_m'] == 2.0}
    if len(base) != 4:
        raise FreezeAError('the frozen P3 manifest does not have the four d2 cells')
    cells = []
    for d600 in STRESS_D600:
        for budget in BUDGETS:
            for presence in PRESENCE:
                cell = copy.deepcopy(base[(budget, presence)])
                geo = georeference_for(d600).to_dict()
                cell.update(cell_index=len(cells), cell_id=f'd{d600:g}__{budget}__{presence}', d600_rms_m=d600,
                            georeference=geo, georeference_sha256=canonical_sha256(geo), pair_id=f'dqA__{budget}',
                            weight=(0.4 if presence == 'positive' else 0.1), seeds={},
                            out_of_training_range=d600 not in IN_TRAINING_RANGE)
                cells.append(cell)
    return cells


def stress_seed(cell, realization):
    space = dq.DQ_NAMESPACES[NAMESPACE]
    if not 0 <= realization < 1000:
        raise FreezeAError(f'realisation {realization} outside [0, 1000)')
    seed = space['start'] + BUDGETS.index(cell['budget_name']) * 1000 + realization
    if not space['start'] <= seed < space['end']:
        raise FreezeAError('seed outside dq_stress_eval')
    return seed


def learned_generator(cell, realization):
    return torch.Generator().manual_seed(dq.DQ_STUDY.stream(LEARNED_STREAM, cell['cell_index'], realization))


def random_generator(cell, realization):
    return torch.Generator().manual_seed(dq.DQ_STUDY.stream(RANDOM_STREAM, cell['cell_index'], realization))


def stress_episodes(cells):
    return [(c, r) for c in cells for r in STRESS_REALIZATIONS]


def reset_episodes(cells):
    return [(c, r) for c in cells if c['d600_rms_m'] in RESET_D600 for r in RESET_REALIZATIONS]


# -- the 48 finals ------------------------------------------------------------------------------------------------------

def selected_finals():
    """[(family, block, run_dir, final checkpoint path, sha256, update-0 path, sha256)], bound by summary.json."""
    selection = json.loads(E1_SELECTION.read_text())
    families = selection['families'] if 'families' in selection else selection['selection']['families']
    out = []
    for family in FAMILIES:
        candidate = families[family] if isinstance(families[family], str) else families[family]['candidate']
        for block in BLOCKS:
            run = E1_CAMPAIGN / 'evaluation' / f'{family}__{candidate}__b{block}' / 'run'
            summary = json.loads((run / 'summary.json').read_text())
            finals = [c for arm in summary['arms'] for c in arm['checkpoints'] if c.get('final')]
            if len(finals) != 1:
                raise FreezeAError(f'{run}: expected one final checkpoint, found {len(finals)}')
            final = finals[0]
            path = Path(final['path'])
            path = path if path.is_absolute() else run / family / 'checkpoints' / path.name
            if not path.exists():
                path = run / family / 'checkpoints' / Path(final['path']).name
            zero = run / family / 'checkpoints' / 'update00000_decisions00000000.pt'
            out.append({'family': family, 'role': FAMILIES[family], 'block': block, 'candidate': candidate,
                        'final': str(path), 'final_sha256': final['sha256'], 'update0': str(zero),
                        'update0_sha256': dq.sha256_file(zero)})
    return out


def load_final(entry):
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    if dq.sha256_file(entry['final']) != entry['final_sha256']:
        raise FreezeAError(f'{entry["final"]} is not the bound final checkpoint')
    return load_checkpoint(entry['final'])[0].eval()


def reset_model(entry):
    """The final with its RESET_TARGETS replaced by the same run's update-0 values; every other tensor untouched."""
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    final = load_final(entry)
    if dq.sha256_file(entry['update0']) != entry['update0_sha256']:
        raise FreezeAError(f'{entry["update0"]} changed')
    zero = load_checkpoint(entry['update0'])[0]
    state = final.state_dict()
    zero_state = zero.state_dict()
    for key in RESET_TARGETS[entry['family']]:
        state[key] = zero_state[key].clone()
    final.load_state_dict(state)
    return final.eval()


# -- episode jobs --------------------------------------------------------------------------------------------------------

def run_learned(runtime, cell, model, realization, label):
    return p4.run_learned_episode(runtime, cell, NAMESPACE, stress_seed(cell, realization), model,
                                  learned_generator(cell, realization), False, label, 0, 1.0, 60.0)['record']


def job_list(finals):
    jobs = [{'job': f'{f["family"]}__b{f["block"]}__{kind}', 'kind': kind, 'final': f}
            for f in finals for kind in ('stress', 'reset_suite_final', 'reset_suite_reset')]
    jobs += [{'job': 'p3_rule__stress', 'kind': 'rule_stress'}, {'job': 'random__stress', 'kind': 'random_stress'}]
    return jobs


def run_job(job):
    """One job; returns its record (failures are recorded, never retried)."""
    import e1_baselines
    torch.set_num_threads(1)
    started = time.perf_counter()
    record = {'job': job['job'], 'kind': job['kind'], 'status': 'started'}
    try:
        require_open()
        runtime = base_runtime()
        cells = stress_cells(runtime)
        episodes = []
        if job['kind'] in ('stress', 'reset_suite_final', 'reset_suite_reset'):
            model = reset_model(job['final']) if job['kind'] == 'reset_suite_reset' else load_final(job['final'])
            plan = stress_episodes(cells) if job['kind'] == 'stress' else reset_episodes(cells)
            for cell, r in plan:
                rec = run_learned(runtime, cell, model, r, f'dqA-{job["job"]}-c{cell["cell_index"]}-r{r}')
                episodes.append(_keep(rec, cell, r))
            record['final_sha256'] = job['final']['final_sha256']
        elif job['kind'] == 'rule_stress':
            rule = e1_baselines.p3_rule()
            for cell, r in stress_episodes(cells):
                rec = e1_baselines.run_policy_episode(runtime, cell, NAMESPACE, stress_seed(cell, r), 'p3_rule',
                                                      f'dqA-rule-c{cell["cell_index"]}-r{r}', rule=rule)
                episodes.append(_keep(rec, cell, r))
            record['rule'] = rule
        else:
            model = e1_baselines.zero_actor_model()
            for cell, r in stress_episodes(cells):
                rec = e1_baselines.run_policy_episode(runtime, cell, NAMESPACE, stress_seed(cell, r), 'random',
                                                      f'dqA-random-c{cell["cell_index"]}-r{r}',
                                                      generator=random_generator(cell, r), model=model)
                episodes.append(_keep(rec, cell, r))
        record.update(status='complete', episodes=episodes)
    except dq.DQError as exc:
        record.update(status='refused', error=str(exc))
    except Exception as exc:
        record.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    record['wall_s'] = time.perf_counter() - started
    return record


def _keep(rec, cell, r):
    return {'cell_id': cell['cell_id'], 'cell_index': cell['cell_index'], 'd600': cell['d600_rms_m'],
            'budget': cell['budget_name'], 'presence': cell['presence'], 'realization': r,
            **{k: rec[k] for k in ('seed', 'crn_sha256', 'J', 'tp_timely', 'fp_spatial', 'return_ok',
                                   'fixes_attempted', 'termination_reason')}}


# -- A0 and A1 ---------------------------------------------------------------------------------------------------------

def a0_rows():
    """The never_fix and tuned-rule rows of the P3 tuning campaign of record, keyed by (cell_id, realisation)."""
    import e1_baselines
    rule_sha = e1_baselines.RULE_SHA256
    never, rule = {}, {}
    for path in sorted(P3_TUNING_RAW.glob('unit_*.json')):
        unit = json.loads(path.read_text())
        for cell_id, row in unit['rows'].items():
            key = (cell_id, row['realization'])
            if row['comparator'] == NEVER_FIX:
                never[key] = row
            elif row['candidate_sha256'] == rule_sha:
                rule[key] = row
    return never, rule


def rescore(row, positions):
    from aerial_qrl.mission.evaluation import EvaluatorSpec
    from aerial_qrl.training.reward_gnss import TimelyScoreSpec, terminal_score
    spec = TimelyScoreSpec(row['score_spec']['score_version'], row['mission_budget_s_applied'], row['n_ref'],
                           row['score_spec']['tolerance_m'], row['score_spec']['metric'],
                           row['score_spec']['false_report_weight'], row['score_spec']['provenance'])
    targets = _targets(row['presence'])
    times = [s['sim_time_s'] for s in row['submissions']]
    return terminal_score([tuple(p) for p in positions], times, targets, spec, row['return_ok'])[0]


_TARGETS = {}


def _targets(presence):
    if presence not in _TARGETS:
        runtime = base_runtime()
        cell = next(c for c in runtime.cells if c['presence'] == presence)
        _TARGETS[presence] = runtime.static(cell)['context']['evaluator'].targets
    return _TARGETS[presence]


def a0():
    require_open()
    never, rule = a0_rows()
    if len(never) != 160:
        raise FreezeAError(f'expected 160 never_fix rows, found {len(never)}')
    per = []
    for key, row in sorted(never.items()):
        recorded = rescore(row, [s['global_position_enu'] for s in row['submissions']])
        if abs(recorded - row['score_J']) > 1e-12:
            raise FreezeAError(f'{key}: the scorer does not reproduce the stored score ({recorded} vs {row["score_J"]})')
        ceiling = rescore(row, [s['local_position_enu'] for s in row['submissions']])
        per.append({'cell_id': key[0], 'realization': key[1], 'weight': row['cell_weight'], 'recorded_J': recorded,
                    'A0_J': ceiling, 'rule_J': rule[key]['score_J'] if key in rule else None})
    cells = sorted({p['cell_id'] for p in per})
    summary = {}
    for cell in cells:
        rows = [p for p in per if p['cell_id'] == cell]
        summary[cell] = {k: float(np.mean([r[k] for r in rows])) for k in ('recorded_J', 'A0_J')}
        paired = [r['A0_J'] - r['rule_J'] for r in rows if r['rule_J'] is not None]
        summary[cell]['A0_minus_rule_RE'] = float(np.mean(paired) * N_REF) if paired else None
        summary[cell]['A0_minus_recorded_RE'] = (summary[cell]['A0_J'] - summary[cell]['recorded_J']) * N_REF
        summary[cell]['weight'] = rows[0]['weight']
    total = sum(s['weight'] for s in summary.values())
    weighted = {k: sum(s['weight'] * s[k] for s in summary.values()) / total
                for k in ('recorded_J', 'A0_J', 'A0_minus_rule_RE', 'A0_minus_recorded_RE')}
    return {'label': 'conditional replay ceiling (not an oracle, not total headroom)', 'rows': per,
            'per_cell': summary, 'cell_weighted': weighted}


def fourier_power(C):
    """Power of C (features x wires) by harmonic order and interaction order, constant term excluded."""
    order = torch.tensor([0, 1, 1, 2, 2])
    idx = torch.cartesian_prod(*[torch.arange(5)] * 4)
    orders = order[idx]                                            # (625, 4) per-wire frequency order
    harmonic, interaction = orders.max(1).values, (orders > 0).sum(1)
    power = (C ** 2).sum(1)
    ac = power[1:].sum()
    return {'harmonic_order': {int(h): float(power[harmonic == h].sum() / ac) for h in (1, 2)},
            'interaction_order': {int(i): float(power[interaction == i].sum() / ac) for i in (1, 2, 3, 4)}}


def a1():
    require_open()
    from aerial_qrl.training.smdp_ppo import load_checkpoint
    import dq_instruments as ins
    support, _ = tp.support(qc.SHIPPED)
    finals = selected_finals()
    hq, s625 = [], []
    for entry in finals:
        if entry['family'] == 'quantum':
            w0 = load_checkpoint(entry['update0'])[0].actor.weights.detach()
            w1 = load_final(entry).actor.weights.detach()
            rec = {'block': entry['block']}
            for tag, w in (('update0', w0), ('final', w1)):
                C = qc.coefficients(qc.SHIPPED, w)
                A = qc.jacobian(qc.SHIPPED, w)[support]
                k, s, amb = ins.chart_rank(A, 1e-8)
                groups = {'ry_offsets': [l * 8 + j * 2 for l in range(2) for j in range(4)],
                          'rz_layer1': [j * 2 + 1 for j in range(4)], 'rz_layer2_inert': [8 + j * 2 + 1 for j in range(4)]}
                rec[tag] = {'fourier_power': fourier_power(C), 'rank': k,
                            'singular_values': [float(x) for x in s],
                            'by_group': {g: [float(x) for x in torch.linalg.svdvals(A[:, i])] for g, i in groups.items()}}
            d = (w1 - w0).abs()
            rec['displacement_rad'] = {'mean': float(d.mean()), 'max': float(d.max())}
            hq.append(rec)
        elif entry['family'] == 'surrogate625':
            C = load_final(entry).actor.surrogate.weight.detach().T.reshape(-1)
            s625.append({'block': entry['block'],
                         'power_outside_circuit_support': float((C[~support] ** 2).sum() / (C ** 2).sum())})
    return {'hq': hq, 'surrogate625': s625}


# -- manifest, freeze, launch ----------------------------------------------------------------------------------------------

def manifest():
    runtime = base_runtime()
    cells = stress_cells(runtime)
    finals = selected_finals()
    return {'manifest_version': 'gnss-dq-freeze-a-manifest/1', 'study': 'dq', 'salt': [dq.DQ_SALT],
            'namespace': NAMESPACE, 'namespace_range': dq.DQ_NAMESPACES[NAMESPACE],
            'cells': [{k: c[k] for k in ('cell_index', 'cell_id', 'd600_rms_m', 'budget_name', 'mission_budget_s',
                                         'presence', 'pair_id', 'weight', 'georeference_sha256', 'n_ref',
                                         'out_of_training_range')} for c in cells],
            'stress_seeds': {c['cell_id']: [stress_seed(c, r) for r in STRESS_REALIZATIONS] for c in cells},
            'reset_seeds': {c['cell_id']: [stress_seed(c, r) for r in RESET_REALIZATIONS] for c in cells
                            if c['d600_rms_m'] in RESET_D600},
            'streams': {'learned': f'SeedSequence([{dq.DQ_SALT}, {LEARNED_STREAM}, cell_index, r])',
                        'random': f'SeedSequence([{dq.DQ_SALT}, {RANDOM_STREAM}, cell_index, r])'},
            'finals': finals, 'reset_targets': RESET_TARGETS,
            'jobs': [j['job'] for j in job_list(finals)],
            'a0_source': {'raw_dir': str(P3_TUNING_RAW.relative_to(ROOT)), 'never_fix_rows': 160},
            'episode_counts': {'stress_per_policy': len(STRESS_REALIZATIONS) * len(cells),
                               'reset_suite_per_model': len(RESET_REALIZATIONS) * 8}}


FROZEN_FILES = ('scripts/gnss_qrl_runtime/dq_freeze_a.py', 'scripts/gnss_qrl_runtime/dq_freeze_a_analysis.py',
                'scripts/gnss_qrl_runtime/dq_runtime.py', 'scripts/gnss_qrl_runtime/dq_circuits.py',
                'scripts/gnss_qrl_runtime/dq_transplant.py', 'scripts/gnss_qrl_runtime/dq_instruments.py',
                'scripts/gnss_qrl_runtime/e1_baselines.py', 'scripts/gnss_qrl_runtime/p4_train.py',
                'tests/gnss_qrl_dq/test_dq_freeze_a.py', 'docs/study/gnss_qrl_v1/frozen/DQ_FREEZE_A_PROTOCOL_EN.md')


def write_manifest(out):
    dq.claim_output(out, directory=True)
    m = manifest()
    m['manifest_sha256'] = dq.canonical_sha256(m)
    path = out / 'freeze_a_manifest.json'
    dq.write_json(path, {'utc': dq.utc(), 'seed_manifest': m})
    return path, m['manifest_sha256']


def write_freeze(out, manifest_path):
    dq.watch_trees()
    dq.claim_output(out, directory=True)
    m = json.loads(Path(manifest_path).read_text())['seed_manifest']
    record = {'tool': TOOL, 'utc': dq.utc(), 'protocol': 'docs/study/gnss_qrl_v1/frozen/DQ_FREEZE_A_PROTOCOL_EN.md',
              'files': {f: dq.sha256_file(ROOT / f) for f in FROZEN_FILES},
              'seed_manifest': str(Path(manifest_path).relative_to(ROOT)), 'seed_manifest_sha256': m['manifest_sha256'],
              'copy_code_tree_sha256': dq.dq_tree()[0], 'statement': 'Freeze A package; nothing has run under it'}
    path = out / 'freeze.json'
    dq.write_json(path, record)
    return path, dq.sha256_file(path)


def launch(campaign, workers):
    import multiprocessing as mp
    require_open()
    campaign = Path(campaign)
    jobs_dir = campaign / 'jobs'
    dq.claim_output(jobs_dir, directory=True)
    dq.append_tree_watch('freeze A launch')
    jobs = job_list(selected_finals())
    jobs.sort(key=lambda j: 0 if j['kind'] == 'rule_stress' else 1)             # the slowest job first
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(run_job, jobs):
            dq.write_json(jobs_dir / f'{record["job"]}.json', json.loads(json.dumps(record, default=float)))
    dq.append_tree_watch('freeze A done')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('manifest', 'a0', 'a1'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True, type=Path)
    fr = sub.add_parser('freeze')
    fr.add_argument('--out', required=True, type=Path)
    fr.add_argument('--manifest', required=True, type=Path)
    la = sub.add_parser('launch')
    la.add_argument('--campaign', required=True, type=Path)
    la.add_argument('--workers', type=int, default=10)
    args = parser.parse_args(argv)
    if args.command == 'manifest':
        print(json.dumps(dict(zip(('path', 'manifest_sha256'), map(str, write_manifest(args.out))))))
    elif args.command == 'freeze':
        print(json.dumps(dict(zip(('path', 'sha256'), map(str, write_freeze(args.out, args.manifest))))))
    elif args.command in ('a0', 'a1'):
        result = a0() if args.command == 'a0' else a1()
        dq.claim_output(args.out, directory=True)
        dq.write_json(args.out / f'{args.command}.json', json.loads(json.dumps(result, default=float)))
    else:
        launch(args.campaign, args.workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
