"""E1: a learned correction trigger inside the tuned rule's executive (see PLAN_DRAFT.md / PLAN.md).

Imports the idea_dq code tree unchanged (its live hash must stay ffe9dfef...; gnss_policies.py is only subclassed,
and e1_baselines.p3_rule() re-checks its hash). Writes only inside this directory.

    python runner.py test                         # unit tests (synthetic records)
    python runner.py precheck [--workers 8] [--tag _draft]
        (a) code tree + rule source hashes; (b) unit tests; (c) the rule-imitating gate (rms >= 1.5) run through THIS
        gate code path reproduces all 80 stored e1_evaluation rule rows exactly (J, fixes, decisions, crn_sha256 and
        every other stored field) with the segment identity on every episode; (d) a training smoke on development
        seeds (about 30 episodes, 3 updates) end to end, checkpoint reload and evaluation determinism; timings.
    python runner.py smoke [--tag _x]             # (d) alone
    python runner.py launch --workers 12          # the 12 main runs of PLAN.md (refused unless PLAN.md is hashed and
                                                  # precheck.json passed for this plan and this code)
    python runner.py precheck-stress [--workers 8] [--per-cell N]   # rule reproduction on the stored stress rows
    python runner.py evaluate-stress --workers 12 # secondary read-out of the finals on the 320-episode stress map
    python runner.py pilot --lrs 1e-3 3e-4 --blocks 0 1 --budget 6400 --workers 4   # OPTIONAL development pilot
    python runner.py pilot --lrs 1e-3 --blocks 2 3 --budget 2560 --workers 2 --quick  # learning check (dev only)
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_core as gc                                                                            # noqa: E402

import numpy as np                                                                                # noqa: E402
import torch                                                                                      # noqa: E402

import dq_runtime as dq                                                                           # noqa: E402
import e1_baselines                                                                               # noqa: E402
import e1_seed_manifest                                                                           # noqa: E402
import fast_mission                                                                               # noqa: E402
import p4_train as p4                                                                             # noqa: E402
import run_p3_cells as runner                                                                     # noqa: E402
from aerial_qrl.mission.contracts import INFRASTRUCTURE                                           # noqa: E402
from aerial_qrl.training.gnss_rewards import place_gnss_rewards                                   # noqa: E402
from aerial_qrl.training.reward_gnss import check_reward_identity, episode_rewards                # noqa: E402
from aerial_qrl.training.smdp_ppo import RolloutBuffer, compute_advantages, file_sha256, ppo_update  # noqa: E402

IDEA_DQ = gc.IDEA_DQ
TOOL = 'e1-gate-train/1'
EXPECTED_TREE = 'ffe9dfef2b24ba48a48772e696656e3b0d2aa1abb6de8fa73cf1d51c7f745952'
PLAN = HERE / 'PLAN.md'
PLAN_SHA256 = hashlib.sha256(PLAN.read_bytes()).hexdigest() if PLAN.exists() else None
BASELINES = Path('/home/konstantinos/Documents/ChatGPT/idea_e1/output/study/gnss_qrl_v1/e1/e1_campaign_v2/'
                 'baselines/baselines.json')
BASELINES_SHA256 = '00202039dda01d814146db77e85590996abda974b4e813bbb17c10d081572fd7'
STRESS_RULE = IDEA_DQ / 'output/study/gnss_qrl_v1/dq/freeze_a_campaign_v1/jobs/p3_rule__stress.json'

# ------------------------------------------------------------------ DESIGN (copied into PLAN_DRAFT.md) --------------
#: Two new seed namespaces after the last DQ namespace (dq_e2_evaluation = index 18), by P3's own rule
#: start = 20260921 * 1000 + index * 100,000. CellRuntime refuses any overlap with a frozen or E1/DQ range.
GATE_NAMESPACES = {name: {'start': dq.NAMESPACE_BASE * 1000 + index * dq.NAMESPACE_WIDTH,
                          'end': dq.NAMESPACE_BASE * 1000 + (index + 1) * dq.NAMESPACE_WIDTH}
                   for index, name in ((19, 'gate_training'), (20, 'gate_development'))}
EXTRA_NAMESPACES = {**p4.E1_NAMESPACES, **dq.DQ_NAMESPACES, **GATE_NAMESPACES}
SALT_MAIN, SALT_DEV = (20261025, 1), (20261025, 2)
STRIDE = 5_000                      # training seeds per block: seed = start + block * STRIDE + episode
BLOCKS = tuple(range(12))           # main runs (gate_training)
SMOKE_BLOCK, PILOT_BLOCKS = 19, (0, 1, 2, 3)   # development runs (gate_development)
BUDGET = 12_800                     # gate decisions (valid SMDP rows) per run
GRID = [0, 3_200, 6_400, 9_600, 12_800]
TRANSITIONS_PER_UPDATE = 256
HIDDEN = 64
PPO = {'learning_rate': 1e-3, 'clip_range': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.0, 'max_grad_norm': 0.5,
       'epochs': 10, 'minibatch_transitions': 64, 'advantage_std_floor': 0.0056, 'max_segment_length': 1}
GAMMA_REF, TAU_S, LAMBDA_REF, LAMBDA_MODE = 1.0, 60.0, 0.95, 'time'
CURVE_SUITE, FINAL_SUITE = 'e1_development', 'e1_evaluation'
# ------------------------------------------------------------------ END DESIGN ---------------------------------------

p4.BLOCK_STRIDE = STRIDE            # read at call time by CellRuntime.schedule; this process only


def _study(name, salt, training_namespace):
    return p4.Study(name=name, tool=TOOL, salt=salt, training_namespace=training_namespace,
                    extra_namespaces=EXTRA_NAMESPACES,
                    non_claim='A post-hoc E1 gate run (PLAN.md); labelled post hoc, not part of any frozen family.')


STUDY_MAIN = _study('e1gate', SALT_MAIN, 'gate_training')
STUDY_DEV = _study('e1gate_dev', SALT_DEV, 'gate_development')
STUDIES = {s.name: s for s in (STUDY_MAIN, STUDY_DEV)}
EPISODE_FIELDS = ('cell_id', 'seed', 'crn_sha256', 'termination_reason', 'return_ok', 'J', 'tp_timely', 'tp_spatial',
                  'fp_spatial', 'duplicates', 'fixes_attempted', 'decisions', 'rejections', 'elapsed_s')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {f: sha(HERE / f) for f in ('runner.py', 'gate_core.py', 'test_gate.py', 'gate_analysis.py')
            if (HERE / f).exists()}


def runtime(study):
    c = 'output/study/gnss_qrl_v1/claude/'
    return p4.CellRuntime(IDEA_DQ / (c + 'p3_freeze_v1/p3_frozen_configuration.json'),
                          IDEA_DQ / (c + 'p3_execution_manifest_v1/p3_execution_manifest.json'),
                          IDEA_DQ / 'output/training/time_model_village_v1/20260917T074923Z/time_model.json',
                          study=study)


def stored_rule_rows(suite):
    """{(cell_id, seed): stored P3-rule row} of an E1 suite (baselines.json, bound by SHA-256)."""
    if sha(BASELINES) != BASELINES_SHA256:
        raise RuntimeError('baselines.json is not the bound file')
    rows = json.loads(BASELINES.read_text())['results'][f'p3_rule/{suite}']['episodes']
    return {(r['cell_id'], r['seed']): r for r in rows}


def config(block, study=STUDY_MAIN, **over):
    c = {'version': 'e1-gate-config/1', 'tool': TOOL, 'study': study.name, 'salt': list(study.salt),
         'training_namespace': study.training_namespace,
         'training_seed_rule': f'{study.training_namespace}.start + block * {STRIDE} + episode',
         'block': int(block), 'rule': e1_baselines.p3_rule(), 'model': {'kind': 'gate-mlp/1', 'hidden': HIDDEN,
                                                                        'features': list(gc.FEATURES)},
         'budget_gate_decisions': BUDGET, 'grid': list(GRID), 'transitions_per_update': TRANSITIONS_PER_UPDATE,
         'ppo': dict(PPO), 'gamma_ref': GAMMA_REF, 'reference_duration_s': TAU_S, 'lambda_ref': LAMBDA_REF,
         'lambda_mode': LAMBDA_MODE, 'curve_suite': CURVE_SUITE, 'final_suite': FINAL_SUITE, 'readout': 'argmax',
         'curve_realizations': None, 'final_realizations': None, 'plan_sha256': PLAN_SHA256,
         'code_sha256': code_hashes()}
    for k, v in over.items():
        if k == 'ppo':
            c['ppo'] = dict(c['ppo'], **v)
        else:
            c[k] = v
    if c['final_suite'] == 'e1_evaluation' and study is not STUDY_MAIN:
        raise RuntimeError('a development run never reads the e1_evaluation suite')
    return c


def build_model(study, block, hidden=HIDDEN):
    torch.manual_seed(study.stream(block, p4.INIT_STREAM))
    return gc.GateActorCritic(gc.N_FEATURES, hidden).double()


# ------------------------------------------------------------------ one episode ---------------------------------------

def run_gate_episode(rt, cell, namespace, seed, gate, rule, episode_id, episode_number, gamma=GAMMA_REF, tau=TAU_S):
    """One episode of GatePolicy on a prepared cell, scored exactly as e1_baselines.run_policy_episode scores the rule;
    rewards placed on the records, gate segments built, and the segment identity asserted."""
    static = rt.static(cell)
    backend_config, group, service = runner.cell_episode_parts(static, cell, namespace, seed)
    holder = {}

    def factory(briefing, obstacle_map, mission_config, reserve):
        holder['policy'] = gc.GatePolicy(briefing, obstacle_map, mission_config, reserve, gate=gate,
                                         **rule['parameters'])
        return holder['policy']
    started = time.perf_counter()
    result = fast_mission.run_fast_episode(static['context'], static['banks'], rt.time_model, episode_id,
                                           backend_config=backend_config,
                                           executive_kwargs={'georeference': service, 'anchor_set': static['anchors']},
                                           policy_factory=factory, reports_for_matching=service.global_report_positions)
    wall = time.perf_counter() - started
    outcome, policy = result['outcome'], holder['policy']
    return_ok = outcome.termination_reason == 'mission_complete' and outcome.returned and outcome.landed
    times = [s.sim_time_s for s in service.submissions]
    breakdown = episode_rewards(service.global_report_positions(), times, static['context']['evaluator'].targets,
                                static['score'], return_ok)
    check_reward_identity(breakdown)
    placed = place_gnss_rewards(result['records'], breakdown, times)
    segments = gc.build_gate_segments(policy.decisions, result['records'], placed.per_record,
                                      outcome.termination_reason, episode_number, gamma, tau)
    residual = gc.check_segment_identity(segments, breakdown.score, placed.dropped)
    final = breakdown.final
    durations = [r['duration_s'] for r in segments.rows if r['valid']]
    record = {'cell_id': cell['cell_id'], 'seed': seed, 'crn_sha256': group['crn_sha256'],
              'termination_reason': outcome.termination_reason, 'return_ok': return_ok, 'J': breakdown.score,
              'tp_timely': final.tp_timely, 'tp_spatial': final.tp_spatial, 'fp_spatial': final.fp_spatial,
              'duplicates': final.duplicates, 'fixes_attempted': len(service.measurements),
              'decisions': outcome.decisions, 'rejections': outcome.rejections, 'elapsed_s': outcome.elapsed_s,
              'wall_s': wall, 'infrastructure': outcome.termination_reason == INFRASTRUCTURE,
              'gate_rows': len(segments.rows), 'gate_valid_rows': segments.valid_rows,
              'gate_fix_rows': sum(1 for r in segments.rows if r['action'] == gc.FIX_FIRST),
              'gate_stats': dict(policy.stats), 'kept_reward': segments.kept_reward,
              'pre_gate_reward': segments.info['pre_gate_reward'],
              'invalid_segment_reward': segments.info['invalid_segment_reward'], 'dropped_reward': placed.dropped,
              'identity_residual': residual, 'min_segment_duration_s': min(durations) if durations else None,
              'gates': [[round(m['start_s'], 3), int(m['action_index']),
                         round(policy.decisions[m['decision_index']].info.get('rms_m', float('nan')), 4)]
                        for m in segments.meta]}
    return {'record': record, 'segments': segments, 'policy': policy}


def suite_episodes(rt, suite, realizations=None):
    eps = e1_seed_manifest.suite_episodes(rt, suite)
    return [e for e in eps if realizations is None or e[1] in realizations]


def evaluate_suite(model, rt, suite, rule, label, realizations=None):
    """Argmax read-out of a gate model on an E1 suite, paired row by row with the stored rule rows."""
    namespace = p4.STUDIES['e1'].suites[suite][0]
    stored = stored_rule_rows(suite)
    gate = gc.NetworkGate(model, deterministic=True)
    rows = []
    for cell, r, seed in suite_episodes(rt, suite, realizations):
        run = run_gate_episode(rt, cell, namespace, seed, gate, rule, f'{label}-{suite}-c{cell["cell_index"]}-r{r}', -1)
        rec = run['record']
        s = stored[(cell['cell_id'], seed)]
        if s['crn_sha256'] != rec['crn_sha256']:
            raise RuntimeError(f'{cell["cell_id"]} r{r}: crn_sha256 differs from the stored rule row')
        rows.append(dict(rec, realization=r, rule_J=s['J'], d_J=rec['J'] - s['J'],
                         rule_fixes=s['fixes_attempted'], rule_fp=s['fp_spatial'], rule_tp_timely=s['tp_timely']))
    out = {'suite': suite, 'readout': 'argmax', 'episodes': rows, 'n': len(rows)}
    if {r['cell_id'] for r in rows} == {c['cell_id'] for c in rt.cells}:
        per_cell, weighted = p4.aggregate_evaluation(rt, rows)
        _, rule_weighted = p4.aggregate_evaluation(rt, [dict(r, J=r['rule_J']) for r in rows])
        out.update(per_cell_J=per_cell, weighted_J=weighted, rule_weighted_J=rule_weighted,
                   weighted_d=weighted - rule_weighted, unweighted_mean_d=float(np.mean([r['d_J'] for r in rows])))
    return out


# ------------------------------------------------------------------ training (adapted from p4_train.train_arm) -------

def train_gate(cfg, rt, run_dir, max_updates=None):
    """Train one gate run; returns its summary. Checkpoints and curve read-outs on cfg['grid'], final read-out on
    cfg['final_suite']. An algorithmic failure (non-finite value, schedule exhausted) stops the run without a final."""
    study = STUDIES[cfg['study']]
    if rt.study is not study:
        raise RuntimeError('runtime/study mismatch')
    block, ppo, rule = cfg['block'], cfg['ppo'], cfg['rule']
    gamma, tau = cfg['gamma_ref'], cfg['reference_duration_s']
    torch.set_num_threads(1)
    model = build_model(study, block, cfg['model']['hidden'])
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo['learning_rate'])
    action_generator = torch.Generator().manual_seed(study.stream(block, p4.ACTION_STREAM))
    minibatch_generator = torch.Generator().manual_seed(study.stream(block, p4.MINIBATCH_STREAM))
    run_dir.mkdir(parents=True)
    (run_dir / 'checkpoints').mkdir()
    log = p4.JsonLines(run_dir / 'metrics.jsonl')
    totals = {'updates': 0, 'gate_decisions': 0, 'episodes': 0, 'empty_episodes': 0, 'infrastructure_episodes': 0,
              'dropped_reward': 0., 'pre_gate_reward': 0., 'invalid_segment_reward': 0.}
    walls = {'rollout_s': 0., 'update_s': 0., 'checkpoint_s': 0., 'evaluation_s': 0.}
    cpu = {'rollout_s': 0., 'update_s': 0., 'evaluation_s': 0.}
    load_average = {'start': list(os.getloadavg())}
    done_points, checkpoints, curve = [], [], []
    start = time.perf_counter()
    label = f'{study.name}-b{block}'
    salt = ''.join(f'{s}, ' for s in study.salt)
    log.write({'kind': 'arm_start', 'utc': p4.utc(), 'tool': TOOL, 'config': cfg, 'model': model.describe(),
               'code_tree_sha256': p4.live_copy_tree(),
               'seeds': {'model_init': f'SeedSequence([{salt}{block}, {p4.INIT_STREAM}])',
                         'action_sampling': f'SeedSequence([{salt}{block}, {p4.ACTION_STREAM}])',
                         'minibatch_order': f'SeedSequence([{salt}{block}, {p4.MINIBATCH_STREAM}])',
                         'schedule': f'cell: SeedSequence([{salt}{block}, {p4.SCHEDULE_STREAM}, episode]); seed: '
                                     f'{cfg["training_seed_rule"]}'},
               'non_claim': study.non_claim})

    def checkpoint(points, final=False):
        t0 = time.perf_counter()
        name = f'{"final_" if final else ""}update{totals["updates"]:05d}_decisions{totals["gate_decisions"]:08d}.pt'
        path = run_dir / 'checkpoints' / name
        gc.save_gate_checkpoint(path, model, optimizer, cfg, {'points': points, 'final': final, **totals})
        walls['checkpoint_s'] += time.perf_counter() - t0
        entry = {'kind': 'checkpoint', 'path': str(path.relative_to(run_dir)), 'sha256': file_sha256(path),
                 'points': points, 'final': final, 'gate_decisions': totals['gate_decisions']}
        log.write(entry)
        checkpoints.append(entry)

    def evaluate_curve(points):
        t0, c0 = time.perf_counter(), time.process_time()
        res = evaluate_suite(model, rt, cfg['curve_suite'], rule, f'{label}-curve{len(curve):02d}',
                             cfg.get('curve_realizations'))
        walls['evaluation_s'] += time.perf_counter() - t0
        cpu['evaluation_s'] += time.process_time() - c0
        for p in points:
            curve.append((p, totals['gate_decisions'], res.get('weighted_J'), res.get('weighted_d')))
        log.write({'kind': 'evaluation', 'points': points, 'gate_decisions': totals['gate_decisions'], 'result': res,
                   'wall_s': time.perf_counter() - t0})

    def due():
        hit = [p for p in cfg['grid'] if p not in done_points and totals['gate_decisions'] >= p]
        done_points.extend(hit)
        return hit

    def one_update():
        buffer = RolloutBuffer(gc.N_FEATURES, gc.N_GATE_ACTIONS, model.hidden)
        r0, c0, kept, here, sim_s = time.perf_counter(), time.process_time(), 0, 0, 0.
        gate = gc.NetworkGate(model, action_generator, deterministic=False)
        while kept < cfg['transitions_per_update']:
            number = totals['episodes']
            cell, seed = rt.schedule(block, number)
            run = run_gate_episode(rt, cell, study.training_namespace, seed, gate, rule,
                                   f'{label}-e{number:06d}', number, gamma, tau)
            rec, segs = run['record'], run['segments']
            totals['episodes'] += 1
            here += 1
            totals['dropped_reward'] += rec['dropped_reward']
            totals['pre_gate_reward'] += rec['pre_gate_reward']
            totals['invalid_segment_reward'] += rec['invalid_segment_reward']
            totals['infrastructure_episodes'] += rec['infrastructure']
            sim_s += rec['elapsed_s'] or 0.
            if segs.valid_rows:
                segs.add_to(buffer)
                kept += segs.valid_rows
            else:
                totals['empty_episodes'] += 1
            log.write(dict(rec, kind='episode', episode=number, update_index=totals['updates'] + 1))
        rollout_wall, rollout_cpu = time.perf_counter() - r0, time.process_time() - c0
        batch = compute_advantages(buffer, gamma, tau, cfg['lambda_ref'], cfg['lambda_mode'], ppo['max_segment_length'])
        fix_rate = float(np.mean(batch['actions'] == gc.FIX_FIRST))
        u0, uc0 = time.perf_counter(), time.process_time()
        metrics = ppo_update(model, optimizer, batch, ppo['clip_range'], ppo['value_coef'], ppo['entropy_coef'],
                             ppo['max_grad_norm'], ppo['epochs'], generator=minibatch_generator,
                             minibatch_transitions=ppo['minibatch_transitions'],
                             advantage_std_floor=ppo['advantage_std_floor'])
        update_wall, update_cpu = time.perf_counter() - u0, time.process_time() - uc0
        bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        if bad:
            raise FloatingPointError(f'non-finite parameters after update {totals["updates"] + 1}: {bad[:4]}')
        used = int(metrics['transitions'])
        if used != kept:
            raise RuntimeError(f'internal: {kept} kept rows, {used} used by the update')
        with torch.no_grad():                    # the updated policy's P(fix) on this rollout's gate states
            x = torch.as_tensor(batch['obs'], dtype=torch.float64).unsqueeze(1)
            p_fix = torch.softmax(model(x)[0], -1)[:, 0, gc.FIX_FIRST].numpy()
        rms = batch['obs'][:, 0] * gc.TOLERANCE_SCALE_M
        totals['updates'] += 1
        totals['gate_decisions'] += used
        walls['rollout_s'] += rollout_wall
        walls['update_s'] += update_wall
        cpu['rollout_s'] += rollout_cpu
        cpu['update_s'] += update_cpu
        log.write({'kind': 'update', 'update_index': totals['updates'], 'metrics': metrics, 'transitions': used,
                   'episodes': here, 'sim_s': sim_s, 'rollout_fix_rate': fix_rate,
                   'p_fix_after': {'mean': float(p_fix.mean()),
                                   'rms_lt_1.5': float(p_fix[rms < 1.5].mean()) if (rms < 1.5).any() else None,
                                   'rms_ge_1.5': float(p_fix[rms >= 1.5].mean()) if (rms >= 1.5).any() else None},
                   'wall': {'rollout_s': rollout_wall, 'update_s': update_wall,
                            'rollout_per_1000_decisions_s': 1000. * rollout_wall / used,
                            'rollout_per_episode_s': rollout_wall / here},
                   'cumulative': dict(totals)})
        points = due()
        if points:
            checkpoint(points)
            evaluate_curve(points)

    points = due()
    if points:
        checkpoint(points)
        evaluate_curve(points)
    failure = None
    while failure is None and totals['gate_decisions'] < cfg['budget_gate_decisions'] and \
            (max_updates is None or totals['updates'] < max_updates):
        try:
            one_update()
        except (FloatingPointError, p4.ScheduleExhausted) as error:
            failure = {'kind': type(error).__name__, 'message': str(error), **totals}
            log.write({'kind': 'algorithmic_failure', **failure})
    final_eval = None
    if failure is None:
        checkpoint([], final=True)
        t0, c0 = time.perf_counter(), time.process_time()
        final_eval = evaluate_suite(model, rt, cfg['final_suite'], rule, f'{label}-final',
                                    cfg.get('final_realizations'))
        walls['evaluation_s'] += time.perf_counter() - t0
        cpu['evaluation_s'] += time.process_time() - c0
        log.write({'kind': 'final_evaluation', 'result': final_eval, 'wall_s': time.perf_counter() - t0})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    load_average['end'] = list(os.getloadavg())
    summary = {'study': study.name, 'block': block, **totals, 'status': 'completed' if failure is None
               else 'algorithmic_failure', 'failure': failure, 'wall_s': time.perf_counter() - start, 'walls': walls,
               'cpu': cpu, 'load_average': load_average,
               'process_rusage': {'user_s': usage.ru_utime, 'system_s': usage.ru_stime, 'max_rss_kib': usage.ru_maxrss},
               'rollout_per_1000_decisions_s': 1000. * walls['rollout_s'] / max(totals['gate_decisions'], 1),
               'rollout_per_episode_s': walls['rollout_s'] / max(totals['episodes'], 1),
               'model_parameters': model.parameter_counts(), 'checkpoints': checkpoints,
               'learning_curve': curve,
               'final_evaluation': None if final_eval is None else
               {k: v for k, v in final_eval.items() if k != 'episodes'}}
    log.write(dict(summary, kind='arm_end', utc=p4.utc()))
    return summary, final_eval


# ------------------------------------------------------------------ rule reproduction (precheck) ---------------------

COMPARE = EPISODE_FIELDS


def _reproduce_cell(task):
    """Rule-imitating gate through the gate code path on one cell's suite episodes; compares with stored rows."""
    suite, cell_index, realizations = task
    torch.set_num_threads(1)
    rt = runtime(STUDY_MAIN)
    rule = e1_baselines.p3_rule()
    stored = stored_rule_rows(suite)
    namespace = p4.STUDIES['e1'].suites[suite][0]
    out = []
    for cell, r, seed in suite_episodes(rt, suite, realizations):
        if cell['cell_index'] != cell_index:
            continue
        try:
            run = run_gate_episode(rt, cell, namespace, seed, gc.RuleGate(rule['parameters']['sigma_threshold_m']),
                                   rule, f'precheck-{suite}-c{cell_index}-r{r}', -1)
            rec = run['record']
            s = stored[(cell['cell_id'], seed)]
            diff = {k: [rec[k], s[k]] for k in COMPARE if rec[k] != s[k] or type(rec[k]) is not type(s[k])}
            out.append({'cell_id': cell['cell_id'], 'r': r, 'seed': seed, 'identical': not diff, 'diff': diff,
                        'J_repr': [repr(rec['J']), repr(s['J'])], 'wall_s': rec['wall_s'],
                        'gate_valid_rows': rec['gate_valid_rows'], 'gate_fix_rows': rec['gate_fix_rows'],
                        'fixes_attempted': rec['fixes_attempted'], 'identity_residual': rec['identity_residual'],
                        'pre_gate_reward': rec['pre_gate_reward'], 'dropped_reward': rec['dropped_reward'],
                        'invalid_segment_reward': rec['invalid_segment_reward'],
                        'min_segment_duration_s': rec['min_segment_duration_s'], 'gate_stats': rec['gate_stats'],
                        'kept_plus_pre_equals_J': abs(rec['kept_reward'] + rec['pre_gate_reward'] - rec['J']) <= 1e-9})
        except Exception as exc:                                                       # recorded
            out.append({'cell_id': cell['cell_id'], 'r': r, 'seed': seed, 'identical': False,
                        'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
    return out


def reproduce(suite, workers, realizations=None):
    import multiprocessing as mp
    rt = runtime(STUDY_MAIN)
    tasks = [(suite, c['cell_index'], realizations) for c in rt.cells]
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        rows = [row for part in pool.imap(_reproduce_cell, tasks) for row in part]
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r.get('identical')]
    positive = [r for r in rows if r['cell_id'].endswith('__positive') and 'wall_s' in r]
    negative = [r for r in rows if r['cell_id'].endswith('__no_human') and 'wall_s' in r]
    return {'suite': suite, 'episodes': len(rows), 'identical': len(ok), 'all_identical': len(ok) == len(rows) > 0,
            'segment_identity_all': all(r.get('identity_residual') is not None and abs(r['identity_residual']) <= 1e-9
                                        for r in rows),
            'pre_gate_reward_all_zero': all(r.get('pre_gate_reward') == 0.0 for r in rows),
            'kept_plus_pre_equals_J_all': all(r.get('kept_plus_pre_equals_J') for r in rows),
            'gate_fix_rows_equal_fixes_all': all(r.get('gate_fix_rows') == r.get('fixes_attempted') for r in rows),
            'min_segment_duration_s': min((r['min_segment_duration_s'] for r in rows
                                           if r.get('min_segment_duration_s') is not None), default=None),
            'gate_rows_total': sum(r.get('gate_valid_rows', 0) for r in rows),
            'gate_rows_per_positive_episode': float(np.mean([r['gate_valid_rows'] for r in positive])) if positive else None,
            'gate_rows_per_no_human_episode': float(np.mean([r['gate_valid_rows'] for r in negative])) if negative else None,
            'resamples_total': sum(r.get('gate_stats', {}).get('resamples', 0) for r in rows),
            'no_anchor_skips_total': sum(r.get('gate_stats', {}).get('no_anchor_skips', 0) for r in rows),
            'wall_s': wall, 'workers': workers,
            'episode_wall_s_mean': {'positive': float(np.mean([r['wall_s'] for r in positive])) if positive else None,
                                    'no_human': float(np.mean([r['wall_s'] for r in negative])) if negative else None},
            'rows': rows}


# ------------------------------------------------------------------ smoke --------------------------------------------

SMOKE = {'budget_gate_decisions': 192, 'grid': [0, 192], 'transitions_per_update': 64, 'curve_suite': 'e1_development',
         'final_suite': 'e1_development', 'curve_realizations': [0], 'final_realizations': [0]}


def smoke(tag=''):
    out = HERE / f'smoke{tag}'
    if out.exists():
        raise SystemExit(f'{out} exists')
    torch.set_num_threads(1)
    rt = runtime(STUDY_DEV)
    cfg = config(SMOKE_BLOCK, STUDY_DEV, **SMOKE)
    t0 = time.perf_counter()
    summary, final = train_gate(cfg, rt, out / 'run')
    wall = time.perf_counter() - t0
    events = [json.loads(x) for x in (out / 'run' / 'metrics.jsonl').read_text().splitlines()]
    kinds = {}
    for e in events:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    eps = [e for e in events if e['kind'] == 'episode']
    seeds_ok = all(GATE_NAMESPACES['gate_development']['start'] + SMOKE_BLOCK * STRIDE <= e['seed'] <
                   GATE_NAMESPACES['gate_development']['start'] + (SMOKE_BLOCK + 1) * STRIDE for e in eps)
    # reload the final checkpoint and re-run the first final-evaluation episode: identical row (determinism)
    final_ck = next((out / 'run' / 'checkpoints').glob('final_*.pt'))
    model, payload = gc.load_gate_checkpoint(final_ck)
    cell = next(c for c in rt.cells if c['cell_id'] == final['episodes'][0]['cell_id'])
    first = final['episodes'][0]
    again = run_gate_episode(rt, cell, 'e1_development', first['seed'], gc.NetworkGate(model, deterministic=True),
                             cfg['rule'], 'smoke-reload', -1)['record']
    same = {k: again[k] == first[k] for k in EPISODE_FIELDS + ('gate_fix_rows', 'gate_valid_rows')}
    pos = [e for e in eps if e['cell_id'].endswith('__positive')]
    neg = [e for e in eps if e['cell_id'].endswith('__no_human')]
    upd = [e for e in events if e['kind'] == 'update']
    result = {'wall_s': wall, 'event_counts': kinds, 'status': summary['status'], 'episodes': len(eps),
              'updates': len(upd), 'gate_decisions': summary['gate_decisions'],
              'training_seeds_in_dev_range': seeds_ok,
              'identity_max_abs_residual': max(abs(e['identity_residual']) for e in eps),
              'pre_gate_reward_total': sum(e['pre_gate_reward'] for e in eps),
              'infrastructure_episodes': summary['infrastructure_episodes'],
              'reload_rerun_identical': all(same.values()), 'reload_rerun_fields': same,
              'timing': {'training_episode_wall_s_mean': float(np.mean([e['wall_s'] for e in eps])),
                         'positive_episode_wall_s_mean': float(np.mean([e['wall_s'] for e in pos])) if pos else None,
                         'no_human_episode_wall_s_mean': float(np.mean([e['wall_s'] for e in neg])) if neg else None,
                         'rollout_per_1000_gate_decisions_s': summary['rollout_per_1000_decisions_s'],
                         'gate_rows_per_episode': summary['gate_decisions'] / max(summary['episodes'], 1),
                         'update_wall_s_mean': float(np.mean([u['wall']['update_s'] for u in upd])) if upd else None,
                         'evaluation_wall_s': summary['walls']['evaluation_s'],
                         'evaluation_episodes': sum(e['result']['n'] for e in events if e['kind'] == 'evaluation')
                         + (final['n'] if final else 0),
                         'load_average': summary['load_average']},
              'approx_kl': [u['metrics']['approx_kl'] for u in upd], 'p_fix_after': [u['p_fix_after'] for u in upd]}
    expected = {'arm_start': 1, 'update': 3, 'evaluation': 2, 'final_evaluation': 1, 'arm_end': 1, 'checkpoint': 3}
    result['passed'] = (summary['status'] == 'completed' and all(kinds.get(k) == v for k, v in expected.items())
                        and seeds_ok and result['reload_rerun_identical'] and result['identity_max_abs_residual'] <= 1e-9
                        and summary['gate_decisions'] >= SMOKE['budget_gate_decisions'])
    (out / 'smoke.json').write_text(json.dumps(result, indent=1, default=float))
    return result


# ------------------------------------------------------------------ precheck -----------------------------------------

def precheck(workers, tag=''):
    out_json = HERE / f'precheck{tag}.json'
    if out_json.exists():
        raise SystemExit(f'{out_json} exists')
    t0 = time.perf_counter()
    result = {'utc': p4.utc(), 'plan_sha256': PLAN_SHA256, 'code_sha256': code_hashes(), 'tag': tag,
              'load_average_start': list(os.getloadavg())}
    tree = p4.live_copy_tree()
    rule = e1_baselines.p3_rule()                       # refuses if gnss_policies.py is not the frozen source
    result['a_hashes'] = {'code_tree': tree, 'expected': EXPECTED_TREE, 'tree_ok': tree == EXPECTED_TREE,
                          'rule': rule, 'baselines_sha256_ok': sha(BASELINES) == BASELINES_SHA256}
    import test_gate
    tests = test_gate.run_all()
    result['b_unit_tests'] = {'passed': all(v == 'passed' for v in tests.values()), 'results': tests}
    rep = reproduce('e1_evaluation', workers)
    result['c_reproduction'] = rep
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1, default=float), flush=True)
    result['d_smoke'] = smoke(tag)
    result['a_hashes']['code_tree_after'] = p4.live_copy_tree()
    result['passed'] = bool(result['a_hashes']['tree_ok'] and result['a_hashes']['baselines_sha256_ok']
                            and result['a_hashes']['code_tree_after'] == EXPECTED_TREE
                            and result['b_unit_tests']['passed'] and rep['all_identical']
                            and rep['segment_identity_all'] and rep['pre_gate_reward_all_zero']
                            and rep['gate_fix_rows_equal_fixes_all'] and result['d_smoke']['passed'])
    result['wall_s'] = time.perf_counter() - t0
    result['load_average_end'] = list(os.getloadavg())
    out_json.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({'passed': result['passed'], 'reproduced': f'{rep["identical"]}/{rep["episodes"]}',
                      'unit_tests': result['b_unit_tests']['passed'], 'smoke': result['d_smoke']['passed'],
                      'wall_s': result['wall_s']}, indent=1))


# ------------------------------------------------------------------ launch -------------------------------------------

def plan_hashed():
    f = HERE / 'PLAN.sha256'
    return PLAN_SHA256 is not None and f.exists() and f.read_text().split()[0] == PLAN_SHA256


def run_one(task):
    block, out = task
    torch.set_num_threads(1)
    rec = {'block': block, 'status': 'started', 'plan_sha256': PLAN_SHA256, 'stride': p4.BLOCK_STRIDE}
    t0 = time.perf_counter()
    try:
        rt = runtime(STUDY_MAIN)
        cfg = config(block)
        run_dir = Path(out) / f'gate__b{block}'
        if run_dir.exists():
            raise RuntimeError(f'{run_dir} exists: no overwrite, no resume')
        space = rt.namespaces[STUDY_MAIN.training_namespace]
        lo, hi = space['start'] + block * STRIDE, space['start'] + (block + 1) * STRIDE
        if not (space['start'] <= lo and hi <= space['end']):
            raise RuntimeError('training seeds would leave the namespace')
        rec.update(config=cfg, training_seed_range=[lo, hi])
        summary, final = train_gate(cfg, rt, run_dir / 'run')
        rec.update(status=summary['status'], summary=summary, final_evaluation=final)
    except Exception as exc:                                                       # recorded, never retried
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def _require_launchable():
    if not plan_hashed():
        raise SystemExit('PLAN.md is not hashed (PLAN.sha256 missing or different): no main run before the plan')
    pre = json.loads((HERE / 'precheck.json').read_text()) if (HERE / 'precheck.json').exists() else {}
    if not (pre.get('passed') and pre.get('plan_sha256') == PLAN_SHA256):
        raise SystemExit('precheck.json has not passed for this plan')
    now = code_hashes()
    if any(pre['code_sha256'].get(k) != v for k, v in now.items() if k != 'gate_analysis.py'):
        raise SystemExit('runner.py / gate_core.py / test_gate.py changed since the precheck')


def launch(workers):
    _require_launchable()
    runs, records = HERE / 'runs', HERE / 'records'
    runs.mkdir(exist_ok=True)
    records.mkdir(exist_ok=True)
    utc = time.strftime('%Y-%m-%dT%H%M%SZ', time.gmtime())
    with open(HERE / 'launches.jsonl', 'a') as f:
        f.write(json.dumps({'utc': utc, 'plan_sha256': PLAN_SHA256, 'workers': workers, 'code_sha256': code_hashes(),
                            'idea_dq_code_tree': p4.live_copy_tree()}) + '\n')
    tasks = []
    for b in BLOCKS:
        name = f'gate__b{b}'
        if (records / f'{name}.json').exists():
            continue
        if (runs / name).exists():                           # killed by a crash: moved aside, restarted from scratch
            import shutil
            (HERE / 'interrupted' / utc).mkdir(parents=True, exist_ok=True)
            shutil.move(str(runs / name), str(HERE / 'interrupted' / utc / name))
        tasks.append((b, str(runs)))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(run_one, tasks):
            (records / f'gate__b{rec["block"]}.json').write_text(json.dumps(rec, indent=1, default=float))
            print('gate', rec['block'], rec['status'], round(rec['wall_s']), flush=True)


# ------------------------------------------------------------------ stress map (secondary) -----------------------------

def stress_parts():
    import dq_freeze_a as fa
    rt = runtime(STUDY_MAIN)
    cells = fa.stress_cells(rt)
    payload = json.loads(STRESS_RULE.read_text())
    if payload.get('status') != 'complete' or len(payload['episodes']) != 320:
        raise RuntimeError('the stored rule stress job is not the complete 320-episode job')
    stored = {(e['cell_id'], e['realization']): e for e in payload['episodes']}
    return fa, rt, cells, stored


STRESS_FIELDS = ('seed', 'crn_sha256', 'J', 'tp_timely', 'fp_spatial', 'return_ok', 'fixes_attempted',
                 'termination_reason')


def _stress_cell(task):
    """One stress cell: the rule-imitating gate (kind 'rule') or a final checkpoint (kind 'final') on its episodes."""
    kind, cell_index, realizations, ck = task
    torch.set_num_threads(1)
    fa, rt, cells, stored = stress_parts()
    rule = e1_baselines.p3_rule()
    cell = cells[cell_index]
    if kind == 'rule':
        gate = gc.RuleGate(rule['parameters']['sigma_threshold_m'])
    else:
        model, _ = gc.load_gate_checkpoint(ck)
        gate = gc.NetworkGate(model, deterministic=True)
    out = []
    for r in realizations:
        seed = fa.stress_seed(cell, r)
        try:
            rec = run_gate_episode(rt, cell, fa.NAMESPACE, seed, gate, rule, f'stress-{kind}-c{cell_index}-r{r}', -1)['record']
            s = stored[(cell['cell_id'], r)]
            row = {'cell_id': cell['cell_id'], 'cell_index': cell_index, 'd600': cell['d600_rms_m'],
                   'budget': cell['budget_name'], 'presence': cell['presence'], 'weight': cell['weight'],
                   'realization': r, **{k: rec[k] for k in STRESS_FIELDS}, 'rule_J': s['J'], 'd_J': rec['J'] - s['J'],
                   'gate_valid_rows': rec['gate_valid_rows'], 'gate_fix_rows': rec['gate_fix_rows'],
                   'wall_s': rec['wall_s'], 'identity_residual': rec['identity_residual']}
            if kind == 'rule':
                row['diff'] = {k: [rec[k], s[k]] for k in STRESS_FIELDS if rec[k] != s[k]}
                row['identical'] = not row['diff']
            elif rec['crn_sha256'] != s['crn_sha256']:
                raise RuntimeError('crn_sha256 differs from the stored rule stress row')
            out.append(row)
        except Exception as exc:
            out.append({'cell_id': cell['cell_id'], 'realization': r, 'identical': False,
                        'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]})
    return out


def precheck_stress(workers, per_cell=None, tag=''):
    import multiprocessing as mp
    out_json = HERE / f'precheck_stress{tag}.json'
    if out_json.exists():
        raise SystemExit(f'{out_json} exists')
    fa, rt, cells, stored = stress_parts()
    reals = list(fa.STRESS_REALIZATIONS)[:per_cell] if per_cell else list(fa.STRESS_REALIZATIONS)
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        rows = [x for part in pool.imap(_stress_cell, [('rule', c['cell_index'], reals, None) for c in cells])
                for x in part]
    res = {'utc': p4.utc(), 'plan_sha256': PLAN_SHA256, 'code_sha256': code_hashes(), 'episodes': len(rows),
           'identical': sum(1 for r in rows if r.get('identical')), 'per_cell': len(reals),
           'full': len(reals) == len(fa.STRESS_REALIZATIONS), 'wall_s': time.perf_counter() - t0, 'workers': workers,
           'segment_identity_all': all(abs(r.get('identity_residual', 1.)) <= 1e-9 for r in rows), 'rows': rows}
    res['passed'] = res['identical'] == res['episodes'] > 0 and res['segment_identity_all']
    out_json.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))


def evaluate_stress(workers):
    import multiprocessing as mp
    _require_launchable()
    pre = json.loads((HERE / 'precheck_stress.json').read_text()) if (HERE / 'precheck_stress.json').exists() else {}
    if not (pre.get('passed') and pre.get('full') and pre.get('plan_sha256') == PLAN_SHA256):
        raise SystemExit('precheck_stress.json (all 320 stored rule rows reproduced) has not passed for this plan')
    fa, rt, cells, stored = stress_parts()
    out = HERE / 'stress_records'
    out.mkdir(exist_ok=True)
    for f in sorted((HERE / 'records').glob('gate__b*.json')):
        rec = json.loads(f.read_text())
        dest = out / f.name
        if rec.get('status') != 'completed' or dest.exists():
            continue
        final = [c for c in rec['summary']['checkpoints'] if c['final']][0]
        ck = HERE / 'runs' / f'gate__b{rec["block"]}' / 'run' / final['path']
        if file_sha256(ck) != final['sha256']:
            raise SystemExit(f'{ck} is not the recorded final checkpoint')
        t0 = time.perf_counter()
        with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
            rows = [x for part in pool.imap(_stress_cell, [('final', c['cell_index'], list(fa.STRESS_REALIZATIONS),
                                                            str(ck)) for c in cells]) for x in part]
        dest.write_text(json.dumps({'block': rec['block'], 'final_sha256': final['sha256'], 'plan_sha256': PLAN_SHA256,
                                    'episodes': rows, 'errors': sum(1 for r in rows if 'error' in r),
                                    'wall_s': time.perf_counter() - t0}, indent=1, default=float))
        print('stress', rec['block'], len(rows), round(time.perf_counter() - t0), flush=True)


# ------------------------------------------------------------------ optional development pilot ----------------------

def _pilot_one(task):
    lr, block, budget, out, quick = task
    torch.set_num_threads(1)
    rec = {'lr': lr, 'block': block, 'status': 'started', 'quick': quick}
    t0 = time.perf_counter()
    try:
        rt = runtime(STUDY_DEV)
        grid = [0, budget] if quick else sorted({0, budget // 2, budget})
        reals = [0] if quick else None             # quick: e1_development r0 only (8 episodes per read-out)
        cfg = config(block, STUDY_DEV, budget_gate_decisions=budget, grid=grid, final_suite='e1_development',
                     curve_realizations=reals, final_realizations=reals, ppo={'learning_rate': lr})
        summary, final = train_gate(cfg, rt, Path(out) / f'lr{lr:g}__b{block}' / 'run')
        rec.update(status=summary['status'], summary=summary)
    except Exception as exc:
        rec.update(status='failure', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-4000:])
    rec['wall_s'] = time.perf_counter() - t0
    return rec


def pilot(lrs, blocks, budget, workers, quick=False):
    """Development seeds (gate_development) and the e1_development suite only; never e1_evaluation."""
    import multiprocessing as mp
    out = HERE / ('pilot_quick' if quick else 'pilot')
    out.mkdir(exist_ok=True)
    tasks = [(float(lr), int(b), int(budget), str(out), bool(quick)) for lr in lrs for b in blocks
             if not (out / f'lr{float(lr):g}__b{b}').exists()]
    if any(b not in PILOT_BLOCKS for b in blocks):
        raise SystemExit(f'pilot blocks are {PILOT_BLOCKS}')
    with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(_pilot_one, tasks):
            (out / f'lr{rec["lr"]:g}__b{rec["block"]}.json').write_text(json.dumps(rec, indent=1, default=float))
            print('pilot', rec['lr'], rec['block'], rec['status'], round(rec['wall_s']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('test')
    pc = sub.add_parser('precheck')
    pc.add_argument('--workers', type=int, default=8)
    pc.add_argument('--tag', default='')
    sm = sub.add_parser('smoke')
    sm.add_argument('--tag', default='_manual')
    la = sub.add_parser('launch')
    la.add_argument('--workers', type=int, default=12)
    ps = sub.add_parser('precheck-stress')
    ps.add_argument('--workers', type=int, default=8)
    ps.add_argument('--per-cell', type=int, default=None)
    ps.add_argument('--tag', default='')
    es = sub.add_parser('evaluate-stress')
    es.add_argument('--workers', type=int, default=12)
    pl = sub.add_parser('pilot')
    pl.add_argument('--lrs', type=float, nargs='+', required=True)
    pl.add_argument('--blocks', type=int, nargs='+', required=True)
    pl.add_argument('--budget', type=int, default=6400)
    pl.add_argument('--workers', type=int, default=4)
    pl.add_argument('--quick', action='store_true', help='grid [0, budget], read-outs on e1_development r0 only')
    a = ap.parse_args()
    if a.cmd == 'test':
        import test_gate
        res = test_gate.run_all()
        print(json.dumps(res, indent=1))
        raise SystemExit(0 if all(v == 'passed' for v in res.values()) else 1)
    elif a.cmd == 'precheck':
        precheck(a.workers, a.tag)
    elif a.cmd == 'smoke':
        print(json.dumps(smoke(a.tag), indent=1, default=float))
    elif a.cmd == 'launch':
        launch(a.workers)
    elif a.cmd == 'precheck-stress':
        precheck_stress(a.workers, a.per_cell, a.tag)
    elif a.cmd == 'evaluate-stress':
        evaluate_stress(a.workers)
    elif a.cmd == 'pilot':
        pilot(a.lrs, a.blocks, a.budget, a.workers, a.quick)
