"""Fast cached-view mission episodes (ADR 006): the executive, policies, perception, tracker, reserve and evaluator of
scripts/run_mission.py, with CachedViewBackend instead of Gazebo/PX4.

  .venv/bin/python scripts/fast_mission.py --protocol <mission protocol.json> --bank <bank dir> --time-model <time_model.json> \
      --out output/training/fast_runs/<name> [--episodes 1] [--policy coverage_baseline|information_cost] [--log jsonl|memory|none] \
      [--reobservation-model frame_periodic/1|visit_permutation/2|visit_frame_permutation/3] [--reobservation-seed 0] \
      [--repeated-capture serve|invalid] [--evidence-serving all|first_only]

Inputs
    The protocol supplies scene, variant, vehicle, camera, perception, mission config, tracker, reserve and budgets exactly as
    for a Gazebo run (run_mission.briefing_with_overrides and run_mission.build_mission). Its freeze file is not required:
    fast runs are development tools, not frozen experiments; its sha256 is recorded. The bank must come from the protocol's
    scene manifest and variant, otherwise the run is refused.
Per episode
    A fresh stack (policy, tracker, coverage grid) and a reset backend; truth sampled after every macro and at the end
    (EpisodeAccounting, the evaluator of evaluate_mission_run.py); all frames recorded for sensor-loss accounting. Frames
    at off-graph poses (pad, landing) have invalid RGB/depth in this backend, so sensor-loss counts are not comparable
    with Gazebo for those frames (ADR 006).
Observation serving (CachedViewConfig; module docstring of cached_view_backend)
    --reobservation-model and --reobservation-seed (episode k uses seed + k) choose the repeat and frame served on each
    visit, --repeated-capture whether an image already served in the episode is served again, and --evidence-serving
    whether every frame at a node serves the node's first served capture (first_only, r6 factor D 'deterministic evidence')
    or the re-observation model decides (all). The defaults reproduce earlier runs.
Outputs (new --out directory)
    episode_NNN/evaluation.json (EpisodeResult, matching, backend statistics, timing, backend_config) and, with --log jsonl,
    the executive's public logs; summary.json with inputs, per-episode results, the observation-serving settings
    ('reobservation': model, seed_base, episode_seed, repeated_capture, evidence_serving) and throughput (wall per episode
    and per decision).
Finite-shot RL policies (protocol mission.rl.actor_shots, run_mission.build_rl_policy)
    build_rl_policy seeds the quantum actor head's shot stream with actor_shots.seed. A fast episode re-seeds it with
    episode_shot_seed = SeedSequence([actor_shots.seed, reobservation_seed of the episode]).generate_state(1, uint32)[0],
    so repeated episodes of one protocol get independent shot noise (ADR 012 addendum 1, D3); the seed is recorded per
    episode ('actor_shot_seed', None without finite shots). A Gazebo run keeps actor_shots.seed itself (one flight per
    protocol).
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from aerial_qrl.mission.backend import LifecycleState
from aerial_qrl.mission.cached_view_backend import EVIDENCE_SERVING, CachedViewBackend, CachedViewConfig, make_cached_perceive_fn
from aerial_qrl.mission.contracts import (SCHEMA_VERSION, TERMINAL_REASONS, EvaluatorSpec, RngStreams, ScenarioSpec, SemiMarkovDiscount,
                                          TimingPolicy)
from aerial_qrl.mission.evaluation import EpisodeAccounting
from aerial_qrl.mission.executive import JsonlLog, MemoryLog, MissionExecutive
from aerial_qrl.mission.observation_bank import ObservationBank, open_banks  # noqa: F401 (ObservationBank kept for callers)
from aerial_qrl.mission.perception import CameraModel, LocalizationConfig
from aerial_qrl.mission.time_model import TimeModel


class NullLog:
    """Discards records (throughput runs); the executive still checks every record with the leakage guard."""

    def write(self, relative, record):
        pass


def load_context(protocol_path, bank, policy=None, budget_overrides=None):
    """Build the mission context. `budget_overrides` merges into the protocol's own budget overrides BEFORE the
    briefing is built, so a changed mission_time_s reaches the briefing, the ReturnReserve, the executive
    deadline, the observation's remaining_time_s and the action mask together. Overriding the budget anywhere
    downstream would change the score while the episode still flew under the protocol's dynamics."""
    from run_mission import briefing_with_overrides, build_mission, sha256_file, sha256_json
    from run_px4_integration_episode import vehicle_inputs
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes)
    if policy is not None:
        protocol = copy.deepcopy(protocol)
        protocol['mission']['policy'] = policy
    if budget_overrides:
        protocol = copy.deepcopy(protocol)
        protocol['mission']['budget_overrides'] = dict(protocol['mission']['budget_overrides'], **budget_overrides)
    inputs = protocol['inputs']
    scene = ROOT / inputs['scene_dir']
    manifest_sha = sha256_file(scene / 'manifest.json')
    if manifest_sha != inputs['scene_manifest_sha256']:
        raise SystemExit('scene manifest differs from the protocol')
    if bank.manifest.get('scene_manifest_sha256') != manifest_sha or bank.manifest.get('variant') != inputs['variant']:
        raise SystemExit('the bank was collected for another scene manifest or variant')
    scene_manifest = json.loads((scene / 'manifest.json').read_text())
    variant = scene_manifest['variants'][inputs['variant']]
    public = json.loads((scene / variant['public']).read_text())
    obstacle_data = json.loads((scene / 'obstacle_map.json').read_text())
    if public['obstacle_map'] != obstacle_data:
        raise SystemExit('public scenario obstacle map differs from obstacle_map.json')
    briefing, budget_record = briefing_with_overrides(public, protocol['mission']['budget_overrides'])
    vehicle_manifest, _ = vehicle_inputs(protocol)
    evaluator = EvaluatorSpec.from_dict(json.loads((scene / variant['private']).read_text())['evaluator_spec'])
    block = protocol['scenario']
    scenario = ScenarioSpec(SCHEMA_VERSION, inputs['scenario_id'], block['split'], block['layout_family'], manifest_sha, 'cached_view',
                            bank.manifest_sha256, sha256_json(protocol['mission']), briefing, SemiMarkovDiscount.from_dict(block['discount']),
                            TimingPolicy.from_dict(block['timing']), RngStreams.from_dict(block['rng']))
    from run_mission import task_view_graph
    stack = build_mission(protocol, briefing, obstacle_data, vehicle_manifest)
    view_graph, task_record = task_view_graph(protocol, briefing, stack)
    identity = check_bank_identity(bank, protocol, stack)
    return {'protocol': protocol, 'protocol_path': str(Path(protocol_path).resolve()), 'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
            'briefing': briefing, 'budget_record': budget_record, 'obstacle_data': obstacle_data, 'vehicle_manifest': vehicle_manifest,
            'evaluator': evaluator, 'scenario': scenario, 'build_mission': build_mission, 'view_graph': view_graph, 'task': task_record,
            'bank_identity': identity}


def check_bank_identity(bank, protocol, stack):
    """Every bank part must have been collected on the protocol's view graph and camera mount (fast-env audit B9). Parts
    whose manifest predates these fields are listed as not checked; a mismatch raises SystemExit."""
    parts = getattr(bank, 'banks', None) or [bank]
    expected_graph = (protocol.get('task') or {}).get('view_graph_sha256')
    mount = stack['mount'].to_dict()
    record = []
    for part in parts:
        manifest = part.manifest
        graph = (manifest.get('graph') or {}).get('sha256')
        entry = {'bank_manifest_sha256': part.manifest_sha256, 'graph': 'not_checked', 'camera_mount': 'not_checked'}
        if expected_graph is not None and graph is not None:
            if graph != expected_graph:
                raise SystemExit(f'bank {part.manifest_sha256[:12]} was collected on view graph {graph[:12]}, the protocol names {expected_graph[:12]}')
            entry['graph'] = 'equal'
        bank_mount = manifest.get('camera_mount')
        if bank_mount is not None:
            for key in ('translation_flu', 'rpy_rad', 'sensor_offset_link'):
                if key in bank_mount and any(abs(a - b) > 1e-9 for a, b in zip(bank_mount[key], mount[key])):
                    raise SystemExit(f'bank {part.manifest_sha256[:12]} camera mount {key} differs from the protocol vehicle')
            entry['camera_mount'] = 'equal'
        record.append(entry)
    return record


def run_fast_episode(context, bank, time_model, episode_id, log=None, executive_kwargs=None, backend_config=None,
                     policy_factory=None, reports_for_matching=None):
    """One episode; returns outcome, EpisodeAccount, decision records, backend statistics and timing.

    policy_factory: None uses the protocol policy built by build_mission; otherwise policy_factory(briefing, obstacle_map,
    config, reserve) builds the episode's policy from the same public stack (e.g. scripts/train_smdp_ppo.py).
    reports_for_matching: None scores outcome.reports, which is right for flood-sar-mission/1 where the public report IS
    the submitted coordinate. Under flood-sar-village-gnss/1 the public record is a local-frame receipt that
    evaluation.match_reports refuses by design, so the GNSS runner passes a callable returning the frozen GLOBAL
    coordinates of the private ledger (GeoreferenceService.global_report_positions)."""
    protocol, briefing = context['protocol'], context['briefing']
    wall0 = time.perf_counter()
    stack = context['build_mission'](protocol, briefing, context['obstacle_data'], context['vehicle_manifest'])
    policy = stack['policy'] if policy_factory is None else policy_factory(briefing, stack['obstacle_map'], stack['config'], stack['reserve'])
    shot_seed = reseed_actor_shots(policy, backend_config)
    perception = protocol['perception']
    rgb_camera = CameraModel.from_sensor_model(briefing.sensor_model, 'rgb', stack['mount'])
    depth_camera = CameraModel.from_sensor_model(briefing.sensor_model, 'depth', stack['mount']) if perception['use_depth'] else None
    depth_range = (briefing.sensor_model.depth_min_m, briefing.sensor_model.depth_max_m)
    backend = CachedViewBackend(bank, time_model, backend_config)
    perceive = make_cached_perceive_fn(backend, bank, rgb_camera, stack['ground'], depth_camera, depth_range,
                                       LocalizationConfig.from_dict(perception['localization']))
    accounting = EpisodeAccounting(episode_id, context['evaluator'], briefing.landing_zone, stack['obstacle_map'])
    backend.reset(context['scenario'], context['evaluator'])
    setup_s = time.perf_counter() - wall0
    kwargs = dict({'view_graph': context.get('view_graph')}, **(executive_kwargs or {}))
    executive = MissionExecutive(backend, briefing, stack['obstacle_map'], stack['config'], perceive, stack['tracker'], stack['grid'],
                                 stack['reserve'], policy, NullLog() if log is None else log,
                                 after_macro=lambda info: accounting.record_truth(backend.evaluator_truth()), **kwargs)
    run0 = time.perf_counter()
    outcome = executive.run()
    run_s = time.perf_counter() - run0
    if backend.state == LifecycleState.ACTIVE:
        backend.end_episode()
    accounting.record_truth(backend.evaluator_truth())
    for frame in backend.frames:
        accounting.record_frame(frame)
    completed = outcome.phase.value == 'COMPLETE' or outcome.termination_reason in TERMINAL_REASONS
    wall = time.perf_counter() - wall0
    reports = outcome.reports if reports_for_matching is None else reports_for_matching()
    account = accounting.build(outcome.termination_reason, reports, outcome.elapsed_s or 0., wall,
                               'completed' if completed else 'interrupted')
    backend.stop()
    return {'outcome': outcome, 'account': account, 'records': executive.records, 'stats': backend.stats, 'policy': policy.name,
            'wall_s': wall, 'setup_wall_s': setup_s, 'executive_wall_s': run_s, 'actor_shot_seed': shot_seed}


def reseed_actor_shots(policy, backend_config):
    """Per-episode shot seed of a policy whose quantum actor head runs with finite shots (module docstring); None otherwise."""
    model = getattr(policy, 'model', None)
    shots = model.actor_shots() if hasattr(model, 'actor_shots') else None
    if shots is None:
        return None
    reobservation_seed = (backend_config or CachedViewConfig()).reobservation_seed
    seed = int(np.random.SeedSequence([shots['seed'], reobservation_seed]).generate_state(1)[0])
    model.actor.set_shots(shots['shots'], seed)
    return seed


def report_classes(outcome, account):
    """EVALUATOR side: per submitted report, correct (matched target) / duplicate / false (evaluation.MatchResult)."""
    matching = account.matching
    by_report = {m.report_index: m for m in matching.matches}
    classes = []
    for i, report in enumerate(outcome.reports):
        entry = {'report_index': i, 'track_id': report.track_id, 'sim_time_s': report.sim_time_s}
        if i in by_report:
            entry.update(result='correct', target_id=by_report[i].target_id, error_xy_m=by_report[i].error_xy_m)
        else:
            entry['result'] = 'duplicate' if i in matching.duplicate_report_indices else 'false'
        classes.append(entry)
    return classes


def episode_summary(result):
    account, outcome = result['account'], result['outcome']
    valid = sum(r.valid for r in result['records'])
    return {'episode_result': account.to_episode_result().to_dict(), 'policy': result['policy'], 'phase': outcome.phase.value,
            'decisions': outcome.decisions, 'valid_decisions': valid, 'rejections': outcome.rejections,
            'reports_evaluator': report_classes(outcome, account), 'unresolved_target_ids': list(account.matching.unresolved_target_ids),
            'backend_stats': result['stats'], 'wall_s': result['wall_s'], 'setup_wall_s': result['setup_wall_s'],
            'executive_wall_s': result['executive_wall_s'],
            'executive_wall_per_decision_s': result['executive_wall_s'] / valid if valid else None,
            'sim_s': outcome.elapsed_s, 'actor_shot_seed': result.get('actor_shot_seed')}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--bank', required=True, nargs='+', help='one or more bank directories of the same scene and variant (MultiBank)')
    parser.add_argument('--time-model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--policy', choices=['coverage_baseline', 'coverage_first_sight', 'coverage_recheck', 'information_cost'])
    parser.add_argument('--log', choices=['jsonl', 'memory', 'none'], default='jsonl')
    parser.add_argument('--verify-bank', action='store_true')
    parser.add_argument('--reobservation-model', choices=['frame_periodic/1', 'visit_permutation/2', 'visit_frame_permutation/3'], default='frame_periodic/1',
                        help='CachedViewConfig.reobservation_model (ADR 006 revision 2); the default keeps earlier runs reproducible')
    parser.add_argument('--repeated-capture', choices=['serve', 'invalid'], default='serve',
                        help='CachedViewConfig.repeated_capture (fast-env audit B1): invalid = an image already served in the episode '
                             'is not served again as new evidence; the default keeps earlier runs reproducible')
    parser.add_argument('--reobservation-seed', type=int, default=0,
                        help='episode k uses reobservation_seed + k (visit_permutation/2 and visit_frame_permutation/3 with evidence '
                             'serving all; never an observation)')
    parser.add_argument('--evidence-serving', choices=EVIDENCE_SERVING, default='all',
                        help='CachedViewConfig.evidence_serving (r6 factor D): first_only = every frame at a node serves the '
                             "node's first served capture; the default keeps earlier runs reproducible")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from code_version import file_hashes, tree_hash
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    load0 = time.perf_counter()
    bank = open_banks(args.bank, verify_files=args.verify_bank)
    time_model = TimeModel.from_file(args.time_model)
    context = load_context(args.protocol, bank, args.policy)
    load_s = time.perf_counter() - load0
    episodes = []
    for k in range(args.episodes):
        directory = out / f'episode_{k:03d}'
        directory.mkdir()
        log = JsonlLog(directory) if args.log == 'jsonl' else MemoryLog() if args.log == 'memory' else None
        backend_config = CachedViewConfig(reobservation_model=args.reobservation_model, reobservation_seed=args.reobservation_seed + k,
                                          repeated_capture=args.repeated_capture, evidence_serving=args.evidence_serving)
        summary = episode_summary(run_fast_episode(context, bank, time_model, f'fast-{k:03d}', log, backend_config=backend_config))
        summary['backend_config'] = backend_config.to_dict()
        with open(directory / 'evaluation.json', 'x') as f:
            f.write(json.dumps(summary, indent=2, default=str) + '\n')
        episodes.append(summary)
        print(json.dumps({'episode': k, 'termination': summary['episode_result']['termination_reason'],
                          'correct': summary['episode_result']['correct_distinct_reports'], 'false': summary['episode_result']['false_reports'],
                          'decisions': summary['valid_decisions'], 'sim_s': round(summary['sim_s'] or 0., 1), 'wall_s': round(summary['wall_s'], 3),
                          'off_node_frames': summary['backend_stats']['off_node_frames'], 'flags': summary['backend_stats']['time_model_flags']}), flush=True)
    walls = [e['executive_wall_s'] for e in episodes]
    per_decision = [e['executive_wall_per_decision_s'] for e in episodes if e['executive_wall_per_decision_s']]
    record = {'tool': 'fast_mission/1', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': sys.argv if argv is None else argv,
              'code_tree_sha256': tree_hash(file_hashes()), 'protocol': context['protocol_path'], 'protocol_sha256': context['protocol_sha256'],
              'policy': context['protocol']['mission']['policy'], 'bank': bank.describe(), 'time_model': time_model.describe(),
              'log_mode': args.log, 'load_wall_s': load_s, 'episodes': episodes,
              'reobservation': {'model': args.reobservation_model, 'seed_base': args.reobservation_seed, 'episode_seed': 'seed_base + k',
                                'repeated_capture': args.repeated_capture, 'evidence_serving': args.evidence_serving},
              'bank_identity': context['bank_identity'],
              'throughput': {'episodes': len(episodes), 'median_executive_wall_s': statistics.median(walls),
                             'median_executive_wall_per_decision_s': statistics.median(per_decision) if per_decision else None,
                             'median_sim_s': statistics.median([e['sim_s'] or 0. for e in episodes]),
                             'note': 'single process, sequential episodes; excludes bank/protocol loading (load_wall_s)'}}
    with open(out / 'summary.json', 'x') as f:
        f.write(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({'out': str(out), 'throughput': record['throughput']}, indent=1))


if __name__ == '__main__':
    main()
