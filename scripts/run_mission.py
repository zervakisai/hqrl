"""Run one frozen public autonomous mission (segment-mission protocol) on Gazebo/PX4 into a fresh run directory.

From the project root with the ML venv:
  .venv/bin/python scripts/run_mission.py --protocol output/mission/<dir>/protocol.json [--run-id RUN]

Refusals  The protocol differs from its last protocol_freeze.sha256 entry (a draft without a freeze file is refused);
          the scene manifest, variant world/public/private files or obstacle map differ from the protocol/manifest; the
          vehicle manifest or model, PX4 binary or detector weights differ; the run directory exists.
World     worlds/<variant>.sdf is copied to <run>/world/<world_name>.sdf (PX4 requires world name == file stem) and hashed.
          GZ resource paths = scene manifest resource_paths + <scene>/models + vehicle resource paths (backend vehicle dict).
Channels  Public (public/): briefing with the recorded protocol budget overrides, obstacle map, mission config, frames,
          perception (detector results, candidates, rejections), tracks, observations, actions, transitions, outcome.
          Private (evaluator/): the evaluator file is read only for backend.reset and copied here; EvaluatorTruth is sampled
          after reset, after every macro (runner hook, result never returned to the executive) and at the end.
Ground    Public height model: top of the highest terrain box, or of a terrain-like obstacle box (protocol kinds), whose XY
          footprint contains the point; outside every box mission.ground_model.outside_height_m. Water surfaces are visual
          only and excluded. Height steps can stop the ground-ray fixed point near edges (perception rejects, never repairs).
          Planning (footprints, views, coverage) uses the flat plane mission.config.ground_z_m (declared limit).
Clock     Simulator seconds from the backend; wall time only in logs, status and decision latency.
Failure   Safety LAND while the backend is ACTIVE and the vehicle has not landed; end_episode; backend.stop and detector
          close in finally; status.json, errors.json and checksums.json are always written. KeyboardInterrupt -> interrupted.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from xml.etree import ElementTree

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from aerial_qrl.mission.backend import CommandKind, FlightCommand, LifecycleState
from aerial_qrl.mission.contracts import (SCHEMA_VERSION, TERMINAL_REASONS, EvaluatorSpec, MissionBriefing, RngStreams, ScenarioSpec,
                                          SemiMarkovDiscount, TimingPolicy)
from aerial_qrl.mission.executive import POLICIES, CoverageBaselinePolicy, JsonlLog, MissionConfig, MissionExecutive, bind_view_graph
from aerial_qrl.mission.leakage import assert_public_payload
from aerial_qrl.mission.perception import (FROZEN_WEIGHTS_SHA256, CameraModel, CameraMount, DetectorClient, LocalizationConfig,
                                           perceive_sensor_frame, rpy_matrix)
from aerial_qrl.mission.planning import CoverageGrid, ObstacleBox, ReturnReserve, StaticObstacleMap
from aerial_qrl.mission.tracking import Tracker, TrackerConfig

SNAPSHOT = ['src/aerial_qrl/mission', 'scripts/px4_mission_server.py', 'scripts/run_mission.py', 'scripts/evaluate_mission_run.py',
            'scripts/run_px4_integration_episode.py', 'scripts/person_detector_worker.py', 'scripts/recover_mission_run.py',
            'scripts/code_version.py', 'scripts/px4_env.sh']
BUDGET_OVERRIDE_KEYS = ('mission_time_s', 'return_reserve_s')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# --- Pure helpers (unit tested) ------------------------------------------------------------------

def verify_freeze(protocol_path):
    """(bytes, protocol) when the file matches the last freeze-file entry naming it; SystemExit otherwise."""
    protocol_path = Path(protocol_path)
    data, freeze = protocol_path.read_bytes(), protocol_path.parent / 'protocol_freeze.sha256'
    lines = [l for l in freeze.read_text().splitlines() if l.endswith(protocol_path.name)] if freeze.is_file() else []
    if not lines or hashlib.sha256(data).hexdigest() != lines[-1].split()[0]:
        raise SystemExit(f'{protocol_path.name} does not match its last recorded freeze hash; refusing to run')
    return data, json.loads(data)


def copy_world(scene_dir, variant, run_dir, world_name, expected_sha256):
    """Copy worlds/<variant>.sdf to <run_dir>/world/<world_name>.sdf (exclusive create); returns (path, sha256)."""
    data = (Path(scene_dir) / 'worlds' / f'{variant}.sdf').read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f'world {variant}.sdf sha256 {digest} differs from the scene manifest')
    world = ElementTree.fromstring(data).find('world')
    if world is None or world.get('name') != world_name:
        raise ValueError(f"SDF world name {None if world is None else world.get('name')!r} differs from {world_name!r}")
    target = Path(run_dir) / 'world' / f'{world_name}.sdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'xb') as f:
        f.write(data)
    return target, digest


def camera_mount_from_vehicle(manifest, base_link_offset_m, sensor_offset_link_m, sensor='IMX214', tolerance_m=1e-3, source=None):
    """base_link FLU <- camera link from the vehicle manifest (model frame) minus the base_link offset in the model frame.

    Cross-check: camera link + R(pitch) @ sensor offset must equal the manifest's recorded optical origin within tolerance_m."""
    link, pitch = np.asarray(manifest['camera_link_in_model_m'], float), float(manifest['pitch_rad'])
    offset = np.asarray(sensor_offset_link_m, float)
    recorded = (manifest.get('sensor_optical_origin_in_model_m') or {}).get(sensor)
    optical = link + rpy_matrix(0., pitch, 0.) @ offset
    if recorded is None or float(np.linalg.norm(optical - np.asarray(recorded, float))) > tolerance_m:
        raise ValueError(f'{sensor}: sensor offset inconsistent with the manifest optical origin ({optical.tolist()} vs {recorded})')
    translation = link - np.asarray(base_link_offset_m, float)
    return CameraMount(tuple(float(v) for v in translation), (0., pitch, 0.), tuple(float(v) for v in offset), 'specification_only',
                       source or f"vehicle manifest {manifest['model']} camera_link_in_model_m/pitch_rad; SDF sensor pose")


class GroundHeightModel:
    """Public static ground height (ENU z, metres) from obstacle-map terrain and terrain-like obstacle boxes (module docstring)."""

    def __init__(self, obstacle_map_data, terrain_like_kinds=('embankment', 'bridge_abutment'), outside_height_m=0.):
        self.terrain_like_kinds, self.outside_height_m = tuple(terrain_like_kinds), float(outside_height_m)
        entries = [(t['surface_id'], t['min_enu'], t['max_enu']) for t in obstacle_map_data['terrain'] if t.get('collision', True)]
        entries += [(o['obstacle_id'], o['min_enu'], o['max_enu']) for o in obstacle_map_data['obstacles'] if o['kind'] in self.terrain_like_kinds]
        if not entries or not np.isfinite(self.outside_height_m):
            raise ValueError('ground model needs terrain boxes and a finite outside height')
        self.names = [e[0] for e in entries]
        self.lo, self.hi = np.array([e[1] for e in entries], float), np.array([e[2] for e in entries], float)

    def __call__(self, x, y):
        inside = (self.lo[:, 0] <= x) & (x <= self.hi[:, 0]) & (self.lo[:, 1] <= y) & (y <= self.hi[:, 1])
        return float(self.hi[inside, 2].max()) if inside.any() else self.outside_height_m

    def describe(self):
        return {'boxes': [{'name': n, 'min_enu': lo.tolist(), 'max_enu': hi.tolist()} for n, lo, hi in zip(self.names, self.lo, self.hi)],
                'terrain_like_kinds': list(self.terrain_like_kinds), 'outside_height_m': self.outside_height_m,
                'rule': 'max top z of boxes whose closed XY footprint contains the point; water surfaces excluded'}


def obstacle_map_from_json(data, bounds, z_band_m):
    boxes = [ObstacleBox(o['obstacle_id'], tuple(o['min_enu']), tuple(o['max_enu'])) for o in data['obstacles']]
    return StaticObstacleMap(bounds, boxes, z_min_m=z_band_m[0], z_max_m=z_band_m[1])


def briefing_with_overrides(public_scenario, overrides):
    """MissionBriefing with the protocol budget overrides and a record of original and applied budgets."""
    unknown = set(overrides) - set(BUDGET_OVERRIDE_KEYS) - {'provenance', 'justification'}
    if unknown:
        raise ValueError(f'budget_overrides: unsupported keys {sorted(unknown)}')
    briefing = copy.deepcopy(public_scenario['briefing'])
    original = dict(briefing['budgets'])
    briefing['budgets'].update({k: overrides[k] for k in BUDGET_OVERRIDE_KEYS if k in overrides})
    briefing['budgets']['provenance'] = overrides.get('provenance', original['provenance'])
    result = MissionBriefing.from_dict(briefing)
    return result, {'scene_budgets': original, 'applied_budgets': result.budgets.to_dict(), 'overrides': dict(overrides)}


def build_scenario_spec(protocol, briefing, evaluator, scene_manifest_sha256, px4_binary_sha256):
    block = protocol['scenario']
    if evaluator.scenario_id != protocol['inputs']['scenario_id']:
        raise ValueError('evaluator scenario_id differs from protocol inputs.scenario_id')
    return ScenarioSpec(SCHEMA_VERSION, protocol['inputs']['scenario_id'], block['split'], block['layout_family'], scene_manifest_sha256,
                        'gazebo_px4', px4_binary_sha256, sha256_json(protocol['mission']), briefing,
                        SemiMarkovDiscount.from_dict(block['discount']), TimingPolicy.from_dict(block['timing']), RngStreams.from_dict(block['rng']))


RL_POLICY = 'rl_checkpoint'


def build_rl_policy(mission, briefing, obstacle_map, config, reserve):
    """Frozen RL policy for a Gazebo or fast run (policy 'rl_checkpoint'): mission.rl = {checkpoint, checkpoint_sha256,
    obs_encoding} and optionally actor_shots = {shots: integer in [1, actor_heads.MAX_SHOTS], seed: integer >= 0} and
    return_mask_rule (encoding.RETURN_MASK_RULES). The file hash must equal the protocol's; actions are the argmax of the
    masked logits (no sampling, no critic evaluation); the model is loaded with weights_only.
    The RETURN mask rule is the checkpoint's own (train_config.return_mask_rule; absent, i.e. every train config before
    train-smdp-ppo-config/6, means 'always'), so a checkpoint is never flown under a rule it was not trained with by leaving
    the key out. A protocol may declare it, and a declaration differing from the checkpoint's is refused naming both.
    actor_shots (simulated finite-shot execution of a quantum actor head) is refused, before the checkpoint is read, unless
    it is well formed, and afterwards unless the checkpoint's actor head is quantum; it is applied with
    actor.set_shots(shots, seed) on the freshly loaded model before the policy exists, so every call of this function (one
    per episode: fast_mission.run_fast_episode builds the mission stack per episode) restarts the shot stream at call 0.
    Repeated episodes of one protocol therefore share one shot-noise stream (common random numbers: the same observation
    sequence gives the same finite-shot logits in every repeat, and only the re-observation seed makes repeats differ). A
    study that needs independent shot noise per repeat must freeze one protocol per repeat, each with its own
    actor_shots.seed; the seed is recorded in RLMissionPolicy.describe()['actor_shots']."""
    from aerial_qrl.training.actor_heads import MAX_SHOTS
    from aerial_qrl.training.encoding import RETURN_MASK_RULES
    from aerial_qrl.training.rl_policy import RLMissionPolicy
    from aerial_qrl.training.smdp_ppo import QUANTUM_ACTOR_HEADS, file_sha256, load_checkpoint
    rl = mission.get('rl') or {}
    if set(rl) - {'actor_shots', 'return_mask_rule'} != {'checkpoint', 'checkpoint_sha256', 'obs_encoding'}:
        raise ValueError('mission.rl: expected exactly checkpoint, checkpoint_sha256, obs_encoding and optionally actor_shots '
                         'and return_mask_rule')
    declared = rl.get('return_mask_rule')
    if 'return_mask_rule' in rl and declared not in RETURN_MASK_RULES:
        raise ValueError(f'mission.rl.return_mask_rule: expected one of {RETURN_MASK_RULES}, got {declared!r}')
    shots = rl.get('actor_shots')
    if 'actor_shots' in rl:
        if not isinstance(shots, dict) or set(shots) != {'shots', 'seed'} or \
                any(isinstance(shots[k], bool) or not isinstance(shots[k], int) or not low <= shots[k] <= high
                    for k, low, high in (('shots', 1, MAX_SHOTS), ('seed', 0, float('inf')))):
            raise ValueError(f'mission.rl.actor_shots: expected exactly {{shots: integer in [1, {MAX_SHOTS}], seed: integer >= 0}}')
    path = Path(rl['checkpoint'])
    path = path if path.is_absolute() else ROOT / path
    if file_sha256(path) != rl['checkpoint_sha256']:
        raise ValueError(f'checkpoint {path} differs from mission.rl.checkpoint_sha256')
    model, _, payload = load_checkpoint(path)
    model.eval()
    train = payload.get('train_config') if isinstance(payload.get('train_config'), dict) else {}
    trained = train.get('return_mask_rule', 'always')
    if declared is not None and declared != trained:
        raise ValueError(f'mission.rl.return_mask_rule {declared!r} differs from the checkpoint training rule {trained!r}')
    if shots is not None:
        if model.config['actor_head'] not in QUANTUM_ACTOR_HEADS:
            raise ValueError(f"mission.rl.actor_shots: the checkpoint actor head {model.config['actor_head']!r} is not quantum")
        model.actor.set_shots(shots['shots'], shots['seed'])
    return RLMissionPolicy(briefing, obstacle_map, config, reserve, model, deterministic=True, compute_values=False,
                           obs_encoding=rl['obs_encoding'], return_mask_rule=trained)


def build_mission(protocol, briefing, obstacle_data, vehicle_manifest):
    """Public mission stack from the protocol: config, obstacle map, ground model, reserve, grid, tracker, policy."""
    mission, camera = protocol['mission'], protocol['camera']
    if obstacle_data['world_bounds'] != {k: v for k, v in briefing.world_bounds.to_dict().items()}:
        raise ValueError('obstacle map bounds differ from the briefing')
    mount = camera_mount_from_vehicle(vehicle_manifest, protocol['spawn']['base_link_offset_in_model_frame_m'], camera['sensor_offset_link_m'],
                                      camera['rgb_sensor'], source=camera.get('source'))
    config = MissionConfig.from_dict(dict(mission['config'], camera_mount=mount.to_dict(), discount=protocol['scenario']['discount']))
    obstacle_map = obstacle_map_from_json(obstacle_data, briefing.world_bounds, mission['obstacle_z_band_m'])
    ground = GroundHeightModel(obstacle_data, mission['ground_model']['terrain_like_obstacle_kinds'], mission['ground_model']['outside_height_m'])
    r = mission['reserve']
    reserve = ReturnReserve(briefing.budgets, briefing.landing_zone, speed_m_s=config.transit_speed_m_s, climb_speed_m_s=r['climb_speed_m_s'],
                            descent_speed_m_s=r['descent_speed_m_s'], time_margin_factor=r['time_margin_factor'], fixed_margin_s=r['fixed_margin_s'],
                            transit_altitude_m=briefing.landing_zone.center_enu[2] + config.transit_altitude_agl_m, obstacle_map=obstacle_map,
                            clearance_radius_m=config.clearance_radius_m)
    grid = CoverageGrid(briefing, config.coverage_cell_m, config.mount_pitch_rad, config.footprint_max_range_m, config.ground_z_m)
    if mission['policy'] == RL_POLICY:
        policy = build_rl_policy(mission, briefing, obstacle_map, config, reserve)
    elif mission['policy'] not in POLICIES:
        raise ValueError(f"unknown policy {mission['policy']!r}; expected one of {sorted(POLICIES) + [RL_POLICY]}")
    else:
        policy = POLICIES[mission['policy']](briefing, obstacle_map, config, reserve)
    return {'config': config, 'mount': mount, 'obstacle_map': obstacle_map, 'ground': ground, 'reserve': reserve, 'grid': grid,
            'tracker': Tracker(TrackerConfig.from_dict(mission['tracker'])), 'policy': policy}


def task_view_graph(protocol, briefing, stack):
    """(ViewGraph or None, public task record). A protocol without a 'task' block runs flood-sar-mission/1 unchanged. With
    task.version 'flood-sar-viewgraph/1' (ADR 006) or 'flood-sar-village/1' (ADR 010: the same plus public tentative tracks, which
    must match mission.config.public_tentative_tracks) the graph is built from public inputs with task.view_graph
    (ViewGraphConfig) and its sha256 must equal task.view_graph_sha256, which is frozen with the protocol."""
    from aerial_qrl.mission.view_graph import TASK_TENTATIVE_TRACKS, ViewGraphConfig, build_view_graph
    task = protocol.get('task')
    if task is None:
        return None, {'task_version': 'flood-sar-mission/1'}
    version = task.get('version')
    if version not in TASK_TENTATIVE_TRACKS:
        raise ValueError(f"task.version: expected one of {sorted(TASK_TENTATIVE_TRACKS)}, got {version!r}")
    if stack['config'].public_tentative_tracks != TASK_TENTATIVE_TRACKS[version]:
        raise ValueError(f"task.version {version!r} requires mission.config.public_tentative_tracks = {TASK_TENTATIVE_TRACKS[version]}")
    graph = build_view_graph(briefing, stack['obstacle_map'], stack['config'], ViewGraphConfig.from_dict(task['view_graph']))
    if graph.sha256 != task.get('view_graph_sha256'):
        raise ValueError(f"view graph sha256 {graph.sha256} differs from the protocol task.view_graph_sha256")
    return graph, {'task_version': version, 'graph_version': graph.config.graph_version, 'graph_sha256': graph.sha256,
                   'graph_config': graph.config.to_dict(), 'node_counts': graph.counts()}


def plan_summary(policy):
    if not isinstance(policy, CoverageBaselinePolicy):
        return {'policy': policy.name, 'candidates': len(getattr(policy, 'candidates', ())), 'candidate_source': getattr(policy, 'candidate_source', None)}
    plan = policy.plan
    counts = {}
    for w in plan.waypoints:
        counts[w.sector_id] = counts.get(w.sector_id, 0) + 1
    return {'policy': policy.name, 'waypoints': len(plan.waypoints), 'per_sector': counts, 'path_length_m': plan.path_length_m,
            'issues': len(plan.issues), 'swath': vars(plan.swath), 'achieved_overlap': plan.achieved_overlap}


def make_perceive_fn(detector, work_dir, rgb_camera, ground, depth_camera, depth_range_m, localization, log, retain_npy):
    """perception.perceive_sensor_frame with the detector result logged publicly; the .npy copy is deleted unless retained
    (the verified raw bridge array stays under generation_NN/arrays)."""
    def perceive(frame):
        result, detected = perceive_sensor_frame(frame, detector, work_dir, rgb_camera, ground, depth_camera, depth_range_m, localization)
        record = {'kind': 'detector_result', 'sim_time_s': frame.sim_time_s, 'detector': None if detected is None else detected.to_dict()}
        if detected is not None and not retain_npy:
            npy = Path(work_dir) / f'{Path(frame.rgb_path).stem}.npy'
            npy.unlink()
            record['npy_deleted'] = str(npy)
        log.write('public/perception.jsonl', assert_public_payload(record))
        return result
    return perceive


# --- Run -----------------------------------------------------------------------------------------

def nvidia_snapshot():
    import subprocess
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20)
        return {'exit_code': r.returncode, 'stdout': r.stdout.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {'error': repr(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--run-id')
    args = parser.parse_args(argv)
    from code_version import file_hashes, tree_hash
    from run_px4_integration_episode import vehicle_inputs
    protocol_path = (ROOT / args.protocol).resolve()
    protocol_bytes, protocol = verify_freeze(protocol_path)
    inputs = protocol['inputs']
    scene = ROOT / inputs['scene_dir']
    if sha256_file(scene / 'manifest.json') != inputs['scene_manifest_sha256']:
        raise SystemExit('scene manifest differs from protocol')
    scene_manifest = json.loads((scene / 'manifest.json').read_text())
    variant = scene_manifest['variants'][inputs['variant']]
    for rel in (variant['public'], variant['private'], 'obstacle_map.json'):
        if sha256_file(scene / rel) != scene_manifest['files'][rel]:
            raise SystemExit(f'{rel} differs from the scene manifest')
    if scene_manifest['world_name'] != inputs['world_name']:
        raise SystemExit('scene world_name differs from protocol inputs.world_name')
    config_path = ROOT / inputs['development_config']
    development = json.loads(config_path.read_text())
    px4_binary = Path(development['backend']['px4_sitl_binary'])
    if sha256_file(px4_binary) != inputs['px4_sitl_binary_sha256']:
        raise SystemExit('PX4 SITL binary differs from protocol')
    vehicle_manifest, vehicle_params = vehicle_inputs(protocol)
    if vehicle_manifest is None:
        raise SystemExit('protocol inputs.vehicle is required (camera mount source)')
    det = inputs['detector']
    weights = Path(det['weights']).expanduser()
    if not det['weights_sha256'] == FROZEN_WEIGHTS_SHA256 == sha256_file(weights):
        raise SystemExit('detector weights differ from the frozen hash')
    public_scenario = json.loads((scene / variant['public']).read_text())
    obstacle_data = json.loads((scene / 'obstacle_map.json').read_text())
    if public_scenario['obstacle_map'] != obstacle_data:
        raise SystemExit('public scenario obstacle map differs from obstacle_map.json')
    briefing, budget_record = briefing_with_overrides(public_scenario, protocol['mission']['budget_overrides'])
    evaluator_bytes = (scene / variant['private']).read_bytes()
    evaluator = EvaluatorSpec.from_dict(json.loads(evaluator_bytes)['evaluator_spec'])
    scenario = build_scenario_spec(protocol, briefing, evaluator, inputs['scene_manifest_sha256'], inputs['px4_sitl_binary_sha256'])
    stack = build_mission(protocol, briefing, obstacle_data, vehicle_manifest)
    view_graph, task_record = task_view_graph(protocol, briefing, stack)
    if view_graph is not None:
        bind_view_graph(stack['policy'], view_graph)  # before plan_summary; the executive binds the same graph again (no-op)

    run_id =args.run_id or 'run_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = protocol_path.parent / run_id
    prior_runs = sorted(p.name for p in protocol_path.parent.iterdir() if p.is_dir() and p.name.startswith('run_'))
    run_dir.mkdir(exist_ok=False)
    for sub in ('public', 'evaluator', 'diagnostics', 'source_snapshot', 'perception_inputs'):
        (run_dir / sub).mkdir()
    shutil.copy2(protocol_path, run_dir / 'protocol.json')
    snapshot = {}
    for entry in SNAPSHOT + [inputs['development_config']]:
        source = ROOT / entry
        for f in (sorted(source.glob('*.py')) if source.is_dir() else [source]):
            target = run_dir / 'source_snapshot' / f.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            snapshot[str(f.relative_to(ROOT))] = sha256_file(f)
    code_hashes = file_hashes()
    (run_dir / 'code_version.json').write_text(json.dumps({'code_tree_sha256': tree_hash(code_hashes), 'files': code_hashes}, indent=2) + '\n')
    world_path, world_sha = copy_world(scene, inputs['variant'], run_dir, inputs['world_name'], variant['world_sha256'])
    resource_paths = [str(p) for p in scene_manifest['resource_paths']] + [str(scene / p) for p in scene_manifest['scene_resource_paths_relative']]
    vehicle = dict(vehicle_params, resource_paths=resource_paths + list(vehicle_params['resource_paths']))
    manifest = {'protocol_version': protocol['protocol_version'], 'run_id': run_id, 'started_utc': utc(), 'argv': sys.argv,
                'code_tree_sha256': tree_hash(code_hashes), 'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
                'scene_manifest_sha256': inputs['scene_manifest_sha256'], 'variant': inputs['variant'],
                'world': {'source': variant['world'], 'copy': str(world_path.relative_to(run_dir)), 'sha256': world_sha},
                'px4_binary_sha256': inputs['px4_sitl_binary_sha256'], 'development_config_sha256': sha256_file(config_path),
                'vehicle': {'params': vehicle, 'manifest_sha256': inputs['vehicle']['manifest_sha256']},
                'detector': {'weights_sha256': det['weights_sha256'], 'python': str(Path(det['python']).expanduser()),
                             'worker_sha256': sha256_file(ROOT / det['worker'])},
                'camera_mount': stack['mount'].to_dict(), 'budgets': budget_record, 'ground_model': stack['ground'].describe(),
                'plan': plan_summary(stack['policy']), 'source_snapshot_sha256': snapshot, 'prior_runs_in_protocol_dir': prior_runs,
                'host': {'platform': platform.platform(), 'python': sys.version}, 'nvidia_smi_before': nvidia_snapshot()}
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    (run_dir / 'public/task.json').write_text(json.dumps(assert_public_payload(task_record), indent=2) + '\n')
    (run_dir / 'status.json').write_text(json.dumps({'state': 'running', 'updated_utc': utc()}, indent=2) + '\n')
    public_files = {'public/briefing.json': {'briefing': briefing.to_dict(), 'budgets': budget_record},
                    'public/obstacle_map.json': obstacle_data,
                    'public/mission_config.json': {'policy': protocol['mission']['policy'], 'config': stack['config'].to_dict(),
                                                   'tracker': stack['tracker'].config.to_dict(), 'reserve': protocol['mission']['reserve']}}
    for rel, value in public_files.items():
        (run_dir / rel).write_text(json.dumps(assert_public_payload(value), indent=2) + '\n')
    (run_dir / 'evaluator/evaluator_source.json').write_bytes(evaluator_bytes)
    (run_dir / 'evaluator/evaluator_spec.json').write_text(evaluator.to_json() + '\n')
    (run_dir / 'evaluator/scenario_spec.json').write_text(scenario.to_json() + '\n')

    from aerial_qrl.mission.gazebo_backend import GazeboPx4Backend
    from aerial_qrl.mission.journal import LifecycleJournal, read_journal, summarize
    from recover_mission_run import recover
    server_argv = [str(ROOT / 'scripts/px4_env.sh'), development['backend']['px4_python'], str(ROOT / 'scripts/px4_mission_server.py')]
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    health = protocol['health']
    backend = GazeboPx4Backend(server_argv, run_dir, protocol, px4_binary.parents[1], run_dir / 'world', env=env,
                               journal=LifecycleJournal(run_dir / 'lifecycle_journal.jsonl'),
                               health={k: v for k, v in health.items() if k not in ('provenance', 'fault_injection_enabled')},
                               fault_injection_enabled=bool(health.get('fault_injection_enabled')), vehicle=vehicle)
    log = JsonlLog(run_dir)
    errors, detector, outcome, landed = [], None, None, False
    state, failure, t0 = 'failed', None, time.monotonic()

    def flush():
        for record in backend.drain_records():
            log.write('diagnostics/adapter_records.jsonl', record)

    def event(kind, **fields):
        log.write('events.jsonl', {'kind': kind, 'source': 'runner', 'lifecycle_state': backend.state.value, **fields})

    def sample_truth(label, **info):
        flush()
        log.write('evaluator/truth.jsonl', {'label': label, **info, 'truth': backend.evaluator_truth().to_dict()})

    def safety_land():
        if backend.state == LifecycleState.ACTIVE and not landed:
            try:
                result = backend.command(FlightCommand(CommandKind.LAND)).to_dict()
                event('safety_land', outcome=result)
            except Exception as exc:
                errors.append({'utc': utc(), 'error': f'safety land: {exc.__class__.__name__}: {exc}', 'traceback': traceback.format_exc()})
            flush()

    try:
        detector = DetectorClient(python=Path(det['python']).expanduser(), worker=ROOT / det['worker'], worker_args=['--weights', str(weights)],
                                  timeout_s=det['timeout_s'], init_timeout_s=det['init_timeout_s'], expected_weights_sha256=det['weights_sha256'],
                                  stderr_path=run_dir / 'diagnostics/detector_stderr.log')
        (run_dir / 'diagnostics/detector_init.json').write_text(json.dumps(detector.init_record, indent=2) + '\n')
        perception = protocol['perception']
        rgb_camera = CameraModel.from_sensor_model(briefing.sensor_model, 'rgb', stack['mount'])
        depth_camera = CameraModel.from_sensor_model(briefing.sensor_model, 'depth', stack['mount']) if perception['use_depth'] else None
        depth_range = (briefing.sensor_model.depth_min_m, briefing.sensor_model.depth_max_m)
        perceive = make_perceive_fn(detector, run_dir / 'perception_inputs', rgb_camera, stack['ground'], depth_camera, depth_range,
                                    LocalizationConfig.from_dict(perception['localization']), log, perception['retain_detector_npy'])
        event('reset_begin')
        backend.reset(scenario, evaluator)
        # No truth sample here: readiness runs inside executive.run() (begin_episode); the first sample is the executive's
        # EPISODE_BEGIN hook. Regression: run_20260915T082106Z sampled before readiness and was stopped by the failure.
        executive = MissionExecutive(backend, briefing, stack['obstacle_map'], stack['config'], perceive, stack['tracker'], stack['grid'],
                                     stack['reserve'], stack['policy'], log, after_macro=lambda info: sample_truth('after_macro', **info),
                                     view_graph=view_graph)
        outcome = executive.run()
        landed = outcome.landed
        flush()
        (run_dir / 'public/outcome.json').write_text(json.dumps(assert_public_payload(outcome.to_dict()), indent=2) + '\n')
        safety_land()
        if backend.state == LifecycleState.ACTIVE:
            backend.end_episode()
            event('episode_ended')
        if backend.state in (LifecycleState.PREPARED, LifecycleState.ACTIVE, LifecycleState.ENDED):
            sample_truth('final')
        completed = outcome.phase.value == 'COMPLETE' or outcome.termination_reason in TERMINAL_REASONS
        state, failure = ('completed', None) if completed else ('failed', outcome.reason)
    except KeyboardInterrupt:
        state, failure = 'interrupted', 'KeyboardInterrupt'
        errors.append({'utc': utc(), 'error': 'KeyboardInterrupt'})
        safety_land()
    except Exception as exc:
        state, failure = 'failed', f'{exc.__class__.__name__}: {exc}'
        errors.append({'utc': utc(), 'error': failure, 'traceback': traceback.format_exc()})
        safety_land()
    finally:
        try:
            backend.stop()
        except Exception as exc:
            errors.append({'utc': utc(), 'error': f'stop: {exc.__class__.__name__}: {exc}', 'traceback': traceback.format_exc()})
            state = 'failed' if state == 'completed' else state
        if detector is not None:
            try:
                errors.append({'utc': utc(), 'note': f'detector exit code {detector.close()}'})
            except Exception as exc:
                errors.append({'utc': utc(), 'error': f'detector close: {exc.__class__.__name__}: {exc}'})
        flush()
        event('stopped')
        (run_dir / 'diagnostics/lifecycle_events.json').write_text(json.dumps(backend.events, indent=2) + '\n')
        journal = None
        try:
            journal = summarize(*read_journal(run_dir / 'lifecycle_journal.jsonl'))
            (run_dir / 'diagnostics/recovery_dry_run.json').write_text(json.dumps(recover(run_dir, execute=False), indent=2, default=str) + '\n')
        except Exception as exc:
            errors.append({'utc': utc(), 'error': f'journal/recovery check: {exc.__class__.__name__}: {exc}', 'traceback': traceback.format_exc()})
        (run_dir / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
        status = {'state': state, 'failure': failure, 'finished_utc': utc(), 'wall_s': round(time.monotonic() - t0, 3),
                  'outcome': None if outcome is None else outcome.to_dict(), 'final_lifecycle_state': backend.state.value,
                  'journal': journal and {k: journal[k] for k in ('status', 'records', 'torn_tail', 'open_intents')},
                  'nvidia_smi_after': nvidia_snapshot()}
        (run_dir / 'status.json').write_text(json.dumps(status, indent=2, default=str) + '\n')
        checksums = {str(p.relative_to(run_dir)): sha256_file(p) for p in sorted(run_dir.rglob('*'))
                     if p.is_file() and not p.is_symlink() and 'etc' not in p.relative_to(run_dir).parts and p.name != 'checksums.json'}
        (run_dir / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
        print(json.dumps({'run_dir': str(run_dir), 'state': state, 'failure': failure}, indent=2))
    return 0 if state == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
