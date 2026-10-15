"""Run the frozen px4-integration-episode/1 protocol once, into a fresh run directory.

From the project root with the ML venv:
  .venv/bin/python scripts/run_px4_integration_episode.py --protocol output/mission/px4_integration_v1/protocol.json

Refuses to start when the protocol differs from its last recorded freeze hash, when the scene
manifest or PX4 binary differ from the protocol, or when the run directory exists. Steps,
failures and adapter records are appended as they happen. The run ends as completed, failed or
interrupted and is checksummed. Gates G7/G8 need scripts/analyze_px4_integration_episode.py.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from aerial_qrl.mission.backend import CommandKind, FlightCommand, LifecycleState
from aerial_qrl.mission.contracts import EvaluatorSpec, ScenarioSpec
from aerial_qrl.mission.gazebo_backend import GazeboPx4Backend
from aerial_qrl.mission.journal import LifecycleJournal, read_journal, summarize
from code_version import file_hashes, tree_hash
from recover_mission_run import recover

SNAPSHOT = ['src/aerial_qrl/mission', 'scripts/px4_mission_server.py', 'scripts/run_px4_integration_episode.py',
            'scripts/recover_mission_run.py', 'scripts/code_version.py', 'scripts/px4_env.sh', 'configs/mission/development.json',
            'scripts/build_vehicle_model.py']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def nvidia_snapshot():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=20)
        return {'exit_code': r.returncode, 'stdout': r.stdout.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {'error': repr(exc)}


class RunLog:
    def __init__(self, run_dir):
        self.run_dir, self.t0 = run_dir, time.monotonic()

    def write(self, relative, record):
        with open(self.run_dir / relative, 'a') as f:
            f.write(json.dumps({'utc': utc(), 'wall_elapsed_s': round(time.monotonic() - self.t0, 3), **record}, default=str) + '\n')


def percentiles(values):
    if not values:
        return None
    a = np.asarray(values)
    return {'n': len(a), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def vehicle_inputs(protocol):
    """Derived vehicle (protocol v4+): verified manifest and the lifecycle.start parameters; None for the stock model."""
    vehicle = protocol['inputs'].get('vehicle')
    if not vehicle:
        return None, None
    manifest_path = ROOT / vehicle['manifest']
    if sha256_file(manifest_path) != vehicle['manifest_sha256']:
        raise SystemExit('vehicle model manifest differs from protocol')
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(manifest_path.parent / manifest['model'] / 'model.sdf') != manifest['model_sdf_sha256']:
        raise SystemExit('vehicle model.sdf differs from its manifest')
    params = {'vehicle_models_dir': str(manifest_path.parent), 'sys_autostart': vehicle['sys_autostart'],
              'resource_paths': [str(ROOT / p) for p in vehicle.get('resource_paths', [])]}
    return manifest, params


def depth_validity_gate(protocol, frames, steps=('C1', 'C2', 'C3', 'C4')):
    """G16 (protocol v5+): fraction of depth pixels below the near clip (-inf) in each airborne capture.
    Uses the public frame's depth_path (raw little-endian float32, size from the readiness spec); a missing depth fails."""
    shape = protocol['readiness']['sensor_conditioning']['depth']
    fractions = {}
    for step in steps:
        path = (frames.get(step) or {}).get('depth_path')
        if path:
            depth = np.fromfile(path, '<f4').reshape(shape['height'], shape['width'])
            fractions[step] = float(np.isneginf(depth).mean())
    limit = protocol['gate_parameters']['max_depth_near_clip_fraction']
    return len(fractions) == len(steps) and all(f <= limit for f in fractions.values()), fractions


def compute_metrics(protocol, backend_events, records, frames, outcomes, truths, final_state, journal_summary=None, recovery=None, health_checks=(),
                    vehicle_manifest=None):
    readiness = [r for r in records if r['kind'] == 'readiness']
    captures = [r for r in records if r['kind'] == 'capture']
    flight_steps = [s['step'] for s in protocol['sequence'] if s['kind'] != 'CAPTURE']
    airborne = ['C1', 'C2', 'C3', 'C4']
    latency = {}
    for r in records:
        if r['kind'] == 'bridge_call':
            latency.setdefault(r['method'], []).append(r['round_trip_s'])
    g6 = {}
    for step in airborne:
        if step in frames and step in truths:
            estimate, truth = frames[step]['vehicle']['position_enu'], truths[step]['vehicle_position_enu']
            g6[step] = {'horizontal_error_m': float(np.hypot(estimate[0] - truth[0], estimate[1] - truth[1])),
                        'vertical_error_m': float(estimate[2] - truth[2]),
                        'estimate_time_s': [s for s in frames[step]['sensors'] if s['stream'] == 'vehicle_state'][0]['timestamp_s'],
                        'evaluator_query_time_s': truths[step]['sim_time_s']}
    fresh = {step: {s['stream']: s['valid'] for s in frames[step]['sensors']} for step in frames}
    gates = {
        'G1_ready_first_attempt': bool(readiness) and readiness[0]['history'][-1] == 'READY' and not readiness[0]['errors'],
        'G2_flight': all(step in outcomes and outcomes[step]['completed'] for step in flight_steps),
        'G3a_payload_integrity': {'verified_files': 2 * len(captures), 'expected': 2 * sum(s['kind'] == 'CAPTURE' for s in protocol['sequence'])},
        'G3b_freshness': all(step in fresh and all(fresh[step].values()) for step in airborne),
        'G4_lifecycle': all(e['ok'] for e in backend_events) and final_state == LifecycleState.STOPPED.value
                        and any(e['operation'] == 'end_episode' and e['ok'] for e in backend_events),
        'G5_clock': bool(readiness) and readiness[0]['history'][-1] == 'READY',
        'G6_estimate_vs_evaluator': bool(g6) and len(g6) == len(airborne) and all(v['horizontal_error_m'] <= .5 for v in g6.values()),
        'G7_ulog': 'pending_offline_analysis',
        'G8_clearance_offline': 'pending_offline_analysis',
    }
    gates['G3a_payload_integrity']['pass'] = gates['G3a_payload_integrity']['verified_files'] == gates['G3a_payload_integrity']['expected']
    if 'G9_journal_and_recovery' in protocol['gates']:
        decisions = [a['decision'] for j in (recovery or {}).get('journals', []) for a in j['actions']]
        gates['G9_journal_and_recovery'] = bool(journal_summary) and journal_summary['status'] == 'clean_stop' and not journal_summary['open_intents'] \
            and bool(decisions) and all(d == 'not_running' for d in decisions)
        server_faults = [r for r in records if r['kind'] == 'stop_report' and r['report'].get('health_fault')]
        gates['G10_health'] = bool(health_checks) and all(h['ok'] for h in health_checks) and not server_faults
        stage_timing = readiness[0].get('stage_timing', []) if readiness else []
        gates['G11_stage_timing'] = bool(stage_timing) and all(t.get('start_sim_s') is not None or t['stage'] in ('START', 'RPC_READY')
                                                              for t in stage_timing) and all('wall_s' in t for t in stage_timing)
        gates['G12_evaluator_frame_offline'] = 'pending_offline_analysis'
    if 'G13_speed_offline' in protocol['gates']:
        gates['G13_speed_offline'] = 'pending_offline_analysis'
    depth_near_clip = {}
    if 'G16_depth_validity' in protocol['gates']:
        gates['G16_depth_validity'], depth_near_clip = depth_validity_gate(protocol, frames)
    heading_errors = {}
    if 'G15_heading' in protocol['gates']:  # public PX4 attitude at captures that follow a commanded heading
        expected = {s['step']: s['expected_yaw_enu_rad'] for s in protocol['sequence'] if 'expected_yaw_enu_rad' in s}
        for step, yaw in expected.items():
            if step in frames:
                w, x, y, z = frames[step]['vehicle']['attitude_wxyz']
                actual = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                heading_errors[step] = abs((actual - yaw + math.pi) % (2 * math.pi) - math.pi)
        gates['G15_heading'] = bool(expected) and len(heading_errors) == len(expected) and \
            all(e <= protocol['gate_parameters']['heading_tolerance_rad'] for e in heading_errors.values())
    if 'G14_propeller_geometry' in protocol['gates']:
        checks = (vehicle_manifest or {}).get('propeller_check_design', {}).get('at_build_pitch_radius_0.2m', {})
        gates['G14_propeller_geometry'] = bool(checks) and all(v['points_in_view'] == 0 for v in checks.values())
    return {'gates_live': gates, 'estimate_vs_evaluator': g6, 'freshness_by_capture': fresh, 'heading_errors_rad': heading_errors,
            'depth_near_clip_fraction': depth_near_clip,
            'journal_summary': journal_summary and {k: journal_summary[k] for k in ('status', 'records', 'torn_tail', 'open_intents')},
            'recovery_decisions': [a['decision'] for j in (recovery or {}).get('journals', []) for a in j['actions']],
            'health_checks': list(health_checks),
            'readiness_stage_timing': readiness[0].get('stage_timing') if readiness else None,
            'capture_timing': [{'label': r['label'], 'sim_time_s': r['sim_time_s'], 'stamps': r['stamps'],
                                'ages_s': {k: r['sim_time_s'] - v for k, v in r['stamps'].items()},
                                'capture_request_to_frame_s': r['sim_time_s'] - r['request_sim_time_s'],
                                'invalid_reasons': r['invalid_reasons']} for r in captures],
            'bridge_round_trip_s': {m: percentiles(v) for m, v in latency.items()},
            'command_outcomes': outcomes,
            'readiness': [{'history': r['history'], 'errors': r['errors']} for r in readiness]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='output/mission/px4_integration_v1/protocol.json')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    freeze_lines = [l for l in (protocol_path.parent / 'protocol_freeze.sha256').read_text().splitlines() if l.endswith('protocol.json')]
    if not freeze_lines or hashlib.sha256(protocol_bytes).hexdigest() != freeze_lines[-1].split()[0]:
        raise SystemExit('protocol.json does not match its last recorded freeze hash; refusing to run')
    config_path = ROOT / protocol['inputs']['development_config']
    config = json.loads(config_path.read_text())
    scene = ROOT / protocol['inputs']['scene_dir']
    if sha256_file(scene / 'manifest.json') != protocol['inputs']['scene_manifest_sha256']:
        raise SystemExit('scene manifest differs from protocol')
    px4_binary = Path(config['backend']['px4_sitl_binary'])
    if sha256_file(px4_binary) != protocol['inputs']['px4_sitl_binary_sha256']:
        raise SystemExit('PX4 SITL binary differs from protocol')
    vehicle_manifest, vehicle_params = vehicle_inputs(protocol)

    run_id = args.run_id or 'run_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = protocol_path.parent / run_id
    prior_runs = sorted(p.name for p in protocol_path.parent.iterdir() if p.is_dir() and p.name.startswith('run_'))
    run_dir.mkdir(exist_ok=False)
    for sub in ('public', 'evaluator', 'diagnostics', 'source_snapshot'):
        (run_dir / sub).mkdir()
    shutil.copy2(protocol_path, run_dir / 'protocol.json')
    snapshot_hashes = {}
    for entry in SNAPSHOT:
        source = ROOT / entry
        files = sorted(source.glob('*.py')) if source.is_dir() else [source]
        for f in files:
            target = run_dir / 'source_snapshot' / f.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            snapshot_hashes[str(f.relative_to(ROOT))] = sha256_file(f)
    code_hashes = file_hashes()
    (run_dir / 'code_version.json').write_text(json.dumps({'code_tree_sha256': tree_hash(code_hashes), 'files': code_hashes}, indent=2) + '\n')
    manifest = {'protocol_version': protocol['protocol_version'], 'run_id': run_id, 'started_utc': utc(), 'argv': sys.argv,
                'code_tree_sha256': tree_hash(code_hashes),
                'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(), 'scene_manifest_sha256': protocol['inputs']['scene_manifest_sha256'],
                'px4_binary_sha256': protocol['inputs']['px4_sitl_binary_sha256'], 'config_sha256': sha256_file(config_path),
                'vehicle': vehicle_params and {'params': vehicle_params, 'manifest_sha256': protocol['inputs']['vehicle']['manifest_sha256']},
                'source_snapshot_sha256': snapshot_hashes, 'prior_runs_in_protocol_dir': prior_runs,
                'host': {'platform': platform.platform(), 'python': sys.version}, 'nvidia_smi_before': nvidia_snapshot()}
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (run_dir / 'status.json').write_text(json.dumps({'state': 'running', 'updated_utc': utc()}, indent=2) + '\n')

    scenario = ScenarioSpec.from_json((scene / 'private/scenario.json').read_text())
    evaluator = EvaluatorSpec.from_json((scene / 'evaluator/evaluator.json').read_text())
    server_argv = [str(ROOT / 'scripts/px4_env.sh'), config['backend']['px4_python'], str(ROOT / 'scripts/px4_mission_server.py')]
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    health = protocol.get('health')
    backend = GazeboPx4Backend(server_argv, run_dir, protocol, px4_binary.parents[1], scene / 'world', env=env,
                               journal=LifecycleJournal(run_dir / 'lifecycle_journal.jsonl'),
                               health={k: v for k, v in health.items() if k not in ('provenance', 'fault_injection_enabled')} if health else None,
                               fault_injection_enabled=bool(health and health.get('fault_injection_enabled')), vehicle=vehicle_params)
    log = RunLog(run_dir)
    records, frames, outcomes, truths, errors, health_checks = [], {}, {}, {}, [], []
    journal_summary, recovery_report = None, None
    state, failure = 'failed', None
    t0 = time.monotonic()

    def flush():
        for record in backend.drain_records():
            records.append(record)
            log.write('diagnostics/adapter_records.jsonl', record)

    def event(step, **fields):
        log.write('events.jsonl', {'step': step, 'lifecycle_state': backend.state.value, **fields})

    try:
        event('reset_begin')
        backend.reset(scenario, evaluator)
        flush()
        event('reset_done')
        backend.begin_episode()
        flush()
        event('episode_begun')
        for item in protocol['sequence']:
            step, kind = item['step'], item['kind']
            event(step + '_begin', kind=kind)
            if kind == 'CAPTURE':
                frame = backend.read_sensors().to_dict()
                frames[step] = frame
                log.write('public/frames.jsonl', {'step': step, 'frame': frame})
                truth = backend.evaluator_truth().to_dict()
                truths[step] = truth
                log.write('evaluator/truth.jsonl', {'step': step, 'truth': truth})
            else:
                command = FlightCommand(kind, tuple(item['position_enu']) if 'position_enu' in item else None,
                                        item.get('max_speed_m_s'), item.get('hold_s'), item.get('yaw_enu_rad'))
                outcome = backend.command(command).to_dict()
                outcomes[step] = outcome
                log.write('public/commands.jsonl', {'step': step, 'command': command.to_dict(), 'outcome': outcome})
                if not outcome['completed']:
                    failure = f"{step} not completed: {outcome['reason']}"
            if health and backend.state == LifecycleState.ACTIVE:
                check = backend.check_health().to_dict()
                health_checks.append({'step': step, **check})
                log.write('diagnostics/health.jsonl', {'step': step, 'health': check})
            flush()
            event(step + '_done')
            if failure:
                break
        if failure and backend.state == LifecycleState.ACTIVE and not outcomes.get('S6', {}).get('completed'):
            safety = backend.command(FlightCommand(CommandKind.LAND)).to_dict()
            log.write('public/commands.jsonl', {'step': 'SAFETY_LAND_outside_sequence', 'outcome': safety})
            flush()
        if backend.state == LifecycleState.ACTIVE:
            backend.end_episode()
            flush()
            event('episode_ended')
        if backend.state in (LifecycleState.PREPARED, LifecycleState.ACTIVE, LifecycleState.ENDED):
            final = backend.evaluator_truth().to_dict()
            log.write('evaluator/truth.jsonl', {'step': 'FINAL', 'truth': final})
            flush()
        state = 'completed' if failure is None else 'failed'
    except KeyboardInterrupt:
        state, failure = 'interrupted', 'KeyboardInterrupt'
        errors.append({'utc': utc(), 'error': 'KeyboardInterrupt'})
    except Exception as exc:
        state, failure = 'failed', f'{exc.__class__.__name__}: {exc}'
        errors.append({'utc': utc(), 'error': failure, 'traceback': traceback.format_exc()})
    finally:
        try:
            backend.stop()
        except Exception as exc:
            errors.append({'utc': utc(), 'error': f'stop: {exc.__class__.__name__}: {exc}', 'traceback': traceback.format_exc()})
            state = 'failed' if state == 'completed' else state
        flush()
        event('stopped')
        (run_dir / 'diagnostics/lifecycle_events.json').write_text(json.dumps(backend.events, indent=2) + '\n')
        try:
            journal_summary = summarize(*read_journal(run_dir / 'lifecycle_journal.jsonl'))
            recovery_report = recover(run_dir, execute=False)
            (run_dir / 'diagnostics/recovery_dry_run.json').write_text(json.dumps(recovery_report, indent=2, default=str) + '\n')
        except Exception as exc:
            errors.append({'utc': utc(), 'error': f'journal/recovery check: {exc.__class__.__name__}: {exc}', 'traceback': traceback.format_exc()})
        metrics = compute_metrics(protocol, backend.events, records, frames, outcomes, truths, backend.state.value,
                                  journal_summary, recovery_report, health_checks, vehicle_manifest)
        metrics['stop_report'] = backend.stop_report
        metrics['nvidia_smi_after'] = nvidia_snapshot()
        (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str) + '\n')
        (run_dir / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
        (run_dir / 'status.json').write_text(json.dumps({'state': state, 'failure': failure, 'finished_utc': utc(),
                                                         'wall_s': round(time.monotonic() - t0, 3)}, indent=2) + '\n')
        checksums = {str(p.relative_to(run_dir)): sha256_file(p) for p in sorted(run_dir.rglob('*'))
                     if p.is_file() and not p.is_symlink() and 'etc' not in p.relative_to(run_dir).parts and p.name != 'checksums.json'}
        (run_dir / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
        print(json.dumps({'run_dir': str(run_dir), 'state': state, 'failure': failure, 'gates_live': metrics['gates_live']}, indent=2))
    return 0 if state == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
