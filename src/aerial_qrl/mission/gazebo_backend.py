"""Gazebo/PX4 adapter for MissionBackend through scripts/px4_mission_server.py.

Frames      The server speaks PX4 local NED metres and FRD quaternions. This adapter converts to
            world ENU/FLU with a declared local origin: the nominal base_link position at rest
            (public vehicle spec). Gazebo pose/info and PX4_GZ_MODEL_POSE refer to the x500 model
            frame, 0.24 m below base_link; FRAME_CALIBRATION therefore checks the model-frame
            spawn (protocol spawn.model_frame_spawn_enu), never the local origin.
Clock       PX4 time_boot seconds (SITL lockstep with Gazebo). Each reset starts a fresh server
            and PX4/Gazebo process group, so the clock restarts with the reset generation.
Public      sensors.capture and flight.* use the public bridge channel. evaluator.* calls happen
            only in readiness evidence and _evaluator_truth; they never enter SensorFrame.
Freshness   Deterministic rules applied before building the public frame (simulator seconds):
              1. a stream older than TimingPolicy.max_sample_age_s is invalid ('stale');
              2. vehicle_state (timestamp = position sample) is invalid if its attitude sample is
                 older than max_sample_age_s ('attitude_stale') or differs from the position
                 sample by more than max_image_state_skew_s ('attitude_position_skew');
              3. if RGB and depth are both valid but skewed beyond max_rgb_depth_skew_s, both
                 images are invalid;
              4. a valid image skewed from a valid vehicle state beyond max_image_state_skew_s
                 is invalid. An invalid vehicle state leaves images valid: only the pose
                 association is lost, which the public frame exposes through the status.
            Invalid images keep their files only in diagnostics (labelled stale); the public frame
            carries no path for them.
Failures    Bridge timeouts or remote errors propagate to the base class, which invalidates the
            lifecycle. _stop asks the server to stop PX4; if the bridge is unusable it terminates
            the server, whose SIGTERM handler still stops the owned PX4 process group.
Not measured Live collisions and clearance (EvaluatorTruth reports None).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import os

from .backend import (CommandKind, CommandOutcome, EvaluatorTruth, HealthReport, MissionBackend, ReadinessReport, SensorFrame)
from .bridge import BridgeClient, read_array
from .contracts import TIME_TOLERANCE_S, SensorStatus, VehicleEstimate
from .frames import enu_to_ned, enu_yaw_to_ned_yaw, ned_frd_to_enu_flu, ned_to_enu, quat_to_matrix


def server_identity(pid):
    """pid, process group and start time of the bridge server (for journal-based recovery)."""
    stat = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
    return {'pid': pid, 'process_group': os.getpgid(pid), 'start_ticks': stat[19]}


def model_frame_to_base_link_enu(model_position_enu, model_q_wxyz, offset_m):
    """Gazebo pose/info reports the x500 model frame (ENU, FLU axes); base_link is offset inside it."""
    return np.asarray(model_position_enu, float) + quat_to_matrix(model_q_wxyz) @ np.asarray(offset_m, float)


def classify_freshness(frame_time_s, stamps, policy, *, attitude_time_s):
    """Return {stream: reason or None} for rgb/depth/vehicle_state under the four rules above."""
    reasons = {s: None for s in stamps}
    for stream, stamp in stamps.items():
        if frame_time_s - stamp > policy.max_sample_age_s + TIME_TOLERANCE_S:
            reasons[stream] = 'stale'
    fresh = lambda s: reasons[s] is None
    if fresh('vehicle_state'):
        if frame_time_s - attitude_time_s > policy.max_sample_age_s + TIME_TOLERANCE_S:
            reasons['vehicle_state'] = 'attitude_stale'
        elif abs(attitude_time_s - stamps['vehicle_state']) > policy.max_image_state_skew_s + TIME_TOLERANCE_S:
            reasons['vehicle_state'] = 'attitude_position_skew'
    if fresh('rgb') and fresh('depth') and abs(stamps['rgb'] - stamps['depth']) > policy.max_rgb_depth_skew_s + TIME_TOLERANCE_S:
        reasons['rgb'] = reasons['depth'] = 'rgb_depth_skew'
    for image in ('rgb', 'depth'):
        if fresh(image) and fresh('vehicle_state') and \
                abs(stamps[image] - stamps['vehicle_state']) > policy.max_image_state_skew_s + TIME_TOLERANCE_S:
            reasons[image] = 'image_state_skew'
    return reasons


class GazeboPx4Backend(MissionBackend):
    def __init__(self, server_argv, run_dir, protocol, px4_build_dir, world_dir, env=None, request_timeout_s=30.,
                 capture_timeout_s=120., stop_timeout_s=120., journal=None, health=None, fault_injection_enabled=False, vehicle=None):
        super().__init__(journal)
        # Derived vehicle (optional): {'vehicle_models_dir', 'sys_autostart', 'resource_paths'} passed to lifecycle.start.
        self.vehicle = {k: v for k, v in (vehicle or {}).items() if k in ('vehicle_models_dir', 'sys_autostart', 'resource_paths')}
        self.health_limits = dict(health or {'max_heartbeat_age_wall_s': 3., 'max_local_position_age_s': .5,
                                             'max_attitude_age_s': .5, 'check_period_wall_s': .1})
        self.fault_injection_enabled = bool(fault_injection_enabled)
        self.capture_timeout_s = capture_timeout_s
        self.stop_timeout_s = stop_timeout_s
        self.readiness_timeout_s = 600.  # qualification runners lower this to the remaining startup budget
        self.server_argv = list(server_argv)
        self.run_dir = Path(run_dir)
        self.protocol = protocol
        self.px4_build_dir = str(px4_build_dir)
        self.world_dir = str(world_dir)
        self.env = env
        self.request_timeout_s = request_timeout_s
        self.origin_enu = np.asarray(protocol['spawn']['local_ned_origin_enu'], float)
        self.client = None
        self.generation_dir = None
        self.reset_count = 0
        self.frame_count = 0
        self.records = []
        self.process = None
        self.stop_report = None

    def _record(self, kind, **fields):
        self.records.append({'kind': kind, 'generation': self.reset_count, **fields})

    def drain_records(self):
        records, self.records = self.records, []
        return records

    def _call(self, method, params=None, timeout_s=None):
        reply = self.client.call(method, {} if params is None else params, timeout_s or self.request_timeout_s)
        self._record('bridge_call', method=method, seq=reply.seq, round_trip_s=reply.round_trip_s)
        return reply

    def _reset(self, scenario, evaluator):
        if self.client is not None:
            self._stop()
        world = Path(self.world_dir) / f"{self.protocol['inputs']['world_name']}.sdf"
        if not world.is_file():
            raise FileNotFoundError(f'world file missing: {world}')
        self.reset_count += 1
        self.generation_dir = self.run_dir / f'generation_{self.reset_count:02d}'
        self.generation_dir.mkdir(parents=True, exist_ok=False)
        self._truth = evaluator
        self.client = BridgeClient(self.server_argv, timeout_s=self.request_timeout_s, env=self.env,
                                   stderr_path=self.generation_dir / 'server_stderr.log')
        spawn = self.protocol['spawn']
        params = {'work_dir': str(self.generation_dir / 'rootfs'), 'arrays_dir': str(self.generation_dir / 'arrays'),
                  'diagnostics_dir': str(self.generation_dir / 'diagnostics'), 'px4_build_dir': self.px4_build_dir,
                  'world_dir': self.world_dir, 'world_name': self.protocol['inputs']['world_name'],
                  'spawn_pose': spawn['px4_gz_model_pose'], 'model_frame_spawn_enu': list(spawn['model_frame_spawn_enu']),
                  'sim_model': self.protocol['inputs']['vehicle_model'], 'partition': self.protocol['inputs']['gz_partition'],
                  'readiness': self.protocol['readiness'], 'controller': self.protocol['controller'],
                  'health': self.health_limits, 'fault_injection_enabled': self.fault_injection_enabled, **self.vehicle}
        self.journal_record('identity', role='bridge_server', **server_identity(self.client.pid))
        self.process = self._call('lifecycle.start', params, timeout_s=60).result
        self.journal_record('identity', role='px4', pid=self.process['pid'], process_group=self.process['process_group'],
                            start_ticks=self.process['start_ticks'])
        self._record('started', process=self.process)

    def _readiness(self):
        result = self._call('lifecycle.readiness', timeout_s=self.readiness_timeout_s).result
        evidence = self._call('evaluator.readiness_evidence').result
        self._record('readiness', history=result['history'], errors=result['errors'], stage_timing=result.get('stage_timing'),
                     evaluator_evidence=evidence)
        return ReadinessReport(tuple(result['history']), tuple(result['errors']))

    def _read_sensors(self):
        self.frame_count += 1
        reply = self._call('sensors.capture', {'label': f'frame_{self.frame_count:03d}'}, timeout_s=self.capture_timeout_s)
        result, refs = reply.result, {ref['name']: ref for ref in reply.arrays}
        if set(refs) != {'rgb', 'depth'}:
            raise RuntimeError('capture must return exactly rgb and depth arrays')
        for ref in refs.values():
            read_array(ref)  # size and SHA256 verification before any public use
        t = result['sim_time_s']
        state = result['vehicle_state']
        stamps = {'rgb': result['rgb']['timestamp_s'], 'depth': result['depth']['timestamp_s'], 'vehicle_state': state['timestamp_s']}
        reasons = classify_freshness(t, stamps, self._scenario.timing, attitude_time_s=state['attitude_timestamp_s'])
        sensors = tuple(SensorStatus(s, reasons[s] is None, stamps[s], t - stamps[s]) for s in ('rgb', 'depth', 'vehicle_state'))
        vehicle = VehicleEstimate(tuple(self.origin_enu + ned_to_enu(state['position_ned'])), tuple(ned_to_enu(state['velocity_ned'])),
                                  tuple(ned_frd_to_enu_flu(state['q_ned_frd'])), 'px4_estimator')
        frame = SensorFrame(t, vehicle, sensors, refs['rgb']['path'] if reasons['rgb'] is None else None,
                            refs['depth']['path'] if reasons['depth'] is None else None, ())
        self._record('capture', label=result['label'], sim_time_s=t, stamps=stamps, invalid_reasons=reasons,
                     attitude_timestamp_s=state['attitude_timestamp_s'], request_sim_time_s=result['request_sim_time_s'],
                     arrays={name: dict(ref, stale=reasons[name] is not None) for name, ref in refs.items()})
        return frame

    def _command(self, command):
        if command.kind in (CommandKind.TAKEOFF, CommandKind.GOTO):
            target = enu_to_ned(np.asarray(command.position_enu) - self.origin_enu).tolist()
            method = 'flight.takeoff' if command.kind == CommandKind.TAKEOFF else 'flight.goto'
            params = {'target_ned': target, 'max_speed_m_s': command.max_speed_m_s}
            if command.yaw_enu_rad is not None:
                params['yaw_ned_rad'] = enu_yaw_to_ned_yaw(command.yaw_enu_rad)
            distance = float(np.linalg.norm(np.asarray(command.position_enu) - self.origin_enu))
            timeout_s = 4 * (distance / command.max_speed_m_s + 20) + 180
        elif command.kind == CommandKind.HOLD:
            method, params, timeout_s = 'flight.hold', {'hold_s': command.hold_s}, 4 * command.hold_s + 60
        else:
            method, params, timeout_s = 'flight.land', {}, 300
        result = self._call(method, params, timeout_s=timeout_s).result
        self._record('command', method=method, params=params, outcome=result)
        if not result['completed'] and str(result['reason']).startswith('health_fault:'):
            # Hard stop already requested by the server; the episode must not continue.
            raise RuntimeError(f"{method} ended by {result['reason']}")
        return CommandOutcome.from_dict(result)

    def _health(self):
        result = self._call('flight.health').result
        self._record('health', result=result)
        return HealthReport(result['ok'], None if result['ok'] else result['fault']['reason'], result['sim_time_s'])

    def inject_position_blackout(self, duration_wall_s):
        """Fault-injection runs only (server started with fault_injection_enabled)."""
        result = self._call('fault.position_blackout', {'duration_wall_s': duration_wall_s}).result
        self._record('fault_injection', result=result)
        self.journal_record('note', text='fault injection: position blackout', duration_wall_s=duration_wall_s)
        return result

    def _end_episode(self):
        status = self._call('flight.status').result
        self._record('end_status', status=status)
        if status['armed'] or status['landed_state'] != 1:
            raise RuntimeError(f'vehicle not landed and disarmed at episode end: {status}')

    def _evaluator_truth(self):
        pose = self._call('evaluator.pose', timeout_s=60).result
        base_link = model_frame_to_base_link_enu(pose['position_enu'], pose['orientation_wxyz'],
                                                 self.protocol['spawn']['base_link_offset_in_model_frame_m'])
        self._record('evaluator_pose', pose=pose, base_link_position_enu=base_link.tolist())
        return EvaluatorTruth(pose['px4_time_s'], tuple(base_link), None, None, self.reset_count, 0, self._truth)

    def _stop(self):
        if self.client is None:
            return
        try:
            if not self.client.broken:
                self.stop_report = self._call('lifecycle.stop', timeout_s=180).result
                self._record('stop_report', report=self.stop_report)
        finally:
            exit_code = self.client.close(timeout_s=self.stop_timeout_s)
            self._record('server_closed', exit_code=exit_code)
            self.client = None
