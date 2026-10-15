"""CachedViewBackend: a MissionBackend with calibrated command durations and real Gazebo observations from a bank (ADR 006).

Clock and motion (simulator seconds from reset; kinematic, no dynamics, no hover jitter, no collision model)
    A command starts after the time-model gap for (previous command kind -> this kind), which stands for the capture,
    detector and planning latency during which Gazebo keeps running, and lasts the time-model duration. The vehicle is
    then exactly at the goal. TAKEOFF and GOTO keep the current yaw unless the command gives one. LAND puts base_link
    rest_height_m above the landing-zone height at the current xy. A TAKEOFF/GOTO goal outside the world bounds is accepted
    but not completed ('goal_outside_world'), as a live backend reports a failed goal.
Sensing
    read_sensors serves the bank capture of the node whose pose equals the vehicle pose within the configured tolerances.
    At any other pose (launch pad, landing, off-graph) RGB and depth are INVALID: no image of another node and no oracle
    fallback. The vehicle estimate is the kinematic pose, declared 'simulator_pose_declared_ideal'.
Re-observation model (CachedViewConfig.reobservation_model; ADR 006 revision 2)
    'frame_periodic/1' (ADR 006 revision 1, the default so earlier runs reproduce): the v-th FRAME read at a node in the
    episode gets repeat v mod R. Frames of one hover (e.g. the 3 inspection frames) get different repeats, and the order is
    the same in every episode, so a policy with memory could learn node-specific revisit counts.
    'visit_permutation/2': a VISIT starts with every completed TAKEOFF/GOTO; all frames read at the node until the next
    completed TAKEOFF/GOTO belong to that visit and get the same capture (one hover realisation). Visit k of a node gets
    order[k mod R] with order = numpy default_rng([reobservation_seed, crc32(node_id), k // R]).permutation(R): within an
    episode each cycle of R visits uses every repeat once, in an order that changes with reobservation_seed. The seed is
    chosen by the runner per episode and is never an observation. Limitation: within a cycle repeats are drawn without
    replacement, so after a miss the next visit is slightly more likely to show a detecting repeat than an independent draw.
    'visit_frame_permutation/3' (perception model v3, validation-capture/4): the repeat of a visit as in /2, and frame j of the
    visit serves frame j mod K of that repeat (observation-bank/2: K frames per repeat at image-time poses of one real hover).
    With a one-frame bank it equals /2.
Evidence serving (CachedViewConfig.evidence_serving; factor D 'deterministic evidence' of the r6 study)
    'all' (default, so earlier runs reproduce): the re-observation model above chooses the capture of every frame.
    'first_only': every frame read at a node, whatever the visit and frame, serves the node's first served capture
    bank.capture_for_visit_frame(node, 0, 0), i.e. frame 0 of the node's first served repeat in repeat order (with
    frame_periodic/1 the same capture is read as bank.capture_for_visit(node, 0)). The re-observation model's visit and frame counters and permutation
    are neither consulted nor advanced, so reobservation_seed has no effect. Each node therefore shows one image, as a
    one-repeat, one-frame bank holding only these captures would under any re-observation model. With repeated_capture
    'invalid' (below) that image counts once per episode and every later frame read at the node is invalid: the
    one-image-per-node evidence structure of the old r5 banks, reproduced on a multi-repeat, multi-frame bank (the image
    itself differs: r5 images were rendered at the ideal node pose, an r6 repeat 0 frame 0 at a measured image-time pose,
    so factor D isolates evidence variability, not pose realism). With 'serve' the same
    image is served again and counted again by the tracker.
Repeated captures (CachedViewConfig.repeated_capture; fast-env audit of 16 Sep 2026, B1)
    'serve' (default, so earlier runs reproduce): a capture already served in the episode is served again, so the tracker
    counts the same image as new evidence (3 inspection frames + the report capture of a one-frame bank are one image).
    'invalid': a capture whose RGB sha256 was already served in the episode is served with INVALID RGB and depth (vehicle
    state valid): no candidates, no tracker or coverage update, time still charged; counted in stats['repeated_capture_frames'].
    The same image therefore counts once per episode. Gazebo frames of one hover are different images, so with a one-frame
    bank 'invalid' undercounts evidence relative to Gazebo; a bank with K >= frames per visit removes the gap.
Perception
    make_cached_perceive_fn replays the frozen detector result stored for the served RGB sha256 and runs the same
    perception.perceive (ground ray, depth, missing-data rules) with the bank depth array; the tracker then runs as live.
Evaluator
    evaluator_truth gives the kinematic pose; collisions and min_clearance_m are None (not modelled).
Public statistics (.stats, reset every episode): frames, bank frames, off-node frames, repeated-capture frames, commands by
kind, time-model flags.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import zlib

import numpy as np

from .backend import (STARTUP_SEQUENCE, BackendError, CommandKind, CommandOutcome, EvaluatorTruth, HealthReport, MissionBackend,
                      ReadinessReport, SensorFrame)
from .contracts import PROVENANCE, Contract, SensorStatus, VehicleEstimate, _choice, _count, _nested, _real
from .perception import PerceptionError, perceive

STREAMS = ('rgb', 'depth', 'vehicle_state')
REOBSERVATION_MODELS = ('frame_periodic/1', 'visit_permutation/2', 'visit_frame_permutation/3')
REPEATED_CAPTURE_POLICIES = ('serve', 'invalid')
EVIDENCE_SERVING = ('all', 'first_only')


@dataclass(frozen=True)
class CachedViewConfig(Contract):
    rest_height_m: float = .24           # base_link above the pad at rest (x500 base_link offset, mission protocols' spawn)
    initial_yaw_rad: float = 0.          # PX4 spawn yaw of the mission protocols ('20,20,0,0,0,0')
    node_xy_tolerance_m: float = 1e-3
    node_z_tolerance_m: float = 1e-3
    node_yaw_tolerance_rad: float = 1e-3
    provenance: str = 'engineering_default_unfrozen'
    reobservation_model: str = 'frame_periodic/1'
    reobservation_seed: int = 0
    repeated_capture: str = 'serve'
    evidence_serving: str = 'all'

    def __post_init__(self):
        _count(self.reobservation_seed, 'reobservation_seed', 0)
        self._set(reobservation_model=_choice(self.reobservation_model, 'reobservation_model', REOBSERVATION_MODELS),
                  repeated_capture=_choice(self.repeated_capture, 'repeated_capture', REPEATED_CAPTURE_POLICIES),
                  evidence_serving=_choice(self.evidence_serving, 'evidence_serving', EVIDENCE_SERVING))
        self._set(rest_height_m=_real(self.rest_height_m, 'rest_height_m', low=0), initial_yaw_rad=_real(self.initial_yaw_rad, 'initial_yaw_rad'),
                  node_xy_tolerance_m=_real(self.node_xy_tolerance_m, 'node_xy_tolerance_m', low=0, open_low=True),
                  node_z_tolerance_m=_real(self.node_z_tolerance_m, 'node_z_tolerance_m', low=0, open_low=True),
                  node_yaw_tolerance_rad=_real(self.node_yaw_tolerance_rad, 'node_yaw_tolerance_rad', low=0, open_low=True),
                  provenance=_choice(self.provenance, 'provenance', PROVENANCE))


def yaw_quaternion(yaw):
    return (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class CachedViewBackend(MissionBackend):
    def __init__(self, bank, time_model, config=None, journal=None):
        super().__init__(journal)
        self.bank, self.time_model = bank, time_model
        self.config = CachedViewConfig() if config is None else _nested(CachedViewConfig, config, 'config')
        self.pre_episode_teleports = 0
        self.served_capture = None
        self.frames = []
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return {'frames': 0, 'bank_frames': 0, 'off_node_frames': 0, 'repeated_capture_frames': 0, 'commands': {}, 'time_model_flags': {}}

    def _reset(self, scenario, evaluator):
        centre = scenario.briefing.launch_zone.center_enu
        self._evaluator_spec = evaluator
        self._position = np.array([centre[0], centre[1], centre[2] + self.config.rest_height_m], float)
        self._yaw = self.config.initial_yaw_rad
        self._sim_time, self._last_kind, self._visits = 0., None, {}
        self._arrivals, self._node_visit, self._served_rgb = 0, {}, set()
        self.served_capture, self.frames, self.stats = None, [], self._new_stats()
        self.pre_episode_teleports += 1

    def _readiness(self):
        return ReadinessReport(STARTUP_SEQUENCE, ())

    def _vehicle(self):
        return VehicleEstimate(tuple(float(v) for v in self._position), (0., 0., 0.), yaw_quaternion(self._yaw), 'simulator_pose_declared_ideal')

    def _read_sensors(self):
        t, c = self._sim_time, self.config
        node = self.bank.find(self._position, self._yaw, c.node_xy_tolerance_m, c.node_z_tolerance_m, c.node_yaw_tolerance_rad)
        self.stats['frames'] += 1
        invalid = (SensorStatus('rgb', False, None, None), SensorStatus('depth', False, None, None), SensorStatus('vehicle_state', True, t, 0.))
        capture = None if node is None else self._capture(node)
        if node is None:
            self.served_capture = None
            self.stats['off_node_frames'] += 1
            frame = SensorFrame(t, self._vehicle(), invalid, None, None, ())
        elif c.repeated_capture == 'invalid' and capture.rgb_sha256 in self._served_rgb:
            self.served_capture = None
            self.stats['repeated_capture_frames'] += 1
            frame = SensorFrame(t, self._vehicle(), invalid, None, None, ())
        else:
            self._served_rgb.add(capture.rgb_sha256)
            self.served_capture = capture
            self.stats['bank_frames'] += 1
            frame = SensorFrame(t, self._vehicle(), tuple(SensorStatus(s, True, t, 0.) for s in STREAMS), capture.rgb_path, capture.depth_path, ())
        self.frames.append(frame)
        return frame

    def _capture(self, node):
        """Bank capture for a frame read at node under the configured evidence serving and re-observation model (module docstring)."""
        if self.config.evidence_serving == 'first_only':  # the node's first served capture; no counter or permutation is touched
            if self.config.reobservation_model == 'frame_periodic/1':
                return self.bank.capture_for_visit(node, 0)
            return self.bank.capture_for_visit_frame(node, 0, 0)
        if self.config.reobservation_model == 'frame_periodic/1':
            visit = self._visits.get(node, 0)
            self._visits[node] = visit + 1
            return self.bank.capture_for_visit(node, visit)
        token, visit, frame = self._node_visit.get(node, (None, -1, -1))
        if token != self._arrivals:
            visit, frame = visit + 1, -1
        frame += 1
        self._node_visit[node] = (self._arrivals, visit, frame)
        repeats = self.bank.repeats(node)
        cycle, position = divmod(visit, repeats)
        order = np.random.default_rng([self.config.reobservation_seed, zlib.crc32(node.encode()), cycle]).permutation(repeats)
        if self.config.reobservation_model == 'visit_frame_permutation/3':
            return self.bank.capture_for_visit_frame(node, int(order[position]), frame)
        return self.bank.capture_for_visit(node, int(order[position]))

    def _count_flags(self, flags):
        for flag in flags:
            self.stats['time_model_flags'][flag] = self.stats['time_model_flags'].get(flag, 0) + 1

    def _command(self, command):
        kind = command.kind.value
        gap, flags = self.time_model.gap_s(self._last_kind, kind)
        self._count_flags(flags)
        start = self._sim_time + gap
        self._sim_time = start
        briefing, distance = self._scenario.briefing, 0.
        if command.kind in (CommandKind.TAKEOFF, CommandKind.GOTO):
            goal = np.asarray(command.position_enu, float)
            if not briefing.world_bounds.contains(goal[0], goal[1]):
                return CommandOutcome(True, False, start, start, 0., 'goal_outside_world')
            distance = float(np.linalg.norm(goal - self._position))
            if command.kind == CommandKind.TAKEOFF:
                duration, flags = self.time_model.takeoff_s(goal[2] - briefing.launch_zone.center_enu[2])
            else:
                change = 0. if command.yaw_enu_rad is None else abs(wrap_angle(command.yaw_enu_rad - self._yaw))
                duration, flags = self.time_model.goto_s(distance, command.max_speed_m_s, change)
            self._position = goal
            if command.yaw_enu_rad is not None:
                self._yaw = float(command.yaw_enu_rad)
            self._arrivals += 1  # a completed TAKEOFF/GOTO starts a new visit (visit_permutation/2)
        elif command.kind == CommandKind.HOLD:
            duration, flags = self.time_model.hold_s(command.hold_s)
        else:
            zone_z = briefing.landing_zone.center_enu[2]
            height = float(self._position[2] - zone_z)
            duration, flags = self.time_model.land_s(height)
            distance = max(height - self.config.rest_height_m, 0.)
            self._position = np.array([self._position[0], self._position[1], zone_z + self.config.rest_height_m], float)
        self._count_flags(flags)
        if not (math.isfinite(duration) and duration > 0):
            raise BackendError(f'time model gave a non-positive duration {duration} for {kind}')
        self._sim_time = start + duration
        self._last_kind = kind
        self.stats['commands'][kind] = self.stats['commands'].get(kind, 0) + 1
        return CommandOutcome(True, True, start, self._sim_time, distance, None)

    def _evaluator_truth(self):
        return EvaluatorTruth(self._sim_time, tuple(float(v) for v in self._position), None, None, self.pre_episode_teleports, 0, self._evaluator_spec)

    def _health(self):
        return HealthReport(True, None, self._sim_time)

    def _stop(self):
        self.served_capture = None


def make_cached_perceive_fn(backend, bank, rgb_camera, ground_height_fn, depth_camera=None, depth_range_m=None, localization=None):
    """perceive_fn for MissionExecutive: frozen detector result of the served capture + perception.perceive."""
    def perceive_frame(frame):
        detector_result = depth = None
        if frame.rgb_path is not None:
            capture = backend.served_capture
            if capture is None or capture.rgb_path != frame.rgb_path:
                raise PerceptionError('frame RGB does not match the capture served by the backend')
            detector_result = bank.detection(capture.rgb_sha256)
            if depth_camera is not None and frame.depth_path is not None:
                depth = bank.depth(capture)
        return perceive(frame.sim_time_s, frame.vehicle, frame.sensors, detector_result, rgb_camera, ground_height_fn, depth,
                        depth_camera if depth is not None else None, depth_range_m if depth is not None else None, localization)
    return perceive_frame
