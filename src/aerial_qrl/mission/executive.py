"""Mission executive (episode state machine) and deterministic baseline policies for the public mission (M5).

State machine
    INITIALIZING -> TAKEOFF -> decision loop (SEARCH / INSPECT / REPORT, any order) -> RETURN -> LAND -> COMPLETE,
    with FAILURE (before the decision loop) or ABORT (inside it) and a reason from any phase. The runner owns
    backend.reset (it needs the private evaluator file) and end_episode/stop; the executive calls begin_episode.

Decisions (semi-Markov)
    After TAKEOFF every decision builds a PublicObservation from the last SensorFrame (its vehicle estimate and
    sensor statuses; sim_time_s is that frame's time), CoverageGrid sector fractions, the tracker's confirmed
    public tracks (all tracks with max_score when MissionConfig.public_tentative_tracks, task flood-sar-village/1,
    ADR 010), valid planning.inspection_views of those unreported tracks, remaining time, the
    ReturnReserve estimate and the submitted reports. The policy returns an Action, the executive runs it as a
    macro and records a DecisionRecord with duration_s = next.sim_time_s - observation.sim_time_s (measured
    simulator time, including perception and bookkeeping latency while the simulator runs) and
    discount = gamma_ref ** (duration_s / reference_duration_s).
    REWARD: contracts.Transition requires a finite reward for valid transitions, but rewards are evaluator-side
    and computed offline. DecisionRecord therefore keeps reward None and converts with with_reward() offline;
    invalid (infrastructure) records are valid contracts.Transition objects as they are.

Macros (every macro ends with read_sensors so the next observation is a fresh frame)
    SEARCH_SECTOR  policy.search_goal(sector_id) -> GOTO (goal yaw, search speed), HOLD capture_settle_s, capture.
    INSPECT_VIEW   GOTO view (yaw toward the track, inspect speed), HOLD settle, inspect_frames captures separated
                   by HOLD inspect_interval_s.
    REPORT         appends SubmittedReport(track_id, action.report_position_enu, decision sim time); no movement.
                   HOLD report_hold_s then capture: a declared report-transmission time that keeps duration_s > 0
                   (the contract requires it). Correctness is never revealed.
    RETURN         climb in place to the transit altitude if below it, GOTO above the landing-zone centre, LAND,
                   capture without perception. Terminal ('mission_complete').
    A capture reads the frame, calls perceive_fn(frame) (object with .candidates), and only when RGB and
    vehicle_state are valid updates the tracker (in-view predicate from the camera footprint of the estimate)
    and CoverageGrid.record_frame. Every GOTO leg is checked with StaticObstacleMap.check_segment(clearance
    radius) before any command; an infeasible action is not flown, is logged as rejected, the policy receives
    feedback and chooses again (max_consecutive_rejections, then a forced RETURN).

Budget and failures
    elapsed_s is measured from the first frame after begin_episode (pad frame; a lower bound on the true episode
    time). ReturnReserve.must_return forces RETURN (forced='return_reserve'); the mission deadline makes the
    decision that crosses it terminal ('mission_deadline', zero bootstrap), after which the vehicle returns and
    lands outside the decision process. check_health runs after every macro. An incomplete command ends the
    decision process with a 'safety_abort' terminal record when a fresh observation is still available; any
    exception (backend, health fault, contract) gives an invalid INFRASTRUCTURE record. Both ABORT; the runner
    lands the vehicle while the backend is ACTIVE.

Public/private
    The executive and policies receive only public objects: MissionBriefing, the public static obstacle map,
    SensorFrame, perception candidates, public tracks, the coverage raster, ReturnReserve and their own plan
    state. They never receive EvaluatorSpec/EvaluatorTruth, target poses/IDs, segmentation or report correctness;
    after_macro is a runner hook whose return value is ignored. Every record the executive logs passes
    leakage.assert_public_payload.

View graph (optional; task flood-sar-viewgraph/1, docs/adr/006-fast-decision-environment.md §3)
    MissionExecutive(..., view_graph=None) keeps flood-sar-mission/1 unchanged. With a view_graph.ViewGraph every requested view
    that inspection_views marks valid is snapped to an inspect node (view_graph.snap_views, checks re-run at the node pose with
    planning.evaluate_viewpoint); the observation offers only node poses (view_id f'{track_id}-{node_id}'), every snap record
    is logged to public/view_snaps.jsonl, and a SEARCH_SECTOR goal that is not a graph node (ViewGraph.find_node, default
    tolerances) is rejected with 'goal_not_graph_node'. TAKEOFF, RETURN and LAND are not restricted to the graph.
    MissionConfig.max_inspections_per_track = 0 offers no inspection views at all (a search-only task: every policy then
    flies only search nodes, e.g. the pilot task covered by search-node banks).
    A policy that defines use_view_graph(view_graph) is bound to the graph before the first decision (bind_view_graph, called here
    and by the runners before they record the plan summary): InformationCostPolicy then scores only the graph search and inspect
    nodes (validation-mission/1 phase 2).

Units, frames, clock: metres, seconds, radians; world ENU, yaw counter-clockwise from East; simulator seconds.
Every numeric default is an engineering default (provenance engineering_default_unfrozen).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional, Tuple

import numpy as np

from .backend import CommandKind, FlightCommand
from .contracts import (GNSS_SCHEMA_VERSION, INFRASTRUCTURE, PROVENANCE, SCHEMA_VERSION, TERMINAL_REASONS, TIME_TOLERANCE_S,
                        Action, ActionType, Contract, ContractError, LocalReportReceipt, PublicObservation,
                        SemiMarkovDiscount, SubmittedReport, Transition, _choice, _count, _nested, _optional_real, _real,
                        dedup_position_xy)
from .relocalization import AnchorSet, anchor_statuses
from .frames import quat_to_matrix
from .leakage import assert_public_payload
from .perception import CameraModel, CameraMount, footprint_fn
from .planning import Candidate as GoalCandidate, CoverageBaseline, CoverageGrid, InformationCostPlanner, inspection_views, points_in_polygon
from .view_graph import ViewGraph, node_validity_fn, snap_views, task_version_for


class Phase(str, Enum):
    INITIALIZING = 'INITIALIZING'
    TAKEOFF = 'TAKEOFF'
    SEARCH = 'SEARCH'
    INSPECT = 'INSPECT'
    REPORT = 'REPORT'
    RETURN = 'RETURN'
    LAND = 'LAND'
    COMPLETE = 'COMPLETE'
    FAILURE = 'FAILURE'
    ABORT = 'ABORT'
    RELOCALIZE = 'RELOCALIZE'   # flood-sar-village-gnss/1 only; never emitted by a legacy run


ACTION_PHASE = {ActionType.SEARCH_SECTOR: Phase.SEARCH, ActionType.INSPECT_VIEW: Phase.INSPECT,
                ActionType.REPORT: Phase.REPORT, ActionType.RETURN: Phase.RETURN,
                ActionType.RELOCALIZE: Phase.RELOCALIZE}
assert set(ACTION_PHASE) == set(ActionType), 'every action type needs a phase: no action may borrow another one'


class ExecutiveError(RuntimeError):
    pass


class CommandIncomplete(ExecutiveError):
    """A command was accepted by a live backend but did not complete (goal timeout, rejection)."""


class SafetyAbort(ExecutiveError):
    """A public safety rule stops the decision process (blocked return path, infeasible forced return)."""


# --- Configuration -------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionConfig(Contract):
    """Public executive/policy configuration (metres, m/s, s). Altitudes are above the flat planning ground ground_z_m."""
    camera_mount: CameraMount
    discount: SemiMarkovDiscount
    altitude_agl_m: float = 15.
    transit_altitude_agl_m: float = 15.
    ground_z_m: float = 0.
    takeoff_speed_m_s: float = 1.5
    search_speed_m_s: float = 3.
    inspect_speed_m_s: float = 2.
    transit_speed_m_s: float = 3.
    capture_settle_s: float = 2.
    inspect_frames: int = 3
    inspect_interval_s: float = 1.
    report_hold_s: float = 1.
    clearance_radius_m: float = 3.
    goal_bounds_margin_m: float = 0.  # extra inset for generated goals (waypoints, views, candidates) beyond clearance_radius_m
    overlap_fraction: float = .2
    footprint_max_range_m: float = 60.
    coverage_cell_m: float = 2.
    inspect_radii_m: Tuple[float, ...] = (15., 20.)
    inspect_angles: int = 8
    centre_window_fraction: float = .5
    sigma_scale: float = 2.
    vertical_sigma_m: float = .5
    in_view_max_range_m: float = 60.
    report_min_detections: int = 4
    report_min_views: int = 2
    report_merge_radius_m: float = 5.
    view_separation_m: float = 5.
    max_inspections_per_track: int = 3
    public_tentative_tracks: bool = False  # task flood-sar-village/1 (ADR 010): tentative tracks and their max_score are public
    recheck_min_score: Optional[float] = None  # CoverageRecheckPolicy (ADR 011): inspect a track only if its max_score >= this
    recheck_inspections: int = 1               # CoverageRecheckPolicy: inspections per track that qualifies
    first_sight_min_score: Optional[float] = None  # CoverageFirstSightPolicy (diagnostic): report a track at once if max_score >= this
    info_candidate_spacing_m: Optional[float] = None
    info_yaw_count: int = 4
    info_min_gain: float = 1e-3
    info_graph_node_kinds: Tuple[str, ...] = ('search', 'inspect')  # InformationCostPolicy candidates under a view graph
    max_consecutive_rejections: int = 20
    max_decisions: int = 2000
    provenance: str = 'engineering_default_unfrozen'

    def __post_init__(self):
        positive = dict(low=0, open_low=True)
        values = {k: _real(getattr(self, k), k, **positive) for k in (
            'altitude_agl_m', 'transit_altitude_agl_m', 'takeoff_speed_m_s', 'search_speed_m_s', 'inspect_speed_m_s',
            'transit_speed_m_s', 'capture_settle_s', 'inspect_interval_s', 'report_hold_s', 'footprint_max_range_m',
            'coverage_cell_m', 'centre_window_fraction', 'sigma_scale', 'vertical_sigma_m', 'in_view_max_range_m',
            'report_merge_radius_m', 'view_separation_m')}
        radii = tuple(_real(r, 'inspect_radii_m', **positive) for r in self.inspect_radii_m)
        if not radii:
            raise ContractError('inspect_radii_m: at least one radius')
        self._set(**values, inspect_radii_m=radii, camera_mount=_nested(CameraMount, self.camera_mount, 'camera_mount'),
                  discount=_nested(SemiMarkovDiscount, self.discount, 'discount'), ground_z_m=_real(self.ground_z_m, 'ground_z_m'),
                  clearance_radius_m=_real(self.clearance_radius_m, 'clearance_radius_m', low=0),
                  goal_bounds_margin_m=_real(self.goal_bounds_margin_m, 'goal_bounds_margin_m', low=0),
                  overlap_fraction=_real(self.overlap_fraction, 'overlap_fraction', low=0, high=1, open_high=True),
                  info_min_gain=_real(self.info_min_gain, 'info_min_gain', low=0),
                  info_candidate_spacing_m=None if self.info_candidate_spacing_m is None else
                  _real(self.info_candidate_spacing_m, 'info_candidate_spacing_m', **positive),
                  provenance=_choice(self.provenance, 'provenance', PROVENANCE))
        for key in ('inspect_frames', 'inspect_angles', 'report_min_detections', 'report_min_views', 'info_yaw_count',
                    'max_consecutive_rejections', 'max_decisions'):
            _count(getattr(self, key), key, 1)
        _count(self.max_inspections_per_track, 'max_inspections_per_track', 0)  # 0: no inspection views are offered
        if not isinstance(self.public_tentative_tracks, bool):
            raise ContractError('public_tentative_tracks: expected a boolean')
        self._set(recheck_min_score=_optional_real(self.recheck_min_score, 'recheck_min_score', low=0, high=1),
                  first_sight_min_score=_optional_real(self.first_sight_min_score, 'first_sight_min_score', low=0, high=1))
        _count(self.recheck_inspections, 'recheck_inspections', 1)
        kinds = tuple(_choice(k, 'info_graph_node_kinds', ('search', 'inspect')) for k in self.info_graph_node_kinds)
        if not kinds or len(set(kinds)) != len(kinds):
            raise ContractError('info_graph_node_kinds: expected distinct kinds from search, inspect')
        self._set(info_graph_node_kinds=kinds)
        rpy = self.camera_mount.rpy_rad
        if abs(rpy[0]) > 1e-6 or abs(rpy[2]) > 1e-6 or not 0 < rpy[1] <= math.pi / 2:
            raise ContractError('camera_mount: planning needs zero roll/yaw and a pitch in (0, pi/2]')

    @property
    def mount_pitch_rad(self):
        return self.camera_mount.rpy_rad[1]

    @property
    def goal_inset_m(self):
        """Inset from the world bounds for every generated goal. The safety checks keep clearance_radius_m, so a vehicle
        estimate within goal_bounds_margin_m of a goal stays inside them (segment_mission_absent_v1 run_20260915T090154Z:
        waypoint on the inset line, estimate 3 cm beyond it, SafetyAbort)."""
        return self.clearance_radius_m + self.goal_bounds_margin_m

    @property
    def search_z(self):
        return self.ground_z_m + self.altitude_agl_m

    @property
    def inspect_capture_s(self):
        return self.capture_settle_s + (self.inspect_frames - 1) * self.inspect_interval_s


@dataclass(frozen=True)
class SearchGoal:
    """Concrete public goal flown by SEARCH_SECTOR; goal_id is a policy plan identifier."""
    goal_id: str
    sector_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float

    def to_dict(self):
        return {'goal_id': self.goal_id, 'sector_id': self.sector_id, 'position_enu': list(self.position_enu), 'yaw_rad': self.yaw_rad}


# --- Records ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionRecord:
    """One semi-Markov decision with the reward left to the evaluator (module docstring, REWARD).

    Valid records satisfy every contracts.Transition rule except the reward type; invalid ones are checked by
    constructing the contracts.Transition itself. forced names an executive override ('return_reserve',
    'rejection_cap', 'max_decisions') or None for a policy choice.
    """
    index: int
    observation: PublicObservation
    action: Action
    next_observation: Optional[PublicObservation]
    duration_s: Optional[float]
    discount: Optional[float]
    terminated: bool
    truncated: bool
    reason: Optional[str]
    valid: bool
    forced: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.valid:
            Transition(self.observation, self.action, self.next_observation, None, self.duration_s, None, self.terminated,
                       self.truncated, self.reason, False)
            return
        before, after = self.observation, self.next_observation
        if after is None or self.truncated:
            raise ContractError('valid decision requires next_observation and is never truncated here')
        duration = _real(self.duration_s, 'duration_s', low=0, open_low=True)
        _real(self.discount, 'discount', low=0, high=1, open_low=True)
        if not math.isclose(after.sim_time_s - before.sim_time_s, duration, abs_tol=TIME_TOLERANCE_S) or \
                not math.isclose(after.elapsed_s - before.elapsed_s, duration, abs_tol=TIME_TOLERANCE_S):
            raise ContractError('duration_s must equal the measured sim-time and elapsed-time differences')
        if after.reports[:len(before.reports)] != before.reports:
            raise ContractError('submitted-report history is append-only')
        if self.terminated:
            _choice(self.reason, 'reason', TERMINAL_REASONS)
        elif self.reason is not None:
            raise ContractError('reason only for terminated or invalid decisions')

    def bootstrap_weight(self):
        if not self.valid:
            raise ContractError('invalid decision cannot be bootstrapped')
        return 0. if self.terminated else self.discount

    def with_reward(self, reward):
        """contracts.Transition with an evaluator-side reward (offline use only)."""
        return Transition(self.observation, self.action, self.next_observation, reward if self.valid else None, self.duration_s,
                          self.discount, self.terminated, self.truncated, self.reason, self.valid)

    def to_dict(self):
        return {'index': self.index, 'observation': self.observation.to_dict(), 'action': self.action.to_dict(),
                'next_observation': None if self.next_observation is None else self.next_observation.to_dict(),
                'reward': None, 'reward_source': 'evaluator_offline', 'duration_s': self.duration_s, 'discount': self.discount,
                'terminated': self.terminated, 'truncated': self.truncated, 'reason': self.reason, 'valid': self.valid,
                'forced': self.forced, 'error': self.error}


@dataclass(frozen=True)
class MissionOutcome:
    """Public end state of the executive. returned/landed are the public command outcomes, not evaluator flags."""
    phase: Phase
    reason: Optional[str]
    termination_reason: Optional[str]
    decisions: int
    rejections: int
    valid_decisions: int
    invalid_decisions: int
    reports: Tuple[SubmittedReport, ...]
    returned: bool
    landed: bool
    estimated_distance_m: float
    elapsed_s: Optional[float]
    frames: int

    def to_dict(self):
        return {'phase': self.phase.value, 'reason': self.reason, 'termination_reason': self.termination_reason,
                'decisions': self.decisions, 'rejections': self.rejections, 'valid_decisions': self.valid_decisions,
                'invalid_decisions': self.invalid_decisions, 'reports': [r.to_dict() for r in self.reports],
                'returned': self.returned, 'landed': self.landed, 'estimated_distance_m': self.estimated_distance_m,
                'elapsed_s': self.elapsed_s, 'frames': self.frames}


class MemoryLog:
    """Append-only in-memory writer: records[relative_path] is a list of JSON round-tripped records."""

    def __init__(self):
        self.records = {}

    def write(self, relative, record):
        self.records.setdefault(relative, []).append(json.loads(json.dumps(record, allow_nan=False, default=str)))


class JsonlLog:
    """Append-only JSON-lines writer under run_dir; adds host wall time (logs only, never a decision input)."""

    def __init__(self, run_dir):
        self.run_dir, self.t0 = Path(run_dir), time.monotonic()

    def write(self, relative, record):
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({'wall_elapsed_s': round(time.monotonic() - self.t0, 3), **record}, allow_nan=False, default=str)
        with open(path, 'a') as f:
            f.write(line + '\n')


def yaw_of(attitude_wxyz):
    r = quat_to_matrix(attitude_wxyz)
    return math.atan2(r[1, 0], r[0, 0])


def _stream(frame_or_observation, name):
    return next((s for s in frame_or_observation.sensors if s.stream == name), None)


# --- Policies --------------------------------------------------------------------------------

class MissionPolicy:
    """Deterministic public policy. act(observation) -> Action; search_goal(sector_id) -> SearchGoal or None names the
    goal SEARCH_SECTOR flies; feedback(action, executed, reason, goal) reports execution or rejection (never correctness)."""
    name = 'policy'

    def __init__(self, briefing, obstacle_map, config, reserve):
        self.briefing, self.obstacle_map, self.config, self.reserve = briefing, obstacle_map, config, reserve

    def act(self, observation):
        raise NotImplementedError

    def search_goal(self, sector_id):
        return None

    def feedback(self, action, executed, reason=None, goal=None):
        pass

    def use_view_graph(self, view_graph):
        """Task flood-sar-viewgraph/1 hook (bind_view_graph). The default changes nothing: the executive checks goals against the graph."""
        return None


class EvidencePolicy(MissionPolicy):
    """Shared inspect/report rules (public tracks and the policy's own evidence bookkeeping only).

    A track counts as reported when its ID was reported or it lies within report_merge_radius_m (XY) of an earlier
    report; such a track is never reported or inspected again. Distinct viewpoints of a track: vehicle XY positions at
    observations where its public detection count increased, merged within view_separation_m (detections gained before
    confirmation are credited to the first observation that shows the track, a conservative undercount). A track with
    >= report_min_detections detections from >= report_min_views viewpoints is reported (lowest track_id first). Otherwise
    the cheapest valid view (estimated_duration_s, then view_id) that is not rejected and lies farther than
    view_separation_m from credited and already inspected positions is inspected; with no such view or after
    max_inspections_per_track inspections the track is abandoned for inspection (it may still be reported later).
    """

    def __init__(self, briefing, obstacle_map, config, reserve):
        super().__init__(briefing, obstacle_map, config, reserve)
        self._detections, self._credited, self._inspected, self._inspections = {}, {}, {}, {}
        self._abandoned, self._rejected_views = set(), set()

    def _update_evidence(self, observation):
        xy = observation.vehicle.position_enu[:2]
        for track in observation.tracks:
            if track.detections > self._detections.get(track.track_id, 0):
                points = self._credited.setdefault(track.track_id, [])
                if all(math.dist(xy, p) > self.config.view_separation_m for p in points):
                    points.append(tuple(xy))
            self._detections[track.track_id] = track.detections

    def is_reported(self, track, observation):
        """Local-frame association only (contracts.dedup_position_xy): never a scoring comparison."""
        return any(r.track_id == track.track_id
                   or math.dist(dedup_position_xy(r), track.position_enu[:2]) <= self.config.report_merge_radius_m
                   for r in observation.reports)

    def _evidence_action(self, observation, allow_report=True):
        """The REPORT or INSPECT_VIEW this evidence rule wants next, or None.

        `allow_report=False` asks the same question with the report branch suppressed, i.e. "is there
        inspection work still outstanding?". A policy that HOLDS its reports needs that: with reports allowed,
        one ready track masks every remaining inspection, and the holder would stop gathering evidence the
        moment its first candidate became reportable.
        """
        c = self.config
        open_tracks = [t for t in sorted(observation.tracks, key=lambda t: t.track_id) if not self.is_reported(t, observation)]
        for track in open_tracks if allow_report else ():
            if track.detections >= c.report_min_detections and len(self._credited.get(track.track_id, ())) >= c.report_min_views:
                return Action(ActionType.REPORT, track_id=track.track_id, report_position_enu=track.position_enu)
        options = []
        for track in open_tracks:
            if track.track_id in self._abandoned:
                continue
            tried = self._credited.get(track.track_id, []) + self._inspected.get(track.track_id, [])
            views = [v for v in observation.views if v.track_id == track.track_id and v.view_id not in self._rejected_views
                     and all(math.dist(v.position_enu[:2], p) > c.view_separation_m for p in tried)]
            if not views or self._inspections.get(track.track_id, 0) >= self._inspection_cap(track):
                self._abandoned.add(track.track_id)
                continue
            options += views
        if not options:
            return None
        view = min(options, key=lambda v: (v.estimated_duration_s, v.view_id))
        return Action(ActionType.INSPECT_VIEW, view_id=view.view_id, track_id=view.track_id)

    def _inspection_cap(self, track):
        """Inspections allowed for a track (max_inspections_per_track; CoverageRecheckPolicy overrides)."""
        return self.config.max_inspections_per_track

    def _evidence_feedback(self, action, executed, goal=None):
        """goal: the ViewOption flown by INSPECT_VIEW (None when the view was unknown)."""
        if action.action_type == ActionType.INSPECT_VIEW:
            if executed:
                self._inspections[action.track_id] = self._inspections.get(action.track_id, 0) + 1
                if goal is not None:
                    self._inspected.setdefault(action.track_id, []).append(tuple(goal.position_enu[:2]))
            else:
                self._rejected_views.add(action.view_id)
        elif action.action_type == ActionType.REPORT and not executed:
            self._abandoned.add(action.track_id)


class CoverageBaselinePolicy(EvidencePolicy):
    """Systematic coverage: planning.CoverageBaseline waypoints in plan order (sectors by prior weight), one per decision.
    Waypoints with a plan issue at the waypoint itself are skipped at construction; rejected waypoints are skipped.
    RETURN when no waypoint is pending (the executive forces RETURN earlier when the reserve requires it)."""
    name = 'coverage_baseline'

    def __init__(self, briefing, obstacle_map, config, reserve):
        super().__init__(briefing, obstacle_map, config, reserve)
        c = config
        start = (*briefing.launch_zone.center_enu[:2], c.search_z)
        self.plan = CoverageBaseline(briefing, obstacle_map, c.mount_pitch_rad, c.altitude_agl_m, c.overlap_fraction,
                                     c.clearance_radius_m, c.footprint_max_range_m, c.ground_z_m, bounds_margin_m=c.goal_inset_m).plan(start)
        self.status = {w.index: None for w in self.plan.waypoints}
        for issue in self.plan.issues:
            if issue.kind == 'waypoint':
                self.status[issue.index] = f'plan_issue:{issue.reason}'

    def _pending(self, sector_id=None):
        return next((w for w in self.plan.waypoints if self.status[w.index] is None and sector_id in (None, w.sector_id)), None)

    def act(self, observation):
        self._update_evidence(observation)
        action = self._evidence_action(observation)
        if action is not None:
            return action
        waypoint = self._pending()
        return Action(ActionType.RETURN) if waypoint is None else Action(ActionType.SEARCH_SECTOR, sector_id=waypoint.sector_id)

    def search_goal(self, sector_id):
        w = self._pending(sector_id)
        return None if w is None else SearchGoal(f'wp{w.index}', w.sector_id, w.position_enu, w.yaw_rad)

    def feedback(self, action, executed, reason=None, goal=None):
        if action.action_type == ActionType.SEARCH_SECTOR and goal is not None:
            self.status[int(goal.goal_id[2:])] = 'flown' if executed else f'rejected:{reason}'
        self._evidence_feedback(action, executed, goal)


class InformationCostPolicy(EvidencePolicy):
    """Greedy information per cost: planning.InformationCostPlanner over its candidate lattice, on a coverage raster owned
    by the policy and updated from observations (vehicle estimate and RGB timestamp when RGB and vehicle_state are valid;
    one capture per observation). Flown or rejected candidates are excluded afterwards. RETURN when no feasible candidate
    remains or the best gain is below info_min_gain (fraction of prior mass). The SEARCH_SECTOR sector is the first briefing
    sector containing the goal XY, else the sector with the nearest vertex mean. Under task flood-sar-viewgraph/1
    (use_view_graph) the candidates are the graph search and inspect nodes instead of the lattice."""
    name = 'information_cost'

    def __init__(self, briefing, obstacle_map, config, reserve):
        super().__init__(briefing, obstacle_map, config, reserve)
        c = config
        self.grid = CoverageGrid(briefing, c.coverage_cell_m, c.mount_pitch_rad, c.footprint_max_range_m, c.ground_z_m)
        self.planner = InformationCostPlanner(briefing, obstacle_map, self.grid, reserve, c.altitude_agl_m, c.clearance_radius_m,
                                              c.capture_settle_s, candidate_spacing_m=c.info_candidate_spacing_m, yaw_count=c.info_yaw_count,
                                              bounds_margin_m=c.goal_inset_m)
        self.candidates = self.planner.candidates()
        self.candidate_source = {'kind': 'lattice', 'count': len(self.candidates)}
        self._excluded, self._goal, self._last_capture_s, self._acted = set(), None, None, False

    def use_view_graph(self, view_graph):
        """Candidates := the graph nodes of config.info_graph_node_kinds (default search and inspect) in graph order
        (candidate_id = node_id), scored by the unchanged planner; takeoff_top is not a candidate. Idempotent for the same graph (sha256). Another graph, a call after the first
        decision, or a node away from the planner altitude (grid ground + altitude_agl_m) raises ExecutiveError."""
        graph = _nested(ViewGraph, view_graph, 'view_graph')
        if self.candidate_source.get('graph_sha256') == graph.sha256:
            return None
        if self.candidate_source['kind'] != 'lattice' or self._acted:
            raise ExecutiveError('information_cost: a view graph is bound once, before the first decision')
        z, kinds = self.grid.ground_z_m + self.config.altitude_agl_m, self.config.info_graph_node_kinds
        nodes = [n for n in graph.nodes if n.kind in kinds]
        off = [n.node_id for n in nodes if abs(n.position_enu[2] - z) > 1e-6]
        if off:
            raise ExecutiveError(f'information_cost: graph nodes away from the planner altitude {z} m: {off[:3]}')
        self.candidates = tuple(GoalCandidate(n.node_id, n.position_enu, n.yaw_rad) for n in nodes)
        self.candidate_source = {'kind': 'view_graph', 'graph_sha256': graph.sha256, 'node_kinds': list(kinds),
                                 'count': len(self.candidates)}
        return None

    def _record_coverage(self, observation):
        rgb, state = _stream(observation, 'rgb'), _stream(observation, 'vehicle_state')
        if rgb is None or not rgb.valid or state is None or not state.valid:
            return
        if self._last_capture_s is None or rgb.timestamp_s > self._last_capture_s + TIME_TOLERANCE_S:
            self.grid.record_capture(rgb.timestamp_s, observation.vehicle.position_enu, yaw_of(observation.vehicle.attitude_wxyz))
            self._last_capture_s = rgb.timestamp_s

    def sector_of(self, position_enu):
        for sector in self.briefing.search_sectors:
            if points_in_polygon([position_enu[:2]], sector.polygon_enu)[0]:
                return sector.sector_id
        return min(self.briefing.search_sectors,
                   key=lambda s: (math.dist(np.mean(s.polygon_enu, axis=0), position_enu[:2]), s.sector_id)).sector_id

    def act(self, observation):
        self._acted = True
        self._record_coverage(observation)
        self._update_evidence(observation)
        action = self._evidence_action(observation)
        if action is not None:
            return action
        options = tuple(c for c in self.candidates if c.candidate_id not in self._excluded)
        best = self.planner.choose(observation.vehicle.position_enu, observation.elapsed_s, options)
        if best is None or best.gain < self.config.info_min_gain:
            self._goal = None
            return Action(ActionType.RETURN)
        self._goal = SearchGoal(best.candidate_id, self.sector_of(best.position_enu), best.position_enu, best.yaw_rad)
        return Action(ActionType.SEARCH_SECTOR, sector_id=self._goal.sector_id)

    def search_goal(self, sector_id):
        return self._goal if self._goal is not None and self._goal.sector_id == sector_id else None

    def feedback(self, action, executed, reason=None, goal=None):
        if action.action_type == ActionType.SEARCH_SECTOR and goal is not None:
            self._excluded.add(goal.goal_id)
            self._goal = None
        self._evidence_feedback(action, executed, goal)


class CoverageRecheckPolicy(CoverageBaselinePolicy):
    """Coverage with a threshold re-check rule (ADR 011; after Schedl et al. 2021): a public track is inspected only when its
    public max_score >= recheck_min_score, at most min(recheck_inspections, max_inspections_per_track) times, always from the
    cheapest offered view (the shared EvidencePolicy choice); other tracks
    are never inspected but may still be reported by the shared report rule. Requires task flood-sar-village/1
    (public_tentative_tracks, so scores are public) and a declared recheck_min_score."""
    name = 'coverage_recheck'

    def __init__(self, briefing, obstacle_map, config, reserve):
        if not config.public_tentative_tracks or config.recheck_min_score is None:
            raise ContractError('coverage_recheck needs public_tentative_tracks and a recheck_min_score')
        super().__init__(briefing, obstacle_map, config, reserve)

    def _inspection_cap(self, track):
        c = self.config
        if track.max_score is None or track.max_score < c.recheck_min_score:
            return 0
        return min(c.recheck_inspections, c.max_inspections_per_track)


class CoverageFirstSightPolicy(CoverageBaselinePolicy):
    """Diagnostic rule (pilot-village-v1): coverage that reports an unreported public track as soon as its public max_score >=
    first_sight_min_score (lowest track_id first) and otherwise never inspects. It measures what a report-on-first-sight
    rule achieves without learning. Requires task flood-sar-village/1 and a declared first_sight_min_score."""
    name = 'coverage_first_sight'

    def __init__(self, briefing, obstacle_map, config, reserve):
        if not config.public_tentative_tracks or config.first_sight_min_score is None:
            raise ContractError('coverage_first_sight needs public_tentative_tracks and a first_sight_min_score')
        super().__init__(briefing, obstacle_map, config, reserve)

    def _evidence_action(self, observation):
        for track in sorted(observation.tracks, key=lambda t: t.track_id):
            if (not self.is_reported(track, observation) and track.max_score is not None
                    and track.max_score >= self.config.first_sight_min_score and track.track_id not in self._abandoned):
                return Action(ActionType.REPORT, track_id=track.track_id, report_position_enu=track.position_enu)
        return None


POLICIES = {CoverageBaselinePolicy.name: CoverageBaselinePolicy, InformationCostPolicy.name: InformationCostPolicy,
            CoverageRecheckPolicy.name: CoverageRecheckPolicy, CoverageFirstSightPolicy.name: CoverageFirstSightPolicy}


def bind_view_graph(policy, view_graph):
    """policy.use_view_graph(view_graph) when the policy defines it; returns the policy (MissionExecutive and the runners)."""
    hook = getattr(policy, 'use_view_graph', None)
    if hook is not None:
        hook(view_graph)
    return policy


# --- Executive -------------------------------------------------------------------------------

class MissionExecutive:
    """Runs one episode on a PREPARED backend (module docstring). run() returns a MissionOutcome and never raises for
    backend, contract or policy failures; KeyboardInterrupt propagates. after_macro(info) is a runner hook called after
    TAKEOFF, every executed decision and the post-deadline return, with public info only; its result is ignored.
    view_graph: optional view_graph.ViewGraph (task flood-sar-viewgraph/1, module docstring); None keeps flood-sar-mission/1."""

    def __init__(self, backend, briefing, obstacle_map, config, perceive_fn, tracker, coverage_grid, reserve, policy, log,
                 after_macro=None, clock=time.monotonic, view_graph=None, georeference=None, anchor_set=None,
                 macro_audit=None):
        self.backend, self.briefing, self.obstacle_map = backend, briefing, obstacle_map
        self.config = _nested(MissionConfig, config, 'config')
        # Task selection. Both None: flood-sar-mission/1, byte-for-byte as before. Both given:
        # flood-sar-village-gnss/1. One without the other is a configuration error, never a silent half-task.
        if (georeference is None) != (anchor_set is None):
            raise ExecutiveError('georeference and anchor_set must be given together (task flood-sar-village-gnss/1)')
        if georeference is not None and any(v != 0. for v in georeference.gauge_enu):
            # review 1, ruling 2: v1 runs with a zero runtime gauge. contracts.check_scenario_pair validates
            # evaluator targets against the LOCAL world bounds, so a non-zero gauge is not representable end to
            # end; gauge invariance stays a service-level analytical test, explicitly bounded by that contract.
            raise ExecutiveError(f'georeference.gauge_enu must be zero at runtime in v1, got {georeference.gauge_enu}')
        self.georeference = georeference
        self.anchor_set = None if anchor_set is None else _nested(AnchorSet, anchor_set, 'anchor_set')
        self.schema_version = SCHEMA_VERSION if georeference is None else GNSS_SCHEMA_VERSION
        self.view_graph = None if view_graph is None else _nested(ViewGraph, view_graph, 'view_graph')
        if self.view_graph is not None and any(abs(n.pitch_rad - self.config.mount_pitch_rad) > 1e-9 for n in self.view_graph.nodes):
            raise ExecutiveError('view_graph: node camera pitch differs from the config camera mount pitch')
        self.perceive_fn, self.tracker, self.coverage_grid, self.reserve = perceive_fn, tracker, coverage_grid, reserve
        self.policy, self.log, self.after_macro, self.clock = policy, log, after_macro, clock
        # macro_audit(stage, action, snapshot): an optional read-only probe called immediately BEFORE and AFTER
        # every executed macro, with stage in ('before', 'after'). It exists so an auditor can attribute a state
        # change to ONE transition instead of differencing whole episodes, which cannot separate a macro's
        # effect from a different trajectory. Its return value is ignored and it must not mutate anything.
        self.macro_audit = macro_audit
        if self.view_graph is not None:
            bind_view_graph(policy, self.view_graph)
        self.camera = CameraModel.from_sensor_model(briefing.sensor_model, 'rgb', self.config.camera_mount)
        self.sector_ids = {s.sector_id for s in briefing.search_sectors}
        self.phase, self.frame, self.t0 = Phase.INITIALIZING, None, None
        self.reports, self.records = [], []
        self.decisions = self.rejections = self.frames = self.observations = self.perceptions = 0
        self.distance_m, self.returned, self.landed = 0., False, False

    # -- logging --
    def _write(self, relative, record):
        self.log.write(relative, assert_public_payload(record))

    def _write_private(self, relative, record):
        """Evaluator stream: truth-derived diagnostics that must never be read back as an observation.

        It bypasses the public key guard on purpose, so the caller carries the burden: only `evaluator/` paths
        are allowed here, and nothing written through this method is ever returned by `_observation`.
        """
        if not relative.startswith('evaluator/'):
            raise ExecutiveError(f'_write_private is only for the evaluator stream, got {relative!r}')
        self.log.write(relative, record)

    def _event(self, kind, **fields):
        self._write('events.jsonl', {'kind': kind, 'phase': self.phase.value, **fields})

    def _set_phase(self, phase, reason=None):
        if phase != self.phase:
            self._event('phase', previous=self.phase.value, next=phase.value, reason=reason)
            self.phase = phase

    # -- backend primitives --
    def _command(self, kind, position=None, speed=None, hold_s=None, yaw=None):
        command = FlightCommand(kind, None if position is None else tuple(float(v) for v in position), speed, hold_s, yaw)
        outcome = self.backend.command(command)
        self.distance_m += outcome.estimated_distance_m
        self._event('command', command=command.to_dict(), outcome=outcome.to_dict())
        if not outcome.completed:
            raise CommandIncomplete(f'{command.kind.value} not completed: {outcome.reason}')
        return outcome

    def _capture(self, perceive=True):
        frame = self.backend.read_sensors()
        self.frames += 1
        self.frame = frame
        self._write('public/frames.jsonl', {'frame_index': self.frames, 'frame': frame.to_dict()})
        if not perceive:
            return frame
        result = self.perceive_fn(frame)
        rgb, state = _stream(frame, 'rgb'), _stream(frame, 'vehicle_state')
        record = {'frame_index': self.frames, 'sim_time_s': frame.sim_time_s,
                  'perception': result.to_dict() if hasattr(result, 'to_dict') else {'candidates': [c.to_dict() for c in result.candidates]},
                  'tracker_update': None, 'coverage_recorded': False}
        if rgb is not None and rgb.valid and state is not None and state.valid:
            in_view = footprint_fn(self.camera, self.camera.pose(frame.vehicle), self.config.in_view_max_range_m)
            record['tracker_update'] = self.tracker.update(tuple(result.candidates), rgb.timestamp_s, in_view).to_dict()
            record['coverage_recorded'] = self.coverage_grid.record_frame(frame)
        elif tuple(result.candidates):
            raise ExecutiveError('perception returned candidates for a frame without valid RGB and vehicle state')
        self.perceptions += 1
        self._write('public/perception.jsonl', record)
        self._write('public/tracks.jsonl', {'frame_index': self.frames, 'sim_time_s': frame.sim_time_s,
                                            'tracks': [s.to_dict() for s in self.tracker.snapshot()]})
        return frame

    def _audit_snapshot(self):
        """Read-only digest of everything a macro could change. Used by macro_audit only.

        It deliberately includes the counters that would move if a macro consumed a cached observation
        (`bank_frames`, `repeated_capture_frames`), the perception/tracker/coverage state that would move if the
        detector ran, and the georeference state that a RELOCALIZE is allowed to move.
        """
        stats = dict(getattr(self.backend, 'stats', {}) or {})
        tracks = tuple(sorted((s.track_id, s.hits, s.frames_with_detection, tuple(s.position_enu))
                              for s in self.tracker.snapshot()))
        coverage = tuple(sorted((c.sector_id, c.covered_fraction) for c in self.coverage_grid.sector_coverage()))
        snapshot = {'sim_time_s': None if self.frame is None else self.frame.sim_time_s,
                    'frames': self.frames, 'perception_records': self.perceptions,
                    'bank_frames': stats.get('bank_frames'), 'off_node_frames': stats.get('off_node_frames'),
                    'repeated_capture_frames': stats.get('repeated_capture_frames'),
                    'tracks': tracks, 'tracks_digest': hashlib.sha256(repr(tracks).encode()).hexdigest(),
                    'coverage_digest': hashlib.sha256(repr(coverage).encode()).hexdigest(),
                    'reports': len(self.reports)}
        if self.georeference is not None:
            summary = self.georeference.public_summary(snapshot['sim_time_s'])
            snapshot['georeference'] = summary.to_dict()
            snapshot['measurements'] = len(self.georeference.measurements)
            snapshot['submissions'] = len(self.georeference.submissions)
        return snapshot

    def _audit(self, stage, action):
        if self.macro_audit is not None:
            self.macro_audit(stage, action, self._audit_snapshot())

    def _hook(self, **info):
        if self.after_macro is not None:
            self.after_macro(dict(info, phase=self.phase.value, frame_index=self.frames))

    # -- observation --
    def _observation(self):
        c, frame = self.config, self.frame
        t, position = frame.sim_time_s, frame.vehicle.position_enu
        elapsed = max(0., t - self.t0)
        estimate = self.reserve.estimate_s(position)
        if not math.isfinite(estimate):
            raise SafetyAbort('return path from the vehicle estimate is not free')
        tracks = self.tracker.public_tracks(t, c.vertical_sigma_m, confirmed_only=not c.public_tentative_tracks,
                                            with_score=c.public_tentative_tracks)
        reported = {r.track_id for r in self.reports}
        views, snaps = [], []
        for track in tracks:
            if track.track_id in reported or c.max_inspections_per_track == 0:
                continue
            requested = inspection_views(track.position_enu, track.covariance_enu, c.altitude_agl_m, c.inspect_radii_m, c.inspect_angles,
                                         self.obstacle_map, self.briefing.sensor_model, c.mount_pitch_rad, position, c.inspect_speed_m_s,
                                         c.clearance_radius_m, c.inspect_capture_s, c.ground_z_m, c.footprint_max_range_m,
                                         c.centre_window_fraction, c.sigma_scale, reserve=self.reserve, elapsed_s=elapsed,
                                         view_prefix=track.track_id, bounds_margin_m=c.goal_inset_m)
            if self.view_graph is None:
                for view in requested:
                    if view.valid:
                        views.append(view.to_view_option(track.track_id))
                continue
            validity = node_validity_fn(track, self.obstacle_map, self.briefing.sensor_model, c.mount_pitch_rad, position, c.inspect_speed_m_s,
                                        c.clearance_radius_m, c.inspect_capture_s, c.footprint_max_range_m, c.centre_window_fraction,
                                        c.sigma_scale, reserve=self.reserve, elapsed_s=elapsed, bounds_margin_m=c.goal_inset_m)
            offered, records = snap_views(track, [view for view in requested if view.valid], self.view_graph, validity)
            views.extend(offered)
            snaps.extend(records)
        georeference, anchors = None, ()
        if self.georeference is not None:
            # The summary is a function of the filter's covariance at this absolute time; querying it neither
            # advances the process nor consumes a draw, so observing costs nothing and changes nothing.
            georeference = self.georeference.public_summary(t)
            anchors = anchor_statuses(self.anchor_set, position, c.transit_speed_m_s, self.obstacle_map,
                                      c.clearance_radius_m, self.reserve, elapsed, t)
        observation = PublicObservation(self.schema_version, t, elapsed, frame.vehicle, frame.sensors, self.coverage_grid.sector_coverage(),
                                        tracks, tuple(views), max(0., self.briefing.budgets.mission_time_s - elapsed), estimate,
                                        tuple(self.reports), georeference, anchors)
        self.observations += 1
        for record in snaps:
            self._write('public/view_snaps.jsonl', {'observation_index': self.observations, 'sim_time_s': t, 'task_version': task_version_for(c),
                                                    'graph_version': self.view_graph.config.graph_version,
                                                    'graph_sha256': self.view_graph.sha256, 'snap': record.to_dict()})
        self._write('public/observations.jsonl', {'observation_index': self.observations, 'observation': observation.to_dict()})
        return observation

    # -- action preparation (public feasibility only) --
    def _legs_free(self, start, points):
        for point in points:
            check = self.obstacle_map.check_segment(start, point, self.config.clearance_radius_m)
            if not check.free:
                return f'leg_{check.reason}:{check.nearest_box}'
            start = point
        return None

    def _return_points(self, position):
        z = self.briefing.landing_zone.center_enu[2] + self.config.transit_altitude_agl_m
        points = [] if position[2] >= z - .5 else [(position[0], position[1], z)]
        return points + [(*self.briefing.landing_zone.center_enu[:2], z)]

    def _prepare(self, action, observation):
        """(goal, rejection_reason): goal is the SearchGoal, ViewOption, public Track or return points."""
        kind, position = action.action_type, observation.vehicle.position_enu
        if kind == ActionType.SEARCH_SECTOR:
            if action.sector_id not in self.sector_ids:
                return None, 'unknown_sector'
            goal = self.policy.search_goal(action.sector_id)
            if goal is None:
                return None, 'no_pending_goal'
            if goal.sector_id != action.sector_id:
                return goal, 'goal_sector_mismatch'
            if self.view_graph is not None and self.view_graph.find_node(goal.position_enu, goal.yaw_rad) is None:
                return goal, 'goal_not_graph_node'
            return goal, self._legs_free(position, [goal.position_enu])
        if kind == ActionType.INSPECT_VIEW:
            view = next((v for v in observation.views if v.view_id == action.view_id), None)
            if view is None:
                return None, 'unknown_view'
            if action.track_id is not None and action.track_id != view.track_id:
                return view, 'view_track_mismatch'
            return view, self._legs_free(position, [view.position_enu])
        if kind == ActionType.REPORT:
            track = next((t for t in observation.tracks if t.track_id == action.track_id), None)
            if track is None:
                return None, 'unknown_track'
            return track, 'track_already_reported' if any(r.track_id == track.track_id for r in self.reports) else None
        if kind == ActionType.RELOCALIZE:
            if self.anchor_set is None:
                return None, 'relocalize_not_enabled'
            anchor = self.anchor_set.find(action.anchor_id)
            if anchor is None:
                return None, 'unknown_anchor'
            status = next((s for s in observation.anchors if s.anchor_id == anchor.anchor_id), None)
            if status is None:
                return anchor, 'anchor_not_in_observation'
            if not status.reachable:
                return anchor, 'anchor_unreachable'
            return anchor, self._legs_free(position, [anchor.position_enu])
        if kind == ActionType.RETURN:
            points = self._return_points(position)
            return points, self._legs_free(position, points)
        raise ExecutiveError(f'no _prepare branch for action {getattr(kind, "value", kind)!r}: '
                             'an action must never borrow another action\'s preparation path')

    # -- macros --
    def _macro(self, action, goal, observation):
        c = self.config
        kind = action.action_type
        if kind == ActionType.SEARCH_SECTOR:
            self._command(CommandKind.GOTO, goal.position_enu, c.search_speed_m_s, yaw=goal.yaw_rad)
            self._command(CommandKind.HOLD, hold_s=c.capture_settle_s)
            self._capture()
        elif kind == ActionType.INSPECT_VIEW:
            self._command(CommandKind.GOTO, goal.position_enu, c.inspect_speed_m_s, yaw=goal.yaw_rad)
            self._command(CommandKind.HOLD, hold_s=c.capture_settle_s)
            for i in range(c.inspect_frames):
                if i:
                    self._command(CommandKind.HOLD, hold_s=c.inspect_interval_s)
                self._capture()
        elif kind == ActionType.REPORT:
            # The public report carries the LOCAL track coordinate and the submission time. Under
            # flood-sar-village-gnss/1 the transmitted GLOBAL coordinate is frozen here on the evaluator stream
            # and never appears in a public payload: publishing it beside the local one would hand the policy
            # the residual d - dhat, which is exactly the quantity it must not observe.
            if self.georeference is None:
                report = SubmittedReport(action.track_id, action.report_position_enu, observation.sim_time_s)
                self.reports.append(report)
            else:
                # A typed receipt, not a scorable coordinate: it has no `position_enu` attribute at all, and both
                # evaluation.match_reports and reward_gnss refuse it (review 1, ruling R1).
                report = LocalReportReceipt(action.track_id, action.report_position_enu, observation.sim_time_s)
                self.reports.append(report)
                submission = self.georeference.submit_report(len(self.reports) - 1, action.track_id,
                                                             action.report_position_enu, observation.sim_time_s)
                self._write_private('evaluator/global_submissions.jsonl', submission.to_dict())
            self._event('report_submitted', report=report.to_dict())
            self._command(CommandKind.HOLD, hold_s=c.report_hold_s)
            self._capture()
        elif kind == ActionType.RELOCALIZE:
            self._command(CommandKind.GOTO, goal.position_enu, c.transit_speed_m_s, yaw=goal.yaw_rad)
            self._command(CommandKind.HOLD, hold_s=goal.dwell_s)
            # perceive=False: a fix buys no image. The frame only refreshes the clock and the pose; the
            # detector, the tracker and the coverage grid never see it. The measurement is timestamped at the
            # END of the dwell, so the full dwell is paid before anything is received.
            frame = self._capture(perceive=False)
            available = self.anchor_set.available_at(frame.sim_time_s)
            event = self.georeference.attempt_fix(frame.sim_time_s, goal.anchor_id, available)
            self._write_private('evaluator/georeference_measurements.jsonl', event.to_dict())
            self._event('relocalize', anchor_id=goal.anchor_id, available=event.available, accepted=event.accepted)
        elif kind == ActionType.RETURN:
            self._fly_return(goal)
        else:
            raise ExecutiveError(f'no _macro branch for action {getattr(kind, "value", kind)!r}')

    def _fly_return(self, points):
        for point in points:
            self._command(CommandKind.GOTO, point, self.config.transit_speed_m_s)
        self.returned = True
        self._set_phase(Phase.LAND)
        self._command(CommandKind.LAND)
        self.landed = True
        self._capture(perceive=False)

    # -- records and outcome --
    def _record(self, observation, action, next_observation, terminated, reason, forced, valid=True, error=None):
        duration = None if next_observation is None else next_observation.sim_time_s - observation.sim_time_s
        if valid:
            if duration <= TIME_TOLERANCE_S:
                raise ExecutiveError('macro took no simulator time')
            discount = self.config.discount.discount(duration)
        else:
            discount, terminated, reason = None, False, INFRASTRUCTURE
            duration = None if duration is None or duration < 0 else duration
        record = DecisionRecord(len(self.records) + 1, observation, action, next_observation, duration, discount, terminated,
                                False, reason, valid, forced, error)
        self.records.append(record)
        self._write('public/transitions.jsonl', record.to_dict())
        return record

    def _invalid(self, observation, action, forced, exc):
        next_observation = None
        error = f'{exc.__class__.__name__}: {exc}'
        try:
            self._record(observation, action, None, False, INFRASTRUCTURE, forced, valid=False, error=error)
        except Exception as record_exc:  # never mask the original failure
            self._event('record_error', error=f'{record_exc.__class__.__name__}: {record_exc}', original=error)
        return next_observation

    def _finish(self, phase, termination_reason, reason):
        self._set_phase(phase, reason)
        outcome = MissionOutcome(self.phase, reason, termination_reason, self.decisions, self.rejections,
                                 sum(r.valid for r in self.records), sum(not r.valid for r in self.records), tuple(self.reports),
                                 self.returned, self.landed, self.distance_m,
                                 None if self.frame is None or self.t0 is None else max(0., self.frame.sim_time_s - self.t0), self.frames)
        self._event('outcome', outcome=outcome.to_dict())
        return outcome

    # -- episode --
    def run(self):
        try:
            self.backend.begin_episode()  # readiness happens here: evaluator truth is unavailable before it on Gazebo
            self._capture(perceive=False)
            self.t0 = self.frame.sim_time_s
            if self.georeference is not None:
                # d0, dhat0 and P0 belong to the episode's origin, not to the backend clock's zero.
                self.georeference.begin_episode(self.t0)
            self._hook(decision_index=None, action_type='EPISODE_BEGIN')  # first runner truth sample: on the pad, after READY
            self._set_phase(Phase.TAKEOFF)
            position = self.frame.vehicle.position_enu
            goal = (position[0], position[1], self.config.search_z)
            blocked = self._legs_free(position, [goal])
            if blocked:
                return self._finish(Phase.FAILURE, 'safety_abort', f'takeoff {blocked}')
            self._command(CommandKind.TAKEOFF, goal, self.config.takeoff_speed_m_s)
            self._capture()
            self.backend.check_health()
            self._hook(decision_index=None, action_type='TAKEOFF')
            observation = self._observation()
        except KeyboardInterrupt:
            raise
        except (CommandIncomplete, SafetyAbort) as exc:
            return self._finish(Phase.FAILURE, 'safety_abort', f'{exc.__class__.__name__}: {exc}')
        except Exception as exc:
            return self._finish(Phase.FAILURE, INFRASTRUCTURE, f'{exc.__class__.__name__}: {exc}')
        return self._loop(observation)

    def _loop(self, observation):
        c, consecutive = self.config, 0
        while True:
            self._set_phase(Phase.SEARCH)
            forced = 'return_reserve' if self.reserve.must_return(observation.elapsed_s, observation.vehicle.position_enu) else \
                'rejection_cap' if consecutive >= c.max_consecutive_rejections else 'max_decisions' if self.decisions >= c.max_decisions else None
            latency = None
            try:
                if forced:
                    action = Action(ActionType.RETURN)
                else:
                    started = self.clock()
                    action = self.policy.act(observation)
                    latency = self.clock() - started
                    if not isinstance(action, Action):
                        raise ContractError('policy must return a contracts.Action')
                goal, rejection = self._prepare(action, observation)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                return self._finish(Phase.ABORT, INFRASTRUCTURE, f'policy or preparation error: {exc.__class__.__name__}: {exc}')
            entry = {'observation_index': self.observations, 'action': action.to_dict(), 'forced': forced,
                     'decision_wall_s': latency, 'goal': goal.to_dict() if hasattr(goal, 'to_dict') else goal}
            if rejection:
                self.rejections += 1
                consecutive += 1
                self._write('public/actions.jsonl', dict(entry, status='rejected', rejection=rejection))
                if forced:
                    return self._finish(Phase.ABORT, 'safety_abort', f'forced RETURN ({forced}) infeasible: {rejection}')
                self.policy.feedback(action, False, rejection, goal)
                continue
            consecutive = 0
            self.decisions += 1
            self._write('public/actions.jsonl', dict(entry, status='executed', decision_index=self.decisions))
            try:
                self._set_phase(ACTION_PHASE[action.action_type], forced)
                self._audit('before', action)
                self._macro(action, goal, observation)
                self._audit('after', action)
                self.backend.check_health()
                self._hook(decision_index=self.decisions, action_type=action.action_type.value)
                next_observation = self._observation()
            except KeyboardInterrupt:
                raise
            except CommandIncomplete as exc:
                return self._incomplete(observation, action, forced, exc)
            except Exception as exc:
                self._invalid(observation, action, forced, exc)
                return self._finish(Phase.ABORT, INFRASTRUCTURE, f'{exc.__class__.__name__}: {exc}')
            self.policy.feedback(action, True, None, goal)
            deadline = next_observation.elapsed_s >= self.briefing.budgets.mission_time_s - TIME_TOLERANCE_S
            terminal = action.action_type == ActionType.RETURN
            reason = 'mission_deadline' if deadline else 'mission_complete' if terminal else None
            try:
                self._record(observation, action, next_observation, deadline or terminal, reason, forced)
            except Exception as exc:
                return self._finish(Phase.ABORT, INFRASTRUCTURE, f'record error: {exc.__class__.__name__}: {exc}')
            if terminal:
                return self._finish(Phase.COMPLETE, reason, forced or 'policy RETURN')
            if deadline:
                return self._post_deadline_return()
            observation = next_observation

    def _incomplete(self, observation, action, forced, exc):
        try:
            self._capture(perceive=False)
            self._record(observation, action, self._observation(), True, 'safety_abort', forced, error=str(exc))
        except KeyboardInterrupt:
            raise
        except Exception as inner:
            self._invalid(observation, action, forced, inner)
            return self._finish(Phase.ABORT, INFRASTRUCTURE, f'{exc}; then {inner.__class__.__name__}: {inner}')
        return self._finish(Phase.ABORT, 'safety_abort', str(exc))

    def _post_deadline_return(self):
        """Outside the decision process: the deadline decision was terminal; the vehicle still returns and lands."""
        self._set_phase(Phase.RETURN, 'mission_deadline')
        try:
            position = self.frame.vehicle.position_enu
            points = self._return_points(position)
            blocked = self._legs_free(position, points)
            if blocked:
                return self._finish(Phase.ABORT, 'mission_deadline', f'post-deadline return {blocked}')
            self._fly_return(points)
            self.backend.check_health()
            self._hook(decision_index=None, action_type='POST_DEADLINE_RETURN')
        except KeyboardInterrupt:
            raise
        except CommandIncomplete as exc:
            return self._finish(Phase.ABORT, 'mission_deadline', f'post-deadline return: {exc}')
        except Exception as exc:
            return self._finish(Phase.ABORT, INFRASTRUCTURE, f'post-deadline return: {exc.__class__.__name__}: {exc}')
        return self._finish(Phase.COMPLETE, 'mission_deadline', 'post-deadline return and landing')
