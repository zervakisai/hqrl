"""Evaluator-side report matching and episode accounting (PRIVATE).

Private channel
    Inputs are EvaluatorSpec/TargetTruth and EvaluatorTruth samples, plus public data the evaluator may
    also read (submitted reports, sensor frames, the public static obstacle map). Outputs (MatchResult,
    EpisodeAccount, contracts.EpisodeResult) go to evaluator/ files and reward code only; they must never
    reach actor, critic or planner inputs. planning.py does not import this module.

Units, frames, time
    Metres, world ENU, simulator seconds; wall-clock seconds only in wall_time_s.

Matching rule (declared before use)
    A report and a target are compatible when their distance is <= tolerance_m, in 3D ('xyz', default)
    or horizontally ('xy'). The one-to-one matching maximises the number of matched pairs, then
    minimises the summed distance (quantised to 1 micrometre), then prefers earlier reports; an exact
    integer Hungarian algorithm solves it (no scipy). Report times do not change the matching. An
    unmatched report compatible with some target is a duplicate (maximality implies that target is
    already matched); any other unmatched report is false. Unmatched targets are unresolved. With zero
    targets every report is false and recall is None, never 100%.

Failure handling
    Invalid inputs raise ContractError. Unmeasured quantities stay None (EpisodeResult.collisions is Optional:
    unmeasured contacts are carried as None, never 0). EpisodeResult cannot represent in-episode teleports, so
    EpisodeAccount.to_episode_result refuses to convert such accounts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .backend import EvaluatorTruth
from .contracts import (INFRASTRUCTURE, SCHEMA_VERSION, TERMINAL_REASONS, TIME_TOLERANCE_S, ContractError, EpisodeResult,
                        EvaluatorSpec, TargetTruth, Zone, _flag, _ident, _optional_real, _real, _vec)

METRICS = ('xyz', 'xy')
RUN_STATES = ('planned', 'started', 'completed', 'interrupted')
_QUANTUM_M = 1e-6


def _assignment(cost):
    """Exact minimum-cost assignment of every row of an n x m integer matrix (n <= m); column per row."""
    n, m = len(cost), len(cost[0])
    u, v, p, way = [0] * (n + 1), [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0], i0, delta, j1 = True, p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    reduced = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if reduced < minv[j]:
                        minv[j], way[j] = reduced, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    assigned = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            assigned[p[j] - 1] = j - 1
    return assigned


@dataclass(frozen=True)
class ReportMatch:
    report_index: int
    target_id: str
    distance_m: float             # in the matching metric
    error_xyz_m: float
    error_xy_m: float


@dataclass(frozen=True)
class MatchResult:
    tolerance_m: float
    metric: str
    targets_present: int
    report_count: int
    matches: Tuple[ReportMatch, ...]
    duplicate_report_indices: Tuple[int, ...]
    false_report_indices: Tuple[int, ...]
    unresolved_target_ids: Tuple[str, ...]

    @property
    def correct_distinct_reports(self):
        return len(self.matches)

    @property
    def false_reports(self):
        return len(self.false_report_indices)

    @property
    def duplicate_reports(self):
        return len(self.duplicate_report_indices)

    @property
    def unresolved_targets(self):
        return len(self.unresolved_target_ids)

    def recall(self):
        return None if self.targets_present == 0 else self.correct_distinct_reports / self.targets_present

    def mean_error_m(self):
        return None if not self.matches else float(np.mean([m.distance_m for m in self.matches]))


def match_reports(reports, targets, tolerance_m, metric='xyz'):
    """Match submitted reports (SubmittedReport or 3-vectors, in submission order) to evaluator targets."""
    tolerance = _real(tolerance_m, 'tolerance_m', low=0, open_low=True)
    if metric not in METRICS:
        raise ContractError(f'metric: expected one of {METRICS}')
    targets = tuple(t if isinstance(t, TargetTruth) else TargetTruth.from_dict(t) for t in targets)
    if len({t.target_id for t in targets}) != len(targets):
        raise ContractError('targets: duplicate IDs')
    # Fail closed on a public local-frame receipt (review 1, ruling R1). Under flood-sar-village-gnss/1 the
    # scorable coordinate is the frozen global one in georeference.GlobalSubmission; a public receipt carries the
    # LOCAL track coordinate and scoring it would silently measure a task without georeferencing error.
    for i, report in enumerate(reports):
        if hasattr(report, 'local_position_enu'):
            raise ContractError(
                f'reports[{i}]: a local-frame record ({type(report).__name__}) cannot be scored. Pass the frozen '
                'global coordinates, e.g. GeoreferenceService.global_report_positions().')
    points = np.array([_vec(getattr(r, 'position_enu', r), f'reports[{i}]') for i, r in enumerate(reports)], float).reshape(-1, 3)
    truth = np.array([t.position_enu for t in targets], float).reshape(-1, 3)
    diff = points[:, None, :] - truth[None, :, :]
    error_xyz, error_xy = np.linalg.norm(diff, axis=2), np.linalg.norm(diff[..., :2], axis=2)
    distance = error_xyz if metric == 'xyz' else error_xy
    compatible = distance <= tolerance
    n_reports, n_targets = distance.shape
    k, pairs = min(n_reports, n_targets), []
    if k and compatible.any():
        scale = k * n_reports + 1   # summed report indices of any matching stay below one quantum of distance
        quantised = [[int(round(distance[r, t] / _QUANTUM_M)) * scale + r if compatible[r, t] else None
                      for t in range(n_targets)] for r in range(n_reports)]
        big = (max(c for row in quantised for c in row if c is not None) + 1) * (k + 1)
        cost = [[big if c is None else c for c in row] for row in quantised]
        if n_reports <= n_targets:
            pairs = list(enumerate(_assignment(cost)))
        else:
            pairs = [(r, t) for t, r in enumerate(_assignment([list(col) for col in zip(*cost)]))]
        pairs = sorted((r, t) for r, t in pairs if compatible[r, t])
    matched_reports, matched_targets = {r for r, _ in pairs}, {t for _, t in pairs}
    duplicates = tuple(r for r in range(n_reports) if r not in matched_reports and compatible[r].any())
    if any(t not in matched_targets for r in duplicates for t in np.nonzero(compatible[r])[0]):
        raise ContractError('internal: matching is not maximal')
    return MatchResult(tolerance, metric, n_targets, n_reports,
                       tuple(ReportMatch(r, targets[t].target_id, float(distance[r, t]), float(error_xyz[r, t]), float(error_xy[r, t]))
                             for r, t in pairs),
                       duplicates, tuple(r for r in range(n_reports) if r not in matched_reports and not compatible[r].any()),
                       tuple(t.target_id for i, t in enumerate(targets) if i not in matched_targets))


@dataclass(frozen=True)
class EpisodeAccount:
    """Full private accounting; to_episode_result() maps it onto contracts.EpisodeResult."""
    episode_id: str
    run_state: str
    termination_reason: Optional[str]
    matching: MatchResult
    returned: bool
    landed: bool
    collisions: Optional[int]                   # None: not measured by every evaluator sample
    min_clearance_m: Optional[float]            # min of the measured clearances below
    trajectory_min_clearance_m: Optional[float]  # evaluator trajectory vs public static boxes
    backend_min_clearance_m: Optional[float]    # EvaluatorTruth.min_clearance_m, when always measured
    interventions: int
    flight_distance_m: float
    sim_time_s: float
    wall_time_s: float
    sensor_loss_events: int
    invalid_sensor_samples: int
    episode_teleports: int
    truth_samples: int
    decision_latency_p95_s: Optional[float]

    def to_episode_result(self):
        if self.episode_teleports:
            raise ContractError('in-episode teleports recorded; EpisodeResult has no field for this invalidity')
        m = self.matching
        return EpisodeResult(SCHEMA_VERSION, self.episode_id, self.run_state, self.termination_reason == INFRASTRUCTURE,
                             self.termination_reason, m.targets_present, m.correct_distinct_reports, m.false_reports,
                             m.duplicate_reports, m.unresolved_targets, self.returned, self.landed, self.collisions,
                             self.min_clearance_m, self.interventions, self.flight_distance_m, self.sim_time_s,
                             self.wall_time_s, self.sensor_loss_events, self.decision_latency_p95_s, None, None)


class EpisodeAccounting:
    """Accumulates private samples of one episode and builds an EpisodeAccount.

    Declared rules: the trajectory is the sequence of EvaluatorTruth.vehicle_position_enu samples joined by
    straight segments, so flight distance is a sampling-dependent lower bound and trajectory clearance is
    that of the base_link reference point to the public static boxes (airframe radius not subtracted,
    ground excluded, None without boxes). Collisions are None unless every sample measured them, else the
    maximum (a cumulative count); backend clearance likewise None unless always measured, else the
    minimum. returned: the last sample lies horizontally within the landing-zone radius. landed: the
    backend landed status when given, else the last sample is within landed_height_m above the zone z
    (flat ground). A sensor loss event is the start of a run of invalid samples of one stream.
    """

    def __init__(self, episode_id, evaluator, landing_zone, obstacle_map=None, metric='xyz', landed_height_m=.5):
        if not isinstance(evaluator, EvaluatorSpec) or not isinstance(landing_zone, Zone):
            raise ContractError('EvaluatorSpec and Zone required')
        if metric not in METRICS:
            raise ContractError(f'metric: expected one of {METRICS}')
        self.episode_id, self.evaluator, self.landing_zone = _ident(episode_id, 'episode_id'), evaluator, landing_zone
        self.obstacle_map, self.metric = obstacle_map, metric
        self.landed_height_m = _real(landed_height_m, 'landed_height_m', low=0)
        self.truth, self.interventions, self.sensor_loss_events, self.invalid_sensor_samples = [], 0, 0, 0
        self._invalid_streams = set()

    def record_truth(self, truth):
        if not isinstance(truth, EvaluatorTruth):
            raise ContractError('EvaluatorTruth required')
        if truth.evaluator != self.evaluator:
            raise ContractError('evaluator spec changed within the episode')
        if self.truth and truth.sim_time_s < self.truth[-1].sim_time_s - TIME_TOLERANCE_S:
            raise ContractError('evaluator sample time moved backwards')
        self.truth.append(truth)

    def record_frame(self, frame):
        for status in frame.sensors:
            if status.valid:
                self._invalid_streams.discard(status.stream)
                continue
            self.invalid_sensor_samples += 1
            if status.stream not in self._invalid_streams:
                self.sensor_loss_events += 1
                self._invalid_streams.add(status.stream)

    def record_intervention(self):
        self.interventions += 1

    def trajectory(self):
        return np.array([t.vehicle_position_enu for t in self.truth], float).reshape(-1, 3)

    def build(self, termination_reason, reports, sim_time_s, wall_time_s, run_state='completed', landed=None,
              decision_latency_p95_s=None):
        if termination_reason is not None and termination_reason not in TERMINAL_REASONS + (INFRASTRUCTURE,):
            raise ContractError('termination_reason: unknown')
        if run_state not in RUN_STATES:
            raise ContractError(f'run_state: expected one of {RUN_STATES}')
        matching = match_reports(reports, self.evaluator.targets, self.evaluator.match_tolerance_m, self.metric)
        path, samples = self.trajectory(), self.truth
        distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.
        trajectory_clearance = None
        if self.obstacle_map is not None and len(path) and self.obstacle_map.boxes:
            segments = list(zip(path[:-1], path[1:])) or [(path[0], path[0])]
            trajectory_clearance = float(min(self.obstacle_map.segment_clearance(a, b) for a, b in segments))
        collisions = None if not samples or any(t.collisions is None for t in samples) else max(t.collisions for t in samples)
        backend_clearance = None if not samples or any(t.min_clearance_m is None for t in samples) else min(t.min_clearance_m for t in samples)
        measured = [c for c in (trajectory_clearance, backend_clearance) if c is not None]
        centre = np.asarray(self.landing_zone.center_enu, float)
        returned = bool(len(path)) and math.hypot(*(path[-1, :2] - centre[:2])) <= self.landing_zone.radius_m
        on_ground = bool(len(path)) and path[-1, 2] - centre[2] <= self.landed_height_m
        return EpisodeAccount(self.episode_id, run_state, termination_reason, matching, bool(returned),
                              bool(on_ground) if landed is None else _flag(landed, 'landed'), collisions,
                              min(measured) if measured else None, trajectory_clearance, backend_clearance, self.interventions,
                              distance, _real(sim_time_s, 'sim_time_s', low=0), _real(wall_time_s, 'wall_time_s', low=0),
                              self.sensor_loss_events, self.invalid_sensor_samples,
                              max((t.episode_teleports for t in samples), default=0), len(samples),
                              _optional_real(decision_latency_p95_s, 'decision_latency_p95_s', low=0))
