"""Public view graph of task flood-sar-viewgraph/1 (docs/adr/006-fast-decision-environment.md §3).

Task
    flood-sar-viewgraph/1 restricts every goal to a finite public graph of vehicle poses so that a library of real renders
    can cover it. It is not the same problem as flood-sar-mission/1 and is always named explicitly.

Public inputs only (§3.1)
    MissionBriefing, the public StaticObstacleMap, the public MissionConfig (camera mount pitch, altitude, clearance radius,
    goal inset, coverage parameters) and a ViewGraphConfig. Never human positions, segmentation, evaluator objects or
    scenario seeds; tests check the imports and the parameter names. build_view_graph duck-types mission_config (the
    executive imports this module, so importing MissionConfig here would be circular).

Nodes (world ENU metres, yaw radians counter-clockwise from East, camera pitch below the horizontal)
    takeoff_top  'takeoff_top': launch-zone centre XY at config.search_z, yaw 0.0 (declared: the pad heading is not part of
                 the briefing). Optional (ViewGraphConfig.include_takeoff_top).
    search       's{index}': planning.CoverageBaseline waypoints built exactly as CoverageBaselinePolicy builds its plan
                 (same arguments, bounds_margin_m = goal_inset_m, start above the launch-zone centre at search_z), with
                 the waypoint lane yaw. Waypoints with a plan issue at the waypoint itself are excluded and listed.
    inspect      'i{ix}-{iy}-y{k}': lattice over the bounding box of the search sectors intersected with the world bounds
                 shrunk by goal_inset_m, anchored at its (x_min, y_min) corner, spacing inspect_spacing_m (the last
                 row/column is clamped onto the box edge when the extent is within 1e-9 spacings of a multiple), yaw
                 wrap_angle(2*pi*k/inspect_yaw_count), z = search_z. A lattice cell is kept only when
                 StaticObstacleMap.check_segment(p, p, clearance_radius_m) is free; rejected cells are listed as 'i{ix}-{iy}'.
    Every node has pitch_rad = config.mount_pitch_rad and records the clearance radius of the obstacle check it passed.
    Order: takeoff_top, search nodes by waypoint index, inspect nodes by (ix, iy, k). Deterministic.

Version (§3.1)
    ViewGraph.sha256 = sha256 of the canonical JSON (sorted keys, no whitespace, no NaN) of {'config', 'nodes'}. Excluded
    entries are reported but not hashed. Any change of generator, configuration or inputs that changes a node is a new graph.

Snapping a requested inspection view (§3.2)
    nearest_inspect: minimum horizontal distance to an inspect node; ties (within 1e-9 m) by the smallest wrapped yaw
    difference; remaining ties (within 1e-9 rad) by node order. A spatial hash grid of cell inspect_spacing_m makes the
    search exact without scanning every node. snap_views rejects a snap when the horizontal error exceeds max_snap_xy_m
    ('beyond_snap_xy') or the yaw error exceeds max_snap_yaw_rad ('beyond_snap_yaw'); equality is accepted. Otherwise the
    validity checks of planning.inspection_views are re-run at the node pose through validity_fn (node_validity_fn uses
    planning.evaluate_viewpoint with standoff and bearing recomputed from the node pose). A valid node is offered as a
    ViewOption with the node position and yaw and view_id f'{track_id}-{node_id}'; a later request that snaps to a node
    already evaluated for the same track is 'merged_duplicate' when that node was offered, or repeats the node's
    'invalid_at_node:<reason>'. Every requested view passed in gets one SnapRecord (errors and outcome). snap_views does
    not consult the validity of the requested pose: the caller decides which requested views to pass (the executive
    passes the views that are valid at their requested pose, i.e. the flood-sar-mission/1 offer set).

Failure handling: invalid inputs raise ContractError/PlanningError (ValueError); nothing is silently repaired.
Every numeric default is an engineering default (provenance engineering_default_unfrozen).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional, Tuple

import numpy as np

from .contracts import (PROVENANCE, Contract, ContractError, MissionBriefing, ViewOption, _choice, _count, _flag, _ident, _nested,
                        _nested_seq, _optional_ident, _optional_real, _real, _unique, _vec)
from .frames import vector3, wrap_angle
from .leakage import assert_public_payload
from .planning import (DEFAULT_CAPTURE_TIME_S, DEFAULT_CLEARANCE_RADIUS_M, DEFAULT_MAX_RANGE_M, CoverageBaseline, PlanningError,
                       StaticObstacleMap, evaluate_viewpoint)

TASK_VERSION = 'flood-sar-viewgraph/1'
VILLAGE_TASK_VERSION = 'flood-sar-village/1'  # TASK_VERSION + public tentative tracks with max_score (ADR 010)
TASK_TENTATIVE_TRACKS = {TASK_VERSION: False, VILLAGE_TASK_VERSION: True}  # version -> required MissionConfig.public_tentative_tracks


def task_version_for(config):
    """The graph-bound task version a MissionConfig runs (flood-sar-village/1 when tentative tracks are public)."""
    return VILLAGE_TASK_VERSION if config.public_tentative_tracks else TASK_VERSION
GRAPH_VERSION = 'viewgraph/1'
NODE_KINDS = ('takeoff_top', 'search', 'inspect')
TAKEOFF_TOP_ID = 'takeoff_top'
TAKEOFF_TOP_YAW_RAD = 0.
SNAP_OUTCOMES = ('offered', 'merged_duplicate', 'beyond_snap_xy', 'beyond_snap_yaw')
INVALID_AT_NODE = 'invalid_at_node:'
_TIE_XY_M = 1e-9
_TIE_YAW_RAD = 1e-9


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ContractError(f'{name}: expected non-empty string')
    return value


@dataclass(frozen=True)
class ViewGraphConfig(Contract):
    graph_version: str
    inspect_spacing_m: float
    inspect_yaw_count: int
    max_snap_xy_m: float
    max_snap_yaw_rad: float
    include_takeoff_top: bool
    provenance: str = 'engineering_default_unfrozen'

    def __post_init__(self):
        _count(self.inspect_yaw_count, 'inspect_yaw_count', 1)
        self._set(graph_version=_choice(self.graph_version, 'graph_version', (GRAPH_VERSION,)),
                  inspect_spacing_m=_real(self.inspect_spacing_m, 'inspect_spacing_m', low=0, open_low=True),
                  max_snap_xy_m=_real(self.max_snap_xy_m, 'max_snap_xy_m', low=0),
                  max_snap_yaw_rad=_real(self.max_snap_yaw_rad, 'max_snap_yaw_rad', low=0, high=math.pi),
                  include_takeoff_top=_flag(self.include_takeoff_top, 'include_takeoff_top'),
                  provenance=_choice(self.provenance, 'provenance', PROVENANCE))


@dataclass(frozen=True)
class ViewNode(Contract):
    """Vehicle pose of the graph. clearance_radius_m: StaticObstacleMap.check_segment(p, p, clearance_radius_m) was free."""
    node_id: str
    kind: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float
    clearance_radius_m: float

    def __post_init__(self):
        self._set(node_id=_ident(self.node_id, 'node_id'), kind=_choice(self.kind, 'kind', NODE_KINDS),
                  position_enu=_vec(self.position_enu, 'position_enu'),
                  yaw_rad=_real(self.yaw_rad, 'yaw_rad', low=-math.pi, high=math.pi),
                  pitch_rad=_real(self.pitch_rad, 'pitch_rad', low=0, high=math.pi / 2, open_low=True),
                  clearance_radius_m=_real(self.clearance_radius_m, 'clearance_radius_m', low=0))


@dataclass(frozen=True)
class ExcludedNode(Contract):
    """A candidate that is not a node: 's{index}' ('plan_issue:<reason>'), 'takeoff_top' or a lattice cell 'i{ix}-{iy}'
    (all yaws; the StaticObstacleMap.check_segment reason)."""
    node_id: str
    kind: str
    position_enu: Tuple[float, float, float]
    reason: str
    nearest_box: Optional[str]

    def __post_init__(self):
        self._set(node_id=_ident(self.node_id, 'node_id'), kind=_choice(self.kind, 'kind', NODE_KINDS),
                  position_enu=_vec(self.position_enu, 'position_enu'), reason=_text(self.reason, 'reason'),
                  nearest_box=_optional_ident(self.nearest_box, 'nearest_box'))


def graph_sha256(config, nodes):
    return hashlib.sha256(_canonical({'config': config.to_dict(), 'nodes': [n.to_dict() for n in nodes]}).encode()).hexdigest()


@dataclass(frozen=True)
class ViewGraph(Contract):
    """Immutable public view graph (module docstring). sha256 is computed; a given value must match it."""
    config: ViewGraphConfig
    nodes: Tuple[ViewNode, ...]
    excluded: Tuple[ExcludedNode, ...] = ()
    sha256: Optional[str] = None

    def __post_init__(self):
        config, nodes = _nested(ViewGraphConfig, self.config, 'config'), _nested_seq(ViewNode, self.nodes, 'nodes')
        excluded = _nested_seq(ExcludedNode, self.excluded, 'excluded')
        _unique([n.node_id for n in nodes], 'nodes')
        digest = graph_sha256(config, nodes)
        if self.sha256 is not None and self.sha256 != digest:
            raise ContractError('ViewGraph: sha256 does not match config and nodes')
        self._set(config=config, nodes=nodes, excluded=excluded, sha256=digest)
        assert_public_payload(self.to_dict())
        xyz = np.array([n.position_enu for n in nodes], float).reshape(-1, 3)
        inspect = [i for i, n in enumerate(nodes) if n.kind == 'inspect']
        buckets, origin, low, high = {}, None, None, None
        if inspect:
            origin = xyz[inspect, :2].min(axis=0)
            keys = np.floor((xyz[inspect, :2] - origin) / config.inspect_spacing_m + .5).astype(int)
            for i, key in zip(inspect, map(tuple, keys.tolist())):
                buckets.setdefault(key, []).append(i)
            low, high = tuple(keys.min(axis=0).tolist()), tuple(keys.max(axis=0).tolist())
        self._set(_index={n.node_id: i for i, n in enumerate(nodes)}, _xyz=xyz, _yaw=np.array([n.yaw_rad for n in nodes], float),
                  _kinds=np.array([n.kind for n in nodes], dtype=object), _buckets=buckets, _origin=origin, _low=low, _high=high)

    def get(self, node_id):
        return self.nodes[self._index[node_id]]

    def counts(self):
        return {kind: sum(n.kind == kind for n in self.nodes) for kind in NODE_KINDS}

    def find_node(self, position_enu, yaw_rad, xy_tol_m=1e-3, yaw_tol_rad=1e-3, z_tol_m=1e-3, kind=None):
        """First node (graph order) within the horizontal, vertical and wrapped-yaw tolerances, or None."""
        p, yaw = vector3(position_enu, 'position_enu'), _real(yaw_rad, 'yaw_rad')
        if not self.nodes:
            return None
        dyaw = np.abs((self._yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        mask = (np.hypot(self._xyz[:, 0] - p[0], self._xyz[:, 1] - p[1]) <= xy_tol_m) & (np.abs(self._xyz[:, 2] - p[2]) <= z_tol_m) \
            & (dyaw <= yaw_tol_rad)
        if kind is not None:
            mask &= self._kinds == _choice(kind, 'kind', NODE_KINDS)
        hits = np.flatnonzero(mask)
        return self.nodes[int(hits[0])] if len(hits) else None

    def _ring(self, cx, cy, k):
        (lx, ly), (hx, hy) = self._low, self._high
        if k == 0:
            if lx <= cx <= hx and ly <= cy <= hy:
                yield cx, cy
            return
        for i in range(max(cx - k, lx), min(cx + k, hx) + 1):
            for j in (cy - k, cy + k):
                if ly <= j <= hy:
                    yield i, j
        for j in range(max(cy - k + 1, ly), min(cy + k - 1, hy) + 1):
            for i in (cx - k, cx + k):
                if lx <= i <= hx:
                    yield i, j

    def nearest_inspect(self, position_enu, yaw_rad):
        """(node, horizontal error m, wrapped yaw error rad) of the snapping rule (module docstring); None without inspect nodes.

        Bucket key = floor((xy - origin) / spacing + 0.5), so a node lies within half a cell of its key and the query within
        half a cell of its own key: every node in ring m (Chebyshev key distance) is at least (m - 1) * spacing away."""
        p, yaw = vector3(position_enu, 'position_enu'), _real(yaw_rad, 'yaw_rad')
        if not self._buckets:
            return None
        s = self.config.inspect_spacing_m
        cx, cy = (int(math.floor((p[i] - self._origin[i]) / s + .5)) for i in (0, 1))
        (lx, ly), (hx, hy) = self._low, self._high
        k, last = max(0, lx - cx, cx - hx, ly - cy, cy - hy), max(cx - lx, hx - cx, cy - ly, hy - cy)
        found = []
        while True:
            for key in self._ring(cx, cy, k):
                for i in self._buckets.get(key, ()):
                    found.append((math.hypot(self._xyz[i, 0] - p[0], self._xyz[i, 1] - p[1]), abs(wrap_angle(self._yaw[i] - yaw)), i))
            if (found and min(f[0] for f in found) + _TIE_XY_M < s * k) or k >= last:
                break
            k += 1
        best_xy = min(f[0] for f in found)
        found = [f for f in found if f[0] <= best_xy + _TIE_XY_M]
        best_yaw = min(f[1] for f in found)
        xy, dyaw, i = min((f for f in found if f[1] <= best_yaw + _TIE_YAW_RAD), key=lambda f: f[2])
        return self.nodes[i], float(xy), float(dyaw)


def build_view_graph(briefing, obstacle_map, mission_config, graph_config):
    """ViewGraph from public inputs only (module docstring). mission_config: executive.MissionConfig."""
    briefing, graph_config = _nested(MissionBriefing, briefing, 'briefing'), _nested(ViewGraphConfig, graph_config, 'graph_config')
    if not isinstance(obstacle_map, StaticObstacleMap):
        raise PlanningError('obstacle_map: expected the public StaticObstacleMap')
    c = mission_config
    pitch, z, radius, inset = c.mount_pitch_rad, c.search_z, c.clearance_radius_m, c.goal_inset_m
    launch_xy = briefing.launch_zone.center_enu[:2]
    nodes, excluded = [], []
    if graph_config.include_takeoff_top:
        top = (float(launch_xy[0]), float(launch_xy[1]), float(z))
        check = obstacle_map.check_segment(top, top, radius)
        if check.free:
            nodes.append(ViewNode(TAKEOFF_TOP_ID, 'takeoff_top', top, TAKEOFF_TOP_YAW_RAD, pitch, radius))
        else:
            excluded.append(ExcludedNode(TAKEOFF_TOP_ID, 'takeoff_top', top, check.reason, check.nearest_box))
    start = (*launch_xy, c.search_z)  # as CoverageBaselinePolicy
    plan = CoverageBaseline(briefing, obstacle_map, c.mount_pitch_rad, c.altitude_agl_m, c.overlap_fraction, c.clearance_radius_m,
                            c.footprint_max_range_m, c.ground_z_m, bounds_margin_m=c.goal_inset_m).plan(start)
    issues = {i.index: i for i in plan.issues if i.kind == 'waypoint'}
    for w in plan.waypoints:
        issue = issues.get(w.index)
        if issue is None:
            nodes.append(ViewNode(f's{w.index}', 'search', w.position_enu, w.yaw_rad, pitch, radius))
        else:
            excluded.append(ExcludedNode(f's{w.index}', 'search', w.position_enu, f'plan_issue:{issue.reason}', issue.nearest_box))
    b, s, n = briefing.world_bounds, graph_config.inspect_spacing_m, graph_config.inspect_yaw_count
    corners = np.concatenate([np.asarray(sector.polygon_enu, float) for sector in briefing.search_sectors])
    x0, x1 = max(b.x_min + inset, float(corners[:, 0].min())), min(b.x_max - inset, float(corners[:, 0].max()))
    y0, y1 = max(b.y_min + inset, float(corners[:, 1].min())), min(b.y_max - inset, float(corners[:, 1].max()))
    yaws = [wrap_angle(2 * math.pi * k / n) for k in range(n)]
    if x0 <= x1 and y0 <= y1:
        nx, ny = int(math.floor((x1 - x0) / s + 1e-9)) + 1, int(math.floor((y1 - y0) / s + 1e-9)) + 1
        for ix in range(nx):
            for iy in range(ny):
                p = (float(min(x0 + ix * s, x1)), float(min(y0 + iy * s, y1)), float(z))
                check = obstacle_map.check_segment(p, p, radius)
                if not check.free:
                    excluded.append(ExcludedNode(f'i{ix}-{iy}', 'inspect', p, check.reason, check.nearest_box))
                    continue
                nodes.extend(ViewNode(f'i{ix}-{iy}-y{k}', 'inspect', p, yaw, pitch, radius) for k, yaw in enumerate(yaws))
    return ViewGraph(graph_config, tuple(nodes), tuple(excluded))


# --- Snapping ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SnapRecord(Contract):
    """One requested view and what the graph made of it. view_id: the offered option ('offered') or the option it was merged
    into ('merged_duplicate'); None otherwise. node_id and the errors are None only when the graph has no inspect node."""
    track_id: str
    requested_view_id: str
    requested_position_enu: Tuple[float, float, float]
    requested_yaw_rad: float
    node_id: Optional[str]
    xy_error_m: Optional[float]
    yaw_error_rad: Optional[float]
    outcome: str
    view_id: Optional[str]

    def __post_init__(self):
        outcome = self.outcome
        if not isinstance(outcome, str) or not (outcome in SNAP_OUTCOMES or (outcome.startswith(INVALID_AT_NODE) and len(outcome) > len(INVALID_AT_NODE))):
            raise ContractError(f'outcome: expected one of {SNAP_OUTCOMES} or {INVALID_AT_NODE}<reason>')
        if (self.view_id is not None) != (outcome in ('offered', 'merged_duplicate')):
            raise ContractError('view_id only for offered or merged views')
        self._set(track_id=_ident(self.track_id, 'track_id'), requested_view_id=_ident(self.requested_view_id, 'requested_view_id'),
                  requested_position_enu=_vec(self.requested_position_enu, 'requested_position_enu'),
                  requested_yaw_rad=_real(self.requested_yaw_rad, 'requested_yaw_rad'), node_id=_optional_ident(self.node_id, 'node_id'),
                  xy_error_m=_optional_real(self.xy_error_m, 'xy_error_m', low=0),
                  yaw_error_rad=_optional_real(self.yaw_error_rad, 'yaw_error_rad', low=0, high=math.pi),
                  view_id=_optional_ident(self.view_id, 'view_id'))


def node_validity_fn(track, obstacle_map, sensor_model, mount_pitch_rad, current_position_enu, speed_m_s,
                     clearance_radius_m=DEFAULT_CLEARANCE_RADIUS_M, capture_time_s=DEFAULT_CAPTURE_TIME_S, max_range_m=DEFAULT_MAX_RANGE_M,
                     centre_window_fraction=.5, sigma_scale=2., climb_over_altitude_m=None, reserve=None, elapsed_s=None,
                     bounds_margin_m=None):
    """validity_fn(node, view_id) -> planning.InspectionView at the node pose, with the inspection_views parameters of the
    requested views and standoff/bearing recomputed from the node pose. The node camera pitch must equal mount_pitch_rad."""
    def check(node, view_id):
        if abs(node.pitch_rad - mount_pitch_rad) > 1e-9:
            raise PlanningError(f'{node.node_id}: node camera pitch differs from mount_pitch_rad')
        dx, dy = node.position_enu[0] - track.position_enu[0], node.position_enu[1] - track.position_enu[1]
        return evaluate_viewpoint(track.position_enu, track.covariance_enu, view_id, node.position_enu, node.yaw_rad, math.hypot(dx, dy),
                                  math.atan2(dy, dx), obstacle_map, sensor_model, mount_pitch_rad, current_position_enu, speed_m_s,
                                  clearance_radius_m, capture_time_s, max_range_m, centre_window_fraction, sigma_scale,
                                  climb_over_altitude_m, reserve, elapsed_s, bounds_margin_m)
    return check


def snap_views(track, requested_views, graph, validity_fn):
    """(offered ViewOptions, SnapRecords) for one public track (module docstring, §3.2)."""
    cfg, offered, records, evaluated = graph.config, [], [], {}
    for view in requested_views:
        base = dict(track_id=track.track_id, requested_view_id=view.view_id, requested_position_enu=view.position_enu,
                    requested_yaw_rad=view.yaw_rad)
        found = graph.nearest_inspect(view.position_enu, view.yaw_rad)
        if found is None:
            records.append(SnapRecord(**base, node_id=None, xy_error_m=None, yaw_error_rad=None, outcome='beyond_snap_xy', view_id=None))
            continue
        node, xy_error, yaw_error = found
        view_id = None
        if xy_error > cfg.max_snap_xy_m:
            outcome = 'beyond_snap_xy'
        elif yaw_error > cfg.max_snap_yaw_rad:
            outcome = 'beyond_snap_yaw'
        elif node.node_id in evaluated:
            first, view_id = evaluated[node.node_id]
            outcome = 'merged_duplicate' if first == 'offered' else first
        else:
            view_id = f'{track.track_id}-{node.node_id}'
            checked = validity_fn(node, view_id)
            if checked.view_id != view_id or tuple(checked.position_enu) != tuple(node.position_enu) or checked.yaw_rad != node.yaw_rad:
                raise PlanningError(f'{view_id}: validity_fn must evaluate the node pose under the given view_id')
            if checked.valid:
                offered.append(ViewOption(view_id, node.position_enu, node.yaw_rad, checked.path_length_m, checked.estimated_duration_s,
                                          track.track_id))
                outcome = 'offered'
            else:
                outcome, view_id = f'{INVALID_AT_NODE}{checked.reason}', None
            evaluated[node.node_id] = (outcome, view_id)
        records.append(SnapRecord(**base, node_id=node.node_id, xy_error_m=xy_error, yaw_error_rad=yaw_error, outcome=outcome, view_id=view_id))
    return tuple(offered), tuple(records)
