"""Public planning geometry and heuristics for the Flood-SAR mission (M5).

Public inputs only
    MissionBriefing (bounds, search sectors with prior weights, zones, sensor model, budgets), the
    public static obstacle map (axis-aligned boxes of buildings, bridge, trees; never humans), the
    vehicle estimate, public tracks (position, covariance) and the public coverage raster. Nothing here
    accepts or imports evaluator truth, target IDs, segmentation or report correctness; tests check the
    imports and the parameter names.

Units, frames, time
    Metres, seconds, radians. World ENU (x East, y North, z Up); yaw counter-clockwise from East.
    Times are simulator seconds (contracts.py). Durations are kinematic estimates, not measurements.

Camera model (sensor calibration_status is specification_only)
    Pinhole, square pixels, principal point at the image centre, no distortion; vertical fov from the
    aspect ratio. The camera is fixed to the body, looks along the vehicle yaw and is pitched
    mount_pitch_rad below the horizontal (pi/2 is nadir). Vehicle roll/pitch and the mount offset are
    ignored (hover approximation). The ground is a flat plane at ground_z_m.

Controller model
    Straight-line goto segments at a constant speed <= Budgets.max_speed_m_s, vertical climb/descent at
    declared speeds. Acceleration, yaw turns and settling are covered only by capture_time_s and the
    declared margins. Yaws produced here are sent as FlightCommand.yaw_enu_rad (TAKEOFF/GOTO; heading convergence is part
    of GOTO completion when controller.reach_yaw_rad is set, validated live in px4_integration_v5/v6 G15).

Failure handling
    Invalid arguments raise ValueError (ContractError or PlanningError). Obstacle, bounds and reserve
    violations are returned as explicit issues or reasons, never silently dropped. Space outside the
    declared bounds is not free. InformationCostPlanner scores are heuristics, not calibrated
    probabilities. Every numeric default is an engineering default, unfrozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .contracts import SectorCoverage, ViewOption, _ident, _real, _vec
from .frames import FLU_FROM_OPTICAL, quat_to_matrix, vector3, wrap_angle
from .leakage import assert_public_payload

DEFAULT_MAX_RANGE_M = 60.          # horizontal clip of the ground footprint; not a detection range
DEFAULT_CAPTURE_TIME_S = 2.        # settle + capture at a goal
DEFAULT_CLEARANCE_RADIUS_M = 3.    # obstacle inflation radius for the vehicle reference point
DEFAULT_CLIMB_SPEED_M_S = 1.
DEFAULT_DESCENT_SPEED_M_S = .7
DEFAULT_TIME_MARGIN_FACTOR = 1.2
_EPS = 1e-9


class PlanningError(ValueError):
    pass


def _positive(value, name):
    return _real(value, name, low=0, open_low=True)


def _pitch(value):
    return _real(value, 'mount_pitch_rad', low=0, high=math.pi / 2, open_low=True)


# --- Camera geometry -------------------------------------------------------------------

def camera_rotation(yaw_rad, mount_pitch_rad):
    """World(ENU)<-optical rotation for a level body at yaw_rad, camera pitched below the horizontal."""
    c, s = math.cos(_real(yaw_rad, 'yaw_rad')), math.sin(yaw_rad)
    cp, sp = math.cos(_pitch(mount_pitch_rad)), math.sin(mount_pitch_rad)
    yaw = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    pitch = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])  # about body y (left): forward tilts down
    return yaw @ pitch @ FLU_FROM_OPTICAL


def focal_length_px(sensor_model):
    return sensor_model.rgb_width / 2 / math.tan(sensor_model.rgb_hfov_rad / 2)


def vertical_fov_rad(sensor_model):
    return 2 * math.atan(math.tan(sensor_model.rgb_hfov_rad / 2) * sensor_model.rgb_height / sensor_model.rgb_width)


def project_to_image(point_enu, camera_position_enu, yaw_rad, mount_pitch_rad, sensor_model):
    """(u, v, depth): pixel coordinates (origin top-left) and optical depth; u, v are nan when depth <= 0."""
    p = camera_rotation(yaw_rad, mount_pitch_rad).T @ (vector3(point_enu, 'point_enu') - vector3(camera_position_enu, 'camera_position_enu'))
    if p[2] <= _EPS:
        return math.nan, math.nan, float(p[2])
    f = focal_length_px(sensor_model)
    return float(sensor_model.rgb_width / 2 + f * p[0] / p[2]), float(sensor_model.rgb_height / 2 + f * p[1] / p[2]), float(p[2])


def camera_ground_footprint(position_enu, yaw_rad, altitude_agl, sensor_model, mount_pitch_rad,
                            max_range_m=DEFAULT_MAX_RANGE_M, return_clipped=False):
    """Ground polygon of the full RGB image as a (4, 2) array of ENU xy corners.

    Corners follow the image corners top-left, top-right, bottom-right, bottom-left, which is
    clockwise seen from above. The ground plane is z = position_enu[2] - altitude_agl. A corner ray
    that meets the ground at or above the horizon, or beyond max_range_m horizontal distance, is
    replaced by the point at max_range_m along the ray azimuth (flagged in the optional clipped
    tuple). The clipped polygon is a declared approximation of frustum-intersect-range-disc.
    """
    position = vector3(position_enu, 'position_enu')
    height, max_range = _positive(altitude_agl, 'altitude_agl'), _positive(max_range_m, 'max_range_m')
    rotation = camera_rotation(yaw_rad, mount_pitch_rad)
    a = math.tan(sensor_model.rgb_hfov_rad / 2)
    b = a * sensor_model.rgb_height / sensor_model.rgb_width
    corners, clipped = [], []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        ray = rotation @ np.array([sx * a, sy * b, 1.])
        horizontal = math.hypot(ray[0], ray[1])
        if horizontal < _EPS:
            raise PlanningError('degenerate corner ray')
        reach = height / -ray[2] * horizontal if ray[2] < -_EPS else math.inf
        scale = min(reach, max_range) / horizontal
        corners.append(position[:2] + scale * ray[:2])
        clipped.append(reach > max_range)
    polygon = np.asarray(corners)
    return (polygon, tuple(clipped)) if return_clipped else polygon


# --- Polygons ----------------------------------------------------------------------------

def polygon_area(polygon_xy):
    """Signed shoelace area: positive for counter-clockwise vertices."""
    p = np.asarray(polygon_xy, float)
    return .5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def is_convex(polygon_xy):
    p = np.asarray(polygon_xy, float)
    e = np.roll(p, -1, axis=0) - p
    cross = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
    scale = max(float(np.abs(e).max()) ** 2, 1.)
    return bool(np.all(cross >= -1e-9 * scale) or np.all(cross <= 1e-9 * scale))


def points_in_polygon(points_xy, polygon_xy):
    """Even-odd rule for simple polygons; points exactly on an edge may fall on either side."""
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    poly = np.asarray(polygon_xy, float)
    x, y = pts[:, :1], pts[:, 1:]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return np.count_nonzero(crosses & (x < xi), axis=1) % 2 == 1


def _chord(poly_al, level):
    """(min, max) along-coordinate where the line lateral == level meets a polygon given as (along, lateral)."""
    a0, l0 = poly_al[:, 0], poly_al[:, 1]
    a1, l1 = np.roll(a0, -1), np.roll(l0, -1)
    values = list(a0[np.abs(l0 - level) <= 1e-9])
    crossing = (l0 - level) * (l1 - level) < 0
    t = (level - l0[crossing]) / (l1[crossing] - l0[crossing])
    values += list(a0[crossing] + t * (a1[crossing] - a0[crossing]))
    return (float(min(values)), float(max(values))) if values else None


# --- Public static obstacle map ----------------------------------------------------------

@dataclass(frozen=True)
class ObstacleBox:
    """Axis-aligned public static obstacle (building, bridge part, tree block) in world ENU."""
    name: str
    min_enu: Tuple[float, float, float]
    max_enu: Tuple[float, float, float]

    def __post_init__(self):
        low, high = _vec(self.min_enu, 'min_enu'), _vec(self.max_enu, 'max_enu')
        if any(lo >= hi for lo, hi in zip(low, high)):
            raise PlanningError(f'{self.name}: empty box')
        object.__setattr__(self, 'name', _ident(self.name, 'name'))
        object.__setattr__(self, 'min_enu', low)
        object.__setattr__(self, 'max_enu', high)


@dataclass(frozen=True)
class SegmentCheck:
    free: bool
    clearance_m: float            # distance to the nearest box (inf without boxes, 0 when touching)
    nearest_box: Optional[str]
    reason: Optional[str]         # None, 'outside_bounds' or 'obstacle_clearance'


def _segment_box_sq_distance(a, b, lo, hi):
    """Exact squared distance from segment a-b to each box (N,): the piecewise-quadratic minimum."""
    d, n = b - a, len(lo)
    with np.errstate(divide='ignore', invalid='ignore'):
        breaks = np.concatenate([(lo - a) / d, (hi - a) / d], axis=1)
    breaks = np.where(np.isfinite(breaks), np.clip(breaks, 0., 1.), 0.)
    t = np.sort(np.concatenate([breaks, np.zeros((n, 1)), np.ones((n, 1))], axis=1), axis=1)
    mid = (t[:, :-1] + t[:, 1:]) / 2
    p = a + mid[..., None] * d
    below, above = p < lo[:, None], p > hi[:, None]
    bound = np.where(below, lo[:, None], np.where(above, hi[:, None], 0.))
    active = below | above
    num = np.sum(np.where(active, d * (bound - a), 0.), axis=2)
    den = np.sum(np.where(active, d * d, 0.), axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        stationary = np.where(den > 0, num / den, mid)
    candidates = np.concatenate([t, np.clip(stationary, t[:, :-1], t[:, 1:])], axis=1)
    q = a + candidates[..., None] * d
    gap = np.maximum(np.maximum(lo[:, None] - q, 0.), q - hi[:, None])
    return np.min(np.sum(gap * gap, axis=2), axis=1)


class StaticObstacleMap:
    """Public static boxes inside declared bounds; the same map for every policy arm.

    A segment is free with inflation radius r when both endpoints lie in the bounds shrunk by r
    horizontally, z_min_m <= z <= z_max_m, and the exact segment-to-box distance is strictly greater
    than r for every box. Outside the bounds is unknown and never free. The ground is not a box:
    altitude is set by the planners and vertical takeoff/landing belong to the controller.
    """

    def __init__(self, bounds, boxes=(), z_min_m=0., z_max_m=120.):
        self.bounds = bounds
        self.boxes = tuple(b if isinstance(b, ObstacleBox) else ObstacleBox(**b) for b in boxes)
        if len({b.name for b in self.boxes}) != len(self.boxes):
            raise PlanningError('duplicate obstacle names')
        self.z_min_m, self.z_max_m = _real(z_min_m, 'z_min_m'), _real(z_max_m, 'z_max_m')
        if self.z_min_m >= self.z_max_m:
            raise PlanningError('empty altitude band')
        self._lo = np.array([b.min_enu for b in self.boxes], float).reshape(-1, 3)
        self._hi = np.array([b.max_enu for b in self.boxes], float).reshape(-1, 3)
        assert_public_payload(self.to_dict())

    @classmethod
    def from_scene_config(cls, scene, bounds, z_max_m=120.):
        """Collision boxes of a minimal_scene config (centre/size); visual-only surfaces are skipped."""
        entries = list(scene['static_boxes']) + [e for e in scene.get('water_surfaces', ()) if e.get('collision')]
        boxes = [ObstacleBox(e['name'], tuple(c - s / 2 for c, s in zip(e['center_enu'], e['size_m'])),
                             tuple(c + s / 2 for c, s in zip(e['center_enu'], e['size_m']))) for e in entries]
        return cls(bounds, boxes, z_max_m=z_max_m)

    def to_dict(self):
        return {'bounds': self.bounds.to_dict(), 'z_min_m': self.z_min_m, 'z_max_m': self.z_max_m,
                'boxes': [{'name': b.name, 'min_enu': list(b.min_enu), 'max_enu': list(b.max_enu)} for b in self.boxes]}

    def inside_bounds(self, point_enu, radius_m=0.):
        x, y, z = vector3(point_enu, 'point_enu')
        b, r = self.bounds, radius_m
        return b.x_min + r <= x <= b.x_max - r and b.y_min + r <= y <= b.y_max - r and self.z_min_m <= z <= self.z_max_m

    def _candidate_boxes(self, a, b):
        """Indices (ascending) of the boxes that can hold the minimum distance to segment a-b: a box whose distance to the
        segment's bounding box (a lower bound) exceeds the smallest point-a distance (an upper bound on the minimum) cannot.
        The small relative slack only admits more boxes, so the exact minimum and its first index are unchanged."""
        seg_lo, seg_hi = np.minimum(a, b), np.maximum(a, b)
        gap = np.maximum(np.maximum(self._lo - seg_hi, 0.), seg_lo - self._hi)
        lower = np.einsum('ij,ij->i', gap, gap)
        point_gap = np.maximum(np.maximum(self._lo - a, 0.), a - self._hi)
        upper = float(np.min(np.einsum('ij,ij->i', point_gap, point_gap)))
        return np.flatnonzero(lower <= upper * (1. + 1e-9) + 1e-12)

    def nearest(self, a, b):
        """(clearance_m, box name) for segment a-b; (inf, None) without boxes. Exact distances are computed only for the
        candidate boxes (_candidate_boxes); the result equals the all-box minimum with the same first-index tie rule."""
        a, b = vector3(a, 'a'), vector3(b, 'b')
        if not self.boxes:
            return math.inf, None
        index = self._candidate_boxes(a, b)
        sq = _segment_box_sq_distance(a, b, self._lo[index], self._hi[index])
        i = int(np.argmin(sq))
        return math.sqrt(max(float(sq[i]), 0.)), self.boxes[int(index[i])].name

    def segment_clearance(self, a, b):
        return self.nearest(a, b)[0]

    def check_segment(self, a, b, radius_m):
        radius = _real(radius_m, 'radius_m', low=0)
        clearance, name = self.nearest(a, b)
        if not (self.inside_bounds(a, radius) and self.inside_bounds(b, radius)):
            return SegmentCheck(False, clearance, name, 'outside_bounds')
        if clearance <= radius:
            return SegmentCheck(False, clearance, name, 'obstacle_clearance')
        return SegmentCheck(True, clearance, name, None)

    def segment_is_free(self, a, b, radius_m):
        return self.check_segment(a, b, radius_m).free

    def point_is_free(self, point_enu, radius_m):
        return self.check_segment(point_enu, point_enu, radius_m).free


@dataclass(frozen=True)
class Leg:
    """Path between two points: direct, or a declared climb-over detour (start, up, across, down, goal)."""
    points: Tuple[Tuple[float, float, float], ...]
    length_m: float
    free: bool
    detour: bool
    clearance_m: float
    nearest_box: Optional[str]
    reason: Optional[str]


def plan_leg(obstacle_map, start_enu, goal_enu, radius_m, climb_over_altitude_m=None):
    """Direct segment if free; otherwise the climb-over detour only when climb_over_altitude_m is declared."""
    start, goal = vector3(start_enu, 'start_enu'), vector3(goal_enu, 'goal_enu')
    direct = obstacle_map.check_segment(start, goal, radius_m)
    length = float(np.linalg.norm(goal - start))
    if direct.free or direct.reason == 'outside_bounds' or climb_over_altitude_m is None:
        return Leg((tuple(start), tuple(goal)), length, direct.free, False, direct.clearance_m, direct.nearest_box, direct.reason)
    z = max(_real(climb_over_altitude_m, 'climb_over_altitude_m'), start[2], goal[2])
    points = [start, np.array([start[0], start[1], z]), np.array([goal[0], goal[1], z]), goal]
    points = [p for i, p in enumerate(points) if i == 0 or not np.allclose(p, points[i - 1])]
    checks = [obstacle_map.check_segment(p, q, radius_m) for p, q in zip(points[:-1], points[1:])]
    free, worst = all(c.free for c in checks), min(checks, key=lambda c: c.clearance_m)
    return Leg(tuple(tuple(p) for p in points), float(sum(np.linalg.norm(q - p) for p, q in zip(points[:-1], points[1:]))),
               free, True, worst.clearance_m, worst.nearest_box, None if free else 'blocked_after_climb_over')


# --- Return reserve ----------------------------------------------------------------------

def time_to_return(position_enu, landing_zone, speed_m_s, climb_speed_m_s=DEFAULT_CLIMB_SPEED_M_S,
                   descent_speed_m_s=DEFAULT_DESCENT_SPEED_M_S, transit_altitude_m=None):
    """Kinematic seconds to fly above the landing-zone centre and descend to its z (no margins).

    Climb to transit_altitude_m (ENU z) first when declared and higher than the current altitude.
    The horizontal distance goes to the zone centre; the zone radius is not subtracted.
    """
    p, centre = vector3(position_enu, 'position_enu'), np.asarray(landing_zone.center_enu, float)
    transit = p[2] if transit_altitude_m is None else max(p[2], _real(transit_altitude_m, 'transit_altitude_m'))
    return ((transit - p[2]) / _positive(climb_speed_m_s, 'climb_speed_m_s')
            + math.hypot(*(centre[:2] - p[:2])) / _positive(speed_m_s, 'speed_m_s')
            + max(transit - centre[2], 0.) / _positive(descent_speed_m_s, 'descent_speed_m_s'))


class ReturnReserve:
    """Conservative return check on the mission budget (simulator seconds).

    estimate_s = time_margin_factor * time_to_return + fixed_margin_s, or inf when an obstacle map is
    given and the return path is not free. Budgets.return_reserve_s is kept as untouched slack on top:
    must_return(elapsed) is true once elapsed + estimate_s + return_reserve_s >= mission_time_s.
    """

    def __init__(self, budgets, landing_zone, speed_m_s=None, climb_speed_m_s=DEFAULT_CLIMB_SPEED_M_S,
                 descent_speed_m_s=DEFAULT_DESCENT_SPEED_M_S, time_margin_factor=DEFAULT_TIME_MARGIN_FACTOR,
                 fixed_margin_s=0., transit_altitude_m=None, obstacle_map=None, clearance_radius_m=DEFAULT_CLEARANCE_RADIUS_M):
        self.budgets, self.landing_zone = budgets, landing_zone
        self.speed_m_s = budgets.max_speed_m_s if speed_m_s is None else _positive(speed_m_s, 'speed_m_s')
        if self.speed_m_s > budgets.max_speed_m_s:
            raise PlanningError('speed_m_s exceeds Budgets.max_speed_m_s')
        self.climb_speed_m_s, self.descent_speed_m_s = _positive(climb_speed_m_s, 'climb_speed_m_s'), _positive(descent_speed_m_s, 'descent_speed_m_s')
        self.time_margin_factor = _real(time_margin_factor, 'time_margin_factor', low=1)
        self.fixed_margin_s = _real(fixed_margin_s, 'fixed_margin_s', low=0)
        self.transit_altitude_m, self.obstacle_map, self.clearance_radius_m = transit_altitude_m, obstacle_map, clearance_radius_m

    def return_path(self, position_enu):
        p, centre = vector3(position_enu, 'position_enu'), np.asarray(self.landing_zone.center_enu, float)
        z = p[2] if self.transit_altitude_m is None else max(p[2], self.transit_altitude_m)
        points = [p, np.array([p[0], p[1], z]), np.array([centre[0], centre[1], z]), centre]
        return tuple(tuple(q) for i, q in enumerate(points) if i == 0 or not np.allclose(q, points[i - 1]))

    def path_free(self, position_enu):
        if self.obstacle_map is None:
            return True
        path = self.return_path(position_enu)
        return all(self.obstacle_map.segment_is_free(a, b, self.clearance_radius_m) for a, b in zip(path[:-1], path[1:]))

    def estimate_s(self, position_enu):
        if not self.path_free(position_enu):
            return math.inf
        return self.time_margin_factor * time_to_return(position_enu, self.landing_zone, self.speed_m_s, self.climb_speed_m_s,
                                                        self.descent_speed_m_s, self.transit_altitude_m) + self.fixed_margin_s

    def remaining_s(self, elapsed_s):
        return self.budgets.mission_time_s - _real(elapsed_s, 'elapsed_s', low=0)

    def must_return(self, elapsed_s, position_enu):
        return self.remaining_s(elapsed_s) <= self.estimate_s(position_enu) + self.budgets.return_reserve_s

    def allows(self, elapsed_s, duration_s, end_position_enu, estimate_s=None):
        """True when after an action of duration_s ending at end_position_enu a return is still affordable. estimate_s: an
        already computed estimate_s(end_position_enu) (callers that memoise it); None computes it."""
        estimate = self.estimate_s(end_position_enu) if estimate_s is None else estimate_s
        return self.remaining_s(elapsed_s) - _real(duration_s, 'duration_s', low=0) > estimate + self.budgets.return_reserve_s


# --- Swath and coverage raster -----------------------------------------------------------

@dataclass(frozen=True)
class Swath:
    """Centred rectangle guaranteed inside the footprint, in the vehicle heading frame (metres).

    width_m is the narrower of the near and far image edges on the ground; forward_min_m and
    forward_max_m bound the along-track interval ahead of the vehicle covered over that width.
    """
    width_m: float
    forward_min_m: float
    forward_max_m: float

    @property
    def depth_m(self):
        return self.forward_max_m - self.forward_min_m


def swath_geometry(altitude_agl, sensor_model, mount_pitch_rad, max_range_m=DEFAULT_MAX_RANGE_M):
    polygon = camera_ground_footprint((0., 0., altitude_agl), 0., altitude_agl, sensor_model, mount_pitch_rad, max_range_m)
    if not is_convex(polygon):
        raise PlanningError('footprint polygon is not convex; change max_range_m or the mount pitch')
    width = min(abs(polygon[3, 1] - polygon[2, 1]), abs(polygon[0, 1] - polygon[1, 1]))
    chords = [_chord(polygon, level) for level in (-width / 2, width / 2)]
    if width <= _EPS or None in chords:
        raise PlanningError('degenerate footprint width')
    swath = Swath(float(width), max(c[0] for c in chords), min(c[1] for c in chords))
    if swath.depth_m <= _EPS:
        raise PlanningError('degenerate footprint depth')
    return swath


class CoverageGrid:
    """Public coverage raster over the world bounds, updated only from frames actually captured.

    A cell is covered once its centre lies inside the footprint of a recorded capture; the first
    capture sim time is kept per cell (nan while uncovered). Fractions are raster approximations.
    """

    def __init__(self, briefing, cell_size_m, mount_pitch_rad, max_range_m=DEFAULT_MAX_RANGE_M, ground_z_m=0.):
        self.briefing, self.sensor_model = briefing, briefing.sensor_model
        self.cell_size_m, self.mount_pitch_rad = _positive(cell_size_m, 'cell_size_m'), _pitch(mount_pitch_rad)
        self.max_range_m, self.ground_z_m = _positive(max_range_m, 'max_range_m'), _real(ground_z_m, 'ground_z_m')
        b = briefing.world_bounds
        self.nx = math.ceil((b.x_max - b.x_min) / self.cell_size_m - 1e-9)
        self.ny = math.ceil((b.y_max - b.y_min) / self.cell_size_m - 1e-9)
        self.xs = b.x_min + (np.arange(self.nx) + .5) * self.cell_size_m
        self.ys = b.y_min + (np.arange(self.ny) + .5) * self.cell_size_m
        gx, gy = np.meshgrid(self.xs, self.ys)
        centres = np.column_stack([gx.ravel(), gy.ravel()])
        self.sector_masks = {}
        for sector in briefing.search_sectors:
            mask = points_in_polygon(centres, sector.polygon_enu).reshape(self.ny, self.nx)
            if not mask.any():
                raise PlanningError(f'sector {sector.sector_id}: no cell centre inside; reduce cell_size_m')
            self.sector_masks[sector.sector_id] = mask
        self.covered_at_s = np.full((self.ny, self.nx), np.nan)
        self.captures = []
        self.last_capture_s = None

    @property
    def cell_area_m2(self):
        return self.cell_size_m ** 2

    def footprint(self, position_enu, yaw_rad):
        position = vector3(position_enu, 'position_enu')
        return camera_ground_footprint(position, yaw_rad, position[2] - self.ground_z_m, self.sensor_model,
                                       self.mount_pitch_rad, self.max_range_m)

    def cells_in_polygon(self, polygon_xy):
        """(rows, cols) of cells whose centre lies inside the polygon."""
        p = np.asarray(polygon_xy, float)
        cols = np.nonzero((self.xs >= p[:, 0].min()) & (self.xs <= p[:, 0].max()))[0]
        rows = np.nonzero((self.ys >= p[:, 1].min()) & (self.ys <= p[:, 1].max()))[0]
        rr, cc = (a.ravel() for a in np.meshgrid(rows, cols, indexing='ij'))
        inside = points_in_polygon(np.column_stack([self.xs[cc], self.ys[rr]]), p) if len(rr) else np.zeros(0, bool)
        return rr[inside], cc[inside]

    def record_capture(self, sim_time_s, position_enu, yaw_rad):
        """Mark the footprint of one captured frame; returns the number of newly covered cells."""
        t = _real(sim_time_s, 'sim_time_s', low=0)
        if self.last_capture_s is not None and t < self.last_capture_s - 1e-6:
            raise PlanningError('capture sim time moved backwards')
        rows, cols = self.cells_in_polygon(self.footprint(position_enu, yaw_rad))
        new = np.isnan(self.covered_at_s[rows, cols])
        self.covered_at_s[rows[new], cols[new]] = t
        self.last_capture_s = t
        self.captures.append((t, tuple(vector3(position_enu)), float(yaw_rad)))
        return int(new.sum())

    def record_frame(self, frame):
        """Record a public SensorFrame only when its RGB sample is valid; returns whether it was recorded."""
        rgb = next((s for s in frame.sensors if s.stream == 'rgb'), None)
        if rgb is None or not rgb.valid:
            return False
        r = quat_to_matrix(frame.vehicle.attitude_wxyz)
        self.record_capture(rgb.timestamp_s, frame.vehicle.position_enu, math.atan2(r[1, 0], r[0, 0]))
        return True

    def covered_mask(self):
        return np.isfinite(self.covered_at_s)

    def covered_fraction(self, sector_id):
        mask = self.sector_masks[sector_id]
        return float(np.count_nonzero(mask & self.covered_mask()) / np.count_nonzero(mask))

    def sector_coverage(self):
        return tuple(SectorCoverage(s, self.covered_fraction(s)) for s in self.sector_masks)

    def prior_density(self):
        """Prior mass per m^2: each sector's prior_weight spread uniformly over its cells."""
        density = np.zeros((self.ny, self.nx))
        for sector in self.briefing.search_sectors:
            mask = self.sector_masks[sector.sector_id]
            density[mask] += sector.prior_weight / (np.count_nonzero(mask) * self.cell_area_m2)
        return density


# --- Systematic coverage baseline ----------------------------------------------------------

@dataclass(frozen=True)
class Waypoint:
    index: int
    position_enu: Tuple[float, float, float]
    yaw_rad: float                # requested yaw along the lane
    sector_id: str
    lane: int
    clamped: bool                 # moved inside the bounds margin; flagged, never silent


@dataclass(frozen=True)
class PlanIssue:
    kind: str                     # 'waypoint', or 'leg' for the leg ending at waypoint index
    index: int
    reason: str
    clearance_m: float
    nearest_box: Optional[str]


@dataclass(frozen=True)
class CoveragePlan:
    start_enu: Tuple[float, float, float]
    waypoints: Tuple[Waypoint, ...]
    legs: Tuple[Leg, ...]         # legs[i] ends at waypoints[i]
    issues: Tuple[PlanIssue, ...]
    swath: Swath
    overlap_fraction: float       # declared minimum
    achieved_overlap: Tuple[Tuple[str, Optional[float], Optional[float]], ...]  # (sector, across, along); None: one lane/station

    @property
    def valid(self):
        return not self.issues

    @property
    def path_length_m(self):
        return float(sum(leg.length_m for leg in self.legs))

    def raise_if_invalid(self):
        if self.issues:
            raise PlanningError('coverage plan violates constraints: ' + '; '.join(
                f'{i.kind}[{i.index}] {i.reason} (clearance {i.clearance_m:.2f} m, {i.nearest_box})' for i in self.issues))

    def departure_times_s(self, speed_m_s, capture_time_s=DEFAULT_CAPTURE_TIME_S):
        """Seconds from plan start until the capture at each waypoint has finished."""
        speed = _positive(speed_m_s, 'speed_m_s')
        return tuple(float(t) for t in np.cumsum([leg.length_m / speed + capture_time_s for leg in self.legs]))

    def reserve_prefix(self, reserve, elapsed_s, capture_time_s=DEFAULT_CAPTURE_TIME_S):
        """Number of leading waypoints that keep a return affordable after each capture."""
        for n, (waypoint, t) in enumerate(zip(self.waypoints, self.departure_times_s(reserve.speed_m_s, capture_time_s))):
            if not reserve.allows(elapsed_s, t, waypoint.position_enu):
                return n
        return len(self.waypoints)


class CoverageBaseline:
    """Boustrophedon lawnmower over the public search sectors (convex sectors only; others rejected).

    Lanes run parallel to the sector edge that minimises the sector width across them. Lane spacing
    is at most swath.width_m * (1 - overlap_fraction) and capture stations along a lane at most
    swath.depth_m * (1 - overlap_fraction) apart, so neighbouring footprints overlap by at least the
    declared fraction when the requested yaw is flown; stations trail the sector edge by the swath
    forward offset. Sectors are visited by descending prior weight, then distance to their nearest
    entry, then sector_id. Waypoints outside the bounds shrunk by bounds_margin_m are clamped and
    flagged. Waypoint and leg obstacle/bounds violations are PlanIssues (strict=True raises); a
    climb-over detour is used only when climb_over_altitude_m is declared.
    """

    def __init__(self, briefing, obstacle_map, mount_pitch_rad, altitude_agl, overlap_fraction=.2,
                 clearance_radius_m=DEFAULT_CLEARANCE_RADIUS_M, max_range_m=DEFAULT_MAX_RANGE_M, ground_z_m=0.,
                 climb_over_altitude_m=None, bounds_margin_m=None):
        self.briefing, self.obstacle_map = briefing, obstacle_map
        self.altitude_agl, self.ground_z_m = _positive(altitude_agl, 'altitude_agl'), _real(ground_z_m, 'ground_z_m')
        self.overlap_fraction = _real(overlap_fraction, 'overlap_fraction', low=0, high=1, open_high=True)
        self.clearance_radius_m = _real(clearance_radius_m, 'clearance_radius_m', low=0)
        self.climb_over_altitude_m = climb_over_altitude_m
        self.bounds_margin_m = self.clearance_radius_m if bounds_margin_m is None else _real(bounds_margin_m, 'bounds_margin_m', low=0)
        self.swath = swath_geometry(self.altitude_agl, briefing.sensor_model, mount_pitch_rad, max_range_m)
        for sector in briefing.search_sectors:
            if not is_convex(sector.polygon_enu):
                raise PlanningError(f'sector {sector.sector_id}: only convex sectors are supported')

    def _orderings(self, sector):
        """Four boustrophedon orderings of (xy, yaw, lane) plus the achieved (across, along) overlap."""
        poly = np.asarray(sector.polygon_enu, float)
        best = None
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            if np.linalg.norm(edge) <= _EPS:
                continue
            u = edge / np.linalg.norm(edge)
            normal = np.array([-u[1], u[0]])
            width = float(np.ptp(poly @ normal))
            if best is None or width < best[0] - 1e-9:
                best = (width, u, normal)
        width, u, normal = best
        al = np.column_stack([poly @ u, poly @ normal])
        w, q, sw = self.swath.width_m, 1 - self.overlap_fraction, self.swath
        lo, hi = float(al[:, 1].min()), float(al[:, 1].max())
        if width <= w:
            offsets, across = [(lo + hi) / 2], None
        else:
            k = math.ceil((width - w) / (w * q) - 1e-9) + 1
            offsets, across = np.linspace(lo + w / 2, hi - w / 2, k).tolist(), 1 - (width - w) / (k - 1) / w
        lanes, along = [], None
        for c in offsets:
            levels = [min(max(c - w / 2, lo), hi), min(max(c + w / 2, lo), hi)] + [float(v) for v in al[:, 1] if c - w / 2 < v < c + w / 2]
            chords = [ch for ch in (_chord(al, level) for level in levels) if ch is not None]
            s0, s1 = min(ch[0] for ch in chords), max(ch[1] for ch in chords)
            span = s1 - s0 - sw.depth_m
            if span <= 0:
                forward = [(s0 + s1) / 2 - (sw.forward_min_m + sw.forward_max_m) / 2]
            else:
                m = math.ceil(span / (sw.depth_m * q) - 1e-9) + 1
                forward = np.linspace(s0 - sw.forward_min_m, s1 - sw.forward_max_m, m).tolist()
                along = min(1 - span / (m - 1) / sw.depth_m, 1. if along is None else along)
            lanes.append((c, forward, [s0 + s1 - s for s in forward]))  # stations heading +u, mirrored for -u
        orderings = []
        for lane_order in (range(len(lanes)), range(len(lanes) - 1, -1, -1)):
            for first in (1, -1):
                sequence = []
                for j, lane in enumerate(lane_order):
                    heading = first * (-1) ** j
                    c, plus, minus = lanes[lane]
                    yaw = math.atan2(heading * u[1], heading * u[0])
                    sequence += [(s * u + c * normal, yaw, lane) for s in (plus if heading > 0 else minus)]
                orderings.append(sequence)
        return orderings, across, along

    def plan(self, start_enu, strict=False):
        start = vector3(start_enu, 'start_enu')
        z, b, margin = self.ground_z_m + self.altitude_agl, self.briefing.world_bounds, self.bounds_margin_m
        prepared = {s.sector_id: self._orderings(s) for s in self.briefing.search_sectors}
        remaining, waypoints, overlaps, current = list(self.briefing.search_sectors), [], [], start

        def entry(sector):
            options = prepared[sector.sector_id][0]
            k = min(range(len(options)), key=lambda i: (math.dist(options[i][0][0], current[:2]), i))
            return k, math.dist(options[k][0][0], current[:2])

        while remaining:
            sector = min(remaining, key=lambda s: (-s.prior_weight, entry(s)[1], s.sector_id))
            remaining.remove(sector)
            options, across, along = prepared[sector.sector_id]
            for xy, yaw, lane in options[entry(sector)[0]]:
                x, y = min(max(xy[0], b.x_min + margin), b.x_max - margin), min(max(xy[1], b.y_min + margin), b.y_max - margin)
                clamped = abs(x - xy[0]) > 1e-9 or abs(y - xy[1]) > 1e-9
                waypoints.append(Waypoint(len(waypoints), (float(x), float(y), z), float(yaw), sector.sector_id, lane, clamped))
            overlaps.append((sector.sector_id, across, along))
            current = np.asarray(waypoints[-1].position_enu)
        legs, issues, previous = [], [], start
        for waypoint in waypoints:
            point = self.obstacle_map.check_segment(waypoint.position_enu, waypoint.position_enu, self.clearance_radius_m)
            if not point.free:
                issues.append(PlanIssue('waypoint', waypoint.index, point.reason, point.clearance_m, point.nearest_box))
            leg = plan_leg(self.obstacle_map, previous, waypoint.position_enu, self.clearance_radius_m, self.climb_over_altitude_m)
            if not leg.free:
                issues.append(PlanIssue('leg', waypoint.index, leg.reason, leg.clearance_m, leg.nearest_box))
            legs.append(leg)
            previous = waypoint.position_enu
        plan = CoveragePlan(tuple(start), tuple(waypoints), tuple(legs), tuple(issues), self.swath, self.overlap_fraction, tuple(overlaps))
        if strict:
            plan.raise_if_invalid()
        return plan


# --- Information-per-cost planner --------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True)
class GoalOption:
    candidate_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    feasible: bool
    reason: Optional[str]         # None, 'no_uncovered_prior', a Leg reason, or 'return_reserve'
    gain: float                   # range-weighted uncovered prior mass in the footprint (heuristic)
    travel_time_s: float
    duration_s: float             # travel + capture
    score: float                  # gain / duration_s when feasible, else 0
    leg: Leg


class InformationCostPlanner:
    """Greedy information-per-cost goal choice. A declared heuristic, not a calibrated probability.

    gain(c)  = sum over uncovered cells inside the footprint at c of prior_density * cell_area * range_proxy
    score(c) = gain(c) / (travel_time_s + capture_time_s)
    prior_density spreads each sector prior uniformly over its raster cells; range_proxy (the detection
    probability proxy) is 1 up to full_weight_range_m slant range and falls linearly to 0 at the grid
    max_range_m. Travel follows plan_leg at reserve.speed_m_s; yaw turns are not costed. A candidate is
    feasible only with a free leg and when ReturnReserve.allows the action. Ties: shorter travel, then
    candidate_id. Only public inputs: briefing, obstacle map, coverage grid, vehicle position, elapsed time.
    Exact speed-ups (tests/test_information_cost_choose.py): the footprint cells with their range proxy and the return estimate
    of a pose are memoised (they depend only on the pose and the fixed grid geometry, obstacle map and reserve; coverage is read
    at every call), and choose visits candidates by a score upper bound instead of planning every leg.
    """

    def __init__(self, briefing, obstacle_map, grid, reserve, altitude_agl, clearance_radius_m=DEFAULT_CLEARANCE_RADIUS_M,
                 capture_time_s=DEFAULT_CAPTURE_TIME_S, full_weight_range_m=None, candidate_spacing_m=None, yaw_count=4,
                 climb_over_altitude_m=None, bounds_margin_m=None):
        self.briefing, self.obstacle_map, self.grid, self.reserve = briefing, obstacle_map, grid, reserve
        self.altitude_agl = _positive(altitude_agl, 'altitude_agl')
        self.clearance_radius_m = _real(clearance_radius_m, 'clearance_radius_m', low=0)
        self.bounds_margin_m = self.clearance_radius_m if bounds_margin_m is None else _real(bounds_margin_m, 'bounds_margin_m', low=0)
        self.capture_time_s = _positive(capture_time_s, 'capture_time_s')
        self.full_weight_range_m = grid.max_range_m / 2 if full_weight_range_m is None else \
            _real(full_weight_range_m, 'full_weight_range_m', low=0, high=grid.max_range_m, open_high=True)
        swath = swath_geometry(self.altitude_agl, briefing.sensor_model, grid.mount_pitch_rad, grid.max_range_m)
        self.candidate_spacing_m = swath.width_m / 2 if candidate_spacing_m is None else _positive(candidate_spacing_m, 'candidate_spacing_m')
        if isinstance(yaw_count, bool) or not isinstance(yaw_count, int) or yaw_count < 1:
            raise PlanningError('yaw_count: expected integer >= 1')
        self.yaw_count, self.climb_over_altitude_m = yaw_count, climb_over_altitude_m
        self._density = grid.prior_density()
        self._cells, self._estimates = {}, {}

    @staticmethod
    def from_waypoints(waypoints):
        return tuple(Candidate(f'wp{w.index}', w.position_enu, w.yaw_rad) for w in waypoints)

    def candidates(self):
        """Lattice over the sector bounding boxes grown by max_range_m, inside the bounds shrunk by bounds_margin_m
        (default the clearance radius)."""
        b, r, reach = self.briefing.world_bounds, self.bounds_margin_m, self.grid.max_range_m
        corners = np.concatenate([np.asarray(s.polygon_enu, float) for s in self.briefing.search_sectors])
        x0, x1 = max(b.x_min + r, corners[:, 0].min() - reach), min(b.x_max - r, corners[:, 0].max() + reach)
        y0, y1 = max(b.y_min + r, corners[:, 1].min() - reach), min(b.y_max - r, corners[:, 1].max() + reach)
        z = self.grid.ground_z_m + self.altitude_agl
        xs, ys = np.arange(x0, x1 + 1e-9, self.candidate_spacing_m), np.arange(y0, y1 + 1e-9, self.candidate_spacing_m)
        return tuple(Candidate(f'c{i}-{j}-y{k}', (float(x), float(y), z), wrap_angle(2 * math.pi * k / self.yaw_count))
                     for i, x in enumerate(xs) for j, y in enumerate(ys) for k in range(self.yaw_count))

    def _footprint_cells(self, position_enu, yaw_rad):
        """(rows, cols, range proxy) of the footprint cells at a pose, memoised: the pose and the grid geometry fix them."""
        p = vector3(position_enu, 'position_enu')
        key = (float(p[0]), float(p[1]), float(p[2]), float(yaw_rad))
        cells = self._cells.get(key)
        if cells is None:
            rows, cols = self.grid.cells_in_polygon(self.grid.footprint(p, yaw_rad))
            slant = np.sqrt((self.grid.xs[cols] - p[0]) ** 2 + (self.grid.ys[rows] - p[1]) ** 2 + (p[2] - self.grid.ground_z_m) ** 2)
            proxy = np.clip((self.grid.max_range_m - slant) / max(self.grid.max_range_m - self.full_weight_range_m, _EPS), 0., 1.)
            cells = self._cells[key] = (rows, cols, proxy)
        return cells

    def gain(self, position_enu, yaw_rad):
        rows, cols, proxy = self._footprint_cells(position_enu, yaw_rad)
        keep = np.isnan(self.grid.covered_at_s[rows, cols])
        return float(np.sum(self._density[rows[keep], cols[keep]] * proxy[keep]) * self.grid.cell_area_m2)

    def _return_estimate(self, position_enu):
        key = tuple(float(v) for v in position_enu)
        if key not in self._estimates:
            self._estimates[key] = self.reserve.estimate_s(key)
        return self._estimates[key]

    def _option(self, candidate, position, elapsed, gain):
        c = candidate
        leg = plan_leg(self.obstacle_map, position, c.position_enu, self.clearance_radius_m, self.climb_over_altitude_m)
        travel = leg.length_m / self.reserve.speed_m_s
        duration = travel + self.capture_time_s
        reason = 'no_uncovered_prior' if gain <= 0 else leg.reason if not leg.free else \
            None if self.reserve.allows(elapsed, duration, c.position_enu, self._return_estimate(c.position_enu)) else 'return_reserve'
        return GoalOption(c.candidate_id, tuple(c.position_enu), c.yaw_rad, reason is None, reason, gain, travel,
                          duration, gain / duration if reason is None else 0., leg)

    def rank(self, position_enu, elapsed_s, candidates=None):
        """All candidates: feasible ones by (-score, travel, id), then infeasible ones by id with reasons."""
        position, elapsed = vector3(position_enu, 'position_enu'), _real(elapsed_s, 'elapsed_s', low=0)
        options = [self._option(c, position, elapsed, self.gain(c.position_enu, c.yaw_rad))
                   for c in (self.candidates() if candidates is None else candidates)]
        feasible = sorted((o for o in options if o.feasible), key=lambda o: (-o.score, o.travel_time_s, o.candidate_id))
        return tuple(feasible) + tuple(sorted((o for o in options if not o.feasible), key=lambda o: o.candidate_id))

    def choose(self, position_enu, elapsed_s, candidates=None):
        """Best feasible goal, or None (nothing affordable: the caller should RETURN); equal to rank(...)[0] when that option
        is feasible, else None. A leg is never shorter than the straight line, so gain / (distance / speed + capture_time_s)
        bounds the score from above; candidates with gain > 0 are visited by decreasing bound and the search stops once the
        bound falls below the best feasible score by more than a relative 1e-9 (ties are still evaluated). With a declared
        climb_over_altitude_m or duplicate candidate ids the full rank is used."""
        position, elapsed = vector3(position_enu, 'position_enu'), _real(elapsed_s, 'elapsed_s', low=0)
        pool = tuple(self.candidates() if candidates is None else candidates)
        if self.climb_over_altitude_m is not None or len({c.candidate_id for c in pool}) != len(pool):
            # the bound and the tie order are exact only for direct legs and unique ids (second review): full rank instead
            ranked = self.rank(position, elapsed, pool)
            return ranked[0] if ranked and ranked[0].feasible else None
        bounded = []
        for c in pool:
            gain = self.gain(c.position_enu, c.yaw_rad)
            if gain > 0:
                distance = float(np.linalg.norm(vector3(c.position_enu, 'position_enu') - position))
                bounded.append((gain / (distance / self.reserve.speed_m_s + self.capture_time_s), gain, c))
        bounded.sort(key=lambda item: -item[0])
        best = None
        for bound, gain, c in bounded:
            if best is not None and bound < best.score * (1 - 1e-9):
                break
            option = self._option(c, position, elapsed, gain)
            if option.feasible and (best is None or (-option.score, option.travel_time_s, option.candidate_id)
                                    < (-best.score, best.travel_time_s, best.candidate_id)):
                best = option
        return best


# --- Inspection viewpoints ---------------------------------------------------------------

@dataclass(frozen=True)
class InspectionView:
    view_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    standoff_m: float             # horizontal distance to the track
    bearing_rad: float            # ENU direction from the track to the viewpoint
    pixel_uv: Tuple[float, float]  # projected track position (nan behind the camera)
    path_length_m: float
    estimated_duration_s: float   # travel at speed + capture
    valid: bool
    reason: Optional[str]
    leg: Leg

    def to_view_option(self, track_id=None):
        if not self.valid:
            raise PlanningError(f'{self.view_id}: invalid view ({self.reason})')
        return ViewOption(self.view_id, self.position_enu, self.yaw_rad, self.path_length_m, self.estimated_duration_s, track_id)


def _track_covariance(track_covariance_enu):
    cov = np.asarray(track_covariance_enu, float)
    if cov.shape != (3, 3) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-9) or np.linalg.eigvalsh(cov).min() < -1e-9:
        raise PlanningError('track_covariance_enu: expected finite symmetric positive semidefinite 3x3')
    return cov


def evaluate_viewpoint(track_position_enu, track_covariance_enu, view_id, position_enu, yaw_rad, standoff_m, bearing_rad, obstacle_map,
                       sensor_model, mount_pitch_rad, current_position_enu, speed_m_s, clearance_radius_m=DEFAULT_CLEARANCE_RADIUS_M,
                       capture_time_s=DEFAULT_CAPTURE_TIME_S, max_range_m=DEFAULT_MAX_RANGE_M, centre_window_fraction=.5,
                       sigma_scale=2., climb_over_altitude_m=None, reserve=None, elapsed_s=None, bounds_margin_m=None):
    """The inspection_views checks for one vehicle pose (position_enu, yaw_rad) looking at a public track.

    standoff_m and bearing_rad are recorded, not used by the checks (inspection_views passes its radius and bearing;
    view_graph passes values recomputed from a node pose). Reasons and their order are those of inspection_views.
    """
    track, current = vector3(track_position_enu, 'track_position_enu'), vector3(current_position_enu, 'current_position_enu')
    cov = _track_covariance(track_covariance_enu)
    if reserve is not None and elapsed_s is None:
        raise PlanningError('elapsed_s required with a reserve')
    speed, pitch, frac = _positive(speed_m_s, 'speed_m_s'), _pitch(mount_pitch_rad), _real(centre_window_fraction, 'centre_window_fraction', low=0, high=1, open_low=True)
    eigval, eigvec = np.linalg.eigh(cov)
    spread = [s * _real(sigma_scale, 'sigma_scale', low=0) * math.sqrt(max(val, 0.)) * eigvec[:, i] for i, val in enumerate(eigval) for s in (-1, 1)]
    width, height = sensor_model.rgb_width, sensor_model.rgb_height
    position, yaw = vector3(position_enu, 'position_enu'), _real(yaw_rad, 'yaw_rad')

    def in_image(point):
        u, v, depth = project_to_image(point, position, yaw, pitch, sensor_model)
        return depth > 0 and 0 <= u <= width and 0 <= v <= height

    u, v, depth = project_to_image(track, position, yaw, pitch, sensor_model)
    leg = plan_leg(obstacle_map, current, position, clearance_radius_m, climb_over_altitude_m)
    duration = leg.length_m / speed + capture_time_s
    point = obstacle_map.check_segment(position, position, clearance_radius_m)
    ray = track - position
    slant = float(np.linalg.norm(ray))
    reason = None
    if not point.free:
        reason = 'viewpoint_outside_bounds' if point.reason == 'outside_bounds' else 'viewpoint_in_obstacle'
    elif bounds_margin_m is not None and not obstacle_map.inside_bounds(position, bounds_margin_m):
        reason = 'viewpoint_outside_bounds'  # inside the clearance radius but within the declared goal margin band
    elif depth <= 0:
        reason = 'track_behind_camera'
    elif abs(u - width / 2) > frac * width / 2 or abs(v - height / 2) > frac * height / 2:
        reason = 'track_off_centre'
    elif slant > max_range_m:
        reason = 'beyond_max_range'
    elif not all(in_image(track + offset) for offset in spread):
        reason = 'uncertainty_exceeds_image'
    elif obstacle_map.segment_clearance(position, track - ray / slant * min(1., slant / 2)) <= 0:
        reason = 'line_of_sight_blocked'
    elif not leg.free:
        reason = f'path_{leg.reason}'
    elif reserve is not None and not reserve.allows(elapsed_s, duration, position):
        reason = 'return_reserve'
    return InspectionView(view_id, tuple(float(x) for x in position), yaw, _real(standoff_m, 'standoff_m', low=0),
                          _real(bearing_rad, 'bearing_rad'), (u, v), leg.length_m, duration, reason is None, reason, leg)


def inspection_views(track_position_enu, track_covariance_enu, altitude_agl, radii_m, n_angles, obstacle_map, sensor_model,
                     mount_pitch_rad, current_position_enu, speed_m_s, clearance_radius_m=DEFAULT_CLEARANCE_RADIUS_M,
                     capture_time_s=DEFAULT_CAPTURE_TIME_S, ground_z_m=0., max_range_m=DEFAULT_MAX_RANGE_M,
                     centre_window_fraction=.5, sigma_scale=2., start_bearing_rad=0., climb_over_altitude_m=None,
                     reserve=None, elapsed_s=None, view_prefix='view', bounds_margin_m=None):
    """Candidate re-inspection viewpoints around a public track, each with its path cost.

    Viewpoints sit at altitude_agl above ground_z_m, at horizontal standoff r in radii_m and bearing
    start_bearing_rad + 2*pi*k/n_angles from the track, yawed to face it. With a fixed pitch the track
    is exactly centred only at r = (z_camera - z_track) / tan(mount_pitch_rad); other radii stay valid
    while the track projects inside the central window (centre_window_fraction of width and height).
    The first failing check is the reason: viewpoint_outside_bounds, viewpoint_in_obstacle,
    track_behind_camera, track_off_centre, beyond_max_range (slant), uncertainty_exceeds_image (the
    sigma_scale-sigma principal-axis points must project inside the image), line_of_sight_blocked
    (public static boxes only), path_<leg reason>, return_reserve (when a reserve is given). Geometric
    view quality is not detectability. The per-viewpoint checks are evaluate_viewpoint.
    """
    track, current = vector3(track_position_enu, 'track_position_enu'), vector3(current_position_enu, 'current_position_enu')
    cov = _track_covariance(track_covariance_enu)
    if isinstance(n_angles, bool) or not isinstance(n_angles, int) or n_angles < 1 or not len(radii_m):
        raise PlanningError('n_angles >= 1 and at least one radius required')
    if reserve is not None and elapsed_s is None:
        raise PlanningError('elapsed_s required with a reserve')
    _positive(speed_m_s, 'speed_m_s'), _pitch(mount_pitch_rad), _real(centre_window_fraction, 'centre_window_fraction', low=0, high=1, open_low=True)
    _real(sigma_scale, 'sigma_scale', low=0)
    z = _real(ground_z_m, 'ground_z_m') + _positive(altitude_agl, 'altitude_agl')
    views = []
    for ri, radius in enumerate(_positive(r, 'radii_m') for r in radii_m):
        for k in range(n_angles):
            bearing = wrap_angle(_real(start_bearing_rad, 'start_bearing_rad') + 2 * math.pi * k / n_angles)
            position = np.array([track[0] + radius * math.cos(bearing), track[1] + radius * math.sin(bearing), z])
            views.append(evaluate_viewpoint(track, cov, f'{view_prefix}-r{ri}-a{k}', position, wrap_angle(bearing + math.pi), radius, bearing,
                                            obstacle_map, sensor_model, mount_pitch_rad, current, speed_m_s, clearance_radius_m,
                                            capture_time_s, max_range_m, centre_window_fraction, sigma_scale, climb_over_altitude_m,
                                            reserve, elapsed_s, bounds_margin_m))
    return tuple(views)
