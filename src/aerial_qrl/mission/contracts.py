"""Typed mission contracts with strict validation (schema flood-sar-mission/1).

Public/private split
    Public: MissionBriefing, PublicObservation, Action (and the SensorFrame/CommandOutcome of
    backend.py). They never carry evaluator truth, scenario IDs or seeds; a key-name guard
    (leakage.py) is applied on construction. Private: ScenarioSpec bookkeeping (IDs, hashes,
    RNG streams, timing policy), EvaluatorSpec, EpisodeResult. Keep them in separate files.

Units and frames
    Metres, seconds, radians. Positions/velocities in world ENU (see frames.py);
    quaternions scalar-first wxyz, parent<-child.

Time base
    sim_time_s: simulator clock in seconds since the backend's last successful reset.
    elapsed_s: simulator seconds since the episode began (so elapsed_s <= sim_time_s).
    Sensor timestamp_s uses the same clock; age_s = sim_time_s - timestamp_s.
    Wall-clock time appears only in EpisodeResult.wall_time_s and run logs.

Failure handling
    Validation rejects unknown or missing fields, NaN/inf, booleans used as numbers,
    inconsistent counts or times; it never repairs input. Missing sensor data is an invalid
    SensorStatus, never an old sample relabelled as fresh.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
import json
import math
import re
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .leakage import assert_public_payload

SCHEMA_VERSION = 'flood-sar-mission/1'
#: Task flood-sar-village-gnss/1 (docs/study/gnss_qrl_v1/MODEL_CONTRACT_EN.md): the same scene, local route graph and
#: cached observations, plus an uncertain GLOBAL georeference. A payload carrying this version MUST populate the
#: georeference/anchor fields of PublicObservation; a payload carrying SCHEMA_VERSION must leave them empty, and
#: serializes byte-for-byte as before (PublicObservation.to_dict).
GNSS_SCHEMA_VERSION = 'flood-sar-village-gnss/1'
SCHEMA_VERSIONS = (SCHEMA_VERSION, GNSS_SCHEMA_VERSION)
SPLITS = ('train', 'development', 'test')
BACKENDS = ('gazebo_px4', 'esar_airsim', 'fake', 'cached_view')  # cached_view: ADR 006 fast decision environment
PROVENANCE = ('engineering_default_unfrozen', 'frozen')
TERMINAL_REASONS = ('mission_complete', 'mission_deadline', 'collision', 'safety_abort')
TRUNCATION_REASONS = ('training_horizon',)
INFRASTRUCTURE = 'infrastructure_failure'
TIME_TOLERANCE_S = 1e-6
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_IDENT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')


class ContractError(ValueError):
    pass


def _reject_constant(name):
    raise ContractError(f'non-finite JSON constant {name}')


def loads(text):
    """json.loads that refuses NaN/Infinity literals instead of producing floats."""
    return json.loads(text, parse_constant=_reject_constant)


def _real(value, name, low=None, high=None, open_low=False, open_high=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f'{name}: expected finite number')
    value = float(value)
    if low is not None and (value < low or (open_low and value == low)):
        raise ContractError(f'{name}: out of range')
    if high is not None and (value > high or (open_high and value == high)):
        raise ContractError(f'{name}: out of range')
    return value


def _optional_real(value, name, **limits):
    return None if value is None else _real(value, name, **limits)


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f'{name}: expected integer >= {minimum}')
    return value


def _flag(value, name):
    if not isinstance(value, bool):
        raise ContractError(f'{name}: expected boolean')
    return value


def _ident(value, name):
    if not isinstance(value, str) or not _IDENT.match(value):
        raise ContractError(f'{name}: invalid identifier')
    return value


def _optional_ident(value, name):
    return None if value is None else _ident(value, name)


def _sha256(value, name):
    if not isinstance(value, str) or not _SHA256.match(value):
        raise ContractError(f'{name}: expected lowercase sha256 hex')
    return value


def _choice(value, name, options):
    if value not in options:
        raise ContractError(f'{name}: expected one of {options}')
    return value


def _list(value, name):
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise ContractError(f'{name}: expected list')
    return value


def _vec(value, name, n=3):
    value = _list(value, name)
    if len(value) != n:
        raise ContractError(f'{name}: expected {n} values')
    return tuple(_real(x, f'{name}[{i}]') for i, x in enumerate(value))


def _nested(cls, value, name):
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise ContractError(f'{name}: expected {cls.__name__} object')
    return cls.from_dict(value)


def _nested_seq(cls, value, name):
    return tuple(_nested(cls, x, f'{name}[{i}]') for i, x in enumerate(_list(value, name)))


def _unique(values, name):
    if len(set(values)) != len(values):
        raise ContractError(f'{name}: duplicate IDs')


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class Contract:
    def _set(self, **values):
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def to_dict(self):
        return _plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise ContractError(f'{cls.__name__}: expected object')
        names = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
        extra, missing = set(data) - names, required - set(data)
        if extra or missing:
            raise ContractError(f'{cls.__name__}: unknown fields {sorted(extra)}, missing fields {sorted(missing)}')
        return cls(**data)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(loads(text))


def _check_schema(value, allowed=(SCHEMA_VERSION,)):
    """Schema gate. The default keeps every legacy contract on flood-sar-mission/1; contracts that the GNSS task
    versions pass SCHEMA_VERSIONS explicitly, so a GNSS payload can never be read as a legacy one by accident."""
    if value not in allowed:
        raise ContractError(f'schema_version: expected {allowed[0] if len(allowed) == 1 else sorted(allowed)}')
    return value


# --- Scenario (public briefing plus private bookkeeping) -------------------------------

@dataclass(frozen=True)
class Bounds2D(Contract):
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self):
        self._set(**{k: _real(getattr(self, k), k) for k in ('x_min', 'x_max', 'y_min', 'y_max')})
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ContractError('Bounds2D: empty extent')

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class Zone(Contract):
    zone_id: str
    center_enu: Tuple[float, float, float]
    radius_m: float

    def __post_init__(self):
        self._set(zone_id=_ident(self.zone_id, 'zone_id'), center_enu=_vec(self.center_enu, 'center_enu'),
                  radius_m=_real(self.radius_m, 'radius_m', low=0, open_low=True))


@dataclass(frozen=True)
class SearchSector(Contract):
    sector_id: str
    polygon_enu: Tuple[Tuple[float, float], ...]
    prior_weight: float

    def __post_init__(self):
        polygon = tuple(_vec(p, f'polygon_enu[{i}]', 2) for i, p in enumerate(_list(self.polygon_enu, 'polygon_enu')))
        if len(polygon) < 3:
            raise ContractError('polygon_enu: at least three vertices required')
        xy = np.asarray(polygon)
        area = .5 * abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))))
        if area <= 0:
            raise ContractError('polygon_enu: degenerate polygon')
        self._set(sector_id=_ident(self.sector_id, 'sector_id'), polygon_enu=polygon,
                  prior_weight=_real(self.prior_weight, 'prior_weight', low=0))


@dataclass(frozen=True)
class SensorModel(Contract):
    model_id: str
    rgb_width: int
    rgb_height: int
    rgb_hfov_rad: float
    depth_width: int
    depth_height: int
    depth_hfov_rad: float
    depth_min_m: float
    depth_max_m: float
    calibration_status: str
    source: str

    def __post_init__(self):
        self._set(model_id=_ident(self.model_id, 'model_id'),
                  calibration_status=_choice(self.calibration_status, 'calibration_status', ('specification_only', 'calibrated')))
        for key in ('rgb_width', 'rgb_height', 'depth_width', 'depth_height'):
            _count(getattr(self, key), key, minimum=1)
        for key in ('rgb_hfov_rad', 'depth_hfov_rad'):
            self._set(**{key: _real(getattr(self, key), key, low=0, high=math.pi, open_low=True, open_high=True)})
        self._set(depth_min_m=_real(self.depth_min_m, 'depth_min_m', low=0, open_low=True),
                  depth_max_m=_real(self.depth_max_m, 'depth_max_m', low=0, open_low=True))
        if self.depth_min_m >= self.depth_max_m:
            raise ContractError('depth range: min must be below max')
        if not isinstance(self.source, str) or not self.source:
            raise ContractError('source: required provenance path or URL')


@dataclass(frozen=True)
class Budgets(Contract):
    mission_time_s: float
    return_reserve_s: float
    max_speed_m_s: float
    provenance: str

    def __post_init__(self):
        self._set(mission_time_s=_real(self.mission_time_s, 'mission_time_s', low=0, open_low=True),
                  return_reserve_s=_real(self.return_reserve_s, 'return_reserve_s', low=0),
                  max_speed_m_s=_real(self.max_speed_m_s, 'max_speed_m_s', low=0, open_low=True),
                  provenance=_choice(self.provenance, 'provenance', PROVENANCE))
        if self.return_reserve_s >= self.mission_time_s:
            raise ContractError('return_reserve_s must be below mission_time_s')


@dataclass(frozen=True)
class SemiMarkovDiscount(Contract):
    """gamma(duration) = gamma_ref ** (duration_s / reference_duration_s)."""
    gamma_ref: float
    reference_duration_s: float
    provenance: str

    def __post_init__(self):
        self._set(gamma_ref=_real(self.gamma_ref, 'gamma_ref', low=0, high=1, open_low=True),
                  reference_duration_s=_real(self.reference_duration_s, 'reference_duration_s', low=0, open_low=True),
                  provenance=_choice(self.provenance, 'provenance', PROVENANCE))

    def discount(self, duration_s):
        return self.gamma_ref ** (_real(duration_s, 'duration_s', low=0, open_low=True) / self.reference_duration_s)


@dataclass(frozen=True)
class TimingPolicy(Contract):
    """Freshness and synchronisation limits on the simulator clock, in seconds.

    A sample older than max_sample_age_s must be reported invalid; a backend that marks it
    valid violates the contract. Skews compare timestamps of samples that are all valid.
    """
    max_sample_age_s: float
    max_rgb_depth_skew_s: float
    max_image_state_skew_s: float
    provenance: str

    def __post_init__(self):
        for key in ('max_sample_age_s', 'max_rgb_depth_skew_s', 'max_image_state_skew_s'):
            self._set(**{key: _real(getattr(self, key), key, low=0, open_low=True)})
        self._set(provenance=_choice(self.provenance, 'provenance', PROVENANCE))

    def check(self, sim_time_s, sensors):
        check_sensor_times(sim_time_s, sensors)
        valid = {s.stream: s for s in sensors if s.valid}
        for status in valid.values():
            if status.age_s > self.max_sample_age_s + TIME_TOLERANCE_S:
                raise ContractError(f'{status.stream}: stale sample marked valid')
        if 'rgb' in valid and 'depth' in valid and \
                abs(valid['rgb'].timestamp_s - valid['depth'].timestamp_s) > self.max_rgb_depth_skew_s + TIME_TOLERANCE_S:
            raise ContractError('rgb/depth timestamps exceed max_rgb_depth_skew_s')
        for image in ('rgb', 'depth'):
            if image in valid and 'vehicle_state' in valid and \
                    abs(valid[image].timestamp_s - valid['vehicle_state'].timestamp_s) > self.max_image_state_skew_s + TIME_TOLERANCE_S:
                raise ContractError(f'{image}/vehicle_state timestamps exceed max_image_state_skew_s')


def check_sensor_times(sim_time_s, sensors):
    """Clock consistency that holds for any policy: unique streams, no future samples, exact ages."""
    _unique([s.stream for s in sensors], 'sensors')
    for status in sensors:
        if status.timestamp_s is None:
            continue
        if status.timestamp_s > sim_time_s + TIME_TOLERANCE_S:
            raise ContractError(f'{status.stream}: timestamp_s is after sim_time_s')
        if abs(status.age_s - (sim_time_s - status.timestamp_s)) > TIME_TOLERANCE_S:
            raise ContractError(f'{status.stream}: age_s inconsistent with sim_time_s - timestamp_s')


@dataclass(frozen=True)
class RngStreams(Contract):
    """Independent streams: numpy default_rng([stream_id, seed]) per purpose."""
    seed: int
    layout: int
    prior: int
    targets: int
    nuisance: int
    sensor_noise: int

    def __post_init__(self):
        ids = [_count(getattr(self, k), k) for k in ('layout', 'prior', 'targets', 'nuisance', 'sensor_noise')]
        _count(self.seed, 'seed')
        _unique(ids, 'rng stream')


@dataclass(frozen=True)
class MissionBriefing(Contract):
    """The only scenario information a policy or planner may receive."""
    world_bounds: Bounds2D
    search_sectors: Tuple[SearchSector, ...]
    launch_zone: Zone
    landing_zone: Zone
    sensor_model: SensorModel
    budgets: Budgets

    def __post_init__(self):
        sectors = _nested_seq(SearchSector, self.search_sectors, 'search_sectors')
        if not sectors:
            raise ContractError('search_sectors: at least one sector required')
        _unique([s.sector_id for s in sectors], 'search_sectors')
        if not math.isclose(sum(s.prior_weight for s in sectors), 1., abs_tol=1e-9):
            raise ContractError('search_sectors: prior weights must sum to 1')
        bounds = _nested(Bounds2D, self.world_bounds, 'world_bounds')
        launch, landing = _nested(Zone, self.launch_zone, 'launch_zone'), _nested(Zone, self.landing_zone, 'landing_zone')
        for sector in sectors:
            if not all(bounds.contains(x, y) for x, y in sector.polygon_enu):
                raise ContractError(f'sector {sector.sector_id}: outside world bounds')
        for zone in (launch, landing):
            if not bounds.contains(*zone.center_enu[:2]):
                raise ContractError(f'zone {zone.zone_id}: outside world bounds')
        self._set(world_bounds=bounds, search_sectors=sectors, launch_zone=launch, landing_zone=landing,
                  sensor_model=_nested(SensorModel, self.sensor_model, 'sensor_model'),
                  budgets=_nested(Budgets, self.budgets, 'budgets'))
        assert_public_payload(self.to_dict())


@dataclass(frozen=True)
class ScenarioSpec(Contract):
    schema_version: str
    scenario_id: str
    split: str
    layout_family: str
    asset_manifest_sha256: str
    backend: str
    backend_build_sha256: str
    settings_sha256: str
    briefing: MissionBriefing
    discount: SemiMarkovDiscount
    timing: TimingPolicy
    rng: RngStreams

    def __post_init__(self):
        self._set(schema_version=_check_schema(self.schema_version), scenario_id=_ident(self.scenario_id, 'scenario_id'),
                  split=_choice(self.split, 'split', SPLITS), layout_family=_ident(self.layout_family, 'layout_family'),
                  asset_manifest_sha256=_sha256(self.asset_manifest_sha256, 'asset_manifest_sha256'),
                  backend=_choice(self.backend, 'backend', BACKENDS),
                  backend_build_sha256=_sha256(self.backend_build_sha256, 'backend_build_sha256'),
                  settings_sha256=_sha256(self.settings_sha256, 'settings_sha256'),
                  briefing=_nested(MissionBriefing, self.briefing, 'briefing'),
                  discount=_nested(SemiMarkovDiscount, self.discount, 'discount'),
                  timing=_nested(TimingPolicy, self.timing, 'timing'),
                  rng=_nested(RngStreams, self.rng, 'rng'))


# --- Evaluator (private) ---------------------------------------------------------------

@dataclass(frozen=True)
class TargetTruth(Contract):
    target_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    asset_id: str

    def __post_init__(self):
        self._set(target_id=_ident(self.target_id, 'target_id'), position_enu=_vec(self.position_enu, 'position_enu'),
                  yaw_rad=_real(self.yaw_rad, 'yaw_rad'), asset_id=_ident(self.asset_id, 'asset_id'))


@dataclass(frozen=True)
class EvaluatorSpec(Contract):
    schema_version: str
    scenario_id: str
    targets: Tuple[TargetTruth, ...]
    match_tolerance_m: float
    semantic_label_ids: Tuple[int, ...]

    def __post_init__(self):
        targets = _nested_seq(TargetTruth, self.targets, 'targets')
        _unique([t.target_id for t in targets], 'targets')
        labels = tuple(_count(x, 'semantic_label_ids') for x in _list(self.semantic_label_ids, 'semantic_label_ids'))
        _unique(labels, 'semantic_label_ids')
        self._set(schema_version=_check_schema(self.schema_version), scenario_id=_ident(self.scenario_id, 'scenario_id'),
                  targets=targets, semantic_label_ids=labels,
                  match_tolerance_m=_real(self.match_tolerance_m, 'match_tolerance_m', low=0, open_low=True))

    @property
    def present(self):
        return bool(self.targets)


def check_scenario_pair(scenario, evaluator):
    if scenario.scenario_id != evaluator.scenario_id:
        raise ContractError('scenario/evaluator IDs differ')
    bounds = scenario.briefing.world_bounds
    for target in evaluator.targets:
        if not bounds.contains(*target.position_enu[:2]):
            raise ContractError(f'target {target.target_id}: outside world bounds')


# --- Policy interface (public) ---------------------------------------------------------

class ActionType(str, Enum):
    SEARCH_SECTOR = 'SEARCH_SECTOR'
    INSPECT_VIEW = 'INSPECT_VIEW'
    REPORT = 'REPORT'
    RETURN = 'RETURN'
    #: Task flood-sar-village-gnss/1 only. A NON-TERMINAL visit to a public anchor: fly, dwell, at most one
    #: measurement of the global georeference error, no image. It is never a synonym for RETURN: the executive
    #: has its own prepare/execute branch for it and RELOCALIZE over the landing pad does not land.
    RELOCALIZE = 'RELOCALIZE'


_ACTION_FIELDS = {
    ActionType.SEARCH_SECTOR: ({'sector_id'}, set()),
    ActionType.INSPECT_VIEW: ({'view_id'}, {'track_id'}),
    ActionType.REPORT: ({'track_id', 'report_position_enu'}, set()),
    ActionType.RETURN: (set(), set()),
    ActionType.RELOCALIZE: ({'anchor_id'}, set()),
}
_ACTION_KEYS = ('sector_id', 'view_id', 'track_id', 'report_position_enu', 'anchor_id')


def action_from_dict(payload):
    """Rebuild an `Action` from its serialized form, refusing exactly what the constructor refuses.

    review 19: the campaign validator re-implemented the payload rules and therefore did NOT enforce
    them - it computed the present keys from a fixed list, so an unknown key was invisible, and it accepted any
    non-empty string where the contract applies an identifier grammar. Two enforcers with one table is still
    two enforcers. There is one now: the validator builds the object, and whatever `Action.__post_init__`
    refuses is refused on disk too.

    An unknown key reaches the constructor as a `TypeError`; this turns it into a `ContractError` so a caller
    has one exception to catch.
    """
    if not isinstance(payload, dict):
        raise ContractError('action: expected an object')
    unknown = sorted(set(payload) - {'action_type', *_ACTION_KEYS})
    if unknown:
        raise ContractError(f'action: unknown {unknown}')
    if 'action_type' not in payload:
        raise ContractError('action: no action_type')
    try:
        return Action(**payload)
    except TypeError as error:                       # unreachable once `unknown` is empty; kept as a backstop
        raise ContractError(f'action: {error}') from None


def action_requirements(action_type):
    """The frozen payload contract for one action type: (required keys, optional keys).

    `Action.__post_init__` enforces this when an action is constructed, and the P3 campaign validator has to
    enforce the same thing on a PERSISTED action long after the object is gone (review 18: a genuine
    executed REPORT could lose its `track_id` and still pass). Two enforcers, one table, read from here so
    they cannot drift apart.
    """
    required, optional = _ACTION_FIELDS[ActionType(action_type)]
    return tuple(sorted(required)), tuple(sorted(optional))


@dataclass(frozen=True)
class Action(Contract):
    """REPORT submits an evidence claim; its correctness is never revealed to the policy.

    report_position_enu is in the LOCAL mission frame. Under flood-sar-village-gnss/1 the executive converts it
    to the global frame through the private georeference service at submission; the policy never sees, and never
    chooses, a global coordinate.
    """
    action_type: ActionType
    sector_id: Optional[str] = None
    view_id: Optional[str] = None
    track_id: Optional[str] = None
    report_position_enu: Optional[Tuple[float, float, float]] = None
    anchor_id: Optional[str] = None

    def __post_init__(self):
        try:
            kind = ActionType(self.action_type)
        except ValueError:
            raise ContractError('action_type: unknown') from None
        required, optional = _ACTION_FIELDS[kind]
        present = {k for k in _ACTION_KEYS if getattr(self, k) is not None}
        if not required <= present or present - required - optional:
            raise ContractError(f'{kind.value}: requires {sorted(required)}, allows {sorted(optional)}, got {sorted(present)}')
        self._set(action_type=kind, sector_id=_optional_ident(self.sector_id, 'sector_id'),
                  view_id=_optional_ident(self.view_id, 'view_id'), track_id=_optional_ident(self.track_id, 'track_id'),
                  anchor_id=_optional_ident(self.anchor_id, 'anchor_id'),
                  report_position_enu=None if self.report_position_enu is None else _vec(self.report_position_enu, 'report_position_enu'))

    def to_dict(self):
        """Legacy actions serialize exactly as before: the GNSS-only key appears only when it is used."""
        data = _plain(self)
        if self.anchor_id is None:
            data.pop('anchor_id', None)
        return data


@dataclass(frozen=True)
class VehicleEstimate(Contract):
    position_enu: Tuple[float, float, float]
    velocity_enu: Tuple[float, float, float]
    attitude_wxyz: Tuple[float, float, float, float]
    source: str

    def __post_init__(self):
        q = _vec(self.attitude_wxyz, 'attitude_wxyz', 4)
        if abs(math.sqrt(sum(x * x for x in q)) - 1) > 1e-3:
            raise ContractError('attitude_wxyz: not unit length')
        # Simulator pose used as an ideal estimate must say so; it is not localization.
        self._set(position_enu=_vec(self.position_enu, 'position_enu'), velocity_enu=_vec(self.velocity_enu, 'velocity_enu'),
                  attitude_wxyz=q, source=_choice(self.source, 'source', ('px4_estimator', 'simulator_pose_declared_ideal')))


@dataclass(frozen=True)
class SensorStatus(Contract):
    stream: str
    valid: bool
    timestamp_s: Optional[float]
    age_s: Optional[float]

    def __post_init__(self):
        self._set(stream=_choice(self.stream, 'stream', ('rgb', 'depth', 'vehicle_state')), valid=_flag(self.valid, 'valid'),
                  timestamp_s=_optional_real(self.timestamp_s, 'timestamp_s', low=0),
                  age_s=_optional_real(self.age_s, 'age_s', low=0))
        if (self.timestamp_s is None) != (self.age_s is None):
            raise ContractError('timestamp_s and age_s must be given together')
        if self.valid and self.timestamp_s is None:
            raise ContractError('valid sensor sample requires timestamp_s and age_s')


@dataclass(frozen=True)
class Track(Contract):
    track_id: str
    position_enu: Tuple[float, float, float]
    covariance_enu: Tuple[Tuple[float, float, float], ...]
    age_s: float
    detections: int
    max_score: Optional[float] = None  # highest frozen-detector confidence of the track's detections (public; ADR 010)

    def __post_init__(self):
        rows = _list(self.covariance_enu, 'covariance_enu')
        cov = tuple(_vec(r, f'covariance_enu[{i}]') for i, r in enumerate(rows))
        if len(cov) != 3:
            raise ContractError('covariance_enu: expected 3x3')
        c = np.asarray(cov)
        if not np.allclose(c, c.T, atol=1e-9) or np.linalg.eigvalsh(c).min() < -1e-9:
            raise ContractError('covariance_enu: must be symmetric positive semidefinite')
        self._set(track_id=_ident(self.track_id, 'track_id'), position_enu=_vec(self.position_enu, 'position_enu'),
                  covariance_enu=cov, age_s=_real(self.age_s, 'age_s', low=0),
                  max_score=_optional_real(self.max_score, 'max_score', low=0, high=1))
        _count(self.detections, 'detections', minimum=1)


@dataclass(frozen=True)
class ViewOption(Contract):
    view_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    path_length_m: float
    estimated_duration_s: float
    track_id: Optional[str] = None

    def __post_init__(self):
        self._set(view_id=_ident(self.view_id, 'view_id'), position_enu=_vec(self.position_enu, 'position_enu'),
                  yaw_rad=_real(self.yaw_rad, 'yaw_rad'), path_length_m=_real(self.path_length_m, 'path_length_m', low=0),
                  estimated_duration_s=_real(self.estimated_duration_s, 'estimated_duration_s', low=0),
                  track_id=_optional_ident(self.track_id, 'track_id'))


@dataclass(frozen=True)
class SectorCoverage(Contract):
    sector_id: str
    covered_fraction: float

    def __post_init__(self):
        self._set(sector_id=_ident(self.sector_id, 'sector_id'),
                  covered_fraction=_real(self.covered_fraction, 'covered_fraction', low=0, high=1))


@dataclass(frozen=True)
class SubmittedReport(Contract):
    """Task flood-sar-mission/1. `position_enu` IS the submitted coordinate and is what the evaluator scores."""
    track_id: str
    position_enu: Tuple[float, float, float]
    sim_time_s: float

    def __post_init__(self):
        self._set(track_id=_ident(self.track_id, 'track_id'), position_enu=_vec(self.position_enu, 'position_enu'),
                  sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0))


@dataclass(frozen=True)
class LocalReportReceipt(Contract):
    """Task flood-sar-village-gnss/1: the PUBLIC receipt of a transmission. It is not a scorable coordinate.

    The field is called `local_position_enu` and there is deliberately no `position_enu` attribute, so an
    evaluator or reward path cannot read a scorable coordinate off it by habit. The transmitted GLOBAL
    coordinate is frozen in the private `georeference.GlobalSubmission` ledger and nowhere else: publishing it
    beside the local one would hand the policy `d - dhat` by subtraction.

    `evaluation.match_reports` and `training.reward_gnss` refuse this type outright (review 1, ruling R1).
    """
    track_id: str
    local_position_enu: Tuple[float, float, float]
    sim_time_s: float
    frame: str = 'L'

    def __post_init__(self):
        if self.frame != 'L':
            raise ContractError("LocalReportReceipt.frame: must be 'L'; a receipt is never a global coordinate")
        self._set(track_id=_ident(self.track_id, 'track_id'),
                  local_position_enu=_vec(self.local_position_enu, 'local_position_enu'),
                  sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0))


def dedup_position_xy(receipt):
    """XY of a public report record, for merge-radius DEDUPLICATION ONLY. Never a scoring coordinate.

    Deduplication is a local-frame association question (model contract section 3), so both public record types
    answer it; scoring is not, and only `SubmittedReport` can be scored.
    """
    if isinstance(receipt, LocalReportReceipt):
        return receipt.local_position_enu[:2]
    if isinstance(receipt, SubmittedReport):
        return receipt.position_enu[:2]
    raise ContractError(f'dedup_position_xy: expected SubmittedReport or LocalReportReceipt, got {type(receipt).__name__}')


@dataclass(frozen=True)
class PublicObservation(Contract):
    schema_version: str
    sim_time_s: float
    elapsed_s: float
    vehicle: VehicleEstimate
    sensors: Tuple[SensorStatus, ...]
    coverage: Tuple[SectorCoverage, ...]
    tracks: Tuple[Track, ...]
    views: Tuple[ViewOption, ...]
    remaining_time_s: float
    return_estimate_s: float
    reports: Tuple[SubmittedReport, ...]
    #: flood-sar-village-gnss/1 only; empty under flood-sar-mission/1 and then absent from to_dict().
    #: georeference: georeference_contracts.GeoreferenceSummary - the filter's own belief, never the true error.
    #: anchors: tuple of relocalization.AnchorStatus - public anchor geometry, availability and route cost.
    #: Both are typed Any here only to keep contracts.py free of an import cycle; __post_init__ validates them.
    georeference: Optional[Any] = None
    anchors: Tuple[Any, ...] = ()

    def __post_init__(self):
        sensors = _nested_seq(SensorStatus, self.sensors, 'sensors')
        coverage = _nested_seq(SectorCoverage, self.coverage, 'coverage')
        tracks, views = _nested_seq(Track, self.tracks, 'tracks'), _nested_seq(ViewOption, self.views, 'views')
        _unique([c.sector_id for c in coverage], 'coverage')
        _unique([t.track_id for t in tracks], 'tracks')
        _unique([v.view_id for v in views], 'views')
        known = {t.track_id for t in tracks}
        if any(v.track_id is not None and v.track_id not in known for v in views):
            raise ContractError('views: reference unknown track')
        self._set(schema_version=_check_schema(self.schema_version, SCHEMA_VERSIONS),
                  sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0),
                  elapsed_s=_real(self.elapsed_s, 'elapsed_s', low=0), vehicle=_nested(VehicleEstimate, self.vehicle, 'vehicle'),
                  sensors=sensors, coverage=coverage, tracks=tracks, views=views,
                  remaining_time_s=_real(self.remaining_time_s, 'remaining_time_s', low=0),
                  return_estimate_s=_real(self.return_estimate_s, 'return_estimate_s', low=0),
                  reports=_nested_seq(LocalReportReceipt if self.schema_version == GNSS_SCHEMA_VERSION else SubmittedReport,
                                      self.reports, 'reports'))
        self._set(**self._gnss_fields())
        if self.elapsed_s > self.sim_time_s + TIME_TOLERANCE_S:
            raise ContractError('elapsed_s exceeds sim_time_s')
        check_sensor_times(self.sim_time_s, sensors)
        if any(t.age_s > self.elapsed_s + TIME_TOLERANCE_S for t in tracks):
            raise ContractError('track older than the episode')
        start, times = self.sim_time_s - self.elapsed_s, [r.sim_time_s for r in self.reports]
        if times != sorted(times) or any(t < start - TIME_TOLERANCE_S or t > self.sim_time_s + TIME_TOLERANCE_S for t in times):
            raise ContractError('reports must be time-ordered within the episode')
        assert_public_payload(self.to_dict())

    def _gnss_fields(self):
        """Validate the task-versioned fields. Imported late so contracts.py stays free of an import cycle."""
        if self.schema_version != GNSS_SCHEMA_VERSION:
            if self.georeference is not None or tuple(self.anchors):
                raise ContractError(f'georeference/anchors belong to schema_version {GNSS_SCHEMA_VERSION}')
            return {'georeference': None, 'anchors': ()}
        from .georeference_contracts import GeoreferenceSummary
        from .relocalization import AnchorStatus
        if self.georeference is None:
            raise ContractError(f'schema_version {GNSS_SCHEMA_VERSION} requires a georeference summary')
        anchors = _nested_seq(AnchorStatus, self.anchors, 'anchors')
        _unique([a.anchor_id for a in anchors], 'anchors')
        return {'georeference': _nested(GeoreferenceSummary, self.georeference, 'georeference'), 'anchors': anchors}

    def to_dict(self):
        """A legacy observation serializes byte-for-byte as before: the two GNSS keys appear only for the GNSS task."""
        data = _plain(self)
        if self.schema_version != GNSS_SCHEMA_VERSION:
            data.pop('georeference', None)
            data.pop('anchors', None)
        return data


@dataclass(frozen=True)
class Transition(Contract):
    """One semi-Markov decision. Invalid (infrastructure) transitions carry no reward.

    The mission deadline is a terminal event with zero bootstrap, not a truncation.
    """
    observation: PublicObservation
    action: Action
    next_observation: Optional[PublicObservation]
    reward: Optional[float]
    duration_s: Optional[float]
    discount: Optional[float]
    terminated: bool
    truncated: bool
    reason: Optional[str]
    valid: bool

    def __post_init__(self):
        self._set(observation=_nested(PublicObservation, self.observation, 'observation'),
                  action=_nested(Action, self.action, 'action'),
                  next_observation=None if self.next_observation is None else _nested(PublicObservation, self.next_observation, 'next_observation'),
                  terminated=_flag(self.terminated, 'terminated'), truncated=_flag(self.truncated, 'truncated'),
                  valid=_flag(self.valid, 'valid'))
        if not self.valid:
            if self.reason != INFRASTRUCTURE or self.terminated or self.truncated or self.reward is not None or self.discount is not None:
                raise ContractError('invalid transition: infrastructure_failure only, no reward/discount/termination')
            self._set(duration_s=_optional_real(self.duration_s, 'duration_s', low=0))
            return
        if self.next_observation is None:
            raise ContractError('valid transition requires next_observation')
        self._set(reward=_real(self.reward, 'reward'), duration_s=_real(self.duration_s, 'duration_s', low=0, open_low=True),
                  discount=_real(self.discount, 'discount', low=0, high=1, open_low=True))
        before, after = self.observation, self.next_observation
        if not math.isclose(after.sim_time_s - before.sim_time_s, self.duration_s, abs_tol=TIME_TOLERANCE_S):
            raise ContractError('duration_s must equal the measured sim-time difference')
        if not math.isclose(after.elapsed_s - before.elapsed_s, self.duration_s, abs_tol=TIME_TOLERANCE_S):
            raise ContractError('elapsed_s must advance by duration_s (same episode, same clock)')
        if after.reports[:len(before.reports)] != before.reports:
            raise ContractError('submitted-report history is append-only')
        if self.terminated and self.truncated:
            raise ContractError('terminated and truncated are exclusive')
        if self.terminated:
            _choice(self.reason, 'reason', TERMINAL_REASONS)
        elif self.truncated:
            _choice(self.reason, 'reason', TRUNCATION_REASONS)
        elif self.reason is not None:
            raise ContractError('reason only for terminated/truncated/invalid transitions')

    def bootstrap_weight(self):
        """Weight of V(next) in the target; zero at terminal events, undefined when invalid."""
        if not self.valid:
            raise ContractError('invalid transition cannot be bootstrapped')
        return 0. if self.terminated else self.discount


# --- Episode outcome (evaluator side) --------------------------------------------------

@dataclass(frozen=True)
class EpisodeResult(Contract):
    schema_version: str
    episode_id: str
    run_state: str
    infrastructure_failure: bool
    termination_reason: Optional[str]
    targets_present: int
    correct_distinct_reports: int
    false_reports: int
    duplicate_reports: int
    unresolved_targets: int
    returned: bool
    landed: bool
    collisions: Optional[int]
    min_clearance_m: Optional[float]
    interventions: int
    flight_distance_m: float
    sim_time_s: float
    wall_time_s: float
    sensor_loss_events: int
    decision_latency_p95_s: Optional[float]
    energy_proxy: Optional[float]
    energy_model: Optional[str]

    def __post_init__(self):
        counts = ('targets_present', 'correct_distinct_reports', 'false_reports', 'duplicate_reports',
                  'unresolved_targets', 'interventions', 'sensor_loss_events')
        for key in counts:
            _count(getattr(self, key), key)
        if self.collisions is not None:  # None = not measured (Gazebo backend); never written as 0
            _count(self.collisions, 'collisions')
        for key in ('infrastructure_failure', 'returned', 'landed'):
            _flag(getattr(self, key), key)
        self._set(schema_version=_check_schema(self.schema_version), episode_id=_ident(self.episode_id, 'episode_id'),
                  run_state=_choice(self.run_state, 'run_state', ('planned', 'started', 'completed', 'interrupted')),
                  min_clearance_m=_optional_real(self.min_clearance_m, 'min_clearance_m', low=0),
                  flight_distance_m=_real(self.flight_distance_m, 'flight_distance_m', low=0),
                  sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0), wall_time_s=_real(self.wall_time_s, 'wall_time_s', low=0),
                  decision_latency_p95_s=_optional_real(self.decision_latency_p95_s, 'decision_latency_p95_s', low=0),
                  energy_proxy=_optional_real(self.energy_proxy, 'energy_proxy', low=0))
        if (self.energy_proxy is None) != (self.energy_model is None):
            raise ContractError('energy_proxy requires a named calibrated energy_model (and vice versa)')
        if self.correct_distinct_reports + self.unresolved_targets != self.targets_present:
            raise ContractError('correct_distinct_reports + unresolved_targets must equal targets_present')
        if self.infrastructure_failure != (self.termination_reason == INFRASTRUCTURE):
            raise ContractError('infrastructure_failure must match termination_reason')
        if self.infrastructure_failure and self.run_state == 'completed':
            raise ContractError('infrastructure failure cannot be a completed episode')
        if self.termination_reason is not None:
            _choice(self.termination_reason, 'termination_reason', TERMINAL_REASONS + (INFRASTRUCTURE,))
        if self.run_state == 'completed' and self.termination_reason is None:
            raise ContractError('completed episode requires termination_reason')
        if self.run_state == 'planned':
            activity = [getattr(self, k) for k in counts if k not in ('targets_present', 'unresolved_targets')]
            if any(activity) or self.returned or self.landed or self.flight_distance_m or self.sim_time_s or self.termination_reason:
                raise ContractError('planned episode cannot report activity')

    def recall(self):
        """None for target-absent episodes: undefined recall is never reported as 100%."""
        return None if self.targets_present == 0 else self.correct_distinct_reports / self.targets_present

    def any_false_report(self):
        return self.false_reports > 0
