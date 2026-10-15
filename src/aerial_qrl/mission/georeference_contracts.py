"""Typed state and serialized metadata for georeferencing under intermittent global-position aid.

Task
    flood-sar-village-gnss/1 (docs/study/gnss_qrl_v1/MODEL_CONTRACT_EN.md). The scene, the local route graph, the
    local vehicle state and the cached RGB/depth observations stay exactly those of the validated fast tier. The
    new stochastic variable is the horizontal calibration error of the GLOBAL georeferencing service. It changes
    the global coordinates written into a submitted report. It never changes a bank lookup pose, an image, the
    flown route or the local track estimate.

Frames
    L   stable local mission-map frame: every public quantity (vehicle pose, tracks, anchors, routes) lives here.
    G   evaluator global ENU frame: only submitted report coordinates and evaluator targets live here.
    T0: L -> G is a private constant transform. This v1 models it as a pure translation (its gauge), so the
    evaluator targets and every report translate together and the matching is invariant to the gauge. The gauge
    is never an actor input and never leaves the private service.

Public and private, stated once
    PUBLIC (may reach actor, critic, planner, masks): GeoreferenceSummary below, built only from the filter's own
    mean-free quantities (covariance, fix age, fix counts, last-fix acceptance).
    PRIVATE (evaluator side only): the true bias d_t, the gauge of T0, the measurement noise draw, the residual
    (d_t - dhat_t) and therefore the instantaneous report error. `GeoreferenceSummary` carries none of them, and
    its key names pass mission.leakage.assert_public_payload.

Units
    Metres and simulator seconds throughout. Covariances are m^2. The process is two-dimensional (ENU east,
    north); the evaluator's tolerance stays 3-D, so the vertical error is unchanged by this model.

Provenance of the numbers
    Every scientific quantity here is a REQUIRED field: this module has no silent default for D600, for the
    initial bias, correction and covariance, for the measurement noise or for the fix cost. A configuration must
    state them, and `GeoreferenceConfig.provenance` records whether they are frozen or an unfrozen development
    proposal. freeze_candidate.json lists these as pending pre-lock fields; nothing here asserts they are locked.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .contracts import PROVENANCE, Contract, ContractError, _choice, _count, _flag, _ident, _optional_real, _real

GEOREFERENCE_VERSION = 'georeference/1'
PROCESS_MODELS = ('brownian_translation_2d/1',)
#: How a query time between two exactly represented path points is served (see georeference.BrownianPath).
INTERPOLATIONS = ('dyadic_bridge_then_linear/1',)


def _cov2(value, name):
    """A 2x2 symmetric positive-semidefinite matrix as a tuple of tuples."""
    rows = tuple(tuple(float(v) for v in row) for row in value)
    if len(rows) != 2 or any(len(r) != 2 for r in rows):
        raise ContractError(f'{name}: expected a 2x2 matrix')
    matrix = np.asarray(rows, float)
    if not np.isfinite(matrix).all():
        raise ContractError(f'{name}: non-finite entry')
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ContractError(f'{name}: must be symmetric')
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-12:
        raise ContractError(f'{name}: must be positive semidefinite')
    return rows


def _vec2(value, name):
    values = tuple(float(v) for v in value)
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ContractError(f'{name}: expected 2 finite numbers (east, north)')
    return values


@dataclass(frozen=True)
class GeoreferenceConfig(Contract):
    """Versioned parameters of the private georeferencing error and of the common filter.

    d600_rms_m
        D600: the RMS of the horizontal displacement of the raw service error 600 s after the last measurement,
        with no measurement in between. The per-axis diffusion is q = D600^2 / (2 * 600) m^2/s, so that
        E[|d(t0+600) - d(t0)|^2] = 2 * q * 600 = D600^2. D600 = 0 disables the perturbation exactly.
    initial_bias_enu, initial_correction_enu, initial_covariance_m2
        d0, dhat0 and P0 at episode start. The contract requires these to be explicit; they are pre-lock fields.
    measurement_sigma_m
        Per-axis standard deviation of the anchor measurement noise v; R = measurement_sigma_m^2 * I_2.
    path_horizon_s, path_resolution_s
        The private path is built on [0, path_horizon_s] and is exact at dyadic points; between the finest
        points of width <= path_resolution_s it is interpolated by the declared rule. Both are discretization
        choices, not physics: make path_horizon_s longer than any episode, and path_resolution_s small against
        the shortest macro.
    """
    version: str
    d600_rms_m: float
    initial_bias_enu: Tuple[float, float]
    initial_correction_enu: Tuple[float, float]
    initial_covariance_m2: Tuple[Tuple[float, float], Tuple[float, float]]
    measurement_sigma_m: float
    path_horizon_s: float
    path_resolution_s: float
    provenance: str
    process_model: str = PROCESS_MODELS[0]
    interpolation: str = INTERPOLATIONS[0]

    def __post_init__(self):
        if self.version != GEOREFERENCE_VERSION:
            raise ContractError(f'version: expected {GEOREFERENCE_VERSION}')
        self._set(
            d600_rms_m=_real(self.d600_rms_m, 'd600_rms_m', low=0),
            initial_bias_enu=_vec2(self.initial_bias_enu, 'initial_bias_enu'),
            initial_correction_enu=_vec2(self.initial_correction_enu, 'initial_correction_enu'),
            initial_covariance_m2=_cov2(self.initial_covariance_m2, 'initial_covariance_m2'),
            measurement_sigma_m=_real(self.measurement_sigma_m, 'measurement_sigma_m', low=0),
            path_horizon_s=_real(self.path_horizon_s, 'path_horizon_s', low=0, open_low=True),
            path_resolution_s=_real(self.path_resolution_s, 'path_resolution_s', low=0, open_low=True),
            provenance=_choice(self.provenance, 'provenance', PROVENANCE),
            process_model=_choice(self.process_model, 'process_model', PROCESS_MODELS),
            interpolation=_choice(self.interpolation, 'interpolation', INTERPOLATIONS))
        if self.path_resolution_s > self.path_horizon_s:
            raise ContractError('path_resolution_s cannot exceed path_horizon_s')

    @property
    def q_m2_per_s(self):
        """Per-axis diffusion coefficient q = D600^2 / (2 * 600); zero when the perturbation is disabled."""
        return self.d600_rms_m ** 2 / 1200.

    @property
    def measurement_covariance_m2(self):
        r = self.measurement_sigma_m ** 2
        return ((r, 0.), (0., r))

    @property
    def enabled(self):
        """False when D600 = 0 AND the initial bias is zero: the georeferencing error is then identically zero."""
        return self.d600_rms_m > 0 or any(v != 0. for v in self.initial_bias_enu)


@dataclass(frozen=True)
class FilterState(Contract):
    """The common estimator's public-side state: correction mean, covariance and fix history.

    `correction_enu` is dhat: the filter's estimate of the raw service error. It is NOT public on its own,
    because with the true bias it would give the residual; only `GeoreferenceSummary` is published.
    """
    correction_enu: Tuple[float, float]
    covariance_m2: Tuple[Tuple[float, float], Tuple[float, float]]
    updated_at_s: float
    fixes_attempted: int
    fixes_accepted: int
    last_fix_at_s: Optional[float] = None
    last_fix_accepted: Optional[bool] = None

    def __post_init__(self):
        _count(self.fixes_attempted, 'fixes_attempted')
        _count(self.fixes_accepted, 'fixes_accepted')
        self._set(correction_enu=_vec2(self.correction_enu, 'correction_enu'),
                  covariance_m2=_cov2(self.covariance_m2, 'covariance_m2'),
                  updated_at_s=_real(self.updated_at_s, 'updated_at_s', low=0),
                  last_fix_at_s=_optional_real(self.last_fix_at_s, 'last_fix_at_s', low=0),
                  last_fix_accepted=None if self.last_fix_accepted is None else _flag(self.last_fix_accepted, 'last_fix_accepted'))
        if self.fixes_accepted > self.fixes_attempted:
            raise ContractError('fixes_accepted cannot exceed fixes_attempted')


@dataclass(frozen=True)
class GeoreferenceSummary(Contract):
    """PUBLIC belief summary: the only georeferencing quantity an actor, critic, planner or mask may read.

    Every field is a function of the filter's covariance and of the fix bookkeeping. None is a function of the
    true bias, of the gauge or of the realised report error, so this object cannot carry the answer.

    rms_horizontal_error_m
        sqrt(trace(P)): the RMS magnitude the filter expects for the two-dimensional residual it has not
        corrected. It is the quantity to compare against the evaluator's 3.0 m tolerance, and it is exactly what
        a mission system could compute for itself.
    time_since_fix_s
        Simulator seconds since the last ACCEPTED measurement, or since episode start when none was accepted.
    """
    schema_version: str
    rms_horizontal_error_m: float
    covariance_trace_m2: float
    time_since_fix_s: float
    fixes_attempted: int
    fixes_accepted: int
    last_fix_accepted: Optional[bool] = None

    def __post_init__(self):
        _count(self.fixes_attempted, 'fixes_attempted')
        _count(self.fixes_accepted, 'fixes_accepted')
        if self.schema_version != GEOREFERENCE_VERSION:
            raise ContractError(f'schema_version: expected {GEOREFERENCE_VERSION}')
        self._set(rms_horizontal_error_m=_real(self.rms_horizontal_error_m, 'rms_horizontal_error_m', low=0),
                  covariance_trace_m2=_real(self.covariance_trace_m2, 'covariance_trace_m2', low=0),
                  time_since_fix_s=_real(self.time_since_fix_s, 'time_since_fix_s', low=0),
                  last_fix_accepted=None if self.last_fix_accepted is None else _flag(self.last_fix_accepted, 'last_fix_accepted'))
        if not math.isclose(self.rms_horizontal_error_m ** 2, self.covariance_trace_m2, rel_tol=1e-9, abs_tol=1e-12):
            raise ContractError('rms_horizontal_error_m must be the square root of covariance_trace_m2')


@dataclass(frozen=True)
class MeasurementEvent(Contract):
    """PRIVATE record of one attempted fix. Written to the evaluator stream, never to a public payload.

    `innovation_enu` is z - dhat^-: public in principle (the filter computes it), but this v1 keeps it on the
    private stream because together with a published correction it would narrow the true bias.
    """
    sim_time_s: float
    anchor_id: str
    available: bool
    accepted: bool
    innovation_enu: Optional[Tuple[float, float]] = None
    residual_before_m: Optional[float] = None    # |d - dhat| before the update: diagnosis only
    residual_after_m: Optional[float] = None

    def __post_init__(self):
        self._set(sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0), anchor_id=_ident(self.anchor_id, 'anchor_id'),
                  available=_flag(self.available, 'available'), accepted=_flag(self.accepted, 'accepted'),
                  innovation_enu=None if self.innovation_enu is None else _vec2(self.innovation_enu, 'innovation_enu'),
                  residual_before_m=_optional_real(self.residual_before_m, 'residual_before_m', low=0),
                  residual_after_m=_optional_real(self.residual_after_m, 'residual_after_m', low=0))
        if self.accepted and not self.available:
            raise ContractError('an unavailable anchor cannot produce an accepted measurement')
        if self.accepted and self.innovation_enu is None:
            raise ContractError('an accepted measurement must record its innovation')


__all__ = ['GEOREFERENCE_VERSION', 'INTERPOLATIONS', 'PROCESS_MODELS', 'FilterState', 'GeoreferenceConfig',
           'GeoreferenceSummary', 'MeasurementEvent']
