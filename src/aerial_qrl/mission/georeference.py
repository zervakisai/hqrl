"""Private georeferencing process, common filter and the service that joins them (task flood-sar-village-gnss/1).

Read `georeference_contracts.py` first for the frames, the public/private boundary and the units.

The path is a function of absolute time, not a stream of increments
    A shared seed does NOT give two policies the same disturbance once their action durations differ: a
    sequentially consumed generator hands out its draws in query order, so a policy that takes more or shorter
    macros walks a different path. `BrownianPath` therefore defines one realisation W: [0, H] -> R^2 as a
    deterministic function of absolute simulator time, built by the Levy dyadic construction:

        W(0) = 0,  W(H) ~ N(0, q H I_2),
        for an interval [a, b] with both endpoints already fixed, the midpoint m = (a+b)/2 is
        W(m) = (W(a) + W(b))/2 + sqrt(q (b-a) / 4) Z,  Z ~ N(0, I_2),

    where every Z is drawn from a counter-based key (path_key, level, index) rather than from a sequence. The
    law of W at dyadic points is exactly that of the two-dimensional Brownian motion with per-axis diffusion q.
    Two consequences are the reason for this construction, and both are tested:
      * partition invariance - W(t) does not depend on which times were queried before it, in what order, how
        many times, or how the macros happened to split the interval. Adding a diagnostic read changes nothing.
      * restart invariance - the path is stateless given (path_key, config), so a checkpoint/restart reproduces
        it without carrying generator state.
    Below the finest dyadic width (config.path_resolution_s) the value is linearly interpolated between the two
    surrounding exact points. That is a declared discretization: the interpolation error has standard deviation
    at most sqrt(q * path_resolution_s / 4), which the configuration must keep negligible against the mission's
    3.0 m tolerance. `BrownianPath.interpolation_sigma_m` reports it so a protocol can record the number.

Counter-based draws
    _standard_normal_pair(key) = Box-Muller of two 64-bit words of blake2b(key, digest_size=32). Deterministic
    across platforms and interpreter runs, independent of NumPy's generator stream, and cheap enough to call
    once per dyadic node. The key is b'<path_key>|<level>|<index>'.

Covariance is a function of time, not an accumulator
    `GeoreferenceFilter` stores (dhat, P, t) at the last measurement only. The covariance at time t is
    P_last + q (t - t_last) I. There is no `advance()` to call, so "the drift advanced twice for one macro" is
    not a state this code can reach; the model-contract invariant about advancing time exactly once holds by
    construction rather than by discipline. Queries never mutate anything: only `attempt_fix` does.

What touches truth
    Only `GeoreferenceService`: it reads the true bias to build a noisy measurement and to compute the global
    coordinate of a report. `public_summary()` is built from the filter alone. There is no method that returns
    the true bias, the gauge or the residual to a caller that is not the evaluator stream; the private
    diagnostics on `MeasurementEvent` are written to evaluator files only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Optional, Tuple

import numpy as np

from .contracts import ContractError, TIME_TOLERANCE_S, _ident, _real, _vec
from .georeference_contracts import (GEOREFERENCE_VERSION, FilterState, GeoreferenceConfig, GeoreferenceSummary,
                                     MeasurementEvent)

import json  # noqa: E402  (used by GeoreferenceService.disturbance_identity)

_TWO_PI = 2. * math.pi
_UINT64 = float(1 << 64)


def _standard_normal_pair(key):
    """Two independent standard normals from a counter key, by Box-Muller on blake2b words (module docstring)."""
    digest = hashlib.blake2b(key, digest_size=32).digest()
    u1 = (int.from_bytes(digest[0:8], 'big') + 0.5) / _UINT64
    u2 = (int.from_bytes(digest[8:16], 'big') + 0.5) / _UINT64
    radius = math.sqrt(-2. * math.log(u1))
    angle = _TWO_PI * u2
    return radius * math.cos(angle), radius * math.sin(angle)


class BrownianPath:
    """One realisation of a 2-D Brownian translation on [0, horizon_s], queryable at any absolute time.

    `at(t)` returns the displacement W(t) (metres, east/north) with W(0) = 0. Deterministic in
    (path_key, q, horizon_s, resolution_s); see the module docstring for the construction and its guarantees.
    """

    def __init__(self, path_key, q_m2_per_s, horizon_s, resolution_s):
        self.path_key = _ident(path_key, 'path_key')
        self.q_m2_per_s = _real(q_m2_per_s, 'q_m2_per_s', low=0)
        self.horizon_s = _real(horizon_s, 'horizon_s', low=0, open_low=True)
        self.resolution_s = _real(resolution_s, 'resolution_s', low=0, open_low=True)
        self.depth = 0 if self.resolution_s >= self.horizon_s else int(math.ceil(math.log2(self.horizon_s / self.resolution_s)))
        self._key_prefix = self.path_key.encode()
        self._nodes = {(0, 0): (0., 0.)}
        self._nodes[(0, 1)] = self._endpoint()

    @property
    def finest_width_s(self):
        return self.horizon_s / float(1 << self.depth)

    @property
    def interpolation_sigma_m(self):
        """Per-axis standard deviation of the linear interpolation error inside the finest dyadic interval."""
        return math.sqrt(self.q_m2_per_s * self.finest_width_s / 4.)

    def _draw(self, level, index):
        return _standard_normal_pair(b'%s|%d|%d' % (self._key_prefix, level, index))

    def _endpoint(self):
        scale = math.sqrt(self.q_m2_per_s * self.horizon_s)
        east, north = self._draw(0, 1)
        return (scale * east, scale * north)

    def _value(self, level, index, low, high, width):
        """W at time index * horizon / 2^level, given its parent endpoint values and the parent width."""
        cached = self._nodes.get((level, index))
        if cached is not None:
            return cached
        scale = math.sqrt(self.q_m2_per_s * width / 4.)
        east, north = self._draw(level, index)
        value = ((low[0] + high[0]) / 2. + scale * east, (low[1] + high[1]) / 2. + scale * north)
        self._nodes[(level, index)] = value
        return value

    def at(self, sim_time_s):
        t = _real(sim_time_s, 'sim_time_s', low=0)
        if t > self.horizon_s + TIME_TOLERANCE_S:
            raise ContractError(f'sim_time_s {t} exceeds the path horizon {self.horizon_s}; lengthen path_horizon_s')
        t = min(t, self.horizon_s)
        if self.q_m2_per_s == 0.:
            return (0., 0.)
        level, index, width = 0, 0, self.horizon_s
        low, high = self._nodes[(0, 0)], self._nodes[(0, 1)]
        while level < self.depth:
            middle = self._value(level + 1, 2 * index + 1, low, high, width)
            width /= 2.
            level += 1
            if t < (index * 2 + 1) * width:
                index, high = 2 * index, middle
            else:
                index, low = 2 * index + 1, middle
        start = index * width
        alpha = 0. if width <= 0 else min(1., max(0., (t - start) / width))
        return (low[0] + alpha * (high[0] - low[0]), low[1] + alpha * (high[1] - low[1]))


class GeoreferenceFilter:
    """Common two-dimensional estimator of the raw service error: prediction by diffusion, Joseph-form update.

    State is (correction mean dhat, covariance P, the time they are valid at) plus fix bookkeeping. Prediction
    is a pure function of the query time, so it cannot be applied twice for one macro.
    """

    def __init__(self, config, episode_start_s=0.):
        if not isinstance(config, GeoreferenceConfig):
            raise ContractError('GeoreferenceConfig required')
        self.config = config
        self._mean = np.asarray(config.initial_correction_enu, float)
        self._covariance = np.asarray(config.initial_covariance_m2, float)
        self.episode_start_s = _real(episode_start_s, 'episode_start_s', low=0)
        self._valid_at_s = self.episode_start_s
        self._last_fix_at_s = None
        self._last_fix_accepted = None
        self._attempted = 0
        self._accepted = 0

    def begin_episode(self, sim_time_s):
        """Anchor the filter to the episode's first simulator timestamp.

        Backends do not all start an episode at clock zero, so d0/dhat0/P0 must be attached to the episode's own
        origin. Without this, an episode starting at t0 > 0 would begin with a covariance already grown by q*t0
        and with a bias sampled from the middle of the path.
        """
        t0 = _real(sim_time_s, 'sim_time_s', low=0)
        if self._attempted:
            raise ContractError('begin_episode after a measurement: build a new filter instead')
        self.episode_start_s = t0
        self._valid_at_s = t0

    def correction_at(self, sim_time_s):
        """dhat(t): the prediction step does not move the mean, so this is the mean since the last update."""
        self._check_time(sim_time_s)
        return (float(self._mean[0]), float(self._mean[1]))

    def covariance_at(self, sim_time_s):
        """P(t) = P_last + q (t - t_last) I_2."""
        t = self._check_time(sim_time_s)
        grown = self._covariance + np.eye(2) * (self.config.q_m2_per_s * (t - self._valid_at_s))
        return (grown + grown.T) / 2.

    def _check_time(self, sim_time_s):
        t = _real(sim_time_s, 'sim_time_s', low=0)
        if t < self._valid_at_s - TIME_TOLERANCE_S:
            raise ContractError(f'query time {t} precedes the last filter update at {self._valid_at_s}')
        return max(t, self._valid_at_s)

    def state_at(self, sim_time_s):
        covariance = self.covariance_at(sim_time_s)
        return FilterState(self.correction_at(sim_time_s), tuple(tuple(float(v) for v in row) for row in covariance),
                           _real(sim_time_s, 'sim_time_s', low=0), self._attempted, self._accepted,
                           self._last_fix_at_s, self._last_fix_accepted)

    def summary_at(self, sim_time_s):
        """PUBLIC summary. time_since_fix_s counts from the last ACCEPTED fix, else from episode start."""
        t = self._check_time(sim_time_s)
        trace = float(np.trace(self.covariance_at(t)))
        # With no accepted fix the age is measured from the EPISODE's origin, not from the backend clock's zero.
        since = t - self.episode_start_s if self._last_accepted_at_s is None else t - self._last_accepted_at_s
        return GeoreferenceSummary(GEOREFERENCE_VERSION, math.sqrt(max(0., trace)), max(0., trace), max(0., since),
                                   self._attempted, self._accepted, self._last_fix_accepted)

    @property
    def _last_accepted_at_s(self):
        return self._last_fix_at_s if self._last_fix_accepted else None

    def record_unavailable(self, sim_time_s):
        """A fix attempt that produced no measurement: counted, time already paid by the caller, no update."""
        t = self._check_time(sim_time_s)
        self._covariance = self.covariance_at(t)
        self._valid_at_s = t
        self._attempted += 1
        self._last_fix_at_s, self._last_fix_accepted = t, False

    def update(self, sim_time_s, measurement_enu, measurement_covariance_m2):
        """Joseph-form update with z = measurement_enu; returns the innovation z - dhat^-."""
        t = self._check_time(sim_time_s)
        prior = self.covariance_at(t)
        z = np.asarray(measurement_enu, float).reshape(2)
        r = np.asarray(measurement_covariance_m2, float).reshape(2, 2)
        innovation = z - self._mean
        gain = prior @ np.linalg.inv(prior + r)
        identity_minus_gain = np.eye(2) - gain
        posterior = identity_minus_gain @ prior @ identity_minus_gain.T + gain @ r @ gain.T
        self._mean = self._mean + gain @ innovation
        self._covariance = (posterior + posterior.T) / 2.
        self._valid_at_s = t
        self._attempted += 1
        self._accepted += 1
        self._last_fix_at_s, self._last_fix_accepted = t, True
        return (float(innovation[0]), float(innovation[1])), gain


@dataclass(frozen=True)
class GlobalSubmission:
    """PRIVATE ledger entry of one transmitted report: the global coordinate actually sent, frozen on submission.

    A later fix never rewrites one of these. The public observation carries only the track identity, the local
    coordinate and the submission time (see the executive's task docstring), so the actor can never difference
    two global coordinates against their local ones and read the residual off.
    """
    report_index: int
    track_id: str
    sim_time_s: float
    local_position_enu: Tuple[float, float, float]
    global_position_enu: Tuple[float, float, float]
    correction_enu: Tuple[float, float]
    bias_enu: Tuple[float, float]
    residual_enu: Tuple[float, float]

    @property
    def residual_norm_m(self):
        return math.hypot(*self.residual_enu)

    def to_dict(self):
        return {'report_index': self.report_index, 'track_id': self.track_id, 'sim_time_s': self.sim_time_s,
                'local_position_enu': list(self.local_position_enu), 'global_position_enu': list(self.global_position_enu),
                'correction_enu': list(self.correction_enu), 'bias_enu': list(self.bias_enu),
                'residual_enu': list(self.residual_enu), 'residual_norm_m': self.residual_norm_m}


class GeoreferenceService:
    """Joins the private path, the private gauge and the common filter. The only object that reads the truth.

    gauge_enu is the translation of the private transform T0: L -> G. Translating it, together with the
    evaluator targets, leaves every public payload and the report matching unchanged (model contract section 5).
    """

    def __init__(self, config, path_key, gauge_enu=(0., 0., 0.), episode_start_s=0.):
        if not isinstance(config, GeoreferenceConfig):
            raise ContractError('GeoreferenceConfig required')
        self.config = config
        self.gauge_enu = _vec(gauge_enu, 'gauge_enu')
        self.filter = GeoreferenceFilter(config, episode_start_s)
        self._path = BrownianPath(f'{path_key}.bias', config.q_m2_per_s, config.path_horizon_s, config.path_resolution_s)
        self._noise_key = f'{path_key}.anchor_noise'
        self._initial_bias = np.asarray(config.initial_bias_enu, float)
        self.submissions = []
        self.measurements = []

    @property
    def episode_start_s(self):
        return self.filter.episode_start_s

    def begin_episode(self, sim_time_s):
        """Attach the episode origin: the path is indexed by time SINCE this instant, so d(t0) = d0 exactly."""
        self.filter.begin_episode(sim_time_s)
        return self

    # -- private quantities: evaluator stream only ---------------------------------------------------
    def _elapsed(self, sim_time_s):
        t = _real(sim_time_s, 'sim_time_s', low=0)
        if t < self.episode_start_s - TIME_TOLERANCE_S:
            raise ContractError(f'sim_time_s {t} precedes the episode origin {self.episode_start_s}')
        return max(0., t - self.episode_start_s)

    def _bias_at(self, sim_time_s):
        """d(t) = d0 + W(t - t0). PRIVATE: never an observation, a mask, a reward feature or a filter input."""
        east, north = self._path.at(self._elapsed(sim_time_s))
        return np.array([self._initial_bias[0] + east, self._initial_bias[1] + north], float)

    def _residual_at(self, sim_time_s):
        """d(t) - dhat(t): the error the next submitted report would carry. PRIVATE."""
        return self._bias_at(sim_time_s) - np.asarray(self.filter.correction_at(sim_time_s), float)

    def _measurement_noise(self, sim_time_s, anchor_id):
        """v ~ N(0, R), keyed by the event rather than by a consumed sequence, so it replays after a restart."""
        sigma = self.config.measurement_sigma_m
        if sigma == 0.:
            return np.zeros(2)
        key = b'%s|%s|%d' % (self._noise_key.encode(), _ident(anchor_id, 'anchor_id').encode(),
                             int(round(self._elapsed(sim_time_s) / TIME_TOLERANCE_S)))
        east, north = _standard_normal_pair(key)
        return np.array([sigma * east, sigma * north], float)

    # -- public interface ------------------------------------------------------------------------------
    def public_summary(self, sim_time_s):
        return self.filter.summary_at(sim_time_s)

    def attempt_fix(self, sim_time_s, anchor_id, available):
        """One measurement opportunity. The caller has already charged the travel and dwell time.

        An unavailable anchor is recorded, counted and leaves the covariance growing: it buys nothing at all.
        An available anchor returns z = d(t) + v and runs the common update.
        """
        t = _real(sim_time_s, 'sim_time_s', low=0)
        anchor = _ident(anchor_id, 'anchor_id')
        if not available:
            before = float(np.linalg.norm(self._residual_at(t)))
            self.filter.record_unavailable(t)
            event = MeasurementEvent(t, anchor, False, False, None, before, before)
            self.measurements.append(event)
            return event
        before = float(np.linalg.norm(self._residual_at(t)))
        z = self._bias_at(t) + self._measurement_noise(t, anchor)
        innovation, _ = self.filter.update(t, z, self.config.measurement_covariance_m2)
        after = float(np.linalg.norm(self._residual_at(t)))
        event = MeasurementEvent(t, anchor, True, True, innovation, before, after)
        self.measurements.append(event)
        return event

    def submit_report(self, report_index, track_id, local_position_enu, sim_time_s):
        """Freeze the global coordinate of one transmission: T0(p_local) + d(t) - dhat(t).

        Returns the GlobalSubmission. The returned global coordinate goes to the evaluator ledger; the public
        report the executive appends carries the LOCAL coordinate and the submission time.
        """
        t = _real(sim_time_s, 'sim_time_s', low=0)
        local = np.asarray(_vec(local_position_enu, 'local_position_enu'), float)
        residual = self._residual_at(t)
        gauge = np.asarray(self.gauge_enu, float)
        global_position = local + gauge + np.array([residual[0], residual[1], 0.], float)
        submission = GlobalSubmission(int(report_index), _ident(track_id, 'track_id'), t,
                                      tuple(float(v) for v in local), tuple(float(v) for v in global_position),
                                      self.filter.correction_at(t), tuple(float(v) for v in self._bias_at(t)),
                                      (float(residual[0]), float(residual[1])))
        self.submissions.append(submission)
        return submission

    def global_report_positions(self):
        """The frozen global coordinates, in submission order, for evaluator matching."""
        return tuple(s.global_position_enu for s in self.submissions)

    @property
    def disturbance_identity(self):
        """The ACTUAL stochastic identity this service is using, for recording and for pairing checks.

        Two services share a disturbance exactly when these hashes agree. Recording the derived hashes rather
        than a descriptive seed field is the point: a numeric seed can be equal while the counter keys differ,
        which is how a "paired" run can silently use two different Brownian paths.
        """
        return {'path_key': self._path.path_key,
                'path_sha256': hashlib.sha256(self._path.path_key.encode()).hexdigest(),
                'noise_key': self._noise_key,
                'noise_sha256': hashlib.sha256(self._noise_key.encode()).hexdigest(),
                'config_sha256': hashlib.sha256(
                    json.dumps(self.config.to_dict(), sort_keys=True, allow_nan=False,
                               separators=(',', ':')).encode()).hexdigest()}

    def evaluator_stream(self):
        return {'georeference_version': GEOREFERENCE_VERSION, 'config': self.config.to_dict(),
                'gauge_enu': list(self.gauge_enu), 'path_key': self._path.path_key,
                'path_depth': self._path.depth, 'path_finest_width_s': self._path.finest_width_s,
                'path_interpolation_sigma_m': self._path.interpolation_sigma_m,
                'measurements': [m.to_dict() for m in self.measurements],
                'submissions': [s.to_dict() for s in self.submissions]}


def translate_evaluator_targets(evaluator, gauge_enu):
    """A copy of an EvaluatorSpec with every target translated by the gauge: the other half of an A06 check."""
    from dataclasses import replace
    shift = np.asarray(_vec(gauge_enu, 'gauge_enu'), float)
    targets = tuple(replace(t, position_enu=tuple(float(v) for v in (np.asarray(t.position_enu, float) + shift)))
                    for t in evaluator.targets)
    return replace(evaluator, targets=targets)


__all__ = ['BrownianPath', 'GeoreferenceFilter', 'GeoreferenceService', 'GlobalSubmission', 'translate_evaluator_targets']
