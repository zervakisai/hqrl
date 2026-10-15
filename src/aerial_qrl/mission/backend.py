"""Backend interface: lifecycle, readiness, public sensing/commands, private evaluator truth.

Units, frames, clock
    Metres, seconds, radians; world ENU; quaternions wxyz (frames.py). Every time value is on
    the simulator clock, in seconds since the last successful reset (contracts.py).

Public/private
    SensorFrame and CommandOutcome are public and pass the key guard. EvaluatorTruth is a
    separate channel that must never reach actor, critic or planner inputs.

Lifecycle (MissionBackend enforces it; adapters implement the underscored hooks)

    UNPREPARED --reset ok--> PREPARED --begin (READY)--> ACTIVE --end--> ENDED --reset ok--> PREPARED
    any of UNPREPARED/PREPARED/ENDED/ACTIVE --stop ok--> STOPPED (terminal)

    - Invalidate first: reset, end and stop change state *before* calling the hook, so a hook
      that raises can never leave a previous preparation or episode usable.
    - Failed/rejected reset -> UNPREPARED. Not-READY begin -> UNPREPARED (a new reset is a new,
      recorded attempt; no silent retry). Failed readiness in PREPARED -> UNPREPARED.
    - During ACTIVE, any hook exception, untyped result, timing violation or non-READY readiness
      -> ENDED (infrastructure failure; the episode cannot continue).
    - end_episode -> ENDED even if its hook fails; ENDED requires a new reset.
    - stop -> FAULTED until the hook succeeds, then STOPPED. FAULTED allows only stop retries.
    - reset during ACTIVE is refused (no teleport inside an episode) and leaves the episode running.
    Every operation and failure is appended to MissionBackend.events. With a LifecycleJournal, an
    'intent' record is fsync'ed to disk before each operation runs and a 'result' record after it,
    so an operation interrupted by power loss or SIGKILL remains visible as an open intent.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .contracts import (TIME_TOLERANCE_S, Contract, ContractError, EvaluatorSpec, SensorStatus, VehicleEstimate,
                        _count, _flag, _nested, _nested_seq, _optional_real, _real, _vec, check_scenario_pair,
                        check_sensor_times)
from .leakage import assert_public_payload


class BackendError(RuntimeError):
    pass


class ReadinessState(str, Enum):
    START = 'START'
    RPC_READY = 'RPC_READY'
    RESET = 'RESET'
    SETTINGS_READBACK = 'SETTINGS_READBACK'
    FRAME_CALIBRATION = 'FRAME_CALIBRATION'
    SENSOR_CONDITIONING = 'SENSOR_CONDITIONING'
    POSE_AND_SENSOR_GATE = 'POSE_AND_SENSOR_GATE'
    READY = 'READY'
    FAILED = 'FAILED'


STARTUP_SEQUENCE = (ReadinessState.START, ReadinessState.RPC_READY, ReadinessState.RESET,
                    ReadinessState.SETTINGS_READBACK, ReadinessState.FRAME_CALIBRATION,
                    ReadinessState.SENSOR_CONDITIONING, ReadinessState.POSE_AND_SENSOR_GATE, ReadinessState.READY)


class LifecycleState(str, Enum):
    UNPREPARED = 'UNPREPARED'
    PREPARED = 'PREPARED'
    ACTIVE = 'ACTIVE'
    ENDED = 'ENDED'
    STOPPED = 'STOPPED'
    FAULTED = 'FAULTED'


@dataclass(frozen=True)
class ReadinessReport(Contract):
    """READY only after every startup stage passed in order with no recorded error."""
    history: Tuple[ReadinessState, ...]
    errors: Tuple[str, ...]

    def __post_init__(self):
        try:
            history = tuple(ReadinessState(s) for s in self.history)
        except ValueError:
            raise ContractError('history: unknown readiness state') from None
        errors = tuple(self.errors)
        if not all(isinstance(e, str) and e for e in errors):
            raise ContractError('errors: non-empty strings required')
        stages = history[:-1] if history and history[-1] == ReadinessState.FAILED else history
        if not history or stages != STARTUP_SEQUENCE[:len(stages)]:
            raise ContractError('history: stages must follow the startup sequence without skipping')
        if (history[-1] == ReadinessState.FAILED) != bool(errors):
            raise ContractError('FAILED requires errors, and errors require FAILED')
        self._set(history=history, errors=errors)

    @property
    def state(self):
        return self.history[-1]

    @property
    def ready(self):
        return self.state == ReadinessState.READY


class CommandKind(str, Enum):
    TAKEOFF = 'TAKEOFF'
    GOTO = 'GOTO'
    HOLD = 'HOLD'
    LAND = 'LAND'


@dataclass(frozen=True)
class FlightCommand(Contract):
    """Goal for the shared controller (ENU metres, m/s, s); policies never send motor commands.

    yaw_enu_rad (optional, TAKEOFF/GOTO only): body heading in ENU (0 = East, counter-clockwise), wrapped to [-pi, pi);
    it is held after the command ends. Without it the controller keeps the current heading."""
    kind: CommandKind
    position_enu: Optional[Tuple[float, float, float]] = None
    max_speed_m_s: Optional[float] = None
    hold_s: Optional[float] = None
    yaw_enu_rad: Optional[float] = None

    def __post_init__(self):
        try:
            kind = CommandKind(self.kind)
        except ValueError:
            raise ContractError('kind: unknown command') from None
        need = {CommandKind.TAKEOFF: {'position_enu', 'max_speed_m_s'}, CommandKind.GOTO: {'position_enu', 'max_speed_m_s'},
                CommandKind.HOLD: {'hold_s'}, CommandKind.LAND: set()}[kind]
        present = {k for k in ('position_enu', 'max_speed_m_s', 'hold_s') if getattr(self, k) is not None}
        if present != need:
            raise ContractError(f'{kind.value}: requires exactly {sorted(need)}')
        if self.yaw_enu_rad is not None and kind not in (CommandKind.TAKEOFF, CommandKind.GOTO):
            raise ContractError(f'{kind.value}: yaw_enu_rad is only allowed for TAKEOFF and GOTO')
        yaw = _optional_real(self.yaw_enu_rad, 'yaw_enu_rad')
        self._set(kind=kind, position_enu=None if self.position_enu is None else _vec(self.position_enu, 'position_enu'),
                  max_speed_m_s=_optional_real(self.max_speed_m_s, 'max_speed_m_s', low=0, open_low=True),
                  hold_s=_optional_real(self.hold_s, 'hold_s', low=0, open_low=True),
                  yaw_enu_rad=None if yaw is None else (yaw + math.pi) % (2 * math.pi) - math.pi)


@dataclass(frozen=True)
class CommandOutcome(Contract):
    """Public result from the vehicle's own estimate; collisions are reported by the evaluator."""
    accepted: bool
    completed: bool
    sim_time_start_s: float
    sim_time_end_s: float
    estimated_distance_m: float
    reason: Optional[str]

    def __post_init__(self):
        self._set(accepted=_flag(self.accepted, 'accepted'), completed=_flag(self.completed, 'completed'),
                  sim_time_start_s=_real(self.sim_time_start_s, 'sim_time_start_s', low=0),
                  sim_time_end_s=_real(self.sim_time_end_s, 'sim_time_end_s', low=0),
                  estimated_distance_m=_real(self.estimated_distance_m, 'estimated_distance_m', low=0))
        if self.sim_time_end_s < self.sim_time_start_s:
            raise ContractError('command outcome: time runs backwards')
        if self.completed and not self.accepted:
            raise ContractError('command outcome: completed requires accepted')
        if (not self.completed) != (self.reason is not None):
            raise ContractError('command outcome: reason required exactly when not completed')


@dataclass(frozen=True)
class PointMeasurement(Contract):
    """Synthetic detector output from fake backends only; no identity and no confidence."""
    position_enu: Tuple[float, float, float]

    def __post_init__(self):
        self._set(position_enu=_vec(self.position_enu, 'position_enu'))


@dataclass(frozen=True)
class SensorFrame(Contract):
    """Public sensor sample at sim_time_s. Real backends give image paths; perception detects."""
    sim_time_s: float
    vehicle: VehicleEstimate
    sensors: Tuple[SensorStatus, ...]
    rgb_path: Optional[str]
    depth_path: Optional[str]
    synthetic_measurements: Tuple[PointMeasurement, ...]

    def __post_init__(self):
        self._set(sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0), vehicle=_nested(VehicleEstimate, self.vehicle, 'vehicle'),
                  sensors=_nested_seq(SensorStatus, self.sensors, 'sensors'),
                  synthetic_measurements=_nested_seq(PointMeasurement, self.synthetic_measurements, 'synthetic_measurements'))
        check_sensor_times(self.sim_time_s, self.sensors)
        valid = {s.stream for s in self.sensors if s.valid}
        for key, stream in (('rgb_path', 'rgb'), ('depth_path', 'depth')):
            path = getattr(self, key)
            if path is not None and not isinstance(path, str):
                raise ContractError(f'{key}: expected path string')
            if path is not None and stream not in valid:
                raise ContractError(f'{key}: payload given for a stream reported invalid')
        assert_public_payload(self.to_dict())


@dataclass(frozen=True)
class HealthReport(Contract):
    """In-flight health from the vehicle side (heartbeat, pose and process liveness)."""
    ok: bool
    reason: Optional[str]
    sim_time_s: Optional[float]

    def __post_init__(self):
        self._set(ok=_flag(self.ok, 'ok'), sim_time_s=_optional_real(self.sim_time_s, 'sim_time_s', low=0))
        if self.ok != (self.reason is None) or (self.reason is not None and not (isinstance(self.reason, str) and self.reason)):
            raise ContractError('health: reason required exactly when not ok')


@dataclass(frozen=True)
class EvaluatorTruth(Contract):
    """Private channel. Never convert into observations, masks, rewards-as-inputs or planner costs.

    collisions and min_clearance_m are None when the backend has no such measurement; an
    unmeasured quantity is never reported as zero.
    """
    sim_time_s: float
    vehicle_position_enu: Tuple[float, float, float]
    collisions: Optional[int]
    min_clearance_m: Optional[float]
    pre_episode_teleports: int
    episode_teleports: int
    evaluator: EvaluatorSpec

    def __post_init__(self):
        self._set(sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0),
                  vehicle_position_enu=_vec(self.vehicle_position_enu, 'vehicle_position_enu'),
                  min_clearance_m=_optional_real(self.min_clearance_m, 'min_clearance_m', low=0),
                  evaluator=_nested(EvaluatorSpec, self.evaluator, 'evaluator'))
        if self.collisions is not None:
            _count(self.collisions, 'collisions')
        for key in ('pre_episode_teleports', 'episode_teleports'):
            _count(getattr(self, key), key)


S = LifecycleState


class MissionBackend(ABC):
    """Template methods enforce lifecycle and clock rules; subclasses implement the hooks."""

    def __init__(self, journal=None):
        self._state = S.UNPREPARED
        self._scenario = None
        self._evaluator = None
        self._generation = 0
        self._clock = None
        self._stamps = {}
        self._intents = {}
        self.events = []
        self.journal = journal

    @property
    def state(self):
        return self._state

    @property
    def episode_active(self):
        return self._state == S.ACTIVE

    def journal_record(self, kind, **fields):
        """Adapters record owned process identities and notes; no-op without a journal."""
        if self.journal is not None:
            self.journal.append(kind, generation=self._generation, **fields)

    def _intent(self, operation):
        if self.journal is not None:
            record = self.journal.append('intent', operation=operation, from_state=self._state.value, generation=self._generation)
            self._intents[operation] = record['seq']

    def _log(self, operation, before, ok, error=None):
        event = {'operation': operation, 'from': before.value, 'to': self._state.value, 'ok': ok, 'generation': self._generation}
        if error is not None:
            event['error'] = f'{error.__class__.__name__}: {error}'
        self.events.append(event)
        if self.journal is not None:
            self.journal.append('result', intent_seq=self._intents.pop(operation, None), **event)

    def _require(self, operation, *allowed):
        if self._state not in allowed:
            raise BackendError(f'{operation} is not allowed in lifecycle state {self._state.value}')

    def _fail(self, operation, before, error, state):
        self._state = state
        if state in (S.UNPREPARED, S.FAULTED, S.STOPPED):
            self._scenario = self._evaluator = None
        self._log(operation, before, False, error)

    def reset(self, scenario, evaluator):
        self._require('reset', S.UNPREPARED, S.PREPARED, S.ENDED)
        self._intent('reset')
        before = self._state
        self._state, self._scenario, self._evaluator = S.UNPREPARED, None, None
        self._generation += 1
        self._clock, self._stamps = None, {}
        try:
            check_scenario_pair(scenario, evaluator)
        except ContractError as exc:
            self._fail('reset', before, exc, S.UNPREPARED)
            raise
        try:
            self._reset(scenario, evaluator)
        except Exception as exc:
            self._fail('reset', before, exc, S.UNPREPARED)
            raise BackendError(f'reset failed; backend is UNPREPARED: {exc}') from exc
        self._scenario, self._evaluator, self._state = scenario, evaluator, S.PREPARED
        self._log('reset', before, True)

    def readiness(self):
        self._require('readiness', S.PREPARED, S.ACTIVE)
        self._intent('readiness')
        before = self._state
        failure_state = S.ENDED if before == S.ACTIVE else S.UNPREPARED
        try:
            report = self._readiness()
            if not isinstance(report, ReadinessReport):
                raise BackendError('backend returned an untyped readiness report')
        except Exception as exc:
            self._fail('readiness', before, exc, failure_state)
            raise BackendError(f'readiness check failed; state {failure_state.value}: {exc}') from exc
        if before == S.ACTIVE and not report.ready:
            error = BackendError(f'readiness lost in flight at {report.state.value}: {list(report.errors)}')
            self._fail('readiness', before, error, S.ENDED)
            raise error
        self._log('readiness', before, True)
        return report

    def begin_episode(self):
        self._require('begin_episode', S.PREPARED)
        self._intent('begin_episode')
        try:
            report = self.readiness()
        except BackendError as exc:
            self._log('begin_episode', S.PREPARED, False, exc)
            raise
        if not report.ready:
            error = BackendError(f'begin_episode requires READY, got {report.state.value}: {list(report.errors)}; '
                                 'a new reset (new recorded attempt) is required')
            self._fail('begin_episode', S.PREPARED, error, S.UNPREPARED)
            raise error
        self._state = S.ACTIVE
        self._log('begin_episode', S.PREPARED, True)

    def read_sensors(self):
        self._require('read_sensors', S.ACTIVE)
        self._intent('read_sensors')
        try:
            frame = self._read_sensors()
            if not isinstance(frame, SensorFrame):
                raise BackendError('backend returned an untyped sensor frame')
            self._scenario.timing.check(frame.sim_time_s, frame.sensors)
            self._check_clock(frame.sim_time_s, 'sensor frame')
            for status in frame.sensors:
                previous = self._stamps.get(status.stream)
                if status.timestamp_s is not None and previous is not None and status.timestamp_s < previous - TIME_TOLERANCE_S:
                    raise ContractError(f'{status.stream}: timestamp moved backwards')
        except Exception as exc:
            self._fail('read_sensors', S.ACTIVE, exc, S.ENDED)
            raise BackendError(f'sensor read invalidated the episode: {exc}') from exc
        self._clock = frame.sim_time_s
        self._stamps.update({s.stream: s.timestamp_s for s in frame.sensors if s.timestamp_s is not None})
        self._log('read_sensors', S.ACTIVE, True)
        return frame

    def command(self, command):
        self._require('command', S.ACTIVE)
        command = _nested(FlightCommand, command, 'command')  # malformed input: rejected before execution
        self._intent('command')
        try:
            outcome = self._command(command)
            if not isinstance(outcome, CommandOutcome):
                raise BackendError('backend returned an untyped command outcome')
            self._check_clock(outcome.sim_time_start_s, 'command start')
        except Exception as exc:
            self._fail('command', S.ACTIVE, exc, S.ENDED)
            raise BackendError(f'command invalidated the episode: {exc}') from exc
        self._clock = outcome.sim_time_end_s
        self._log('command', S.ACTIVE, True)
        return outcome

    def check_health(self):
        """In-flight hard-stop gate. A fault ends the episode; the adapter has already requested landing."""
        self._require('check_health', S.ACTIVE)
        self._intent('check_health')
        try:
            report = self._health()
            if not isinstance(report, HealthReport):
                raise BackendError('backend returned an untyped health report')
        except Exception as exc:
            self._fail('check_health', S.ACTIVE, exc, S.ENDED)
            raise BackendError(f'health check failed; episode ended: {exc}') from exc
        if not report.ok:
            error = BackendError(f'health fault in flight: {report.reason}; episode ended')
            self._fail('check_health', S.ACTIVE, error, S.ENDED)
            raise error
        self._log('check_health', S.ACTIVE, True)
        return report

    def end_episode(self):
        if self._state == S.ENDED:
            return
        self._require('end_episode', S.ACTIVE)
        self._intent('end_episode')
        self._state = S.ENDED
        try:
            self._end_episode()
        except Exception as exc:
            self._fail('end_episode', S.ACTIVE, exc, S.ENDED)
            raise BackendError(f'end_episode hook failed; state ENDED, new reset required: {exc}') from exc
        self._log('end_episode', S.ACTIVE, True)

    def evaluator_truth(self):
        self._require('evaluator_truth', S.PREPARED, S.ACTIVE, S.ENDED)
        self._intent('evaluator_truth')
        before = self._state
        try:
            truth = self._evaluator_truth()
            if not isinstance(truth, EvaluatorTruth):
                raise BackendError('backend returned untyped evaluator truth')
        except Exception as exc:
            self._fail('evaluator_truth', before, exc, S.UNPREPARED if before == S.PREPARED else S.ENDED)
            raise BackendError(f'evaluator truth unavailable: {exc}') from exc
        self._log('evaluator_truth', before, True)
        return truth

    def stop(self):
        if self._state == S.STOPPED:
            return
        self._intent('stop')
        before = self._state
        self._state, self._scenario, self._evaluator = S.FAULTED, None, None
        try:
            self._stop()
        except Exception as exc:
            self._fail('stop', before, exc, S.FAULTED)
            raise BackendError(f'stop failed; external state unknown (FAULTED): {exc}') from exc
        self._state = S.STOPPED
        self._log('stop', before, True)

    def _check_clock(self, sim_time_s, label):
        if self._clock is not None and sim_time_s < self._clock - TIME_TOLERANCE_S:
            raise ContractError(f'{label}: simulator clock moved backwards ({sim_time_s} < {self._clock})')

    @abstractmethod
    def _reset(self, scenario, evaluator): ...

    @abstractmethod
    def _readiness(self): ...

    @abstractmethod
    def _read_sensors(self): ...

    @abstractmethod
    def _command(self, command): ...

    @abstractmethod
    def _evaluator_truth(self): ...

    @abstractmethod
    def _stop(self): ...

    def _end_episode(self):
        """Adapter hook, e.g. confirm disarm; default does nothing."""

    def _health(self):
        """Adapter hook; backends without in-flight monitoring report healthy at the last known time."""
        return HealthReport(True, None, self._clock)
