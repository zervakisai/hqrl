"""Command time model 'time-model/1' for the cached-view backend (ADR 006 §6).

Parameters come from scripts/calibrate_time_model.py (time_model.json); this module only evaluates them, with the same
rules as that script's validation. Units: simulator seconds, metres, radians.

  GOTO     T = a + b * d / v + c * |dyaw| with the coefficients of the commanded speed when it is calibrated, else of the
           nearest calibrated speed (flag 'extrapolated_speed'); |dyaw| farther than 0.05 rad from every calibrated yaw
           change is flagged 'extrapolated_yaw'.
  HOLD     T = hold_s + median measured excess.
  TAKEOFF  median measured duration when the height is within 0.5 m of a calibrated height, else that duration scaled by
  LAND     height / nearest calibrated height ('extrapolated_height'). TAKEOFF height = goal z above the launch zone, LAND
           height = height above the landing zone before landing (the calibration used the previous commanded goal z).
  gap      median simulator time between the previous command kind and this one (capture, detector, planning and bridge
           latency while Gazebo keeps running); 0 for the first command of an episode; an unseen pair gives 0 and
           'unseen_gap'.
Flags never change a value silently: callers count them.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MODEL_VERSION = 'time-model/1'
HEIGHT_SUPPORT_TOLERANCE_M = .5
YAW_SUPPORT_TOLERANCE_RAD = .05


class TimeModelError(ValueError):
    pass


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise TimeModelError(f'{name}: expected a finite value >= 0, got {value}')
    return value


class TimeModel:
    def __init__(self, data, source=None, sha256=None):
        if not isinstance(data, dict) or data.get('model_version') != MODEL_VERSION:
            raise TimeModelError(f'expected model_version {MODEL_VERSION!r}')
        self.data, self.source, self.sha256 = data, source, sha256
        self._goto = {float(k): v for k, v in data['goto_by_speed_m_s'].items() if v.get('status') == 'calibrated'}
        if not self._goto:
            raise TimeModelError('no calibrated GOTO speed')
        if data.get('hold', {}).get('median_excess_s') is None:
            raise TimeModelError('HOLD is not calibrated')
        for kind in ('takeoff', 'land'):
            if data.get(kind, {}).get('status') != 'calibrated' or not data[kind].get('height_support_m'):
                raise TimeModelError(f'{kind.upper()} is not calibrated')

    @classmethod
    def from_file(cls, path):
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), str(Path(path).resolve()), hashlib.sha256(raw).hexdigest())

    def goto_s(self, distance_m, speed_m_s, yaw_change_rad=0.):
        distance, yaw = _nonnegative(distance_m, 'distance_m'), abs(float(yaw_change_rad))
        speed = float(speed_m_s)
        if not math.isfinite(speed) or speed <= 0 or not math.isfinite(yaw):
            raise TimeModelError('speed must be > 0 and yaw change finite')
        flags = []
        use = speed if speed in self._goto else min(self._goto, key=lambda s: (abs(s - speed), s))
        if use != speed:
            flags.append('extrapolated_speed')
        entry = self._goto[use]
        if not any(abs(yaw - s) <= YAW_SUPPORT_TOLERANCE_RAD for s in entry['yaw_support_rad']):
            flags.append('extrapolated_yaw')
        return entry['intercept_s'] + entry['d_over_v_coefficient'] * distance / speed + entry['abs_yaw_coefficient_s_per_rad'] * yaw, flags

    def hold_s(self, hold_s):
        return _nonnegative(hold_s, 'hold_s') + self.data['hold']['median_excess_s'], []

    def _height(self, kind, height_m):
        entry, height = self.data[kind], _nonnegative(height_m, f'{kind} height')
        nearest = min(entry['height_support_m'], key=lambda h: abs(h - height))
        if abs(nearest - height) > HEIGHT_SUPPORT_TOLERANCE_M:
            return entry['median_duration_s'] * height / nearest, ['extrapolated_height']
        return entry['median_duration_s'], []

    def takeoff_s(self, height_m):
        return self._height('takeoff', height_m)

    def land_s(self, height_m):
        return self._height('land', height_m)

    def gap_s(self, previous_kind, kind):
        if previous_kind is None:
            return 0., []
        entry = self.data.get('gaps_s', {}).get(f'{previous_kind}->{kind}')
        return (0., ['unseen_gap']) if entry is None else (float(entry['median']), [])

    def describe(self):
        return {'model_version': MODEL_VERSION, 'source': self.source, 'sha256': self.sha256,
                'calibration_runs': self.data.get('calibration_runs'),
                'calibrated_goto_speeds_m_s': sorted(self._goto)}
