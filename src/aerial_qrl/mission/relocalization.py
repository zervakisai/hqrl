"""Public anchor geometry, availability and the RELOCALIZE macro definition (task flood-sar-village-gnss/1).

What an anchor is, and what it is not
    An anchor is an ABSTRACT global-position service at a geometry-selected location in the local frame: flying
    to it and dwelling buys at most one measurement of the raw georeferencing error. It is NOT a camera-based
    landmark detector, and this study must never claim that an existing building became one. Nothing about the
    scene's appearance is used to place anchors, and their placement must not depend on where people are: the
    selection rule is recorded next to the anchor set and is a function of route geometry only.

Public, therefore available to the actor, the critic, the planner and the masks
    anchor identity, local position and yaw, dwell time, whether the service is available now, and the route and
    time cost of going there. `AnchorStatus` carries exactly these.

Private, therefore never here
    the surveyed global coordinate of the anchor, the measurement value z, the true bias and the residual. An
    anchor definition that contained a victim field would be a contract violation; `AnchorSet` has no field that
    could hold one, and a test asserts the public payload passes the leakage guard.

Availability schedules (`AVAILABILITY_RULES`)
    'at_anchor/1'        the core condition. GNSS is out everywhere else; standing at an anchor always yields a
                         measurement. Visiting costs travel and dwell, and that is the whole price.
    'intermittent/1'     the declared stress condition. The service has a duty cycle of `off_s` unavailable then
                         `on_s` available, repeating, with an explicit `phase_s`. Arriving during an off window
                         costs the full travel and dwell and returns nothing: the covariance keeps growing. The
                         cycle is a property of the service and of absolute simulator time, not of the policy,
                         so every arm meets the same schedule at the same times.

Timing
    `estimated_duration_s` is a PUBLIC estimate: straight-line path length at the transit speed plus the dwell.
    It is what a planner may budget with. The time actually charged is whatever the backend's calibrated time
    model returns for the executed commands, which is larger because it includes the model's command gaps. The
    two are deliberately different quantities and the runtime logs both.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .contracts import Contract, ContractError, _choice, _flag, _ident, _list, _real, _vec
from .georeference_contracts import GEOREFERENCE_VERSION

AVAILABILITY_RULES = ('at_anchor/1', 'intermittent/1')
#: review 1, ruling 3: the measurement is timestamped at the END of the dwell, so the travel and the full
#: dwell are paid before anything becomes available. Recorded in run metadata and in the manuscript.
FIX_TIMESTAMP_CONVENTION = 'end_of_dwell/1'


@dataclass(frozen=True)
class Anchor(Contract):
    """One public relocalization service location, in the local frame."""
    anchor_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    dwell_s: float

    def __post_init__(self):
        self._set(anchor_id=_ident(self.anchor_id, 'anchor_id'), position_enu=_vec(self.position_enu, 'position_enu'),
                  yaw_rad=_real(self.yaw_rad, 'yaw_rad'), dwell_s=_real(self.dwell_s, 'dwell_s', low=0, open_low=True))


@dataclass(frozen=True)
class AnchorSet(Contract):
    """The public anchor set with its availability schedule and the rule that placed it.

    `selection_rule` is free text recorded for provenance, e.g. which route-geometry criterion chose the
    positions. It exists so a reader can check that victim locations played no part.
    """
    schema_version: str
    anchors: Tuple[Anchor, ...]
    availability_rule: str
    selection_rule: str
    off_s: Optional[float] = None
    on_s: Optional[float] = None
    phase_s: float = 0.

    def __post_init__(self):
        anchors = tuple(Anchor.from_dict(a) if isinstance(a, dict) else a for a in _list(self.anchors, 'anchors'))
        if not anchors:
            raise ContractError('anchors: at least one anchor is required')
        if len({a.anchor_id for a in anchors}) != len(anchors):
            raise ContractError('anchors: duplicate anchor_id')
        if self.schema_version != GEOREFERENCE_VERSION:
            raise ContractError(f'schema_version: expected {GEOREFERENCE_VERSION}')
        rule = _choice(self.availability_rule, 'availability_rule', AVAILABILITY_RULES)
        self._set(anchors=anchors, availability_rule=rule, selection_rule=str(self.selection_rule),
                  phase_s=_real(self.phase_s, 'phase_s', low=0))
        if rule == 'intermittent/1':
            self._set(off_s=_real(self.off_s, 'off_s', low=0, open_low=True), on_s=_real(self.on_s, 'on_s', low=0, open_low=True))
        elif self.off_s is not None or self.on_s is not None:
            raise ContractError("off_s/on_s belong to availability_rule 'intermittent/1'")

    def find(self, anchor_id):
        return next((a for a in self.anchors if a.anchor_id == anchor_id), None)

    def available_at(self, sim_time_s):
        """Whether the service yields a measurement at this absolute simulator time (module docstring)."""
        t = _real(sim_time_s, 'sim_time_s', low=0)
        if self.availability_rule == 'at_anchor/1':
            return True
        cycle = self.off_s + self.on_s
        return ((t + self.phase_s) % cycle) >= self.off_s


@dataclass(frozen=True)
class AnchorStatus(Contract):
    """PUBLIC per-anchor observation entry. No global coordinate, no measurement, nothing truth-derived."""
    anchor_id: str
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    dwell_s: float
    available: bool
    path_length_m: float
    estimated_duration_s: float
    reachable: bool

    def __post_init__(self):
        self._set(anchor_id=_ident(self.anchor_id, 'anchor_id'), position_enu=_vec(self.position_enu, 'position_enu'),
                  yaw_rad=_real(self.yaw_rad, 'yaw_rad'), dwell_s=_real(self.dwell_s, 'dwell_s', low=0, open_low=True),
                  available=_flag(self.available, 'available'),
                  path_length_m=_real(self.path_length_m, 'path_length_m', low=0),
                  estimated_duration_s=_real(self.estimated_duration_s, 'estimated_duration_s', low=0),
                  reachable=_flag(self.reachable, 'reachable'))


def anchor_statuses(anchor_set, position_enu, transit_speed_m_s, obstacle_map=None, clearance_radius_m=None,
                    reserve=None, elapsed_s=None, sim_time_s=0.):
    """Public AnchorStatus for every anchor, from the vehicle's local position.

    reachable is the conjunction of two public checks: the straight leg to the anchor is free of the public
    obstacle map, and after arriving and dwelling a return is still affordable under the common reserve. An
    anchor that fails either is masked; it is not silently turned into some other action.
    """
    start = np.asarray(_vec(position_enu, 'position_enu'), float)
    speed = _real(transit_speed_m_s, 'transit_speed_m_s', low=0, open_low=True)
    statuses = []
    for anchor in anchor_set.anchors:
        goal = np.asarray(anchor.position_enu, float)
        length = float(np.linalg.norm(goal - start))
        duration = length / speed + anchor.dwell_s
        free = True
        if obstacle_map is not None and clearance_radius_m is not None:
            free = obstacle_map.check_segment(tuple(start), tuple(goal), clearance_radius_m).free
        affordable = True
        if reserve is not None and elapsed_s is not None:
            affordable = bool(reserve.allows(elapsed_s, duration, tuple(goal)))
        statuses.append(AnchorStatus(anchor.anchor_id, anchor.position_enu, anchor.yaw_rad, anchor.dwell_s,
                                     bool(anchor_set.available_at(sim_time_s)), length, duration, bool(free and affordable)))
    return tuple(statuses)


def home_anchor(landing_zone, transit_altitude_agl_m, dwell_s, yaw_rad=0., anchor_id='A_home'):
    """The anchor over the landing pad, used by the end-batch comparator.

    It is an ordinary anchor: visiting it is NON-TERMINAL. The vehicle hovers at transit altitude above the pad,
    takes its measurement, may then submit paid reports, and only a separate RETURN lands the vehicle.
    """
    centre = _vec(landing_zone.center_enu, 'landing_zone.center_enu')
    z = centre[2] + _real(transit_altitude_agl_m, 'transit_altitude_agl_m', low=0)
    return Anchor(anchor_id, (centre[0], centre[1], z), _real(yaw_rad, 'yaw_rad'), dwell_s)


def assert_anchors_off_bank(anchor_set, bank, xy_tolerance_m, z_tolerance_m, yaw_tolerance_rad):
    """Anchors must not coincide with a bank node, or the dwell would consume that node's cached capture.

    The RELOCALIZE macro reads a frame without perception so that a fix buys no image. On the cached-view
    backend a frame read at a bank node still marks that node's capture as served, which under
    repeated_capture='invalid' would silently destroy evidence the policy had not used. Raises ContractError
    naming every offending anchor.
    """
    offenders = [a.anchor_id for a in anchor_set.anchors
                 if bank.find(a.position_enu, a.yaw_rad, xy_tolerance_m, z_tolerance_m, yaw_tolerance_rad) is not None]
    if offenders:
        raise ContractError(f'anchors coincide with bank nodes and would consume their captures: {offenders}')
    return True


__all__ = ['AVAILABILITY_RULES', 'FIX_TIMESTAMP_CONVENTION', 'Anchor', 'AnchorSet', 'AnchorStatus', 'anchor_statuses', 'assert_anchors_off_bank',
           'home_anchor']
