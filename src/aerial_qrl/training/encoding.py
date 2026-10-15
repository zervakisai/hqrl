"""Observation encoding obs-encoding/2 and action space action-space/1 of the RL mission policy (ADR 008 sections 2-3 and
revision 1).

Versions
    obs-encoding/2 keeps the obs-encoding/1 layout (118 features, same order and normalisation) and changes only the view
    slot assignment (see 'Slot order and overflow'): views of tracks already treated as reported take no slot, and the
    view slots are shared round-robin across track slots. obs-encoding/1 is not implemented any more; a checkpoint trained
    with it must not be evaluated with this module.

obs-encoding/3 and action-space/2 (task flood-sar-village/1, ADR 010; EncodingLayout LAYOUT_V3)
    Same groups and normalisation as obs-encoding/2 with these differences: K_T = 8 track slots and K_V = 12 view slots;
    only tracks NOT treated as reported take track slots (reported tracks and their views take no slot); tracks are
    ordered by descending detections, then descending max_score, then track_id; the track feature 'reported' (always 0
    under this rule) is replaced by max_score (the track's public highest detector confidence, 0 when absent); the report
    count feature is min(len(reports), 10)/10. OBS_DIM = 4 + 5 + 12 + 1 + 8*8 + 12*(1 + 8 + 3) = 230;
    action-space/2 has 3 search + 12 inspect + 8 report + 1 return = 24 actions, masks as action-space/1 under the same
    RETURN mask rule (a slotted track is never reported). LAYOUT_V2 keeps obs-encoding/2 and action-space/1 bit for bit
    (module constants below).

Public inputs only
    ObservationEncoder and ActionSpace are built from the MissionBriefing (world bounds, sector order and prior weights,
    budgets) and the public MissionConfig (altitude_agl_m, ground_z_m, report_merge_radius_m). encode and masks read one
    contracts.PublicObservation and the policy's own public CoveragePlanState. Nothing else is accepted: no evaluator
    object, target, segmentation, report correctness, seed, scenario or variant name (ADR 008 section 2, 'Απαγορεύονται').

Public constants
    T = briefing.budgets.mission_time_s     R = briefing.budgets.return_reserve_s
    W = world_bounds.x_max - x_min          H = world_bounds.y_max - y_min          D = hypot(W, H)
    A = config.altitude_agl_m               z0 = config.ground_z_m                  r_merge = config.report_merge_radius_m
    Slots: S = 3 sectors (briefing order), K_T = 4 tracks, K_V = 8 views.

Feature vector (float64, OBS_DIM = 118, this exact order; FEATURE_NAMES lists every entry)
    0-3    time      elapsed_s/T, remaining_time_s/T, return_estimate_s/T, (remaining_time_s - return_estimate_s - R)/T
    4-8    vehicle   (x - x_min)/W, (y - y_min)/H, (z - z0)/A, sin(yaw), cos(yaw)
                     position = observation.vehicle.position_enu; yaw = executive.yaw_of(attitude_wxyz) (ENU, CCW from East)
    9-20   sectors   slot k (4 each, offset 9 + 4k), k-th briefing sector:
                     covered_fraction (observation.coverage entry of that sector, 0 when absent), prior_weight,
                     pending_fraction = pending nodes / searchable nodes of the sector in the policy's CoveragePlanState
                     (0 when the sector has no searchable node), XY distance from the vehicle to the sector's next pending
                     node / D (0 when no node is pending). Slots k >= number of briefing sectors are all zero.
    21     reports   min(len(observation.reports), 3)/3
    22-53  tracks    slot i (8 each, offset 22 + 8i): present (1), (track.x - vehicle.x)/100, (track.y - vehicle.y)/100,
                     sqrt(cov_xx + cov_yy)/10 (covariance_enu XY block), min(detections, 20)/20, age_s/T,
                     reported (1 when a submitted report has the track_id or lies within r_merge in XY: the rule of
                     executive.EvidencePolicy.is_reported), min(number of offered views of the track, 16)/16
                     (all observation.views of that track, including views without a view slot). Empty slots are zero.
    54-117 views     slot j (8 each, offset 54 + 8j): present (1), one-hot track slot (4), (view.x - vehicle.x)/100,
                     (view.y - vehicle.y)/100, estimated_duration_s/60. Empty slots are all zero.
    Nothing is clipped except the three min() caps; values outside [0, 1] (negative slack, large deltas) are kept.

Slot order and overflow (ADR 008 section 2; view rule of revision 1)
    Tracks: descending detections, then track_id; the first K_T take the track slots.
    Views, in this order of precedence:
      1. a view of a track treated as reported (track_is_reported, any track) takes no slot: views_of_reported_tracks;
      2. a view of a track outside the track slots, or without a track_id, takes no slot: views_without_track_slot;
      3. the remaining views get rank r = position within their track by (estimated_duration_s, view_id) and are ordered by
         (r, track slot), so the slots go round-robin over the track slots (cheapest view of every slotted track first,
         then the second cheapest, ...); the first K_V take the view slots in that order.
    overflow_views = views_without_track_slot + ranked views beyond K_V (views_of_reported_tracks is not overflow);
    overflow_tracks = tracks beyond K_T. masks, decode and index_of use the same assignment.

Action space action-space/1 (N_ACTIONS = 16; mask True = publicly feasible)
    0-2    SEARCH_SECTOR(k)       k < number of briefing sectors and the sector has a pending search node in the plan
    3-10   INSPECT_VIEW(slot j)   view slot j (obs-encoding/2 assignment) is occupied
    11-14  REPORT(track slot i)   track slot i is occupied and the track is not reported (id or merge radius)
    15     RETURN                 under the RETURN mask rule (below)
    decode(index, observation) builds the contracts.Action: SEARCH_SECTOR(sector_id), INSPECT_VIEW(view_id, track_id),
    REPORT(track_id, report_position_enu = the public track position), RETURN. The search goal itself is the policy's
    search_goal (next pending CoverageBaseline node of the sector). decode raises ValueError for an empty slot. Rejection
    masks (for one observation) are applied by rl_policy.RLMissionPolicy on top of these public masks.

RETURN mask rule (ActionSpace(..., return_mask_rule=...), RETURN_MASK_RULES; default 'always' = every earlier version, bit
for bit; ADR 012 addendum 2 D9)
    always            RETURN is always publicly feasible.
    search_exhausted  RETURN is publicly feasible only when no SEARCH_SECTOR action is feasible, i.e. no briefing sector has
                      a pending search node in the policy plan. The voluntary return is therefore aligned with the rule
                      comparators, which return only under the executive's reserve.
    The public mask can never be all-False: RETURN is masked only in the branch where a SEARCH_SECTOR action is feasible.
    A rule other than 'always' is named in the version ('action-space/2+return/search_exhausted') and in describe(), so a
    checkpoint cannot be read as trained under another rule. The executive's forced RETURN (reserve, rejection cap, decision
    cap) is not a policy action: executive._loop builds it without calling policy.act, so no mask can suppress it.

Failure handling: inconsistent inputs (more than S sectors, a plan state for other sectors, non-finite features) raise
ValueError; nothing is repaired.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from ..mission.contracts import (Action, ActionType, MissionBriefing, PublicObservation, Track, ViewOption,
                                 dedup_position_xy)
from ..mission.executive import yaw_of

OBS_ENCODING_VERSION = 'obs-encoding/2'
ACTION_SPACE_VERSION = 'action-space/1'
SECTOR_SLOTS, TRACK_SLOTS, VIEW_SLOTS = 3, 4, 8
TIME_FEATURES, VEHICLE_FEATURES, SECTOR_FEATURES, REPORT_FEATURES, TRACK_FEATURES = 4, 5, 4, 1, 8
VIEW_FEATURES = 1 + TRACK_SLOTS + 3
OBS_DIM = (TIME_FEATURES + VEHICLE_FEATURES + SECTOR_SLOTS * SECTOR_FEATURES + REPORT_FEATURES + TRACK_SLOTS * TRACK_FEATURES
           + VIEW_SLOTS * VIEW_FEATURES)
TIME_OFFSET, VEHICLE_OFFSET, SECTOR_OFFSET = 0, 4, 9
REPORT_INDEX = SECTOR_OFFSET + SECTOR_SLOTS * SECTOR_FEATURES
TRACK_OFFSET = REPORT_INDEX + REPORT_FEATURES
VIEW_OFFSET = TRACK_OFFSET + TRACK_SLOTS * TRACK_FEATURES
DELTA_SCALE_M = 100.
TRACK_SPREAD_SCALE_M = 10.
DETECTION_CAP = 20
REPORT_CAP = 3
VIEW_COUNT_CAP = 16
VIEW_DURATION_SCALE_S = 60.
VIEW_SLOT_RULE = ('views of reported tracks (id or merge radius) take no slot; remaining views of slotted tracks ordered by '
                  '(rank within track by estimated_duration_s then view_id, track slot); first 8 take the slots')
RETURN_MASK_RULES = ('always', 'search_exhausted', 'gnss_exhausted/1')
RETURN_MASK_TEXT = {'always': 'always',
                    'search_exhausted': ('only when no SEARCH_SECTOR action is feasible (no sector has a pending search node in '
                                         'the policy plan); the executive still forces a return by reserve'),
                    'gnss_exhausted/1': ('only when no SEARCH_SECTOR, no REPORT and no RELOCALIZE action is feasible: the '
                                         'voluntary finish accounts for reportable pending tracks and available correction '
                                         'decisions instead of inheriting the search-only rule. The executive still forces a '
                                         'return by reserve, so an episode always terminates')}

# --- flood-sar-village-gnss/1 feature block (appended after the view slots; absent from every legacy layout) ---
GEOREFERENCE_FEATURES = 6
ANCHOR_FEATURES = 6
#: Fixed normalisation of the georeference belief: the evaluator's own matching tolerance, so a feature value of
#: 1.0 means "the filter expects an error the size of the tolerance". Declared, not tuned.
TOLERANCE_SCALE_M = 3.0
FIX_COUNT_CAP = 5

N_ACTIONS = 16
SEARCH_ACTIONS = tuple(range(0, SECTOR_SLOTS))
INSPECT_ACTIONS = tuple(range(SECTOR_SLOTS, SECTOR_SLOTS + VIEW_SLOTS))
REPORT_ACTIONS = tuple(range(SECTOR_SLOTS + VIEW_SLOTS, SECTOR_SLOTS + VIEW_SLOTS + TRACK_SLOTS))
RETURN_ACTION = SECTOR_SLOTS + VIEW_SLOTS + TRACK_SLOTS
assert OBS_DIM == 118 and N_ACTIONS == RETURN_ACTION + 1


@dataclass(frozen=True)
class EncodingLayout:
    """Slot layout of one observation-encoding / action-space version pair (module docstring)."""
    obs_version: str
    action_version: str
    track_slots: int
    view_slots: int
    report_cap: int
    exclude_reported_tracks: bool      # True: reported tracks take no track slot (obs-encoding/3)
    score_feature: bool                # True: track feature 6 is max_score instead of 'reported' (obs-encoding/3)
    anchor_slots: int = 0              # > 0 only for flood-sar-village-gnss/1: adds the georeference and anchor block

    @property
    def view_features(self):
        return 1 + self.track_slots + 3

    @property
    def track_offset(self):
        return TRACK_OFFSET

    @property
    def view_offset(self):
        return TRACK_OFFSET + self.track_slots * TRACK_FEATURES

    @property
    def georeference_offset(self):
        """Start of the GNSS block; equals dim for a legacy layout, which has no such block."""
        return self.view_offset + self.view_slots * self.view_features

    @property
    def anchor_offset(self):
        return self.georeference_offset + (GEOREFERENCE_FEATURES if self.anchor_slots else 0)

    @property
    def dim(self):
        return self.anchor_offset + self.anchor_slots * ANCHOR_FEATURES

    @property
    def search_actions(self):
        return tuple(range(0, SECTOR_SLOTS))

    @property
    def inspect_actions(self):
        return tuple(range(SECTOR_SLOTS, SECTOR_SLOTS + self.view_slots))

    @property
    def report_actions(self):
        return tuple(range(SECTOR_SLOTS + self.view_slots, SECTOR_SLOTS + self.view_slots + self.track_slots))

    @property
    def relocalize_actions(self):
        first = SECTOR_SLOTS + self.view_slots + self.track_slots
        return tuple(range(first, first + self.anchor_slots))

    @property
    def return_action(self):
        return SECTOR_SLOTS + self.view_slots + self.track_slots + self.anchor_slots

    @property
    def n_actions(self):
        return self.return_action + 1


LAYOUT_V2 = EncodingLayout(OBS_ENCODING_VERSION, ACTION_SPACE_VERSION, TRACK_SLOTS, VIEW_SLOTS, REPORT_CAP, False, False)
LAYOUT_V3 = EncodingLayout('obs-encoding/3', 'action-space/2', 8, 12, 10, True, True)
#: flood-sar-village-gnss/1: obs-encoding/3 plus the georeference belief block and 3 anchor slots.
LAYOUT_V4_GNSS = EncodingLayout('obs-encoding/4-gnss', 'action-space/4-gnss', 8, 12, 10, True, True, anchor_slots=3)
LAYOUTS = {layout.obs_version: layout for layout in (LAYOUT_V2, LAYOUT_V3, LAYOUT_V4_GNSS)}
assert LAYOUT_V2.dim == OBS_DIM and LAYOUT_V2.n_actions == N_ACTIONS and LAYOUT_V3.dim == 230 and LAYOUT_V3.n_actions == 24
assert LAYOUT_V2.anchor_offset == LAYOUT_V2.dim and LAYOUT_V3.anchor_offset == LAYOUT_V3.dim  # legacy: no GNSS block
assert LAYOUT_V4_GNSS.dim == 230 + GEOREFERENCE_FEATURES + 3 * ANCHOR_FEATURES == 254
assert LAYOUT_V4_GNSS.n_actions == 27 and LAYOUT_V4_GNSS.relocalize_actions == (23, 24, 25)


def get_layout(layout):
    """EncodingLayout from None (obs-encoding/2), a version string or a layout."""
    if layout is None:
        return LAYOUT_V2
    if isinstance(layout, EncodingLayout):
        return layout
    if layout not in LAYOUTS:
        raise ValueError(f'obs_encoding: expected one of {sorted(LAYOUTS)}, got {layout!r}')
    return LAYOUTS[layout]


def check_return_mask_rule(rule):
    """RETURN mask rule from None ('always', every version before the rule existed) or a name in RETURN_MASK_RULES."""
    if rule is None:
        return RETURN_MASK_RULES[0]
    if rule not in RETURN_MASK_RULES:
        raise ValueError(f'return_mask_rule: expected one of {RETURN_MASK_RULES}, got {rule!r}')
    return rule


def _gnss_feature_names(layout):
    if not layout.anchor_slots:
        return []
    names = ['geo.rms_horizontal_error_over_tolerance', 'geo.time_since_fix_over_T', 'geo.fixes_accepted_capped',
             'geo.fixes_failed_capped', 'geo.last_fix_accepted', 'geo.no_fix_yet']
    for k in range(layout.anchor_slots):
        names += [f'anchor{k}.present', f'anchor{k}.dx_over_D', f'anchor{k}.dy_over_D',
                  f'anchor{k}.estimated_duration_over_T', f'anchor{k}.available_now', f'anchor{k}.reachable']
    return names


def _feature_names(layout=None):
    layout = get_layout(layout)
    names = ['time.elapsed_over_T', 'time.remaining_over_T', 'time.return_estimate_over_T', 'time.return_slack_over_T',
             'vehicle.x_over_W', 'vehicle.y_over_H', 'vehicle.z_agl_over_A', 'vehicle.sin_yaw', 'vehicle.cos_yaw']
    for k in range(SECTOR_SLOTS):
        names += [f'sector{k}.covered_fraction', f'sector{k}.prior_weight', f'sector{k}.pending_fraction',
                  f'sector{k}.next_node_xy_distance_over_D']
    names.append(f'reports.count_capped{layout.report_cap}_over_{layout.report_cap}')
    for i in range(layout.track_slots):
        names += [f'track{i}.present', f'track{i}.dx_over_100m', f'track{i}.dy_over_100m', f'track{i}.sqrt_trace_cov_xy_over_10m',
                  f'track{i}.detections_capped20_over_20', f'track{i}.age_over_T',
                  f'track{i}.max_score' if layout.score_feature else f'track{i}.reported', f'track{i}.views_capped16_over_16']
    for j in range(layout.view_slots):
        names += [f'view{j}.present'] + [f'view{j}.track_slot{i}' for i in range(layout.track_slots)] + \
                 [f'view{j}.dx_over_100m', f'view{j}.dy_over_100m', f'view{j}.estimated_duration_over_60s']
    return tuple(names + _gnss_feature_names(layout))


def _action_names(layout=None):
    layout = get_layout(layout)
    return tuple([f'SEARCH_SECTOR(sector slot {k})' for k in range(SECTOR_SLOTS)]
                 + [f'INSPECT_VIEW(view slot {j})' for j in range(layout.view_slots)]
                 + [f'REPORT(track slot {i})' for i in range(layout.track_slots)]
                 + [f'RELOCALIZE(anchor slot {k})' for k in range(layout.anchor_slots)] + ['RETURN'])


FEATURE_NAMES = _feature_names()
ACTION_NAMES = _action_names()
assert len(FEATURE_NAMES) == OBS_DIM and len(ACTION_NAMES) == N_ACTIONS
assert len(_feature_names(LAYOUT_V3)) == LAYOUT_V3.dim and len(_action_names(LAYOUT_V3)) == LAYOUT_V3.n_actions


# --- Plan state (public: the policy's own coverage plan) -------------------------------------------------------------

@dataclass(frozen=True)
class CoveragePlanState:
    """Pending search nodes of the policy's public CoverageBaseline plan, per sector in briefing order.

    searchable_nodes[k]: plan waypoints of sector k without a plan issue at the waypoint; pending_nodes[k]: those not yet
    flown; next_node_enu[k]: position of the first pending one in plan order (None when none is pending).
    """
    sector_ids: Tuple[str, ...]
    searchable_nodes: Tuple[int, ...]
    pending_nodes: Tuple[int, ...]
    next_node_enu: Tuple[Optional[Tuple[float, float, float]], ...]

    def __post_init__(self):
        n = len(self.sector_ids)
        if not (len(self.searchable_nodes) == len(self.pending_nodes) == len(self.next_node_enu) == n):
            raise ValueError('CoveragePlanState: one entry per sector required')
        for k in range(n):
            total, pending = self.searchable_nodes[k], self.pending_nodes[k]
            if isinstance(total, bool) or isinstance(pending, bool) or not 0 <= int(pending) <= int(total):
                raise ValueError(f'CoveragePlanState: sector {self.sector_ids[k]} counts invalid')
            if (self.next_node_enu[k] is None) != (pending == 0):
                raise ValueError(f'CoveragePlanState: sector {self.sector_ids[k]} next node must exist exactly when a node is pending')


def plan_state_from_coverage(plan, status, sector_ids):
    """CoveragePlanState from a planning.CoveragePlan and a waypoint status dict (None = pending, 'plan_issue:...' = never
    searchable, anything else = done), as kept by executive.CoverageBaselinePolicy and the RL policy."""
    sector_ids = tuple(sector_ids)
    searchable, pending, nxt = dict.fromkeys(sector_ids, 0), dict.fromkeys(sector_ids, 0), dict.fromkeys(sector_ids)
    for waypoint in plan.waypoints:
        if waypoint.sector_id not in searchable:
            raise ValueError(f'plan waypoint {waypoint.index}: sector {waypoint.sector_id!r} is not a briefing sector')
        state = status[waypoint.index]
        if isinstance(state, str) and state.startswith('plan_issue:'):
            continue
        searchable[waypoint.sector_id] += 1
        if state is None:
            pending[waypoint.sector_id] += 1
            if nxt[waypoint.sector_id] is None:
                nxt[waypoint.sector_id] = tuple(float(v) for v in waypoint.position_enu)
    return CoveragePlanState(sector_ids, tuple(searchable[s] for s in sector_ids), tuple(pending[s] for s in sector_ids),
                             tuple(nxt[s] for s in sector_ids))


# --- Slots ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotAssignment:
    tracks: Tuple[Optional[Track], ...]                          # TRACK_SLOTS entries
    views: Tuple[Optional[Tuple[ViewOption, int]], ...]         # VIEW_SLOTS entries: (view, track slot)
    overflow_tracks: int                                         # tracks not shown
    overflow_views: int                                          # views of unreported tracks not shown
    views_without_track_slot: int                                # part of overflow_views: track not slotted or no track_id
    views_of_reported_tracks: int                                # not shown by rule, not overflow


def track_is_reported(track, reports, merge_radius_m):
    """executive.EvidencePolicy.is_reported: a report with the track_id or within merge_radius_m (XY) of the track.

    Deduplication is a LOCAL-frame association question, so it reads `contracts.dedup_position_xy`, which serves
    both public record types. It is never a scoring coordinate: only the private global ledger is scored.
    """
    return any(r.track_id == track.track_id or math.dist(dedup_position_xy(r), track.position_enu[:2]) <= merge_radius_m
               for r in reports)


def assign_slots(observation, merge_radius_m, layout=None):
    """Track and view slots of obs-encoding/2 or /3 (module docstring, 'Slot order and overflow')."""
    layout = get_layout(layout)
    reported = {t.track_id for t in observation.tracks if track_is_reported(t, observation.reports, merge_radius_m)}
    if layout.exclude_reported_tracks:
        tracks = sorted((t for t in observation.tracks if t.track_id not in reported),
                        key=lambda t: (-t.detections, -(t.max_score or 0.), t.track_id))
    else:
        tracks = sorted(observation.tracks, key=lambda t: (-t.detections, t.track_id))
    slotted = tracks[:layout.track_slots]
    slot_of = {t.track_id: i for i, t in enumerate(slotted)}
    by_slot, of_reported, without_slot = {}, 0, 0
    for view in observation.views:
        if view.track_id is not None and view.track_id in reported:
            of_reported += 1
        elif view.track_id in slot_of:
            by_slot.setdefault(slot_of[view.track_id], []).append(view)
        else:
            without_slot += 1
    ranked = [(rank, slot, view) for slot, views in by_slot.items()
              for rank, view in enumerate(sorted(views, key=lambda v: (v.estimated_duration_s, v.view_id)))]
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    shown = ranked[:layout.view_slots]
    return SlotAssignment(tuple(slotted) + (None,) * (layout.track_slots - len(slotted)),
                          tuple((view, slot) for _, slot, view in shown) + (None,) * (layout.view_slots - len(shown)),
                          max(0, len(tracks) - layout.track_slots), without_slot + len(ranked) - len(shown), without_slot, of_reported)


def _public_inputs(briefing, config, layout=None):
    if not isinstance(briefing, MissionBriefing):
        raise TypeError('briefing: expected contracts.MissionBriefing')
    sectors = briefing.search_sectors
    if len(sectors) > SECTOR_SLOTS:
        raise ValueError(f'{get_layout(layout).obs_version} has {SECTOR_SLOTS} sector slots; the briefing has {len(sectors)} sectors')
    values = {}
    for key in ('altitude_agl_m', 'ground_z_m', 'report_merge_radius_m'):
        value = float(getattr(config, key))
        if not math.isfinite(value):
            raise ValueError(f'config.{key}: expected a finite number')
        values[key] = value
    if values['altitude_agl_m'] <= 0 or values['report_merge_radius_m'] <= 0:
        raise ValueError('config: altitude_agl_m and report_merge_radius_m must be positive')
    return tuple(s.sector_id for s in sectors), tuple(float(s.prior_weight) for s in sectors), values


def _check_plan_state(plan_state, sector_ids):
    if not isinstance(plan_state, CoveragePlanState):
        raise TypeError('plan_state: expected CoveragePlanState')
    if plan_state.sector_ids != sector_ids:
        raise ValueError(f'plan_state sectors {plan_state.sector_ids} differ from the briefing order {sector_ids}')


def _check_observation(observation):
    if not isinstance(observation, PublicObservation):
        raise TypeError('observation: expected contracts.PublicObservation')


@dataclass(frozen=True)
class EncodingInfo:
    tracks: int
    views: int
    overflow_tracks: int
    overflow_views: int
    views_without_track_slot: int
    views_of_reported_tracks: int
    slots: SlotAssignment


class ObservationEncoder:
    """obs-encoding/2 or /3 (module docstring). encode(observation, plan_state) -> (float64 vector of dim, EncodingInfo)."""
    version = OBS_ENCODING_VERSION
    dim = OBS_DIM

    def __init__(self, briefing, config, layout=None, anchor_ids=None):
        self.layout = get_layout(layout)
        self.anchor_ids = None if anchor_ids is None else tuple(anchor_ids)
        if self.layout.anchor_slots and self.anchor_ids is not None and len(self.anchor_ids) > self.layout.anchor_slots:
            raise ValueError(f'{len(self.anchor_ids)} anchors exceed the {self.layout.anchor_slots} slots of {self.layout.obs_version}')
        self.version, self.dim = self.layout.obs_version, self.layout.dim
        self.sector_ids, self.prior_weights, values = _public_inputs(briefing, config, self.layout)
        bounds, budgets = briefing.world_bounds, briefing.budgets
        self.x_min, self.y_min = float(bounds.x_min), float(bounds.y_min)
        self.W, self.H = float(bounds.x_max - bounds.x_min), float(bounds.y_max - bounds.y_min)
        self.D = math.hypot(self.W, self.H)
        self.T, self.R = float(budgets.mission_time_s), float(budgets.return_reserve_s)
        self.A, self.z0, self.merge_radius_m = values['altitude_agl_m'], values['ground_z_m'], values['report_merge_radius_m']

    def describe(self):
        layout = self.layout
        return {'version': self.version, 'dim': self.dim, 'feature_names': list(_feature_names(layout)),
                'slots': {'sectors': SECTOR_SLOTS, 'tracks': layout.track_slots, 'views': layout.view_slots,
                          'anchors': layout.anchor_slots},
                'anchor_order': None if self.anchor_ids is None else list(self.anchor_ids),
                'view_slot_rule': VIEW_SLOT_RULE.replace('first 8', f'first {layout.view_slots}'),
                'track_slot_rule': ('unreported tracks only; descending detections, descending max_score, track_id'
                                    if layout.exclude_reported_tracks else 'descending detections, then track_id'),
                'sector_order': list(self.sector_ids),
                'constants': {'T_mission_time_s': self.T, 'R_return_reserve_s': self.R, 'W_m': self.W, 'H_m': self.H,
                              'D_m': self.D, 'A_altitude_agl_m': self.A, 'x_min_m': self.x_min, 'y_min_m': self.y_min,
                              'ground_z_m': self.z0, 'report_merge_radius_m': self.merge_radius_m,
                              'delta_scale_m': DELTA_SCALE_M, 'track_spread_scale_m': TRACK_SPREAD_SCALE_M,
                              'detection_cap': DETECTION_CAP, 'report_cap': layout.report_cap, 'view_count_cap': VIEW_COUNT_CAP,
                              'view_duration_scale_s': VIEW_DURATION_SCALE_S,
                              'tolerance_scale_m': TOLERANCE_SCALE_M, 'fix_count_cap': FIX_COUNT_CAP}}

    def _encode_gnss(self, x, o, px, py):
        """The flood-sar-village-gnss/1 block: the filter's own belief and the public anchor geometry.

        Every value is public (model contract section 5): the covariance summary, the fix age and counts, and
        the anchor positions, availability and route costs. The true bias, the residual and the transmitted
        global coordinate are not here and cannot be: they never reach a PublicObservation.
        `available_now` is availability at THIS instant; under an intermittent schedule the anchor may be off
        by the time the vehicle arrives, which is the point of that stress condition.
        """
        layout = self.layout
        geo, T = o.georeference, self.T
        if geo is None:
            raise ValueError(f'{layout.obs_version} requires an observation with a georeference summary')
        if len(o.anchors) > layout.anchor_slots:
            raise ValueError(f'{len(o.anchors)} anchors exceed the {layout.anchor_slots} anchor slots of {layout.obs_version}')
        if self.anchor_ids is not None and [a.anchor_id for a in o.anchors] != list(self.anchor_ids):
            raise ValueError(f'anchor order {[a.anchor_id for a in o.anchors]} differs from the declared '
                             f'{list(self.anchor_ids)}: the anchor slot mapping is part of the frozen schema')
        base = layout.georeference_offset
        failed = geo.fixes_attempted - geo.fixes_accepted
        x[base:base + GEOREFERENCE_FEATURES] = (
            geo.rms_horizontal_error_m / TOLERANCE_SCALE_M,
            min(geo.time_since_fix_s, T) / T,
            min(geo.fixes_accepted, FIX_COUNT_CAP) / FIX_COUNT_CAP,
            min(failed, FIX_COUNT_CAP) / FIX_COUNT_CAP,
            1. if geo.last_fix_accepted else 0.,
            1. if geo.last_fix_accepted is None else 0.)
        for k, anchor in enumerate(o.anchors):
            slot = layout.anchor_offset + k * ANCHOR_FEATURES
            x[slot:slot + ANCHOR_FEATURES] = (
                1., (anchor.position_enu[0] - px) / self.D, (anchor.position_enu[1] - py) / self.D,
                min(anchor.estimated_duration_s, T) / T, float(anchor.available), float(anchor.reachable))

    def encode(self, observation, plan_state):
        _check_observation(observation)
        _check_plan_state(plan_state, self.sector_ids)
        o, T, layout = observation, self.T, self.layout
        x = np.zeros(layout.dim, dtype=np.float64)
        x[TIME_OFFSET:TIME_OFFSET + 4] = (o.elapsed_s / T, o.remaining_time_s / T, o.return_estimate_s / T,
                                          (o.remaining_time_s - o.return_estimate_s - self.R) / T)
        px, py, pz = o.vehicle.position_enu
        yaw = yaw_of(o.vehicle.attitude_wxyz)
        x[VEHICLE_OFFSET:VEHICLE_OFFSET + 5] = ((px - self.x_min) / self.W, (py - self.y_min) / self.H, (pz - self.z0) / self.A,
                                                math.sin(yaw), math.cos(yaw))
        coverage = {c.sector_id: c.covered_fraction for c in o.coverage}
        unknown = set(coverage) - set(self.sector_ids)
        if unknown:
            raise ValueError(f'observation coverage names sectors outside the briefing: {sorted(unknown)}')
        for k, sector_id in enumerate(self.sector_ids):
            base = SECTOR_OFFSET + k * SECTOR_FEATURES
            total, pending, node = plan_state.searchable_nodes[k], plan_state.pending_nodes[k], plan_state.next_node_enu[k]
            x[base:base + 4] = (coverage.get(sector_id, 0.), self.prior_weights[k], pending / total if total else 0.,
                                math.hypot(node[0] - px, node[1] - py) / self.D if node is not None else 0.)
        x[REPORT_INDEX] = min(len(o.reports), layout.report_cap) / layout.report_cap
        slots = assign_slots(o, self.merge_radius_m, layout)
        views_per_track = {}
        for view in o.views:
            views_per_track[view.track_id] = views_per_track.get(view.track_id, 0) + 1
        for i, track in enumerate(slots.tracks):
            if track is None:
                continue
            base = TRACK_OFFSET + i * TRACK_FEATURES
            c = track.covariance_enu
            sixth = (float(track.max_score or 0.) if layout.score_feature
                     else float(track_is_reported(track, o.reports, self.merge_radius_m)))
            x[base:base + 8] = (1., (track.position_enu[0] - px) / DELTA_SCALE_M, (track.position_enu[1] - py) / DELTA_SCALE_M,
                                math.sqrt(max(c[0][0] + c[1][1], 0.)) / TRACK_SPREAD_SCALE_M,
                                min(track.detections, DETECTION_CAP) / DETECTION_CAP, track.age_s / T, sixth,
                                min(views_per_track.get(track.track_id, 0), VIEW_COUNT_CAP) / VIEW_COUNT_CAP)
        for j, entry in enumerate(slots.views):
            if entry is None:
                continue
            view, track_slot = entry
            base = layout.view_offset + j * layout.view_features
            x[base] = 1.
            x[base + 1 + track_slot] = 1.
            x[base + 1 + layout.track_slots:base + layout.view_features] = ((view.position_enu[0] - px) / DELTA_SCALE_M,
                                                              (view.position_enu[1] - py) / DELTA_SCALE_M,
                                                              view.estimated_duration_s / VIEW_DURATION_SCALE_S)
        if layout.anchor_slots:
            self._encode_gnss(x, o, px, py)
        if not np.isfinite(x).all():
            raise ValueError('non-finite feature in the observation encoding')
        info = EncodingInfo(len(o.tracks), len(o.views), slots.overflow_tracks, slots.overflow_views,
                            slots.views_without_track_slot, slots.views_of_reported_tracks, slots)
        return x, info


@dataclass(frozen=True)
class ActionSlot:
    """Decoded action-space/1 entry: kind in ('search_sector', 'inspect_view', 'report_track', 'return'), the slot within
    its group (None for RETURN) and the public identifier it refers to (sector_id, view_id, track_id or None)."""
    index: int
    kind: str
    slot: Optional[int]
    ref: Optional[str]


class ActionSpace:
    """action-space/1 or /2 under a RETURN mask rule (module docstring). masks(observation, plan_state) -> bool array (n,),
    True = feasible; the rule is named in .version unless it is 'always'."""
    version = ACTION_SPACE_VERSION
    n = N_ACTIONS

    def __init__(self, briefing, config, layout=None, return_mask_rule=None, anchor_ids=None):
        self.layout = get_layout(layout)
        self.anchor_ids = None if anchor_ids is None else tuple(anchor_ids)
        self.return_mask_rule = check_return_mask_rule(return_mask_rule)
        self.version = self.layout.action_version + ('' if self.return_mask_rule == 'always' else f'+return/{self.return_mask_rule}')
        self.n = self.layout.n_actions
        self.sector_ids, _, values = _public_inputs(briefing, config, self.layout)
        self.merge_radius_m = values['report_merge_radius_m']

    def describe(self):
        return {'version': self.version, 'n': self.n, 'action_names': list(_action_names(self.layout)), 'sector_order': list(self.sector_ids),
                'anchor_order': None if self.anchor_ids is None else list(self.anchor_ids),
                'view_slots_from': self.layout.obs_version, 'return_mask_rule': self.return_mask_rule,
                'masks': {'SEARCH_SECTOR': 'sector exists and has a pending search node in the policy plan',
                          'INSPECT_VIEW': 'view slot occupied', 'REPORT': 'track slot occupied and track not reported (id or merge radius)',
                          'RELOCALIZE': ('anchor slot occupied and the anchor is reachable (free leg and an affordable return '
                                         'after travel and dwell); availability is a feature, never a mask'),
                          'RETURN': RETURN_MASK_TEXT[self.return_mask_rule]},
                'rejections': 'rl_policy: a rejected action is masked for that observation; a rejected RETURN only while another '
                              'action is feasible'}

    def masks(self, observation, plan_state):
        _check_observation(observation)
        _check_plan_state(plan_state, self.sector_ids)
        layout = self.layout
        mask = np.zeros(layout.n_actions, dtype=np.bool_)
        for k in range(len(self.sector_ids)):
            mask[layout.search_actions[k]] = plan_state.pending_nodes[k] > 0
        slots = assign_slots(observation, self.merge_radius_m, layout)
        for j, entry in enumerate(slots.views):
            mask[layout.inspect_actions[j]] = entry is not None
        for i, track in enumerate(slots.tracks):
            mask[layout.report_actions[i]] = track is not None and not track_is_reported(track, observation.reports, self.merge_radius_m)
        for k, anchor in enumerate(observation.anchors[:layout.anchor_slots]):
            # `reachable` is the common public check (free leg AND an affordable return after the visit).
            # `available` is NOT a mask: an unavailable service must remain takeable so that a failed fix can
            # cost its travel and dwell, which is the whole content of the intermittent stress condition.
            mask[layout.relocalize_actions[k]] = bool(anchor.reachable)
        mask[layout.return_action] = self._return_feasible(mask)
        return mask

    def _return_feasible(self, mask):
        """The voluntary finish rule. The mask can never be all-False, and the executive's forced RETURN
        (reserve, rejection cap, decision cap) is not a policy action, so an episode always terminates."""
        layout = self.layout
        if self.return_mask_rule == 'always':
            return True
        if self.return_mask_rule == 'search_exhausted':
            return not mask[:SECTOR_SLOTS].any()
        useful = (list(mask[:SECTOR_SLOTS]) + [mask[i] for i in layout.report_actions]
                  + [mask[i] for i in layout.relocalize_actions])
        return not any(useful)

    def decode(self, index, observation):
        """(contracts.Action, ActionSlot) for an action index; ValueError for an empty or unknown slot."""
        _check_observation(observation)
        layout = self.layout
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < layout.n_actions:
            raise ValueError(f'action index {index!r} outside 0..{layout.n_actions - 1}')
        index = int(index)
        if index == layout.return_action:
            return Action(ActionType.RETURN), ActionSlot(index, 'return', None, None)
        if index in layout.search_actions:
            k = index - layout.search_actions[0]
            if k >= len(self.sector_ids):
                raise ValueError(f'action {index}: sector slot {k} is empty ({len(self.sector_ids)} briefing sectors)')
            return Action(ActionType.SEARCH_SECTOR, sector_id=self.sector_ids[k]), ActionSlot(index, 'search_sector', k, self.sector_ids[k])
        slots = assign_slots(observation, self.merge_radius_m, layout)
        if index in layout.inspect_actions:
            j = index - layout.inspect_actions[0]
            if slots.views[j] is None:
                raise ValueError(f'action {index}: view slot {j} is empty')
            view = slots.views[j][0]
            return Action(ActionType.INSPECT_VIEW, view_id=view.view_id, track_id=view.track_id), ActionSlot(index, 'inspect_view', j, view.view_id)
        if index in layout.relocalize_actions:
            k = index - layout.relocalize_actions[0]
            if k >= len(observation.anchors):
                raise ValueError(f'action {index}: anchor slot {k} is empty ({len(observation.anchors)} anchors offered)')
            anchor_id = observation.anchors[k].anchor_id
            return Action(ActionType.RELOCALIZE, anchor_id=anchor_id), ActionSlot(index, 'relocalize_anchor', k, anchor_id)
        i = index - layout.report_actions[0]
        track = slots.tracks[i]
        if track is None:
            raise ValueError(f'action {index}: track slot {i} is empty')
        return (Action(ActionType.REPORT, track_id=track.track_id, report_position_enu=track.position_enu),
                ActionSlot(index, 'report_track', i, track.track_id))

    def index_of(self, action, observation):
        """Inverse of decode for actions expressible in action-space/1; ValueError otherwise."""
        _check_observation(observation)
        kind, layout = action.action_type, self.layout
        if kind == ActionType.RETURN:
            return layout.return_action
        if kind == ActionType.SEARCH_SECTOR:
            if action.sector_id not in self.sector_ids:
                raise ValueError(f'sector {action.sector_id!r} not in the briefing')
            return layout.search_actions[self.sector_ids.index(action.sector_id)]
        slots = assign_slots(observation, self.merge_radius_m, layout)
        if kind == ActionType.INSPECT_VIEW:
            for j, entry in enumerate(slots.views):
                if entry is not None and entry[0].view_id == action.view_id and action.track_id in (None, entry[0].track_id):
                    return layout.inspect_actions[j]
            raise ValueError(f'view {action.view_id!r} is not in a view slot')
        if kind == ActionType.RELOCALIZE:
            for k, anchor in enumerate(observation.anchors[:layout.anchor_slots]):
                if anchor.anchor_id == action.anchor_id:
                    return layout.relocalize_actions[k]
            raise ValueError(f'anchor {action.anchor_id!r} is not in an anchor slot')
        for i, track in enumerate(slots.tracks):
            if track is not None and track.track_id == action.track_id and tuple(action.report_position_enu) == tuple(track.position_enu):
                return layout.report_actions[i]
        raise ValueError(f'report of track {action.track_id!r} is not expressible in a track slot')


__all__ = ['ACTION_NAMES', 'ACTION_SPACE_VERSION', 'ANCHOR_FEATURES', 'ActionSlot', 'ActionSpace', 'CoveragePlanState',
           'EncodingInfo', 'EncodingLayout', 'FIX_COUNT_CAP', 'GEOREFERENCE_FEATURES', 'TOLERANCE_SCALE_M',
           'LAYOUTS', 'LAYOUT_V2', 'LAYOUT_V3', 'LAYOUT_V4_GNSS', 'get_layout', 'check_return_mask_rule',
           'FEATURE_NAMES', 'INSPECT_ACTIONS', 'N_ACTIONS', 'OBS_DIM', 'OBS_ENCODING_VERSION', 'ObservationEncoder',
           'REPORT_ACTIONS', 'RETURN_ACTION', 'RETURN_MASK_RULES', 'RETURN_MASK_TEXT', 'SEARCH_ACTIONS', 'SECTOR_SLOTS',
           'SlotAssignment', 'TRACK_SLOTS', 'VIEW_SLOTS', 'VIEW_SLOT_RULE', 'assign_slots', 'plan_state_from_coverage',
           'track_is_reported']
