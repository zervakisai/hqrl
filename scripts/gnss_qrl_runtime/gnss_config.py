"""Versioned runtime configuration for task flood-sar-village-gnss/1, and the safety checks that gate a run.

Why a separate config version
    A legacy mission config says nothing about drift, anchors or the timely score. Loading one into the GNSS
    runner would silently produce a mission without the factor the study is about, so `load_config` refuses
    anything that is not `gnss-runtime-config/1`, refuses unknown keys, and refuses a config missing any
    scientific quantity. There is no default for D600, for the initial bias, correction or covariance, for the
    measurement noise, for the anchor geometry or for the score normalisation.

Anchor placement, and the rule that must survive review
    `AnchorSet.selection_rule` is free text recorded next to the geometry. `derive_anchors` implements the one
    used here: the home anchor sits over the landing pad, and the field anchors sit at declared fractions of the
    world-bound extent. Both are functions of the briefing's own geometry. No victim position, no evaluator
    field and no detector output enters the choice, which is the property `LITERATURE_EN.md` and the model
    contract both require and which a reader can check by reading this function.

Provenance
    `provenance` is `engineering_default_unfrozen` until a lock manifest exists. Nothing in this module claims a
    frozen protocol, and `freeze_candidate.json` still lists these fields as pending.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from aerial_qrl.mission.contracts import ContractError                          # noqa: E402
from aerial_qrl.mission.georeference_contracts import GEOREFERENCE_VERSION, GeoreferenceConfig  # noqa: E402
from aerial_qrl.mission.relocalization import AnchorSet, assert_anchors_off_bank, home_anchor   # noqa: E402
from aerial_qrl.training.encoding import LAYOUT_V4_GNSS, check_return_mask_rule  # noqa: E402
from aerial_qrl.training.reward_gnss import SCORE_VERSION, TimelyScoreSpec       # noqa: E402

CONFIG_VERSION = 'gnss-runtime-config/1'
TASK_ID = 'flood-sar-village-gnss/1'
TOP_LEVEL_KEYS = {'config_version', 'task_id', 'note', 'provenance', 'georeference', 'anchors', 'score', 'encoding',
                  'episodes', 'comparators'}
REQUIRED_KEYS = {'config_version', 'task_id', 'provenance', 'georeference', 'anchors', 'score', 'encoding',
                 'episodes', 'comparators'}
#: Declared fractions of the world-bound extent for the field anchors (see the module docstring).
FIELD_ANCHOR_FRACTIONS = ((0.80, 0.25), (0.25, 0.80))


class GnssConfigError(ValueError):
    """The runtime configuration cannot be used for this task."""


def derive_anchors(briefing, transit_altitude_agl_m, dwell_s, availability_rule='at_anchor/1', off_s=None, on_s=None,
                   phase_s=0.):
    """Three anchors from briefing geometry alone: the pad, plus two at declared extent fractions."""
    bounds = briefing.world_bounds
    width, height = bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min
    z = briefing.landing_zone.center_enu[2] + transit_altitude_agl_m
    anchors = [home_anchor(briefing.landing_zone, transit_altitude_agl_m, dwell_s)]
    for i, (fx, fy) in enumerate(FIELD_ANCHOR_FRACTIONS):
        anchors.append({'anchor_id': f'A_field{i}', 'yaw_rad': 0., 'dwell_s': dwell_s,
                        'position_enu': (bounds.x_min + fx * width, bounds.y_min + fy * height, z)})
    extra = {}
    if availability_rule == 'intermittent/1':
        extra = {'off_s': off_s, 'on_s': on_s}
    return AnchorSet(schema_version=GEOREFERENCE_VERSION, anchors=tuple(anchors), availability_rule=availability_rule,
                     selection_rule=('pad centre plus world-bound extent fractions '
                                     f'{list(FIELD_ANCHOR_FRACTIONS)} at transit altitude; briefing geometry only, '
                                     'no victim, evaluator or detector input'),
                     phase_s=phase_s, **extra)


def _require(block, name, keys):
    if not isinstance(block, dict):
        raise GnssConfigError(f'{name}: expected an object')
    missing = sorted(set(keys) - set(block))
    extra = sorted(set(block) - set(keys))
    if missing or extra:
        raise GnssConfigError(f'{name}: missing {missing}, unknown {extra}')
    return block


def load_config(path_or_dict):
    """Parse and validate a GNSS runtime config. Every failure is a refusal, never a repaired default."""
    if isinstance(path_or_dict, (str, Path)):
        raw = json.loads(Path(path_or_dict).read_text())
    else:
        raw = dict(path_or_dict)
    if not isinstance(raw, dict):
        raise GnssConfigError('config: expected an object')
    if raw.get('config_version') != CONFIG_VERSION:
        raise GnssConfigError(f'config_version: expected {CONFIG_VERSION}, got {raw.get("config_version")!r}. '
                              'A legacy mission config describes no drift, no anchors and no timely score.')
    if raw.get('task_id') != TASK_ID:
        raise GnssConfigError(f'task_id: expected {TASK_ID}, got {raw.get("task_id")!r}')
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    missing = sorted(REQUIRED_KEYS - set(raw))
    if unknown or missing:
        raise GnssConfigError(f'config: unknown keys {unknown}, missing keys {missing}')

    georeference = GeoreferenceConfig.from_dict(raw['georeference'])
    anchors = AnchorSet.from_dict(raw['anchors'])
    score_block = _require(raw['score'], 'score', {'mission_budget_s', 'n_ref', 'tolerance_m', 'metric',
                                                   'false_report_weight'})
    score = TimelyScoreSpec(SCORE_VERSION, provenance=raw['provenance'], **score_block)
    encoding = _require(raw['encoding'], 'encoding', {'obs_encoding', 'return_mask_rule'})
    if encoding['obs_encoding'] != LAYOUT_V4_GNSS.obs_version:
        raise GnssConfigError(f'encoding.obs_encoding: expected {LAYOUT_V4_GNSS.obs_version}')
    check_return_mask_rule(encoding['return_mask_rule'])
    if len(anchors.anchors) > LAYOUT_V4_GNSS.anchor_slots:
        raise GnssConfigError(f'{len(anchors.anchors)} anchors exceed the {LAYOUT_V4_GNSS.anchor_slots} anchor slots '
                              f'of {LAYOUT_V4_GNSS.obs_version}')
    episodes = _require(raw['episodes'], 'episodes', {'realizations', 'reobservation_seed_base', 'path_key_prefix'})
    if not isinstance(episodes['realizations'], int) or episodes['realizations'] < 1:
        raise GnssConfigError('episodes.realizations: a positive integer is required')
    comparators = raw['comparators']
    if not isinstance(comparators, list) or not comparators:
        raise GnssConfigError('comparators: a non-empty list is required')
    from gnss_policies import COMPARATORS
    for entry in comparators:
        _require(entry, 'comparators[]', {'name', 'parameters'})
        if entry['name'] not in COMPARATORS:
            raise GnssConfigError(f'comparators[].name: expected one of {COMPARATORS}, got {entry["name"]!r}')
        if not isinstance(entry['parameters'], dict):
            raise GnssConfigError('comparators[].parameters: expected an object')
    return {'raw': raw, 'georeference': georeference, 'anchors': anchors, 'score': score, 'encoding': encoding,
            'episodes': episodes, 'comparators': comparators, 'provenance': raw['provenance']}


def check_anchor_safety(anchors, bank, backend_config):
    """Fail closed before the first cached-view episode (review 3, carry-forward 3).

    A RELOCALIZE reads a frame without perception so that a fix buys no image. On the cached-view backend a
    frame read AT a bank node still marks that node's capture as served, and under repeated_capture='invalid'
    every later read of it is void. An anchor sitting on a node would therefore destroy evidence the policy had
    not used, silently and only sometimes. This refuses the run instead.
    """
    return assert_anchors_off_bank(anchors, bank, backend_config.node_xy_tolerance_m,
                                   backend_config.node_z_tolerance_m, backend_config.node_yaw_tolerance_rad)


def candidate_config(briefing, d600_rms_m, measurement_sigma_m, dwell_s, transit_altitude_agl_m, n_ref,
                     realizations, comparators, mission_budget_s=None, availability_rule='at_anchor/1',
                     off_s=None, on_s=None, path_key_prefix='gnss-dev', note=None):
    """A development candidate config. Everything it writes is `engineering_default_unfrozen`."""
    anchors = derive_anchors(briefing, transit_altitude_agl_m, dwell_s, availability_rule, off_s, on_s)
    budget = float(briefing.budgets.mission_time_s if mission_budget_s is None else mission_budget_s)
    georeference = GeoreferenceConfig(
        version=GEOREFERENCE_VERSION, d600_rms_m=float(d600_rms_m), initial_bias_enu=(0., 0.),
        initial_correction_enu=(0., 0.), initial_covariance_m2=((0., 0.), (0., 0.)),
        measurement_sigma_m=float(measurement_sigma_m), path_horizon_s=4096., path_resolution_s=0.05,
        provenance='engineering_default_unfrozen')
    return {
        'config_version': CONFIG_VERSION, 'task_id': TASK_ID, 'provenance': 'engineering_default_unfrozen',
        'note': note or ('development candidate; d0, dhat0, P0, R, anchor geometry, dwell and budget are pending '
                         'pre-lock fields of freeze_candidate.json'),
        'georeference': georeference.to_dict(), 'anchors': anchors.to_dict(),
        'score': {'mission_budget_s': budget, 'n_ref': int(n_ref), 'tolerance_m': 3.0, 'metric': 'xyz',
                  'false_report_weight': 1.0},
        'encoding': {'obs_encoding': LAYOUT_V4_GNSS.obs_version, 'return_mask_rule': 'gnss_exhausted/1'},
        'episodes': {'realizations': int(realizations), 'reobservation_seed_base': 20260920,
                     'path_key_prefix': path_key_prefix},
        'comparators': [{'name': name, 'parameters': dict(parameters)} for name, parameters in comparators],
    }


__all__ = ['CONFIG_VERSION', 'FIELD_ANCHOR_FRACTIONS', 'TASK_ID', 'GnssConfigError', 'candidate_config',
           'check_anchor_safety', 'derive_anchors', 'load_config']
