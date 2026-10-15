"""Read-only public index of an observation bank written by scripts/collect_observation_bank.py (ADR 006 §5).

Only manifest.json and public/ are read; the evaluator/ folder is never opened by this module.
A capture is served only when it is valid (RGB and depth present, stamps consistent) and, unless require_detections is
False, a frozen-detector result exists for its RGB sha256. Unusable captures are listed in .unusable; a node without a
usable capture is not in the index, so a backend reports invalid RGB/depth there (no image of another node, no fallback).
Re-observation model (declared in ADR 006 §5): visit v (0-based, per episode) of a node serves repeat v mod R, where R is
the number of usable repeats of that node, in repeat order. MultiBank applies the same rule to a node whose repeats a
declared repeat extension splits over several parts (MultiBank docstring).
Units: metres, radians (ENU); stamps are the bank simulator seconds and are not reused as mission time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np

from .array_store import array_sha256, load_array
from .perception import DetectorResult

BANK_VERSION = 'observation-bank/1'
BANK_VERSION_FRAMES = 'observation-bank/2'  # K frames per repeat ('frame' field; perception model v3)
BANK_VERSIONS = (BANK_VERSION, BANK_VERSION_FRAMES)


class BankError(RuntimeError):
    pass


@dataclass(frozen=True)
class BankCapture:
    node_id: str
    repeat: int
    position_enu: Tuple[float, float, float]
    yaw_rad: float
    rgb_path: str
    rgb_sha256: str
    rgb_stamp_s: float
    depth_path: str
    depth_sha256: str
    depth_shape: Tuple[int, ...]
    frame: int = 0


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class ObservationBank:
    def __init__(self, bank_dir, verify_files=False, require_detections=True):
        self.root = Path(bank_dir).resolve()
        raw = (self.root / 'manifest.json').read_bytes()
        self.manifest, self.manifest_sha256 = json.loads(raw), hashlib.sha256(raw).hexdigest()
        if self.manifest.get('bank_version') not in BANK_VERSIONS:
            raise BankError(f'{self.root}: expected bank_version in {BANK_VERSIONS}')
        self._detections = {}
        detections = self.root / 'public' / 'detections.jsonl'
        if detections.is_file():
            for line in detections.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._detections[record['rgb_sha256']] = DetectorResult.from_dict(record['result'])
        all_captures, self.unusable = {}, []
        for line in (self.root / 'public' / 'captures.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            streams = entry.get('streams', {})
            reason = None if entry.get('valid') and 'rgb' in streams and 'depth' in streams else 'invalid_capture'
            if reason is None and require_detections and streams['rgb']['sha256'] not in self._detections:
                reason = 'no_detector_result'
            if reason is not None:
                self.unusable.append({'node_id': entry.get('node_id'), 'repeat': entry.get('repeat'), 'reason': reason})
                continue
            capture = BankCapture(entry['node_id'], int(entry['repeat']), tuple(float(v) for v in entry['node_position_enu']),
                                  float(entry['node_yaw_rad']), str(self.root / streams['rgb']['file']), streams['rgb']['sha256'],
                                  float(streams['rgb']['stamp_s']), str(self.root / streams['depth']['file']), streams['depth']['sha256'],
                                  tuple(int(v) for v in streams['depth']['shape']), int(entry.get('frame', 0)))
            all_captures.setdefault(capture.node_id, []).append(capture)
        by_node, frames = {}, {}
        for node_id, captures in all_captures.items():
            if len({(c.repeat, c.frame) for c in captures}) != len(captures):
                raise BankError(f'node {node_id}: duplicate (repeat, frame) index')
            if len({(c.position_enu, c.yaw_rad) for c in captures}) != 1:
                raise BankError(f'node {node_id}: repeats disagree on the node pose')
            for c in sorted(captures, key=lambda c: (c.repeat, c.frame)):
                frames.setdefault((node_id, c.repeat), []).append(c)
            # a repeat is served only when its frame 0 is usable; its other usable frames are cycled within a visit
            served = [frames[(node_id, r)][0] for r in sorted({c.repeat for c in captures}) if frames[(node_id, r)][0].frame == 0]
            if served:
                by_node[node_id] = served
        self._by_node, self._frames = by_node, frames
        self._ids = sorted(by_node)
        self._positions = np.array([by_node[i][0].position_enu for i in self._ids], float).reshape(-1, 3)
        self._yaws = np.array([by_node[i][0].yaw_rad for i in self._ids], float)
        if verify_files:
            for captures in frames.values():
                for c in captures:
                    if array_sha256(c.rgb_path) != c.rgb_sha256 or array_sha256(c.depth_path) != c.depth_sha256:
                        raise BankError(f'{c.node_id} repeat {c.repeat}: file sha256 differs from captures.jsonl')

    @property
    def node_ids(self):
        return tuple(self._ids)

    def find(self, position_enu, yaw_rad, xy_tol_m=1e-3, z_tol_m=1e-3, yaw_tol_rad=1e-3):
        """node_id whose pose equals the given pose within the tolerances, or None; two matches raise BankError."""
        if not self._ids:
            return None
        p = np.asarray(position_enu, float)
        near = ((np.hypot(self._positions[:, 0] - p[0], self._positions[:, 1] - p[1]) <= xy_tol_m)
                & (np.abs(self._positions[:, 2] - p[2]) <= z_tol_m)
                & (np.abs((self._yaws - float(yaw_rad) + math.pi) % (2 * math.pi) - math.pi) <= yaw_tol_rad))
        hits = np.flatnonzero(near)
        if len(hits) > 1:
            raise BankError(f'ambiguous node lookup at {p.tolist()} yaw {yaw_rad}: {[self._ids[i] for i in hits]}')
        return self._ids[hits[0]] if len(hits) else None

    def capture_for_visit(self, node_id, visit):
        captures = self._by_node[node_id]
        return captures[int(visit) % len(captures)]

    def repeats(self, node_id):
        """Number of usable repeats R of a served node."""
        return len(self._by_node[node_id])

    def capture_for_visit_frame(self, node_id, visit, frame):
        """Frame `frame` (cycling over the usable frames) of the repeat served at visit `visit` (visit mod R)."""
        served = self._by_node[node_id]
        frames = self._frames[(node_id, served[int(visit) % len(served)].repeat)]
        return frames[int(frame) % len(frames)]

    def detection(self, rgb_sha256):
        try:
            return self._detections[rgb_sha256]
        except KeyError:
            raise BankError(f'no frozen detector result for RGB {rgb_sha256}') from None

    def depth(self, capture):
        array = load_array(capture.depth_path)
        if array.shape != capture.depth_shape or array.dtype != np.float32:
            raise BankError(f'{capture.depth_path}: shape/dtype {array.shape}/{array.dtype} differs from the record')
        return array

    def describe(self):
        return {'bank_dir': str(self.root), 'bank_version': self.manifest.get('bank_version'), 'manifest_sha256': self.manifest_sha256,
                'scene_manifest_sha256': self.manifest.get('scene_manifest_sha256'), 'variant': self.manifest.get('variant'),
                'nodes_served': len(self._ids), 'captures_served': sum(len(v) for v in self._by_node.values()),
                'unusable_captures': len(self.unusable), 'missing_in_manifest': len(self.manifest.get('missing', [])),
                'repeats': self.manifest.get('repeats', 1), 'perturbation': self.manifest.get('perturbation'),
                'sensor_source': self.manifest.get('sensor_source'), 'detector_init': self.manifest.get('detector_init')}


def _held_repeats(bank):
    """{node_id: every repeat index the bank recorded for the node, usable or not}."""
    held = {}
    for node_id, repeat in bank._frames:
        held.setdefault(node_id, set()).add(repeat)
    for entry in bank.unusable:
        held.setdefault(entry['node_id'], set()).add(entry['repeat'])
    return held


def _repeat_offset(bank):
    """manifest['repeat_offset'] of a part: 0 when the key is absent; any value other than an integer >= 0 raises BankError."""
    value = bank.manifest.get('repeat_offset', 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BankError(f'{bank.root}: manifest repeat_offset must be an integer >= 0 (absent means 0), got {value!r}')
    return value


# Collector modes pooled_frames and pooled_frames_distinct draw repeat 0 identically; a distinct part's repeat k avoids the
# samples of repeats 0 .. k-1 (scripts/collect_observation_bank.py choose_distinct; output/mission/village_r6_banks/
# DESIGN_ADDENDUM_3_EN.md). Parts of the two modes (same samples and seed) are therefore one consistent collection only when
# every pooled_frames part holds the node at repeat 0 alone, which MultiBank checks.
POOLED_MODE_CLASS = {'pooled_frames': 'pooled_frames', 'pooled_frames_distinct': 'pooled_frames'}


def _repeat_definition(manifest):
    """Manifest fields that decide how a repeat is rendered and detected, on which every part of a merged node must agree:
    frame lags (absent, as in bank v1, means [0.]), perturbation sha256/mode/seed (not the source path), rig sensors, camera
    mount, sensor SDF sha256, and the frozen detector's config, weights sha256 and protocol (not its library versions)."""
    perturbation, detector = manifest.get('perturbation') or {}, manifest.get('detector_init') or {}
    mode = perturbation.get('mode')
    mode = POOLED_MODE_CLASS.get(mode, mode)
    return {'frame_lags_s': manifest.get('frame_lags_s', [0.]), 'perturbation': {'sha256': perturbation.get('sha256'), 'mode': mode,
                                                                                  'seed': perturbation.get('seed')},
            'rig_sensors': manifest.get('rig_sensors'), 'camera_mount': manifest.get('camera_mount'),
            'sensor_source_sha256': (manifest.get('sensor_source') or {}).get('sha256'),
            'detector': {k: detector.get(k) for k in ('config', 'weights_sha256', 'protocol')}}


class MultiBank:
    """Several ObservationBank directories of the same scene manifest and variant served as one (for example a search-node
    and an inspection-node collection). A node id served by more than one bank raises BankError (no silent precedence),
    except for a declared repeat extension (scripts/collect_observation_bank.py --repeat-offset), which is served as ONE node
    when all of these hold for the parts serving it:
      (a) the node pose (position and yaw) is identical in every part;
      (b) the repeat indices the parts recorded for the node (usable or not) are disjoint;
      (c) at least one part declares manifest['repeat_offset'] > 0 (absent means 0);
      (d) the parts agree on every repeat-defining manifest field (_repeat_definition: frame lags, perturbation sha256, mode
          and seed, rig sensors, camera mount, sensor SDF sha256, detector config, weights and protocol); the modes
          pooled_frames and pooled_frames_distinct count as one (POOLED_MODE_CLASS) only when every pooled_frames part
          holds the node at repeat 0 alone;
      (e) a part declaring repeat_offset K0 > 0 has an integer manifest 'repeats' R (absent means 1) and recorded the node
          only at repeats K0 .. K0 + R - 1.
    Every part's repeat_offset must be absent or an integer >= 0 (BankError otherwise). A part that records a node only as
    unusable captures does not serve it and is not one of its parts. A merged node serves the union of the parts' served
    repeats in repeat order: R is the size of the union, visit v serves the repeat at position v mod R, and
    capture_for_visit / capture_for_visit_frame route to the part that holds that repeat (frames cycle within the repeat as
    in ObservationBank). depth(capture) is read by the part holding the capture's (node, repeat). manifest_sha256 hashes the
    sorted part manifest sha256s, as before."""

    def __init__(self, banks):
        self.banks = tuple(banks)
        if not self.banks:
            raise BankError('MultiBank needs at least one bank')
        scene = {b.manifest.get('scene_manifest_sha256') for b in self.banks}
        variant = {b.manifest.get('variant') for b in self.banks}
        if len(scene) != 1 or len(variant) != 1:
            raise BankError(f'banks differ in scene manifest or variant: {sorted(map(str, scene))} {sorted(map(str, variant))}')
        offsets = {id(b): _repeat_offset(b) for b in self.banks}
        serving, frame_repeats = {}, {}
        for bank in self.banks:
            for node_id in bank.node_ids:
                serving.setdefault(node_id, []).append(bank)
            for node_id, repeat in bank._frames:
                frame_repeats.setdefault((id(bank), node_id), []).append(repeat)
        held, definitions = {}, {}
        # _served[node]: ((part, visit index of the repeat in that part), ...) in repeat order;
        # _part[(node, repeat)]: the part holding that repeat's usable frames (served or not)
        self._served, self._part, merged = {}, {}, []
        for node_id, parts in serving.items():
            if len(parts) > 1:
                for bank in parts:
                    if id(bank) not in held:
                        held[id(bank)], definitions[id(bank)] = _held_repeats(bank), _repeat_definition(bank.manifest)
                problems = []
                if len({(b._by_node[node_id][0].position_enu, b._by_node[node_id][0].yaw_rad) for b in parts}) != 1:
                    problems.append('node pose differs between parts')
                indices = [r for b in parts for r in held[id(b)][node_id]]
                if len(set(indices)) != len(indices):
                    problems.append(f'repeat indices {sorted({r for r in indices if indices.count(r) > 1}, key=str)} are in more than one part')
                if not any(offsets[id(b)] > 0 for b in parts):
                    problems.append('no part declares repeat_offset > 0')
                first = definitions[id(parts[0])]
                differing = sorted({key for b in parts[1:] for key, value in definitions[id(b)].items() if value != first[key]})
                if differing:
                    problems.append(f'parts differ in {", ".join(differing)}')
                # only repeat 0 is the same draw in both pooled modes: beside a pooled_frames_distinct part, a pooled_frames part
                # may hold the node at repeat 0 alone (its later repeats can reuse a sample the distinct part avoided)
                modes = {(b.manifest.get('perturbation') or {}).get('mode') for b in parts}
                if modes == set(POOLED_MODE_CLASS):
                    late = [str(b.root) for b in parts if (b.manifest.get('perturbation') or {}).get('mode') == 'pooled_frames'
                            and set(held[id(b)][node_id]) - {0}]
                    if late:
                        problems.append(f'pooled_frames part(s) {late} hold repeats other than 0 beside a pooled_frames_distinct part')
                for b in parts:
                    offset, repeats, own = offsets[id(b)], b.manifest.get('repeats', 1), held[id(b)][node_id]
                    if offset and (isinstance(repeats, bool) or not isinstance(repeats, int)
                                   or not all(isinstance(r, int) and offset <= r < offset + repeats for r in own)):
                        problems.append(f'{b.root} declares repeat_offset {offset} and repeats {repeats!r} but recorded repeats '
                                        f'{sorted(own, key=str)}')
                if problems:
                    raise BankError(f'node {node_id} is served by more than one bank ({[str(b.root) for b in parts]}) and is not a '
                                    f'declared repeat extension: {"; ".join(problems)}')
                merged.append(node_id)
            served = sorted(((b._by_node[node_id][i].repeat, i, b) for b in parts for i in range(len(b._by_node[node_id]))),
                            key=lambda item: item[0])
            self._served[node_id] = tuple((b, i) for _, i, b in served)
            self._part.update({(node_id, repeat): b for b in parts for repeat in frame_repeats[(id(b), node_id)]})
        self.merged_node_ids = tuple(sorted(merged))
        parts = sorted(b.manifest_sha256 for b in self.banks)
        self.manifest_sha256 = hashlib.sha256(''.join(parts).encode()).hexdigest()
        versions = {b.manifest.get('bank_version') for b in self.banks}
        if len(versions) != 1:
            raise BankError(f'banks differ in bank_version: {sorted(map(str, versions))}')
        self.manifest = {'bank_version': versions.pop(), 'scene_manifest_sha256': scene.pop(), 'variant': variant.pop(), 'parts': parts}
        self.unusable = [u for b in self.banks for u in b.unusable]

    @property
    def node_ids(self):
        return tuple(sorted(self._served))

    def find(self, position_enu, yaw_rad, xy_tol_m=1e-3, z_tol_m=1e-3, yaw_tol_rad=1e-3):
        """node_id matched by any part, or None; different node ids from different parts raise BankError (the parts of a
        merged node return the same id, which is not an ambiguity)."""
        hits = [n for n in (b.find(position_enu, yaw_rad, xy_tol_m, z_tol_m, yaw_tol_rad) for b in self.banks) if n is not None]
        if len(set(hits)) > 1:
            raise BankError(f'ambiguous node lookup across banks: {hits}')
        return hits[0] if hits else None

    def capture_for_visit(self, node_id, visit):
        """Frame 0 of the served repeat at position visit mod R (repeat order over all parts)."""
        served = self._served[node_id]
        bank, index = served[int(visit) % len(served)]
        return bank.capture_for_visit(node_id, index)

    def repeats(self, node_id):
        """Number of usable repeats R of a served node (the union over its parts)."""
        return len(self._served[node_id])

    def capture_for_visit_frame(self, node_id, visit, frame):
        """Frame `frame` (cycling over the usable frames of that repeat, in the part holding it) of the repeat at visit mod R."""
        served = self._served[node_id]
        bank, index = served[int(visit) % len(served)]
        return bank.capture_for_visit_frame(node_id, index, frame)

    def detection(self, rgb_sha256):
        for bank in self.banks:
            if rgb_sha256 in bank._detections:
                return bank._detections[rgb_sha256]
        raise BankError(f'no frozen detector result for RGB {rgb_sha256}')

    def depth(self, capture):
        """Depth of a capture of a served node, read and checked by the part holding its (node, repeat); that part must hold
        this exact capture (every capture capture_for_visit / capture_for_visit_frame returns does), else BankError. An
        unknown (node, repeat) raises KeyError, as an unknown node did before."""
        bank = self._part[(capture.node_id, capture.repeat)]
        if capture not in bank._frames[(capture.node_id, capture.repeat)]:
            raise BankError(f'{capture.node_id} repeat {capture.repeat} frame {capture.frame}: not a capture of {bank.root}')
        return bank.depth(capture)

    def describe(self):
        return {'bank_version': self.manifest['bank_version'], 'manifest_sha256': self.manifest_sha256, 'parts': [b.describe() for b in self.banks],
                'scene_manifest_sha256': self.manifest['scene_manifest_sha256'], 'variant': self.manifest['variant'],
                'nodes_served': len(self._served), 'merged_nodes': len(self.merged_node_ids), 'unusable_captures': len(self.unusable)}


def open_banks(bank_dirs, verify_files=False):
    """ObservationBank for one directory, MultiBank for several."""
    banks = [ObservationBank(d, verify_files=verify_files) for d in bank_dirs]
    return banks[0] if len(banks) == 1 else MultiBank(banks)
