"""Public tracker: gated nearest-neighbour association of perception candidates in world XY.

Units, frames, clock
    Metres, world ENU (association uses XY only), simulator seconds. A frame is identified by the
    update's sim_time_s, which must equal the sim_time_s (RGB image timestamp) of every candidate.

Public/private
    Inputs are perception.Candidate objects and a public in-view predicate (for example
    perception.footprint_fn from the camera model and the vehicle estimate). Track IDs 'T0001',
    'T0002', ... are issued sequentially by the tracker. Simulator object IDs, segmentation or
    evaluator matches are never inputs; snapshots pass the leakage key guard.

Association (TrackerConfig: engineering defaults, provenance engineering_default_unfrozen)
    1. Candidates are ranked canonically (score descending, then x, y, box), so input order does not
       change the tracks.
    2. Track/candidate pairs with XY distance <= gate_m are accepted greedily in order of (distance,
       track creation, candidate rank): at most one candidate per track and one track per candidate.
    3. Unassociated candidates open new tracks in canonical order.
    4. Every pre-existing unassociated track whose position satisfies in_view_fn gains one
       in_view_misses; tracks outside the camera footprint are not penalised. Occlusion is not modelled.
    Greedy nearest neighbour is not globally optimal: two people closer than the gate that are not seen
    in the same frame merge into one track (documented by test).

Estimate
    XY is the information-weighted mean (weights covariance_xy^-1); z is the weighted mean with weights
    trace(covariance_xy^-1). covariance_xy is the harmonic mean of the candidate covariances plus the
    weighted scatter of candidate XY about the mean. Successive images are NOT assumed independent, so
    the covariance does not shrink with the number of detections.

Confirmation
    hits >= confirm_min_hits and frames_with_detection >= confirm_min_frames (defaults 2 and 2).
    frames_with_detection counts distinct sim times, so reprocessing one frame cannot confirm a track.
    Counts only grow; confirmation is never withdrawn and tracks are never deleted.

Failure handling
    Time running backwards, candidates from another time, non-Candidate inputs or a non-boolean
    in_view_fn result raise TrackingError; invalid configuration raises ContractError. Nothing is
    repaired. Frames with invalid RGB must not be passed to update: they carry no evidence of absence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Tuple

import numpy as np

from .contracts import PROVENANCE, TIME_TOLERANCE_S, Contract, ContractError, _choice, _count, _nested, _real
from .contracts import Track as PublicTrack
from .leakage import assert_public_payload
from .perception import Candidate


class TrackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrackerConfig(Contract):
    gate_m: float = 3.
    confirm_min_hits: int = 2
    confirm_min_frames: int = 2
    provenance: str = 'engineering_default_unfrozen'

    def __post_init__(self):
        self._set(gate_m=_real(self.gate_m, 'gate_m', low=0, open_low=True), provenance=_choice(self.provenance, 'provenance', PROVENANCE))
        _count(self.confirm_min_hits, 'confirm_min_hits', 1)
        _count(self.confirm_min_frames, 'confirm_min_frames', 1)
        if self.confirm_min_frames > self.confirm_min_hits:
            raise ContractError('confirm_min_frames cannot exceed confirm_min_hits')


@dataclass(frozen=True)
class Track:
    """Tracker state; history holds the associated candidates in update order."""
    track_id: str
    position_enu: Tuple[float, float, float]
    covariance_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    first_seen_s: float
    last_seen_s: float
    hits: int
    frames_with_detection: int
    in_view_misses: int
    max_score: float
    history: Tuple[Candidate, ...]


@dataclass(frozen=True)
class TrackSnapshot(Contract):
    track_id: str
    position_enu: Tuple[float, float, float]
    covariance_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    first_seen_s: float
    last_seen_s: float
    hits: int
    frames_with_detection: int
    in_view_misses: int
    max_score: float
    confirmed: bool

    def __post_init__(self):
        assert_public_payload(self.to_dict())


@dataclass(frozen=True)
class UpdateReport(Contract):
    """associations: (track_id, candidate index in the caller's order), sorted by track_id."""
    sim_time_s: float
    associations: Tuple[Tuple[str, int], ...]
    new_tracks: Tuple[str, ...]
    in_view_misses: Tuple[str, ...]
    out_of_view: Tuple[str, ...]


def fuse(candidates):
    """(position_enu, covariance_xy) of associated candidates; no independence assumption (module docstring)."""
    xy = np.array([c.position_enu[:2] for c in candidates])
    info = np.linalg.inv(np.array([c.covariance_xy for c in candidates]))
    total = info.sum(axis=0)
    mean = np.linalg.solve(total, np.einsum('nij,nj->i', info, xy))
    alpha = np.trace(info, axis1=1, axis2=2)
    alpha = alpha / alpha.sum()
    z = float(alpha @ np.array([c.position_enu[2] for c in candidates]))
    delta = xy - mean
    cov = np.linalg.inv(total / len(candidates)) + np.einsum('n,ni,nj->ij', alpha, delta, delta)
    cov = (cov + cov.T) / 2
    return (float(mean[0]), float(mean[1]), z), tuple(tuple(float(v) for v in row) for row in cov)


class Tracker:
    def __init__(self, config=None):
        self.config = TrackerConfig() if config is None else _nested(TrackerConfig, config, 'config')
        self._tracks = []
        self._time = None

    @property
    def tracks(self):
        return tuple(self._tracks)

    def confirmed(self, track):
        return track.hits >= self.config.confirm_min_hits and track.frames_with_detection >= self.config.confirm_min_frames

    def update(self, candidates, sim_time_s, in_view_fn):
        t = _real(sim_time_s, 'sim_time_s', low=0)
        if self._time is not None and t < self._time - TIME_TOLERANCE_S:
            raise TrackingError(f'sim_time_s moved backwards ({t} < {self._time})')
        candidates = tuple(candidates)
        for i, c in enumerate(candidates):
            if not isinstance(c, Candidate):
                raise TrackingError(f'candidate {i}: expected perception.Candidate')
            if abs(c.sim_time_s - t) > TIME_TOLERANCE_S:
                raise TrackingError(f'candidate {i}: sim_time_s {c.sim_time_s} differs from the update time {t}')
        order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].score, candidates[i].position_enu[0],
                                                              candidates[i].position_enu[1], candidates[i].box))
        pairs = sorted((math.hypot(candidates[i].position_enu[0] - track.position_enu[0],
                                   candidates[i].position_enu[1] - track.position_enu[1]), k, rank, i)
                       for k, track in enumerate(self._tracks) for rank, i in enumerate(order))
        assigned, used = {}, set()
        for distance, k, _, i in pairs:
            if distance > self.config.gate_m:
                break
            if k not in assigned and i not in used:
                assigned[k] = i
                used.add(i)
        tracks, misses, outside = list(self._tracks), [], []  # committed only if the whole update succeeds
        for k, track in enumerate(self._tracks):
            if k in assigned:
                tracks[k] = self._hit(track, candidates[assigned[k]], t)
                continue
            inside = in_view_fn(track.position_enu)
            if not isinstance(inside, (bool, np.bool_)):
                raise TrackingError('in_view_fn must return a boolean')
            if inside:
                tracks[k] = replace(track, in_view_misses=track.in_view_misses + 1)
                misses.append(track.track_id)
            else:
                outside.append(track.track_id)
        new = []
        for i in order:
            if i not in used:
                c = candidates[i]
                tracks.append(Track(f'T{len(tracks) + 1:04d}', *fuse((c,)), t, t, 1, 1, 0, c.score, (c,)))
                new.append(tracks[-1].track_id)
        self._tracks, self._time = tracks, t
        return UpdateReport(t, tuple(sorted((tracks[k].track_id, i) for k, i in assigned.items())), tuple(new),
                            tuple(misses), tuple(outside))

    @staticmethod
    def _hit(track, candidate, t):
        history = track.history + (candidate,)
        position, covariance = fuse(history)
        new_frame = abs(t - track.last_seen_s) > TIME_TOLERANCE_S
        return replace(track, position_enu=position, covariance_xy=covariance, last_seen_s=t, hits=track.hits + 1,
                       frames_with_detection=track.frames_with_detection + int(new_frame),
                       max_score=max(track.max_score, candidate.score), history=history)

    def snapshot(self):
        """Public fields only (no candidate history), in track_id order."""
        return tuple(TrackSnapshot(t.track_id, t.position_enu, t.covariance_xy, t.first_seen_s, t.last_seen_s, t.hits,
                                   t.frames_with_detection, t.in_view_misses, t.max_score, self.confirmed(t)) for t in self._tracks)

    def public_tracks(self, sim_time_s, vertical_sigma_m, confirmed_only=True, with_score=False):
        """contracts.Track records: age_s = sim_time_s - first_seen_s, detections = hits, covariance_enu from
        covariance_xy with a declared vertical variance (the tracker estimates horizontal position only).
        confirmed_only=False also lists tentative tracks; with_score=True fills max_score (both: task flood-sar-village/1, ADR 010)."""
        t = _real(sim_time_s, 'sim_time_s', low=0)
        vz = _real(vertical_sigma_m, 'vertical_sigma_m', low=0, open_low=True) ** 2
        out = []
        for track in self._tracks:
            if confirmed_only and not self.confirmed(track):
                continue
            if t < track.last_seen_s - TIME_TOLERANCE_S:
                raise TrackingError('sim_time_s precedes the last detection of ' + track.track_id)
            (a, b), (c, d) = track.covariance_xy
            out.append(PublicTrack(track.track_id, track.position_enu, ((a, b, 0.), (c, d, 0.), (0., 0., vz)),
                                   max(0., t - track.first_seen_s), track.hits, track.max_score if with_score else None))
        return tuple(out)
