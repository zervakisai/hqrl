"""Public perception: frozen person-detector client, pinhole camera geometry, candidate localisation.

Units, frames, clock
    Metres, radians, simulator seconds (the SensorStatus clock). World ENU; base_link FLU; the camera
    link has FLU axes and Gazebo cameras look along link +x; optical frame x right, y down, z forward
    (frames.py). Quaternions wxyz, parent<-child. Pixels are continuous edge coordinates: pixel
    (column i, row j) covers [i, i+1) x [j, j+1) and its centre is (i + 0.5, j + 0.5); detector boxes
    use the same convention.

Camera model (calibration_status specification_only until M4 calibration)
    Pinhole, square pixels, no distortion: fx = fy = width / (2 tan(hfov / 2)), cx = width / 2,
    cy = height / 2. CameraMount is a constructor parameter (never hard-coded): base_link FLU <- camera
    link from the SDF joint pose, translation plus roll/pitch/yaw with R = Rz(yaw) Ry(pitch) Rx(roll)
    (positive pitch tilts the camera down), the sensor translation inside the link (<sensor><pose>,
    zero rotation) and the link <- optical rotation (FLU_FROM_OPTICAL).

Public/private
    Inputs are RGB/depth pixels, their sim timestamps and validity, the VehicleEstimate, the sensor
    model/mount and a public static ground-height function. Simulator object IDs, segmentation, target
    poses and evaluator data are never accepted; Candidate and Rejection pass the leakage key guard.

Detector transport (protocol aerial-qrl-detector/1, scripts/person_detector_worker.py)
    One worker subprocess under the perception interpreter; JSON lines; the first line is the init
    record, checked against the frozen weights hash and config. Requests carry an absolute image path
    and the sha256 of its bytes; seq starts at 1 and must be echoed. After a timeout or protocol
    violation the client is unusable (a late reply could be matched to the next request). A well-formed
    ok:false reply is a typed DetectorRemoteError and keeps the client usable.

Localisation (LocalizationConfig: engineering defaults, provenance engineering_default_unfrozen)
    ground_ray  Ray through the box bottom centre, intersected with ground_height_fn by marching along the
                ray in 0.1 m steps to the first sample at or below the terrain, then bisecting (exact on a
                plane; a vertical face returns the face point). Rejected when the ray does not descend, the
                ground under the camera is not below it, or no ground lies within max_range_m
                ('ground_beyond_max_range'). The earlier fixed-point iteration oscillated at terrain steps
                ('ground_iteration_not_converged', kept in FLAGS for old logs only; segment_mission_v1
                run_20260915T083019Z frame 119).
    depth       DECLARED ASSUMPTION: RGB and depth share optical origin and orientation (the OakD-Lite
                SDF gives both sensors the same pose; not calibrated; enforced by comparing the mounts).
                The RGB box maps into the depth image through normalised coordinates; the median of
                valid optical-z samples (finite, inside the depth range) whose pixel centres lie in the
                mapped box is placed on the box-centre ray. When valid it is the Candidate's method;
                the ground-ray estimate is kept beside it and the horizontal disagreement is reported,
                never averaged away. Only the ray/box geometry is used, not object shape.
    covariance  First order and uncalibrated. Angular sigma = hypot(pixel_sigma_px / fx,
                angular_sigma_rad). ground_ray: across-view (r sigma)^2, along-view (r sigma / sin g)^2 +
                (ground_height_sigma_m / tan g)^2 with grazing angle g. depth: depth_sigma_m along the
                ray and r sigma across it, projected to XY. Both add position_sigma_m^2 in XY.
                Detector scores are not probabilities.

Missing data
    Invalid RGB -> status 'rgb_invalid' and no detections (supplying detector output is an error; an
    older frame is never reused). Invalid vehicle state -> 'vehicle_state_invalid', detections are
    rejected unlocalised. Invalid or absent depth -> 'depth_invalid', ground_ray only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import time
from typing import Optional, Tuple

import numpy as np

from .backend import SensorFrame
from .contracts import (PROVENANCE, Contract, ContractError, SensorModel, SensorStatus, VehicleEstimate, _choice, _count,
                        _flag, _list, _nested, _nested_seq, _optional_real, _real, _sha256, _unique, _vec, check_sensor_times)
from .contracts import loads as strict_loads
from .frames import FLU_FROM_OPTICAL, compose, quat_to_matrix, rotation, vector3, vectors
from .leakage import assert_public_payload

PROTOCOL = 'aerial-qrl-detector/1'
FROZEN_WEIGHTS_SHA256 = '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'
FROZEN_DETECTOR_CONFIG = {'imgsz': 640, 'rect': False, 'conf': .25, 'iou': .7, 'max_det': 100, 'classes': [0], 'device': 'cpu',
                          'quantize': 32, 'augment': False, 'verbose': False, 'save': False, 'batch': 1}
INIT_KEYS = frozenset({'protocol', 'kind', 'versions', 'weights_sha256', 'config', 'runtime', 'declared_difference', 'input_contract'})
OK_KEYS = frozenset({'seq', 'ok', 'input_sha256', 'width', 'height', 'predictions', 'predict_wall_s'})
ERROR_KEYS = frozenset({'seq', 'ok', 'error'})
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PYTHON = Path.home() / '.local/share/aerial-qrl/perception/python/bin/python'
DEFAULT_WORKER = ROOT / 'scripts' / 'person_detector_worker.py'
MAX_LINE_BYTES = 1024 * 1024
METHODS = ('ground_ray', 'depth')
FLAGS = ('ray_not_descending', 'ground_not_below_camera', 'ground_iteration_not_converged', 'ground_not_found_within_range', 'ground_beyond_max_range',
         'depth_invalid', 'depth_insufficient_valid_pixels', 'depth_beyond_max_range')
STATUSES = ('rgb_invalid', 'vehicle_state_invalid', 'depth_invalid')
RAW_DTYPES = {'uint8': '<u1', 'float32': '<f4'}
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class PerceptionError(RuntimeError):
    pass


class DetectorError(PerceptionError):
    pass


class DetectorTimeout(DetectorError):
    pass


class DetectorProtocolError(DetectorError):
    pass


class DetectorRemoteError(DetectorError):
    pass


# --- Detector output and transport ------------------------------------------------------

@dataclass(frozen=True)
class Detection(Contract):
    """One person box in continuous pixel-edge coordinates of its image; score is not a probability."""
    xyxy: Tuple[float, float, float, float]
    score: float

    def __post_init__(self):
        box = _vec(self.xyxy, 'xyxy', 4)
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ContractError('xyxy: expected x0 < x1 and y0 < y1')
        self._set(xyxy=box, score=_real(self.score, 'score', low=0, high=1))


@dataclass(frozen=True)
class DetectorResult(Contract):
    """Worker reply for one image; predict_wall_s and round_trip_s are host wall times (logs only)."""
    seq: int
    input_sha256: str
    width: int
    height: int
    detections: Tuple[Detection, ...]
    predict_wall_s: float
    round_trip_s: float

    def __post_init__(self):
        for key, low in (('seq', 1), ('width', 1), ('height', 1)):
            _count(getattr(self, key), key, low)
        detections = _nested_seq(Detection, self.detections, 'detections')
        if any(d.xyxy[0] < 0 or d.xyxy[1] < 0 or d.xyxy[2] > self.width or d.xyxy[3] > self.height for d in detections):
            raise ContractError('detections: box outside the image')
        self._set(input_sha256=_sha256(self.input_sha256, 'input_sha256'), detections=detections,
                  predict_wall_s=_real(self.predict_wall_s, 'predict_wall_s', low=0),
                  round_trip_s=_real(self.round_trip_s, 'round_trip_s', low=0))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_rgb_npy(rgb, path):
    """Write H x W x 3 uint8 RGB pixels as .npy (exclusive create); returns the sha256 of the file bytes."""
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError('expected H x W x 3 uint8 RGB pixels')
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(a), allow_pickle=False)
    data = buffer.getvalue()
    with open(path, 'xb') as f:  # never overwrite an earlier frame
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def load_raw_array(path, dtype, shape):
    """Raw C-order little-endian bridge array of known dtype/shape (the backend verified its sha256)."""
    data = Path(path).read_bytes()
    expected = int(np.prod(shape)) * np.dtype(RAW_DTYPES[dtype]).itemsize
    if len(data) != expected:
        raise PerceptionError(f'{path}: {len(data)} bytes, expected {expected} for {dtype} {tuple(shape)}')
    return np.frombuffer(data, dtype=RAW_DTYPES[dtype]).reshape(shape)


class DetectorClient:
    """Owns exactly one worker process; close() never touches any other process."""

    def __init__(self, python=DEFAULT_PYTHON, worker=DEFAULT_WORKER, worker_args=(), timeout_s=60., init_timeout_s=300.,
                 expected_weights_sha256=FROZEN_WEIGHTS_SHA256, expected_config=FROZEN_DETECTOR_CONFIG, python_flags=('-I',),
                 cwd=None, env=None, stderr_path=None):
        for name, value in (('timeout_s', timeout_s), ('init_timeout_s', init_timeout_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
                raise ValueError(f'{name} must be positive and finite')
        self.timeout_s = float(timeout_s)
        self.argv = [str(python), *python_flags, str(worker), *map(str, worker_args)]
        self._stderr = open(stderr_path, 'ab') if stderr_path else subprocess.DEVNULL
        try:
            self._proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                          cwd=cwd, env=env, start_new_session=True)
        except BaseException:
            if self._stderr is not subprocess.DEVNULL:
                self._stderr.close()
            raise
        self._seq, self._buffer, self._closed = 0, b'', False
        self.broken = None
        try:
            self.init_record = self._read_init(init_timeout_s, expected_weights_sha256, expected_config)
        except BaseException:
            self.close(timeout_s=5.)
            raise

    @property
    def pid(self):
        return self._proc.pid

    def poll(self):
        return self._proc.poll()

    def _violation(self, message):
        self.broken = 'protocol violation'
        raise DetectorProtocolError(message)

    def _read_init(self, timeout_s, weights_sha256, config):
        message = self._message(self._readline(timeout_s, 'init record'))
        if set(message) != INIT_KEYS or message['protocol'] != PROTOCOL or message['kind'] != 'init':
            self._violation(f'init record needs exactly {sorted(INIT_KEYS)} with protocol {PROTOCOL}')
        if weights_sha256 is not None and message['weights_sha256'] != weights_sha256:
            self._violation(f"worker weights {message['weights_sha256']!r} differ from expected {weights_sha256}")
        if config is not None and message['config'] != config:
            self._violation('worker detector config differs from the frozen config')
        return message

    def detect(self, image_path, sha256=None, timeout_s=None):
        """Detect persons in an .npy/.png file; sha256 (of the file bytes) is computed when not given."""
        if self.broken:
            raise DetectorError('detector client unusable after ' + self.broken)
        path = os.path.abspath(os.fspath(image_path))
        digest = file_sha256(path) if sha256 is None else sha256
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise ValueError('sha256 must be lowercase hex')
        self._seq += 1
        seq = self._seq
        line = json.dumps({'seq': seq, 'image_path': path, 'sha256': digest}, sort_keys=True, separators=(',', ':')) + '\n'
        sent = time.monotonic()
        try:
            self._proc.stdin.write(line.encode('utf-8'))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.broken = 'write failure'
            raise DetectorError(f'cannot write request: {exc}') from exc
        message = self._message(self._readline(self.timeout_s if timeout_s is None else timeout_s, f'reply to seq {seq}'))
        round_trip = time.monotonic() - sent
        if isinstance(message.get('seq'), bool) or message.get('seq') != seq:
            self._violation(f"sequence mismatch: expected {seq}, got {message.get('seq')!r}")
        if message.get('ok') is False and set(message) == ERROR_KEYS and isinstance(message['error'], str):
            raise DetectorRemoteError(message['error'])
        if message.get('ok') is not True or set(message) != OK_KEYS:
            self._violation(f'malformed reply fields {sorted(message)}')
        try:
            if message['input_sha256'] != digest:
                raise ContractError('input_sha256 differs from the request')
            return DetectorResult(seq, message['input_sha256'], message['width'], message['height'], message['predictions'],
                                  message['predict_wall_s'], round_trip)
        except (ContractError, TypeError) as exc:
            self._violation(f'invalid reply: {exc}')

    def _message(self, raw):
        try:
            message = strict_loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, ValueError) as exc:
            self._violation(f'invalid JSON line: {exc}')
        if not isinstance(message, dict):
            self._violation('line must be a JSON object')
        return message

    def _readline(self, timeout_s, what):
        deadline = time.monotonic() + timeout_s
        fd = self._proc.stdout.fileno()
        while b'\n' not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.broken = 'timeout'
                raise DetectorTimeout(f'no {what} within {timeout_s:.3f} s')
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                self.broken = 'worker exit'
                raise DetectorError(f'detector worker exited (code {self._proc.poll()}) before {what}')
            self._buffer += chunk
            if len(self._buffer) > MAX_LINE_BYTES:
                self._violation('line exceeds limit')
        line, _, self._buffer = self._buffer.partition(b'\n')
        return line

    def close(self, timeout_s=5.):
        """EOF on stdin first (the worker exits cleanly), then SIGTERM, then SIGKILL. Idempotent."""
        if self._closed:
            return self._proc.returncode
        self._closed = True
        self.broken = self.broken or 'close'
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc.stdout.close()
        if self._stderr is not subprocess.DEVNULL:
            self._stderr.close()
        return self._proc.returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- Camera geometry --------------------------------------------------------------------

def rpy_matrix(roll, pitch, yaw):
    """Gazebo/SDF pose rotation parent <- child: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return (np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]]) @ np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
            @ np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]]))


@dataclass(frozen=True)
class CameraMount(Contract):
    """base_link FLU <- camera link (SDF joint pose) plus the sensor translation inside the link."""
    translation_flu: Tuple[float, float, float]
    rpy_rad: Tuple[float, float, float]
    sensor_offset_link: Tuple[float, float, float]
    calibration_status: str
    source: str

    def __post_init__(self):
        self._set(translation_flu=_vec(self.translation_flu, 'translation_flu'), rpy_rad=_vec(self.rpy_rad, 'rpy_rad'),
                  sensor_offset_link=_vec(self.sensor_offset_link, 'sensor_offset_link'),
                  calibration_status=_choice(self.calibration_status, 'calibration_status', ('specification_only', 'calibrated')))
        if not isinstance(self.source, str) or not self.source:
            raise ContractError('source: required provenance path or URL')


@dataclass(frozen=True, eq=False)
class CameraPose:
    """World(ENU) <- optical rotation and optical centre in ENU metres."""
    rotation: np.ndarray
    position_enu: np.ndarray


class CameraModel:
    """Pinhole camera rigidly mounted on base_link (conventions in the module docstring)."""

    def __init__(self, width, height, hfov_rad, mount, link_from_optical=FLU_FROM_OPTICAL):
        self.width, self.height = _count(width, 'width', 1), _count(height, 'height', 1)
        self.hfov_rad = _real(hfov_rad, 'hfov_rad', low=0, high=math.pi, open_low=True, open_high=True)
        self.fx = self.fy = width / (2 * math.tan(self.hfov_rad / 2))
        self.cx, self.cy = width / 2, height / 2
        self.mount = _nested(CameraMount, mount, 'mount')
        link = rpy_matrix(*self.mount.rpy_rad)
        self.body_from_optical = link @ rotation(link_from_optical, 'link_from_optical')
        self.optical_origin_body = np.asarray(self.mount.translation_flu) + link @ np.asarray(self.mount.sensor_offset_link)

    @classmethod
    def from_sensor_model(cls, sensor_model, stream, mount):
        model = _nested(SensorModel, sensor_model, 'sensor_model')
        stream = _choice(stream, 'stream', ('rgb', 'depth'))
        return cls(getattr(model, f'{stream}_width'), getattr(model, f'{stream}_height'), getattr(model, f'{stream}_hfov_rad'), mount)

    @property
    def intrinsics(self):
        return np.array([[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]])

    def pose(self, vehicle):
        vehicle = _nested(VehicleEstimate, vehicle, 'vehicle')
        r, t = compose(quat_to_matrix(vehicle.attitude_wxyz), vehicle.position_enu, self.body_from_optical, self.optical_origin_body)
        return CameraPose(r, t)

    def same_optical_frame(self, other, atol=1e-9):
        return bool(np.allclose(self.body_from_optical, other.body_from_optical, atol=atol, rtol=0)
                    and np.allclose(self.optical_origin_body, other.optical_origin_body, atol=atol, rtol=0))

    def normalized(self, uv):
        uv = np.asarray(uv, dtype=float)
        if uv.ndim == 0 or uv.shape[-1] != 2 or not np.isfinite(uv).all():
            raise ValueError('pixels: expected finite (..., 2) values')
        return (uv - [self.cx, self.cy]) / [self.fx, self.fy]

    def pixel_from_normalized(self, xy):
        return np.asarray(xy, dtype=float) * [self.fx, self.fy] + [self.cx, self.cy]

    def pixel_to_ray(self, uv, pose):
        """Unit ENU directions through pixels (..., 2)."""
        n = self.normalized(uv)
        rays = np.concatenate([n, np.ones(n.shape[:-1] + (1,))], axis=-1)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True) @ pose.rotation.T

    def world_to_pixel(self, points_enu, pose):
        """(uv, optical_z) for ENU points (..., 3); uv is NaN where optical z <= 0 (not in front)."""
        p = vectors(points_enu, 'points_enu')
        local = (p.reshape(-1, 3) - pose.position_enu) @ pose.rotation
        z, uv = local[:, 2], np.full((len(local), 2), np.nan)
        front = z > 0
        uv[front] = local[front, :2] / z[front, None] * [self.fx, self.fy] + [self.cx, self.cy]
        return uv.reshape(p.shape[:-1] + (2,)), z.reshape(p.shape[:-1])

    def in_view(self, points_enu, pose, max_range_m=None, margin_px=0.):
        """Inside the image (with margin) in front of the camera and within max_range_m; occlusion is not modelled."""
        uv, z = self.world_to_pixel(points_enu, pose)
        with np.errstate(invalid='ignore'):
            inside = (z > 0) & (uv[..., 0] >= margin_px) & (uv[..., 0] <= self.width - margin_px) \
                & (uv[..., 1] >= margin_px) & (uv[..., 1] <= self.height - margin_px)
        if max_range_m is not None:
            inside &= np.linalg.norm(vectors(points_enu) - pose.position_enu, axis=-1) <= max_range_m
        return inside


def footprint_fn(camera, pose, max_range_m=None, margin_px=0.):
    """Public in-view predicate for Tracker.update from the camera model and the vehicle estimate."""
    return lambda position_enu: bool(camera.in_view(np.asarray(position_enu, float), pose, max_range_m, margin_px))


# --- Localisation -----------------------------------------------------------------------

@dataclass(frozen=True)
class LocalizationConfig(Contract):
    pixel_sigma_px: float = 2.
    angular_sigma_rad: float = .005
    position_sigma_m: float = .1
    ground_height_sigma_m: float = .1
    depth_sigma_m: float = .1
    max_range_m: float = 150.
    min_depth_pixels: int = 3
    ground_tolerance_m: float = 1e-4
    max_ground_iterations: int = 50
    provenance: str = 'engineering_default_unfrozen'

    def __post_init__(self):
        positive = dict(low=0, open_low=True)
        self._set(pixel_sigma_px=_real(self.pixel_sigma_px, 'pixel_sigma_px', **positive),
                  angular_sigma_rad=_real(self.angular_sigma_rad, 'angular_sigma_rad', low=0),
                  position_sigma_m=_real(self.position_sigma_m, 'position_sigma_m', low=0),
                  ground_height_sigma_m=_real(self.ground_height_sigma_m, 'ground_height_sigma_m', low=0),
                  depth_sigma_m=_real(self.depth_sigma_m, 'depth_sigma_m', **positive),
                  max_range_m=_real(self.max_range_m, 'max_range_m', **positive),
                  ground_tolerance_m=_real(self.ground_tolerance_m, 'ground_tolerance_m', **positive),
                  provenance=_choice(self.provenance, 'provenance', PROVENANCE))
        _count(self.min_depth_pixels, 'min_depth_pixels', 1)
        _count(self.max_ground_iterations, 'max_ground_iterations', 1)


DEFAULT_LOCALIZATION = LocalizationConfig()


def _box(value):
    box = _vec(value, 'box', 4)
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ContractError('box: expected x0 < x1 and y0 < y1')
    return box


@dataclass(frozen=True)
class Candidate(Contract):
    """Localised public detection. position_enu is the estimate named by method (depth preferred when valid)."""
    position_enu: Tuple[float, float, float]
    covariance_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    score: float
    box: Tuple[float, float, float, float]
    sim_time_s: float
    method: str
    range_m: float
    ground_valid: bool
    depth_valid: bool
    ground_position_enu: Optional[Tuple[float, float, float]]
    depth_position_enu: Optional[Tuple[float, float, float]]
    disagreement_xy_m: Optional[float]
    flags: Tuple[str, ...]

    def __post_init__(self):
        cov = tuple(_vec(r, f'covariance_xy[{i}]', 2) for i, r in enumerate(_list(self.covariance_xy, 'covariance_xy')))
        c = np.asarray(cov)
        if c.shape != (2, 2) or not np.allclose(c, c.T, rtol=1e-12, atol=1e-12) or np.linalg.eigvalsh(c).min() <= 0:
            raise ContractError('covariance_xy: expected symmetric positive definite 2x2')
        ground = None if self.ground_position_enu is None else _vec(self.ground_position_enu, 'ground_position_enu')
        depth = None if self.depth_position_enu is None else _vec(self.depth_position_enu, 'depth_position_enu')
        flags = tuple(_choice(f, 'flags', FLAGS) for f in _list(self.flags, 'flags'))
        _unique(flags, 'flags')
        self._set(position_enu=_vec(self.position_enu, 'position_enu'), covariance_xy=cov,
                  score=_real(self.score, 'score', low=0, high=1), box=_box(self.box),
                  sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0), method=_choice(self.method, 'method', METHODS),
                  range_m=_real(self.range_m, 'range_m', low=0, open_low=True), ground_valid=_flag(self.ground_valid, 'ground_valid'),
                  depth_valid=_flag(self.depth_valid, 'depth_valid'), ground_position_enu=ground, depth_position_enu=depth,
                  disagreement_xy_m=_optional_real(self.disagreement_xy_m, 'disagreement_xy_m', low=0), flags=flags)
        if self.ground_valid != (ground is not None) or self.depth_valid != (depth is not None):
            raise ContractError('ground_valid/depth_valid must match the estimates present')
        if (self.disagreement_xy_m is None) != (ground is None or depth is None):
            raise ContractError('disagreement_xy_m is required exactly when both estimates exist')
        if self.position_enu != (depth if self.depth_valid else ground) or self.method != ('depth' if self.depth_valid else 'ground_ray'):
            raise ContractError('method/position_enu must be the depth estimate when valid, else the ground-ray estimate')
        assert_public_payload(self.to_dict())


@dataclass(frozen=True)
class Rejection(Contract):
    """Detection that produced no position estimate, with the reasons (FLAGS or vehicle_state_invalid)."""
    box: Tuple[float, float, float, float]
    score: float
    sim_time_s: float
    reasons: Tuple[str, ...]

    def __post_init__(self):
        reasons = tuple(_choice(r, 'reasons', FLAGS + ('vehicle_state_invalid',)) for r in _list(self.reasons, 'reasons'))
        if not reasons:
            raise ContractError('reasons: at least one required')
        _unique(reasons, 'reasons')
        self._set(box=_box(self.box), score=_real(self.score, 'score', low=0, high=1),
                  sim_time_s=_real(self.sim_time_s, 'sim_time_s', low=0), reasons=reasons)
        assert_public_payload(self.to_dict())


def _height(ground_height_fn, x, y):
    value = ground_height_fn(float(x), float(y))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise PerceptionError('ground_height_fn must return a finite height in metres')
    return float(value)


GROUND_MARCH_STEP_M = .1
GROUND_MAX_MARCH_M = 1000.


def intersect_ground(origin_enu, direction_enu, ground_height_fn, tolerance_m=1e-4, max_iterations=50,
                     step_m=GROUND_MARCH_STEP_M, max_distance_m=GROUND_MAX_MARCH_M):
    """First ray/terrain intersection: (point_enu, slant_range_m, None) or (None, None, reason).

    Marches along the ray in step_m increments to the first sample at or below the terrain, then bisects that interval
    (at most max_iterations halvings, down to tolerance_m of slant range). When the terrain height is the same at the
    bisected point and on the plane crossing inside the interval, the exact plane intersection is returned; a crossing on
    a vertical face returns the face point. Terrain features narrower than step_m along the ray can be missed (declared).
    Replaces a fixed-point iteration that alternated between terrain levels at height steps and rejected a correct
    detection (segment_mission_v1/run_20260915T083019Z frame 119: channel bed -2 m <-> abutment +1.8 m, shelf -0.5 m hit).
    """
    o, d = vector3(origin_enu, 'origin_enu'), vector3(direction_enu, 'direction_enu')
    if not np.linalg.norm(d) > 0:
        raise ValueError('direction_enu: zero vector')
    d = d / np.linalg.norm(d)
    if d[2] >= -1e-12:
        return None, None, 'ray_not_descending'
    if _height(ground_height_fn, o[0], o[1]) >= o[2]:
        return None, None, 'ground_not_below_camera'

    def clearance(s):
        p = o + s * d
        return p[2] - _height(ground_height_fn, p[0], p[1])

    low, s = 0., step_m
    while s <= max_distance_m:
        if clearance(s) <= 0.:
            high = s
            for _ in range(max(1, int(max_iterations))):
                if high - low <= tolerance_m:
                    break
                mid = (low + high) / 2
                low, high = (low, mid) if clearance(mid) <= 0. else (mid, high)
            level = _height(ground_height_fn, *(o + high * d)[:2])
            plane = (level - o[2]) / d[2]
            if low - tolerance_m <= plane <= high + tolerance_m and abs(_height(ground_height_fn, *(o + plane * d)[:2]) - level) <= 1e-9:
                high = plane
            return o + high * d, float(high), None
        low, s = s, s + step_m
    return None, None, 'ground_not_found_within_range'


def ground_ray_covariance_xy(direction_enu, range_m, sigma_rad, config=DEFAULT_LOCALIZATION):
    d = vector3(direction_enu, 'direction_enu') / np.linalg.norm(direction_enu)
    horizontal, sin_g = math.hypot(d[0], d[1]), -d[2]
    along_axis = d[:2] / horizontal if horizontal > 1e-12 else np.array([1., 0.])
    across_axis = np.array([-along_axis[1], along_axis[0]])
    along = (range_m * sigma_rad / sin_g) ** 2 + (config.ground_height_sigma_m * horizontal / sin_g) ** 2
    across = (range_m * sigma_rad) ** 2
    return along * np.outer(along_axis, along_axis) + across * np.outer(across_axis, across_axis) + config.position_sigma_m ** 2 * np.eye(2)


def depth_covariance_xy(direction_enu, range_m, sigma_rad, slant_sigma_m, config=DEFAULT_LOCALIZATION):
    d = vector3(direction_enu, 'direction_enu') / np.linalg.norm(direction_enu)
    cov = slant_sigma_m ** 2 * np.outer(d, d) + (range_m * sigma_rad) ** 2 * (np.eye(3) - np.outer(d, d))
    return cov[:2, :2] + config.position_sigma_m ** 2 * np.eye(2)


def depth_median_in_box(depth, box, rgb_camera, depth_camera, depth_range_m, min_pixels=3):
    """Median optical z (m) of valid depth samples whose pixel centres lie in the RGB box mapped into the depth image.

    DECLARED ASSUMPTION: RGB and depth share optical origin and orientation (checked by the caller). A
    sample is valid if finite and inside depth_range_m; NaN/inf/out-of-range samples are ignored. Returns
    None with fewer than min_pixels valid samples.
    """
    d = np.asarray(depth)
    if d.shape != (depth_camera.height, depth_camera.width) or not np.issubdtype(d.dtype, np.floating):
        raise PerceptionError('depth array shape/dtype does not match the depth camera')
    low, high = _vec(depth_range_m, 'depth_range_m', 2)
    (u0, v0), (u1, v1) = depth_camera.pixel_from_normalized(rgb_camera.normalized([box[:2], box[2:]]))
    cols, rows = np.arange(depth_camera.width) + .5, np.arange(depth_camera.height) + .5
    region = d[np.ix_((rows >= v0) & (rows < v1), (cols >= u0) & (cols < u1))]
    with np.errstate(invalid='ignore'):
        valid = region[np.isfinite(region) & (region >= low) & (region <= high)]
    return float(np.median(valid)) if valid.size >= min_pixels else None


def localize_detection(detection, rgb_camera, pose, ground_height_fn, sim_time_s, depth=None, depth_camera=None,
                       depth_range_m=None, config=None):
    """Candidate or Rejection for one detection. depth=None means no valid depth ('depth_invalid')."""
    config = DEFAULT_LOCALIZATION if config is None else _nested(LocalizationConfig, config, 'config')
    detection = _nested(Detection, detection, 'detection')
    t = _real(sim_time_s, 'sim_time_s', low=0)
    x0, y0, x1, y1 = detection.xyxy
    sigma = math.hypot(config.pixel_sigma_px / rgb_camera.fx, config.angular_sigma_rad)
    flags = []
    ray = rgb_camera.pixel_to_ray([(x0 + x1) / 2, y1], pose)
    ground, ground_range, reason = intersect_ground(pose.position_enu, ray, ground_height_fn, config.ground_tolerance_m,
                                                    config.max_ground_iterations, max_distance_m=config.max_range_m + GROUND_MARCH_STEP_M)
    if reason == 'ground_not_found_within_range' or (ground is not None and ground_range > config.max_range_m):
        ground, reason = None, 'ground_beyond_max_range'  # nothing within the configured range: same public reason as before
    if reason:
        flags.append(reason)
    depth_point = None
    if depth is None:
        flags.append('depth_invalid')
    else:
        if depth_camera is None or depth_range_m is None:
            raise PerceptionError('depth requires depth_camera and depth_range_m')
        if not rgb_camera.same_optical_frame(depth_camera):
            raise PerceptionError('RGB and depth optical frames differ; the co-located assumption does not hold')
        z = depth_median_in_box(depth, detection.xyxy, rgb_camera, depth_camera, depth_range_m, config.min_depth_pixels)
        optical = np.array([*rgb_camera.normalized([(x0 + x1) / 2, (y0 + y1) / 2]), 1.])
        if z is None:
            flags.append('depth_insufficient_valid_pixels')
        elif z * np.linalg.norm(optical) > config.max_range_m:
            flags.append('depth_beyond_max_range')
        else:
            depth_range = float(z * np.linalg.norm(optical))
            depth_point = pose.rotation @ (optical * z) + pose.position_enu
    if ground is None and depth_point is None:
        return Rejection(detection.xyxy, detection.score, t, tuple(flags))
    if depth_point is not None:
        cov = depth_covariance_xy(pose.rotation @ optical, depth_range, sigma, config.depth_sigma_m * np.linalg.norm(optical), config)
        position, method, range_m = depth_point, 'depth', depth_range
    else:
        cov = ground_ray_covariance_xy(ray, ground_range, sigma, config)
        position, method, range_m = ground, 'ground_ray', ground_range
    cov = (cov + cov.T) / 2
    disagreement = None if ground is None or depth_point is None else float(np.hypot(*(depth_point[:2] - ground[:2])))
    return Candidate(tuple(position), tuple(map(tuple, cov)), detection.score, detection.xyxy, t, method, range_m,
                     ground is not None, depth_point is not None, None if ground is None else tuple(ground),
                     None if depth_point is None else tuple(depth_point), disagreement, tuple(flags))


# --- Frame-level missing-data semantics -------------------------------------------------

@dataclass(frozen=True)
class FramePerception(Contract):
    """Perception of one frame. image_time_s is the RGB timestamp (None when RGB is invalid)."""
    frame_sim_time_s: float
    image_time_s: Optional[float]
    status: Tuple[str, ...]
    detections: int
    candidates: Tuple[Candidate, ...]
    rejections: Tuple[Rejection, ...]

    def __post_init__(self):
        status = tuple(_choice(s, 'status', STATUSES) for s in _list(self.status, 'status'))
        _unique(status, 'status')
        self._set(frame_sim_time_s=_real(self.frame_sim_time_s, 'frame_sim_time_s', low=0),
                  image_time_s=_optional_real(self.image_time_s, 'image_time_s', low=0), status=status,
                  candidates=_nested_seq(Candidate, self.candidates, 'candidates'),
                  rejections=_nested_seq(Rejection, self.rejections, 'rejections'))
        _count(self.detections, 'detections')
        if self.detections != len(self.candidates) + len(self.rejections):
            raise ContractError('detections must equal candidates + rejections')
        if ('rgb_invalid' in status) != (self.image_time_s is None) or ('rgb_invalid' in status and self.detections):
            raise ContractError('rgb_invalid frames have no image time and no detections')


def perceive(frame_sim_time_s, vehicle, sensors, detector_result, rgb_camera, ground_height_fn, depth=None,
             depth_camera=None, depth_range_m=None, config=None):
    """Apply the missing-data rules and localise every detection of one frame."""
    sensors = _nested_seq(SensorStatus, sensors, 'sensors')
    check_sensor_times(frame_sim_time_s, sensors)
    by_stream = {s.stream: s for s in sensors}
    if set(by_stream) != {'rgb', 'depth', 'vehicle_state'}:
        raise ContractError('sensors: rgb, depth and vehicle_state statuses required')
    rgb, depth_status, state = by_stream['rgb'], by_stream['depth'], by_stream['vehicle_state']
    if depth is not None and not depth_status.valid:
        raise PerceptionError('depth payload given for a depth sample reported invalid')
    status = tuple(name for name, bad in (('rgb_invalid', not rgb.valid), ('vehicle_state_invalid', not state.valid),
                                          ('depth_invalid', depth is None)) if bad)
    if not rgb.valid:
        if detector_result is not None:
            raise PerceptionError('detector output given for an invalid RGB sample; older frames are never reused')
        return FramePerception(frame_sim_time_s, None, status, 0, (), ())
    if not isinstance(detector_result, DetectorResult):
        raise PerceptionError('a valid RGB sample requires the DetectorResult of that image')
    if (detector_result.width, detector_result.height) != (rgb_camera.width, rgb_camera.height):
        raise PerceptionError('detector image size differs from the RGB camera model')
    t = rgb.timestamp_s
    if not state.valid:
        rejections = tuple(Rejection(d.xyxy, d.score, t, ('vehicle_state_invalid',)) for d in detector_result.detections)
        return FramePerception(frame_sim_time_s, t, status, len(rejections), (), rejections)
    pose = rgb_camera.pose(vehicle)
    results = [localize_detection(d, rgb_camera, pose, ground_height_fn, t, depth, depth_camera, depth_range_m, config)
               for d in detector_result.detections]
    return FramePerception(frame_sim_time_s, t, status, len(results), tuple(r for r in results if isinstance(r, Candidate)),
                           tuple(r for r in results if isinstance(r, Rejection)))


def perceive_sensor_frame(frame, detector, work_dir, rgb_camera, ground_height_fn, depth_camera=None, depth_range_m=None, config=None):
    """Backend glue: raw bridge RGB -> .npy in work_dir (exclusive create) -> detector -> perceive.

    Returns (FramePerception, DetectorResult or None). The backend verified the raw-array SHA256 before it
    built the frame; the .npy written here is hashed again and verified by the worker.
    """
    frame = _nested(SensorFrame, frame, 'frame')
    result = None
    if frame.rgb_path is not None:
        pixels = load_raw_array(frame.rgb_path, 'uint8', (rgb_camera.height, rgb_camera.width, 3))
        target = Path(work_dir) / f'{Path(frame.rgb_path).stem}.npy'
        result = detector.detect(target, write_rgb_npy(pixels, target))
    depth = None
    if frame.depth_path is not None and depth_camera is not None:
        depth = load_raw_array(frame.depth_path, 'float32', (depth_camera.height, depth_camera.width))
    return perceive(frame.sim_time_s, frame.vehicle, frame.sensors, result, rgb_camera, ground_height_fn, depth, depth_camera,
                    depth_range_m, config), result
