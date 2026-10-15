"""Explicit coordinate conventions for the Flood-SAR mission stack.

World       ENU metres: x East, y North, z Up (Gazebo world frame).
Local       NED metres: x North, y East, z Down (PX4 local position, AirSim).
Body        FLU: x forward, y left, z up (Gazebo/ROS).
            FRD: x forward, y right, z down (PX4/AirSim).
Optical     x right, y down, z forward (pinhole, as in flight/camera_geometry.py).
Quaternion  scalar-first [w, x, y, z], Hamilton product. A parent<-child quaternion
            maps child-frame vectors into the parent frame: v_parent = R(q) @ v_child.

Only axis conventions are converted here. Origins (for example the PX4 EKF origin in
the world) and measured offsets such as the ESAR actor/vehicle NED-Z difference are
never built in: they must come from a readback and be passed explicitly.
"""
import math

import numpy as np

ENU_FROM_NED = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
NED_FROM_ENU = ENU_FROM_NED.T
FLU_FROM_FRD = np.diag([1., -1., -1.])
FRD_FROM_FLU = FLU_FROM_FRD.T
# Columns are the optical axes expressed in FLU for a forward-looking, unrotated mount.
FLU_FROM_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def vectors(value, name='vector'):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite (..., 3) values')
    return a


def vector3(value, name='vector'):
    a = vectors(value, name)
    if a.shape != (3,):
        raise ValueError(f'{name}: expected a single 3-vector')
    return a


def rotation(value, name='rotation'):
    r = np.asarray(value, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError(f'{name}: expected finite 3x3 matrix')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1., atol=1e-6):
        raise ValueError(f'{name}: not a proper rotation (reflections are rejected)')
    return r


def quat_to_matrix(q, tolerance=1e-3):
    """Unit wxyz quaternion to rotation; norms off by more than tolerance are rejected, not fixed."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('quaternion: expected finite [w, x, y, z]')
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > tolerance:
        raise ValueError('quaternion: not unit length')
    w, x, y, z = q / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(r):
    """Rotation to unit wxyz quaternion with canonical sign w >= 0."""
    r = rotation(r)
    trace = float(np.trace(r))
    i = int(np.argmax([trace, r[0, 0], r[1, 1], r[2, 2]]))
    if i == 0:
        w = math.sqrt(max(0., 1 + trace)) / 2
        q = [w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)]
    elif i == 1:
        x = math.sqrt(max(0., 1 + 2 * r[0, 0] - trace)) / 2
        q = [(r[2, 1] - r[1, 2]) / (4 * x), x, (r[0, 1] + r[1, 0]) / (4 * x), (r[0, 2] + r[2, 0]) / (4 * x)]
    elif i == 2:
        y = math.sqrt(max(0., 1 + 2 * r[1, 1] - trace)) / 2
        q = [(r[0, 2] - r[2, 0]) / (4 * y), (r[0, 1] + r[1, 0]) / (4 * y), y, (r[1, 2] + r[2, 1]) / (4 * y)]
    else:
        z = math.sqrt(max(0., 1 + 2 * r[2, 2] - trace)) / 2
        q = [(r[1, 0] - r[0, 1]) / (4 * z), (r[0, 2] + r[2, 0]) / (4 * z), (r[1, 2] + r[2, 1]) / (4 * z), z]
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def enu_to_ned(points):
    return vectors(points, 'ENU points') @ NED_FROM_ENU.T


def ned_to_enu(points):
    return vectors(points, 'NED points') @ ENU_FROM_NED.T


def ned_frd_to_enu_flu(q_ned_frd):
    """Attitude of an FRD body in NED to the same attitude of its FLU body in ENU."""
    return matrix_to_quat(ENU_FROM_NED @ quat_to_matrix(q_ned_frd) @ FRD_FROM_FLU)


def enu_flu_to_ned_frd(q_enu_flu):
    return matrix_to_quat(NED_FROM_ENU @ quat_to_matrix(q_enu_flu) @ FLU_FROM_FRD)


def wrap_angle(angle):
    if not math.isfinite(angle):
        raise ValueError('angle: expected finite radians')
    return (angle + math.pi) % (2 * math.pi) - math.pi


def enu_yaw_to_ned_yaw(yaw_enu):
    """ENU yaw is counter-clockwise from East; NED yaw is clockwise from North."""
    return wrap_angle(math.pi / 2 - yaw_enu)


def ned_yaw_to_enu_yaw(yaw_ned):
    return wrap_angle(math.pi / 2 - yaw_ned)


def compose(r_ab, t_ab, r_bc, t_bc):
    """Pose of c in a from pose of b in a and pose of c in b."""
    r_ab, r_bc = rotation(r_ab, 'r_ab'), rotation(r_bc, 'r_bc')
    return r_ab @ r_bc, r_ab @ vector3(t_bc, 't_bc') + vector3(t_ab, 't_ab')


def invert(r_ab, t_ab):
    r_ab = rotation(r_ab, 'r_ab')
    return r_ab.T, -r_ab.T @ vector3(t_ab, 't_ab')


def transform_points(r_ab, t_ab, points_b):
    return vectors(points_b, 'points') @ rotation(r_ab, 'r_ab').T + vector3(t_ab, 't_ab')


def optical_pose_in_world(q_world_from_body_flu, body_position_enu, mount_position_flu,
                          mount_flu_from_optical=FLU_FROM_OPTICAL):
    """World(ENU)<-optical rotation and optical centre from a body pose and a camera mount.

    The mount must come from calibration or the model specification with its status declared.
    """
    return compose(quat_to_matrix(q_world_from_body_flu), body_position_enu,
                   mount_flu_from_optical, mount_position_flu)
