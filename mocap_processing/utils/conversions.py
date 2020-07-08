import numpy as np
import quaternion

from mocap_processing.utils import constants, utils
from scipy.spatial.transform import Rotation

import warnings

"""
Glossary:
p: position (3,)
rad: radians
deg: degrees
A: Axis angle (3,)
E: Euler angle (3,)
Q: Quaternion (4,)
R: Rotation matrix (3,3)
T: Transition matrix (4,4)

Quaternion uses the xyzw order
Rotation matrix matrix is column-wise 
"""


"""
Angle conversions
"""


def rad2deg(rad):
    """Convert from radians to degrees."""
    return rad * 180.0 / np.pi


def deg2rad(deg):
    """Convert from degrees to radians."""
    return deg * np.pi / 180.0


"""
From A to other representations
"""


def A2A(A):
    """
    The same 3D orientation could be represented by
    the two different axis-angle representatons;
    (axis, angle) and (-axis, 2pi - angle) where 
    we assume 0 <= angle <= pi. This function forces
    that it only uses an angle between 0 and 2pi.
    """

    def a2a(a):
        angle = np.linalg.norm(a)
        if angle <= constants.EPSILON:
            return a
        if angle > 2 * np.pi:
            angle = angle % 2 * np.pi
            warnings.warn("!!!Angle is larger than 2PI!!!")
        if angle > np.pi:
            return (-a / angle) * (2 * np.pi - angle)
        else:
            return a

    return utils._apply_fn_agnostic_to_vec_mat(A, a2a)


def A2E(A, order="zyx", degrees=False):
    Rotation.from_rotvec(A).as_euler(order, degrees=degrees)


def A2Q(A):
    return Rotation.from_rotvec(A).as_quat()


def A2R(A):
    return Rotation.from_rotvec(A).as_matrix()


def A2T(A):
    return Rp2T(A2R(A), constants.zero_p())


def Ax2R(theta):
    """
    Convert (axis) angle along x axis Ax to rotation matrix R
    """
    R = constants.eye_R()
    c = np.cos(theta)
    s = np.sin(theta)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def Ay2R(theta):
    """
    Convert (axis) angle along y axis Ay to rotation matrix R
    """
    R = constants.eye_R()
    c = np.cos(theta)
    s = np.sin(theta)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def Az2R(theta):
    """
    Convert (axis) angle along z axis Az to rotation matrix R
    """
    R = constants.eye_R()
    c = np.cos(theta)
    s = np.sin(theta)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


"""
From R to other representations
"""


def R2A(R):
    return Rotation.from_matrix(R).as_rotvec()


def R2E(R, order="zyx", degrees=False):
    Rotation.from_matrix(R).as_euler(order, degrees=degrees)


def R2Q(R):
    return Rotation.from_matrix(R).as_quat()


def R2R(R):
    """
    This returns valid (corrected) rotation if input
    rotations are invalid. Otherwise returns the same values.
    """
    return Rotation.from_matrix(R).as_matrix()


def R2T(R):
    return Rp2T(R, constants.zero_p())


"""
From Q to other representations
"""


def Q2A(Q):
    return Rotation.from_quat(Q).as_rotvec()


def Q2E(Q, order="zyx", degrees=False):
    return Rotation.from_quat(Q).as_euler(order, degrees=degrees)


def Q2Q(Q, op, xyzw_in=True):
    """
    This returns valid (corrected) rotation if input
    rotations are invalid. Otherwise returns the same values.
    """
    return Rotation.from_quat(Q).as_quat()


def Q2R(Q):
    return Rotation.from_quat(Q).as_matrix()


def Q2T(Q):
    return Rp2T(Q2R(Q), constants.zero_p())


"""
From T to other representations
"""


def T2p(T):
    _, p = T2Rp(T)
    return p


def T2R(T):
    R, _ = T2Rp(T)
    return R


def T2Rp(T):
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    return R, p


def T2Qp(T):
    R, p = T2Rp(T)
    Q = R2Q(R)
    return Q, p


def Ap2T(A, p):
    R = A2R(A)
    return Rp2T(R, p)


def Ep2T(E, p, order="zyx", degrees=False):
    R = E2R(E, order, degrees)
    return Rp2T(R, p)


"""
From some representations to T
"""


def Qp2T(Q, p):
    R = Q2R(Q)
    return Rp2T(R, p)


def Rp2T(R, p):
    input_shape = R.shape[:-2] if R.ndim > 2 else p.shape[:-1]
    R_flat = R.reshape((-1, 3, 3))
    p_flat = p.reshape((-1, 3))
    T = np.zeros((int(np.prod(input_shape)), 4, 4))
    T[...] = constants.eye_T()
    T[..., :3, :3] = R_flat
    T[..., :3, 3] = p_flat
    return T.reshape(list(input_shape) + [4, 4])


def p2T(p):
    return Rp2T(constants.eye_R(), np.array(p))
