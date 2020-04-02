import cv2
import numpy as np

from mocap_processing.utils import constants
from mocap_processing.utils import utils


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
"""
# TODO: Add batched input support to all conversion methods


def rad2deg(rad):
    """Convert from radians to degrees."""
    return rad * 180.0 / np.pi


def deg2rad(deg):
    """Convert from degrees to radians."""
    return deg * np.pi / 180.0


def A2R(A):
    # TODO: Write batched version of this method
    input_shape = A.shape
    A_flat = A.reshape((-1, 3))
    R = []
    for A in A_flat:
        theta = np.linalg.norm(A)
        if theta == 0:
            return constants.eye_R
        # normalize axes
        A = A / theta
        x, y, z = A
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.array(
            [
                [c + (1.0 - c) * x * x, (1.0 - c) * x * y - s * z, (1 - c) * x * z + s * y],
                [
                    (1.0 - c) * x * y + s * z,
                    c + (1.0 - c) * y * y,
                    (1.0 - c) * y * z - s * x,
                ],
                [
                    (1.0 - c) * z * x - s * y,
                    (1.0 - c) * z * y + s * x,
                    c + (1.0 - c) * z * z,
                ],
            ]
        )
        R.append(r)
    return np.array(R).reshape(list(input_shape[:-1]) + [3, 3])


def R2A(R):
    """
    Adopted from https://github.com/eth-ait/spl/blob/master/common/
    conversions.py#L172
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        R: A np array of shape (..., 3, 3)
    Returns:
        A np array of shape (..., 3)
    """
    assert (
        R.shape[-1] == 3 and R.shape[-2] == 3 and len(R.shape) >= 3
    ), "Invalid input dimension"
    orig_shape = R.shape[:-2]
    rots = np.reshape(R, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, orig_shape + (3,))


def R2E(R):
    """
    Adopted from https://github.com/eth-ait/spl/blob/master/common/
    conversions.py#L76
    Converts rotation matrices to euler angles. This is an adaptation of
    Martinez et al.'s code to work with batched inputs. Original code can be
    found here:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/
    data_utils.py#L12
    Args:
        R: An np array of shape (..., 3, 3)
    Returns:
        An np array of shape (..., 3) containing the Euler angles for each
        rotation matrix in `R`
    """
    R = np.swapaxes(R, R.ndim - 1, R.ndim - 2)

    # Rest of the method assumes row-wise arrangement of rotation matrix R
    assert R.shape[-1] == 3 and R.shape[-2] == 3
    orig_shape = R.shape[:-2]
    rs = np.reshape(R, [-1, 3, 3])
    n_samples = rs.shape[0]

    # initialize to zeros
    e1 = np.zeros([n_samples])
    e2 = np.zeros([n_samples])
    e3 = np.zeros([n_samples])

    # find indices where we need to treat special cases
    is_one = rs[:, 0, 2] == 1
    is_minus_one = rs[:, 0, 2] == -1
    is_special = np.logical_or(is_one, is_minus_one)

    e1[is_special] = np.arctan2(rs[is_special, 0, 1], rs[is_special, 0, 2])
    e2[is_minus_one] = np.pi / 2
    e2[is_one] = -np.pi / 2

    # normal cases
    is_normal = ~np.logical_or(is_one, is_minus_one)
    # clip inputs to arcsin
    in_ = np.clip(rs[is_normal, 0, 2], -1, 1)
    e2[is_normal] = -np.arcsin(in_)
    e2_cos = np.cos(e2[is_normal])
    e1[is_normal] = np.arctan2(
        rs[is_normal, 1, 2] / e2_cos, rs[is_normal, 2, 2] / e2_cos
    )
    e3[is_normal] = np.arctan2(
        rs[is_normal, 0, 1] / e2_cos, rs[is_normal, 0, 0] / e2_cos
    )

    eul = np.stack([e1, e2, e3], axis=-1)
    eul = np.reshape(eul, np.concatenate([orig_shape, eul.shape[1:]]))
    return eul


def R2Q(R):
    # TODO: Write batched version of R2Q. Implemented here is iterative version
    input_shape = R.shape
    R_flat = R.reshape((-1, 3, 3))
    Q = []
    for R in R_flat:
        R00 = R[0, 0]
        R01 = R[0, 1]
        R02 = R[0, 2]
        R10 = R[1, 0]
        R11 = R[1, 1]
        R12 = R[1, 2]
        R20 = R[2, 0]
        R21 = R[2, 1]
        R22 = R[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [R00 - R11 - R22, 0.0, 0.0, 0.0],
                [R01 + R10, R11 - R00 - R22, 0.0, 0.0],
                [R02 + R20, R12 + R21, R22 - R00 - R11, 0.0],
                [R21 - R12, R02 - R20, R10 - R01, R00 + R11 + R22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        Q.append(q)
    Q = np.array(Q)
    return Q.reshape(list(input_shape[:-2]) + [4])


def Q2R(Q):
    # TODO: Write batched version of R2Q. Implemented here is iterative version
    input_shape = Q.shape
    Q_flat = Q.reshape((-1, 4))
    R = []
    for Q in Q_flat:
        q = np.array(Q, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < constants.EPSILON:
            return np.identity(4)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        R.append(
            np.array([
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ])
        )
    R = np.array(R)
    return R.reshape(list(input_shape[:-1]) + [3, 3])


def Q2A(Q):
    # TODO: Implement quaternion to axis angle conversion
    raise NotImplementedError("")


def A2Q(A):
    """
    Adopted from https://github.com/facebookresearch/QuaterNet/blob/
    ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py#L138
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the
    Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert A.shape[-1] == 3

    original_shape = list(A.shape)
    original_shape[-1] = 4
    A = A.reshape(-1, 3)

    theta = np.linalg.norm(A, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * A
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def Q2E(Q, epsilon=0):
    """
    Adopted from https://github.com/facebookresearch/QuaterNet/blob/
    ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py#L53
    Convert quaternion(s) Q to Euler angles.
    Order is expected to be "wxyz"
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert Q.shape[-1] == 4

    original_shape = list(Q.shape)
    original_shape[-1] = 3
    Q = Q.view(-1, 4)

    q0 = Q[:, 0]
    q1 = Q[:, 1]
    q2 = Q[:, 2]
    q3 = Q[:, 3]

    x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
    z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))

    E = np.stack([x, y, z], axis=-1)
    return np.reshape(E, original_shape)


def T2Rp(T):
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    return R, p


def Rp2T(R, p):
    input_shape = R.shape[:-2] if R.ndim > 2 else p.shape[:-1]
    R_flat = R.reshape((-1, 3, 3))
    p_flat = p.reshape((-1, 3))
    T = np.zeros((int(np.prod(input_shape)), 4, 4))
    T[...] = constants.eye_T.copy()
    T[..., :3, :3] = R_flat
    T[..., :3, 3] = p_flat
    return T.reshape(list(input_shape) + [4, 4])


def p2T(p):
    return Rp2T(constants.eye_R, np.array(p))


def R2T(R):
    return Rp2T(R, constants.zero_p)


def Ax2R(Ax):
    """
    Convert (axis) angle along x axis Ax to rotation matrix R
    """
    return A2R(Ax * utils.str_to_axis("x"))


def Ay2R(Ay):
    """
    Convert (axis) angle along y axis Ay to rotation matrix R
    """
    return A2R(Ay * utils.str_to_axis("y"))


def Az2R(Az):
    """
    Convert (axis) angle along z axis Az to rotation matrix R
    """
    return A2R(Az * utils.str_to_axis("z"))
