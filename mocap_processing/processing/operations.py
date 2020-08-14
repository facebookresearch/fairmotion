import copy
import math
import numpy as np

from mocap_processing.motion import motion as motion_class
from mocap_processing.utils import constants, conversions, utils


def append(motion1, motion2):
    assert isinstance(motion1, motion_class.Motion)
    assert isinstance(motion2, motion_class.Motion)
    assert motion1.skel.num_joints() == motion2.skel.num_joints()

    combined_motion = copy.deepcopy(motion1)
    combined_motion.name = f"{motion1.name}+{motion2.name}"
    combined_motion.poses.extend(motion2.poses)

    return combined_motion


def transform(motion, T, local=False):
    for pose_id in range(len(motion.poses)):
        R0, p0 = conversions.T2Rp(motion.poses[pose_id].get_root_transform())
        R1, p1 = conversions.T2Rp(T)
        if local:
            R, p = np.dot(R0, R1), p0 + np.dot(R0, p1)
        else:
            R, p = np.dot(R1, R0), p0 + p1
        motion.poses[pose_id].set_root_transform(
            conversions.Rp2T(R, p), local=False,
        )
    return motion

def translate(motion, v, local=False):
    return transform(motion, conversions.p2T(v), local)


def rotate(motion, R, local=False):
    return transform(motion, conversions.R2T(R), local)


def cut(motion, frame_start, frame_end):
    """
    Returns motion object with poses from [frame_start, frame_end) only
    """
    cut_motion = motion_class.Motion(skel=motion.skel)
    cut_motion.name = f"{motion.name}_{frame_start}_{frame_end}"
    cut_motion.poses = motion.poses[frame_start:frame_end]

    return cut_motion


def resample(motion, fps):
    """
    Upsample/downsample frame rate of motion object to `fps` Hz
    """
    poses_new = []

    dt = 1.0 / fps
    t = 0
    while t < motion.fps * len(motion.poses):
        pose = motion.get_pose_by_time(t)
        pose.skel = motion.skel
        poses_new.append(pose)
        t += dt

    motion.poses = poses_new
    motion.fps = fps
    return motion


def position_wrt_root(motion):
    matrix = motion.to_matrix(local=False)
    # Extract positions
    matrix = matrix[:, :, 0:2, 3]
    # Subtract root position from all joint positions
    matrix = matrix - matrix[:, np.newaxis, 0]
    return matrix


def slerp(R1, R2, t):
    return np.dot(R1, conversions.A2R(t * conversions.R2A(np.dot(R1.transpose(), R2))))


def linearInterpol(v0, v1, t):
    return v0 + (v1 - v0) * t


def invertT(T):
    R = T[:3, :3]
    p = T[:3, 3]
    invT = constants.eye_T()
    R_trans = R.transpose()
    R_trans_p = np.dot(R_trans, p)
    invT[:3, :3] = R_trans
    invT[:3, 3] = R_trans_p
    return invT


def post_process_Q(Q, normalize=True, half_space=True):
    # Here we assume 'xyzw' order
    if normalize:
        Q /= np.linalg.norm(Q)
    if half_space:
        if Q[3] < 0.0:
            Q *= -1.0
    return Q


def componentOnVector(inputVector, directionVector):
    return np.inner(directionVector, inputVector) / np.dot(
        directionVector, directionVector
    )


def projectionOnVector(inputVector, directionVector):
    # componentOnVector() * vd
    return componentOnVector(inputVector, directionVector) * directionVector


def rotX(theta):
    R = constants.eye_R()
    c = math.cos(theta)
    s = math.sin(theta)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def rotY(theta):
    R = constants.eye_R()
    c = math.cos(theta)
    s = math.sin(theta)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R


def rotZ(theta):
    R = constants.eye_R()
    c = math.cos(theta)
    s = math.sin(theta)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def get_R_from_vectors(vec1, vec2):
    """
    Returns R such that R dot vec1 = vec2
    """
    vec1 = utils.normalize(vec1)
    vec2 = utils.normalize(vec2)

    rot_axis = utils.normalize(np.cross(vec1, vec2))
    inner = np.inner(vec1, vec2)
    theta = math.acos(inner)

    if rot_axis[0] == 0 and rot_axis[1] == 0 and rot_axis[2] == 0:
        rot_axis = [0, 1, 0]

    x, y, z = rot_axis
    c = inner
    s = math.sin(theta)
    R = np.array([
        [c + (1.0-c)*x*x, (1.0-c)*x*y - s*z, (1-c)*x*z + s*y],
        [(1.0-c)*x*y + s*z, c + (1.0-c)*y*y, (1.0-c)*y*z - s*x],
        [(1.0-c)*z*x - s*y, (1.0-c)*z*y + s*x, c + (1.0-c)*z*z]
    ])   
    return R
