import copy
import numpy as np

from mocap_processing.motion import motion as motion_class
from mocap_processing.utils import conversions


def append(motion1, motion2):
    assert isinstance(motion1, motion_class.Motion)
    assert isinstance(motion2, motion_class.Motion)
    assert motion1.skel.num_joints() == motion2.skel.num_joints()

    combined_motion = copy.deepcopy(motion1)
    combined_motion.name = f"{motion1.name}+{motion2.name}"
    combined_motion.poses.extend(motion2.poses)

    combined_motion.times = list(np.append(
        combined_motion.times,
        np.array(motion2.times) + combined_motion.times[-1] + 1.0/combined_motion.fps
    ))
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
            conversions.Rp2T(R, p),
            local=False,
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
    cut_motion.times = motion.times[frame_start: frame_end]
    cut_motion.poses = motion.poses[frame_start: frame_end]

    t_init = cut_motion.times[0]
    for i in range(cut_motion.num_frames()):
        cut_motion.times[i] -= t_init

    return cut_motion


def resample(motion, fps):
    """
    Upsample/downsample frame rate of motion object to `fps` Hz
    """
    times_new = []
    poses_new = []

    dt = 1.0 / fps
    t = motion.times[0]
    while t < motion.times[-1]:
        pose = motion.get_pose_by_time(t)
        pose.skel = motion.skel
        times_new.append(t)
        poses_new.append(pose)
        t += dt

    motion.times = times_new
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
