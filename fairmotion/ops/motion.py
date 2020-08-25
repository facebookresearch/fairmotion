# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import numpy as np

from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, math, quaternion


def append(motion1, motion2):
    """
    Combines two motion sequences into one. motion2 is appended to motion1.
    The operation is not done in place.

    Note that the operation places the sequences next to each other without
    attempting to blend between the poses. To interpolate between the end of
    motion1 and start of motion2, use the `append_and_blend` operation.

    Args:
        motion1, motion2: Motion sequences to be combined.
    """
    assert isinstance(motion1, motion_class.Motion)
    assert isinstance(motion2, motion_class.Motion)
    assert motion1.skel.num_joints() == motion2.skel.num_joints()

    combined_motion = copy.deepcopy(motion1)
    combined_motion.name = f"{motion1.name}+{motion2.name}"
    combined_motion.poses.extend(motion2.poses)

    return combined_motion


def append_and_blend(motion1, motion2, blend_length=0):
    assert isinstance(motion1, motion_class.Motion)
    assert isinstance(motion2, motion_class.Motion)
    assert motion1.skel.num_joints() == motion2.skel.num_joints()

    combined_motion = copy.deepcopy(motion1)
    combined_motion.name = f"{motion1.name}+{motion2.name}"

    if motion1.num_frames() == 0:
        for i in range(motion2.num_frames()):
            combined_motion.poses.append(motion2.poses[i])
            if hasattr(motion1, "velocities"):
                combined_motion.velocities.append(motion2.velocities[i])
        return combined_motion

    frame_target = motion2.time_to_frame(blend_length)
    frame_source = motion1.time_to_frame(motion1.length() - blend_length)

    # Translate and rotate motion2 to location of frame_source
    pose1 = motion1.get_pose_by_frame(frame_source)
    pose2 = motion2.get_pose_by_frame(0)

    R1, p1 = conversions.T2Rp(pose1.get_root_transform())
    R2, p2 = conversions.T2Rp(pose2.get_root_transform())

    # Translation to be applied
    dp = p1 - p2
    dp = dp - math.projectionOnVector(dp, motion1.skel.v_up_env)
    axis = motion1.skel.v_up_env

    # Rotation to be applied
    Q1 = conversions.R2Q(R1)
    Q2 = conversions.R2Q(R2)
    _, theta = quaternion.Q_closest(Q1, Q2, axis)
    dR = conversions.A2R(axis * theta)

    motion2 = translate(motion2, dp)
    motion2 = rotate(motion2, dR)

    t_total = motion1.fps * frame_source
    t_processed = 0.0
    poses_new = []
    for i in range(motion2.num_frames()):
        dt = 1 / motion2.fps
        t_total += dt
        t_processed += dt
        pose_target = motion2.get_pose_by_frame(i)
        # Blend pose for a moment
        if t_processed <= blend_length:
            alpha = t_processed / float(blend_length)
            pose_source = motion1.get_pose_by_time(t_total)
            pose_target = blend(pose_source, pose_target, alpha)
        poses_new.append(pose_target)

    del combined_motion.poses[frame_source + 1 :]
    for i in range(len(poses_new)):
        combined_motion.add_one_frame(0, copy.deepcopy(poses_new[i].data))

    return combined_motion


def blend(pose1, pose2, alpha=0.5):
    assert 0.0 <= alpha <= 1.0
    pose_new = copy.deepcopy(pose1)
    for j in range(pose1.skel.num_joints()):
        R0, p0 = conversions.T2Rp(pose1.get_transform(j, local=True))
        R1, p1 = conversions.T2Rp(pose2.get_transform(j, local=True))
        R, p = math.slerp(R0, R1, alpha), math.lerp(p0, p1, alpha)
        pose_new.set_transform(j, conversions.Rp2T(R, p), local=True)
    return pose_new


def transform(motion, T, local=False):
    """
    Apply transform to all poses of a motion sequence. The operation is done
    in-place.

    Args:
        motion: Motion sequence to be transformed
        T: Transformation matrix of shape (4, 4) to be applied to poses of
            motion
        local: Optional; Set local=True if the transformations are to be
            applied locally, relative to parent of each joint.
    """
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
    """
    Apply translation to motion sequence.

    Args:
        motion: Motion sequence to be translated
        v: Array of shape (3,) indicating translation vector to be applied to
            all poses of motion sequence
        local: Optional; Set local=True if the translation is to be applied
            locally, relative to root position.
    """
    return transform(motion, conversions.p2T(v), local)


def rotate(motion, R, local=False):
    return transform(motion, conversions.R2T(R), local)


def cut(motion, frame_start, frame_end):
    """
    Returns motion object with poses from [frame_start, frame_end) only. The
    operation is not done in-place.

    Args:
        motion: Motion sequence to be cut
        frame_start, frame_end: Frame number range that defines the boundary of
            motion to be cut. Pose at frame_start is included, and pose at
            frame_end is excluded in the returned motion object
    """
    cut_motion = copy.deepcopy(motion)
    cut_motion.name = f"{motion.name}_{frame_start}_{frame_end}"
    cut_motion.poses = motion.poses[frame_start:frame_end]

    return cut_motion


def resample(motion, fps):
    """
    Upsample/downsample frame rate of motion object to `fps` Hz. For
    upsampling, poses are interpolated using `Pose.interpolate` method to
    fill in the gaps.

    Args:
        motion: Motion sequence to be resampled
        fps: Frequency of motion desired
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
    """
    Returns position of joints with respect to the root, for all poses in the
    motion sequence.
    """
    matrix = motion.to_matrix(local=False)
    # Extract positions
    matrix = matrix[:, :, :3, 3]
    # Subtract root position from all joint positions
    matrix = matrix - matrix[:, np.newaxis, 0]
    return matrix
