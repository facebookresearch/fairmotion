# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import numpy as np

from fairmotion.core import motion as motion_class
from fairmotion.core import velocity as vel_class
from fairmotion.ops import conversions, math, quaternion


def blend(pose1, pose2, alpha=0.5):
    """
    Blends two poses, return (1-alpha)*pose1 + alpha*pose2

    Args:
        pose1, pose2: Poses to be blended.
        alpha: Ratio of interpolation which ranges from 0 to 1.
    """
    assert 0.0 <= alpha <= 1.0
    pose_new = copy.deepcopy(pose1)
    for j in range(pose1.skel.num_joints()):
        R0, p0 = conversions.T2Rp(pose1.get_transform(j, local=True))
        R1, p1 = conversions.T2Rp(pose2.get_transform(j, local=True))
        R, p = math.slerp(R0, R1, alpha), math.lerp(p0, p1, alpha)
        pose_new.set_transform(j, conversions.Rp2T(R, p), local=True)
    return pose_new

def stitch(
    motion1, 
    motion2, 
    pivot_offset1=0,
    pivot_offset2=0,
    blend_length=0, 
    blend_method="overlapping"
    ):
    """
    Combines two motion sequences into one, motion2 is appended to motion1.
    Blending is done if requested. The operation is not done in place.
    The second motion is rotate and translated so that they are connected smoothly.
    This method is a subset of the append method

    Args:
        motion1: Previous motion 
        motion2: New motions to be added
        pivot_offset1: Pivot frame offset (sec) to access the pivot pose for motion1.
            The pose at [motion1.length()-pivot_offset1] will be the pivot pose
        pivot_offset2: Pivot frame offset (sec) to access the pivot pose for motion2.
            The pose at [pivot_offset2] will be the pivot pose.
        blend_length: lentgh of blending for stiched area
        blend_method: blending methods 'propagation', 'overlapping', and 'inertialization'.
    """
    return append(
        motion1, 
        motion2, 
        pivot_offset1, 
        pivot_offset2, 
        True, 
        blend_length, 
        blend_method)

def append(
    motion1, 
    motion2, 
    pivot_offset1=0,
    pivot_offset2=0,
    pivot_alignment=False,
    blend_length=0, 
    blend_method="overlapping"
    ):
    """
    Combines two motion sequences into one, motion2 is appended to motion1.
    Blending is done if requested. The operation is not done in place.

    Args:
        motion1: Previous motion 
        motion2: New motions to be added
        pivot_offset1: Pivot frame offset (sec) to access the pivot pose for motion1.
            The pose at [motion1.length()-pivot_offset1] will be the pivot pose
        pivot_offset2: Pivot frame offset (sec) to access the pivot pose for motion2.
            The pose at [pivot_offset2] will be the pivot pose.
        pivot_alignment: Whether motion2 will be aligned to motion1 or not
        blend_length: lentgh of blending for stiched area
        blend_method: blending methods 'propagation', 'overlapping', and 'inertialization'.
    """
    assert isinstance(motion1, (motion_class.Motion, vel_class.MotionWithVelocity))
    assert isinstance(motion2, (motion_class.Motion, vel_class.MotionWithVelocity))
    assert motion1.fps == motion2.fps
    assert motion1.skel.num_joints() == motion2.skel.num_joints()
    assert motion1.num_frames() > 0 or motion2.num_frames() > 0

    if motion1.num_frames() == 0:
        combined_motion = copy.deepcopy(motion2)
        combined_motion.name = f"{motion1.name}+{motion2.name}"
        return combined_motion

    if motion2.num_frames() == 0:
        combined_motion = copy.deepcopy(motion1)
        combined_motion.name = f"{motion1.name}+{motion2.name}"
        return combined_motion

    frame_source = motion1.time_to_frame(motion1.length() - pivot_offset1)
    frame_target = motion2.time_to_frame(pivot_offset2)

    # Translate and rotate motion2 to location of frame_source
    pose1 = motion1.get_pose_by_frame(frame_source)
    pose2 = motion2.get_pose_by_frame(frame_target)

    R1, p1 = conversions.T2Rp(pose1.get_root_transform())
    R2, p2 = conversions.T2Rp(pose2.get_root_transform())

    v_up_env = motion1.skel.v_up_env

    # Remove the translation of the pivot of the motion2
    # so that rotation works correctly
    dp = -(p2 - math.projectionOnVector(p2, v_up_env))
    motion2 = translate(motion2, dp)

    # Translation to be applied
    dp = p1 - math.projectionOnVector(p1, v_up_env)

    # Rotation to be applied
    Q1 = conversions.R2Q(R1)
    Q2 = conversions.R2Q(R2)
    _, theta = quaternion.Q_closest(Q1, Q2, v_up_env)
    dR = conversions.A2R(v_up_env * theta)

    motion2 = transform(
        motion2, conversions.Rp2T(dR, dp), pivot=0, local=False)

    combined_motion = copy.deepcopy(motion1)
    combined_motion.name = f"{motion1.name}+{motion2.name}"
    del combined_motion.poses[frame_source + 1 :]

    t_start = motion1.length()-blend_length
    t_processed = 0.0
    dt = 1 / motion2.fps
    for i in range(frame_target, motion2.num_frames()):
        t_processed += dt
        if blend_length > 0.0:
            alpha = min(1.0, t_processed / float(blend_length))
        else:
            alpha = 1.0
        # Do blending for a while (blend_length)
        if alpha < 1.0:
            if blend_method == "propagation":
                pose_out = blend(
                    motion1.get_pose_by_time(t_start),
                    motion2.get_pose_by_frame(i),
                    alpha)
            elif blend_method == "overlapping":
                pose_out = blend(
                    motion1.get_pose_by_time(t_start+t_processed),
                    motion2.get_pose_by_frame(i),
                    alpha)
            elif blend_method == "inertialization":
                # TODO
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            pose_out = copy.deepcopy(motion2.get_pose_by_frame(i))
        combined_motion.add_one_frame(pose_out.data)

    # Recompute velocities if exists
    if isinstance(combined_motion, vel_class.MotionWithVelocity):
        combined_motion.compute_velocities()

    return combined_motion


def transform(motion, T, pivot=0, local=False):
    """
    Apply transform to all poses of a motion sequence. The operation is done
    in-place.

    Args:
        motion: Motion sequence to be transformed
        T: Transformation matrix of shape (4, 4) to be applied to poses of
            motion
        pivot: Optional; The pivot frame number for the transformation. 
            For example, if it is 0 and the transformation is pure rotation, 
            the entire motion rotates w.r.t. the first frame.
        local: Optional; Set local=True if the transformations are to be
            applied locally, relative to parent of each joint.
    """
    pose_pivot = motion.poses[pivot]
    T_root = pose_pivot.get_root_transform()
    T_root_inv = math.invertT(T_root)
    # Save the relative transform of each pose w.r.t. the pivot pose
    T_rel_wrt_pivot = []
    for pose_id in range(len(motion.poses)):
        T_rel = np.dot(T_root_inv, motion.poses[pose_id].get_root_transform())
        T_rel_wrt_pivot.append(T_rel)
    # Transform the pivot pose
    T_root_new = np.dot(T_root, T) if local else np.dot(T, T_root)
    pose_pivot.set_root_transform(T_root_new, local=False)
    # Transform the remaining poses by using the transformed pivot pose
    for pose_id in range(len(motion.poses)):
        T_new = np.dot(T_root_new, T_rel_wrt_pivot[pose_id])
        motion.poses[pose_id].set_root_transform(T_new, local=False)
    # Recompute velocities if exists
    if isinstance(motion, vel_class.MotionWithVelocity):
        motion.compute_velocities()
    return motion


def translate(motion, v, pivot=0, local=False):
    """
    Apply translation to motion sequence.

    Args:
        motion: Motion sequence to be translated
        v: Array of shape (3,) indicating translation vector to be applied to
            all poses of motion sequence
        pivot: Optional; The pivot frame number for the traslation. 
            In translation, it is only meaningful when local==True.
        local: Optional; Set local=True if the translation is to be applied
            locally, relative to root position.
    """
    return transform(motion, conversions.p2T(v), pivot, local)


def rotate(motion, R, pivot=0, local=False):
    """
    Apply rotation to motion sequence.

    Args:
        motion: Motion sequence to be rotated
        R: Array of shape (3, 3) indicating rotation matrix to be applied
        pivot: Optional; The pivot frame number for the rotation. 
            For example, if it is 0 then the entire motion rotates 
            w.r.t. the first frame.
        local: Optional; Set local=True if the translation is to be applied
            locally, relative to root position.
    """
    return transform(motion, conversions.R2T(R), pivot, local)


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
    cut_motion.poses = cut_motion.poses[frame_start:frame_end]

    # Recompute velocities if exists
    if isinstance(cut_motion, vel_class.MotionWithVelocity):
        cut_motion.vels = cut_motion.vels[frame_start:frame_end]

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
    t = 0.0
    while t <= motion.length():
        pose = motion.get_pose_by_time(t)
        pose.skel = motion.skel
        poses_new.append(pose)
        t += dt

    motion.poses = poses_new
    motion.set_fps(fps)

    # Recompute velocities if exists
    if isinstance(motion, vel_class.MotionWithVelocity):
        motion.compute_velocities()

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
