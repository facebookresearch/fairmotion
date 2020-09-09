# Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np

from fairmotion.ops import conversions, math as math_ops, quaternion


def root_ee_similarity(
    pose1,
    pose2,
    vel1=None,
    vel2=None,
    w_root_pos=1.0,
    w_root_vel=1.0,
    w_ee_pos=1.0,
    w_ee_vel=1.0,
    T_ref_1=None,
    T_ref_2=None,
    auto_weight=True,
    auto_weight_sigma=0.02,
):
    """
    This computes similarity between end_effectors and root for two poses

    Parameters
    ----------
    """

    assert pose1.skel.num_end_effectors() == pose2.skel.num_end_effectors()

    if w_root_vel > 0.0 or w_ee_vel > 0.0:
        assert vel1 is not None and vel2 is not None
        assert vel1.skel.num_end_effectors() == vel2.skel.num_end_effectors()

    skel = pose1.skel

    diff_root_pos = 0.0
    diff_root_vel = 0.0
    diff_ee_pos = 0.0
    diff_ee_vel = 0.0

    """
    Differencse will be computed w.r.t. its own facing frame
    if the reference frame is not given
    """

    if T_ref_1 is None:
        R_face_1, p_face_1 = conversions.T2Rp(pose1.get_facing_transform())
    else:
        R_face_1, p_face_1 = conversions.T2Rp(T_ref_1)
    if T_ref_2 is None:
        R_face_2, p_face_2 = conversions.T2Rp(pose2.get_facing_transform())
    else:
        R_face_2, p_face_2 = conversions.T2Rp(T_ref_2)

    R_face_1_inv = R_face_1.transpose()
    R_face_2_inv = R_face_2.transpose()

    R_root_1, p_root_1 = conversions.T2Rp(pose1.get_root_transform())
    R_root_2, p_root_2 = conversions.T2Rp(pose2.get_root_transform())

    if w_root_pos > 0.0:
        p_root_1_local = np.dot(R_face_1_inv, p_root_1 - p_face_1)
        p_root_2_local = np.dot(R_face_2_inv, p_root_2 - p_face_2)

        diff_root_pos = p_root_2_local - p_root_1_local
        diff_root_pos = np.dot(diff_root_pos, diff_root_pos)

    """ Root Velocity Difference w.r.t. Facing Frame """

    if w_root_vel > 0.0:
        v_root_1 = vel1.get_linear(skel.root_joint, False, R_root_1)
        v_root_2 = vel2.get_linear(skel.root_joint, False, R_root_2)

        v_root_1_local = np.dot(R_face_1_inv, v_root_1)
        v_root_2_local = np.dot(R_face_2_inv, v_root_2)

        diff_root_vel = v_root_2_local - v_root_1_local
        diff_root_vel = np.dot(diff_root_vel, diff_root_vel)

    """ End Effector Position and Velocity Differences w.r.t. Facing Frame """

    num_ee = skel.num_end_effectors()

    if num_ee > 0:
        if w_ee_pos > 0.0 or w_ee_vel > 0.0:
            R1s, p1s = [], []
            R2s, p2s = [], []
            ee_weights = []
            for j in skel.end_effectors:
                R1, p1 = conversions.T2Rp(pose1.get_transform(j, local=False))
                R2, p2 = conversions.T2Rp(pose2.get_transform(j, local=False))
                R1s.append(R1)
                R2s.append(R2)
                p1s.append(p1)
                p2s.append(p2)
                if auto_weight:
                    h = math_ops.projectionOnVector(p1, skel.v_up_env)
                    ee_weights.append(
                        math.exp(-np.dot(h, h) / auto_weight_sigma)
                    )
                else:
                    ee_weights.append(1.0)
            ee_weights_sum = np.sum(ee_weights)
            ee_weights = [w / ee_weights_sum for w in ee_weights]
            # print('--------------------')
            # for j in range(len(skel.end_effectors)):
            #     print(skel.end_effectors[j].name, ee_weights[j])
            # print('--------------------')
            if w_ee_pos > 0.0:
                for j in range(len(skel.end_effectors)):
                    p1_local = np.dot(R_face_1_inv, p1s[j] - p_face_1)
                    p2_local = np.dot(R_face_2_inv, p2s[j] - p_face_2)
                    dp = p2_local - p1_local
                    diff_ee_pos += np.dot(dp, dp)
            if w_ee_vel > 0.0:
                for j in range(len(skel.end_effectors)):
                    v1 = vel1.get_linear(j, False, R1s[j])
                    v2 = vel2.get_linear(j, False, R2s[j])
                    v1_local = np.dot(R_face_1_inv, v1)
                    v2_local = np.dot(R_face_2_inv, v2)
                    dv = v2_local - v1_local
                    diff_ee_vel += np.dot(dv, dv)
            # diff_ee_pos /= float(num_ee)
            # diff_ee_vel /= float(num_ee)

    diff = (
        w_root_pos * diff_root_pos
        + w_root_vel * diff_root_vel
        + w_ee_pos * diff_ee_pos
        + w_ee_vel * diff_ee_vel
    )

    return diff


def pose_similarity(
    pose1,
    pose2,
    vel1=None,
    vel2=None,
    w_joint_pos=0.9,
    w_joint_vel=0.1,
    w_joints=None,
    apply_root_correction=True,
):
    """
    This only measure joint angle difference (i.e. root translation will
    not be considered).
    If 'apply_root_correction' is True, then pose2 will be rotated
    automatically
    in a way that its root rotation is closest to the root rotation of pose1,
    where'root_correction_axis' defines the geodesic curve.
    """
    assert pose1.skel.num_joints() == pose2.skel.num_joints()
    skel = pose1.skel
    if vel1 is not None:
        assert vel2 is not None
        assert vel1.skel.num_joints() == vel2.skel.num_joints()

    if w_joints is None:
        w_joints = np.ones(skel.num_joints())

    """ joint angle difference """
    diff_pos = 0.0
    if w_joint_pos > 0.0:
        root_idx = skel.get_index_joint(skel.root_joint)
        v_up_env = pose1.skel.v_up_env
        for j in range(skel.num_joints()):
            R1, p1 = conversions.T2Rp(pose1.get_transform(j, local=True))
            R2, p2 = conversions.T2Rp(pose2.get_transform(j, local=True))
            if apply_root_correction and j == root_idx:
                Q1 = conversions.R2Q(R1)
                Q2 = conversions.R2Q(R2)
                Q2, _ = quaternion.Q_closest(Q1, Q2, v_up_env)
                R2 = conversions.Q2R(Q2)
            # TODO: Verify if logSO3 is same as R2A
            dR = conversions.R2A(np.dot(np.transpose(R1), R2))
            diff_pos += w_joints[j] * np.dot(dR, dR)
    """ joint angular velocity difference """
    diff_vel = 0.0
    if vel1 is not None and w_joint_vel > 0.0:
        skel = vel1.skel
        for j in range(skel.num_joints()):
            dw = vel2.get_angular(j, local=True) - vel1.get_angular(
                j, local=True
            )
            diff_vel += w_joints[j] * np.dot(dw, dw)
    return (
        w_joint_pos * diff_pos + w_joint_vel * diff_vel
    ) / skel.num_joints()
