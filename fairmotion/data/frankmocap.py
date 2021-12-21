# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle
import torch

from fairmotion.data import amass
from fairmotion.core import motion as motion_classes
from fairmotion.utils import constants, utils
from fairmotion.ops import conversions, motion as motion_ops


def get_smpl_base_position(bm, betas):
    pose_body_zeros = torch.zeros((1, 3 * (22 - 1)))
    body = bm(pose_body=pose_body_zeros, betas=betas)
    base_position = body.Jtr.detach().numpy()[0, 0:22]
    return base_position

def compute_im2sim_scale(
    joints_img,
    base_position,
):
    left_leg_sim = np.linalg.norm(base_position[amass.joint_names.index("lknee")] - base_position[amass.joint_names.index("lankle")])
    # indices from from frankmocap.bodymocap.constants
    left_leg_img = np.linalg.norm(joints_img[29][:2] - joints_img[30][:2])
    right_leg_sim = np.linalg.norm(base_position[amass.joint_names.index("rknee")] - base_position[amass.joint_names.index("rankle")])
    right_leg_img = np.linalg.norm(joints_img[25][:2] - joints_img[26][:2])
    return (left_leg_sim + right_leg_sim)/(left_leg_img + right_leg_img)


def load(
    file,
    motion=None,
    bm_path=None,
    motion_key=None,
    estimate_root=False,
    scale=1.0,
    load_skel=True,
    load_motion=True,
    v_up_skel=np.array([0.0, 1.0, 0.0]),
    v_face_skel=np.array([0.0, 0.0, 1.0]),
    v_up_env=np.array([0.0, 1.0, 0.0]),
):
    all_data = pickle.load(open(file, "rb"))
    if motion_key is None:
        motion_key = list(all_data.keys())[0]
    motion_data = all_data[motion_key]
    bm = amass.load_body_model(bm_path)
    betas = torch.Tensor(np.array(motion_data[0]["pred_output_list"][0]["pred_betas"])[:]).to("cpu")
    img_shape = motion_data[0]["pred_output_list"][0]["img_shape"]
    num_joints = len(amass.joint_names)
    skel = amass.create_skeleton_from_amass_bodymodel(bm, betas, len(amass.joint_names), amass.joint_names)
    joint_names = [j.name for j in skel.joints]
    
    num_frames = len(motion_data)
    T = np.random.rand(num_frames, num_joints, 4, 4)
    T[:] = constants.EYE_T
    # Use lowest point of right/left ankle from first image frame as reference
    ref_root_y = np.min((
        motion_data[0]["pred_output_list"][0]["pred_joints_img"][25][1],
        motion_data[0]["pred_output_list"][0]["pred_joints_img"][30][1]
    ))
    for i in range(num_frames):
        for j in range(num_joints):
            T[i][joint_names.index(amass.joint_names[j])] = conversions.R2T(
                np.array(motion_data[i]["pred_output_list"][0]["pred_rotmat"][0])[j]
            )
        if estimate_root:
            R_root = conversions.T2R(T[i][0])
            p_root = np.zeros(3)

            base_position = get_smpl_base_position(bm, betas)
            # compute scale as ratio of limb length in img and bm
            im2sim_scale = compute_im2sim_scale(
                motion_data[i]["pred_output_list"][0]["pred_joints_img"],
                base_position,
            )
            p_root[0] = np.mean((
                motion_data[i]["pred_output_list"][0]["pred_joints_img"][27][0],
                motion_data[i]["pred_output_list"][0]["pred_joints_img"][28][0]
            )) * im2sim_scale
            root_y = np.mean((
                motion_data[i]["pred_output_list"][0]["pred_joints_img"][27][1],
                motion_data[i]["pred_output_list"][0]["pred_joints_img"][28][1]
            ))
            p_root[2] = (ref_root_y - root_y) * im2sim_scale
            # p_root[1] = np.max((
            #     np.linalg.norm(T[i][amass.joint_names.index("root")] - T[i][amass.joint_names.index("lankle")]),
            #     np.linalg.norm(T[i][amass.joint_names.index("root")] - T[i][amass.joint_names.index("rankle")]),
            # ))
            # print(p_root[1])
            T[i][0] = conversions.Rp2T(R_root, p_root)
    motion = motion_classes.Motion.from_matrix(T, skel)

    motion.set_fps(30)
    motion = motion_ops.rotate(
        motion,
        conversions.Ax2R(conversions.deg2rad(-90)),
    )
    # post process to ensure character stays above floor
    positions = motion.positions(local=False)
    for i in range(motion.num_frames()):
        ltoe = positions[i][amass.joint_names.index("ltoe")][2]
        rtoe = positions[i][amass.joint_names.index("rtoe")][2]
        offset = min(ltoe, rtoe)
        if offset < 0.05:
            # print(offset)
            R, p = conversions.T2Rp(T[i][0])
            p[2] += 0.05 - offset
            T[i][0] = conversions.Rp2T(R, p)

    motion = motion_classes.Motion.from_matrix(T, skel)
    motion = motion_ops.rotate(
        motion,
        conversions.Ax2R(conversions.deg2rad(-90)),
    )
    return motion