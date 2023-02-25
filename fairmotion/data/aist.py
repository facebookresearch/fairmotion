# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle
import torch
from human_body_prior.body_model.body_model import BodyModel
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.data import amass

"""
Structure of pkl file in AIST dataset is as follows.
- smpl_trans (num_frames, 3):  translation (x, y, z) of root joint
- smpl_scaling (1,): 
- smpl_loss (1,)
- smpl_poses (num_frames, 72)
    0-2 Root orientation
    3-65 Body joint orientations
    66-155 Finger articulations
"""

def load(file, bm=None, bm_path=None, model_type="smplh"):
    num_betas = 10
    if bm is None:
        # Download the required body model. For SMPL-H download it from
        # http://mano.is.tue.mpg.de/.
        assert bm_path is not None, "Please provide SMPL body model path"
        bm = amass.load_body_model(bm_path, num_betas, model_type)

    skel = amass.create_skeleton_from_amass_bodymodel(
        bm, None, len(amass.joint_names), amass.joint_names,
    )

    bdata = pickle.load(open(file, "rb"))
    fps = 60
    root_orient = bdata["smpl_poses"][:, :3]  # controls the global root orientation
    pose_body = bdata["smpl_poses"][:, 3:66]  # controls body joint angles
    trans = bdata["smpl_trans"][:, :3] / bdata["smpl_scaling"][0] # controls global position

    motion = motion_class.Motion(skel=skel, fps=fps)

    num_joints = skel.num_joints()
    parents = bm.kintree_table[0].long()[:num_joints]

    for frame in range(pose_body.shape[0]):
        pose_body_frame = pose_body[frame]
        root_orient_frame = root_orient[frame]
        root_trans_frame = trans[frame]
        pose_data = []
        for j in range(num_joints):
            if j == 0:
                T = conversions.Rp2T(
                    conversions.A2R(root_orient_frame), root_trans_frame
                )
            else:
                T = conversions.R2T(
                    conversions.A2R(
                        pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3]
                    )
                )
            pose_data.append(T)
        motion.add_one_frame(pose_data)
    
    grounded_motion = motion_ops.fix_height(motion, axis_up="y")
    return grounded_motion