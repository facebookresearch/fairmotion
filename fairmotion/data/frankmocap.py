# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle
import torch

from fairmotion.data import amass
from fairmotion.core import motion as motion_classes
from fairmotion.utils import constants, utils
from fairmotion.ops import conversions


def load(
    file,
    motion=None,
    bm_path=None,
    motion_key=None,
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
    betas = torch.Tensor(np.array(motion_data[0]["parm_shape"])[:][np.newaxis]).to("cpu")
    num_joints = len(amass.joint_names)
    skel = amass.create_skeleton_from_amass_bodymodel(bm, betas, len(amass.joint_names), amass.joint_names)
    joint_names = [j.name for j in skel.joints]
    
    num_frames = len(motion_data)
    T = np.random.rand(num_frames, num_joints, 4, 4)
    T[:] = constants.EYE_T
    for i in range(num_frames):
        for j in range(num_joints):
            T[i][joint_names.index(amass.joint_names[j])] = conversions.R2T(np.array(motion_data[i]['parm_pose'])[j])
    motion = motion_classes.Motion.from_matrix(T, skel)
    motion.set_fps(30)
    return motion