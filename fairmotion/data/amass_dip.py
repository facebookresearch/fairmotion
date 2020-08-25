# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle as pkl
from fairmotion.core import motion as motion_class
from fairmotion.utils import constants
from fairmotion.ops import conversions


SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPL_NR_JOINTS = 24
SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]
SMPL_JOINTS = [
    "pelvis",
    "l_hip",
    "r_hip",
    "spine1",
    "l_knee",
    "r_knee",
    "spine2",
    "l_ankle",
    "r_ankle",
    "spine3",
    "l_foot",
    "r_foot",
    "neck",
    "l_collar",
    "r_collar",
    "head",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hand",
    "r_hand",
]
SMPL_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}

# this are the offsets stored under `J` in the SMPL model pickle file
OFFSETS = np.array(
    [
        [-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
        [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
        [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
        [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
        [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
        [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
        [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
        [8.95999143e-02, -1.04856032e00, -3.04155922e-02],
        [-9.20120818e-02, -1.05466743e00, -2.80514913e-02],
        [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
        [1.12937580e-01, -1.10320516e00, 8.39545265e-02],
        [-1.14055299e-01, -1.10107698e00, 8.98482216e-02],
        [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
        [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
        [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
        [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
        [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
        [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
        [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
        [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
        [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
        [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
        [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
        [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02],
    ]
)


def load(
    file,
    motion=None,
    scale=1.0,
    load_skel=True,
    load_motion=True,
    v_up_skel=np.array([0.0, 1.0, 0.0]),
    v_face_skel=np.array([0.0, 0.0, 1.0]),
    v_up_env=np.array([0.0, 1.0, 0.0]),
):
    if not motion:
        motion = motion_class.Motion(fps=60)

    if load_skel:
        skel = motion_class.Skeleton(
            v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env,
        )
        smpl_offsets = np.zeros([24, 3])
        smpl_offsets[0] = OFFSETS[0]
        for idx, pid in enumerate(SMPL_PARENTS[1:]):
            smpl_offsets[idx+1] = OFFSETS[idx + 1] - OFFSETS[pid]
        for joint_name, parent_joint, offset in zip(SMPL_JOINTS, SMPL_PARENTS, smpl_offsets):
            joint = motion_class.Joint(name=joint_name)
            if parent_joint == -1:
                parent_joint_name = None
                joint.info["dof"] = 6  # root joint is free
                offset -= offset
            else:
                parent_joint_name = SMPL_JOINTS[parent_joint]
            offset = offset / np.linalg.norm(smpl_offsets[4])
            T1 = conversions.p2T(scale * offset)
            joint.xform_from_parent_joint = T1
            skel.add_joint(joint, parent_joint_name)
        motion.skel = skel
    else:
        assert motion.skel is not None

    if load_motion:
        assert motion.skel is not None
        # Assume 60fps
        motion.fps = 60.0
        dt = float(1 / motion.fps)
        with open(file, "rb") as f:
            data = pkl.load(f, encoding="latin1")
            poses = np.array(data["poses"])  # shape (seq_length, 135)
            assert len(poses) > 0, "file is empty"
            poses = poses.reshape((-1, len(SMPL_MAJOR_JOINTS), 3, 3))

            for pose_id, pose in enumerate(poses):
                pose_data = [constants.eye_T() for _ in range(len(SMPL_JOINTS))]
                major_joint_id = 0
                for joint_id, joint_name in enumerate(SMPL_JOINTS):
                    if joint_id in SMPL_MAJOR_JOINTS:
                        pose_data[
                            motion.skel.get_index_joint(joint_name)
                        ] = conversions.R2T(pose[major_joint_id])
                        major_joint_id += 1
                motion.add_one_frame(pose_id * dt, pose_data)

    return motion


def save():
    raise NotImplementedError("Using bvh.save() is recommended")
