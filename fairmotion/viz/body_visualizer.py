# Copyright (c) Facebook, Inc. and its affiliates.

"""
This code has been adapted from the AMASS Github repository
https://github.com/nghorbani/amass/blob/master/notebooks/
01-AMASS_Visualization.ipynb

You can download SMPL-H body models from http://mano.is.tue.mpg.de/

Sample command:
python fairmotion/viz/body_visualizer.py \
    --input-file $BVH_FILE \
    --body-model-path $BODY_MODEL_PATH \
    --video-output-path $OUTPUT_VIDEO
"""

try:
    import cv2
except ImportError:
    print(
        "ImportError: Please run `pip install opencv-python` to install "
        "OpenCV package required for the visualizer"
    )
    quit()

import argparse
import numpy as np
import pyrender
import torch
import tqdm
import trimesh

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c, colors
from fairmotion.data import bvh
from fairmotion.ops import conversions


def get_dfs_order(parents_np):
    stack = []

    def dfs(stack, joint):
        stack.append(joint)
        for i in range(len(parents_np)):
            if parents_np[i] == joint:
                dfs(stack, i)

    dfs(stack, 0)
    return stack


def prepare_mesh_viewer(img_shape):
    mv = MeshViewer(width=img_shape[0], height=img_shape[1], use_offscreen=True)
    mv.scene = pyrender.Scene(bg_color=colors["white"], ambient_light=(0.3, 0.3, 0.3))
    pc = pyrender.PerspectiveCamera(
        yfov=np.pi / 3.0, aspectRatio=float(img_shape[0]) / img_shape[1]
    )
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 3.75])
    mv.camera_node = mv.scene.add(pc, pose=camera_pose, name="pc-camera")
    mv.viewer = pyrender.OffscreenRenderer(*mv.figsize)
    mv.use_raymond_lighting(5.0)
    return mv


def main(args):
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(
        model_type="smplh", bm_path=args.body_model_path, num_betas=num_betas
    ).to(comp_device)
    faces = c2c(bm.f)

    img_shape = (1600, 1600)
    motion = bvh.load(
        file=args.input_file,
        scale=0.5,
        v_up_skel=np.array([0.0, 1.0, 0.0]),
        v_face_skel=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 0.0, 1.0]),
    )
    motion.rotate(conversions.Ax2R(conversions.deg2Rad(-90)))
    mv = prepare_mesh_viewer(img_shape)

    out = cv2.VideoWriter(
        args.video_output_path, cv2.VideoWriter_fourcc(*"XVID"), 30, img_shape
    )

    parents = bm.kintree_table[0].long()[: 21 + 1]
    parents = parents.cpu().numpy()
    dfs_order = get_dfs_order(parents)
    for frame in tqdm.tqdm(range(motion.num_frames())):
        pose = motion.get_pose_by_frame(frame)

        R, p = conversions.T2Rp(pose.data[0])
        root_orient = conversions.R2A(R)
        trans = p

        num_joints = len(pose.data) - 1
        body_model_pose_data = np.zeros(num_joints * 3)
        for motion_joint, amass_joint in enumerate(dfs_order):
            # motion_joint is idx of joint in Motion class order
            # amass_joint is idx of joint in AMASS skeleton
            if amass_joint == 0:
                continue
            pose_idx = amass_joint - 1
            # Convert rotation matrix to axis angle
            axis_angles = conversions.R2A(conversions.T2R(pose.data[motion_joint]))
            body_model_pose_data[pose_idx * 3 : pose_idx * 3 + 3] = axis_angles

        pose_data_t = torch.Tensor(body_model_pose_data).to(comp_device).unsqueeze(0)
        root_orient_t = torch.Tensor(root_orient).to(comp_device).unsqueeze(0)
        trans_t = torch.Tensor(trans).to(comp_device).unsqueeze(0)
        body = bm(
            pose_body=pose_data_t, root_orient=root_orient_t, trans=trans_t
        )  # , betas=betas)

        body_mesh = trimesh.Trimesh(
            vertices=c2c(body.v[0]),
            faces=faces,
            vertex_colors=np.tile(colors["grey"], (6890, 1)),
        )
        # TODO: Add floor trimesh to the scene to display the ground plane
        mv.set_static_meshes([body_mesh])
        body_image = mv.render()
        img = body_image.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap character from BVH file with SMPL body"
    )
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--body-model-path", type=str, required=True)
    parser.add_argument("--video-output-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
