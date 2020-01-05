import argparse
import cv2
import numpy as np
import sys, os
import torch
import trimesh

from basecode.math import mmMath
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c, colors
from mocap_processing.motion.kinematics import Motion


def get_dfs_order(parents_np):
    stack = []
    def dfs(stack, joint):
        stack.append(joint)
        for i in range(len(parents_np)):
            if parents_np[i] == joint:
                dfs(stack, i)
    dfs(stack, 0)
    return stack


def main(args):
    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(model_type='smplh', bm_path=args.body_model_path, num_betas=num_betas).to(comp_device)
    faces = c2c(bm.f)

    img_shape = (1600, 1600)
    # mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    motion = Motion(file=args.input_file)

    mv = MeshViewer(width=img_shape[0], height=img_shape[1], use_offscreen=False)
    out = cv2.VideoWriter("project.avi", cv2.VideoWriter_fourcc(*"DIVX"), 30, img_shape)

    parents = bm.kintree_table[0].long()[:21 + 1]
    parents = parents.cpu().numpy()
    dfs_order = get_dfs_order(parents)
    for frame in range(motion.num_frame()):
        pose = motion.get_pose_by_frame(frame)

        R, p = mmMath.T2Rp(pose.data[0])
        root_orient = mmMath.logSO3(R)
        trans = p

        num_joints = len(pose.data) - 1
        body_model_pose_data = np.zeros(num_joints*3)
        for joint in range(num_joints):
            pose_idx = dfs_order.index(joint + 1) - 1
            # Convert rotation matrix to axis angle
            axis_angles = mmMath.logSO3(mmMath.T2R(pose.data[joint + 1]))
            body_model_pose_data[pose_idx*3: pose_idx*3 + 3] = axis_angles

        pose_data_t = torch.Tensor(body_model_pose_data).to(comp_device).unsqueeze(0)
        root_orient_t = torch.Tensor(root_orient).to(comp_device).unsqueeze(0)
        trans_t = torch.Tensor(trans).to(comp_device).unsqueeze(0)
        body = bm(pose_body=pose_data_t, root_orient=root_orient_t, trans=trans_t) # , betas=betas)

        body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        mv.set_static_meshes([body_mesh])
        body_image = mv.render()
        img = body_image.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Wrap character from BVH file with SMPL body'
    )
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--body-model-path", type=str, required=True)
    args = parser.parse_args()
    main(args)


