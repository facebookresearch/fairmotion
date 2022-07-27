from fairmotion.core import motion
from fairmotion.core.bullet import bullet_client
from fairmotion.data import bvh
from fairmotion.ops import conversions
from fairmotion.tasks.pfnn import pfnn_char_info, pfnn, sim_agent

import copy
import numpy as np
import os
import pickle
import pybullet as pb
import time

def get_render_data(pb_client, agent):
    joint_data = []
    link_data = []
    model = agent._body_id
    for j in range(pb_client.getNumJoints(model)):
        joint_info = pb_client.getJointInfo(model, j)
        joint_local_p, joint_local_Q, link_idx = joint_info[14], joint_info[15], joint_info[16]
        T_joint_local = conversions.Qp2T(
            np.array(joint_local_Q), np.array(joint_local_p))
        if link_idx == -1:
            link_world_p, link_world_Q = pb_client.getBasePositionAndOrientation(model)
        else:
            link_info = pb_client.getLinkState(model, link_idx)
            link_world_p, link_world_Q = link_info[0], link_info[1]
        T_link_world = conversions.Qp2T(
            np.array(link_world_Q), np.array(link_world_p))
        T_joint_world = np.dot(T_link_world, T_joint_local)
        R, p = conversions.T2Rp(T_joint_world)
        joint_data.append((conversions.R2Q(R), p))
    data_visual = pb_client.getVisualShapeData(model)
    lids = [d[1] for d in data_visual]
    dvs = data_visual
    for lid, dv in zip(lids, dvs):        
        if lid == -1:
            p, Q = pb_client.getBasePositionAndOrientation(model)
        else:
            link_state = pb_client.getLinkState(model, lid)
            p, Q = link_state[4], link_state[5]

        p, Q = np.array(p), np.array(Q)
        R = conversions.Q2R(Q)
        T_joint = conversions.Rp2T(R, p)
        T_visual_from_joint = \
            conversions.Qp2T(np.array(dv[6]),np.array(dv[5]))
        R, p = conversions.T2Rp(np.dot(T_joint, T_visual_from_joint))
        link_data.append((conversions.R2Q(R), p))

    return joint_data, link_data

def main():
    start_time = time.time()
    pb_client = bullet_client.BulletClient(
        connection_mode=pb.GUI, options=' --opengl2'
    )
    print("Initialize sim", time.time() - start_time)
    start_time = time.time()
    dirname = os.path.dirname(__file__)
    character = sim_agent.SimAgent(
        pybullet_client=pb_client, 
        model_file=os.path.join(dirname, "data/character/pfnn.urdf"),
        char_info=pfnn_char_info,
    )
    print("Initialize character", time.time() - start_time)
    start_time = time.time()
    runner = pfnn.Runner(user_input='autonomous')
    m = bvh.load(os.path.join(dirname, "data/motion/pfnn_hierarchy.bvh"))

    # Run PFNN for a second to start with arbitrary pose
    for i in range(0):
        runner.update()
    
    print("Initialize runner", time.time() - start_time)
    start_time = time.time()

    data = {
        "joint_data": [],
        "link_data": []
    }
    for _ in range(1000):
        # for i in range(2):
        runner.update()
        print("Update runner", time.time() - start_time)
        start_time = time.time()
        # print(runner.character.joint_xform_by_ik)
        # print(runner.character.joint_global_anim_xform_by_fk)
        # pose = motion.Pose(skel=m.skel, data=runner.character.joint_xform_by_ik)
        # character.set_pose(pose=pose)
        character.set_pose_by_xform(xform=runner.character.joint_global_anim_xform_by_fk)
        print("Set pose", time.time() - start_time)
        start_time = time.time()
        joint_data, link_data = get_render_data(pb_client, character)
        data["joint_data"].append(joint_data)
        data["link_data"].append(link_data)
        print("Record link data", time.time() - start_time)
        start_time = time.time()

    with open(os.path.join(dirname, "data/render_data.pkl"), "wb") as file:
        pickle.dump(data, file)
    print("Save pickle", time.time() - start_time)
    start_time = time.time()


if __name__ == "__main__":
    main()