import numpy as np
from enum import Enum

from fairmotion.core.bullet import bullet_utils as bu
from fairmotion.core import motion
from fairmotion.ops import conversions, math, quaternion
from fairmotion.utils import constants

import warnings

profile = False
if profile:
    from fairmotion.viz.utils import TimeChecker

    tc = TimeChecker()


class SimAgent(object):
    """
    This defines a simulated character in the scene.
    """
    def __init__(
        self,
        pybullet_client,
        model_file,
        char_info,
        scale=1.0,  # This affects loadURDF
        ref_scale=1.0,  # This will be used when reference motions are appllied to this agent
        ref_height_fix=0.0,
        verbose=False,
        name="agent",
    ):
        self._name = name
        self._pb_client = pybullet_client
        self._char_info = char_info
        # Load self._body_id file
        char_create_flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        self._body_id = self._pb_client.loadURDF(
            model_file,
            [1, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags,
        )
        self._ref_scale = ref_scale
        self._ref_height_fix = ref_height_fix
        self._ref_height_fix_v = ref_height_fix * self._char_info.v_up_env
        self._num_joint = self._pb_client.getNumJoints(self._body_id)
        self._joint_indices = range(self._num_joint)
        self._link_indices = range(-1, self._num_joint)
        self._num_link = len(self._link_indices)
        self._joint_indices_movable = []
        self.setup_kinematics()
        # Pre-compute information about the agent
        self._joint_type = []
        self._joint_axis = []
        self._joint_dofs = []
        for j in self._joint_indices:
            joint_info = self._pb_client.getJointInfo(self._body_id, j)
            self._joint_type.append(joint_info[2])
            self._joint_axis.append(np.array(joint_info[13]))

        for j in self._joint_indices:
            if self._joint_type[j] == self._pb_client.JOINT_SPHERICAL:
                self._joint_dofs.append(3)
                self._joint_indices_movable.append(j)
            elif self._joint_type[j] == self._pb_client.JOINT_REVOLUTE:
                self._joint_dofs.append(1)
                self._joint_indices_movable.append(j)
            elif self._joint_type[j] == self._pb_client.JOINT_FIXED:
                self._joint_dofs.append(0)
            else:
                raise NotImplementedError()
        self._num_dofs = np.sum(self._joint_dofs)
        self._joint_pose_init, self._joint_vel_init = self.get_joint_states()
        self._joint_parent_link = []
        self._joint_xform_from_parent_link = []
        for j in self._joint_indices:
            joint_info = self._pb_client.getJointInfo(self._body_id, j)
            joint_local_p = np.array(joint_info[14])
            joint_local_Q = np.array(joint_info[15])
            link_idx = joint_info[16]
            self._joint_parent_link.append(link_idx)
            self._joint_xform_from_parent_link.append(
                conversions.Qp2T(joint_local_Q, joint_local_p)
            )

    def get_name(self):
        return self._name

    def split_joint_variables(self, states, joint_indices):
        states_out = []
        idx = 0
        for j in joint_indices:
            joint_type = self._joint_type[j]
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                Q = conversions.A2Q(np.array(states[idx : idx + 3]))
                states_out.append(Q)
                idx += 3
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                states_out.append([states[idx]])
                idx += 1
            elif joint_type == self._pb_client.JOINT_FIXED:
                pass
            else:
                raise NotImplementedError()
        assert idx == len(states)
        return states_out

    def setup_kinematics(self):
        # Settings for the kinematic self._body_id so that it does not affect the simulation
        self._pb_client.changeDynamics(
            self._body_id, -1, linearDamping=0, angularDamping=0
        )
        self._pb_client.setCollisionFilterGroupMask(
            self._body_id, -1, collisionFilterGroup=0, collisionFilterMask=0
        )
        for j in range(-1, self._pb_client.getNumJoints(self._body_id)):
            self._pb_client.setCollisionFilterGroupMask(
                self._body_id, j, collisionFilterGroup=0, collisionFilterMask=0
            )
            self._pb_client.changeDynamics(
                self._body_id,
                j,
                activationState=self._pb_client.ACTIVATION_STATE_SLEEP
                + self._pb_client.ACTIVATION_STATE_ENABLE_SLEEPING
                + self._pb_client.ACTIVATION_STATE_DISABLE_WAKEUP,
            )
        self.change_visual_color([0, 1, 0, 0])

    def change_visual_color(self, color):
        self._pb_client.changeVisualShape(self._body_id, -1, rgbaColor=color)
        for j in range(self._pb_client.getNumJoints(self._body_id)):
            self._pb_client.changeVisualShape(self._body_id, j, rgbaColor=color)

    def get_num_dofs(self):
        return self._num_dofs

    def get_num_joint(self):
        return self._num_joint

    def get_joint_type(self, idx):
        return self._joint_type[idx]

    def get_joint_axis(self, idx):
        return self._joint_axis[idx]

    def get_joint_dofs(self, idx):
        return self._joint_dofs[idx]

    def get_root_height_from_ground(self, ground_height):
        p, _, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        vec_root_from_ground = math.projectionOnVector(p, self._char_info.v_up_env)
        return np.linalg.norm(vec_root_from_ground) - ground_height

    def get_root_state(self):
        return bu.get_base_pQvw(self._pb_client, self._body_id)

    def get_root_transform(self):
        p, Q, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        return conversions.Qp2T(Q, p)

    def get_root_position(self):
        p, Q, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        return p

    def set_root_transform(self, T):
        Q, p = conversions.T2Qp(T)
        bu.set_base_pQvw(self._pb_client, self._body_id, p, Q, None, None)

    def get_facing_transform(self, ground_height):
        d, p = self.get_facing_direction_position(ground_height)
        z = d
        y = self._char_info.v_up_env
        x = np.cross(y, z)
        return conversions.Rp2T(np.array([x, y, z]).transpose(), p)

    def get_facing_position(self, ground_height):
        d, p = self.get_facing_direction_position(ground_height)
        return p

    def get_facing_direction(self):
        d, p = self.get_facing_direction_position(0)
        return d

    def get_facing_direction_position(self, ground_height):
        R, p = conversions.T2Rp(self.get_root_transform())
        d = np.dot(R, self._char_info.v_face)
        if np.allclose(d, self._char_info.v_up_env):
            msg = (
                "\n+++++++++++++++++WARNING+++++++++++++++++++\n"
                + "The facing direction is ill-defined "
                + "(i.e. parellel to the world up-vector).\n"
                + "A random direction will be assigned for the direction\n"
                + "Be careful if your system is sensitive to the facing direction\n"
                + "+++++++++++++++++++++++++++++++++++++++++++\n"
            )
            warnings.warn(msg)
            d = math.random_unit_vector()
        d = d - math.projectionOnVector(d, self._char_info.v_up_env)
        p = p - math.projectionOnVector(p, self._char_info.v_up_env)
        if ground_height != 0.0:
            p += ground_height * self._char_info.v_up_env
        return d / np.linalg.norm(d), p

    def project_to_ground(self, v):
        return v - math.projectionOnVector(v, self._char_info.v_up_env)

    def get_link_states(self, indices=None):
        return bu.get_link_pQvw(self._pb_client, self._body_id, indices)

    def get_joint_states(self, indices=None):
        return bu.get_joint_pv(self._pb_client, self._body_id, indices)

    def set_pose_by_xform(self, xform):
        assert len(xform) == len(self._char_info.bvh_map_inv)

        """ Base """
        Q, p = conversions.T2Qp(xform[0])
        p *= self._ref_scale
        p += self._ref_height_fix_v

        bu.set_base_pQvw(self._pb_client, self._body_id, p, Q, None, None)

        """ Others """
        indices = []
        state_pos = []
        state_vel = []
        idx = -1
        for k, j in self._char_info.bvh_map_inv.items():
            idx += 1
            if idx == 0:
                continue
            if j is None:
                continue
            joint_type = self._joint_type[j]
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            T = xform[idx]
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                Q, p = conversions.T2Qp(T)
                w = np.zeros(3)
                state_pos.append(Q)
                state_vel.append(w)
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                joint_axis = self.get_joint_axis(j)
                R, p = conversions.T2Rp(T)
                w = np.zeros(3)
                state_pos.append(math.project_rotation_1D(R, joint_axis))
                state_vel.append(math.project_angular_vel_1D(w, joint_axis))
            else:
                raise NotImplementedError()
            indices.append(j)

        bu.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)

    def set_pose(self, pose, vel=None):
        """
        Velocity should be represented w.r.t. local frame
        """
        # Root joint
        T = pose.get_transform(
            self._char_info.bvh_map[self._char_info.ROOT], local=False
        )
        Q, p = conversions.T2Qp(T)
        p *= self._ref_scale
        p += self._ref_height_fix_v

        v, w = None, None
        if vel is not None:
            # Here we give a root orientation to get velocities represeted in world frame.
            R = conversions.Q2R(Q)
            w = vel.get_angular(self._char_info.bvh_map[self._char_info.ROOT], False, R)
            v = vel.get_linear(self._char_info.bvh_map[self._char_info.ROOT], False, R)
            v *= self._ref_scale

        bu.set_base_pQvw(self._pb_client, self._body_id, p, Q, v, w)

        # Other joints
        indices = []
        state_pos = []
        state_vel = []
        for j in self._joint_indices:
            joint_type = self._joint_type[j]
            # When the target joint do not have dof, we simply ignore it
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            # When there is no matching between the given pose and the simulated character,
            # the character just tries to hold its initial pose
            if self._char_info.bvh_map[j] is None:
                state_pos.append(self._joint_pose_init[j])
                state_vel.append(self._joint_vel_init[j])
            else:
                T = pose.get_transform(self._char_info.bvh_map[j], local=True)
                if joint_type == self._pb_client.JOINT_SPHERICAL:
                    Q, p = conversions.T2Qp(T)
                    w = (
                        np.zeros(3)
                        if vel is None
                        else vel.get_angular(self._char_info.bvh_map[j], local=True)
                    )
                    state_pos.append(Q)
                    state_vel.append(w)
                elif joint_type == self._pb_client.JOINT_REVOLUTE:
                    joint_axis = self.get_joint_axis(j)
                    R, p = conversions.T2Rp(T)
                    w = (
                        np.zeros(3)
                        if vel is None
                        else vel.get_angular(self._char_info.bvh_map[j], local=True)
                    )
                    state_pos.append([math.project_rotation_1D(R, joint_axis)])
                    state_vel.append([math.project_angular_vel_1D(w, joint_axis)])
                else:
                    raise NotImplementedError()
            indices.append(j)
        bu.set_joint_pv(self._pb_client, self._body_id, indices, state_pos, state_vel)

    def get_pose_data(self, skel, apply_height_offset=True):
        p, Q, v, w, ps, vs = bu.get_state_all(self._pb_client, self._body_id)
        pose_data = []
        for i in range(skel.num_joints()):
            joint = skel.joints[i]
            if joint == skel.root_joint:
                if not apply_height_offset:
                    p -= self._ref_height_fix_v
                pose_data.append(conversions.Qp2T(Q, p))
            else:
                j = self._char_info.bvh_map_inv[joint.name]
                if j is None:
                    pose_data.append(constants.eye_T())
                else:
                    joint_type = self._joint_type[j]
                    if joint_type == self._pb_client.JOINT_FIXED:
                        pose_data.append(constants.eye_T())
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        pose_data.append(conversions.Q2T(ps[j]))
                    else:
                        raise NotImplementedError()
        return pose_data

    def get_pose(self, skel, apply_height_offset=True):
        return motion.Pose(skel, self.get_pose_data(skel, apply_height_offset))

    def array_to_pose_data(self, skel, data, T_root_ref=None):
        assert len(data) == self._num_dofs + 6
        T_root = conversions.Rp2T(conversions.A2R(data[3:6]), data[0:3])
        if T_root_ref is not None:
            T_root = np.dot(T_root_ref, T_root)
        pose_data = []
        idx = 6
        for i in range(skel.num_joints()):
            joint = skel.joints[i]
            if joint == skel.root_joint:
                pose_data.append(T_root)
            else:
                j = self._char_info.bvh_map_inv[joint.name]
                if j is None:
                    pose_data.append(constants.eye_T())
                else:
                    joint_type = self._joint_type[j]
                    if joint_type == self._pb_client.JOINT_FIXED:
                        pose_data.append(constants.eye_T())
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        pose_data.append(
                            conversions.R2T(conversions.A2R(data[idx : idx + 3]))
                        )
                        idx += 3
                    else:
                        raise NotImplementedError()
        return pose_data

    def arrary_to_pose(self, skel, data, T_root_ref=None):
        pose_data = self.array_to_pose_data(skel, data)
        return motion.Pose(skel, pose_data)

    def save_states(self):
        return bu.get_state_all(self._pb_client, self._body_id)

    def restore_states(self, states):
        bu.set_state_all(self._pb_client, self._body_id, states)

    def get_com_and_com_vel(self):
        return bu.compute_com_and_com_vel(
            self._pb_client, self._body_id, self._link_indices
        )

    def inverse_kinematics(self, indices, positions):
        assert len(indices) == len(positions)
        new_positions = self._pb_client.calculateInverseKinematics2(
            self._body_id,
            endEffectorLinkIndices=indices,
            targetPositions=positions,
            solver=0,
            maxNumIterations=100,
            residualThreshold=0.01,
        )
        # new_positions = self._pb_client.calculateInverseKinematics(self._body_id, self._char_info.RightHand, np.zeros(3))
        new_positions = self.split_joint_variables(
            new_positions, self._joint_indices_movable
        )
        for p in new_positions:
            print(p)
        self._pb_client.resetJointStatesMultiDof(
            self._body_id, self._joint_indices_movable, new_positions
        )
