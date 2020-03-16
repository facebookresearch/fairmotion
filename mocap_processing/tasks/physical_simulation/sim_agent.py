import numpy as np
from enum import Enum
from basecode.bullet import bullet_utils as bu
from basecode.math import mmMath
from basecode.motion import kinematics_simple as kinematics


class SimAgent(object):
    class TrackingControl(Enum):
        NONE = 0  # No control
        SPD = 1  # Stable PD Control
        PD = 2  # PD Control
        CPD = 3  # PD Control as Constraints of Simulation
        CP = 4  # Position Control as Constraints of Simulation
        V = 5  # Velocity Control as Constraints of Simulation

        @classmethod
        def from_string(cls, string):
            if string == "none":
                return cls.NONE
            if string == "spd":
                return cls.SPD
            if string == "pd":
                return cls.PD
            if string == "cpd":
                return cls.CPD
            if string == "cp":
                return cls.CP
            if string == "v":
                return cls.V
            raise NotImplementedError

    def __init__(
        self,
        pybullet_client,
        model_file,
        char_info,
        scale=1.0,  # This affects loadURDF
        ref_scale=1.0,  # Used when reference motions are applied to this agent
        verbose=False,
        kinematic_only=False,
        self_collision=True,
    ):
        self._pb_client = pybullet_client
        self._char_info = char_info
        # Load self._body_id file
        char_create_flags = self._pb_client.URDF_MAINTAIN_LINK_ORDER
        if self_collision:
            char_create_flags = (
                char_create_flags
                | self._pb_client.URDF_USE_SELF_COLLISION
                | self._pb_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            )
        self._body_id = self._pb_client.loadURDF(
            model_file,
            [0, 0, 0],
            globalScaling=scale,
            useFixedBase=False,
            flags=char_create_flags,
        )
        for pair in self._char_info.collison_ignore_pairs:
            self._pb_client.setCollisionFilterPair(
                self._body_id, self._body_id, pair[0], pair[1], enableCollision=False
            )

        self._ref_scale = ref_scale
        self._num_joint = self._pb_client.getNumJoints(self._body_id)
        self._joint_indices = range(self._num_joint)
        self._link_indices = range(-1, self._num_joint)
        self._joint_indices_movable = []
        if kinematic_only:
            self.setup_kinematics()
        else:
            self.setup_dynamics()
        # Pre-compute informations about the agent
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
            joint_local_p, joint_local_Q, link_idx = (
                joint_info[14],
                joint_info[15],
                joint_info[16],
            )
            self._joint_parent_link.append(link_idx)
            self._joint_xform_from_parent_link.append(
                mmMath.Qp2T(joint_local_Q, joint_local_p)
            )
        self._link_masses = []
        self._link_total_mass = 0.0
        for i in self._link_indices:
            di = self._pb_client.getDynamicsInfo(self._body_id, i)
            mass = di[0]
            self._link_total_mass += mass
            self._link_masses.append(mass)

        if verbose:
            print("[SimAgent] Creating an agent...", model_file)
            print(
                "num_joint <%d>, num_dofs <%d>, total_mass<%f>"
                % (self._num_joint, self._num_dofs, self._link_total_mass)
            )

    def split_joint_variables(self, states, joint_indices):
        states_out = []
        idx = 0
        for j in joint_indices:
            joint_type = self._joint_type[j]
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                Q = mmMath.R2Q(mmMath.exp(np.array(states[idx : idx + 3])))
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

    def setup_dynamics(self):
        # Settings for the simulation self._body_id
        for j in self._link_indices:
            self._pb_client.changeDynamics(
                self._body_id,
                j,
                lateralFriction=self._char_info.friction_lateral,
                spinningFriction=self._char_info.friction_spinning,
                jointDamping=0.0,
                restitution=self._char_info.restitution,
            )
            di = self._pb_client.getDynamicsInfo(self._body_id, j)

        self._pb_client.changeDynamics(
            self._body_id, -1, linearDamping=0, angularDamping=0
        )
        # Disable the initial motor control
        for j in self._joint_indices:
            self._pb_client.setJointMotorControl2(
                self._body_id,
                j,
                self._pb_client.POSITION_CONTROL,
                targetVelocity=0,
                force=0,
            )
            self._pb_client.setJointMotorControlMultiDof(
                self._body_id,
                j,
                self._pb_client.POSITION_CONTROL,
                targetPosition=[0, 0, 0, 1],
                targetVelocity=[0, 0, 0],
                positionGain=0,
                velocityGain=1,
                force=[0, 0, 0],
            )
        for j in self._joint_indices:
            self._pb_client.enableJointForceTorqueSensor(
                self._body_id, j, enableSensor=True
            )

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
            # self._pb_client.changeVisualShape(self._body_id, j, rgbaColor=[1, 1, 1, 0.4])

    def change_visual_color(self, color):
        for j in range(self._pb_client.getNumJoints(self._body_id)):
            self._pb_client.changeVisualShape(
                self._body_id, j, rgbaColor=color,
            )

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

    def get_root_state(self):
        return bu.get_base_pQvw(self._pb_client, self._body_id)

    def get_root_transform(self):
        p, Q, _, _ = bu.get_base_pQvw(self._pb_client, self._body_id)
        return mmMath.Qp2T(Q, p)

    def get_facing_transform(self):
        d, p = self.get_facing_direction_position()
        z = d
        y = self._char_info.v_up_env
        x = np.cross(y, z)
        return mmMath.Rp2T(np.array([x, y, z]).transpose(), p)

    def get_facing_position(self):
        d, p = self.get_facing_direction_position()
        return p

    def get_facing_direction(self):
        d, p = self.get_facing_direction_position()
        return d

    def get_facing_direction_position(self):
        p, Q = self._pb_client.getBasePositionAndOrientation(self._body_id)
        p, Q = np.array(p), np.array(Q)
        R = mmMath.Q2R(Q)
        d = np.dot(R, self._char_info.v_face)
        d = d - mmMath.projectionOnVector(d, self._char_info.v_up_env)
        p = p - mmMath.projectionOnVector(p, self._char_info.v_up_env)
        return d / np.linalg.norm(d), p

    def project_to_ground(self, v):
        return v - mmMath.projectionOnVector(v, self._char_info.v_up_env)

    def get_link_states(self, indices=None):
        return bu.get_link_pQvw(self._pb_client, self._body_id, indices)

    def get_joint_states(self, indices=None):
        return bu.get_joint_pv(self._pb_client, self._body_id, indices)

    def set_pose_by_xform(self, xform):
        assert len(xform) == len(self._char_info.bvh_map_inv)
        """ Base """
        Q, p = mmMath.T2Qp(xform[0])
        self._pb_client.resetBasePositionAndOrientation(
            self._body_id, self._ref_scale * p, Q
        )
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
                Q, p = mmMath.T2Qp(T)
                w = np.zeros(3)
                state_pos.append(Q)
                state_vel.append(w)
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                joint_axis = self.get_joint_axis(j)
                R, p = mmMath.T2Rp(T)
                w = np.zeros(3)
                state_pos.append(kinematics.project_rotation_1D(R, joint_axis))
                state_vel.append(kinematics.project_angular_vel_1D(w, joint_axis))
            else:
                raise NotImplementedError()
            indices.append(j)
        self._pb_client.resetJointStatesMultiDof(
            self._body_id, indices, state_pos, state_vel
        )

    def set_pose(self, pose, vel=None):
        # Root joint
        T = pose.get_transform(
            self._char_info.bvh_map[self._char_info.ROOT], local=False
        )
        Q, p = mmMath.T2Qp(T)
        p *= self._ref_scale
        self._pb_client.resetBasePositionAndOrientation(self._body_id, p, Q)
        if vel is not None:
            # Here we give a root orientation to get velocities represeted in world frame.
            R = mmMath.Q2R(Q)
            w = vel.get_angular_velocity(
                self._char_info.bvh_map[self._char_info.ROOT], False, R
            )
            v = vel.get_linear_velocity(
                self._char_info.bvh_map[self._char_info.ROOT], False, R
            )
            v *= self._ref_scale
            self._pb_client.resetBaseVelocity(self._body_id, v, w)
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
                continue
            T = pose.get_transform(self._char_info.bvh_map[j], local=True)
            if joint_type == self._pb_client.JOINT_SPHERICAL:
                Q, p = mmMath.T2Qp(T)
                w = (
                    np.zeros(3)
                    if vel is None
                    else vel.get_angular_velocity(
                        self._char_info.bvh_map[j], local=True
                    )
                )
                state_pos.append(Q)
                state_vel.append(w)
            elif joint_type == self._pb_client.JOINT_REVOLUTE:
                joint_axis = self.get_joint_axis(j)
                R, p = mmMath.T2Rp(T)
                w = (
                    np.zeros(3)
                    if vel is None
                    else vel.get_angular_velocity(
                        self._char_info.bvh_map[j], local=True
                    )
                )
                state_pos.append([kinematics.project_rotation_1D(R, joint_axis)])
                state_vel.append([kinematics.project_angular_vel_1D(w, joint_axis)])
            else:
                raise NotImplementedError()
            indices.append(j)
        self._pb_client.resetJointStatesMultiDof(
            self._body_id, indices, state_pos, state_vel
        )

    def get_pose(self, skel):
        p, Q = self._pb_client.getBasePositionAndOrientation(self._body_id)
        states = self._pb_client.getJointStatesMultiDof(
            self._body_id, self._joint_indices
        )
        pose_data = []
        for i in range(skel.num_joint()):
            joint = skel.joints[i]
            if joint == skel.root_joint:
                pose_data.append(mmMath.Qp2T(Q, p))
            else:
                j = self._char_info.bvh_map_inv[joint.name]
                if j is None:
                    pose_data.append(mmMath.I_SE3())
                else:
                    joint_type = self._joint_type[j]
                    if joint_type == self._pb_client.JOINT_FIXED:
                        pose_data.append(mmMath.I_SE3())
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        pose_data.append(mmMath.Q2T(states[j][0]))
                    else:
                        raise NotImplementedError()
        return kinematics.Posture(skel, pose_data)

    def array_to_pose_data(self, skel, data, T_root_ref=None):
        assert len(data) == self._num_dofs + 6
        T_root = mmMath.Rp2T(mmMath.exp(data[3:6]), data[0:3])
        if T_root_ref is not None:
            T_root = np.dot(T_root_ref, T_root)
        pose_data = []
        idx = 6
        for i in range(skel.num_joint()):
            joint = skel.joints[i]
            if joint == skel.root_joint:
                pose_data.append(T_root)
            else:
                j = self._char_info.bvh_map_inv[joint.name]
                if j is None:
                    pose_data.append(mmMath.I_SE3())
                else:
                    joint_type = self._joint_type[j]
                    if joint_type == self._pb_client.JOINT_FIXED:
                        pose_data.append(mmMath.I_SE3())
                    elif joint_type == self._pb_client.JOINT_SPHERICAL:
                        pose_data.append(mmMath.R2T(mmMath.exp(data[idx : idx + 3])))
                        idx += 3
                    else:
                        raise NotImplementedError()
        return pose_data

    def arrary_to_pose(self, skel, data, T_root_ref=None):
        pose_data = self.array_to_pose_data(skel, data)
        return kinematics.Posture(skel, pose_data)

    def save_states(self):
        return bu.get_bullet_state(self._pb_client, self._body_id)

    def restore_states(self, states):
        bu.set_bullet_state(self._pb_client, self._body_id, states)

    def get_com_and_com_vel(self):
        return bu.compute_com_and_com_vel(
            self._pb_client, self._body_id, self._link_indices
        )

    def get_joint_torques(self):
        return bu.get_joint_torques(self._pb_client, self._body_id, self._joint_indices)

    def get_joint_weights(self, skel):
        """ Get joint weight values form char_info """
        joint_weights = []
        for j in skel.joints:
            idx = self._char_info.bvh_map_inv[j.name]
            if idx is None:
                joint_weights.append(0.0)
            else:
                w = self._char_info.joint_weight[idx]
                joint_weights.append(w)
        return np.array(joint_weights)

    def interaction_mesh_samples(self):
        assert self._char_info.interaction_mesh_samples is not None

        def get_joint_position(j, p_root, Q_root, p_link, Q_link):
            if (
                j == self._char_info.ROOT
                or self._joint_parent_link[j] == self._char_info.ROOT
            ):
                p, Q = p_root, Q_root
            else:
                p, Q = (
                    p_link[self._joint_parent_link[j]],
                    Q_link[self._joint_parent_link[j]],
                )
            T_link_world = mmMath.Qp2T(Q, p)
            T_joint_local = (
                mmMath.I_SE3()
                if j == self._char_info.ROOT
                else self._joint_xform_from_parent_link[j]
            )
            T_joint_world = np.dot(T_link_world, T_joint_local)
            return mmMath.T2p(T_joint_world)

        points = []
        p_root, Q_root, _, _ = self.get_root_state()
        p_link, Q_link, _, _ = self.get_link_states()
        for j1, j2, alpha in self._char_info.interaction_mesh_samples:
            p1 = get_joint_position(j1, p_root, Q_root, p_link, Q_link)
            p2 = (
                p1
                if j2 is None
                else get_joint_position(j2, p_root, Q_root, p_link, Q_link)
            )
            points.append((1.0 - alpha) * p1 + alpha * p2)
        return points

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

    def tracking_control(self, pose, vel, method):
        tracking_method = SimAgent.TrackingControl.from_string(method)
        if method == SimAgent.TrackingControl.NONE:
            return
        joint_indices = []
        target_positions = []
        target_velocities = []
        kps = []
        kds = []
        max_forces = []
        for j in self._joint_indices:
            joint_type = self.get_joint_type(j)
            if joint_type == self._pb_client.JOINT_FIXED:
                continue
            if self._char_info.bvh_map[j] is None:
                target_pos = self._joint_pose_init[j]
                target_vel = self._joint_vel_init[j]
            else:
                if pose is None:
                    T = mmMath.I_SE3()
                else:
                    T = pose.get_transform(self._char_info.bvh_map[j], local=True,)

                if vel is None:
                    w = np.zeros(3)
                else:
                    w = vel.get_angular_velocity(self._char_info.bvh_map[j])

                if joint_type == self._pb_client.JOINT_REVOLUTE:
                    axis = self.get_joint_axis(j)
                    target_pos = np.array(
                        [kinematics.project_rotation_1D(mmMath.T2R(T), axis)]
                    )
                    target_vel = np.array([kinematics.project_angular_vel_1D(w, axis)])
                    max_force = np.array([self._char_info.max_force[j]])
                elif joint_type == self._pb_client.JOINT_SPHERICAL:
                    Q, p = mmMath.T2Qp(T, quat_order_out="xyzw")
                    target_pos = mmMath.post_process_Q(Q)
                    target_vel = w
                    max_force = np.ones(3) * self._char_info.max_force[j]
                else:
                    raise NotImplementedError
            joint_indices.append(j)
            target_positions.append(target_pos)
            target_velocities.append(target_vel)
            if tracking_method == SimAgent.TrackingControl.SPD:
                kps.append(self._char_info.kp[j])
                kds.append(self._char_info.kd[j])
            elif tracking_method == SimAgent.TrackingControl.PD:
                kps.append(1.5 * self._char_info.kp[j])
                kds.append(0.01 * self._char_info.kd[j])
            elif (
                tracking_method == SimAgent.TrackingControl.CPD
                or tracking_method == SimAgent.TrackingControl.CP
                or tracking_method == SimAgent.TrackingControl.V
            ):
                kps.append(self._char_info.cpd_ratio * self._char_info.kp[j])
                kds.append(self._char_info.cpd_ratio * self._char_info.kd[j])
            max_forces.append(max_force)

        if tracking_method == SimAgent.TrackingControl.SPD:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                joint_indices,
                self._pb_client.STABLE_PD_CONTROL,
                targetPositions=target_positions,
                targetVelocities=target_velocities,
                forces=max_forces,
                positionGains=kps,
                velocityGains=kds,
            )
        elif tracking_method == SimAgent.TrackingControl.PD:
            """ Basic PD in Bullet does not support spherical joint yet """
            # self._pb_client.setJointMotorControlMultiDofArray(self._body_id,
            #                                                   joint_indices,
            #                                                   self._pb_client.PD_CONTROL,
            #                                                   targetPositions=target_positions,
            #                                                   targetVelocities=target_velocities,
            #                                                   forces=max_forces,
            #                                                   positionGains=kps,
            #                                                   velocityGains=kds)
            forces = bu.compute_PD_forces(
                pb_client=self._pb_client,
                body_id=self._body_id,
                joint_indices=joint_indices,
                desired_positions=target_positions,
                desired_velocities=target_velocities,
                kps=kps,
                kds=kds,
                max_forces=max_forces,
            )
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                joint_indices,
                self._pb_client.TORQUE_CONTROL,
                forces=forces,
            )
        elif tracking_method == SimAgent.TrackingControl.CPD:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                joint_indices,
                self._pb_client.POSITION_CONTROL,
                targetPositions=target_positions,
                targetVelocities=target_velocities,
                forces=max_forces,
                positionGains=kps,
                velocityGains=kds,
            )
        elif tracking_method == SimAgent.TrackingControl.CP:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                joint_indices,
                self._pb_client.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=max_forces,
                positionGains=kps,
            )
        elif tracking_method == SimAgent.TrackingControl.V:
            self._pb_client.setJointMotorControlMultiDofArray(
                self._body_id,
                joint_indices,
                self._pb_client.VELOCITY_CONTROL,
                targetVelocities=target_velocities,
                forces=max_forces,
                velocityGains=kds,
            )
        else:
            raise NotImplementedError
