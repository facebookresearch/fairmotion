import numpy as np
import random

from mocap_processing.processing import operations
from mocap_processing.utils import constants
from mocap_processing.utils import conversions
from mocap_processing.utils import utils


class Joint(object):
    def __init__(self, name=None, dof=3):
        self.name = name if name else f"joint_{random.getrandbits(32)}"
        self.parent_joint = None
        self.child_joint = []
        self.index_child_joint = {}
        self.xform_global = constants.eye_T()
        self.xform_from_parent_joint = constants.eye_T()
        self.info = {"dof": dof}  # set ball joint by default

    def get_child_joint(self, key):
        return self.child_joint[utils.get_index(self.index_child_joint, key)]

    def get_child_joint_recursive(self):
        """
        This could have duplicated joints if there exists loops in the chain
        """
        joints = []
        for j in self.child_joint:
            joints.append(j)
            joints += j.get_all_child_joint()
        return joints

    def add_child_joint(self, joint):
        assert isinstance(joint, Joint)
        assert joint.name not in self.index_child_joint.keys()
        self.index_child_joint[joint.name] = len(self.child_joint)
        self.child_joint.append(joint)
        joint.set_parent_joint(self)

    def set_parent_joint(self, joint):
        assert isinstance(joint, Joint)
        self.parent_joint = joint
        self.xform_global = np.dot(
            self.parent_joint.xform_global, self.xform_from_parent_joint,
        )


class Skeleton(object):
    def __init__(
        self,
        name="skeleton",
        v_up=np.array([0.0, 1.0, 0.0]),
        v_face=np.array([0.0, 0.0, 1.0]),
        v_up_env=np.array([0.0, 1.0, 0.0]),
    ):
        self.name = name
        self.joints = []
        self.index_joint = {}
        self.root_joint = None
        self.num_dofs = 0
        self.v_up = v_up
        self.v_face = v_face
        self.v_up_env = v_up_env

    def num_joint(self):
        return len(self.joints)

    def get_index_joint(self, key):
        return utils.get_index(self.index_joint, key)

    def get_joint(self, key):
        return self.joints[self.get_index_joint(key)]

    def add_joint(self, joint, parent_joint):
        if parent_joint is None:
            assert self.num_joint() == 0
            self.root_joint = joint
        else:
            parent_joint = self.get_joint(parent_joint)
            parent_joint.add_child_joint(joint)
        self.index_joint[joint.name] = len(self.joints)
        self.joints.append(joint)
        self.num_dofs += joint.info["dof"]

    def num_joints(self):
        return len(self.joints)


class Pose(object):
    def __init__(self, skel, data=None):
        """
        Construct Pose for a given skeleton and pose data.
        Pose data must be provided with an np.array of shape (num_joints, 4, 4)
        """
        assert isinstance(skel, Skeleton)
        if data is None:
            data = [constants.eye_T for _ in range(skel.num_joint())]
        assert skel.num_joint() == len(data)
        self.skel = skel
        self.data = data

    def get_transform(self, key, local):
        skel = self.skel
        if local:
            return self.data[skel.get_index_joint(key)]
        else:
            joint = skel.get_joint(key)
            T = np.dot(
                joint.xform_from_parent_joint, self.data[skel.get_index_joint(joint)]
            )
            while joint.parent_joint is not None:
                T_j = np.dot(
                    joint.parent_joint.xform_from_parent_joint,
                    self.data[skel.get_index_joint(joint.parent_joint)],
                )
                T = np.dot(T_j, T)
                joint = joint.parent_joint
            return T

    def set_transform(self, key, T, local, do_ortho_norm=True):
        if local:
            T1 = T
        else:
            T0 = self.skel.get_joint(key).xform_global
            T1 = np.dot(operations.invertT(T0), T)
        if do_ortho_norm:
            """
            This insures that the rotation part of
            the given transformation is valid
            """
            Q, p = conversions.T2Qp(T1)
            Q = operations.post_process_Q(Q, normalize=True, half_space=False)
            T1 = conversions.Qp2T(Q, p)
        self.data[self.skel.get_index_joint(key)] = T1

    def get_root_transform(self):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        return self.get_transform(root_idx, local=False)

    def set_root_transform(self, T, local):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        self.set_transform(root_idx, T, local)

    def get_facing_transform(self):
        d, p = self.get_facing_direction_position()
        z = d
        y = self.skel.v_up_env
        x = np.cross(y, z)
        return conversions.Rp2T(np.array([x, y, z]).transpose(), p)

    def get_facing_position(self):
        d, p = self.get_facing_direction_position()
        return p

    def get_facing_direction(self):
        d, p = self.get_facing_direction_position()
        return d

    def get_facing_direction_position(self):
        R, p = conversions.T2Rp(self.get_root_transform())
        d = np.dot(R, self.skel.v_face)
        d = d - operations.projectionOnVector(d, self.skel.v_up_env)
        p = p - operations.projectionOnVector(p, self.skel.v_up_env)
        return d / np.linalg.norm(d), p

    def to_matrix(self, local=True):
        """
        Returns pose data in transformation matrix format, with shape
        (num_joints, 4, 4)
        """
        transforms = []
        for joint in self.skel.joints:
            transforms.append(self.get_transform(joint, local))
        return np.array(transforms)

    @classmethod
    def from_matrix(cls, data, skel, local=True):
        """
        Expects pose data in transformation matrix format, with shape
        (num_joints, 4, 4)
        """
        num_joints, T_0, T_1 = data.shape
        assert num_joints == skel.num_joints(), "Data for all joints not provided"
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "mocap_processing.utils.constants.eye_T() for template identity "
            "matrix"
        )
        pose = cls(skel)
        for joint_id in range(len(skel.joints)):
            pose.set_transform(joint_id, data[joint_id], local)
        return pose


def interpolate_pose(alpha, pose1, pose2):
    skel = pose1.skel
    data = []
    for j in skel.joints:
        R1, p1 = conversions.T2Rp(pose1.get_transform(j, local=True))
        R2, p2 = conversions.T2Rp(pose2.get_transform(j, local=True))
        R, p = (
            operations.slerp(R1, R2, alpha),
            operations.lerp(p1, p2, alpha),
        )
        data.append(conversions.Rp2T(R, p))
    return Pose(pose1.skel, data)


class Motion(object):
    def __init__(
        self, name="motion", skel=None, fps=60,
    ):
        self.name = name
        self.skel = skel
        self.poses = []
        self.fps = fps
        self.info = {}

    def clear(self):
        self.poses = []
        self.info = {}

    def set_skeleton(self, skel):
        self.skel = skel

    def add_one_frame(self, t, pose_data):
        self.poses.append(Pose(self.skel, pose_data))

    def frame_to_time(self, frame):
        frame = np.clip(frame, 0, len(self.poses - 1))
        return self.fps * frame

    def time_to_frame(self, time):
        return int(time * self.fps)

    def get_pose_by_frame(self, frame):
        assert frame < self.num_frames()
        return self.poses[frame]

    def get_pose_by_time(self, time):
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.poses[frame1]
        if np.isclose(time % (1.0/self.fps), 0):
            return self.poses[frame1]

        t1 = int(time/self.fps)*float(1/self.fps)
        t2 = t1 + float(1/self.fps)
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)
        return interpolate_pose(alpha, self.poses[frame1], self.poses[frame2])

    def num_frames(self):
        return len(self.poses)

    def length(self):
        return  (len(self.poses) - 1)/self.fps

    def to_matrix(self, local=True):
        """
        Returns pose data in transformation matrix format, with shape
        (seq_len, num_joints, 4, 4)
        """
        data = []
        for pose in self.poses:
            data.append(pose.to_matrix(local))
        return np.array(data)

    def rotations(self, local=True):
        """
        Returns joint rotations in rotation matrix format, with shape
        (seq_len, num_joints, 3, 3)
        """
        return self.to_matrix(local)[..., :3, :3]

    def positions(self, local=True):
        """
        Returns joint positions with shape (seq_len, num_joints, 3)
        """
        return self.to_matrix(local)[..., :3, 3]

    @classmethod
    def from_matrix(cls, data, skel, local=True, fps=None):
        """
        Expects pose data in transformation matrix format, with shape
        (seq_len, num_joints, 4, 4)
        """
        assert data.ndim == 4, (
            "Data must be provided in transformation matrix format, with shape"
            " (seq_len, num_joints, 4, 4)"
        )
        seq_len, num_joints, T_0, T_1 = data.shape
        assert num_joints == skel.num_joints(), "Data for all joints not provided"
        assert T_0 == 4 and T_1 == 4, (
            "Data not provided in 4x4 transformation matrix format. Use "
            "mocap_processing.utils.constants.eye_T() for template identity "
            "matrix"
        )
        if fps is None:
            fps = 60
        motion = cls(skel=skel, fps=fps)
        for pose_data in data:
            pose = Pose.from_matrix(pose_data, skel, local)
            motion.poses.append(pose)
        return motion


class Velocity(object):
    ''' 
    This contains linear and angluar velocity of joints.
    All velocities are represented w.r.t. the joint frame.
    To get the global velocity, you should give the frame 
    that corresponds to the velocity.
    '''
    def __init__(self, pose1=None, pose2=None, dt=None):
        self.data_local = None
        self.data_global = None
        if pose1:
            assert pose2 and dt
            assert isinstance(pose1, Pose) and isinstance(pose2, Pose)
            self.skel = pose1.skel
            self.data_local, self.data_global = \
                Velocity.compute(pose1, pose2, dt)
    def set(self, skel=None, data_local=None, data_global=None):
        if skel is not None: 
            self.skel = skel
        if data_local is not None: 
            self.data_local = data_local
        if data_global is not None: 
            self.data_global = data_global
    @classmethod
    def compute(cls, pose1, pose2, dt):
        assert pose1.skel == pose2.skel
        data_local = []
        data_global = []
        assert dt > constants.EPSILON
        for joint in pose1.skel.joints:
            T1 = pose1.get_transform(joint, local=True)
            T2 = pose2.get_transform(joint, local=True)
            dR, dp = conversions.T2Rp(np.dot(operations.invertT(T1), T2))
            w, v = conversions.R2A(dR)/dt, dp/dt
            data_local.append(np.hstack((w,v)))
            T1 = pose1.get_transform(joint, local=False)
            T2 = pose2.get_transform(joint, local=False)
            dR, dp = conversions.T2Rp(np.dot(operations.invertT(T1), T2))
            w, v = conversions.R2A(dR)/dt, dp/dt
            data_global.append(np.hstack((w,v)))
        return np.array(data_local), np.array(data_global)
    def get_all(self, key, local, R_ref=None):
        return np.hstack([self.get_angular(key, local, R_ref), 
                          self.get_linear(key, local, R_ref)])
    def get_angular(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        w = data[self.skel.get_index_joint(key),0:3]
        if R_ref is not None: w = np.dot(R_ref, w)
        return w
    def get_linear(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        v = data[self.skel.get_index_joint(key),3:6]
        if R_ref is not None: v = np.dot(R_ref, v)
        return v
    def rotate(self, R):
        data_global_new = []
        for joint in self.skel.joints:
            w = self.get_angular(key, local=False)
            v = self.get_linear(key, local=False)
            w = np.dot(R, w)
            v = np.dot(R, v)
            data_global_new.append(np.hstack([w, v]))
        self.data_global = np.array(data_global_new)
    @classmethod
    def interpolate(cls, v1, v2, alpha):
        data_local = operations.lerp(v1.data_local, v2.data_local, alpha)
        data_global = operations.lerp(v1.data_global, v2.data_global, alpha)
        v = cls()
        v.set(v1.skel, data_local, data_global)
        return v


class MotionWithVelocity(Motion):
    '''
    This is an exteded motion class where precomputed velocities 
    are available to access.
    '''
    def __init__(self, name="motion", skel=None, fps=60):
        super().__init__(name, skel, fps)
        self.vels = []

    def compute_velocities(self):
        self.vels = []
        self.vels += self._compute_velocities()

    def _compute_velocities(self, frame_start=None, frame_end=None):
        vels = []
        if frame_start is None: frame_start = 0
        if frame_end is None: frame_end = self.num_frames()
        for i in range(frame_start, frame_end):
            frame1 = max(0, (i-1))
            frame2 = min(self.num_frames()-1, (i+1))
            dt = (frame2 - frame1) / float(self.fps)
            pose1 = self.get_pose_by_frame(frame1)
            pose2 = self.get_pose_by_frame(frame2)
            vels.append(Velocity(pose1, pose2, dt))
            # print(vels[-1].get_linear('root', local=True))
        return vels
    
    def get_velocity_by_time(self, time):
        assert len(self.vels) > 0, (
            "Velocity was not computed yet.",
            "Please call self.compute_velocities() first")
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.vels[frame1]
        if np.isclose(time % (1.0/self.fps), 0):
            return self.vels[frame1]

        t1 = int(time/self.fps)*float(1/self.fps)
        t2 = t1 + float(1/self.fps)
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)

        v1 = self.get_velocity_by_frame(frame1)
        v2 = self.get_velocity_by_frame(frame2)

        return Velocity.interpolate(v1, v2, alpha)

    def get_velocity_by_frame(self, frame):
        assert len(self.vels) > 0, (
            "Velocity was not computed yet.",
            "Please call self.compute_velocities() first")
        assert frame < self.num_frames()
        return self.vels[frame]

    @classmethod
    def from_motion(cls, motion):
        m = cls(motion.name, motion.skel, motion.fps)
        m.poses = motion.poses
        m.info = motion.info
        m.compute_velocities()
        return m