import numpy as np

from mocap_processing.processing import operations
from mocap_processing.utils import constants
from mocap_processing.utils import conversions
from mocap_processing.motion.motion import Pose, Motion


class Velocity(object):
    """ 
    This contains linear and angluar velocity of joints.
    All velocities are represented w.r.t. the joint frame.
    To get the global velocity, you should give the frame 
    that corresponds to the velocity.
    """

    def __init__(self, pose1=None, pose2=None, dt=None):
        self.data_local = None
        self.data_global = None
        if pose1:
            assert pose2 and dt
            assert isinstance(pose1, Pose) and isinstance(pose2, Pose)
            self.skel = pose1.skel
            self.data_local, self.data_global = Velocity.compute(pose1, pose2, dt)

    def set_skel(self, skel):
        self.skel = skel

    def set_data_local(self, data):
        self.data_local = data

    def set_data_global(self, data):
        self.data_global = data

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
            w, v = conversions.R2A(dR) / dt, dp / dt
            data_local.append(np.hstack((w, v)))
            T1 = pose1.get_transform(joint, local=False)
            T2 = pose2.get_transform(joint, local=False)
            dR, dp = conversions.T2Rp(np.dot(operations.invertT(T1), T2))
            w, v = conversions.R2A(dR) / dt, dp / dt
            data_global.append(np.hstack((w, v)))
        return np.array(data_local), np.array(data_global)

    def get_all(self, key, local, R_ref=None):
        return np.hstack(
            [self.get_angular(key, local, R_ref), self.get_linear(key, local, R_ref)]
        )

    def get_angular(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        w = data[self.skel.get_index_joint(key), 0:3]
        if R_ref is not None:
            w = np.dot(R_ref, w)
        return w

    def get_linear(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        v = data[self.skel.get_index_joint(key), 3:6]
        if R_ref is not None:
            v = np.dot(R_ref, v)
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
        v.set_skel(v1.skel)
        v.set_data_local(data_local)
        v.set_data_global(data_global)
        return v


class MotionWithVelocity(Motion):
    """
    This is an exteded motion class where precomputed velocities 
    are available to access.ss
    """

    def __init__(self, name="motion", skel=None, fps=60):
        super().__init__(name, skel, fps)
        self.vels = []

    def compute_velocities(self):
        self.vels = self._compute_velocities()

    def _compute_velocities(self, frame_start=None, frame_end=None):
        vels = []
        if frame_start is None:
            frame_start = 0
        if frame_end is None:
            frame_end = self.num_frames()
        for i in range(frame_start, frame_end):
            frame1 = max(0, (i - 1))
            frame2 = min(self.num_frames() - 1, (i + 1))
            dt = (frame2 - frame1) / float(self.fps)
            pose1 = self.get_pose_by_frame(frame1)
            pose2 = self.get_pose_by_frame(frame2)
            vels.append(Velocity(pose1, pose2, dt))
            # print(vels[-1].get_linear('root', local=True))
        return vels

    def get_velocity_by_time(self, time):
        assert len(self.vels) > 0, (
            "Velocity was not computed yet.",
            "Please call self.compute_velocities() first",
        )
        time = np.clip(time, 0, self.length())
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1 + 1, self.num_frames() - 1)
        if frame1 == frame2:
            return self.vels[frame1]
        if np.isclose(time % (1.0 / self.fps), 0):
            return self.vels[frame1]

        t1 = int(time / self.fps) * float(1 / self.fps)
        t2 = t1 + float(1 / self.fps)
        alpha = np.clip((time - t1) / (t2 - t1), 0.0, 1.0)

        v1 = self.get_velocity_by_frame(frame1)
        v2 = self.get_velocity_by_frame(frame2)

        return Velocity.interpolate(v1, v2, alpha)

    def get_velocity_by_frame(self, frame):
        assert len(self.vels) > 0, (
            "Velocity was not computed yet.",
            "Please call self.compute_velocities() first",
        )
        assert frame < self.num_frames()
        return self.vels[frame]

    @classmethod
    def from_motion(cls, m):
        mv = cls(m.name, m.skel, m.fps)
        mv.poses = m.poses
        mv.info = m.info
        mv.compute_velocities()
        return mv
