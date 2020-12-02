# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np

from fairmotion.utils import constants
from fairmotion.ops import conversions, math
from fairmotion.core.motion import Pose, Motion


class Velocity(object):
    """Velocity class to compute angular and linear velocity of joints. All
    velocities are represented w.r.t. the joint frame (local).

    Attributes:
        pose1, pose2: Objects of Pose class between which velocity is
            computed.
        dt: Time between poses
    """

    def __init__(self, pose1=None, pose2=None, dt=None):
        self.data_local = None
        self.data_global = None
        if pose1:
            assert pose2 and dt
            assert isinstance(pose1, Pose) and isinstance(pose2, Pose)
            self.skel = pose1.skel
            self.data_local, self.data_global = Velocity.compute(
                pose1, pose2, dt,
            )

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
            dR, dp = conversions.T2Rp(np.dot(math.invertT(T1), T2))
            w, v = conversions.R2A(dR) / dt, dp / dt
            data_local.append(np.hstack((w, v)))
            T1 = pose1.get_transform(joint, local=False)
            T2 = pose2.get_transform(joint, local=False)
            dR, dp = conversions.T2Rp(np.dot(math.invertT(T1), T2))
            w, v = conversions.R2A(dR) / dt, dp / dt
            data_global.append(np.hstack((w, v)))
        return np.array(data_local), np.array(data_global)

    def get_all(self, key, local, R_ref=None):
        """Returns both linear and angular velocity stacked together"""
        return np.hstack(
            [
                self.get_angular(key, local, R_ref),
                self.get_linear(key, local, R_ref),
            ]
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
        # TODO: Add documentation
        data_global_new = []
        for joint in self.skel.joints:
            w = self.get_angular(key=joint, local=False)
            v = self.get_linear(key=joint, local=False)
            w = np.dot(R, w)
            v = np.dot(R, v)
            data_global_new.append(np.hstack([w, v]))
        self.data_global = np.array(data_global_new)

    @classmethod
    def interpolate(cls, v1, v2, alpha):
        """Returns interpolated velocity object between two velocity objects.
        Typically, each velocity object is associated with a frame.
        `interpolate` could be used to calculated interpolated angular and
        linear velocity for any frame between them.

        Args:
            v1, v2: Velocity objects associated with frame between which
                interpolated velocity is calculated
            alpha: Value between 0 and 1 denoting the blending ratio. alpha=0
                returns v1, and alpha=1 returns v2
        """
        data_local = math.lerp(v1.data_local, v2.data_local, alpha)
        data_global = math.lerp(v1.data_global, v2.data_global, alpha)
        v = cls()
        v.set_skel(v1.skel)
        v.set_data_local(data_local)
        v.set_data_global(data_global)
        return v


class MotionWithVelocity(Motion):
    """
    Extension of `Motion` class to additionally pre-compute angular and linear
    velocity of joints.

    Instantiating a `MotionWithVelocity` object uses the constructor of the
    `Motion` class and creates an empty velocity list. After populating the
    empty `MotionWithVelocity` object with poses, use `compute_velocities()`
    to make velocity information available.

    To instantiate `MotionWithVelocity` object from a `Motion` object, use
    the `from_motion` method.
    ```
    from fairmotion.data import bvh
    from fairmotion.core.velocity import MotionWithVelocity

    motion = bvh.load(bvh_filename)
    motion_with_velcoty = MotionWithVelocity.from_motion(motion)
    ```

    Attributes:
        name: Optional; String name of MotionWithVelocity object
        skel: Skeleton object of the character associated with the motion
            sequence
        fps: Rendering frequency in Hz
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

        t1 = self.frame_to_time(frame1)
        t2 = self.frame_to_time(frame2)
        alpha = (time - t1) / (t2 - t1)
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
