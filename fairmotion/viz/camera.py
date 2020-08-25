# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from fairmotion.utils import constants
from fairmotion.ops import conversions


class Camera(object):
    """Camera class for the visualizer.

    Attributes:
        pos: Position of the camera in global coordinates
        origin: Point in global coordinates that the camera is pointing at
        vup: Vertical axis. Defaults to y-axis
        fov: Field of view in degrees
    """
    def __init__(self, pos, origin, vup=np.array([0.0, 1.0, 0.0]), fov=45.0):
        self.pos = pos
        self.origin = origin
        self.vup = vup
        self.fov = fov

    def get_cam_rotation(self):
        def _get_cam_rotation(p_cam, p_obj, vup):
            z = p_obj - p_cam
            z /= np.linalg.norm(z)
            x = np.cross(vup, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            return np.array([x, y, z]).transpose()
        return _get_cam_rotation(self.pos, self.origin, self.vup)

    def translate(self, dp, frame_local=False):
        R = self.get_cam_rotation() if frame_local else constants.eye_R()
        dt = np.dot(R, dp)
        self.pos += dt
        self.origin += dt

    def rotate(self, dx, dy, dz):
        R = self.get_cam_rotation()
        pos_local = np.dot(R.transpose(), self.pos-self.origin)
        dR = np.dot(
            np.dot(
                conversions.Ax2R(dx),
                conversions.Ay2R(dy)
            ),
            conversions.Az2R(dz)
        )
        R = np.dot(R, dR)
        # self.vup = R[:, 1]
        pos_new = self.origin + np.dot(R, pos_local)
        dp = pos_new-self.origin
        dp /= np.linalg.norm(dp)
        if (
            np.linalg.norm(dp-self.vup) > 0.2 and
            np.linalg.norm(dp+self.vup) > 0.2
        ):
            self.pos = pos_new

    def zoom(self, gamma, l_min=0.5):
        vl = self.pos - self.origin
        length = np.linalg.norm(vl)
        self.pos = self.origin + max(l_min, gamma*length) * (vl / length)

    def get_transform_flat(self):
        R = self.get_cam_rotatioin()
        R = R.transpose()
        p = self.pos
        return list(conversions.Rp2T(R, p).ravel())
