# Copyright (c) Facebook, Inc. and its affiliates.

"""
This code is a Python reimplementation, authored by Jungdam Won, 
of the C++ PFNN inference code from Daniel Holden's website
https://theorangeduck.com/page/phase-functioned-neural-networks-character-control

LICENSE

This code and data is free for academic and non-commercial purposes but we 
would ask that you please include the following citations in any published work 
which uses this code or data.


    @inproceedings{Holden:2017:PFNN,
     author = {Holden, Daniel and Komura, Taku, and Saito, Jun},
     title = {Phase-Functioned Neural Networks for Character Control},
     booktitle = {SIGGRAPH 2017},
     year = {2017},
    }
    
    
If you have any questions, or wish to enquire about commerical licenses, please 
contact the original author at `contact@theorangeduck.com`.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"

import numpy as np
import math

from enum import Enum

import ctypes
import sdl2 as sdl

import copy
import pickle
import gzip

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from fairmotion.data import bvh
from fairmotion.ops import conversions, math as fm_math
from fairmotion.utils import constants
from fairmotion.viz import camera, gl_render, glut_viewer


def elu(x):
    return np.maximum(x, 0) + np.exp(np.minimum(x, 0)) - 1


def linear(y0, y1, mu):
    return (1.0 - mu) * y0 + (mu) * y1


def cubic(y0, y1, y2, y3, mu):
    o = (
        (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu
        + (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu
        + (-0.5 * y0 + 0.5 * y2) * mu
        + (y1)
    )
    return o


def mix_directions(x, y, mu):
    R1 = conversions.Ay2R(math.atan2(x[0], x[2]))
    R2 = conversions.Ay2R(math.atan2(y[0], y[2]))
    R = fm_math.slerp(R1, R2, mu)
    return R[:, 2]


_filenames = {}
_filenames["mode"] = "constant"
_filenames["num_point"] = 50
_filenames["x_mean"] = "data/learning/Xmean.bin"
_filenames["y_mean"] = "data/learning/Ymean.bin"
_filenames["x_std"] = "data/learning/Xstd.bin"
_filenames["y_std"] = "data/learning/Ystd.bin"
for i in range(_filenames["num_point"]):
    _filenames["W0_%03i" % (i)] = "data/learning/W0_%03i.bin" % (i)
    _filenames["W1_%03i" % (i)] = "data/learning/W1_%03i.bin" % (i)
    _filenames["W2_%03i" % (i)] = "data/learning/W2_%03i.bin" % (i)
    _filenames["b0_%03i" % (i)] = "data/learning/b0_%03i.bin" % (i)
    _filenames["b1_%03i" % (i)] = "data/learning/b1_%03i.bin" % (i)
    _filenames["b2_%03i" % (i)] = "data/learning/b2_%03i.bin" % (i)


class Areas(object):
    def __init__(self):
        """Crouch area"""
        self.crouch_pos = []
        self.crouch_size = []
        self.CROUCH_WAVE = 50.0
        """ Jump area """
        self.jump_pos = []
        self.jump_size = []
        self.jump_falloff = []
        """ Wall area """
        self.wall_start = []
        self.wall_stop = []
        self.wall_width = []

    def clear(self):
        self.crouch_pos = []
        self.crouch_size = []
        self.jump_pos = []
        self.jump_size = []
        self.jump_falloff = []
        self.wall_start = []
        self.wall_stop = []
        self.wall_width = []

    def add_wall(self, start, stop, width):
        self.wall_start.append(start)
        self.wall_stop.append(stop)
        self.wall_width.append(width)

    def add_crouch(self, pos, size):
        self.crouch_pos.append(pos)
        self.crouch_size.append(size)

    def add_jump(self, pos, size, falloff):
        self.jump_pos.append(pos)
        self.jump_size.append(size)
        self.jump_falloff.append(falloff)

    def num_walls(self):
        return len(self.wall_start)

    def num_crouches(self):
        return len(self.crouch_pos)

    def num_jumps(self):
        return len(self.jump_pos)

    def sample_height(self, pos):
        for j in range(self.num_jumps()):
            d = pos - self.jump_pos[j]
            d[1] = 0.0
            if np.linalg.norm(d) < self.jump_size[j]:
                return 30.0
        return 0.0


class Controller(object):
    def __init__(self, filenames=None):
        if filenames is None:
            filenames = _filenames

        self.mode = filenames["mode"]

        if self.mode == "constant":
            assert filenames["num_point"] == 50
        elif self.mode == "linear":
            assert filenames["num_point"] == 10
        elif self.mode == "cubic":
            assert filenames["num_point"] == 4
        else:
            raise NotImplementedError

        self.x_mean = np.fromfile(filenames["x_mean"], dtype=np.float32)
        self.y_mean = np.fromfile(filenames["y_mean"], dtype=np.float32)
        self.x_std = np.fromfile(filenames["x_std"], dtype=np.float32)
        self.y_std = np.fromfile(filenames["y_std"], dtype=np.float32)
        self.W0, self.W1, self.W2 = [], [], []
        self.b0, self.b1, self.b2 = [], [], []

        XDIM = self.x_mean.size
        YDIM = self.y_mean.size
        HDIM = 512
        for i in range(filenames["num_point"]):
            self.W0.append(
                (
                    np.reshape(
                        np.fromfile(filenames["W0_%03i" % (i)], dtype=np.float32),
                        (HDIM, XDIM),
                    )
                )
            )
            self.W1.append(
                (
                    np.reshape(
                        np.fromfile(filenames["W1_%03i" % (i)], dtype=np.float32),
                        (HDIM, HDIM),
                    )
                )
            )
            self.W2.append(
                (
                    np.reshape(
                        np.fromfile(filenames["W2_%03i" % (i)], dtype=np.float32),
                        (YDIM, HDIM),
                    )
                )
            )
            self.b0.append((np.fromfile(filenames["b0_%03i" % (i)], dtype=np.float32)))
            self.b1.append((np.fromfile(filenames["b1_%03i" % (i)], dtype=np.float32)))
            self.b2.append((np.fromfile(filenames["b2_%03i" % (i)], dtype=np.float32)))

        self.x_p = np.zeros_like(self.x_mean)
        self.y_p = np.zeros_like(self.y_mean)

    def predict(self, phase):
        x = self.x_p
        x = (x - self.x_mean) / self.x_std

        if self.mode == "constant":
            pindex = int(phase / (2 * math.pi) * 50)
            W0p = self.W0[pindex]
            W1p = self.W1[pindex]
            W2p = self.W2[pindex]
            b0p = self.b0[pindex]
            b1p = self.b1[pindex]
            b2p = self.b2[pindex]
        elif self.mode == "linear":
            pamount = math.fmod((phase / (2 * math.pi)) * 10, 1.0)
            pindex1 = int((phase / (2 * math.pi)) * 10)
            pindex2 = (pindex1 + 1) % 10
            W0p = linear(self.W0[pindex1], self.W0[pindex2], pamount)
            W1p = linear(self.W1[pindex1], self.W1[pindex2], pamount)
            W2p = linear(self.W2[pindex1], self.W2[pindex2], pamount)
            b0p = linear(self.b0[pindex1], self.b0[pindex2], pamount)
            b1p = linear(self.b1[pindex1], self.b1[pindex2], pamount)
            b2p = linear(self.b2[pindex1], self.b2[pindex2], pamount)
        elif self.mode == "cubic":
            pamount = math.fmod((phase / (2 * math.pi)) * 4, 1.0)
            pindex1 = int((phase / (2 * math.pi)) * 4)
            pindex0 = (pindex1 + 3) % 4
            pindex2 = (pindex1 + 1) % 4
            pindex3 = (pindex1 + 2) % 4
            W0p = cubic(
                self.W0[pindex0],
                self.W0[pindex1],
                self.W0[pindex2],
                self.W0[pindex3],
                pamount,
            )
            W1p = cubic(
                self.W1[pindex0],
                self.W1[pindex1],
                self.W1[pindex2],
                self.W1[pindex3],
                pamount,
            )
            W2p = cubic(
                self.W2[pindex0],
                self.W2[pindex1],
                self.W2[pindex2],
                self.W2[pindex3],
                pamount,
            )
            b0p = cubic(
                self.b0[pindex0],
                self.b0[pindex1],
                self.b0[pindex2],
                self.b0[pindex3],
                pamount,
            )
            b1p = cubic(
                self.b1[pindex0],
                self.b1[pindex1],
                self.b1[pindex2],
                self.b1[pindex3],
                pamount,
            )
            b2p = cubic(
                self.b2[pindex0],
                self.b2[pindex1],
                self.b2[pindex2],
                self.b2[pindex3],
                pamount,
            )
        else:
            raise NotImplementedError()

        H0 = elu(np.dot(W0p, x) + b0p)
        H1 = elu(np.dot(W1p, H0) + b1p)
        y = np.dot(W2p, H1) + b2p

        self.y_p = (y * self.y_std) + self.y_mean


class Character(object):
    def __init__(self):
        self.JOINT_NUM = 31

        self.JOINT_ROOT_L = 1
        self.JOINT_HIP_L = 2
        self.JOINT_KNEE_L = 3
        self.JOINT_HEEL_L = 4
        self.JOINT_TOE_L = 5

        self.JOINT_ROOT_R = 6
        self.JOINT_HIP_R = 7
        self.JOINT_KNEE_R = 8
        self.JOINT_HEEL_R = 9
        self.JOINT_TOE_R = 10

        self.phase = 0.0
        self.strafe_amount = 0.0
        self.strafe_target = 0.0
        self.crouched_amount = 0.0
        self.crouched_target = 0.0
        self.responsive = 0.0

        self.joint_positions = [np.zeros(3) for j in range(self.JOINT_NUM)]
        self.joint_velocities = [np.zeros(3) for j in range(self.JOINT_NUM)]
        self.joint_rotations = [np.eye(3) for j in range(self.JOINT_NUM)]

        self.joint_anim_xform = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.joint_rest_xform = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.joint_mesh_xform = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.joint_global_rest_xform = [
            constants.eye_T() for j in range(self.JOINT_NUM)
        ]
        self.joint_global_anim_xform = [
            constants.eye_T() for j in range(self.JOINT_NUM)
        ]

        self.joint_parents = [0] * self.JOINT_NUM

        self.link_length = [1.0 for j in range(self.JOINT_NUM)]
        self.link_global_rest_xform = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.link_rest_xform = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.link_xform_offset = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.joint_global_anim_xform_by_fk = [
            constants.eye_T() for j in range(self.JOINT_NUM)
        ]

        self.joint_xform_by_ik = [constants.eye_T() for j in range(self.JOINT_NUM)]
        self.joint_global_xform_by_ik = [
            constants.eye_T() for j in range(self.JOINT_NUM)
        ]

        self.load()

    def reset(self):
        self.phase = 0.0
        self.strafe_amount = 0.0
        self.strafe_target = 0.0
        self.crouched_amount = 0.0
        self.crouched_target = 0.0
        self.responsive = 0.0
        self.joint_anim_xform = self.joint_rest_xform.copy()
        self.forward_kinematics()

    def load(self):
        self.joint_parents = np.fromfile(
            "data/character/character_parents.bin", dtype=np.float32
        ).astype(int)
        self.joint_rest_xform = np.fromfile(
            "data/character/character_xforms.bin", dtype=np.float32
        ).reshape((-1, 4, 4))
        self.joint_anim_xform = self.joint_rest_xform.copy()
        self.forward_kinematics(rest_xform=True)
        (
            self.link_length,
            self.link_rest_xform,
            self.link_global_rest_xform,
        ) = self.compute_link_info()

    def compute_link_info(self):
        link_length = [0.0 for i in range(self.JOINT_NUM)]
        link_rest_xform = [np.eye(4) for i in range(self.JOINT_NUM)]
        link_global_rest_xform = [np.eye(4) for i in range(self.JOINT_NUM)]
        for i in range(self.JOINT_NUM):
            j = self.joint_parents[i]
            if j == -1:
                continue
            else:
                T1 = self.joint_global_rest_xform[j]
                T2 = self.joint_global_rest_xform[i]
                p = T1[:3, 3] - T2[:3, 3]
                l = np.linalg.norm(p)
                link_length[i] = l
                if l >= 1.0e-5:
                    y = p / l
                    x = np.cross(y, T2[:3, 0])
                    x /= np.linalg.norm(x)
                    z = np.cross(x, y)
                    z /= np.linalg.norm(z)
                    R = np.array([x, y, z]).transpose()
                else:
                    R = np.eye(3)
                T = conversions.Rp2T(R, T2[:3, 3])
                link_global_rest_xform[i] = T
                link_rest_xform[i] = np.dot(fm_math.invertT(T2), T)
        return link_length, link_rest_xform, link_global_rest_xform

    def forward_kinematics(self, anim_xform=True, rest_xform=False):
        if anim_xform is False and rest_xform is False:
            return
        for i in range(self.JOINT_NUM):
            if anim_xform:
                self.joint_global_anim_xform[i] = self.joint_anim_xform[i].copy()
            if rest_xform:
                self.joint_global_rest_xform[i] = self.joint_rest_xform[i].copy()
            j = self.joint_parents[i]
            while j != -1:
                if anim_xform:
                    self.joint_global_anim_xform[i] = np.dot(
                        self.joint_anim_xform[j], self.joint_global_anim_xform[i]
                    )
                if rest_xform:
                    self.joint_global_rest_xform[i] = np.dot(
                        self.joint_rest_xform[j], self.joint_global_rest_xform[i]
                    )
                j = self.joint_parents[j]
            self.joint_mesh_xform[i] = np.dot(
                self.joint_global_anim_xform[i],
                fm_math.invertT(self.joint_global_rest_xform[i]),
            )

    def forward_kinematics_correct(self):
        for i in range(self.JOINT_NUM):
            R = conversions.T2R(self.joint_anim_xform[i])
            p = (
                conversions.T2p(self.joint_rest_xform[i])
                if i != 0
                else conversions.T2p(self.joint_anim_xform[i])
            )
            self.joint_global_anim_xform_by_fk[i] = conversions.Rp2T(R, p)
            j = self.joint_parents[i]
            while j != -1:
                R = conversions.T2R(self.joint_anim_xform[j])
                p = (
                    conversions.T2p(self.joint_rest_xform[j])
                    if j != 0
                    else conversions.T2p(self.joint_anim_xform[j])
                )
                self.joint_global_anim_xform_by_fk[i] = np.dot(
                    conversions.Rp2T(R, p), self.joint_global_anim_xform_by_fk[i]
                )
                j = self.joint_parents[j]

    def inverse_kinematics(self, ik_model):
        T0_inv = fm_math.invertT(self.joint_global_anim_xform[0])
        x = np.hstack(
            [
                conversions.T2p(np.dot(T0_inv, self.joint_global_anim_xform[i]))
                for i in range(1, self.JOINT_NUM)
            ]
        )
        y_pred = ik_model.eval(x)
        y_pred = np.hstack([np.zeros(3), y_pred])
        y_pred = np.reshape(y_pred, (-1, 3))

        xform = self.JOINT_NUM * [None]

        for i in range(self.JOINT_NUM):
            if i == 0:
                T = self.joint_global_anim_xform[i].copy()
                self.joint_xform_by_ik[i] = T
            else:
                """
                Transform w.r.t. parent joint.
                joint_rest_xform has rotation compement, which is weired
                because 'bvh' does not have inherently
                """
                self.joint_xform_by_ik[i] = conversions.R2T(conversions.A2R(y_pred[i]))
                T = np.dot(
                    conversions.p2T(conversions.T2p(self.joint_rest_xform[i])),
                    self.joint_xform_by_ik[i],
                )
                j = self.joint_parents[i]
                while j != -1:
                    if xform[j] is not None:
                        T = np.dot(xform[j], T)
                        break
                    else:
                        # T0 = np.dot(self.joint_rest_xform[j], conversions.R2T(conversions.A2R(y_pred[j])))
                        T0 = conversions.p2T(conversions.T2p(self.joint_rest_xform[j]))
                        T0 = np.dot(T0, conversions.R2T(conversions.A2R(y_pred[j])))
                        T = np.dot(T0, T)
                        j = self.joint_parents[j]
            xform[i] = T
        self.joint_global_xform_by_ik = xform


class Trajectory(object):
    def __init__(self):
        self.phase = 0.0
        self.length = 120
        self.width = 25
        self.target_dir = np.array([0.0, 0.0, 1.0])
        self.target_vel = np.array([0.0, 0.0, 2.5])
        self.positions = [np.zeros(3) for i in range(self.length)]
        self.rotations = [np.eye(3) for i in range(self.length)]
        self.directions = [self.rotations[i][:, 2].copy() for i in range(self.length)]
        self.heights = [0.0] * self.length

        self.gait_stand = [0.0] * self.length
        self.gait_walk = [1.0] * self.length
        self.gait_jog = [0.0] * self.length
        self.gait_crouch = [0.0] * self.length
        self.gait_jump = [0.0] * self.length
        self.gait_bump = [0.0] * self.length

    def reset(self, root_pos, root_rot):
        self.target_dir = np.array([0.0, 0.0, 1.0])
        self.target_vel = np.array([0.0, 0.0, 2.5])
        for i in range(self.length):
            self.positions[i] = root_pos.copy()
            self.rotations[i] = root_rot.copy()
            self.directions[i] = self.rotations[i][:, 2].copy()
            self.heights[i] = 0.0  # We currently assume a flat plane
            self.gait_stand[i] = 0.0
            self.gait_walk[i] = 1.0
            self.gait_jog[i] = 0.0
            self.gait_crouch[i] = 0.0
            self.gait_jump[i] = 0.0
            self.gait_bump[i] = 0.0
        dt = 30.0 / (self.length // 2)
        p = self.positions[self.length // 2] + np.zeros(3)
        for i in range(self.length // 2, self.length):
            self.positions[i] = p.copy()
            p += self.target_vel * dt
        p = self.positions[self.length // 2] + np.zeros(3)
        for i in range(0, self.length // 2):
            self.positions[self.length // 2 - i - 1] = p.copy()
            p -= self.target_vel * dt

    def get_base_xform(self):
        R = self.rotations[self.length // 2]
        p = self.positions[self.length // 2]
        return conversions.Rp2T(R, p)


class Options(object):
    def __init__(self):
        self.invert_y = False
        self.enable_ik = True
        self.display_debug = True
        self.display_debug_heights = True
        self.display_debug_joints = False
        self.display_debug_pfnn = False
        self.display_hud_options = True
        self.display_hud_stick = True
        self.display_hud_speed = True
        self.display_areas_jump = False
        self.display_areas_walls = False

        self.display_scale = 2.0

        self.extra_direction_smooth = 0.9
        self.extra_velocity_smooth = 0.9
        self.extra_strafe_smooth = 0.9
        self.extra_crouched_smooth = 0.9
        self.extra_gait_smooth = 0.1
        self.extra_joint_smooth = 0.5


class Momentum(object):
    def __init__(self, x=0.0, dx=0.3):
        self.x_init = x
        self.dx_init = dx
        self.reset()

    def reset(self):
        self.x = self.x_init
        self.dx = self.dx_init
        self.mmt_max = 0.5
        self.dmmt_max = 0.1
        self.mmt = 0.0  # np.random.uniform(-self.mmt_max, self.mmt_max)

    def update(self):
        sign = np.random.uniform(-1.0, 1.0) - self.mmt / self.mmt_max
        dmmt = np.random.uniform(0.0, self.dmmt_max)
        if sign < 0.0:
            dmmt *= -1.0
        self.mmt = np.clip(self.mmt + dmmt, -self.mmt_max, self.mmt_max)
        self.x = np.clip(self.x + self.mmt * self.dx, -1.0, 1.0)

    def get(self):
        return self.x


class Trigger(object):
    def __init__(
        self,
        dt,
        init_state=False,
        prob=0.2,
        decision_duration=1.0,
        min_duration=1.0,
        max_duration=5.0,
    ):
        self.dt = dt
        self.init_state = init_state
        self.prob = prob
        self.state = init_state
        self.decision_duration = decision_duration
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.duration = 0.0
        self.reset()

    def reset(self):
        self.state = self.init_state
        if self.state:
            self.duration = np.random.uniform(self.min_duration, self.max_duration)
        else:
            self.duration = 0.0

    def update(self):
        self.duration -= self.dt
        if self.duration <= 0.0:
            if self.prob > np.random.uniform(0.0, 1.0):
                self.duration = np.random.uniform(self.min_duration, self.max_duration)
                self.state = True
            else:
                self.duration = self.decision_duration
                self.state = False

    def triggered(self):
        return self.state


class Command(object):
    def __init__(self, trajectory, record=False, history_file=None):
        self.scale = 32768
        self.trajectory = trajectory
        self.record = record
        self.recorded_command = False
        if history_file is not None:
            assert not record
            self.recorded_command = True
            self.load_history(history_file)

    def reset(self):
        self.command = {}
        self.command["target_dir"] = self.trajectory.directions[
            self.trajectory.length // 2
        ]
        self.command["trigger_speed"] = -self.scale
        self.command["trigger_strafe"] = -self.scale
        self.command["trigger_crouch"] = -self.scale
        self.command["trigger_stop"] = -self.scale
        self.command["trigger_obstacle"] = -self.scale
        self.command["trigger_jump"] = -self.scale
        self.command["x_move"] = 0
        self.command["y_move"] = 0
        self.command["x_vel"] = 0
        self.command["y_vel"] = 0
        self.command["zoom_i"] = 0
        self.command["zoom_o"] = 0
        if not self.recorded_command:
            self.command_history = []
        if self.recorded_command:
            self.command_cnt = 0
            self.command = self.command_history[0]

    def get(self):
        return self.command

    def _update(self):
        raise NotImplementedError

    def update(self):
        if self.recorded_command:
            self.command_cnt += 1
            idx = min(self.command_cnt, len(self.command_history) - 1)
            self.command = self.command_history[idx]
        else:
            self._update()
            if self.record:
                self.command_history.append(copy.deepcopy(self.command))

    def save_history(self, filename):
        assert self.record
        with gzip.open(filename, "wb") as f:
            pickle.dump(self.command_history, f)
            print("Saved: %s (%d)" % (filename, len(self.command_history)))

    def load_history(self, filename):
        with gzip.open(filename, "rb") as f:
            self.command_history = pickle.load(f)
            print("Loaded: %s (%d)" % (filename, len(self.command_history)))


class JoystickCommand(Command):
    def __init__(self, trajectory, record=False):
        super(JoystickCommand, self).__init__(trajectory, record)
        sdl.SDL_Init(sdl.SDL_INIT_JOYSTICK)
        self.axis = {}
        self.button = {}
        self.reset()

    def reset(self, cam_dir=None):
        super(JoystickCommand, self).reset()
        if cam_dir is not None:
            self.command["target_dir"] = np.array(cam_dir[0], 0, cam_dir[2])

    def _update(self):
        event = sdl.SDL_Event()
        while sdl.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl.SDL_JOYDEVICEADDED:
                self.device = sdl.SDL_JoystickOpen(event.jdevice.which)
            elif event.type == sdl.SDL_JOYAXISMOTION:
                self.axis[event.jaxis.axis] = event.jaxis.value
                if event.jaxis.axis == 0:
                    self.command["x_vel"] = -self.axis[0]
                elif event.jaxis.axis == 1:
                    self.command["y_vel"] = -self.axis[1]
                elif event.jaxis.axis == 3:
                    self.command["x_move"] = -self.axis[3]
                elif event.jaxis.axis == 4:
                    self.command["y_move"] = self.axis[4]
                elif event.jaxis.axis == 5:
                    self.command["zoom_i"] = self.axis[5]
                elif event.jaxis.axis == 2:
                    self.command["zoom_o"] = self.axis[2]
            elif event.type == sdl.SDL_JOYBUTTONDOWN:
                self.button[event.jbutton.button] = True
                if event.jbutton.button == 1:
                    self.command["trigger_crouch"] = self.scale
                elif event.jbutton.button == 5:
                    self.command["trigger_speed"] = self.scale
                elif event.jbutton.button == 0:
                    self.command["trigger_obstacle"] = self.scale
                elif event.jbutton.button == 2:
                    self.command["trigger_jump"] = self.scale
            elif event.type == sdl.SDL_JOYBUTTONUP:
                self.button[event.jbutton.button] = False
                if event.jbutton.button == 1:
                    self.command["trigger_crouch"] = -self.scale
                elif event.jbutton.button == 5:
                    self.command["trigger_speed"] = -self.scale
                elif event.jbutton.button == 0:
                    self.command["trigger_obstacle"] = -self.scale
                elif event.jbutton.button == 2:
                    self.command["trigger_jump"] = -self.scale
        if math.fabs(self.command["x_vel"]) + math.fabs(self.command["y_vel"]) < 10000:
            self.command["x_vel"] = 0
            self.command["y_vel"] = 0
        if (
            math.fabs(self.command["x_move"]) + math.fabs(self.command["y_move"])
            < 10000
        ):
            self.command["x_move"] = 0
            self.command["y_move"] = 0


class AutonomousCommand(Command):
    def __init__(
        self,
        trajectory,
        vel_fwrd_init=1.0,
        vel_side_init=0.0,
        vel_fwrd_change=0.4,
        vel_side_change=0.4,
        record=False,
    ):
        super(AutonomousCommand, self).__init__(trajectory, record)
        self.vel_fwrd = Momentum(vel_fwrd_init, vel_fwrd_change)
        self.vel_side = Momentum(vel_side_init, vel_side_change)
        self.speed = Trigger(dt=1 / 60.0)
        self.crouch = Trigger(dt=1 / 60.0)
        self.stop = Trigger(dt=1 / 60.0, prob=0.1, min_duration=3.0, max_duration=5.0)
        self.reset()

    def reset(self):
        super(AutonomousCommand, self).reset()
        self.vel_fwrd.reset()
        self.vel_side.reset()
        self.command["x_vel"] = self.scale * self.vel_side.get()
        self.command["y_vel"] = self.scale * self.vel_fwrd.get()

    def _update(self):
        self.vel_side.update()
        self.vel_fwrd.update()
        self.speed.update()
        self.crouch.update()
        self.stop.update()
        self.command["x_vel"] = self.scale * self.vel_side.get()
        self.command["y_vel"] = self.scale * self.vel_fwrd.get()
        self.command["trigger_speed"] = (
            self.scale if self.speed.triggered() else -self.scale
        )
        self.command["trigger_crouch"] = (
            self.scale if self.crouch.triggered() else -self.scale
        )
        self.command["trigger_stop"] = (
            self.scale if self.stop.triggered() else -self.scale
        )
        if self.command["trigger_stop"] > 0:
            self.command["x_vel"] = 0
            self.command["y_vel"] = 0
        if math.fabs(self.command["x_vel"]) + math.fabs(self.command["y_vel"]) < 10000:
            self.command["x_vel"] = 0
            self.command["y_vel"] = 0
        if (
            math.fabs(self.command["x_move"]) + math.fabs(self.command["y_move"])
            < 10000
        ):
            self.command["x_move"] = 0
            self.command["y_move"] = 0

    def get(self):
        return self.command

    def set_change(vel_fwrd_change=0.1, vel_side_change=0.1):
        self.vel_fwrd.dx = vel_fwrd_change
        self.vel_side.dx = vel_side_change


class Runner(object):
    class UserInput(Enum):
        Autonomous = 0
        Joystick = 1
        Recorded = 2

        @classmethod
        def from_string(cls, string):
            if string == "autonomous":
                return cls.Autonomous
            if string == "joystick":
                return cls.Joystick
            if string == "recorded":
                return cls.Recorded
            raise NotImplementedError

    def __init__(
        self, ik=True, user_input="autonomous", command_file=None, record=False
    ):
        self.areas = Areas()
        self.controller = Controller()
        self.trajectory = Trajectory()
        self.character = Character()
        self.options = Options()
        if isinstance(user_input, str):
            user_input = Runner.UserInput.from_string(user_input)
        if user_input == Runner.UserInput.Autonomous:
            self.command = AutonomousCommand(self.trajectory, record=record)
        elif user_input == Runner.UserInput.Joystick:
            self.command = JoystickCommand(self.trajectory, record=record)
        elif user_input == Runner.UserInput.Recorded:
            assert command_file is not None
            self.command = Command(self.trajectory, history_file=command_file)
        else:
            raise NotImplementedError
        self.reset()

    def reset(self):
        root_pos = np.zeros(3)
        root_ori = np.eye(3)

        TL = self.trajectory.length
        JN = self.character.JOINT_NUM
        y_p = self.controller.y_mean

        for i in range(self.character.JOINT_NUM):
            opos = 8 + (((TL // 2) // 10) * 4) + (JN * 3 * 0)
            ovel = 8 + (((TL // 2) // 10) * 4) + (JN * 3 * 1)
            orot = 8 + (((TL // 2) // 10) * 4) + (JN * 3 * 2)

            pos = np.dot(root_ori, y_p[opos + i * 3 + 0 : opos + i * 3 + 3] + root_pos)
            vel = np.dot(root_ori, y_p[ovel + i * 3 + 0 : ovel + i * 3 + 3])
            rot = np.dot(
                root_ori, conversions.A2R(y_p[orot + i * 3 + 0 : orot + i * 3 + 3])
            )

            self.character.joint_positions[i] = pos
            self.character.joint_velocities[i] = vel
            self.character.joint_rotations[i] = rot

        self.areas.clear()
        self.character.reset()
        self.trajectory.reset(root_pos=root_pos, root_rot=root_ori)
        self.command.reset()
        self.update()

    def update(self, sim_agent=None):
        if sim_agent is not None:
            """move trajectory to the simulated agent a little bit"""
            trajectory = self.trajectory
            TL = int(trajectory.length)
            TL_div_2 = trajectory.length // 2
            R1, p1 = conversions.T2Rp(sim_agent.get_facing_frame())
            p1 *= 1000 / 9
            R2, p2 = trajectory.rotations[TL_div_2], trajectory.positions[TL_div_2]
            R = fm_math.slerp(R1, R2, 0.5)
            p = 0.5 * (p1 + p2)
            self.transform(conversions.Rp2T(R2, p))
        self.command.update()
        self.update_prev()
        self.update_post()

    def transform(self, T_target):
        trajectory = self.trajectory
        TL = int(trajectory.length)
        TL_div_2 = trajectory.length // 2
        """ Move trajectory """
        T_pivot = conversions.Rp2T(
            trajectory.rotations[TL_div_2], trajectory.positions[TL_div_2]
        )
        T_pivot_inv = fm_math.invertT(T_pivot)
        for i in range(TL):
            T = conversions.Rp2T(trajectory.rotations[i], trajectory.positions[i])
            T_local = np.dot(T_pivot_inv, T)
            T_new = np.dot(T_target, T_local)
            trajectory.rotations[i], trajectory.positions[i] = conversions.T2Rp(T_new)
        trajectory.directions = [
            trajectory.rotations[i][:, 2].copy() for i in range(TL)
        ]
        trajectory.heights = [0.0] * TL
        """ Move character """
        character = self.character
        JN = character.JOINT_NUM
        R_pivot, p_pivot = conversions.T2Rp(T_pivot)
        R_pivot_inv = R_pivot.transpose()
        R_target, p_target = conversions.T2Rp(T_target)
        for i in range(JN):
            pos = np.dot(R_pivot_inv, character.joint_positions[i] - p_pivot)
            vel = np.dot(R_pivot_inv, character.joint_velocities[i])
            rot = np.dot(R_pivot_inv, character.joint_rotations[i])
            character.joint_positions[i] = np.dot(R_target, pos) + p_target
            character.joint_velocities[i] = np.dot(R_target, vel)
            character.joint_rotations[i] = np.dot(R_target, rot)
            character.joint_global_anim_xform[i] = conversions.Rp2T(
                character.joint_rotations[i], character.joint_positions[i]
            )

    def blend_sim_kin(self):
        if self.sim_agent is None:
            return

        TL = int(trajectory.length)
        TL_div_2 = trajectory.length // 2

        R_sim, p_sim = conversions.T2Rp(self.sim_agent.get_facing_frame())
        dir_sim = R_sim[:, 2]
        trajectory_positions_blend[TL_div_2] = linear(
            trajectory.positions[TL_div_2], p_sim, 0.5
        )
        trajectory.directions[TL_div_2] = mix_directions(
            trajectory.directions[TL_div_2], dir_sim, 0.5
        )
        for i in range(TL_div_2 + 1, TL):
            scale_dir = 1.0 - math.pow(1.0 - (float(i - TL_div_2) / (TL_div_2)), 2.0)
            trajectory.directions[i] = mix_directions(
                trajectory.directions[i], dir_sim, scale_dir
            )

    def update_prev(self):
        areas = self.areas
        trajectory = self.trajectory
        character = self.character
        controller = self.controller
        options = self.options
        TL = int(trajectory.length)
        TL_div_2 = trajectory.length // 2

        # command = self.command.get()
        command = self.command.get()
        scale = self.command.scale

        """ Update Target Direction / Velocity """

        trajectory_target_direction_new = command["target_dir"].copy()
        trajectory_target_rotation = conversions.Ay2R(
            math.atan2(
                trajectory_target_direction_new[0], trajectory_target_direction_new[2]
            )
        )

        target_vel_speed = 2.5 + 2.5 * ((command["trigger_speed"] / scale) + 1.0)

        trajectory_target_velocity_new = target_vel_speed * np.dot(
            trajectory_target_rotation,
            np.array([command["x_vel"] / scale, 0, command["y_vel"] / scale]),
        )
        trajectory.target_vel = linear(
            trajectory.target_vel,
            trajectory_target_velocity_new,
            options.extra_velocity_smooth,
        )

        character.strafe_target = ((command["trigger_strafe"] / scale) + 1.0) / 2.0
        character.strafe_amount = linear(
            character.strafe_amount,
            character.strafe_target,
            options.extra_strafe_smooth,
        )

        if np.linalg.norm(trajectory.target_vel) < 1e-05:
            trajectory_target_velocity_dir = trajectory.target_dir
        else:
            trajectory_target_velocity_dir = trajectory.target_vel / np.linalg.norm(
                trajectory.target_vel
            )
        trajectory_target_direction_new = mix_directions(
            trajectory_target_velocity_dir,
            trajectory_target_direction_new,
            character.strafe_amount,
        )
        trajectory.target_dir = mix_directions(
            trajectory.target_dir,
            trajectory_target_direction_new,
            options.extra_direction_smooth,
        )

        character.crouched_target = 1.0 if command["trigger_crouch"] > 0 else 0.0
        character.crouched_amount = linear(
            character.crouched_amount,
            character.crouched_target,
            options.extra_crouched_smooth,
        )

        """ Update Gait """
        if np.linalg.norm(trajectory.target_vel) < 0.5:
            stand_amount = 1.0 - np.clip(
                np.linalg.norm(trajectory.target_vel) / 0.1, 0.0, 1.0
            )
            trajectory.gait_stand[TL_div_2] = linear(
                trajectory.gait_stand[TL_div_2], stand_amount, options.extra_gait_smooth
            )
            trajectory.gait_walk[TL_div_2] = linear(
                trajectory.gait_walk[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_jog[TL_div_2] = linear(
                trajectory.gait_jog[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_crouch[TL_div_2] = linear(
                trajectory.gait_crouch[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_jump[TL_div_2] = linear(
                trajectory.gait_jump[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_bump[TL_div_2] = linear(
                trajectory.gait_bump[TL_div_2], 0.0, options.extra_gait_smooth
            )
        elif character.crouched_amount > 0.1:
            trajectory.gait_stand[TL_div_2] = linear(
                trajectory.gait_stand[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_walk[TL_div_2] = linear(
                trajectory.gait_walk[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_jog[TL_div_2] = linear(
                trajectory.gait_jog[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_crouch[TL_div_2] = linear(
                trajectory.gait_crouch[TL_div_2],
                character.crouched_amount,
                options.extra_gait_smooth,
            )
            trajectory.gait_jump[TL_div_2] = linear(
                trajectory.gait_jump[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_bump[TL_div_2] = linear(
                trajectory.gait_bump[TL_div_2], 0.0, options.extra_gait_smooth
            )
        elif command["trigger_speed"] > 0:
            trajectory.gait_stand[TL_div_2] = linear(
                trajectory.gait_stand[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_walk[TL_div_2] = linear(
                trajectory.gait_walk[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_jog[TL_div_2] = linear(
                trajectory.gait_jog[TL_div_2], 1.0, options.extra_gait_smooth
            )
            trajectory.gait_crouch[TL_div_2] = linear(
                trajectory.gait_crouch[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_jump[TL_div_2] = linear(
                trajectory.gait_jump[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_bump[TL_div_2] = linear(
                trajectory.gait_bump[TL_div_2], 0.0, options.extra_gait_smooth
            )
        else:
            trajectory.gait_stand[TL_div_2] = linear(
                trajectory.gait_stand[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_walk[TL_div_2] = linear(
                trajectory.gait_walk[TL_div_2], 1.0, options.extra_gait_smooth
            )
            trajectory.gait_jog[TL_div_2] = linear(
                trajectory.gait_jog[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_crouch[TL_div_2] = linear(
                trajectory.gait_crouch[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_jump[TL_div_2] = linear(
                trajectory.gait_jump[TL_div_2], 0.0, options.extra_gait_smooth
            )
            trajectory.gait_bump[TL_div_2] = linear(
                trajectory.gait_bump[TL_div_2], 0.0, options.extra_gait_smooth
            )

        """ Predict Future Trajectory """
        trajectory_positions_blend = [np.zeros(3) for i in range(TL)]
        trajectory_positions_blend[TL_div_2] = trajectory.positions[TL_div_2]

        if character.responsive:
            bias_pos = linear(2.0, 2.0, character.strafe_amount)
            bias_dir = linear(5.0, 3.0, character.strafe_amount)
        else:
            bias_pos = linear(0.5, 1.0, character.strafe_amount)
            bias_dir = linear(2.0, 0.5, character.strafe_amount)

        for i in range(TL_div_2 + 1, TL):
            scale_pos = 1.0 - math.pow(
                1.0 - (float(i - TL_div_2) / (TL_div_2)), bias_pos
            )
            scale_dir = 1.0 - math.pow(
                1.0 - (float(i - TL_div_2) / (TL_div_2)), bias_dir
            )

            trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] + linear(
                trajectory.positions[i] - trajectory.positions[i - 1],
                trajectory.target_vel,
                scale_pos,
            )
            trajectory.directions[i] = mix_directions(
                trajectory.directions[i], trajectory.target_dir, scale_dir
            )
            trajectory.heights[i] = trajectory.heights[TL_div_2]

            trajectory.gait_stand[i] = trajectory.gait_stand[TL_div_2]
            trajectory.gait_walk[i] = trajectory.gait_walk[TL_div_2]
            trajectory.gait_jog[i] = trajectory.gait_jog[TL_div_2]
            trajectory.gait_crouch[i] = trajectory.gait_crouch[TL_div_2]
            trajectory.gait_jump[i] = trajectory.gait_jump[TL_div_2]
            trajectory.gait_bump[i] = trajectory.gait_bump[TL_div_2]

        trajectory.positions[TL_div_2 + 1 :] = trajectory_positions_blend[
            TL_div_2 + 1 :
        ]

        """ Trajectory Rotation """
        for i in range(TL):
            trajectory.rotations[i] = conversions.Ay2R(
                math.atan2(trajectory.directions[i][0], trajectory.directions[i][2])
            )

        """ Trajectory Heights """
        if command["trigger_jump"] > 0:
            for i in range(TL_div_2 + 30, TL_div_2 + 40):
                trajectory.gait_jump[i] = 1.0
                h = 100.0
                trajectory.positions[i][1] = h
                trajectory.heights[i] = h

        trajectory.heights[TL_div_2] = 0.0
        for i in range(0, TL, 10):
            trajectory.heights[TL_div_2] += trajectory.positions[i][1] / (TL // 10)

        root_position = np.array(
            [
                trajectory.positions[TL_div_2][0],
                trajectory.heights[TL_div_2],
                trajectory.positions[TL_div_2][2],
            ]
        )
        root_rotation = trajectory.rotations[TL_div_2]
        root_rotation_inv = np.transpose(root_rotation)

        """ Input Trajectory Positions / Directions """

        for i in range(0, TL, 10):
            w = TL // 10
            pos = np.dot(root_rotation_inv, trajectory.positions[i] - root_position)
            dir = np.dot(root_rotation_inv, trajectory.directions[i])
            controller.x_p[(w * 0) + i // 10] = pos[0]
            controller.x_p[(w * 1) + i // 10] = pos[2]
            controller.x_p[(w * 2) + i // 10] = dir[0]
            controller.x_p[(w * 3) + i // 10] = dir[2]

        """ Input Trajectory Gaits """
        for i in range(0, TL, 10):
            w = TL // 10
            controller.x_p[(w * 4) + i // 10] = trajectory.gait_stand[i]
            controller.x_p[(w * 5) + i // 10] = trajectory.gait_walk[i]
            controller.x_p[(w * 6) + i // 10] = trajectory.gait_jog[i]
            controller.x_p[(w * 7) + i // 10] = trajectory.gait_crouch[i]
            controller.x_p[(w * 8) + i // 10] = trajectory.gait_jump[i]
            controller.x_p[(w * 9) + i // 10] = 0.0

        """ Input Joint Previous Positions / Velocities / Rotations """
        prev_root_position = np.array(
            [
                trajectory.positions[TL_div_2 - 1][0],
                trajectory.heights[TL_div_2 - 1],
                trajectory.positions[TL_div_2 - 1][2],
            ]
        )
        prev_root_rotation = trajectory.rotations[TL_div_2 - 1]
        prev_root_rotation_inv = np.transpose(prev_root_rotation)

        JN = character.JOINT_NUM

        for i in range(JN):
            o = (TL // 10) * 10
            pos = np.dot(
                prev_root_rotation_inv,
                character.joint_positions[i] - prev_root_position,
            )
            prv = np.dot(prev_root_rotation_inv, character.joint_velocities[i])
            controller.x_p[o + (JN * 3 * 0) + i * 3 + 0] = pos[0]
            controller.x_p[o + (JN * 3 * 0) + i * 3 + 1] = pos[1]
            controller.x_p[o + (JN * 3 * 0) + i * 3 + 2] = pos[2]
            controller.x_p[o + (JN * 3 * 1) + i * 3 + 0] = prv[0]
            controller.x_p[o + (JN * 3 * 1) + i * 3 + 1] = prv[1]
            controller.x_p[o + (JN * 3 * 1) + i * 3 + 2] = prv[2]

        """ Input Trajectory Heights """
        for i in range(0, TL, 10):
            o = ((TL // 10) * 10) + JN * 3 * 2
            w = TL // 10
            position_r = trajectory.positions[i] + np.dot(
                trajectory.rotations[i], np.array([trajectory.width, 0, 0])
            )
            position_l = trajectory.positions[i] + np.dot(
                trajectory.rotations[i], np.array([-trajectory.width, 0, 0])
            )
            ## Fixation needed
            controller.x_p[o + (w * 0) + (i // 10)] = (
                trajectory.positions[i][1] - root_position[1]
            )
            controller.x_p[o + (w * 1) + (i // 10)] = (
                trajectory.positions[i][1] - root_position[1]
            )
            controller.x_p[o + (w * 2) + (i // 10)] = (
                trajectory.positions[i][1] - root_position[1]
            )

        """ Perform Regression """
        controller.predict(character.phase)
        y_p = controller.y_p

        """ Build Local Transforms """
        for i in range(JN):
            opos = 8 + (((TL_div_2) // 10) * 4) + (JN * 3 * 0)
            ovel = 8 + (((TL_div_2) // 10) * 4) + (JN * 3 * 1)
            orot = 8 + (((TL_div_2) // 10) * 4) + (JN * 3 * 2)

            pos = (
                np.dot(root_rotation, y_p[opos + i * 3 + 0 : opos + i * 3 + 3])
                + root_position
            )
            vel = np.dot(root_rotation, y_p[ovel + i * 3 + 0 : ovel + i * 3 + 3])
            rot = np.dot(
                root_rotation, conversions.A2R(y_p[orot + i * 3 + 0 : orot + i * 3 + 3])
            )

            """
            Blending Between the predicted positions and
            the previous positions plus the velocities 
            smooths out the motion a bit in the case 
            where the two disagree with each other.
            """
            pos = linear(
                character.joint_positions[i] + vel, pos, options.extra_joint_smooth
            )
            character.joint_positions[i] = pos
            character.joint_velocities[i] = vel
            character.joint_rotations[i] = rot

            character.joint_global_anim_xform[i] = np.array(
                [
                    [rot[0][0], rot[0][1], rot[0][2], pos[0]],
                    [rot[1][0], rot[1][1], rot[1][2], pos[1]],
                    [rot[2][0], rot[2][1], rot[2][2], pos[2]],
                    [0, 0, 0, 1],
                ]
            )

        for i in range(JN):
            if i == 0:
                character.joint_anim_xform[i] = character.joint_global_anim_xform[i]
            else:
                T1 = character.joint_global_anim_xform[character.joint_parents[i]]
                T2 = character.joint_global_anim_xform[i]
                T = np.dot(fm_math.invertT(T1), T2)
                character.joint_anim_xform[i] = T

        character.forward_kinematics_correct()

    def update_post(self):
        trajectory = self.trajectory
        character = self.character
        TL = trajectory.length
        TL_div_2 = TL // 2
        y_p = self.controller.y_p

        """Update Past Trajectory """
        for i in range(0, TL_div_2):
            trajectory.positions[i] = trajectory.positions[i + 1]
            trajectory.directions[i] = trajectory.directions[i + 1]
            trajectory.rotations[i] = trajectory.rotations[i + 1]
            trajectory.heights[i] = trajectory.heights[i + 1]
            trajectory.gait_stand[i] = trajectory.gait_stand[i + 1]
            trajectory.gait_walk[i] = trajectory.gait_walk[i + 1]
            trajectory.gait_jog[i] = trajectory.gait_jog[i + 1]
            trajectory.gait_crouch[i] = trajectory.gait_crouch[i + 1]
            trajectory.gait_jump[i] = trajectory.gait_jump[i + 1]
            trajectory.gait_bump[i] = trajectory.gait_bump[i + 1]

        """ Update Current Trajectory """
        stand_amount = math.pow(1.0 - trajectory.gait_stand[TL_div_2], 0.25)

        trajectory_update = np.dot(
            trajectory.rotations[TL_div_2], np.array([y_p[0], 0, y_p[1]])
        )
        trajectory.positions[TL_div_2] = (
            trajectory.positions[TL_div_2] + stand_amount * trajectory_update
        )
        trajectory.directions[TL_div_2] = np.dot(
            conversions.Ay2R(stand_amount * -y_p[2]), trajectory.directions[TL_div_2]
        )
        trajectory.rotations[TL_div_2] = conversions.Ay2R(
            math.atan2(
                trajectory.directions[TL_div_2][0], trajectory.directions[TL_div_2][2]
            )
        )

        """ Update Future Trajectory """
        d_position = np.zeros(3)
        d_direction = np.zeros(3)
        for i in range(TL_div_2 + 1, TL):
            w = TL_div_2 // 10
            m = math.fmod((i - TL_div_2) / 10.0, 1.0)
            d_position[0] = (1 - m) * y_p[8 + (w * 0) + (i // 10) - w] + m * y_p[
                8 + (w * 0) + (i // 10) - w + 1
            ]
            d_position[2] = (1 - m) * y_p[8 + (w * 1) + (i // 10) - w] + m * y_p[
                8 + (w * 1) + (i // 10) - w + 1
            ]
            d_direction[0] = (1 - m) * y_p[8 + (w * 2) + (i // 10) - w] + m * y_p[
                8 + (w * 2) + (i // 10) - w + 1
            ]
            d_direction[2] = (1 - m) * y_p[8 + (w * 3) + (i // 10) - w] + m * y_p[
                8 + (w * 3) + (i // 10) - w + 1
            ]

            position_new = (
                np.dot(trajectory.rotations[TL_div_2], d_position)
                + trajectory.positions[TL_div_2]
            )
            trajectory.positions[i] = linear(
                trajectory.positions[i], position_new, stand_amount
            )

            direction_new = np.dot(trajectory.rotations[TL_div_2], d_direction)
            direction_new /= np.linalg.norm(direction_new)
            trajectory.directions[i] = mix_directions(
                trajectory.directions[i], direction_new, stand_amount
            )

            trajectory.rotations[i] = conversions.Ay2R(
                math.atan2(trajectory.directions[i][0], trajectory.directions[i][2])
            )

        character.phase = math.fmod(
            character.phase + (stand_amount * 0.9 + 0.1) * 2 * math.pi * y_p[3],
            2 * math.pi,
        )


class PfnnViewer(glut_viewer.Viewer):
    def __init__(
        self,
        motion,
        flag,
        toggle,
        play_speed=1.0,
        scale=1.0,
        thickness=1.0,
        render_overlay=True,
        hide_origin=False,
        **kwargs,
    ):
        self.runner = Runner(
            user_input="autonomous",
            # command_file="data/temp/temp.pfnncommand.gzip"
        )
        self.motion = motion
        self.play_speed = play_speed
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.cur_time = 0.0
        self.scale = scale
        self.thickness = thickness
        self.flag = flag
        self.toggle = toggle
        super().__init__(**kwargs)

    def set_camera(self, cam):
        self.cam_cur = cam

    def update_target_pos(self, pos, ignore_x=False, ignore_y=False, ignore_z=False):
        if np.array_equal(pos, self.cam_cur.origin):
            return
        d = pos - self.cam_cur.origin
        if ignore_x:
            d[0] = 0.0
        if ignore_y:
            d[1] = 0.0
        if ignore_z:
            d[2] = 0.0
        self.cam_cur.translate(d)

    def keyboard_callback(self, key):
        if key in self.toggle:
            self.flag[self.toggle[key]] = not self.flag[self.toggle[key]]
            print("Toggle:", self.toggle[key], self.flag[self.toggle[key]])
        elif key == b" ":
            self.runner.update()
        elif key == b"r":
            self.runner.reset()
            self.motion.clear()
            self.time_checker.begin()
        elif key == b"s":
            """Make sure that PFNN uses (cm), we outputs the motion as (m)"""
            self.motion.save_bvh("test.bvh", scale=0.01, verbose=True)
        elif key == b"S":
            """Make sure that PFNN uses (cm), we outputs the motion as (m)"""
            self.runner.command.save_history("data/temp/temp.pfnncommand.gzip")
        elif key == b"c":
            cnt_screenshot = 0
            time_elapsed = 0.0
            dt = 1 / 60.0
            self.runner.reset()
            while time_elapsed <= 60.0:
                self.runner.update()
                name = "screenshot_pfnn_%04d" % (cnt_screenshot)
                self.save_screen(dir="data/screenshot", name=name)
                print("time_elapsed:", time_elapsed, "(", name, ")")
                time_elapsed += dt
                cnt_screenshot += 1
        elif key == b"\x1b":
            glutDestroyWindow(self.window)
            exit(0)
        else:
            raise NotImplementedError("key:" + str(key))

    def idle_callback(self):
        if isinstance(self.runner.command, JoystickCommand):
            command = self.runner.command.get()
            scale = self.runner.command.scale
            self.cam_cur.rotate(
                0.1 * command["y_move"] / scale, 0.1 * command["x_move"] / scale, 0.0
            )
            if command["zoom_o"] > 0.0:
                self.cam_cur.zoom(1.02)
            elif command["zoom_i"] > 0.0:
                self.cam_cur.zoom(0.98)
            command["target_dir"] = self.cam_cur.origin - self.cam_cur.pos

        if (
            self.flag["auto_play"]
            and self.time_checker.get_time(restart=False) >= 1.0 / 30.0
        ):
            """pfnn is 60 FPS"""
            for i in range(2):
                self.runner.update()
            """ we create motion at 30 FPS """
            self.motion.add_one_frame(
                copy.deepcopy(self.runner.character.joint_xform_by_ik)
            )
            self.time_checker.begin()

    def render_callback(self):
        if self.flag["ground"]:
            gl_render.render_ground(
                size=[100, 100],
                color=[0.9, 0.9, 0.9],
                axis="y",
                origin=self.flag["origin"],
                use_arrow=True,
            )

        if self.flag["follow_cam"]:
            p = (
                0.01
                * self.runner.trajectory.positions[self.runner.trajectory.length // 2]
            )
            viewer.update_target_pos(p, ignore_y=True)

        areas = self.runner.areas
        options = self.runner.options
        trajectory = self.runner.trajectory
        character = self.runner.character

        """ pfnn character uses centi-meter """
        glPushMatrix()
        glScalef(0.01, 0.01, 0.01)

        """ Trajectory """
        if self.flag["trajectory"]:
            glDisable(GL_LIGHTING)
            glPointSize(2.0 * options.display_scale)
            glBegin(GL_POINTS)
            for i in range(0, trajectory.length - 10):
                position_c = trajectory.positions[i]
                glColor3f(
                    trajectory.gait_jump[i],
                    trajectory.gait_bump[i],
                    trajectory.gait_crouch[i],
                )
                glVertex3f(position_c[0], position_c[1] + 2, position_c[2])
            glEnd()

            glPointSize(2.0 * options.display_scale)
            glBegin(GL_POINTS)
            for i in range(0, trajectory.length, 10):
                R, p = trajectory.rotations[i], trajectory.positions[i]
                position_c = p
                position_r = p + np.dot(R, np.array([trajectory.width, 0, 0]))
                position_l = p + np.dot(R, np.array([-trajectory.width, 0, 0]))

                glColor3f(
                    trajectory.gait_jump[i],
                    trajectory.gait_bump[i],
                    trajectory.gait_crouch[i],
                )
                glVertex3f(position_c[0], position_c[1] + 2, position_c[2])
                glVertex3f(position_r[0], position_r[1] + 2, position_r[2])
                glVertex3f(position_l[0], position_l[1] + 2, position_l[2])
            glEnd()

            glLineWidth(2.0 * options.display_scale)
            glBegin(GL_LINES)
            for i in range(0, trajectory.length, 10):
                p = trajectory.positions[i]
                d = trajectory.directions[i]
                base = p + np.array([0, 2, 0])
                side = np.cross(d, np.array([0, 1, 0]))
                side /= np.linalg.norm(side)
                fwrd = base + 15 * d
                arw0 = fwrd + 4 * side + 4 * -d
                arw1 = fwrd - 4 * side + 4 * -d
                glColor3f(
                    trajectory.gait_jump[i],
                    trajectory.gait_bump[i],
                    trajectory.gait_crouch[i],
                )
                glVertex3f(base[0], base[1], base[2])
                glVertex3f(fwrd[0], fwrd[1], fwrd[2])
                glVertex3f(fwrd[0], fwrd[1], fwrd[2])
                glVertex3f(arw0[0], fwrd[1], arw0[2])
                glVertex3f(fwrd[0], fwrd[1], fwrd[2])
                glVertex3f(arw1[0], fwrd[1], arw1[2])
            glEnd()

        """ joint positions given from NN """

        if self.flag["character_by_pos"]:
            glEnable(GL_LIGHTING)
            for i in range(character.JOINT_NUM):
                pos = character.joint_positions[i]
                gl_render.render_point(pos, radius=2, color=[0.8, 0.8, 0.0, 1.0])
                j = character.joint_parents[i]
                if j != -1:
                    pos_parent = character.joint_positions[j]
                    gl_render.render_line(p1=pos_parent, p2=pos, color=[0.5, 0.5, 0, 1])

        """ joint positions computed by forward-kinamatics with rotations given from NN """

        if self.flag["character_by_rot_fk"]:
            glPushMatrix()
            glTranslatef(100, 0, 0)

            glEnable(GL_LIGHTING)
            for i in range(character.JOINT_NUM):
                pos = conversions.T2p(character.joint_global_anim_xform_by_fk[i])
                gl_render.render_point(pos, radius=2, color=[0.0, 0.8, 0.8, 1.0])
                j = character.joint_parents[i]
                if j != -1:
                    pos_parent = conversions.T2p(
                        character.joint_global_anim_xform_by_fk[j]
                    )
                    gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0.5, 0.5, 1])

            glPopMatrix()

        """ joint positions computed by forward-kinamatics with rotations given from NN """

        if self.flag["character_by_rot_ik"]:
            glPushMatrix()
            glTranslatef(-100, 0, 0)

            glEnable(GL_LIGHTING)
            for i in range(character.JOINT_NUM):
                pos = conversions.T2p(character.joint_global_xform_by_ik[i])
                gl_render.render_point(pos, radius=2, color=[0.8, 0.0, 0.0, 1.0])
                j = character.joint_parents[i]
                if j != -1:
                    pos_parent = conversions.T2p(character.joint_global_xform_by_ik[j])
                    gl_render.render_line(p1=pos_parent, p2=pos, color=[0.5, 0, 0, 1])

            glPopMatrix()

        """ Render Jump Areas """
        for i in range(areas.num_jumps()):
            glColor3f(1.0, 0.0, 0.0)
            glLineWidth(options.display_scale * 2.0)
            glBegin(GL_LINES)
            for r in np.arange(0.0, 1.0, 0.01):
                glVertex3f(
                    areas.jump_pos[i][0]
                    + areas.jump_size[i] * math.sin((r + 0.00) * 2 * math.pi),
                    areas.jump_pos[i][1],
                    areas.jump_pos[i][2]
                    + areas.jump_size[i] * math.cos((r + 0.00) * 2 * math.pi),
                )
                glVertex3f(
                    areas.jump_pos[i][0]
                    + areas.jump_size[i] * math.sin((r + 0.01) * 2 * math.pi),
                    areas.jump_pos[i][1],
                    areas.jump_pos[i][2]
                    + areas.jump_size[i] * math.cos((r + 0.01) * 2 * math.pi),
                )
            glEnd()

        glPopMatrix()

    def overlay_callback(self):
        command = self.runner.command.get()
        scale = self.runner.command.scale

        glPushAttrib(GL_LIGHTING)
        glDisable(GL_LIGHTING)
        glPointSize(0.001)

        w, h = viewer.window_size
        pad = 10
        w_bar, h_bar = 120, 10
        phase_radius = 50
        joy_radius = 30

        origin = np.array([0.05 * w, 0.05 * h])
        pos = origin.copy()
        gl_render.render_progress_circle_2D(
            self.runner.character.phase / (2 * math.pi),
            origin=(pos[0] + phase_radius, pos[1] + phase_radius),
            radius=phase_radius,
        )
        gl_render.render_text(
            "phase",
            pos=(pos[0] + 2 * phase_radius + pad, pos[1] + phase_radius),
            font=GLUT_BITMAP_9_BY_15,
        )
        pos += np.array([0.0, 2 * phase_radius + 2 * pad])
        gl_render.render_direction_input_2D(
            (-command["x_vel"], -command["y_vel"]),
            (scale, scale),
            origin=(pos[0] + joy_radius, pos[1] + joy_radius),
            radius=joy_radius,
        )
        gl_render.render_direction_input_2D(
            (-command["x_move"], command["y_move"]),
            (scale, scale),
            origin=(pos[0] + 3 * joy_radius + pad, pos[1] + joy_radius),
            radius=joy_radius,
        )
        pos += np.array([0.0, 2 * joy_radius + pad])
        gl_render.render_progress_bar_2D_horizontal(
            command["trigger_speed"] / scale, origin=pos, width=w_bar, height=h_bar
        )
        gl_render.render_text(
            "trigger_speed",
            pos=(pos[0] + w_bar + pad, pos[1] + h_bar),
            font=GLUT_BITMAP_9_BY_15,
        )
        pos += np.array([0.0, h_bar + pad])
        gl_render.render_progress_bar_2D_horizontal(
            command["trigger_crouch"] / scale, origin=pos, width=w_bar, height=h_bar
        )
        gl_render.render_text(
            "trigger_crouch",
            pos=(pos[0] + w_bar + pad, pos[1] + h_bar),
            font=GLUT_BITMAP_9_BY_15,
        )
        pos += np.array([0.0, h_bar + pad])
        gl_render.render_progress_bar_2D_horizontal(
            command["trigger_stop"] / scale, origin=pos, width=w_bar, height=h_bar
        )
        gl_render.render_text(
            "trigger_stop",
            pos=(pos[0] + w_bar + pad, pos[1] + h_bar),
            font=GLUT_BITMAP_9_BY_15,
        )
        pos += np.array([0.0, h_bar + pad])
        gl_render.render_progress_bar_2D_horizontal(
            command["trigger_jump"] / scale, origin=pos, width=w_bar, height=h_bar
        )
        gl_render.render_text(
            "trigger_jump",
            pos=(pos[0] + w_bar + pad, pos[1] + h_bar),
            font=GLUT_BITMAP_9_BY_15,
        )
        pos += np.array([0.0, h_bar + pad])

        glPopAttrib()


if __name__ == "__main__":
    # For viewer
    flag = {
        "follow_cam": True,
        "ground": True,
        "origin": True,
        "character_by_pos": True,
        "character_by_rot_fk": False,
        "character_by_rot_ik": False,
        "trajectory": True,
        "auto_play": False,
    }
    toggle = {
        b"0": "follow_cam",
        b"1": "ground",
        b"2": "origin",
        b"3": "character_by_pos",
        b"4": "character_by_rot_fk",
        b"5": "character_by_rot_ik",
        b"6": "trajectory",
        b"a": "auto_play",
    }
    motion = bvh.load("data/motion/pfnn_hierarchy.bvh")
    motion.clear()

    viewer = PfnnViewer(
        title="PFNN Viewer",
        cam=None,
        size=(1280, 720),
        flag=flag,
        toggle=toggle,
        motion=motion,
    )
    cam_origin = (
        0.01 * viewer.runner.trajectory.positions[viewer.runner.trajectory.length // 2]
    )
    cam_pos = cam_origin + np.array([0.0, 2.0, -3.5])
    cam_vup = np.array([0.0, 1.0, 0.0])
    cam = camera.Camera(
        pos=cam_pos,
        origin=cam_origin,
        vup=cam_vup,
        fov=45.0,
    )
    viewer.set_camera(cam)
    viewer.run()
