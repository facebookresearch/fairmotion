# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from fairmotion.ops import motion as motion_ops
from fairmotion.tasks.clustering.features import utils as feat_utils


class KineticFeatures:
    def __init__(
        self, motion, frame_time, thresholds, up_vec, sliding_window=2
    ):
        self.local_positions = motion.positions(local=True)
        self.positions = motion_ops.position_wrt_root(motion)
        self.frame_time = frame_time
        self.thresholds = thresholds
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
            len(self.positions) - 1.0
        )
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity_horizontal(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += feat_utils.calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val

    def local_position_stats(self, joint):
        return (
            np.mean(self.local_positions[:, joint], axis=0),
            np.std(self.local_positions[:, joint], axis=0),
        )
