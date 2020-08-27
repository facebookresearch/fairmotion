# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import tqdm
from collections import defaultdict
from fairmotion.data import bvh
from fairmotion.tasks.clustering.features import utils as feat_utils


class PercentileThresholds:
    def __init__(self, root_folder, percentile=95, sliding_window=2):
        self.percentile = percentile
        self.velocities = defaultdict(list)
        self.sliding_window = sliding_window
        for root, _, files in os.walk(root_folder, topdown=False):
            for filename in tqdm.tqdm(files):
                motion = bvh.load(os.path.join(root, filename))
                self.joints = [joint.name for joint in motion.skel.joints]
                self._update_velocities(
                    motion.positions(local=False), 1 / motion.fps,
                )
        self.thresholds = self._compute_p95_thresholds()

    def _update_velocities(self, positions, frame_time):
        for i in range(1, len(positions)):
            for joint_idx in range(len(self.joints)):
                velocity = feat_utils.calc_average_velocity(
                    positions, i, joint_idx, self.sliding_window, frame_time
                )
                self.velocities[self.joints[joint_idx]].append(velocity)

    def _compute_p95_thresholds(self):
        return {
            joint: np.percentile(self.velocities[joint], 95)
            for joint in self.joints
        }

    def get_threshold(self, joint_idx):
        return self.thresholds[self.joints[joint_idx]]

    def __str__(self):
        print_str = ""
        for key, value in self.thresholds.items():
            print_str += "%s: %f\n" % (key, value)
        return print_str
