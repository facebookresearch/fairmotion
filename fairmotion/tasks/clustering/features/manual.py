# Copyright (c) Facebook, Inc. and its affiliates.

from fairmotion.tasks.clustering.features import utils as feat_utils
from fairmotion.ops import conversions


class ManualFeatures:
    def __init__(self, motion, skeleton=feat_utils.Skeleton.PFNN):
        self.global_positions = motion.positions(local=False)
        self.joints = [joint.name for joint in motion.skel.joints]
        self.frame_time = 1 / motion.fps
        self.frame_num = 1
        self.offsets = [
            conversions.T2p(joint.xform_from_parent_joint)
            for joint in motion.skel.joints
        ]
        if skeleton == feat_utils.Skeleton.PFNN:
            self.joint_mapping = feat_utils.GENERIC_TO_PFNN_MAPPING
        else:
            self.joint_mapping = feat_utils.GENERIC_TO_CMU_MAPPING

        # humerus length
        self.hl = feat_utils.distance_between_points(
            self.transform_and_fetch_offset("lshoulder"),
            self.transform_and_fetch_offset("lelbow"),
        )
        # shoulder width
        self.sw = feat_utils.distance_between_points(
            self.transform_and_fetch_offset("lshoulder"),
            self.transform_and_fetch_offset("rshoulder"),
        )
        # hip width
        self.hw = feat_utils.distance_between_points(
            self.transform_and_fetch_offset("lhip"),
            self.transform_and_fetch_offset("rhip"),
        )

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [
                0,
                min(
                    [y for (_, y, _) in self.global_positions[self.frame_num]]
                ),
                0,
            ]
        return self.global_positions[self.frame_num][
            self.joints.index(self.joint_mapping[j])
        ]

    def transform_and_fetch_prev_position(self, j):
        return self.global_positions[self.frame_num - 1][
            self.joints.index(self.joint_mapping[j])
        ]

    def transform_and_fetch_offset(self, j):
        return self.offsets[self.joints.index(self.joint_mapping[j])]

    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.velocity_direction_above_threshold(
            j1, j1_prev, j2, j2_prev, j3, j3_prev, range
        )

    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.velocity_direction_above_threshold_normal(
            j1, j1_prev, j2, j3, j4, j4_prev, range
        )

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [
            self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]
        ]
        return feat_utils.angle_within_range(j1, j2, j3, j4, range)

    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return feat_utils.velocity_above_threshold(j1, j1_prev, threshold)
