# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import unittest

from fairmotion.data import bvh
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions


TEST_SINUSOIDAL_FILE = "tests/data/sinusoidal.bvh"


class TestMotion(unittest.TestCase):
    def assert_motion_equal(self, ref_motion, test_motion):
        self.assertEqual(ref_motion.num_frames(), test_motion.num_frames())
        for frame_idx in range(ref_motion.num_frames()):
            for joint in ref_motion.skel.joints:
                np.testing.assert_almost_equal(
                    ref_motion.get_pose_by_frame(frame_idx).get_transform(
                        joint.name, local=True
                    ),
                    test_motion.get_pose_by_frame(frame_idx).get_transform(
                        joint.name, local=True
                    ),
                )

    def test_motion(self):
        motion = bvh.load(file=TEST_SINUSOIDAL_FILE)
        # Inspect 0th frame, root joint
        T = motion.get_pose_by_frame(0).get_transform(0, local=False)
        _, p = conversions.T2Rp(T)
        self.assertListEqual(list(p), [-3, 6, 5])
        # Inspect 100th frame, root joint
        T = motion.get_pose_by_frame(100).get_transform(0, local=False)
        _, p = conversions.T2Rp(T)
        self.assertListEqual(list(p), [-3, 6, 5])
        # Inspect 100th frame, "child2" joint
        T = motion.get_pose_by_frame(100).get_transform("child2", local=False)
        _, p = conversions.T2Rp(T)
        self.assertListEqual(list(p), [-8, 11, 5])

    def test_matrix_representation(self):
        ref_motion = bvh.load(file=TEST_SINUSOIDAL_FILE)
        test_motion = bvh.load(file=TEST_SINUSOIDAL_FILE, load_motion=False,)
        ref_matrix = ref_motion.to_matrix()
        test_motion = motion_class.Motion.from_matrix(
            ref_matrix, ref_motion.skel,
        )
        self.assert_motion_equal(ref_motion, test_motion)


if __name__ == "__main__":
    unittest.main()
