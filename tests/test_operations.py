# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from fairmotion.data import bvh
from fairmotion.ops import motion as motion_ops


class TestMotion(unittest.TestCase):
    def assert_ndarray_equal(self, array1, array2):
        for elem1, elem2 in zip(array1.flatten(), array2.flatten()):
            self.assertAlmostEqual(elem1, elem2)

    def test_cut_motion(self):
        motion = bvh.load(file="tests/data/sinusoidal.bvh")
        # Inspect 0th frame, root joint
        cut_motion = motion_ops.cut(motion, 3, 5)
        self.assertEqual(cut_motion.num_frames(), 2)

    def test_append_motion(self):
        motion1 = bvh.load(file="tests/data/sinusoidal.bvh")
        motion2 = bvh.load(file="tests/data/sinusoidal.bvh")
        num_frames = motion1.num_frames()

        # Append motion 1 and motion 2, and check total combined frames
        combined_motion1 = motion_ops.append(motion1, motion2)
        self.assertEqual(
            combined_motion1.num_frames(),
            motion1.num_frames() + motion2.num_frames(),
        )

        # Append motion 1 and motion 3, and check total combined frames
        motion3 = bvh.load(file="tests/data/sinusoidal.bvh")
        combined_motion2 = motion_ops.append(motion1, motion3)
        self.assertEqual(
            combined_motion2.num_frames(),
            motion1.num_frames() + motion3.num_frames(),
        )
        # Ensure operation has not been done in place
        self.assertEqual(
            motion1.num_frames(), num_frames,
        )

        # Test blending
        blend_length = 0.1
        combined_motion3 = motion_ops.append_and_blend(
            motion1, motion2, blend_length=blend_length,
        )
        self.assertEqual(
            combined_motion3.num_frames(),
            (
                motion1.num_frames()
                + motion2.num_frames()
                - blend_length * motion1.fps
            ),
        )


if __name__ == "__main__":
    unittest.main()
