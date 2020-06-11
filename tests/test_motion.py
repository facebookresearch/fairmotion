import tempfile
import unittest

import mocap_processing.motion.kinematics as kinematics

from mocap_processing.data import bvh
from mocap_processing.motion import motion as motion_class
from mocap_processing.utils import conversions


class TestMotion(unittest.TestCase):
    def assert_ndarray_equal(self, array1, array2):
        for elem1, elem2 in zip(array1.flatten(), array2.flatten()):
            self.assertAlmostEqual(elem1, elem2)

    def assert_motion_equal(self, ref_motion, test_motion):
        self.assertEqual(ref_motion.num_frames(), test_motion.num_frames())
        for frame_idx in range(ref_motion.num_frames()):
            for joint in ref_motion.skel.joints:
                self.assert_ndarray_equal(
                    ref_motion.get_pose_by_frame(frame_idx).get_transform(
                        joint.name, local=True
                    ),
                    test_motion.get_pose_by_frame(frame_idx).get_transform(
                        joint.name, local=True
                    ),
                )

    def test_motion(self):
        motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
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

    def test_save_motion(self):
        # Load file
        motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
        with tempfile.NamedTemporaryFile() as fp:
            # Save loaded file
            motion.save_bvh(fp.name, fps=60.0)
            # Reload saved file and test if it is same as reference file
            ref_motion = kinematics.Motion(file="tests/data/sinusoidal.bvh")
            test_motion = kinematics.Motion(file=fp.name)
            self.assert_motion_equal(ref_motion, test_motion)

    def test_matrix_representation(self):
        ref_motion = bvh.load(file="tests/data/sinusoidal.bvh")
        test_motion = bvh.load(
            file="tests/data/sinusoidal.bvh",
            load_motion=False,
        )
        ref_matrix = ref_motion.to_matrix()
        test_motion = motion_class.Motion.from_matrix(
            ref_matrix, ref_motion.skel,
        )
        self.assert_motion_equal(ref_motion, test_motion)


if __name__ == "__main__":
    unittest.main()
