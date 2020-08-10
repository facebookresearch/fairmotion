# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import tempfile
import unittest
from fairmotion.data import bvh
from fairmotion.utils import conversions, utils


TEST_SINUSOIDAL_FILE = "tests/data/sinusoidal.bvh"
TEST_SINUSOIDAL_2_FILE = "tests/data/sinusoidal_2.bvh"


class TestBVH(unittest.TestCase):
    def test_load_motion(self):
        # Load file
        motion = bvh.load(file=TEST_SINUSOIDAL_FILE)

        # Read pose data from all frames (in Euler angle)
        with open(TEST_SINUSOIDAL_FILE) as f:
            file_content = f.readlines()
        for frame_num, line in enumerate(file_content[-motion.num_frames():]):
            # Skip first 3 entries that store translation data, and read the
            # Euler angle data of joints
            angle_data = np.array(
                list(map(float, line.strip().split()))[3:]
            ).reshape(-1, 3)
            for T, true_E in zip(motion.poses[frame_num].data, angle_data):
                R, _ = conversions.T2Rp(T)
                E = conversions.R2E(R, order="XYZ", degrees=True)
                np.testing.assert_almost_equal(E, true_E)

    def test_load_parallel(self):
        # Load files
        motions = bvh.load_parallel(
            files=[TEST_SINUSOIDAL_FILE, TEST_SINUSOIDAL_2_FILE],
        )
        self.assertEqual(len(motions), 2)
        # Test if the loaded motion objects are not same in the first frame
        self.assertNotEqual(motions[0].poses[0], motions[1].poses[0])

        # Use kwargs
        v_up_skel = utils.str_to_axis("y")
        v_face_skel = utils.str_to_axis("z")
        v_up_env = utils.str_to_axis("x")
        motions = bvh.load_parallel(
            files=[TEST_SINUSOIDAL_FILE, TEST_SINUSOIDAL_2_FILE],
            scale=0.1,
            v_up_skel=v_up_skel,
            v_face_skel=v_face_skel,
            v_up_env=v_up_env,
        )
        np.testing.assert_equal(motions[0].skel.v_up, v_up_skel)
        np.testing.assert_equal(motions[0].skel.v_face, v_face_skel)
        np.testing.assert_equal(motions[0].skel.v_up_env, v_up_env)

    def test_save_motion(self):
        # Load file
        motion = bvh.load(file=TEST_SINUSOIDAL_FILE)

        with tempfile.NamedTemporaryFile() as fp:
            # Save loaded file
            bvh.save(motion, fp.name, rot_order="XYZ")
            # Reload saved file and test if it is same as reference file
            # Read pose data from all frames (in Euler angle)
            with open(TEST_SINUSOIDAL_FILE) as f:
                orig_file = f.readlines()
            with open(fp.name) as f:
                saved_file = f.readlines()
            for orig_line, saved_line in zip(
                orig_file[-motion.num_frames():],
                saved_file[-motion.num_frames():]
            ):
                # Skip first 3 entries that store translation data, and read
                # the Euler angle data of joints
                orig_data, saved_data = [
                    np.array(list(map(float, line.strip().split())))
                    for line in [orig_line, saved_line]
                ]
                np.testing.assert_almost_equal(orig_data, saved_data)

            # Reload saved file and test if it has the same data as original
            # motion object
            with open(TEST_SINUSOIDAL_FILE) as f:
                file_content = f.readlines()
            for frame_num, line in enumerate(
                file_content[-motion.num_frames():]
            ):
                # Skip first 3 entries that store translation data, and read
                # the Euler angle data of joints
                angle_data = np.array(
                    list(map(float, line.strip().split()))[3:]
                ).reshape(-1, 3)
                for T, true_E in zip(motion.poses[frame_num].data, angle_data):
                    R, _ = conversions.T2Rp(T)
                    E = conversions.R2E(R, order="XYZ", degrees=True)
                    np.testing.assert_almost_equal(E, true_E)


if __name__ == "__main__":
    unittest.main()
